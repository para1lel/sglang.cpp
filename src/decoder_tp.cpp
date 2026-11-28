#include "sglang/runtime/execution/decoder_tp.h"

#include "sglang/decoder_internal.h"
#include "sglang/decoder_mtp.h"
#include "sglang/decoder_prefix_cache.h"
#include "sglang/mxfp8_gqa.h"
#include "sglang/runtime/backend/cuda_resources.h"
#include "sglang/runtime/backend/tensor_parallel.h"

#include <algorithm>
#include <iostream>
#include <utility>

namespace sglang {
namespace decoder {

const PrefixCacheLayout* tp2_prefix_layouts() {
    static const PrefixCacheLayout layouts[2] = {tp2_prefix_cache_layout(0),
                                                 tp2_prefix_cache_layout(1)};
    return layouts;
}

}

DecoderBatch::TpDriver::TpDriver(const Decoder::Model& owner, Decoder::TpRuntime& runtime)
    : model(owner), tp(runtime) {}

void DecoderBatch::TpDriver::create_peer_request(decoder::Lane& lane,
                                                 decoder::RequestBlockCache* retired) const {
    const decoder::DeviceScope peer(tp.peer_device);
    lane.peer_request = std::make_unique<decoder::Request>(tp.peer_device, false, retired);
}

void DecoderBatch::TpDriver::initialize_peer(decoder::Lane& lane, int request_capacity,
                                             int resident_limit, const decoder::PrefixSlab* slab,
                                             const decoder::ResidentPrefixRead& reuse,
                                             const decoder::PrefixSlab* draft_slab,
                                             decoder::RequestBlockCache* draft_cache) const {
    decoder::DeviceScope peer(tp.peer_device);
    lane.peer_request->initialize(request_capacity, tp.peer_stream,
                                  decoder::RequestKvStorage{
                                      resident_limit,
                                      gqa_rank_geometry.kv_heads,
                                      slab,
                                      1,
                                      lane.request->shared_prefix_pages,
                                      &reuse,
                                  });
    lane.peer_request->prefix_checkpoint_outputs = lane.checkpoints();
    lane.peer_request->reset(tp.peer_stream);
    if (lane.draft) {
        lane.peer_draft = std::make_unique<decoder::MtpDraftRequest>(
            tp.peer->mtp_weights, lane.capacity(), draft_cache,
            decoder::MtpDraftRequest::PrefixStorage{
                draft_slab,
                lane.draft->resident_prefix_pages(),
                &reuse,
                1,
            });
    }
    decoder::check(cudaStreamSynchronize(tp.peer_stream), "Initialize peer decoder batch slot");
}

void DecoderBatch::TpDriver::grow_peer_kv(decoder::Lane& lane, int positions,
                                          int written) const {
    decoder::DeviceScope peer(tp.peer_device);
    lane.peer_request->grow_kv(positions, written, tp.peer_stream);
}

bool DecoderBatch::TpDriver::sync_peer_stream() const noexcept {
    try {
        const decoder::DeviceScope peer(tp.peer_device);
        decoder::synchronize_cuda(tp.peer_stream, "Drain peer decoder stream");
        return true;
    } catch (const decoder::CudaError& error) {
        std::cerr << "sglang decoder: drain peer stream: " << error.what() << '\n';
    }
    return false;
}

void DecoderBatch::TpDriver::drain_peer(const decoder::Lane& lane) const {
    if (!lane.peer_request) return;
    decoder::DeviceScope peer(tp.peer_device);
    decoder::check(cudaStreamSynchronize(tp.peer_stream), "Drain peer decoder lane");
    lane.peer_request->memory.recycle_after_completion();
    if (lane.peer_draft && (lane.resident_bytes != 0 || lane.checkpoints()))
        lane.peer_draft->recycle_after_completion();
}

void DecoderBatch::TpDriver::reset_peer_cold(decoder::Lane& lane) const {
    if (!lane.peer_request) return;
    decoder::DeviceScope peer(tp.peer_device);
    lane.peer_request->reset(tp.peer_stream);
    if (lane.peer_draft)
        lane.peer_draft->commit_restored_prefix(0);
    decoder::check(cudaStreamSynchronize(tp.peer_stream),
                   "Reset peer decoder after failed prefix restore");
}

void DecoderBatch::TpDriver::register_prefix_restore(
    decoder::Lane& lane, const std::shared_ptr<const decoder::PrefixCacheRead>& read,
    cudaStream_t stream, const decoder::PrefixSnapshots& snapshots,
    const decoder::PrefixRestoreExtent* extent) const {
    auto view =
        decoder::restore_prefix_view(*lane.request, *read, stream, lane.draft.get(),
                                     &decoder::tp2_prefix_layouts()[0], snapshots[0], extent);

    std::optional<decoder::PendingBatchPrefix> peer;
    {
        decoder::DeviceScope peer_scope(tp.peer_device);
        peer.emplace(decoder::restore_prefix_view(
            *lane.peer_request, *read, tp.peer_stream, lane.peer_draft.get(),
            &decoder::tp2_prefix_layouts()[1], snapshots[1], extent));
    }
    lane.prefix = std::make_unique<decoder::PendingBatchPrefix>(std::move(view));
    lane.peer_prefix = std::make_unique<decoder::PendingBatchPrefix>(std::move(*peer));
    lane.prefix->completion = lane.options.restore_prefix(lane.prefix->command);
    lane.peer_prefix->completion = lane.options.restore_prefix(lane.peer_prefix->command);
}

void DecoderBatch::TpDriver::finish_peer_restore(decoder::Lane& lane) const {
    if (!lane.peer_prefix) return;
    decoder::DeviceScope peer(tp.peer_device);
    if (lane.peer_prefix->command.attention_only_restore &&
        *lane.peer_prefix->command.attention_only_restore)
        lane.peer_request->reset(tp.peer_stream);
    decoder::commit_restored_prefix(*lane.peer_request, lane.peer_prefix->command,
                                    lane.peer_draft.get(), tp.peer_stream);
}

decoder::ForwardLane DecoderBatch::TpDriver::mirror_lane(decoder::ForwardLane lane,
                                                         decoder::Request* peer_request) {
    lane.request = peer_request;
    lane.narrow_gqa = false;
    lane.singleton_scores = {};
    return lane;
}

void DecoderBatch::TpDriver::forward_decode(decoder::ForwardLane* work,
                                            decoder::ForwardLane* peer_work, int count,
                                            cudaStream_t stream, bool regular_graph, bool capture,
                                            decoder::DecodeRun& run) const {
    if (!count) return;
    const bool graphed = model.forward_tp(*tp.peer, work, peer_work, count, stream, tp.peer_stream,
                                          *tp.group, tp.partials[0], tp.partials[1], capture);
    if (graphed)
        run.graphed = (std::uint64_t{1} << static_cast<unsigned>(count)) - 1;
    if (regular_graph && model.compute_eager_decode_logits(work, count, stream)) {
        for (int index = 0; index < count; ++index) {
            run.ready[index] = &work[index].request->work;
        }
    }
}

void DecoderBatch::TpDriver::forward_single(decoder::ForwardLane* single,
                                            decoder::ForwardLane* peer_single, int count,
                                            cudaStream_t stream) const {
    if (!count) return;
    model.forward_tp(*tp.peer, single, peer_single, count, stream, tp.peer_stream, *tp.group,
                     tp.partials[0], tp.partials[1]);
}

void DecoderBatch::TpDriver::prefill(const decoder::PrefillGrant* grants,
                                     const decoder::PrefillGrant* peer_grants, int count,
                                     cudaStream_t stream,
                                     DecoderBatchStepStats& stats) const {
    model.prefill_grants_tp(*tp.peer, grants, peer_grants, count, stream, tp.peer_stream,
                            *tp.group, stats);
}

void DecoderBatch::TpDriver::prepare_peer_verification(decoder::Lane& lane, int capacity) const {
    if (lane.peer_verification) return;
    decoder::DeviceScope peer(tp.peer_device);
    lane.peer_verification = std::make_unique<decoder::TargetVerification>(
        *lane.peer_request, lane.capacity(), lane.progress.position, capacity, false,
        tp.peer_stream, nullptr, -1, true);
}

decoder::MtpRound DecoderBatch::TpDriver::peer_round(const decoder::Lane& lane, int rows) {
    return {
        lane.peer_draft.get(),
        lane.peer_verification.get(),
        lane.peer_request.get(),
        nullptr,
        nullptr,
        {lane.progress.last_token},
        lane.progress.position,
        lane.progress.generated,
        lane.remaining_output(),
        rows,
    };
}

namespace {

struct DraftProposalWave {
    int gather(int depth, decoder::MtpRound* const (&ranks)[2], int count) const;
    void finish(int depth, decoder::MtpRound* const (&ranks)[2], int size) const;

    decoder::MtpDraftInput (*inputs)[decoder::draft_projection_rows] = nullptr;
    const decoder::ConstraintCandidates** allowed = nullptr;
    int* lanes = nullptr;
};

int DraftProposalWave::gather(int depth, decoder::MtpRound* const (&ranks)[2], int count) const {
    const int wave = std::max(0, depth);
    int size = 0;
    for (int lane = 0; lane < count; ++lane) {
        const auto& root = ranks[0][lane];
        const auto& peer = ranks[1][lane];
        if (depth < 0 ? root.proposal_limit != 0 : depth >= root.proposal_limit)
            continue;
        const int position = root.position - 1 + wave;
        const int token = root.tokens[wave];
        inputs[0][size] = root.draft_input(position, token, root.draft_hidden);
        inputs[1][size] = peer.draft_input(position, token, peer.draft_hidden);
        allowed[size] =
            depth >= 0 && root.constraint != nullptr ? &root.constraint->candidates(wave) : nullptr;
        lanes[size++] = lane;
    }
    return size;
}

void DraftProposalWave::finish(int depth, decoder::MtpRound* const (&ranks)[2], int size) const {
    for (int index = 0; index < size; ++index) {
        const int lane = lanes[index];
        auto& root = ranks[0][lane];
        auto& peer = ranks[1][lane];
        root.draft_hidden = inputs[0][index].hidden;
        peer.draft_hidden = inputs[1][index].hidden;
        const int token = decoder::MtpDraftRequest::read_tp_token(*root.draft, *peer.draft);
        if (token < 0) {
            root.proposal_limit = depth;
        } else {
            root.tokens[++root.drafted] = token;
            if (root.constraint != nullptr && !root.constraint->advance(depth, token))
                root.proposal_limit = depth + 1;
        }
    }
}

}

bool DecoderBatch::TpDriver::propose(decoder::MtpRound* rounds, decoder::MtpRound* peer_rounds,
                                     int count, cudaStream_t stream,
                                     const decoder::ConstraintWorkspace& scratch,
                                     const std::function<bool()>& cancel) const {
    using namespace decoder;
    if (count < 1 || count > draft_projection_rows)
        throw std::invalid_argument("Invalid tensor-parallel draft batch width");
    const cudaStream_t streams[] = {stream, tp.peer_stream};
    const auto sync = [&] {
        synchronize_cuda(stream, "Complete primary MTP draft wave");
        const DeviceScope peer(tp.peer_device);
        synchronize_cuda(tp.peer_stream, "Complete peer MTP draft wave");
    };
    check(cudaEventRecord(tp.draft_input_ready, stream), "Record primary MTP carry readiness");
    {
        const DeviceScope peer(tp.peer_device);
        check(cudaStreamWaitEvent(tp.peer_stream, tp.draft_input_ready, 0), "Wait for MTP carry");
        for (int lane = 0; lane < count; ++lane) {
            auto& root = rounds[lane];
            auto& other = peer_rounds[lane];
            root.draft_hidden = root.request->work.normalized;
            auto* hidden = other.request->work.normalized;
            check(cudaMemcpyPeerAsync(hidden, tp.peer_device, root.draft_hidden, tp.device,
                                      hidden_size * sizeof(Bf16), tp.peer_stream),
                  "Replicate MTP carry input");
            other.draft_hidden = hidden;
        }
    }
    MtpDraftInput inputs[2][draft_projection_rows];
    const ConstraintCandidates* allowed[draft_projection_rows];
    int lanes[draft_projection_rows];
    const DraftProposalWave wave{inputs, allowed, lanes};
    MtpRound* const ranks[] = {rounds, peer_rounds};
    MtpDraftInput* const rows[] = {inputs[0], inputs[1]};
    for (int depth = -1; depth < decoder_batch_max_mtp_steps; ++depth) {
        if (depth >= 0 && cancel && cancel())
            return false;
        const int size = wave.gather(depth, ranks, count);
        if (size == 0) {
            if (depth < 0)
                continue;
            break;
        }
        MtpDraftRequest::enqueue_tp(rows, size, depth >= 0, streams, tp.partials, tp.packed,
                                    *tp.group, &scratch, allowed);
        if (depth < 0)
            continue;
        sync();
        bool retried = false;
        for (int index = 0; index < size; ++index) {
            const int lane = lanes[index];
            retried |= MtpDraftRequest::retry_invalid_tp(*rounds[lane].draft,
                                                         *peer_rounds[lane].draft, streams);
        }
        if (retried)
            sync();
        wave.finish(depth, ranks, size);
    }
    return !cancel || !cancel();
}

void DecoderBatch::TpDriver::repair(const decoder::MtpRound* rounds,
                                    const decoder::DraftRepairs* repairs,
                                    const decoder::MtpRound* peer_rounds, int count,
                                    cudaStream_t stream) const {
    using namespace decoder;
    const cudaStream_t streams[] = {stream, tp.peer_stream};
    MtpDraftInput inputs[2][draft_projection_rows];
    MtpDraftInput* const rows[] = {inputs[0], inputs[1]};
    for (int lane = 0; lane < count; ++lane) {
        auto& peer = *peer_rounds[lane].draft;
        if (peer.valid_prefix() > rounds[lane].position)
            peer.rewind(rounds[lane].position);
    }
    for (int depth = 0; depth < decoder_batch_max_mtp_steps; ++depth) {
        int size = 0;
        {
            const DeviceScope peer(tp.peer_device);
            for (int lane = 0; lane < count; ++lane) {
                const auto& input = repairs[lane][depth];
                if (input.request == nullptr)
                    continue;
                const auto& pair = peer_rounds[lane];
                auto* hidden = pair.target->checkpoint_row(depth)->normalized;
                check(cudaMemcpyPeerAsync(hidden, tp.peer_device, input.hidden, tp.device,
                                          hidden_size * sizeof(Bf16), tp.peer_stream),
                      "Replicate verified MTP repair input");
                inputs[0][size] = input;
                inputs[1][size++] = pair.draft_input(input.position, input.token, hidden);
            }
        }
        if (size == 0)
            break;
        MtpDraftRequest::enqueue_tp(rows, size, false, streams, tp.partials, tp.packed, *tp.group);
    }
}

void DecoderBatch::TpDriver::seed_draft(decoder::Lane& lane, int position, int token,
                                        cudaStream_t stream) const {
    using namespace decoder;
    auto& root = lane.request->work;
    const auto& peer = lane.peer_request->work;
    if (lane.progress.position < lane.prompt_size)
        model.normalize_final(root, stream);
    check(cudaEventRecord(tp.draft_input_ready, stream), "Record primary MTP seed input");
    {
        const DeviceScope scope(tp.peer_device);
        check(cudaStreamWaitEvent(tp.peer_stream, tp.draft_input_ready, 0), "Wait for MTP seed");
        check(cudaMemcpyPeerAsync(peer.normalized, tp.peer_device, root.normalized, tp.device,
                                  hidden_size * sizeof(Bf16), tp.peer_stream),
              "Replicate MTP seed input");
    }
    MtpDraftRequest::seed_tp({lane.draft.get(), position, token, root.normalized},
                             {lane.peer_draft.get(), position, token, peer.normalized}, *tp.group,
                             stream, tp.peer_stream);
}

bool DecoderBatch::TpDriver::verify(decoder::MtpRound* rounds, decoder::MtpRound* peer_rounds,
                                    int count, cudaStream_t stream,
                                    decoder::ConstraintWorkspace& constraints,
                                    std::byte* sampling_workspace, std::byte* logprob_workspace,
                                    const decoder::TcVocabScratch* vocabulary, bool capture) const {
    const bool wide = count > decoder_batch_max_mtp_lanes;
    return model.verify_mtp_tp(
        *tp.peer, rounds, peer_rounds, count, constraints, sampling_workspace, stream,
        tp.peer_stream, *tp.group, tp.partials[0], tp.partials[1], vocabulary,
        wide ? tp.packed[0] : nullptr, wide ? tp.packed[1] : nullptr, logprob_workspace, capture);
}

void DecoderBatch::TpDriver::commit_peer_round(decoder::Lane& lane, const decoder::MtpRound& round,
                                               const decoder::Workspace* const* saved) const {
    if (!lane.peer_verification)
        return;
    decoder::DeviceScope peer(tp.peer_device);
    lane.peer_verification->commit(round.consumed);
    if (saved != nullptr && *saved != nullptr && *saved != &lane.request->work) {
        lane.peer_checkpoint_row = lane.peer_verification->checkpoint_row(round.consumed - 1);
    }
}
}
