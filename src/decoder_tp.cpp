#include "sglang/runtime/execution/decoder_tp.h"

#include "sglang/decoder_internal.h"
#include "sglang/decoder_prefix_cache.h"
#include "sglang/mxfp8_gqa.h"
#include "sglang/runtime/backend/cuda_resources.h"
#include "sglang/runtime/backend/tensor_parallel.h"

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
                                             const decoder::ResidentPrefixRead& reuse) const {
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
}

void DecoderBatch::TpDriver::reset_peer_cold(decoder::Lane& lane) const {
    if (!lane.peer_request) return;
    decoder::DeviceScope peer(tp.peer_device);
    lane.peer_request->reset(tp.peer_stream);
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

    auto peer_read = std::make_shared<decoder::PrefixCacheRead>(*read);
    peer_read->usable_draft = 0;
    std::optional<decoder::PendingBatchPrefix> peer;
    {
        decoder::DeviceScope peer_scope(tp.peer_device);
        peer.emplace(decoder::restore_prefix_view(*lane.peer_request, *peer_read, tp.peer_stream,
                                                  nullptr, &decoder::tp2_prefix_layouts()[1],
                                                  snapshots[1], extent));
    }
    lane.prefix = std::make_unique<decoder::PendingBatchPrefix>(std::move(view));
    lane.peer_prefix = std::make_unique<decoder::PendingBatchPrefix>(std::move(*peer));
    lane.prefix->completion = lane.options.restore_prefix(lane.prefix->command);
    try {
        lane.peer_prefix->completion = lane.options.restore_prefix(lane.peer_prefix->command);
    } catch (...) {
        lane.prefix.reset();
        lane.peer_prefix.reset();
        throw;
    }
}

void DecoderBatch::TpDriver::finish_peer_restore(decoder::Lane& lane) const {
    if (!lane.peer_prefix) return;
    decoder::DeviceScope peer(tp.peer_device);
    if (lane.peer_prefix->command.attention_only_restore &&
        *lane.peer_prefix->command.attention_only_restore)
        lane.peer_request->reset(tp.peer_stream);
    decoder::commit_restored_prefix(*lane.peer_request, lane.peer_prefix->command, nullptr,
                                    tp.peer_stream);
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
                                            cudaStream_t stream, bool regular_graph,
                                            decoder::DecodeRun& run) const {
    if (!count) return;
    model.forward_tp(*tp.peer, work, peer_work, count, stream, tp.peer_stream, *tp.group,
                     tp.partials[0], tp.partials[1]);
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
    return {nullptr, lane.peer_verification.get(), lane.peer_request.get(), nullptr, nullptr,
            {lane.progress.last_token}, lane.progress.position, lane.progress.generated,
            lane.remaining_output(), rows};
}

bool DecoderBatch::TpDriver::prepare_peer_repair(const decoder::MtpRound* rounds, int count) {
    const std::size_t elements = std::size_t(count) * decoder::target_verify_max_rows *
                                 decoder::hidden_size;
    if (!decoder::PeerDraftRepair::eligible(rounds, count) || !model.is_b200() ||
        !tp.peer->is_b200()) return false;
    const bool transport = count == decoder::wide_mtp_lanes
        ? tp.group->supports_mnnvl_oneshot(elements) : tp.group->supports_mnnvl_twoshot(elements);
    if (!transport) return false;
    if (!peer_repair_prepared) {
        peer_repair = decoder::PeerDraftRepair::prepare(*model.mtp_weights, tp.peer_device);
        peer_repair_prepared = true;
    }
    return peer_repair != nullptr;
}

void DecoderBatch::TpDriver::verify(decoder::MtpRound* rounds, decoder::MtpRound* peer_rounds,
                                    int count, cudaStream_t stream,
                                    decoder::ConstraintWorkspace& constraints,
                                    std::byte* sampling_workspace,
                                    std::byte* logprob_workspace,
                                    const decoder::TcVocabScratch* vocabulary, bool capture,
                                    bool allow_peer_repair) const {
    std::uint8_t* const primary_packed =
        count > decoder_batch_max_mtp_lanes ? tp.packed[0] : nullptr;
    std::uint8_t* const peer_packed =
        count > decoder_batch_max_mtp_lanes ? tp.packed[1] : nullptr;
    auto* repair = allow_peer_repair && !capture &&
        decoder::PeerDraftRepair::eligible(rounds, count) ? peer_repair.get() : nullptr;
    (void)model.verify_mtp_tp(*tp.peer, rounds, peer_rounds, count, constraints,
                              sampling_workspace, stream, tp.peer_stream, *tp.group,
                              tp.partials[0], tp.partials[1], vocabulary, primary_packed,
                              peer_packed, logprob_workspace, capture, repair);
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
