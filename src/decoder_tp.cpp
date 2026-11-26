#include "sglang/runtime/execution/decoder_tp.h"

#include "sglang/decoder_internal.h"
#include "sglang/decoder_prefix_cache.h"
#include "sglang/mxfp8_gqa.h"
#include "sglang/runtime/backend/cuda_resources.h"
#include "sglang/runtime/backend/tensor_parallel.h"

#include <utility>
#include <cstdlib>

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

void DecoderBatch::TpDriver::create_peer_request(decoder::Lane& lane) const {
    decoder::DeviceScope peer(tp.peer_device);
    lane.peer_request = std::make_unique<decoder::Request>(tp.peer_device, false, nullptr);
}

void DecoderBatch::TpDriver::initialize_peer(decoder::Lane& lane, int request_capacity,
                                             int resident_limit) const {
    decoder::DeviceScope peer(tp.peer_device);
    lane.peer_request->initialize(request_capacity, tp.peer_stream, resident_limit,
                                  gqa_rank_geometry.kv_heads);
    lane.peer_request->prefix_checkpoint_outputs = lane.checkpoints();
    lane.peer_request->reset(tp.peer_stream);
    decoder::check(cudaStreamSynchronize(tp.peer_stream), "Initialize peer decoder batch slot");
}

void DecoderBatch::TpDriver::grow_peer_kv(decoder::Lane& lane, int positions,
                                          int written) const {
    decoder::DeviceScope peer(tp.peer_device);
    lane.peer_request->grow_kv(positions, written, tp.peer_stream);
}

void DecoderBatch::TpDriver::sync_peer_stream() const noexcept {
    decoder::DeviceScope peer(tp.peer_device);
    (void)cudaStreamSynchronize(tp.peer_stream);
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
    cudaStream_t stream) const {
    auto view = decoder::restore_prefix_view(*lane.request, *read, stream, lane.draft.get(),
                                             &decoder::tp2_prefix_layouts()[0]);
    
    auto peer_read = std::make_shared<decoder::PrefixCacheRead>(*read);
    peer_read->usable_draft = 0;
    std::optional<decoder::PendingBatchPrefix> peer;
    {
        decoder::DeviceScope peer_scope(tp.peer_device);
        peer.emplace(decoder::restore_prefix_view(*lane.peer_request, *peer_read, tp.peer_stream,
                                                  nullptr,
                                                  &decoder::tp2_prefix_layouts()[1]));
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
    decoder::commit_restored_prefix(*lane.peer_request, lane.peer_prefix->command, nullptr);
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
    constexpr std::size_t elements = std::size_t(decoder::draft_projection_rows) *
        decoder::target_verify_max_rows * decoder::hidden_size;
    if (!decoder::PeerDraftRepair::eligible(rounds, count) || !model.is_b200() ||
        !tp.peer->is_b200() || !tp.group->supports_mnnvl_twoshot(elements)) return false;
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
    auto* repair = allow_peer_repair && !capture && count == decoder::draft_projection_rows
        ? peer_repair.get() : nullptr;
    (void)model.verify_mtp_tp(*tp.peer, rounds, peer_rounds, count, constraints,
                              sampling_workspace, stream, tp.peer_stream, *tp.group,
                              tp.partials[0], tp.partials[1], vocabulary, primary_packed,
                              peer_packed, logprob_workspace, capture, repair);
}

TextGdnReplayBatch* DecoderBatch::TpDriver::prepare_peer_gdn_replays() const {
    
    
    
    
    static const bool batch_disabled =
        std::getenv("SGLANG_DISABLE_GDN_REPLAY_BATCH") != nullptr;
    if (batch_disabled || peer_replays_broken) return nullptr;
    if (peer_replays.capacity() == 0) {
        decoder::DeviceScope peer(tp.peer_device);
        if (!peer_replays.reserve(decoder::gdn_layers * decoder::max_wide_mtp_lanes)) {
            peer_replays_broken = true;
            return nullptr;
        }
    }
    peer_replays.reset();
    return &peer_replays;
}

void DecoderBatch::TpDriver::finish_peer_gdn_replays() const {
    if (peer_replays_broken || peer_replays.size() == 0) return;
    decoder::DeviceScope peer(tp.peer_device);
    decoder::check(peer_replays.launch(true, tp.peer_stream), "Batched peer GDN prefix replay");
}

void DecoderBatch::TpDriver::commit_peer_round(decoder::Lane& lane,
                                               const decoder::MtpRound& round,
                                               const decoder::Workspace* const* saved,
                                               TextGdnReplayBatch* gdn_batch) const {
    if (!lane.peer_verification) return;
    decoder::DeviceScope peer(tp.peer_device);
    lane.peer_verification->commit(round.consumed, gdn_batch);
    if (saved && *saved != &lane.request->work) {
        
        
        lane.peer_checkpoint_row = lane.peer_verification->checkpoint_row(round.consumed - 1);
    }
}

void DecoderBatch::TpDriver::store_peer_checkpoint(
    decoder::Lane& lane, int end, int generated, int count, bool terminal,
    const std::shared_ptr<void>& reservation,
    const std::shared_ptr<std::atomic_uint64_t>& acknowledged) const {
    if (!lane.peer_request) return;
    
    
    decoder::DeviceScope peer(tp.peer_device);
    auto peer_view =
        decoder::export_prefix_view(*lane.peer_request, 0, 0, end, tp.peer_stream, nullptr,
                                    lane.peer_checkpoint_row, terminal,
                                    &decoder::tp2_prefix_layouts()[1]);
    lane.options.store_prefix(std::move(peer_view),
                              count ? lane.prompt.data() + generated : nullptr, count,
                              reservation, acknowledged);
}

}
