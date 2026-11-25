#include "sglang/runtime/execution/decoder_batch_state.h"

#include "sglang/decoder_prefix_cache.h"
#include "sglang/runtime/backend/cuda_resources.h"
#include <algorithm>
#include <utility>

namespace sglang {

using namespace decoder;
using namespace decoder::batch_detail;

void DecoderBatch::release_retired_allocations() try {
    state_->storage->retired.clear();
} catch (...) {
    state_->disable();
    throw;
}

DecoderPrefixCacheStats DecoderBatch::prefix_cache_stats() const {
    auto stats = state_->prefix_stats;
    if (state_->acknowledged_stores) stats.stores = state_->acknowledged_stores->load();
    return stats;
}

void DecoderBatch::reset_prefix_cache() {
    state_->require_ready();
    auto& storage = *state_->storage;
    if (std::any_of(storage.slots.begin(), storage.slots.end(),
                    [](const auto& slot) { return slot.has_value(); })) {
        throw std::logic_error("Cannot reset prefix cache with active decoder slots");
    }
}

DecoderBatchPrefixUpdate DecoderBatch::poll_prefix(int slot) {
    auto& lane = state_->slot_at(slot);
    if (!lane) return {};
    try {
        DeviceScope device(state_->model.device);
        if (lane->prefix_store) {
            const bool terminal = lane->prefix_store->terminal;
            if (!state_->submit_checkpoint(*lane)) return {true, {}};
            if (terminal) State::retire(lane, state_->tp_driver.get());
            return {};
        }
        if (!lane->prefix) return {};
        const auto& prefix = *lane->prefix;
        if (!prefix.ready()) return {true, {}};
        if (prefix.discard) {
            State::retire(lane, state_->tp_driver.get());
            return {};
        }
        bool transferred = false;
        try {
            transferred = prefix.completion.get();
        } catch (const CudaError&) {
            throw;
        } catch (...) {
            // The transfer has ended. Reset any partially restored state before cold prefill.
        }
        if (transferred) {
            const int restored = commit_restored_prefix(*lane->request, prefix.command,
                                                        lane->draft.get());
            if (state_->tp_driver) state_->tp_driver->finish_peer_restore(*lane);
            state_->finish_restore(*lane, restored);
        } else {
            const auto stream = state_->execution.stream();
            lane->request->reset(stream);
            check(cudaStreamSynchronize(stream), "Reset decoder after failed prefix restore");
            // A failed joint GET may have dirtied either rank; both reset cold.
            if (state_->tp_driver) state_->tp_driver->reset_peer_cold(*lane);
        }
        lane->prefix.reset();
        lane->peer_prefix.reset();
        return {false, lane->progress.position};
    } catch (const CudaError& error) {
        if (CudaError::terminal(error.status)) state_->disable();
        throw;
    }
}

void DecoderBatch::State::publish_checkpoint(Lane& lane, DecoderBatchStep& result,
                                             const Workspace* row, bool& publishing) {
    const int end = lane.progress.position;
    if (!lane.request || end <= lane.checkpoint ||
        result.finish_reason == DecoderFinishReason::cancelled) return;
    const bool terminal = result.finish_reason.has_value();
    if (!terminal && end % decoder_batch_prefix_checkpoint_interval != 0) return;
    if (terminal && lane.options.claim_completion && !lane.options.claim_completion()) {
        result.finish_reason = DecoderFinishReason::cancelled;
        return;
    }
    lane.prefix_store.emplace(DeferredPrefixStore{row, nullptr, terminal});
    publishing = true;
    result.prefix_pending = !submit_checkpoint(lane);
    publishing = false;
}

bool DecoderBatch::State::submit_checkpoint(Lane& lane) {
    auto& checkpoint = *lane.prefix_store;
    const auto& opts = lane.options;
    if (opts.reserve_prefix_store && !opts.reserve_prefix_store(checkpoint.reservation)) {
        return false;
    }
    auto reservation = std::move(checkpoint.reservation);
    // Both rank lanes share one reservation ticket: the request allocations
    // stay reserved until BOTH physical stores acknowledge, so a lagged
    // rank-1 daemon copy cannot outlive the pending-store gate.
    const auto peer_reservation = reservation;
    const int end = lane.progress.position;
    const int generated = std::max(lane.prompt_size, lane.checkpoint);
    const int count = std::max(0, end - generated);
    auto view = export_prefix_view(*lane.request, 0, 0, end, execution.stream(),
                                   lane.draft.get(), checkpoint.row, checkpoint.terminal,
                                   tp_driver ? &decoder::tp2_prefix_layouts()[0] : nullptr);
    if (!acknowledged_stores) acknowledged_stores = std::make_shared<std::atomic_uint64_t>(0);
    opts.store_prefix(std::move(view), count ? lane.prompt.data() + generated : nullptr, count,
                      std::move(reservation), acknowledged_stores);
    if (tp_driver && lane.peer_request) {
        tp_driver->store_peer_checkpoint(lane, end, generated, count, checkpoint.terminal,
                                         peer_reservation, acknowledged_stores);
    }
    lane.checkpoint = end;
    lane.prefix_store.reset();
    return true;
}

}
