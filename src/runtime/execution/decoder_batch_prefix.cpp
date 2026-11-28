#include "sglang/runtime/execution/decoder_batch_state.h"

#include "sglang/decoder_prefix_cache.h"
#include "sglang/runtime/backend/cuda_resources.h"
#include "sglang/runtime/cache/prefix_payload.h"
#include "sglang/runtime/cache/prefix_cache_layout.h"
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
            if (!state_->submit_checkpoint(*lane)) return {true, {}};
            State::retire(lane, state_->tp_driver.get());
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
        }
        if (transferred) {
            if (prefix.command.attention_only_restore && *prefix.command.attention_only_restore) {
                lane->request->reset(state_->execution.stream());
                lane->prefill_lookup.reset();
            }
            const int restored = commit_restored_prefix(*lane->request, prefix.command,
                                                        lane->draft.get());
            if (state_->tp_driver) state_->tp_driver->finish_peer_restore(*lane);
            state_->finish_restore(*lane, restored);
        } else {
            lane->prefill_lookup.reset();
            const auto stream = state_->execution.stream();
            lane->request->reset(stream);
            check(cudaStreamSynchronize(stream), "Reset decoder after failed prefix restore");
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
    if (!lane.request || !result.finish_reason ||
        result.finish_reason == DecoderFinishReason::cancelled)
        return;
    if (lane.options.claim_completion && !lane.options.claim_completion()) {
        result.finish_reason = DecoderFinishReason::cancelled;
        return;
    }
    if (end <= lane.checkpoint)
        return;
    lane.prefix_store.emplace(DeferredPrefixStore{row, nullptr});
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
    store_checkpoint(lane, lane.progress.position, checkpoint.row, true,
                     std::move(checkpoint.reservation));
    lane.prefix_store.reset();
    return true;
}

void DecoderBatch::State::plan_prefill_checkpoints(Lane& lane) {
    static_assert(text_gdn_max_prefill_checkpoints == prefix_prefill_retention_limit);
    lane.prefill_endpoint_count = 0;
    const auto read = std::exchange(lane.prefill_lookup, {});
    constexpr int first_distance = 128;
    const int remaining = lane.prompt_size - lane.progress.position;
    for (int distance = 0; distance < remaining;
         distance = distance == 0 ? first_distance : distance * 2) {
        const int end = lane.prompt_size - distance;
        ++prefix_stats.prefill_checkpoints_planned;
        const bool resident = read && read->usable_gqa >= end &&
                              (end % prefix_gqa_chunk_tokens == 0 || read->usable_gqa == end) &&
                              std::binary_search(read->matched_gdn_endpoints.begin(),
                                                 read->matched_gdn_endpoints.end(), end);
        const bool draft_ready =
            !lane.draft || (read && std::binary_search(read->draft_ready_gdn_endpoints.begin(),
                                                       read->draft_ready_gdn_endpoints.end(), end));
        if (resident && draft_ready) {
            ++prefix_stats.prefill_checkpoints_resident;
            lane.checkpoint = std::max(lane.checkpoint, end);
        } else {
            lane.prefill_endpoints[lane.prefill_endpoint_count++] = end;
        }
    }
    std::reverse(lane.prefill_endpoints.begin(),
                 lane.prefill_endpoints.begin() + lane.prefill_endpoint_count);
}

bool DecoderBatch::State::PrefillCheckpointBatch::require(const PrefixPrefillPool& pool,
                                                          std::size_t capacity, std::size_t count) {
    if (pool.available(capacity) >= count)
        return true;
    blocked_pool = &pool;
    blocked_capacity = capacity;
    blocked_count = count;
    return false;
}

DecoderBatch::State::PrefillCheckpointBatch
DecoderBatch::State::prepare_prefill_checkpoints(PrefillRows& rows) {
    PrefillCheckpointBatch batch;
    for (int slot = 0; slot < options.max_lanes; ++slot) {
        const auto& lane = storage->slots[slot];
        if (rows[slot] > 0 && lane && lane->prefill_endpoint_count > 0 &&
            !prepare_lane_checkpoints(*lane, slot, batch, rows[slot])) {
            rows[slot] = 0;
        }
    }
    return batch;
}

bool DecoderBatch::State::prepare_lane_checkpoints(const Lane& lane, int slot,
                                                   PrefillCheckpointBatch& batch, int rows) {
    const auto capacity =
        static_cast<std::size_t>(options.max_lanes) * text_gdn_max_prefill_checkpoints;
    const auto limit = capacity - options.max_lanes;
    const int begin = lane.progress.position;
    const int end = begin + rows;
    const auto* const first =
        std::upper_bound(lane.prefill_endpoints.begin(),
                         lane.prefill_endpoints.begin() + lane.prefill_endpoint_count, begin);
    const auto* const last =
        std::upper_bound(first, lane.prefill_endpoints.begin() + lane.prefill_endpoint_count, end);
    if (first == last)
        return true;
    const auto required = static_cast<std::size_t>(last - first);
    const auto before_exact = *std::prev(last) == lane.prompt_size ? required - 1 : required;
    const auto fits = [&](const PrefixPrefillPool& pool) {
        return batch.require(pool, limit, before_exact) && batch.require(pool, capacity, required);
    };
    if (!fits(prefill_checkpoints) || (tp != nullptr && !fits(peer_prefill_checkpoints))) {
        return false;
    }
    for (const auto* endpoint = first; endpoint != last; ++endpoint) {
        const auto bound = *endpoint == lane.prompt_size ? capacity : limit;
        acquire_prefill_checkpoint(batch, slot, *endpoint, bound);
    }
    return true;
}

void DecoderBatch::State::acquire_prefill_checkpoint(PrefillCheckpointBatch& batch, int slot,
                                                     int end, std::size_t capacity) {
    const auto* layouts = tp != nullptr ? decoder::tp2_prefix_layouts() : nullptr;
    auto local = prefill_checkpoints.acquire(capacity, layouts);
    std::shared_ptr<PrefixPrefillSnapshot> peer;
    if (local && tp != nullptr) {
        const DeviceScope device(tp->peer_device);
        peer = peer_prefill_checkpoints.acquire(capacity, &layouts[1]);
    }
    if (!local || (tp != nullptr && !peer)) {
        ++prefix_stats.prefill_checkpoints_declined;
        return;
    }
    const auto at = batch.counts[slot]++;
    batch.local[slot][at] = {end, std::move(local)};
    batch.peer[slot][at] = {end, std::move(peer)};
}

void DecoderBatch::State::publish_prefill_checkpoint_at(Lane& lane,
                                                        const PrefillCheckpointBatch& batch,
                                                        int slot, int end) {
    for (int index = 0; index < batch.counts[slot]; ++index) {
        const auto& local = batch.local[slot][index];
        if (local.end == end) {
            publish_prefill_checkpoint(lane, local, tp_driver ? &batch.peer[slot][index] : nullptr);
            return;
        }
    }
}

void DecoderBatch::State::publish_prefill_checkpoint(Lane& lane, const PrefillCheckpoint& local,
                                                     const PrefillCheckpoint* peer) {
    store_checkpoint(lane, local.end, nullptr, false, {}, &local, peer);
    lane.prefill_captures[lane.prefill_capture_count++] = local.end;
    ++prefix_stats.prefill_checkpoints_published;
}

void DecoderBatch::State::store_checkpoint(Lane& lane, int end, const Workspace* row, bool terminal,
                                           std::shared_ptr<void> reservation,
                                           const PrefillCheckpoint* local,
                                           const PrefillCheckpoint* peer) {
    const auto peer_reservation = reservation;
    const int generated = std::max(lane.prompt_size, lane.checkpoint);
    const int count = std::max(0, end - generated);
    const auto* draft = lane.draft.get();
    if (local != nullptr && draft != nullptr && draft->valid_prefix() < end - 1)
        draft = nullptr;
    const auto* layouts = tp_driver ? decoder::tp2_prefix_layouts() : nullptr;
    if (terminal && end > lane.prompt_size)
        lane.request->materialize_gdn(execution.stream());
    auto view = export_prefix_view(*lane.request, 0, 0, end, execution.stream(), draft, row,
                                   terminal, tp_driver ? &layouts[0] : nullptr,
                                   local != nullptr ? local->snapshot : nullptr);
    view.retention_endpoints = lane.prefill_captures;
    view.retention_endpoint_count = lane.prefill_capture_count;
    view.retention_requires_draft = lane.prefill_capture_count != 0 && lane.draft != nullptr;
    if (!acknowledged_stores) acknowledged_stores = std::make_shared<std::atomic_uint64_t>(0);
    lane.options.store_prefix(std::move(view), count > 0 ? lane.prompt.data() + generated : nullptr,
                              count, std::move(reservation), acknowledged_stores);
    if (tp_driver && lane.peer_request) {
        const DeviceScope device(tp->peer_device);
        if (terminal && end > lane.prompt_size)
            lane.peer_request->materialize_gdn(tp->peer_stream);
        auto peer_view = export_prefix_view(*lane.peer_request, 0, 0, end, tp->peer_stream, nullptr,
                                            lane.peer_checkpoint_row, terminal, &layouts[1],
                                            peer != nullptr ? peer->snapshot : nullptr);
        peer_view.retention_endpoints = lane.prefill_captures;
        peer_view.retention_endpoint_count = lane.prefill_capture_count;
        lane.options.store_prefix(std::move(peer_view),
                                  count > 0 ? lane.prompt.data() + generated : nullptr, count,
                                  peer_reservation, acknowledged_stores);
    }
    lane.checkpoint = std::max(lane.checkpoint, end);
}
}
