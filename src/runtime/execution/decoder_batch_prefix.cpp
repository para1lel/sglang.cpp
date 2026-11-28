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

namespace {

int cacheable_draft_rows(const Lane& lane, int endpoint) {
    if (!lane.draft)
        return 0;
    const int local = std::min(endpoint - 1, lane.draft->valid_prefix());
    if (!lane.peer_request)
        return local;
    return lane.peer_draft ? std::min(local, lane.peer_draft->valid_prefix()) : 0;
}

void commit_resident_draft(Lane& lane, int rows) {
    if (lane.draft)
        lane.draft->commit_restored_prefix(rows);
    if (lane.peer_draft)
        lane.peer_draft->commit_restored_prefix(rows);
}

bool attention_only(const PrefixDeviceTransfer& transfer) {
    return transfer.attention_only_restore && *transfer.attention_only_restore;
}

}

void DecoderBatch::release_retired_allocations() try {
    auto& storage = *state_->storage;
    std::exception_ptr local_failure;
    try {
        storage.retired.clear();
    } catch (...) {
        local_failure = std::current_exception();
    }
    if (storage.peer_retired)
        storage.peer_retired->clear();
    if (local_failure)
        std::rethrow_exception(local_failure);
} catch (...) {
    state_->disable();
    throw;
}

DecoderPrefixCacheStats DecoderBatch::prefix_cache_stats() const {
    auto stats = state_->prefix_stats;
    if (state_->acknowledged_stores) stats.stores = state_->acknowledged_stores->load();
    const auto& storage = *state_->storage;
    stats.pending_release_bytes = storage.retired.pending_bytes();
    stats.failed_release_bytes = storage.retired.failed_release_bytes();
    if (storage.peer_retired) {
        stats.pending_release_bytes += storage.peer_retired->pending_bytes();
        stats.failed_release_bytes += storage.peer_retired->failed_release_bytes();
    }
    if (state_->prefix_pool)
        state_->prefix_pool->stats(stats);
    return stats;
}

bool DecoderBatch::prefix_backup_pending() const {
    return state_->prefix_pool != nullptr && state_->prefix_pool->backup_pending();
}

void DecoderBatch::reset_prefix_cache() {
    state_->require_ready();
    auto& storage = *state_->storage;
    if (std::any_of(storage.slots.begin(), storage.slots.end(),
                    [](const auto& slot) { return slot.has_value(); })) {
        throw std::logic_error("Cannot reset prefix cache with active decoder slots");
    }
    if (auto* pool = state_->prefix_pool.get(); pool != nullptr)
        pool->reset();
}

bool DecoderBatch::prefix_restore_waiting(int slot) const {
    const auto& lane = state_->slot_at(slot);
    return lane && lane->waiting_for_restore_slot;
}

DecoderBatchPrefixUpdate DecoderBatch::poll_prefix(int slot) {
    state_->require_ready();
    auto& lane = state_->slot_at(slot);
    if (!lane) return {};
    try {
        DeviceScope device(state_->model.device);
        if (lane->prefix_store) {
            if (!state_->submit_checkpoint(*lane)) return {true, {}};
            state_->retire(lane);
            return {};
        }
        if (lane->waiting_for_restore_slot) {
            state_->initialize_resident_prefix(*lane);
            if (lane->waiting_for_restore_slot)
                return {true, {}};
            if (!lane->prefix)
                return {false, lane->progress.position};
        }
        if (!lane->prefix) return {};
        const auto& prefix = *lane->prefix;
        if (!prefix.ready() || (lane->peer_prefix && !lane->peer_prefix->ready())) {
            return {true, {}};
        }
        if (prefix.discard) {
            state_->retire(lane);
            return {};
        }
        bool transferred = false;
        try {
            transferred = prefix.completion.get();
            if (lane->peer_prefix)
                transferred = lane->peer_prefix->completion.get() && transferred;
        } catch (const CudaError&) {
            throw;
        } catch (...) {
            transferred = false;
        }
        if (lane->resident_prefix) {
            state_->finish_resident_restore(*lane, transferred);
        } else if (transferred) {
            if (attention_only(prefix.command) ||
                (lane->peer_prefix && attention_only(lane->peer_prefix->command))) {
                if (prefix.command.attention_only_restore)
                    *prefix.command.attention_only_restore = true;
                if (lane->peer_prefix && lane->peer_prefix->command.attention_only_restore)
                    *lane->peer_prefix->command.attention_only_restore = true;
                lane->request->reset(state_->execution.stream());
                lane->prefill_lookup.reset();
            }
            const int restored = commit_restored_prefix(
                *lane->request, prefix.command, lane->draft.get(), state_->execution.stream());
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
        if (state_->prefix_pool)
            state_->prefix_pool->notify_released();
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
    constexpr int first_distance = 1 << 10;
    const int remaining = lane.prompt_size - lane.progress.position;
    for (int distance = 0; distance < remaining;
         distance = distance == 0 ? first_distance : distance * 2) {
        const int end = lane.prompt_size - distance;
        ++prefix_stats.prefill_checkpoints_planned;
        const bool resident = !lane.resident_prefix && read && read->usable_gqa >= end &&
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
                                                   PrefillCheckpointBatch& batch, int& rows) {
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
    if (lane.resident_prefix) {
        for (const auto* endpoint = first; endpoint != last; ++endpoint) {
            auto existing = prefix_pool->find_checkpoint(lane.options.prefix_cache_keys, *endpoint);
            if (existing.front()) {
                auto& points = batch.local[slot];
                points[points.size() - ++batch.retained[slot]] = {
                    *endpoint,
                    std::move(existing.front()),
                };
                continue;
            }
            bool transfer_pending = false;
            auto snapshots = prefix_pool->reserve_checkpoint(&transfer_pending);
            if (!snapshots.front()) {
                const auto pending = [](int count) { return count != 0; };
                const bool capture_pending =
                    std::any_of(batch.counts.begin(), batch.counts.end(), pending) ||
                    std::any_of(batch.retained.begin(), batch.retained.end(), pending);
                if (transfer_pending || capture_pending) {
                    rows = *endpoint - begin - 1;
                    batch.blocked_resident |= transfer_pending;
                    break;
                }
                ++prefix_stats.prefill_checkpoints_declined;
                continue;
            }
            const int index = batch.counts[slot]++;
            batch.local[slot][index] = {*endpoint, std::move(snapshots[0])};
            batch.peer[slot][index] = {*endpoint, std::move(snapshots[1])};
        }
        return true;
    }
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
    ++prefix_stats.prefill_checkpoints_resident;
    batch.local[slot][at] = {end, std::move(local)};
    batch.peer[slot][at] = {end, std::move(peer)};
}

void DecoderBatch::State::publish_prefill_checkpoint_at(Lane& lane,
                                                        const PrefillCheckpointBatch& batch,
                                                        int slot, int end) {
    if (lane.resident_prefix)
        return;
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
    if ((local != nullptr || tp_driver) && cacheable_draft_rows(lane, end) != end - 1)
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
    PrefixDeviceTransfer peer_view;
    if (tp_driver && lane.peer_request) {
        const DeviceScope device(tp->peer_device);
        if (terminal && end > lane.prompt_size)
            lane.peer_request->materialize_gdn(tp->peer_stream);
        peer_view = export_prefix_view(*lane.peer_request, 0, 0, end, tp->peer_stream,
                                       draft != nullptr ? lane.peer_draft.get() : nullptr,
                                       lane.peer_checkpoint_row, terminal, &layouts[1],
                                       peer != nullptr ? peer->snapshot : nullptr);
        peer_view.retention_endpoints = lane.prefill_captures;
        peer_view.retention_endpoint_count = lane.prefill_capture_count;
        peer_view.retention_requires_draft = view.retention_requires_draft;
    }
    lane.options.store_prefix(std::move(view), count > 0 ? lane.prompt.data() + generated : nullptr,
                              count, std::move(reservation), acknowledged_stores);
    if (tp_driver && lane.peer_request) {
        const DeviceScope device(tp->peer_device);
        lane.options.store_prefix(std::move(peer_view),
                                  count > 0 ? lane.prompt.data() + generated : nullptr, count,
                                  peer_reservation, acknowledged_stores);
    }
    lane.checkpoint = std::max(lane.checkpoint, end);
}

ResidentPrefixRead DecoderBatch::State::prepare_resident_read(const Lane& lane) const {
    if (!lane.resident_prefix)
        return {};
    const int maximum = lane.options.score_prompt
                            ? std::min(lane.options.prompt_logprob_start, lane.prompt_size - 1)
                            : lane.prompt_size;
    auto result =
        prefix_pool->lookup(lane.options.prefix_cache_keys, maximum, lane.options.mtp_steps != 0);
    const auto& cpu = lane.options.prefix_cache_read;
    if (cpu && lane.options.restore_prefix) {
        if (cpu->usable_gqa > result.gqa_tokens) {
            result.gqa_private_from =
                result.gqa_tokens / prefix_gqa_chunk_tokens * prefix_gqa_chunk_tokens;
        }
        if (cpu->matched_draft > result.draft_rows) {
            result.draft_private_from =
                result.draft_rows / prefix_gqa_chunk_tokens * prefix_gqa_chunk_tokens;
        }
    }
    return result;
}

void DecoderBatch::State::apply_resident_read(Lane& lane, const ResidentPrefixRead& hit) {
    lane.request->cached_gqa_tokens = hit.gqa_tokens;
    lane.resident_target_end = hit.gqa_tokens / prefix_gqa_chunk_tokens * prefix_gqa_chunk_tokens;
    lane.resident_draft_end = hit.draft_rows / prefix_gqa_chunk_tokens * prefix_gqa_chunk_tokens;
    commit_resident_draft(lane, hit.draft_rows);
    restore_resident_checkpoint(lane, hit.gdn);
    prefix_stats.gqa_matched_tokens += hit.gqa_tokens;
    prefix_stats.gdn_matched_tokens += hit.matched_gdn;
    finish_restore(lane, hit.gdn_tokens);
    prefix_pool->commit_lookup(hit, lane.options.prefix_cache_keys, hit.gdn_tokens);
}

void DecoderBatch::State::restore_resident_checkpoint(Lane& lane,
                                                      const PrefixSnapshots& snapshots) const {
    if (snapshots[0]) {
        auto* const stream = execution.stream();
        snapshots[0]->restore(*lane.request, stream);
        synchronize_cuda(stream, "Complete resident decoder checkpoint restore");
    }
    if (lane.peer_request) {
        const DeviceScope device(tp->peer_device);
        lane.peer_request->cached_gqa_tokens = lane.request->cached_gqa_tokens;
        if (snapshots[1]) {
            snapshots[1]->restore(*lane.peer_request, tp->peer_stream);
            synchronize_cuda(tp->peer_stream, "Complete resident peer checkpoint restore");
        }
    }
}

void DecoderBatch::State::initialize_resident_prefix(Lane& lane) {
    lane.waiting_for_restore_slot = false;
    if (!stage_resident_restore(lane)) {
        apply_resident_read(lane, lane.restore_fallback);
        lane.restore_fallback = {};
    }
}

bool DecoderBatch::State::stage_resident_restore(Lane& lane) const {
    auto& resident = lane.restore_fallback;
    const auto& cpu = lane.prefill_lookup;
    if (!cpu || !lane.options.restore_prefix)
        return false;
    PrefixRestoreExtent extent;
    if (cpu->usable_gqa > resident.gqa_tokens) {
        extent.gqa_begin = resident.gqa_private_from;
        extent.gqa_end = cpu->usable_gqa;
    }
    if (lane.draft && cpu->matched_draft > resident.draft_rows) {
        extent.draft_begin = resident.draft_private_from;
        extent.draft_end = cpu->matched_draft;
    }
    const int maximum = lane.options.score_prompt
                            ? std::min(lane.options.prompt_logprob_start, lane.prompt_size - 1)
                            : lane.prompt_size;
    const int target = std::max(extent.gqa_end, resident.gqa_tokens);
    const int draft = std::max(extent.draft_end, resident.draft_rows);
    const int coherent = std::min({maximum, target, lane.draft ? draft + 1 : target});
    int hot_end = 0;
    auto hot = prefix_pool->lookup_checkpoint(lane.options.prefix_cache_keys, coherent, hot_end);
    const auto cpu_end = std::upper_bound(cpu->matched_gdn_endpoints.begin(),
                                          cpu->matched_gdn_endpoints.end(), coherent);
    PrefixSnapshots staged;
    if (cpu_end != cpu->matched_gdn_endpoints.begin() && *std::prev(cpu_end) > hot_end) {
        staged = prefix_pool->reserve_checkpoint(&lane.waiting_for_restore_slot);
        if (lane.waiting_for_restore_slot)
            return true;
        if (staged.front())
            extent.gdn_end = *std::prev(cpu_end);
    }
    if (extent.gqa_end == 0 && extent.draft_end == 0 && extent.gdn_end == 0) {
        if (hot_end > resident.gdn_tokens) {
            resident.gdn = std::move(hot);
            resident.gdn_tokens = hot_end;
            resident.matched_gdn = std::max(resident.matched_gdn, hot_end);
        }
        return false;
    }
    auto* const stream = execution.stream();
    if (tp_driver) {
        tp_driver->register_prefix_restore(lane, cpu, stream, staged, &extent);
    } else {
        lane.prefix = std::make_unique<PendingBatchPrefix>(restore_prefix_view(
            *lane.request, *cpu, stream, lane.draft.get(), nullptr, staged[0], &extent));
        lane.prefix->completion = lane.options.restore_prefix(lane.prefix->command);
    }
    lane.desired_gdn = std::move(hot);
    lane.desired_gdn_tokens = hot_end;
    return true;
}

namespace {

void copy_restored_tail(const PrefixSlab& slab, int rank, const PrefixPages& source,
                        const PrefixPages& destination, const PrefixDeviceMxfp8& transfer,
                        int available_rows, cudaStream_t stream) {
    for (int row = transfer.begin; row < available_rows; row += mxfp8_page_size) {
        const int index = row / mxfp8_page_size;
        const int count = std::min(mxfp8_page_size, available_rows - row);
        slab.copy_tail(rank, {*destination[index], *source[index], count}, stream);
    }
    if (transfer.begin < available_rows) {
        synchronize_cuda(stream, "Restore resident pages after transfer miss");
    }
}

}

void DecoderBatch::State::repair_resident_restore(Lane& lane, const PrefixDeviceLayout& layout,
                                                  bool target_transferred,
                                                  bool draft_transferred) const {
    const auto& hit = lane.restore_fallback;
    auto* const stream = execution.stream();
    if (!target_transferred && layout.gqa.end > 0) {
        const auto& slab = *prefix_pool->target();
        copy_restored_tail(slab, 0, hit.gqa, lane.request->shared_prefix_pages, layout.gqa,
                           hit.gqa_tokens, stream);
        if (lane.peer_request) {
            const DeviceScope device(tp->peer_device);
            copy_restored_tail(slab, 1, hit.gqa, lane.peer_request->shared_prefix_pages, layout.gqa,
                               hit.gqa_tokens, tp->peer_stream);
        }
    }
    if (!draft_transferred && layout.draft.end > 0 && lane.draft) {
        const auto& slab = *prefix_pool->draft();
        copy_restored_tail(slab, 0, hit.draft, lane.draft->resident_prefix_pages(), layout.draft,
                           hit.draft_rows, stream);
        if (lane.peer_draft) {
            const DeviceScope device(tp->peer_device);
            copy_restored_tail(slab, 1, hit.draft, lane.peer_draft->resident_prefix_pages(),
                               layout.draft, hit.draft_rows, tp->peer_stream);
        }
    }
}

void DecoderBatch::State::finish_resident_restore(Lane& lane, bool transferred) {
    const auto& hit = lane.restore_fallback;
    prefix_stats.gqa_matched_tokens += std::max(hit.gqa_tokens, lane.prefill_lookup->matched_gqa);
    prefix_stats.gdn_matched_tokens += std::max(hit.matched_gdn, lane.prefill_lookup->matched_gdn);
    const auto& command = lane.prefix->command;
    const bool complete = transferred && !attention_only(command) &&
                          (!lane.peer_prefix || !attention_only(lane.peer_prefix->command));
    repair_resident_restore(lane, command.layout, transferred, complete);
    const bool restored_gqa = transferred && command.layout.gqa.end > 0;
    const bool restored_draft = complete && command.layout.draft.end > 0;
    const int gqa = restored_gqa ? command.layout.gqa.end : hit.gqa_tokens;
    const int draft = restored_draft ? command.layout.draft.end : hit.draft_rows;
    lane.request->cached_gqa_tokens = gqa;
    commit_resident_draft(lane, draft);
    lane.resident_target_end = hit.gqa_tokens / prefix_gqa_chunk_tokens * prefix_gqa_chunk_tokens;
    lane.resident_draft_end = hit.draft_rows / prefix_gqa_chunk_tokens * prefix_gqa_chunk_tokens;
    int end = hit.gdn_tokens;
    PrefixSnapshots snapshots = hit.gdn;
    const bool restored_gdn = complete && command.layout.gdn.end > 0;
    if (restored_gdn) {
        end = command.layout.gdn.end;
        snapshots = {
            command.restore_snapshot,
            lane.peer_prefix ? lane.peer_prefix->command.restore_snapshot : nullptr,
        };
    } else if (lane.desired_gdn_tokens >= end &&
               lane.desired_gdn_tokens <= std::min(gqa, lane.draft ? draft + 1 : gqa)) {
        end = lane.desired_gdn_tokens;
        snapshots = lane.desired_gdn;
    }
    restore_resident_checkpoint(lane, snapshots);
    finish_restore(lane, end);
    prefix_pool->commit_lookup(hit, lane.options.prefix_cache_keys, restored_gdn ? 0 : end);
    if (restored_gdn) {
        prefix_pool->publish_checkpoint(lane.options.prefix_cache_keys,
                                        lane.options.prefix_cache_backing, end, snapshots, true);
    }
    const PrefixRestoreExtent backed{
        command.layout.gqa.begin,
        restored_gqa ? gqa : 0,
        command.layout.draft.begin,
        restored_draft ? draft : 0,
        0,
    };
    publish_resident_pages(lane, gqa, draft, true, &backed);
    lane.desired_gdn = {};
    lane.restore_fallback = {};
    lane.prefill_lookup.reset();
}

void DecoderBatch::State::publish_resident_pages(Lane& lane, int target_rows, int draft_rows,
                                                 bool include_tail,
                                                 const PrefixRestoreExtent* backed) const {
    const auto& keys = lane.options.prefix_cache_keys;
    const auto& backing = lane.options.prefix_cache_backing;
    const auto publish = [&](const PrefixPages& pages, int& full, int rows, bool draft) {
        rows = std::min(rows, keys.front()->cacheable_tokens() - static_cast<int>(draft));
        const auto present = backed == nullptr ? PrefixRestoreExtent{} : *backed;
        const int backed_begin = draft ? present.draft_begin : present.gqa_begin;
        const int backed_end = draft ? present.draft_end : present.gqa_end;
        const auto store = [&](int begin, int end) {
            prefix_pool->publish_pages(keys, backing,
                                       {pages.data() + (begin / mxfp8_page_size), begin, end},
                                       draft, begin >= backed_begin && end <= backed_end);
        };
        while (full + prefix_gqa_chunk_tokens <= rows) {
            const int next = full + prefix_gqa_chunk_tokens;
            store(full, next);
            full = next;
        }
        if (include_tail && rows > 0 && rows % prefix_gqa_chunk_tokens != 0) {
            store(rows / prefix_gqa_chunk_tokens * prefix_gqa_chunk_tokens, rows);
        }
    };
    publish(lane.request->shared_prefix_pages, lane.resident_target_end, target_rows, false);
    if (lane.draft) {
        publish(lane.draft->resident_prefix_pages(), lane.resident_draft_end, draft_rows, true);
    }
}

void DecoderBatch::State::publish_resident_prefill(const PrefillRows& rows,
                                                   PrefillCheckpointBatch& entries) {
    if (!prefix_pool || !options.resident_prefix_cache)
        return;
    bool has_prefill = false;
    for (int slot = 0; slot < options.max_lanes; ++slot) {
        const auto& lane = storage->slots[slot];
        has_prefill |= rows[slot] > 0 && lane && lane->resident_prefix;
    }
    if (!has_prefill)
        return;
    synchronize_cuda(execution.stream(), "Fence resident prefill publication");
    if (tp != nullptr) {
        const DeviceScope device(tp->peer_device);
        synchronize_cuda(tp->peer_stream, "Fence resident peer prefill publication");
    }
    for (int slot = 0; slot < options.max_lanes; ++slot) {
        auto& slot_lane = storage->slots[slot];
        if (rows[slot] == 0 || !slot_lane || !slot_lane->resident_prefix)
            continue;
        auto& lane = *slot_lane;
        const int end = lane.progress.position + rows[slot];
        auto& points = entries.local[slot];
        const auto publish_pages = [&](PrefillCheckpoint& point) {
            publish_resident_pages(lane, point.end, cacheable_draft_rows(lane, point.end), true);
            ++prefix_stats.prefill_checkpoints_resident;
        };
        const auto captured_count = entries.counts[slot];
        auto reused = points.rbegin();
        const auto reused_end = reused + entries.retained[slot];
        for (int index = 0; index <= captured_count; ++index) {
            while (reused != reused_end &&
                   (index == captured_count || reused->end < points[index].end)) {
                publish_pages(*reused);
                (reused++)->snapshot.reset();
            }
            if (index == captured_count)
                break;
            auto& local = points[index];
            publish_pages(local);
            prefix_pool->publish_checkpoint(lane.options.prefix_cache_keys,
                                            lane.options.prefix_cache_backing, local.end,
                                            {local.snapshot, entries.peer[slot][index].snapshot});
            local.snapshot.reset();
            entries.peer[slot][index].snapshot.reset();
            ++prefix_stats.prefill_checkpoints_published;
        }
        const int draft = cacheable_draft_rows(lane, end);
        publish_resident_pages(lane, end, draft, end == lane.prompt_size);
    }
    prefix_pool->notify_released();
}

void DecoderBatch::State::publish_resident_terminal(Lane& lane, const Workspace* row) const {
    const int end = lane.progress.position;
    if (end <= lane.prompt_size)
        return;
    auto& keys = lane.options.prefix_cache_keys;
    for (auto& key : keys) {
        key = std::make_shared<PrefixCacheKeys>(key);
        while (key->tokens() < end)
            key->append(lane.prompt.at(key->tokens()));
    }
    const int draft = cacheable_draft_rows(lane, end);
    publish_resident_pages(lane, end, draft, true);
    if (prefix_pool->find_checkpoint(keys, end).front())
        return;
    auto snapshots = prefix_pool->reserve_checkpoint();
    if (!snapshots.front())
        return;
    const auto capture = [&](Request& request, const Workspace& work,
                             const PrefixPrefillSnapshot& snapshot, cudaStream_t stream) {
        try {
            request.materialize_gdn(stream);
            snapshot.capture_state(request.states.front(), stream);
            snapshot.capture_row(work.hidden, work.residual, stream);
            synchronize_cuda(stream, "Fence resident terminal checkpoint");
        } catch (...) {
            synchronize_cuda(stream, "Drain resident terminal checkpoint");
            throw;
        }
    };
    const auto& local = row != nullptr ? *row : lane.request->work;
    capture(*lane.request, local, *snapshots[0], execution.stream());
    if (lane.peer_request) {
        const DeviceScope device(tp->peer_device);
        auto& peer = *lane.peer_request;
        const auto* saved = lane.peer_checkpoint_row;
        capture(peer, saved != nullptr ? *saved : peer.work, *snapshots[1], tp->peer_stream);
    }
    prefix_pool->publish_checkpoint(keys, lane.options.prefix_cache_backing, end, snapshots);
}
}
