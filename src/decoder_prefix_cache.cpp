#include "sglang/decoder_prefix_cache.h"
#include "sglang/decoder_media_internal.h"
#include "sglang/decoder_mtp.h"
#include "sglang/runtime/cache/mxfp8_prefix_copy.h"

#include <algorithm>
#include <bitset>
#include <utility>

namespace sglang::decoder {
namespace {

constexpr int max_pages = decoder_batch_prefix_max_tokens / mxfp8_page_size;
static_assert(decoder_batch_prefix_max_tokens % mxfp8_page_size == 0);

void drain_snapshot_failure(cudaStream_t stream, const char* operation) {
    const auto pending = cudaGetLastError();
    check(cudaStreamSynchronize(stream), operation);
    if (CudaError::terminal(pending)) check(pending, operation);
}

template <typename Copy>
void copy_page_runs(const GpuPrefixSnapshotStore::PageMapping& mapping, int pages, Copy copy) {
    for (int logical = 0; logical < pages;) {
        const int physical = mapping.physical[logical];
        int length = 1;
        while (logical + length < pages &&
               mapping.physical[logical + length] == physical + length) {
            ++length;
        }
        copy(logical, physical, length);
        logical += length;
    }
}

void copy_layers(std::uint8_t* dst, int dst_pages, int dst_page,
                 const std::uint8_t* src, int src_pages, int src_page,
                 std::size_t width, std::size_t page_bytes, cudaStream_t stream) {
    check(cudaMemcpy2DAsync(dst + dst_page * page_bytes, dst_pages * page_bytes,
                            src + src_page * page_bytes, src_pages * page_bytes,
                            width, full_attention_layers, cudaMemcpyDeviceToDevice,
                            stream), "Copy prefix KV layers");
}

void copy_kv(Mxfp8KvCache dst, int dst_page, Mxfp8KvCache src, int src_page,
             int rows, cudaStream_t stream) {
    const auto payload_width = rows * (mxfp8_payload_bytes_per_page / mxfp8_page_size);
    const auto payload = [&](auto* to, const auto* from) {
        copy_layers(to, dst.page_count, dst_page, from, src.page_count, src_page,
                    payload_width, mxfp8_payload_bytes_per_page, stream);
    };
    const auto scales = [&](auto* to, const auto* from) {
        if (rows % mxfp8_page_size == 0) {
            copy_layers(to, dst.page_count, dst_page, from, src.page_count, src_page,
                        rows / mxfp8_page_size * mxfp8_scale_bytes_per_page,
                        mxfp8_scale_bytes_per_page, stream);
        } else {
            copy_mxfp8_scale_tail(to + dst_page * mxfp8_scale_bytes_per_page, dst.page_count,
                                  from + src_page * mxfp8_scale_bytes_per_page, src.page_count,
                                  full_attention_layers, rows, stream, "Copy prefix KV scales");
        }
    };
    payload(dst.key_payload, src.key_payload);
    payload(dst.value_payload, src.value_payload);
    scales(dst.key_scales, src.key_scales);
    scales(dst.value_scales, src.value_scales);
}

void copy_target_prefix(Mxfp8KvCache snapshot, Mxfp8KvCache live,
                        const GpuPrefixSnapshotStore::PageMapping& mapping, int tokens,
                        bool restoring, cudaStream_t stream) {
    const auto copy = [&](int logical, int physical, int rows) {
        if (restoring) copy_kv(live, physical, snapshot, logical, rows, stream);
        else copy_kv(snapshot, logical, live, physical, rows, stream);
    };
    const int pages = tokens / mxfp8_page_size;
    copy_page_runs(mapping, pages, [&](int logical, int physical, int count) {
        copy(logical, physical, count * mxfp8_page_size);
    });
    if (const int tail = tokens % mxfp8_page_size) copy(pages, mapping.physical[pages], tail);
}

template <typename Element>
void copy_state(Element* dst, const Element* src, std::size_t elements, cudaStream_t stream) {
    check(cudaMemcpyAsync(dst, src, elements * sizeof(Element), cudaMemcpyDeviceToDevice, stream),
          "Copy prefix decoder state");
}

void copy_gdn(TextGdnState dst, TextGdnState src, cudaStream_t stream) {
    copy_state(dst.history, src.history, gdn_layers * gdn_history_elements, stream);
    copy_state(dst.recurrent, src.recurrent, gdn_layers * gdn_state_elements, stream);
}

std::size_t kv_bytes(int layers, int pages) {
    return 2 * layers * pages * (mxfp8_payload_bytes_per_page + mxfp8_scale_bytes_per_page);
}

constexpr auto target_state_bytes = gdn_layers * (gdn_history_elements * sizeof(Bf16) +
    gdn_state_elements * sizeof(float)) + 2 * hidden_size * sizeof(Bf16);

Mxfp8KvCache allocate_kv(Allocations& memory, int layers, int pages) {
    const auto payload = layers * pages * mxfp8_payload_bytes_per_page;
    const auto scales = layers * pages * mxfp8_scale_bytes_per_page;
    auto* const storage = memory.allocate<std::uint8_t>(kv_bytes(layers, pages));
    return {storage, storage + payload, storage + 2 * payload,
            storage + 2 * payload + scales, pages};
}

std::size_t available_snapshot_bytes(std::size_t reusable_bytes = 0) {
    const auto free = available_device_bytes() + reusable_bytes;
    return free - std::min(free, decoder_shared_reserve);
}

}

void copy_mxfp8_scale_tail(std::uint8_t* dst, int dst_pages, const std::uint8_t* src,
                           int src_pages, int layers, int rows, cudaStream_t stream,
                           const char* operation) {
    constexpr auto head_bytes = mxfp8_page_size * mxfp8_groups;
    constexpr auto lane_bytes = head_bytes / mxfp8_group_size;
    const auto copy = [&](std::size_t offset, std::size_t pitch, std::size_t width,
                          std::size_t height) {
        if (layers == 1) {
            check(cudaMemcpy2DAsync(dst + offset, pitch, src + offset, pitch, width, height,
                                    cudaMemcpyDeviceToDevice, stream), operation);
        } else {
            cudaMemcpy3DParms params{};
            params.srcPtr = {const_cast<std::uint8_t*>(src + offset), pitch, pitch,
                             src_pages * mxfp8_scale_bytes_per_page / pitch};
            params.dstPtr = {dst + offset, pitch, pitch,
                             dst_pages * mxfp8_scale_bytes_per_page / pitch};
            params.extent = {width, height, static_cast<std::size_t>(layers)};
            params.kind = cudaMemcpyDeviceToDevice;
            check(cudaMemcpy3DAsync(&params, stream), operation);
        }
    };
    if (rows == mxfp8_page_size - 1) {
        copy(0, head_bytes, head_bytes - mxfp8_groups, mxfp8_kv_heads);
        return;
    }
    // Each lane stores one scale row per group of 32 tokens.
    const auto complete = (rows / mxfp8_group_size) * mxfp8_groups;
    if (complete) copy(0, lane_bytes, complete, mxfp8_group_size * mxfp8_kv_heads);
    if (const int tail = rows % mxfp8_group_size) {
        for (int head = 0; head < mxfp8_kv_heads; ++head) {
            copy(head * head_bytes + complete, lane_bytes, mxfp8_groups, tail);
        }
    }
}

GpuPrefixSnapshotStore::PageMapping GpuPrefixSnapshotStore::read_request_pages(
    const Request& request, int pages, cudaStream_t stream) {
    const auto table = request.pages;
    const int physical_pages = request.caches.front().page_count;
    if (!table.device_pages || pages < 1 || pages > max_pages ||
        table.page_count < pages || table.page_count > max_request_pages ||
        physical_pages < pages || physical_pages > max_request_pages) {
        throw std::invalid_argument("Invalid prefix KV page capacity");
    }
    PageMapping mapping{{}, pages};
    try {
        check(cudaMemcpyAsync(mapping.physical.data(), table.device_pages,
                              table.page_count * sizeof(mapping.physical[0]),
                              cudaMemcpyDeviceToHost, stream), "Read prefix KV page mapping");
        check(cudaStreamSynchronize(stream), "Complete prefix KV page mapping read");
    } catch (const CudaError&) {
        drain_snapshot_failure(stream, "Drain interrupted prefix KV page mapping read");
        throw;
    }

    std::bitset<max_request_pages> used;
    for (int logical = 0; logical < pages; ++logical) {
        const int physical = mapping.physical[logical];
        if (physical < 0 || physical >= physical_pages || used.test(physical)) {
            throw std::invalid_argument("Invalid prefix KV page mapping");
        }
        used.set(physical);
    }
    for (int logical = pages; logical < table.page_count; ++logical) {
        const int physical = mapping.physical[logical];
        if (physical >= 0 && physical < physical_pages && used.test(physical)) {
            throw std::invalid_argument("Prefix KV page aliases a suffix page");
        }
    }
    return mapping;
}

GpuPrefixSnapshotStore::KvAllocation::KvAllocation(int device, int layers, int pages)
    : memory(device), kv(allocate_kv(memory, layers, pages)) {}

GpuPrefixSnapshotStore::Target::Target(int device, int pages)
    : KvAllocation(device, full_attention_layers, pages),
      gdn{memory.allocate<Bf16>(gdn_layers * gdn_history_elements),
          memory.allocate<float>(gdn_layers * gdn_state_elements)},
      hidden(memory.allocate<Bf16>(hidden_size)),
      residual(memory.allocate<Bf16>(hidden_size)) {}

GpuPrefixSnapshotStore::GpuPrefixSnapshotStore(int device, int capacity, bool with_draft)
    : device_(device), with_draft_(with_draft) {
    if (capacity < 1 || capacity > decoder_batch_prefix_max_entries) {
        throw std::invalid_argument("Invalid prefix cache capacity");
    }
}

std::size_t GpuPrefixSnapshotStore::device_bytes() const {
    std::size_t bytes = draft_device_bytes();
    for (const auto& slot : slots_) {
        if (slot.target) bytes += slot.target->memory.bytes();
    }
    return bytes;
}

std::size_t GpuPrefixSnapshotStore::draft_device_bytes() const {
    std::size_t bytes = 0;
    for (const auto& slot : slots_) {
        if (slot.draft) bytes += slot.draft->memory.bytes();
    }
    return bytes;
}

std::optional<GpuPrefixSnapshotStore::Snapshot> GpuPrefixSnapshotStore::prepare(
    int slot, int target_pages, int draft_pages) const {
    const auto& old = slots_[slot];
    const bool target_needed = !old.target || old.target->kv.page_count != target_pages;
    const bool draft_needed =
        draft_pages && (!old.draft || old.draft->kv.page_count != draft_pages);
    const auto bytes = (target_needed ? kv_bytes(full_attention_layers, target_pages) +
                       target_state_bytes : 0) + (draft_needed ? kv_bytes(1, draft_pages) : 0);
    Snapshot prepared;
    if (!bytes) return prepared;
    DeviceScope device(device_);
    if (bytes > available_snapshot_bytes()) return std::nullopt;
    try {
        if (target_needed) prepared.target = std::make_unique<Target>(device_, target_pages);
        if (draft_needed) prepared.draft = std::make_unique<KvAllocation>(device_, 1, draft_pages);
    } catch (const CudaError& error) {
        if (error.status != cudaErrorMemoryAllocation) throw;
        const auto pending = cudaGetLastError();
        if (pending != cudaErrorMemoryAllocation) {
            check(pending, "Check CUDA error after prefix snapshot allocation");
        }
        return std::nullopt;
    }
    return prepared;
}

void GpuPrefixSnapshotStore::release(int slot) { slots_[slot] = {}; }

void GpuPrefixSnapshotStore::clear() { slots_ = {}; }

void GpuPrefixSnapshotStore::store(int slot, Snapshot prepared, const Request& request,
                                  const PageMapping& mapping, int tokens, cudaStream_t stream,
                                  const MtpDraftRequest* draft, int draft_rows) {
    auto& snapshot = slots_[slot];
    if (prepared.target) snapshot.target = std::move(prepared.target);
    if (prepared.draft || draft_rows <= 0) snapshot.draft = std::move(prepared.draft);
    const auto& target = *snapshot.target;
    copy_target_prefix(target.kv, request.caches.front(), mapping, tokens, false, stream);
    copy_gdn(target.gdn, request.states.front(), stream);
    copy_state(target.hidden, request.work.hidden, hidden_size, stream);
    copy_state(target.residual, request.work.residual, hidden_size, stream);
    if (draft_rows > 0) {
        draft->snapshot_prefix(snapshot.draft->kv, draft_rows, stream);
    }
    check(cudaStreamSynchronize(stream), "Complete decoder prefix checkpoint");
}

void GpuPrefixSnapshotStore::restore(int slot, Request& request, int tokens, bool restore_workspace,
                                    cudaStream_t stream, MtpDraftRequest* draft,
                                    int draft_rows) const {
    const auto& snapshot = slots_[slot];
    const auto& target = *snapshot.target;
    const auto mapping = read_request_pages(request, prefix_kv_pages(tokens), stream);
    try {
        copy_target_prefix(target.kv, request.caches.front(), mapping, tokens, true, stream);
        copy_gdn(request.states.front(), target.gdn, stream);
        if (restore_workspace) {
            copy_state(request.work.hidden, target.hidden, hidden_size, stream);
            copy_state(request.work.residual, target.residual, hidden_size, stream);
        }
        if (draft && draft_rows > 0) {
            draft->restore_prefix(snapshot.draft->kv, draft_rows, stream);
        } else {
            check(cudaStreamSynchronize(stream), "Complete decoder prefix restoration");
        }
    } catch (...) {
        drain_snapshot_failure(stream, "Drain interrupted prefix restoration");
        throw;
    }
}

std::optional<PrefixMediaKey> PrefixMediaKey::capture(const MultimodalInput* media, int tokens) {
    PrefixMediaKey key;
    if (!media) return key;
    const auto& prompt = media->prompt;
    if (std::size_t(tokens) > prompt.positions.size()) return std::nullopt;
    for (const auto& span : prompt.spans) {
        if (span.token_begin >= tokens) break;
        if (span.media_index >= media->cache_identities.size()) return std::nullopt;
        const auto& identity = media->cache_identities[span.media_index];
        if (!identity) return std::nullopt;
        key.spans_.push_back({span.token_begin, std::min(span.rows, tokens - span.token_begin),
                              span.feature_begin, identity});
    }
    key.positions_.assign(prompt.positions.begin(), prompt.positions.begin() + tokens);
    return key;
}

bool PrefixMediaKey::matches(const MultimodalInput* media, int tokens) const {
    if (!media) return positions_.empty();
    if (positions_.empty()) return false;
    const auto& prompt = media->prompt;
    if (positions_.size() > prompt.positions.size()) return false;
    std::size_t index = 0;
    for (const auto& span : prompt.spans) {
        if (span.token_begin >= tokens) break;
        if (index == spans_.size() || span.media_index >= media->cache_identities.size()) {
            return false;
        }
        const auto& saved = spans_[index++];
        const auto& identity = media->cache_identities[span.media_index];
        if (saved.token_begin != span.token_begin || saved.feature_begin != span.feature_begin ||
            saved.rows != std::min(span.rows, tokens - span.token_begin) ||
            !identity || !saved.identity->same_content(*identity)) {
            return false;
        }
    }
    return index == spans_.size() && std::equal(positions_.begin(), positions_.end(),
        prompt.positions.begin(), [](const MropePosition& left, const MropePosition& right) {
            return left.temporal == right.temporal && left.height == right.height &&
                left.width == right.width;
        });
}

PrefixCache::PrefixCache(int device, int capacity, bool with_draft)
    : snapshots_(device, capacity, with_draft), index_(capacity) {}

int PrefixCache::checkpoint(int tokens) {
    return std::clamp(tokens, 0, decoder_batch_prefix_max_tokens);
}

int PrefixCache::restore(Request& request, const std::vector<int>& prompt, std::string_view salt,
                         cudaStream_t stream, MtpDraftRequest* draft, int max_tokens,
                         const MultimodalInput* media) {
    const int limit = checkpoint(std::min(static_cast<int>(prompt.size()), max_tokens));
    if (!limit) return 0;
    const auto match = index_.find(prompt, salt, 1, limit, draft != nullptr, media);
    if (match.slot == -1) {
        counters_.draft_misses += match.draft_miss;
        return 0;
    }
    if (request.kv_limit && prefix_kv_pages(match.tokens) > request.pages.page_count) {
        request.grow_kv(match.tokens, 0, stream);
    }
    snapshots_.restore(match.slot, request, match.tokens,
                       draft || match.tokens == static_cast<int>(prompt.size()), stream,
                       draft, match.draft_rows);
    if (draft) {
        counters_.draft_cached_rows += match.draft_rows;
    }
    index_.touch(match);
    ++counters_.hits;
    counters_.cached_tokens += match.tokens;
    return match.tokens;
}

void PrefixCache::store(const Request& request, const std::vector<int>& prompt,
                        std::string_view salt, int tokens, cudaStream_t stream,
                        const MtpDraftRequest* draft, const MultimodalInput* media) {
    if (tokens < 1 || tokens > decoder_batch_prefix_max_tokens ||
        std::size_t(tokens) > prompt.size() || salt.size() > decoder_prefix_max_salt_bytes ||
        (draft && (!snapshots_.supports_draft() || draft->valid_prefix() < tokens - 1))) {
        throw std::invalid_argument("Invalid prefix cache checkpoint");
    }
    const auto exact = index_.find(prompt, salt, tokens, tokens, false, media);
    const int draft_rows = draft ? tokens - 1 : -1;
    if (index_.touch_if_covered(exact, draft_rows)) return;
    auto media_key = PrefixMediaKey::capture(media, tokens);
    if (!media_key) return;
    const auto pages = prefix_kv_pages(tokens);
    const auto mapping = GpuPrefixSnapshotStore::read_request_pages(request, pages, stream);
    const auto write = index_.select_write(exact);
    auto prepared = snapshots_.prepare(write.slot, pages, prefix_kv_pages(std::max(draft_rows, 0)));
    if (!prepared) return;
    index_.invalidate(write.slot);
    counters_.evictions += write.evicted;
    try {
        snapshots_.store(write.slot, std::move(*prepared), request, mapping, tokens, stream,
                         draft, draft_rows);
    } catch (...) {
        drain_snapshot_failure(stream, "Drain interrupted prefix checkpoint");
        snapshots_.release(write.slot);
        throw;
    }
    ++counters_.stores;
    index_.publish(write, prompt, salt, tokens, pages, draft_rows, std::move(*media_key));
}

void PrefixCache::count_prefill(int tokens, int draft_rows) {
    counters_.prefill_tokens += tokens;
    counters_.draft_prefill_rows += draft_rows;
}

void PrefixCache::reclaim_for_admission(std::size_t bytes, std::size_t reusable_bytes) {
    int slot = index_.oldest();
    const auto recoverable = reusable_bytes + snapshots_.device_bytes();
    if (slot == -1 || bytes > available_snapshot_bytes(recoverable)) {
        return;
    }
    while (slot != -1 && bytes > available_snapshot_bytes(reusable_bytes)) {
        index_.invalidate(slot);
        snapshots_.release(slot);
        ++counters_.evictions;
        slot = index_.oldest();
    }
}

void PrefixCache::reset() {
    index_.reset();
    snapshots_.clear();
}

DecoderPrefixCacheStats PrefixCache::stats() const {
    auto stats = counters_;
    stats.device_bytes = snapshots_.device_bytes();
    stats.draft_device_bytes = snapshots_.draft_device_bytes();
    const auto occupancy = index_.occupancy();
    stats.entries = occupancy.entries;
    stats.pages = occupancy.pages;
    return stats;
}

}
