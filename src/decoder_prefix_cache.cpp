#include "sglang/decoder_prefix_cache.h"

#include "sglang/decoder_mtp.h"

#include <algorithm>
#include <bitset>

namespace sglang::decoder {
namespace {

constexpr int max_pages = decoder_batch_prefix_max_tokens / mxfp8_page_size;
static_assert(decoder_batch_prefix_max_tokens % mxfp8_page_size == 0);

template <typename Copy>
void copy_page_runs(const GpuPrefixSnapshotStore::PageMapping& mapping, Copy copy) {
    for (int logical = 0; logical < mapping.count;) {
        const int physical = mapping.physical[logical];
        int length = 1;
        while (logical + length < mapping.count &&
               mapping.physical[logical + length] == physical + length) {
            ++length;
        }
        copy(logical, physical, length);
        logical += length;
    }
}

void copy_layers(std::uint8_t* dst, int dst_pages, int dst_page,
                 const std::uint8_t* src, int src_pages, int src_page,
                 int pages, std::size_t page_bytes, cudaStream_t stream) {
    check(cudaMemcpy2DAsync(dst + dst_page * page_bytes, dst_pages * page_bytes,
                            src + src_page * page_bytes, src_pages * page_bytes,
                            pages * page_bytes, full_attention_layers, cudaMemcpyDeviceToDevice,
                            stream), "Copy prefix KV layers");
}

void copy_kv(Mxfp8KvCache dst, int dst_page, Mxfp8KvCache src, int src_page,
             int pages, cudaStream_t stream) {
    copy_layers(dst.key_payload, dst.page_count, dst_page, src.key_payload, src.page_count,
                src_page, pages, mxfp8_payload_bytes_per_page, stream);
    copy_layers(dst.value_payload, dst.page_count, dst_page, src.value_payload, src.page_count,
                src_page, pages, mxfp8_payload_bytes_per_page, stream);
    copy_layers(dst.key_scales, dst.page_count, dst_page, src.key_scales, src.page_count,
                src_page, pages, mxfp8_scale_bytes_per_page, stream);
    copy_layers(dst.value_scales, dst.page_count, dst_page, src.value_scales, src.page_count,
                src_page, pages, mxfp8_scale_bytes_per_page, stream);
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

Mxfp8KvCache allocate_kv(Allocations& memory, int layers) {
    const auto payload = layers * max_pages * mxfp8_payload_bytes_per_page;
    const auto scales = layers * max_pages * mxfp8_scale_bytes_per_page;
    return {
        memory.allocate<std::uint8_t>(payload), memory.allocate<std::uint8_t>(payload),
        memory.allocate<std::uint8_t>(scales), memory.allocate<std::uint8_t>(scales), max_pages,
    };
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
    check(cudaMemcpyAsync(mapping.physical.data(), table.device_pages,
                          table.page_count * sizeof(mapping.physical[0]),
                          cudaMemcpyDeviceToHost, stream), "Read prefix KV page mapping");
    check(cudaStreamSynchronize(stream), "Complete prefix KV page mapping read");

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

GpuPrefixSnapshotStore::Snapshot::Snapshot(Allocations& memory)
    : kv(allocate_kv(memory, full_attention_layers)),
      gdn{memory.allocate<Bf16>(gdn_layers * gdn_history_elements),
          memory.allocate<float>(gdn_layers * gdn_state_elements)},
      hidden(memory.allocate<Bf16>(hidden_size)),
      residual(memory.allocate<Bf16>(hidden_size)) {}

GpuPrefixSnapshotStore::GpuPrefixSnapshotStore(int device, int capacity, bool with_draft)
    : memory_(device) {
    if (capacity < 1 || capacity > decoder_batch_prefix_max_entries) {
        throw std::invalid_argument("Invalid prefix cache capacity");
    }
    slots_.reserve(capacity);
    for (int slot = 0; slot < capacity; ++slot) {
        auto& snapshot = slots_.emplace_back(memory_);
        if (with_draft) {
            const auto target_bytes = memory_.bytes();
            snapshot.draft = allocate_kv(memory_, 1);
            draft_device_bytes_ += memory_.bytes() - target_bytes;
        }
    }
}

std::size_t GpuPrefixSnapshotStore::device_bytes() const { return memory_.bytes(); }

std::size_t GpuPrefixSnapshotStore::draft_device_bytes() const { return draft_device_bytes_; }

void GpuPrefixSnapshotStore::store(int slot, const Request& request, const PageMapping& mapping,
                                  cudaStream_t stream, const MtpDraftRequest* draft,
                                  int draft_rows) {
    auto& snapshot = slots_[slot];
    copy_page_runs(mapping, [&](int logical, int physical, int length) {
        copy_kv(snapshot.kv, logical, request.caches.front(), physical, length, stream);
    });
    copy_gdn(snapshot.gdn, request.states.front(), stream);
    copy_state(snapshot.hidden, request.work.hidden, hidden_size, stream);
    copy_state(snapshot.residual, request.work.residual, hidden_size, stream);
    if (draft) {
        draft->snapshot_prefix(snapshot.draft, draft_rows, stream);
    }
    check(cudaStreamSynchronize(stream), "Complete decoder prefix checkpoint");
}

void GpuPrefixSnapshotStore::restore(int slot, Request& request, int pages, bool restore_workspace,
                                    cudaStream_t stream, MtpDraftRequest* draft,
                                    int draft_rows) const {
    const auto& snapshot = slots_[slot];
    const auto mapping = read_request_pages(request, pages, stream);
    copy_page_runs(mapping, [&](int logical, int physical, int length) {
        copy_kv(request.caches.front(), physical, snapshot.kv, logical, length, stream);
    });
    copy_gdn(request.states.front(), snapshot.gdn, stream);
    if (restore_workspace) {
        copy_state(request.work.hidden, snapshot.hidden, hidden_size, stream);
        copy_state(request.work.residual, snapshot.residual, hidden_size, stream);
    }
    if (draft) {
        draft->restore_prefix(snapshot.draft, draft_rows, stream);
    } else {
        check(cudaStreamSynchronize(stream), "Complete decoder prefix restoration");
    }
}

PrefixCache::PrefixCache(int device, int capacity, bool with_draft)
    : snapshots_(device, capacity, with_draft), index_(capacity) {}

int PrefixCache::checkpoint(int tokens) {
    return std::min(tokens, decoder_batch_prefix_max_tokens) / mxfp8_page_size * mxfp8_page_size;
}

int PrefixCache::restore(Request& request, const std::vector<int>& prompt, std::string_view salt,
                         cudaStream_t stream, MtpDraftRequest* draft, int max_tokens) {
    const int limit = checkpoint(std::min(static_cast<int>(prompt.size()), max_tokens));
    if (!limit) return 0;
    const auto match = index_.find(prompt, salt, 1, limit, draft != nullptr);
    if (match.slot == -1) {
        counters_.draft_misses += match.draft_miss;
        return 0;
    }
    snapshots_.restore(match.slot, request, match.tokens / mxfp8_page_size,
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
                        const MtpDraftRequest* draft) {
    if (tokens < 1 || tokens > decoder_batch_prefix_max_tokens || tokens % mxfp8_page_size != 0 ||
        std::size_t(tokens) > prompt.size() || salt.size() > decoder_prefix_max_salt_bytes ||
        (draft && (!snapshots_.draft_device_bytes() || draft->valid_prefix() != tokens - 1))) {
        throw std::invalid_argument("Invalid prefix cache checkpoint");
    }
    const auto exact = index_.find(prompt, salt, tokens, tokens);
    const int draft_rows = draft ? tokens - 1 : 0;
    if (index_.touch_if_covered(exact, draft_rows)) return;
    const auto mapping =
        GpuPrefixSnapshotStore::read_request_pages(request, tokens / mxfp8_page_size, stream);
    const auto write = index_.begin_write(exact);
    counters_.evictions += write.evicted;
    snapshots_.store(write.slot, request, mapping, stream, draft, draft_rows);
    ++counters_.stores;
    index_.publish(write, prompt, salt, tokens, draft_rows);
}

void PrefixCache::count_prefill(int tokens, int draft_rows) {
    counters_.prefill_tokens += tokens;
    counters_.draft_prefill_rows += draft_rows;
}

void PrefixCache::reset() { index_.reset(); }

DecoderPrefixCacheStats PrefixCache::stats() const {
    auto stats = counters_;
    stats.device_bytes = snapshots_.device_bytes();
    stats.draft_device_bytes = snapshots_.draft_device_bytes();
    const auto occupancy = index_.occupancy();
    stats.entries = occupancy.entries;
    stats.pages = occupancy.tokens / mxfp8_page_size;
    return stats;
}

}
