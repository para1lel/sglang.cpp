#include "sglang/decoder_prefix_cache.h"

#include <algorithm>
#include <bitset>

namespace sglang::decoder {
namespace {

constexpr int max_pages = decoder_batch_prefix_max_tokens / mxfp8_page_size;
constexpr int max_request_pages = text_attention_max_tokens / mxfp8_page_size;
static_assert(decoder_batch_prefix_max_tokens % mxfp8_page_size == 0);
constexpr auto payload_bytes = full_attention_layers * max_pages * mxfp8_payload_bytes_per_page;
constexpr auto scale_bytes = full_attention_layers * max_pages * mxfp8_scale_bytes_per_page;

using PageMapping = std::array<std::int32_t, max_request_pages>;

PageMapping read_request_pages(const Request& request, int pages, cudaStream_t stream) {
    const auto table = request.pages;
    const int physical_pages = request.caches.front().page_count;
    if (!table.device_pages || pages < 1 || pages > max_pages ||
        table.page_count < pages || table.page_count > max_request_pages ||
        physical_pages < pages || physical_pages > max_request_pages) {
        throw std::invalid_argument("Invalid prefix KV page capacity");
    }
    PageMapping mapping{};
    check(cudaMemcpyAsync(mapping.data(), table.device_pages, table.page_count * sizeof(mapping[0]),
                          cudaMemcpyDeviceToHost, stream), "Read prefix KV page mapping");
    check(cudaStreamSynchronize(stream), "Complete prefix KV page mapping read");

    std::bitset<max_request_pages> used;
    for (int logical = 0; logical < pages; ++logical) {
        const int physical = mapping[logical];
        if (physical < 0 || physical >= physical_pages || used.test(physical)) {
            throw std::invalid_argument("Invalid prefix KV page mapping");
        }
        used.set(physical);
    }
    for (int logical = pages; logical < table.page_count; ++logical) {
        const int physical = mapping[logical];
        if (physical >= 0 && physical < physical_pages && used.test(physical)) {
            throw std::invalid_argument("Prefix KV page aliases a suffix page");
        }
    }
    return mapping;
}

template <typename Copy>
void copy_page_runs(const PageMapping& mapping, int pages, Copy copy) {
    for (int logical = 0; logical < pages;) {
        const int physical = mapping[logical];
        int length = 1;
        while (logical + length < pages && mapping[logical + length] == physical + length) {
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

}

PrefixCache::Entry::Entry(Allocations& memory)
    : kv{memory.allocate<std::uint8_t>(payload_bytes),
         memory.allocate<std::uint8_t>(payload_bytes),
         memory.allocate<std::uint8_t>(scale_bytes),
         memory.allocate<std::uint8_t>(scale_bytes), max_pages},
      gdn{memory.allocate<Bf16>(gdn_layers * gdn_history_elements),
          memory.allocate<float>(gdn_layers * gdn_state_elements)},
      hidden(memory.allocate<Bf16>(hidden_size)),
      residual(memory.allocate<Bf16>(hidden_size)) {}

PrefixCache::PrefixCache(int device, int capacity) : memory_(device) {
    if (capacity < 1 || capacity > decoder_batch_prefix_max_entries) {
        throw std::invalid_argument("Invalid prefix cache capacity");
    }
    entries_.reserve(capacity);
    for (int index = 0; index < capacity; ++index) {
        entries_.push_back(std::make_unique<Entry>(memory_));
    }
    counters_.device_bytes = memory_.bytes();
}

int PrefixCache::checkpoint(int tokens) {
    return std::min(tokens, decoder_batch_prefix_max_tokens) / mxfp8_page_size * mxfp8_page_size;
}

int PrefixCache::find(const std::vector<int>& prompt, std::string_view salt, int limit) const {
    int best = -1;
    int longest = 0;
    for (int index = 0; index < static_cast<int>(entries_.size()); ++index) {
        const auto& entry = *entries_[index];
        if (entry.tokens > longest && entry.tokens <= limit &&
            salt == std::string_view(entry.salt.data(), entry.salt_size) &&
            std::equal(entry.ids.begin(), entry.ids.begin() + entry.tokens, prompt.begin())) {
            longest = entry.tokens;
            best = index;
        }
    }
    return best;
}

void PrefixCache::touch(int index) {
    std::rotate(entries_.begin(), entries_.begin() + index, entries_.begin() + index + 1);
}

int PrefixCache::restore(Request& request, const std::vector<int>& prompt, std::string_view salt,
                         cudaStream_t stream) {
    const int index = find(prompt, salt, checkpoint(static_cast<int>(prompt.size())));
    if (index == -1) {
        return 0;
    }
    const auto& entry = *entries_[index];
    const int pages = entry.tokens / mxfp8_page_size;
    const auto mapping = read_request_pages(request, pages, stream);
    copy_page_runs(mapping, pages, [&](int logical, int physical, int length) {
        copy_kv(request.caches.front(), physical, entry.kv, logical, length, stream);
    });
    copy_gdn(request.states.front(), entry.gdn, stream);
    if (entry.tokens == static_cast<int>(prompt.size())) {
        copy_state(request.work.hidden, entry.hidden, hidden_size, stream);
        copy_state(request.work.residual, entry.residual, hidden_size, stream);
    }
    touch(index);
    ++counters_.hits;
    counters_.cached_tokens += entry.tokens;
    return entry.tokens;
}

void PrefixCache::store(const Request& request, const std::vector<int>& prompt,
                        std::string_view salt, int tokens, cudaStream_t stream) {
    if (tokens < 1 || tokens > decoder_batch_prefix_max_tokens || tokens % mxfp8_page_size != 0 ||
        std::size_t(tokens) > prompt.size() || salt.size() > decoder_prefix_max_salt_bytes) {
        throw std::invalid_argument("Invalid prefix cache checkpoint");
    }
    const int index = find(prompt, salt, tokens);
    if (index != -1 && entries_[index]->tokens == tokens) {
        touch(index);
        return;
    }
    const int pages = tokens / mxfp8_page_size;
    const auto mapping = read_request_pages(request, pages, stream);
    auto& entry = *entries_.back();
    counters_.evictions += entry.tokens != 0;
    entry.tokens = 0;
    copy_page_runs(mapping, pages, [&](int logical, int physical, int length) {
        copy_kv(entry.kv, logical, request.caches.front(), physical, length, stream);
    });
    copy_gdn(entry.gdn, request.states.front(), stream);
    copy_state(entry.hidden, request.work.hidden, hidden_size, stream);
    copy_state(entry.residual, request.work.residual, hidden_size, stream);
    check(cudaStreamSynchronize(stream), "Complete decoder prefix checkpoint");
    ++counters_.stores;
    std::copy_n(prompt.begin(), tokens, entry.ids.begin());
    std::copy(salt.begin(), salt.end(), entry.salt.begin());
    entry.salt_size = salt.size();
    entry.tokens = tokens;
    touch(static_cast<int>(entries_.size()) - 1);
}

void PrefixCache::count_prefill(int tokens) {
    counters_.prefill_tokens += tokens;
}

void PrefixCache::reset() {
    for (auto& entry : entries_) {
        entry->tokens = 0;
    }
}

DecoderPrefixCacheStats PrefixCache::stats() const {
    auto stats = counters_;
    for (const auto& entry : entries_) {
        stats.entries += entry->tokens != 0;
        stats.pages += entry->tokens / mxfp8_page_size;
    }
    return stats;
}

}
