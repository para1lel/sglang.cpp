// Build with xmake f --long-decode-check=y, then xmake build gqa-split-fp64-check.
// Run xmake run gqa-split-fp64-check on the selected GPU. No checkpoint is needed.
// For a baseline without split GQA, compile with SGLANG_GQA_ORDERED_ONLY and link sglang_cuda.
#include "sglang/mxfp8_gqa.h"

#ifndef SGLANG_GQA_ORDERED_ONLY
#include "sglang/decoder_cuda.cuh"
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace {

using namespace sglang;
using Bf16 = __nv_bfloat16;
constexpr int channels = mxfp8_query_heads * mxfp8_head_dim;
constexpr int token_stride = mxfp8_kv_heads * mxfp8_head_dim;
constexpr int scale_head_stride = mxfp8_page_size * mxfp8_groups;
constexpr double relative_tolerance = 0.008;

void check(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}

template <typename Element>
class Device {
public:
    explicit Device(std::size_t size) : count(size) {
        check(cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(Element)));
    }
    ~Device() { cudaFree(data); }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;

    void upload(const Element* source) const {
        check(cudaMemcpy(data, source, count * sizeof(Element), cudaMemcpyHostToDevice));
    }

    std::vector<Element> read() const {
        std::vector<Element> host(count);
        check(cudaMemcpy(host.data(), data, count * sizeof(Element), cudaMemcpyDeviceToHost));
        return host;
    }

    Element* data = nullptr;
    const std::size_t count;
};

struct Case {
    const char* name;
    float query_channel_one;
    int key_scale;
    std::uint8_t second_key;
    int value_scale;
    std::array<std::uint8_t, 2> first_value;
    std::array<std::uint8_t, 2> second_value;
    std::array<double, 2> fp64;
};

bool compare(const Case& test, const char* route, const std::vector<Bf16>& actual) {
    for (int head = 0; head < mxfp8_query_heads; ++head) {
        for (int channel = 0; channel < static_cast<int>(test.fp64.size()); ++channel) {
            const double observed = __bfloat162float(actual[head * mxfp8_head_dim + channel]);
            const double expected = test.fp64[channel];
            if (!std::isfinite(observed) ||
                std::abs(observed - expected) > std::abs(expected) * relative_tolerance) {
                std::cerr << test.name << ' ' << route << " head=" << head
                          << " channel=" << channel << " fp64=" << expected
                          << " actual=" << observed << '\n';
                return false;
            }
        }
    }
    std::cout << test.name << ' ' << route << " matched FP64\n";
    return true;
}

bool run(const Case& test) {
    constexpr int split_tokens = 2048;
    constexpr int split_pages = split_tokens / mxfp8_page_size;
    const int page_id = 0;
    const int position = 1;
    Device<int> page(1);
    page.upload(&page_id);
    Device<std::uint8_t> keys(split_pages * mxfp8_payload_bytes_per_page);
    Device<std::uint8_t> values(split_pages * mxfp8_payload_bytes_per_page);
    Device<std::uint8_t> key_scales(split_pages * mxfp8_scale_bytes_per_page);
    Device<std::uint8_t> value_scales(split_pages * mxfp8_scale_bytes_per_page);
    std::vector<std::uint8_t> host_keys(keys.count), host_values(values.count);
    std::vector<std::uint8_t> host_key_scales(key_scales.count, 127);
    std::vector<std::uint8_t> host_value_scales(value_scales.count, 127);
    for (int head = 0; head < mxfp8_kv_heads; ++head) {
        const int channel = head * mxfp8_head_dim;
        const int scale = head * scale_head_stride;
        host_keys[channel + 1] = 0x7e;
        host_keys[token_stride + channel + 1] = test.second_key;
        host_key_scales[scale] = 127 + test.key_scale;
        host_key_scales[scale + 4 * mxfp8_groups] = 127 + test.key_scale;
        host_value_scales[scale] = 127 + test.value_scale;
        host_value_scales[scale + 4 * mxfp8_groups] = 127 + test.value_scale;
        for (int offset = 0; offset < 2; ++offset) {
            host_values[channel + offset] = test.first_value[offset];
            host_values[token_stride + channel + offset] = test.second_value[offset];
        }
    }
    keys.upload(host_keys.data());
    values.upload(host_values.data());
    key_scales.upload(host_key_scales.data());
    value_scales.upload(host_value_scales.data());
    const Mxfp8KvCache cache{keys.data, values.data, key_scales.data, value_scales.data, 1};
    const Mxfp8PageTable table{page.data, 1};

    std::vector<Bf16> host_queries(channels, __float2bfloat16(0.0f));
    for (int head = 0; head < mxfp8_query_heads; ++head) {
        host_queries[head * mxfp8_head_dim] = __float2bfloat16(1.0f);
        host_queries[head * mxfp8_head_dim + 1] = __float2bfloat16(test.query_channel_one);
    }
    Device<Bf16> queries(channels);
    Device<Bf16> output(channels);
    queries.upload(host_queries.data());
    check(mxfp8_gqa_attention(cache, table, 2, position, 1, queries.data, output.data, nullptr));
    const bool ordered = compare(test, "ordered", output.read());
#ifndef SGLANG_GQA_ORDERED_ONLY
    static_assert(split_tokens == decoder::target_split_first_position + 1);
    static_assert(split_tokens % mxfp8_page_size == 0);
    for (int token = 2; token < mxfp8_page_size; ++token) {
        const int source = token % 2;
        std::copy_n(host_keys.begin() + source * token_stride, token_stride,
                    host_keys.begin() + token * token_stride);
        std::copy_n(host_values.begin() + source * token_stride, token_stride,
                    host_values.begin() + token * token_stride);
        for (int head = 0; head < mxfp8_kv_heads; ++head) {
            const int base = head * scale_head_stride;
            const int target = base + (token % 32 * 4 + token / 32) * mxfp8_groups;
            const int reference = base + source * 4 * mxfp8_groups;
            std::copy_n(host_key_scales.begin() + reference, mxfp8_groups,
                        host_key_scales.begin() + target);
            std::copy_n(host_value_scales.begin() + reference, mxfp8_groups,
                        host_value_scales.begin() + target);
        }
    }
    for (auto* plane : {&host_keys, &host_values, &host_key_scales, &host_value_scales}) {
        const auto page_bytes = plane->size() / split_pages;
        for (int index = 1; index < split_pages; ++index) {
            std::copy_n(plane->begin(), page_bytes, plane->begin() + index * page_bytes);
        }
    }
    keys.upload(host_keys.data());
    values.upload(host_values.data());
    key_scales.upload(host_key_scales.data());
    value_scales.upload(host_value_scales.data());
    Device<int> split_by_page(split_pages);
    std::array<int, split_pages> split_mapping{};
    std::iota(split_mapping.begin(), split_mapping.end(), 0);
    split_by_page.upload(split_mapping.data());
    Device<std::byte> scratch(decoder::paged_decode_scratch_bytes);
    const Mxfp8KvCache split_cache{keys.data, values.data, key_scales.data, value_scales.data,
                                  split_pages};
    const decoder::PagedDecodeLane lane{split_cache, {split_by_page.data, split_pages},
        split_tokens - 1, {queries.data, output.data, scratch.data}};
    check(decoder::target_split_gqa(lane, nullptr));
    const bool split = compare(test, "split", output.read());
    return ordered && split;
#else
    return ordered;
#endif
}

}

int main() try {
    const double symmetric = std::tanh(0.21875);
    const double small_weight = std::exp(-28.0);
    const double large_value = std::ldexp(448.0, 20);
    const std::array cases{
        Case{"well-conditioned", 0x1p-16f, 9, 0xfe, 0, {0x38, 0}, {0xb8, 0},
             {symmetric, 0.0}},
        Case{"query dynamic range", 0x1p-40f, 33, 0xfe, 0, {0x38, 0}, {0xb8, 0},
             {symmetric, 0.0}},
        Case{"probability-weighted value scale", 1.0f, 0, 0, 20, {0, 0x7e}, {0x7e, 0},
             {large_value * small_weight / (1.0 + small_weight),
              large_value / (1.0 + small_weight)}},
    };
    bool passed = true;
    for (const auto& test : cases) {
        passed = run(test) && passed;
    }
    return passed ? 0 : 1;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
