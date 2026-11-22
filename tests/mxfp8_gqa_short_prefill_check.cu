#include "sglang/decoder_cuda.cuh"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace {

using namespace sglang;
using Bf16 = __nv_bfloat16;

constexpr int page_count = 65;
constexpr int capacity = page_count * mxfp8_page_size;
constexpr int position = 8192;
constexpr int channels = mxfp8_query_heads * mxfp8_head_dim;
constexpr auto payload_bytes = page_count * mxfp8_payload_bytes_per_page;
constexpr auto scale_bytes = page_count * mxfp8_scale_bytes_per_page;

void check(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}

template <typename Element>
struct DeviceBuffer {
    explicit DeviceBuffer(std::size_t count) { check(cudaMalloc(&data, count * sizeof(Element))); }
    ~DeviceBuffer() { cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    Element* data = nullptr;
};

using Output = std::array<std::uint16_t, channels + 2>;

Output replay(Mxfp8KvCache cache, Mxfp8PageTable pages, int last, const Bf16* query,
              Bf16* output, decoder::ExactGqaWorkspace workspace, bool prefill,
              cudaStream_t stream) {
    Output host;
    check(cudaMemsetAsync(output, 0xa5, sizeof(host), stream));
    check(decoder::exact_decode_gqa_attention(cache, pages, last, query, output + 1,
                                               workspace, stream, 0, prefill));
    check(cudaMemcpyAsync(host.data(), output, sizeof(host), cudaMemcpyDeviceToHost, stream));
    check(cudaStreamSynchronize(stream));
    if (host.front() != 0xa5a5 || host.back() != 0xa5a5 ||
        std::any_of(host.begin() + 1, host.end() - 1, [](auto value) { return value == 0xa5a5; })) {
        throw std::runtime_error("Short prefill output or canaries were incomplete");
    }
    return host;
}

void check_draft_replay(Mxfp8KvCache cache, Mxfp8PageTable pages, const Bf16* query,
                        Bf16* output, float* scores, cudaStream_t stream) {
    constexpr int short_capacity = 5173;
    constexpr int chunk_capacity = 4368;
    constexpr std::array<std::uint8_t, 4> codes{0x34, 0xb0, 0x39, 0x42};
    std::vector<std::uint8_t> varied_values(payload_bytes);
    for (std::size_t index = 0; index < varied_values.size(); ++index) {
        varied_values[index] = codes[index % codes.size()];
    }
    check(cudaMemcpyAsync(cache.value_payload, varied_values.data(), payload_bytes,
                          cudaMemcpyHostToDevice, stream));
    const auto download = [&] {
        Output host;
        check(cudaMemcpyAsync(host.data(), output, sizeof(host), cudaMemcpyDeviceToHost, stream));
        check(cudaStreamSynchronize(stream));
        if (host.front() != 0xa5a5 || host.back() != 0xa5a5 ||
            std::any_of(host.begin() + 1, host.end() - 1,
                        [](auto value) { return value == 0xa5a5; })) {
            throw std::runtime_error("Draft replay output or canaries were incomplete");
        }
        return host;
    };
    DeviceBuffer<std::uint8_t> offset_values(payload_bytes + 3);
    for (const int visible : {5174, 5173, 257, 256, 8, 7, 8192}) {
        check(cudaMemsetAsync(output, 0xa5, sizeof(Output), stream));
        check(mxfp8_gqa_attention(cache, pages, visible, visible - 1, 1, query, output + 1,
                                  stream));
        const auto expected = download();
        for (const int offset : {0, 1, 2, 3}) {
            auto variant = cache;
            if (offset != 0) {
                variant.value_payload = offset_values.data + offset;
                check(cudaMemcpyAsync(variant.value_payload, cache.value_payload, payload_bytes,
                                      cudaMemcpyDeviceToDevice, stream));
            }
            for (const int capacity : {short_capacity, chunk_capacity, chunk_capacity - 1}) {
                const decoder::ExactGqaWorkspace workspace{scores, capacity};
                check(cudaMemsetAsync(output, 0xa5, sizeof(Output), stream));
                std::printf("Draft V alignment visible=%d capacity=%d offset=%d\n",
                            visible, capacity, offset);
                std::fflush(stdout);
                check(decoder::mtp_gqa_attention(variant, pages, visible - 1, query, output + 1,
                                                 workspace, stream));
                if (download() != expected) {
                    throw std::runtime_error("Draft replay differs bitwise from direct attention");
                }
            }
            if (visible == 257) {
                check(cudaMemsetAsync(output, 0xa5, sizeof(Output), stream));
                check(decoder::vector_decode_gqa_attention(
                    variant, pages, visible - 1, query, output + 1, {scores, visible}, stream));
                if (download() != expected) {
                    throw std::runtime_error("Vector decode differs bitwise from direct attention");
                }
            }
        }
    }
}

}

namespace gqa_test {

void check_short_prefill_replay(cudaStream_t stream) {
    DeviceBuffer<std::uint8_t> keys(payload_bytes), values(payload_bytes);
    DeviceBuffer<std::uint8_t> offset(payload_bytes + 1);
    DeviceBuffer<std::uint8_t> key_scales(scale_bytes), value_scales(scale_bytes);
    DeviceBuffer<std::uint8_t> scale_offset(scale_bytes + 1);
    DeviceBuffer<std::int32_t> mapping(page_count);
    DeviceBuffer<Bf16> query(channels), output(channels + 2);
    DeviceBuffer<float> scores(capacity * mxfp8_query_heads);
    std::array<std::int32_t, page_count> reversed;
    for (int page = 0; page < page_count; ++page) {
        reversed[page] = page_count - 1 - page;
    }
    check(cudaMemcpyAsync(mapping.data, reversed.data(), sizeof(reversed),
                          cudaMemcpyHostToDevice, stream));
    check(cudaMemsetAsync(keys.data, 0x30, payload_bytes, stream));
    check(cudaMemsetAsync(values.data, 0x34, payload_bytes, stream));
    check(cudaMemsetAsync(key_scales.data, 127, scale_bytes, stream));
    check(cudaMemsetAsync(value_scales.data, 127, scale_bytes, stream));
    check(cudaMemsetAsync(query.data, 0x3f, channels * sizeof(Bf16), stream));
    constexpr auto token_bytes = mxfp8_payload_bytes_per_page / mxfp8_page_size;
    check(cudaMemsetAsync(keys.data, 0x40, token_bytes, stream));
    check(cudaMemsetAsync(values.data, 0x42, token_bytes, stream));
    check(cudaMemsetAsync(values.data + token_bytes, 0x7e,
                          mxfp8_payload_bytes_per_page - token_bytes, stream));
    check(cudaMemsetAsync(value_scales.data, 128, mxfp8_groups, stream));
    check(cudaMemcpyAsync(offset.data + 1, values.data, payload_bytes,
                          cudaMemcpyDeviceToDevice, stream));
    check(cudaMemcpyAsync(scale_offset.data + 1, value_scales.data, scale_bytes,
                          cudaMemcpyDeviceToDevice, stream));
    const Mxfp8PageTable pages{mapping.data, page_count};
    const Mxfp8KvCache cache{keys.data, values.data, key_scales.data, value_scales.data,
                             page_count};
    auto unaligned_values = cache;
    unaligned_values.value_payload = offset.data + 1;
    auto unaligned_scales = cache;
    unaligned_scales.value_scales = scale_offset.data + 1;
    const decoder::ExactGqaWorkspace workspace{scores.data, capacity};
    for (const int last : {position - 2, position - 1, position}) {
        const auto expected = replay(cache, pages, last, query.data, output.data, workspace, false,
                                     stream);
        for (auto variant : {cache, unaligned_values, unaligned_scales}) {
            if (replay(variant, pages, last, query.data, output.data, workspace, true, stream) !=
                expected) {
                throw std::runtime_error("Short prefill differs bitwise from exact replay");
            }
        }
    }
    check_draft_replay(unaligned_scales, pages, query.data, output.data, scores.data, stream);
}

}
