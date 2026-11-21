#include "sglang/text/prefill_attn_sm100.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

using namespace sglang;

constexpr int rows = 193;
constexpr int visible = 98304;
constexpr int position = visible - rows;
constexpr int page_count = visible / mxfp8_page_size;
constexpr int groups = mxfp8_query_heads / mxfp8_kv_heads;
constexpr int channels = mxfp8_head_dim;
constexpr std::size_t query_elements = rows * mxfp8_query_heads * channels;
constexpr std::size_t payload_elements = page_count * mxfp8_payload_bytes_per_page;
constexpr std::size_t scale_elements = page_count * mxfp8_scale_bytes_per_page;

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void check(cudaError_t status) {
    require(status == cudaSuccess, cudaGetErrorString(status));
}

template <typename T>
struct Buffer {
    explicit Buffer(std::size_t count) {
        check(cudaMalloc(&data, count * sizeof(T)));
    }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    ~Buffer() { cudaFree(data); }

    void upload(const std::vector<T>& values) {
        check(cudaMemcpy(data, values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice));
    }

    T* data = nullptr;
};

constexpr std::array<int, 4> sample_channels{0, 31, 128, 255};

std::uint8_t key_code(int token, int head) {
    const int page = token / mxfp8_page_size;
    return static_cast<std::uint8_t>(0x28 + 23 * page / (page_count - 1) +
                                     (token % 5 == head ? 1 : 0));
}

std::uint8_t value_code(int token, int head, int channel) {
    const auto bits = static_cast<unsigned>(token) * 0x9e3779b1U ^
                      static_cast<unsigned>(head * channels + channel) * 0x85ebca77U;
    const auto sign = (bits >> 17) & 0x80;
    return static_cast<std::uint8_t>(sign | (0x2c + ((bits >> 20) & 0xf)));
}

double decode(std::uint8_t code) {
    const int exponent = (code >> 3) & 15;
    const double fraction = (code & 7) / 8.0;
    const double value = exponent ? std::ldexp(1.0 + fraction, exponent - 7) :
                                    std::ldexp(fraction, -6);
    return code & 0x80 ? -value : value;
}

double gain(int head) {
    return 0.5 + (head % groups) * 0.25;
}

void fill_kv(const Mxfp8KvCache& cache, Buffer<int>& pages) {
    std::vector<std::uint8_t> keys(payload_elements), values(payload_elements);
    std::vector<int> mapping(page_count);
    for (int page = 0; page < page_count; ++page) {
        mapping[page] = page_count - page - 1;
    }
    for (int token = 0; token < visible; ++token) {
        const int physical = mapping[token / mxfp8_page_size] * mxfp8_page_size +
                             token % mxfp8_page_size;
        for (int head = 0; head < mxfp8_kv_heads; ++head) {
            const auto offset = static_cast<std::size_t>(physical * mxfp8_kv_heads + head) *
                                channels;
            keys[offset] = key_code(token, head);
            for (const int channel : sample_channels) {
                values[offset + channel] = value_code(token, head, channel);
            }
        }
    }
    check(cudaMemcpy(cache.key_payload, keys.data(), keys.size(), cudaMemcpyHostToDevice));
    check(cudaMemcpy(cache.value_payload, values.data(), values.size(), cudaMemcpyHostToDevice));
    check(cudaMemset(cache.key_scales, 127, scale_elements));
    check(cudaMemset(cache.value_scales, 127, scale_elements));
    pages.upload(mapping);
}

std::array<double, sample_channels.size()> reference(int row, int head) {
    std::array<double, sample_channels.size()> weighted{};
    const int kv_head = head / groups;
    double mass = 0.0;
    for (int token = 0; token <= position + row; ++token) {
        const double weight = std::exp(gain(head) * decode(key_code(token, kv_head)));
        mass += weight;
        for (std::size_t sample = 0; sample < weighted.size(); ++sample) {
            const auto code = value_code(token, kv_head, sample_channels[sample]);
            weighted[sample] += weight * decode(code);
        }
    }
    for (double& value : weighted) {
        value /= mass;
    }
    return weighted;
}

void check_reference(const std::vector<__nv_bfloat16>& output) {
    double error = 0.0;
    double norm = 0.0;
    double maximum = 0.0;
    for (const int row : {0, 31, 63, 64, 65, 127, 128, rows - 1}) {
        for (int head = 0; head < mxfp8_query_heads; ++head) {
            const auto result = reference(row, head);
            const auto offset = static_cast<std::size_t>(row * mxfp8_query_heads + head) *
                                channels;
            for (std::size_t sample = 0; sample < result.size(); ++sample) {
                const double got = __bfloat162float(output[offset + sample_channels[sample]]);
                require(std::isfinite(got), "SM100 produced a nonfinite control output");
                const double delta = got - result[sample];
                maximum = std::max(maximum, std::abs(delta));
                error += delta * delta;
                norm += result[sample] * result[sample];
            }
        }
    }
    std::cout << "Rising maxima FP64 control: max abs " << maximum <<
                 ", NRMSE " << std::sqrt(error / norm) << '\n';
    require(maximum < 0.00002, "SM100 rising-max control exceeded the absolute error limit");
}

void check_graph(cudaGraph_t graph) {
    std::size_t count = 0;
    check(cudaGraphGetNodes(graph, nullptr, &count));
    require(count == 5, "Expected five SM100 stream-capture nodes");
    std::array<cudaGraphNode_t, 5> nodes{};
    check(cudaGraphGetNodes(graph, nodes.data(), &count));
    std::array<cudaGraphNodeType, 5> types{};
    for (std::size_t index = 0; index < count; ++index) {
        check(cudaGraphNodeGetType(nodes[index], &types[index]));
    }
    std::sort(types.begin(), types.end());
    const std::array<cudaGraphNodeType, 5> expected{
        cudaGraphNodeTypeKernel, cudaGraphNodeTypeKernel, cudaGraphNodeTypeMemset,
        cudaGraphNodeTypeMemAlloc, cudaGraphNodeTypeMemFree};
    require(types == expected, "Unexpected SM100 stream-capture nodes");
}

void check_replay(Mxfp8KvCache cache, Mxfp8PageTable pages, const __nv_bfloat16* queries,
                  __nv_bfloat16* output, cudaStream_t stream) {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    check(decoder::tensor_core_model_prefill_gqa(cache, pages, position, rows,
                                               queries, output, stream));
    check(cudaStreamEndCapture(stream, &graph));
    check_graph(graph);
    check(cudaGraphInstantiate(&executable, graph, 0));
    std::vector<__nv_bfloat16> first(query_elements), repeated(query_elements);
    for (auto* result : {&first, &repeated}) {
        check(cudaMemsetAsync(output, 0xff, query_elements * sizeof(*output), stream));
        check(cudaGraphLaunch(executable, stream));
        check(cudaStreamSynchronize(stream));
        check(cudaMemcpy(result->data(), output, query_elements * sizeof(*output),
                         cudaMemcpyDeviceToHost));
    }
    check(cudaGraphExecDestroy(executable));
    check(cudaGraphDestroy(graph));
    require(std::memcmp(first.data(), repeated.data(), query_elements * sizeof(*output)) == 0,
            "SM100 graph replay changed after poisoning");
    check_reference(first);
}

}

int main() try {
    require(setenv("SGLANG_PREFILL_ATTN", "sm100", 1) == 0,
            "Could not enable SM100 attention");
    Buffer<std::uint8_t> keys(payload_elements), values(payload_elements);
    Buffer<std::uint8_t> key_scales(scale_elements), value_scales(scale_elements);
    Buffer<int> mapping(page_count);
    Buffer<__nv_bfloat16> queries(query_elements + 1), output(query_elements);
    const Mxfp8KvCache cache{
        keys.data, values.data, key_scales.data, value_scales.data, page_count};
    const Mxfp8PageTable pages{mapping.data, page_count};
    fill_kv(cache, mapping);
    std::vector<__nv_bfloat16> host_queries(query_elements);
    for (int row = 0; row < rows; ++row) {
        for (int head = 0; head < mxfp8_query_heads; ++head) {
            host_queries[(row * mxfp8_query_heads + head) * channels] =
                __float2bfloat16(static_cast<float>(16.0 * gain(head)));
        }
    }
    queries.upload(host_queries);
    cudaStream_t stream = nullptr;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    require(launch_prefill_attn_sm100(cache, {mapping.data, page_count - 1}, position, rows,
                                    queries.data, output.data, stream) == cudaErrorInvalidValue,
            "SM100 launch accepted an undersized page table");
    require(!launch_prefill_attn_sm100(cache, pages, position, rows, queries.data + 1,
                                     output.data, stream), "SM100 launch accepted unaligned Q");
    check(cudaDeviceSynchronize());
    check_replay(cache, pages, queries.data, output.data, stream);
    check(cudaStreamDestroy(stream));
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
