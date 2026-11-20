#include "sglang/mxfp8_gqa.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace gqa_test {

double decode_value(std::uint8_t code, std::uint8_t scale);

namespace {

using namespace sglang;
using Byte = std::uint8_t;
using BFloat = __nv_bfloat16;

constexpr int channels = mxfp8_head_dim;
constexpr int row_width = mxfp8_query_heads * channels;
constexpr int queries_per_kv = mxfp8_query_heads / mxfp8_kv_heads;
constexpr int rows = 1024;
constexpr int guard = 32;
constexpr int first_scale = 116;
constexpr int scale_count = 12;
constexpr unsigned head_blocks = mxfp8_query_heads / 2;
constexpr std::size_t output_elements = static_cast<std::size_t>(rows) * row_width;

void check_cuda(cudaError_t status, const char* action) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(action) + ": " + cudaGetErrorString(status));
    }
}

template <class Element>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count) : count_(count) {
        check_cuda(cudaMalloc(&data_, count * sizeof(Element)), "Allocate Flash check buffer");
    }

    ~DeviceBuffer() { cudaFree(data_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    Element* get() const { return data_; }

    void upload(const std::vector<Element>& source, cudaStream_t stream, int offset = 0) const {
        check_cuda(cudaMemcpyAsync(data_ + offset, source.data(),
                                   source.size() * sizeof(Element), cudaMemcpyHostToDevice, stream),
                   "Upload Flash check input");
    }

    std::vector<Element> download(cudaStream_t stream) const {
        std::vector<Element> result(count_);
        check_cuda(cudaMemcpyAsync(result.data(), data_, count_ * sizeof(Element),
                                   cudaMemcpyDeviceToHost, stream), "Download Flash check output");
        check_cuda(cudaStreamSynchronize(stream), "Complete Flash check");
        return result;
    }

private:
    std::size_t count_;
    Element* data_ = nullptr;
};

struct Input {
    const int physical_pages;
    std::vector<std::int32_t> pages;
    std::vector<Byte> keys, values, key_scales, value_scales;
    std::vector<BFloat> queries;

    explicit Input(int tokens)
        : physical_pages((tokens + mxfp8_page_size - 1) / mxfp8_page_size + 3),
          keys(physical_pages * mxfp8_payload_bytes_per_page, 0x7f), values(keys),
          key_scales(physical_pages * mxfp8_scale_bytes_per_page, 255), value_scales(key_scales),
          queries(output_elements) {
        pages.resize(physical_pages);
        std::iota(pages.begin(), pages.end(), 0);
        std::mt19937 random(1729);
        std::shuffle(pages.begin(), pages.end(), random);
        pages.resize(physical_pages - 3);
        for (int token = 0; token < tokens; ++token) {
            for (int head = 0; head < mxfp8_kv_heads; ++head) {
                const auto payload = payload_offset(token, head);
                const auto scale = scale_offset(token, head);
                for (int channel = 0; channel < channels; ++channel) {
                    const auto key = random() % 254;
                    const auto value = random() % 254;
                    keys[payload + channel] = key + (key >= 127);
                    values[payload + channel] = value + (value >= 127);
                }
                for (int group = 0; group < mxfp8_groups; ++group) {
                    key_scales[scale + group] = first_scale + random() % 6;
                    value_scales[scale + group] = first_scale + random() % 8;
                }
            }
        }
        for (auto& query : queries) {
            query = __float2bfloat16_rn((static_cast<int>(random() % 513) - 256) / 128.0f);
        }
    }

    std::size_t payload_offset(int token, int head) const {
        return pages[token / mxfp8_page_size] * mxfp8_payload_bytes_per_page
            + (token % mxfp8_page_size * mxfp8_kv_heads + head) * channels;
    }

    std::size_t scale_offset(int token, int head) const {
        const auto page_head = static_cast<std::size_t>(pages[token / mxfp8_page_size])
            * mxfp8_kv_heads + head;
        const int in_page = token % mxfp8_page_size;
        constexpr int token_groups = mxfp8_page_size / mxfp8_group_size;
        return ((page_head * mxfp8_group_size + in_page % mxfp8_group_size) * token_groups
            + in_page / mxfp8_group_size) * mxfp8_groups;
    }

    void add_causal_marker(int position, int row) {
        const auto marker = payload_offset(position + row, 0);
        const auto scale = scale_offset(position + row, 0);
        std::fill_n(keys.begin() + marker, mxfp8_group_size, 0);
        std::fill_n(values.begin() + marker, mxfp8_group_size, 0);
        keys[marker] = 0x7e;
        values[marker] = 0x48;
        key_scales[scale] = 127;
        value_scales[scale] = 127;
        for (int query_row : {row - 1, row}) {
            const auto query = queries.begin() + query_row * row_width;
            std::fill_n(query, channels, __float2bfloat16_rn(0));
            *query = __float2bfloat16_rn(2);
        }
    }

    void add_score_overflow(int position) {
        for (int token = 0; token <= position + 1; ++token) {
            std::fill_n(keys.begin() + payload_offset(token, 0), channels, 0);
        }
        const auto key = payload_offset(position + 1, 0);
        const auto scale = scale_offset(position + 1, 0);
        std::fill_n(keys.begin() + key, channels, 0x40);
        std::fill_n(key_scales.begin() + scale, mxfp8_groups, 127);
        std::fill_n(queries.begin() + row_width, channels, __float2bfloat16_rn(0x1p126f));
    }
};

using DecodeTable = std::array<std::array<double, 256>, scale_count>;

DecodeTable make_decode_table() {
    DecodeTable table{};
    for (int scale = 0; scale < scale_count; ++scale) {
        for (int code = 0; code < 256; ++code) {
            table[scale][code] = decode_value(code, scale + first_scale);
        }
    }
    return table;
}

struct Sample {
    std::size_t offset;
    std::array<double, channels> reference;
};

std::vector<Sample> attention_samples(const Input& input, int position) {
    const auto table = make_decode_table();
    std::vector<Sample> samples;
    for (int row : {0, 1, 62, 63, 64, 66, 67, rows - 2, rows - 1}) {
        for (int head : {0, 1, queries_per_kv, 2 * queries_per_kv, mxfp8_query_heads - 1}) {
            const auto offset = static_cast<std::size_t>(row) * row_width + head * channels;
            auto& sample = samples.emplace_back(Sample{offset, {}});
            const int kv_head = head / queries_per_kv;
            const int visible = position + row + 1;
            std::array<double, channels> query;
            for (int channel = 0; channel < channels; ++channel) {
                query[channel] = __bfloat162float(input.queries[offset + channel]);
            }
            std::vector<double> scores(visible);
            for (int token = 0; token < visible; ++token) {
                const auto payload = input.payload_offset(token, kv_head);
                const auto scale = input.scale_offset(token, kv_head);
                double dot = 0;
                for (int group = 0; group < mxfp8_groups; ++group) {
                    const auto& decoded = table[input.key_scales[scale + group] - first_scale];
                    for (int channel = group * mxfp8_group_size;
                         channel < (group + 1) * mxfp8_group_size; ++channel) {
                        dot += query[channel] * decoded[input.keys[payload + channel]];
                    }
                }
                scores[token] = dot / 16;
            }
            const double maximum = *std::max_element(scores.begin(), scores.end());
            double mass = 0;
            for (int token = 0; token < visible; ++token) {
                const double probability = std::exp(scores[token] - maximum);
                const auto payload = input.payload_offset(token, kv_head);
                const auto scale = input.scale_offset(token, kv_head);
                mass += probability;
                for (int group = 0; group < mxfp8_groups; ++group) {
                    const auto& decoded = table[input.value_scales[scale + group] - first_scale];
                    for (int channel = group * mxfp8_group_size;
                         channel < (group + 1) * mxfp8_group_size; ++channel) {
                        sample.reference[channel] +=
                            probability * decoded[input.values[payload + channel]];
                    }
                }
            }
            for (auto& value : sample.reference) {
                value /= mass;
            }
        }
    }
    return samples;
}

struct Dispatch {
    const char* kernel;
    unsigned grid_width;
    unsigned grid_height = head_blocks;
};

constexpr Dispatch flash{"flash_prefill_attention", rows / 64};
constexpr Dispatch exact{"attention_kernel", rows, mxfp8_query_heads};
constexpr Dispatch multirow{"multirow_prefill_attention", rows / 4};
constexpr Dispatch protected_rows{"protected_prefill_attention", rows};

struct Graph {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;

    ~Graph() {
        if (executable) {
            cudaGraphExecDestroy(executable);
        }
        if (graph) {
            cudaGraphDestroy(graph);
        }
    }
};

void check_dispatch_errors(cudaStream_t stream) {
    struct Case {
        const char* label;
        int position;
        int count;
        cudaError_t expected;
    };
    constexpr std::array cases{
        Case{"empty input", 0, 0, cudaSuccess},
        Case{"negative position", -1, rows, cudaErrorInvalidValue},
        Case{"negative row count", 0, -1, cudaErrorInvalidValue},
        Case{"visible length overflow", std::numeric_limits<int>::max(), rows,
             cudaErrorInvalidValue},
    };
    std::array<cudaError_t, cases.size()> statuses{};
    Graph graph;
    check_cuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
               "Begin invalid prefill capture");
    for (std::size_t index = 0; index < cases.size(); ++index) {
        const auto& request = cases[index];
        statuses[index] = decoder::tensor_core_multirow_prefill_gqa(
            {}, {}, request.position, request.count, nullptr, nullptr, stream);
    }
    check_cuda(cudaStreamEndCapture(stream, &graph.graph), "End invalid prefill capture");
    std::size_t count = 0;
    check_cuda(cudaGraphGetNodes(graph.graph, nullptr, &count), "Count invalid prefill launches");
    if (count != 0) {
        throw std::runtime_error("Empty or invalid prefill dispatched a kernel");
    }
    for (std::size_t index = 0; index < cases.size(); ++index) {
        const auto& request = cases[index];
        if (statuses[index] != request.expected) {
            throw std::runtime_error(std::string(request.label) + ": expected " +
                                     cudaGetErrorName(request.expected) + ", got " +
                                     cudaGetErrorName(statuses[index]));
        }
        std::printf("MXFP8 prefill %s status=%s launches=0\n", request.label,
                    cudaGetErrorName(statuses[index]));
    }
}

void check_dispatch(const char* label, cudaGraph_t graph, Dispatch expected) {
    std::size_t count = 0;
    check_cuda(cudaGraphGetNodes(graph, nullptr, &count), "Count Flash check launch nodes");
    if (count != 1) {
        throw std::runtime_error(std::string(label) + ": expected one prefill CUDA graph node");
    }
    cudaGraphNode_t node = nullptr;
    check_cuda(cudaGraphGetNodes(graph, &node, &count), "Get Flash check launch node");
    cudaKernelNodeParams params{};
    check_cuda(cudaGraphKernelNodeGetParams(node, &params), "Get Flash check kernel parameters");
    const char* name = nullptr;
    check_cuda(cudaFuncGetName(&name, params.func), "Name Flash check kernel");
    std::printf("MXFP8 prefill %s launch %s grid=%ux%ux%u block=%ux%ux%u shared_bytes=%u\n",
                label, name, params.gridDim.x, params.gridDim.y, params.gridDim.z,
                params.blockDim.x, params.blockDim.y, params.blockDim.z, params.sharedMemBytes);
    if (std::string_view(name).find(expected.kernel) == std::string_view::npos ||
        params.gridDim.x != expected.grid_width || params.gridDim.y != expected.grid_height ||
        params.gridDim.z != 1) {
        throw std::runtime_error(std::string(label) + ": expected " + expected.kernel +
                                 " grid=" + std::to_string(expected.grid_width) + "x" +
                                 std::to_string(expected.grid_height) + "x1");
    }
}

void check_output(const char* label, const std::vector<BFloat>& output,
                  const std::vector<Sample>& samples) {
    for (int index = 0; index < guard; ++index) {
        if (__bfloat16_as_ushort(output[index]) != 0xffff ||
            __bfloat16_as_ushort(output[guard + output_elements + index]) != 0xffff) {
            throw std::runtime_error(std::string(label) + ": output guard overwritten");
        }
    }
    for (std::size_t index = 0; index < output_elements; ++index) {
        if (!std::isfinite(__bfloat162float(output[guard + index]))) {
            throw std::runtime_error(std::string(label) + ": nonfinite output at " +
                                     std::to_string(index));
        }
    }
    std::vector<double> absolute_errors;
    double squared_error = 0;
    double squared_reference = 0;
    std::size_t worst_offset = 0;
    for (const auto& sample : samples) {
        for (int channel = 0; channel < channels; ++channel) {
            const auto offset = sample.offset + channel;
            const double actual = __bfloat162float(output[guard + offset]);
            const double reference = sample.reference[channel];
            const double error = std::abs(actual - reference);
            if (absolute_errors.empty() || error > absolute_errors[worst_offset]) {
                worst_offset = absolute_errors.size();
            }
            absolute_errors.push_back(error);
            squared_error += error * error;
            squared_reference += reference * reference;
        }
    }
    const auto offset = samples[worst_offset / channels].offset + worst_offset % channels;
    std::sort(absolute_errors.begin(), absolute_errors.end());
    const double max_abs = absolute_errors.back();
    const double p99_abs = absolute_errors[(absolute_errors.size() * 99 - 1) / 100];
    const double nrmse = std::sqrt(squared_error / squared_reference);
    std::printf("MXFP8 prefill %s FP64 samples=%zu p99_abs=%.8g max_abs=%.8g nrmse=%.8g "
                "worst_row=%zu head=%zu channel=%zu\n", label, absolute_errors.size(), p99_abs,
                max_abs, nrmse, offset / row_width, offset % row_width / channels,
                offset % channels);
    if (!(max_abs < 0.08 && nrmse < 0.005)) {
        throw std::runtime_error(std::string(label) + ": FP64 error exceeds regression tolerance");
    }
}

struct DeviceInput {
    DeviceBuffer<Byte> keys, values, key_scales, value_scales;
    DeviceBuffer<std::int32_t> pages;
    DeviceBuffer<BFloat> queries{output_elements};
    DeviceBuffer<BFloat> output{output_elements + 2 * guard};

    explicit DeviceInput(const Input& input)
        : keys(input.keys.size() + 1), values(input.values.size() + 1),
          key_scales(input.key_scales.size()), value_scales(input.value_scales.size()),
          pages(input.pages.size()) {}

    void run(const char* label, const Input& input, int position, Dispatch expected,
             const std::vector<Sample>& samples, cudaStream_t stream,
             int key_offset = 0, int value_offset = 0, bool use_exact = false) {
        keys.upload(input.keys, stream, key_offset);
        values.upload(input.values, stream, value_offset);
        key_scales.upload(input.key_scales, stream);
        value_scales.upload(input.value_scales, stream);
        pages.upload(input.pages, stream);
        queries.upload(input.queries, stream);
        constexpr auto output_bytes = (output_elements + 2 * guard) * sizeof(BFloat);
        check_cuda(cudaMemsetAsync(output.get(), 0xff, output_bytes, stream),
                   "Poison Flash check output");
        const Mxfp8KvCache cache{keys.get() + key_offset, values.get() + value_offset,
                                 key_scales.get(), value_scales.get(), input.physical_pages};
        const Mxfp8PageTable page_table{pages.get(), static_cast<int>(input.pages.size())};
        Graph graph;
        check_cuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
                   "Begin Flash check capture");
        const auto dispatch = use_exact
            ? mxfp8_gqa_attention(cache, page_table, position + rows, position, rows,
                                  queries.get(), output.get() + guard, stream)
            : decoder::tensor_core_multirow_prefill_gqa(
                  cache, page_table, position, rows, queries.get(), output.get() + guard, stream);
        const auto capture = cudaStreamEndCapture(stream, &graph.graph);
        check_cuda(dispatch, "Dispatch captured prefill");
        check_cuda(capture, "End Flash check capture");
        check_dispatch(label, graph.graph, expected);
        check_cuda(cudaGraphInstantiate(&graph.executable, graph.graph, 0),
                   "Instantiate Flash check graph");
        check_cuda(cudaGraphLaunch(graph.executable, stream), "Replay Flash check graph");
        check_output(label, output.download(stream), samples);
    }
};

}

void check_flash_prefill(cudaStream_t stream) {
    check_dispatch_errors(stream);
    constexpr int position = 61;
    {
        Input input(position + rows);
        input.add_causal_marker(position, 1);
        input.add_causal_marker(position, 63);
        input.add_causal_marker(position, 64);
        input.add_causal_marker(position, 67);
        input.add_causal_marker(position, rows - 1);
        const auto samples = attention_samples(input, position);
        DeviceInput device(input);
        device.run("aligned causal tail", input, position, flash, samples, stream);
        device.run("aligned exact", input, position, exact, samples, stream, 0, 0, true);
        device.run("unaligned key", input, position, multirow, samples, stream, 1, 0);
        device.run("unaligned value", input, position, multirow, samples, stream, 0, 1);
        input.add_score_overflow(position);
        device.run("Flash score overflow recovery", input, position, flash,
                   attention_samples(input, position), stream);
    }
    {
        constexpr int max_visible = 32768;
        Input input(max_visible + 1);
        DeviceInput device(input);
        device.run("32768 visible", input, max_visible - rows, flash,
                   attention_samples(input, max_visible - rows), stream);
        device.run("32769 visible", input, max_visible + 1 - rows, protected_rows,
                   attention_samples(input, max_visible + 1 - rows), stream);
    }
}

}
