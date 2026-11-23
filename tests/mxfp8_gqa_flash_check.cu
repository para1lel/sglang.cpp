#include "sglang/mxfp8_gqa.h"
#include "sglang/text/prefill_attn_sm100.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

    explicit Input(int tokens, int count = rows)
        : physical_pages((tokens + mxfp8_page_size - 1) / mxfp8_page_size + 3),
          keys(physical_pages * mxfp8_payload_bytes_per_page, 0x7f), values(keys),
          key_scales(physical_pages * mxfp8_scale_bytes_per_page, 255), value_scales(key_scales),
          queries(static_cast<std::size_t>(count) * row_width) {
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

    void add_score_overflow(int position, int row = 1) {
        for (int token = 0; token <= position + row; ++token) {
            std::fill_n(keys.begin() + payload_offset(token, 0), channels, 0);
        }
        const auto key = payload_offset(position + row, 0);
        const auto scale = scale_offset(position + row, 0);
        std::fill_n(keys.begin() + key, channels, 0x40);
        std::fill_n(key_scales.begin() + scale, mxfp8_groups, 127);
        std::fill_n(queries.begin() + row * row_width, channels, __float2bfloat16_rn(0x1p126f));
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

std::vector<Sample> attention_samples(const Input& input, int position, int count = rows) {
    const auto table = make_decode_table();
    std::vector<Sample> samples;
    for (int row : {0, 1, 62, 63, 64, 65, 66, 67, count - 2, count - 1}) {
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
    unsigned grid_depth = 1;
    unsigned block_width = 0;
    int cluster_width = 0;
};

constexpr Dispatch flash{"flash_prefill_attentionILi88ELb1ELb1EE", 12, head_blocks, 1, 352};
constexpr Dispatch tmem_paired{"tmem_pair_attention", rows / 64, head_blocks, 1, 416, 2};
constexpr Dispatch tmem_paired_tail{"tmem_pair_attentionILb1EE", rows / 64, head_blocks, 1, 416, 2};
constexpr Dispatch exact{"attention_kernel", rows, mxfp8_query_heads};
constexpr Dispatch multirow{"multirow_prefill_attention", rows / 4};
constexpr Dispatch protected_rows{"protected_prefill_attention", rows};

constexpr Dispatch flash_dispatch(int count) {
    return {count % 64 == 0 ? "flash_prefill_attentionILi64ELb0E"
                           : "flash_prefill_attentionILi64ELb1E",
            static_cast<unsigned>((count + 63) / 64)};
}

constexpr Dispatch ordered_dispatch(int count) {
    return {"multirow_prefill_attention", static_cast<unsigned>(count / 4 + count % 4)};
}

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
        Case{"first overflowing visible length", std::numeric_limits<int>::max() - rows + 1,
             rows, cudaErrorInvalidValue},
        Case{"paired missing cache and page table", 8192, rows, cudaErrorInvalidValue},
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

void check_kernel(const char* label, cudaGraphNode_t node, Dispatch expected) {
    cudaKernelNodeParams params{};
    check_cuda(cudaGraphKernelNodeGetParams(node, &params), "Get Flash check kernel parameters");
    const char* name = nullptr;
    check_cuda(cudaFuncGetName(&name, params.func), "Name Flash check kernel");
    std::printf("MXFP8 prefill %s launch %s grid=%ux%ux%u block=%ux%ux%u shared_bytes=%u\n",
                label, name, params.gridDim.x, params.gridDim.y, params.gridDim.z,
                params.blockDim.x, params.blockDim.y, params.blockDim.z, params.sharedMemBytes);
    if (std::string_view(name).find(expected.kernel) == std::string_view::npos ||
        params.gridDim.x != expected.grid_width || params.gridDim.y != expected.grid_height ||
        params.gridDim.z != expected.grid_depth ||
        (expected.block_width && (params.blockDim.x != expected.block_width ||
                                  params.blockDim.y != 1 || params.blockDim.z != 1))) {
        throw std::runtime_error(std::string(label) + ": expected " + expected.kernel +
                                 " grid=" + std::to_string(expected.grid_width) + "x" +
                                 std::to_string(expected.grid_height) + "x" +
                                 std::to_string(expected.grid_depth));
    }
    if (expected.cluster_width) {
        cudaFuncAttributes attributes{};
        check_cuda(cudaFuncGetAttributes(&attributes, params.func), "Read prefill cluster shape");
        if (attributes.requiredClusterWidth != expected.cluster_width ||
            attributes.requiredClusterHeight != 1 || attributes.requiredClusterDepth != 1) {
            throw std::runtime_error(std::string(label) + ": unexpected prefill cluster shape");
        }
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
    check_kernel(label, node, expected);
}

void check_guards(const char* label, const std::vector<BFloat>& output, int count) {
    const auto elements = static_cast<std::size_t>(count) * row_width;
    for (int index = 0; index < guard; ++index) {
        if (__bfloat16_as_ushort(output[index]) != 0xffff) {
            throw std::runtime_error(std::string(label) + ": prefix output guard overwritten");
        }
    }
    for (std::size_t index = guard + elements; index < output.size(); ++index) {
        if (__bfloat16_as_ushort(output[index]) != 0xffff) {
            throw std::runtime_error(std::string(label) + ": inactive output overwritten");
        }
    }
}

void check_output(const char* label, const std::vector<BFloat>& output,
                  const std::vector<Sample>& samples, int count) {
    check_guards(label, output, count);
    const auto elements = static_cast<std::size_t>(count) * row_width;
    for (std::size_t index = 0; index < elements; ++index) {
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

struct RunOptions {
    int count = rows;
    int key_offset = 0;
    int value_offset = 0;
    int key_scale_offset = 0;
    int value_scale_offset = 0;
    bool use_exact = false;
    bool repeat_graph = false;
};

struct DeviceInput {
    DeviceBuffer<Byte> keys, values, key_scales, value_scales;
    DeviceBuffer<std::int32_t> pages;
    DeviceBuffer<BFloat> queries;
    DeviceBuffer<BFloat> output{output_elements + 2 * guard};

    explicit DeviceInput(const Input& input)
        : keys(input.keys.size() + 1), values(input.values.size() + 1),
          key_scales(input.key_scales.size() + 1), value_scales(input.value_scales.size() + 8),
          pages(input.pages.size()), queries(input.queries.size()) {}

    void check_missing_tail_page(int position, cudaStream_t stream) const {
        const int floor_pages = (position + rows) / mxfp8_page_size;
        for (bool short_table : {false, true}) {
            const Mxfp8KvCache cache{keys.get(), values.get(), key_scales.get(),
                                     value_scales.get(), floor_pages + short_table};
            const Mxfp8PageTable table{pages.get(), floor_pages + !short_table};
            Graph graph;
            check_cuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
                       "Begin missing tail page capture");
            const auto status = decoder::tensor_core_multirow_prefill_gqa(
                cache, table, position, rows, queries.get(), output.get() + guard, stream);
            check_cuda(cudaStreamEndCapture(stream, &graph.graph), "End missing tail page capture");
            std::size_t count = 0;
            check_cuda(cudaGraphGetNodes(graph.graph, nullptr, &count), "Count missing page nodes");
            if (status != cudaErrorInvalidValue || count) {
                throw std::runtime_error("Short page count admitted an unaligned paired tile");
            }
        }
        std::puts("MXFP8 paired tail rejects short physical cache and logical page table");
    }

    std::vector<BFloat> run(const char* label, const Input& input, int position, Dispatch expected,
                            const std::vector<Sample>& samples, cudaStream_t stream,
                            RunOptions options = RunOptions()) {
        keys.upload(input.keys, stream, options.key_offset);
        values.upload(input.values, stream, options.value_offset);
        key_scales.upload(input.key_scales, stream, options.key_scale_offset);
        value_scales.upload(input.value_scales, stream, options.value_scale_offset);
        pages.upload(input.pages, stream);
        queries.upload(input.queries, stream);
        constexpr auto output_bytes = (output_elements + 2 * guard) * sizeof(BFloat);
        check_cuda(cudaMemsetAsync(output.get(), 0xff, output_bytes, stream),
                   "Poison Flash check output");
        const Mxfp8KvCache cache{keys.get() + options.key_offset,
                                 values.get() + options.value_offset,
                                 key_scales.get() + options.key_scale_offset,
                                 value_scales.get() + options.value_scale_offset,
                                 input.physical_pages};
        const Mxfp8PageTable page_table{pages.get(), static_cast<int>(input.pages.size())};
        Graph graph;
        check_cuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
                   "Begin Flash check capture");
        const auto dispatch = options.use_exact
            ? mxfp8_gqa_attention(cache, page_table, position + options.count, position,
                                  options.count, queries.get(), output.get() + guard, stream)
            : decoder::tensor_core_multirow_prefill_gqa(
                  cache, page_table, position, options.count, queries.get(),
                  output.get() + guard, stream);
        const auto capture = cudaStreamEndCapture(stream, &graph.graph);
        check_cuda(dispatch, "Dispatch captured prefill");
        check_cuda(capture, "End Flash check capture");
        check_dispatch(label, graph.graph, expected);
        check_cuda(cudaGraphInstantiate(&graph.executable, graph.graph, 0),
                   "Instantiate Flash check graph");
        check_cuda(cudaGraphLaunch(graph.executable, stream), "Replay Flash check graph");
        auto result = output.download(stream);
        if (options.repeat_graph) {
            check_cuda(cudaMemsetAsync(output.get(), 0xff, output_bytes, stream),
                       "Poison output before second TMEM graph replay");
            check_cuda(cudaGraphLaunch(graph.executable, stream), "Replay the TMEM graph again");
            const auto repeated = output.download(stream);
            if (std::memcmp(result.data(), repeated.data(), output_bytes) != 0) {
                throw std::runtime_error("TMEM graph reuse changed its output");
            }
            std::printf("MXFP8 prefill %s repeated graph matched %zu output bytes\n",
                        label, output_bytes);
        }
        if (samples.empty()) {
            check_guards(label, result, options.count);
        } else {
            check_output(label, result, samples, options.count);
        }
        return result;
    }

    void check_staged_scratch(const Input& input, int position, const std::vector<BFloat>& raw,
                              cudaStream_t stream, bool stages) const {
        constexpr std::size_t scratch_guard = 128;
        constexpr Byte poison = 0xa5;
        const auto page_heads = static_cast<std::size_t>((position + rows) / mxfp8_page_size) *
                                mxfp8_kv_heads;
        const auto key_bytes = page_heads * mxfp8_page_size * channels * sizeof(BFloat);
        const auto value_bytes = key_bytes + page_heads * sizeof(int);
        DeviceBuffer<Byte> key_scratch(key_bytes + 2 * scratch_guard);
        DeviceBuffer<Byte> value_scratch(value_bytes + 2 * scratch_guard);
        const Mxfp8KvCache cache{keys.get(), values.get(), key_scales.get(), value_scales.get(),
                                 input.physical_pages};
        const Mxfp8PageTable table{pages.get(), static_cast<int>(input.pages.size())};
        const auto check_scratch = [&](const DeviceBuffer<Byte>& buffer, std::size_t bytes,
                                       bool written) {
            const auto contents = buffer.download(stream);
            const auto clean = [](Byte value) { return value == poison; };
            const auto begin = contents.begin() + scratch_guard;
            const auto end = begin + bytes;
            if (!std::all_of(contents.begin(), begin, clean) ||
                !std::all_of(end, contents.end(), clean) ||
                std::all_of(begin, end, clean) == written) {
                throw std::runtime_error("Paired prefill wrote the wrong scratch span");
            }
        };
        for (const auto mode : {std::string_view("fit"), std::string_view("short"),
                                std::string_view("capture")}) {
            check_cuda(cudaMemsetAsync(key_scratch.get(), poison, key_bytes + 2 * scratch_guard,
                                      stream), "Poison decoded keys");
            check_cuda(cudaMemsetAsync(value_scratch.get(), poison,
                                      value_bytes + 2 * scratch_guard, stream),
                       "Poison decoded values");
            check_cuda(cudaMemsetAsync(output.get(), 0xff, raw.size() * sizeof(BFloat), stream),
                       "Poison staged attention output");
            const decoder::PairedPrefillScratch scratch{
                reinterpret_cast<std::byte*>(key_scratch.get() + scratch_guard), key_bytes,
                reinterpret_cast<std::byte*>(value_scratch.get() + scratch_guard),
                value_bytes - (mode == "short")};
            Graph graph;
            if (mode == "capture") {
                check_cuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
                           "Begin paired capture with scratch");
            }
            check_cuda(decoder::tensor_core_model_prefill_gqa(
                           cache, table, position, rows, queries.get(), output.get() + guard,
                           stream, scratch), "Dispatch paired attention with scratch");
            if (mode == "capture") {
                check_cuda(cudaStreamEndCapture(stream, &graph.graph),
                           "End paired capture with scratch");
                check_dispatch(stages ? "staged tile capture with scratch"
                                      : "early tile capture with scratch", graph.graph,
                               stages ? tmem_paired_tail : tmem_paired);
                check_cuda(cudaGraphInstantiate(&graph.executable, graph.graph, 0),
                           "Instantiate paired capture with scratch");
                check_cuda(cudaGraphLaunch(graph.executable, stream),
                           "Replay paired capture with scratch");
            }
            const auto result = output.download(stream);
            if (std::memcmp(result.data(), raw.data(), raw.size() * sizeof(BFloat))) {
                throw std::runtime_error("Paired workspace changed the raw attention output");
            }
            check_scratch(key_scratch, key_bytes, stages && mode == "fit");
            check_scratch(value_scratch, value_bytes, stages && mode == "fit");
        }
        std::puts("MXFP8 paired scratch exact-fit, one-byte-short and capture matched raw");
    }
};

void check_future_values(const char* label, Input& input, DeviceInput& device, int position,
                         int first_changed_row, const std::vector<BFloat>& baseline,
                         Dispatch dispatch, cudaStream_t stream) {
    for (int row = first_changed_row; row < std::min(rows, first_changed_row + 64); ++row) {
        const auto payload = input.payload_offset(position + row, 0);
        const auto scale = input.scale_offset(position + row, 0);
        std::fill_n(input.values.begin() + payload, channels, 0x48);
        std::fill_n(input.value_scales.begin() + scale, mxfp8_groups, 127);
    }
    const auto changed = device.run(label, input, position, dispatch,
                                    attention_samples(input, position), stream);
    const auto prefix = static_cast<std::size_t>(first_changed_row) * row_width;
    std::size_t suffix_differences = 0;
    for (std::size_t index = 0; index < output_elements; ++index) {
        if (__bfloat16_as_ushort(baseline[guard + index]) ==
            __bfloat16_as_ushort(changed[guard + index])) {
            continue;
        }
        if (index < prefix) {
            throw std::runtime_error(std::string(label) + ": changed a causally masked row");
        }
        ++suffix_differences;
    }
    if (suffix_differences == 0) {
        throw std::runtime_error(std::string(label) + ": visible rows ignored the changed values");
    }
    std::printf("MXFP8 prefill %s unchanged_prefix=%zu changed_suffix=%zu\n",
                label, prefix, suffix_differences);
}

Dispatch long_1023_dispatch() {
    int device = 0;
    int multiprocessors = 0;
    int shared_limit = 0;
    check_cuda(cudaGetDevice(&device), "Get Flash check device");
    check_cuda(cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, device),
               "Read Flash check multiprocessor count");
    check_cuda(cudaDeviceGetAttribute(&shared_limit, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                                      device),
               "Read Flash check shared memory limit");
    if (multiprocessors >= 384 && shared_limit >= 137776) {
        return {"flash_prefill_attentionILi32ELb1ELb1EE", 32, head_blocks, 1, 256};
    }
    if (multiprocessors >= 264 && shared_limit >= 154800) {
        return {"flash_prefill_attentionILi48ELb1ELb1EE", 22, head_blocks, 1, 256};
    }
    if (multiprocessors >= 144 && multiprocessors < 192 && shared_limit >= 197360) {
        return {"flash_prefill_attentionILi88ELb1ELb1EE", 12, head_blocks, 1, 352};
    }
    if (shared_limit >= 171824) {
        return {"flash_prefill_attentionILi64ELb1ELb1EE", 16, head_blocks, 1, 256};
    }
    if (shared_limit >= 138032) {
        return {"flash_prefill_attentionILi64ELb1ELb0EE", 16, head_blocks, 1, 256};
    }
    return ordered_dispatch(1023);
}

void check_paired_prefill(cudaStream_t stream) {
    constexpr int position = 8192;
    constexpr int partial = rows - 1;
    constexpr Dispatch synchronous{"flash_prefill_attentionILi64ELb0ELb0EE", rows / 64};
    Input input(position + rows + mxfp8_page_size);
    input.add_causal_marker(position, 64);
    input.add_causal_marker(position, rows - 1);
    DeviceInput device(input);
    const auto samples = attention_samples(input, position);
    const auto baseline = device.run("default paired start", input, position, tmem_paired,
                                     samples, stream, {.repeat_graph = true});
    device.run("paired value scales aligned to eight bytes", input, position, flash,
               samples, stream, {.value_scale_offset = 8});
    device.run("paired unaligned value scales", input, position, synchronous, samples, stream,
               {.value_scale_offset = 1});
    device.run("paired page-aligned before staging bound", input, position - mxfp8_page_size, flash,
               attention_samples(input, position - mxfp8_page_size), stream);
    device.run("paired page-aligned nonchunk start", input, position + mxfp8_page_size, flash,
               attention_samples(input, position + mxfp8_page_size), stream);
    const auto partial_samples = attention_samples(input, position, partial);
    const auto aligned_partial = device.run("paired partial tile", input, position,
                                            long_1023_dispatch(), partial_samples, stream,
                                            {.count = partial});
    constexpr Dispatch synchronous_partial{"flash_prefill_attentionILi64ELb1ELb0EE",
                                            16, head_blocks, 1, 256};
    const auto unaligned_partial = device.run("paired partial tile with unaligned value scales",
                                              input, position, synchronous_partial, partial_samples,
                                              stream, {.count = partial, .value_scale_offset = 1});
    if (std::memcmp(aligned_partial.data(), unaligned_partial.data(),
                    aligned_partial.size() * sizeof(BFloat))) {
        throw std::runtime_error("Unaligned scales changed the partial Flash output");
    }
    check_future_values("paired final page causality", input, device, position,
                        rows - mxfp8_page_size / 2, baseline, tmem_paired, stream);
}

std::vector<BFloat> check_paired_tail_matches_wide(const char* label, const Input& input,
                                                 DeviceInput& device, int position,
                                                 const std::vector<Sample>& samples,
                                                 cudaStream_t stream) {
    const auto paired = device.run(label, input, position, tmem_paired_tail, samples, stream,
                                   {.repeat_graph = true});
    const auto control = std::string(label) + " scale-alignment fallback";
    const auto wide = device.run(control.c_str(), input, position, flash, {}, stream,
                                 {.value_scale_offset = 8});
    if (std::memcmp(paired.data(), wide.data(), paired.size() * sizeof(BFloat))) {
        throw std::runtime_error(std::string(label) +
                                 ": unaligned paired output differs from wide");
    }
    return paired;
}

void check_paired_tail_repair(Input& input, DeviceInput& device, int position,
                              const std::vector<BFloat>& clean, cudaStream_t stream) {
    constexpr int rows_per_cta = 64;
    constexpr int keys_per_rank = 32;
    const int partial_start = position + 1;
    for (int rank : {0, 1}) {
        const int token = partial_start + rank * keys_per_rank;
        const auto offset = input.payload_offset(token, 0);
        const auto saved = input.keys[offset];
        input.keys[offset] = 0x7f;
        const auto repaired = check_paired_tail_matches_wide(
            rank == 0 ? "rank zero dirty tail key" : "rank one dirty tail key",
            input, device, position, {}, stream);
        input.keys[offset] = saved;
        const auto prefix = static_cast<std::size_t>(token - position) * row_width;
        if (std::memcmp(clean.data() + guard, repaired.data() + guard, prefix * sizeof(BFloat))) {
            throw std::runtime_error("Dirty tail key changed a causally masked row");
        }
        for (int row : {token - position, rows_per_cta}) {
            for (int head : {0, queries_per_kv - 1, queries_per_kv}) {
                const auto index = guard + static_cast<std::size_t>(row) * row_width
                    + head * channels;
                if (std::isfinite(__bfloat162float(repaired[index])) == (head < queries_per_kv)) {
                    throw std::runtime_error("Dirty tail key repaired the wrong CTA row or head");
                }
            }
        }
    }
}

void check_unaligned_paired_prefill(cudaStream_t stream) {
    constexpr int aligned = 8192;
    for (int offset : {1, 63, 64, 127}) {
        const int position = aligned + offset;
        Input input(position + rows);
        input.add_causal_marker(position, 64);
        input.add_causal_marker(position, rows - 1);
        DeviceInput device(input);
        const auto label = "paired page offset " + std::to_string(offset);
        const auto clean = check_paired_tail_matches_wide(
            label.c_str(), input, device, position, attention_samples(input, position),
            stream);
        if (offset == 1) {
            device.check_missing_tail_page(position, stream);
        }
        if (offset == 127) {
            check_paired_tail_repair(input, device, position, clean, stream);
        }
    }
}

void check_paired_lower_edge(cudaStream_t stream) {
    constexpr int first = 4096;
    constexpr int key_tile = 64;
    Input input(first + rows + 1);
    DeviceInput device(input);
    device.run("TMEM aligned start below lower bound", input, first - key_tile, flash,
               attention_samples(input, first - key_tile), stream);
    device.run("TMEM lower edge predecessor", input, first - 1, flash,
               attention_samples(input, first - 1), stream);
    const auto samples = attention_samples(input, first);
    const auto raw = device.run("TMEM first aligned start", input, first, tmem_paired,
                                samples, stream, {.repeat_graph = true});
    device.check_staged_scratch(input, first, raw, stream, false);
    device.run("TMEM nonaligned start", input, first + 1, flash,
               attention_samples(input, first + 1), stream);
    device.run("TMEM unaligned key", input, first, multirow, samples, stream,
               {.key_offset = 1});
    device.run("TMEM unaligned value", input, first, multirow, samples, stream,
               {.value_offset = 1});
}

void check_long_flash(cudaStream_t stream) {
    constexpr int ordered_visible = 32768;
    constexpr int flash_visible = 131072;
    constexpr int aligned_above_limit = flash_visible + 64;
    constexpr int partial = rows - 1;
    constexpr Dispatch partial_protected{"protected_prefill_attention", partial};
    Input input(aligned_above_limit);
    DeviceInput device(input);
    for (int visible : {ordered_visible, ordered_visible + 1}) {
        const int position = visible - rows;
        const auto label = std::to_string(visible) + " visible";
        const auto samples = attention_samples(input, position);
        const auto expected = visible == ordered_visible ? tmem_paired : flash;
        const auto baseline = device.run(label.c_str(), input, position, expected, samples, stream);
        if (visible == ordered_visible + 1) {
            constexpr Dispatch sync{"flash_prefill_attentionILi64ELb0ELb0EE", rows / 64};
            const auto scale_label = label + " with unaligned key scales";
            const auto unaligned = device.run(scale_label.c_str(), input, position,
                                              sync, samples, stream, {.key_scale_offset = 1});
            if (std::memcmp(unaligned.data(), baseline.data(), baseline.size() * sizeof(BFloat))) {
                throw std::runtime_error("Unaligned Flash scales changed the wide tile output");
            }
            device.run("32769 unaligned key", input, position, protected_rows, samples, stream,
                       {.key_offset = 1});
            device.run("32769 unaligned value", input, position, protected_rows, samples, stream,
                       {.value_offset = 1});
        }
    }
    device.run("32768 partial tile", input, ordered_visible - partial, long_1023_dispatch(),
               attention_samples(input, ordered_visible - partial, partial), stream,
               {.count = partial});
    device.run("32769 partial tile", input, ordered_visible + 1 - partial, partial_protected,
               attention_samples(input, ordered_visible + 1 - partial, partial), stream,
               {.count = partial});
    device.run("TMEM first long aligned start", input, ordered_visible, tmem_paired,
               attention_samples(input, ordered_visible), stream, {.repeat_graph = true});
    for (int visible : {65536, 98304}) {
        const int position = visible - rows;
        const auto label = std::to_string(visible) + " visible";
        device.run(label.c_str(), input, position, tmem_paired,
                   attention_samples(input, position), stream);
    }
    const int position = flash_visible - rows;
    const auto samples = attention_samples(input, position);
    const auto baseline =
        device.run("131072 visible", input, position, tmem_paired, samples, stream);
    device.run("131072 exact", input, position, exact, samples, stream, {.use_exact = true});
    device.run("131073 visible", input, position + 1, protected_rows,
               attention_samples(input, position + 1), stream);
    device.run("TMEM aligned end above upper bound", input, aligned_above_limit - rows,
               protected_rows, attention_samples(input, aligned_above_limit - rows), stream);
    check_future_values("131072 future value perturbation", input, device, position, 65,
                        baseline, tmem_paired, stream);
    input.add_score_overflow(position);
    device.run("131072 score overflow recovery", input, position, tmem_paired,
               attention_samples(input, position), stream);
}

void check_partial_boundaries(cudaStream_t stream) {
    constexpr int position = 61;
    for (int count : {143, 144, 191, 192, 193, 1023, rows}) {
        Input input(position + count, count);
        input.add_causal_marker(position, count - 1);
        DeviceInput device(input);
        const auto samples = attention_samples(input, position, count);
        const auto label = std::to_string(count) + " query rows";
        const auto expected = count == 143 ? ordered_dispatch(count) : flash_dispatch(count);
        const auto baseline = device.run(label.c_str(), input, position, expected, samples,
                                         stream, {.count = count});
        if (count == 144) {
            device.run("144 rows unaligned key", input, position, ordered_dispatch(count),
                       samples, stream, {.count = count, .key_offset = 1});
            device.run("144 rows unaligned value", input, position, ordered_dispatch(count),
                       samples, stream, {.count = count, .value_offset = 1});
        }
        if (count == 193) {
            constexpr Dispatch sync{"flash_prefill_attentionILi64ELb1ELb0EE", 4};
            const auto key = device.run("193 rows unaligned key scales", input, position, sync,
                                        samples, stream, {.count = count, .key_scale_offset = 1});
            const auto value = device.run("193 rows unaligned value scales", input, position,
                                          sync, samples, stream,
                                          {.count = count, .value_scale_offset = 1});
            const auto bytes = baseline.size() * sizeof(BFloat);
            if (std::memcmp(key.data(), baseline.data(), bytes) != 0 ||
                std::memcmp(value.data(), baseline.data(), bytes) != 0) {
                throw std::runtime_error("Unaligned Flash scales changed the output");
            }
        }
    }
}

void check_tail_recovery(cudaStream_t stream, int position, int count, Dispatch expected) {
    const int last = count - 1;
    Input input(position + count, count);
    DeviceInput device(input);
    const RunOptions options{.count = count};
    input.add_score_overflow(position, last);
    const auto baseline = device.run("final row score overflow", input, position,
                                     expected, attention_samples(input, position, count),
                                     stream, options);
    const auto changed_value = input.payload_offset(position + last, 0);
    input.values[changed_value] = 0x7f;
    input.queries[last * row_width + 2 * queries_per_kv * channels] =
        __ushort_as_bfloat16(0x7f80);
    const auto changed = device.run("final row nonfinite inputs", input, position,
                                    expected, {}, stream, options);
    const auto last_offset = static_cast<std::size_t>(last) * row_width;
    for (std::size_t index = 0; index < last_offset; ++index) {
        if (__bfloat16_as_ushort(baseline[guard + index]) !=
            __bfloat16_as_ushort(changed[guard + index])) {
            throw std::runtime_error("Nonfinite future input changed a causal prefix output");
        }
    }
    std::size_t nonfinite = 0;
    for (int head = 0; head < mxfp8_query_heads; ++head) {
        for (int channel = 0; channel < channels; ++channel) {
            const bool poisoned = head == 2 * queries_per_kv ||
                                  (head < queries_per_kv && channel == 0);
            const auto value = changed[guard + last_offset + head * channels + channel];
            if (std::isfinite(__bfloat162float(value)) == poisoned) {
                throw std::runtime_error("Nonfinite recovery affected the wrong output channel");
            }
            nonfinite += static_cast<std::size_t>(poisoned);
        }
    }
    std::printf("MXFP8 prefill tail nonfinite causal_prefix=%zu nonfinite_outputs=%zu\n",
                last_offset, nonfinite);
}

}

void check_flash_prefill(cudaStream_t stream) {
    check_dispatch_errors(stream);
    check_paired_lower_edge(stream);
    check_unaligned_paired_prefill(stream);
    check_paired_prefill(stream);
    check_tail_recovery(stream, 8192, rows, tmem_paired);
    {
        constexpr int position = 8193;
        Input input(position + rows);
        DeviceInput device(input);
        const auto raw = device.run("paired scratch raw reference", input, position,
                                    tmem_paired_tail, attention_samples(input, position), stream);
        device.check_staged_scratch(input, position, raw, stream, true);
    }
    check_partial_boundaries(stream);
    check_tail_recovery(stream, 61, 193, flash_dispatch(193));
    constexpr int position = 61;
    constexpr auto early = flash_dispatch(rows);
    {
        Input input(position + rows);
        input.add_causal_marker(position, 1);
        input.add_causal_marker(position, 63);
        input.add_causal_marker(position, 64);
        input.add_causal_marker(position, 67);
        input.add_causal_marker(position, rows - 1);
        const auto samples = attention_samples(input, position);
        DeviceInput device(input);
        device.run("aligned causal tail", input, position, early, samples, stream);
        device.run("aligned exact", input, position, exact, samples, stream, {.use_exact = true});
        device.run("unaligned key", input, position, multirow, samples, stream, {.key_offset = 1});
        device.run("unaligned value", input, position, multirow, samples, stream,
                   {.value_offset = 1});
        input.add_score_overflow(position);
        device.run("Flash score overflow recovery", input, position, early,
                   attention_samples(input, position), stream);
    }
    check_long_flash(stream);
}

}
