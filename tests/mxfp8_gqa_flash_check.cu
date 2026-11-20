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
};

constexpr Dispatch flash{"flash_prefill_attention", rows / 64};
constexpr Dispatch exact{"attention_kernel", rows, mxfp8_query_heads};
constexpr Dispatch multirow{"multirow_prefill_attention", rows / 4};
constexpr Dispatch protected_rows{"protected_prefill_attention", rows};

constexpr Dispatch flash_dispatch(int count) {
    return {count % 64 == 0 ? "flash_prefill_attentionILb0EE" : "flash_prefill_attentionILb1EE",
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
    bool use_exact = false;
};

struct DeviceInput {
    DeviceBuffer<Byte> keys, values, key_scales, value_scales;
    DeviceBuffer<std::int32_t> pages;
    DeviceBuffer<BFloat> queries;
    DeviceBuffer<BFloat> output{output_elements + 2 * guard};

    explicit DeviceInput(const Input& input)
        : keys(input.keys.size() + 1), values(input.values.size() + 1),
          key_scales(input.key_scales.size()), value_scales(input.value_scales.size()),
          pages(input.pages.size()), queries(input.queries.size()) {}

    std::vector<BFloat> run(const char* label, const Input& input, int position, Dispatch expected,
                            const std::vector<Sample>& samples, cudaStream_t stream,
                            RunOptions options = RunOptions()) {
        keys.upload(input.keys, stream, options.key_offset);
        values.upload(input.values, stream, options.value_offset);
        key_scales.upload(input.key_scales, stream);
        value_scales.upload(input.value_scales, stream);
        pages.upload(input.pages, stream);
        queries.upload(input.queries, stream);
        constexpr auto output_bytes = (output_elements + 2 * guard) * sizeof(BFloat);
        check_cuda(cudaMemsetAsync(output.get(), 0xff, output_bytes, stream),
                   "Poison Flash check output");
        const Mxfp8KvCache cache{keys.get() + options.key_offset,
                                 values.get() + options.value_offset, key_scales.get(),
                                 value_scales.get(), input.physical_pages};
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
        if (samples.empty()) {
            check_guards(label, result, options.count);
        } else {
            check_output(label, result, samples, options.count);
        }
        return result;
    }
};

void check_long_causality(Input& input, DeviceInput& device, int position,
                          const std::vector<BFloat>& baseline, cudaStream_t stream) {
    constexpr int first_changed_row = 65;
    for (int row = first_changed_row; row < first_changed_row + 64; ++row) {
        const auto payload = input.payload_offset(position + row, 0);
        const auto scale = input.scale_offset(position + row, 0);
        std::fill_n(input.values.begin() + payload, channels, 0x48);
        std::fill_n(input.value_scales.begin() + scale, mxfp8_groups, 127);
    }
    const auto changed = device.run("131072 future value perturbation", input, position, flash,
                                    attention_samples(input, position), stream);
    constexpr std::size_t prefix = first_changed_row * row_width;
    std::size_t suffix_differences = 0;
    for (std::size_t index = 0; index < output_elements; ++index) {
        if (__bfloat16_as_ushort(baseline[guard + index]) ==
            __bfloat16_as_ushort(changed[guard + index])) {
            continue;
        }
        if (index < prefix) {
            throw std::runtime_error("Long Flash changed an output before the future value range");
        }
        ++suffix_differences;
    }
    if (suffix_differences == 0) {
        throw std::runtime_error("Long Flash ignored the future value range in visible rows");
    }
    std::printf("MXFP8 prefill long causality unchanged_prefix=%zu changed_suffix=%zu\n",
                prefix, suffix_differences);
}

void check_long_flash(cudaStream_t stream) {
    constexpr int ordered_visible = 32768;
    constexpr int flash_visible = 131072;
    constexpr int partial = rows - 1;
    constexpr Dispatch partial_protected{"protected_prefill_attention", partial};
    Input input(flash_visible + 1);
    DeviceInput device(input);
    for (int visible : {ordered_visible, ordered_visible + 1, 65536}) {
        const int position = visible - rows;
        const auto label = std::to_string(visible) + " visible";
        const auto samples = attention_samples(input, position);
        device.run(label.c_str(), input, position, flash, samples, stream);
        if (visible == ordered_visible + 1) {
            device.run("32769 unaligned key", input, position, protected_rows, samples, stream,
                       {.key_offset = 1});
            device.run("32769 unaligned value", input, position, protected_rows, samples, stream,
                       {.value_offset = 1});
        }
    }
    device.run("32768 partial tile", input, ordered_visible - partial, flash_dispatch(partial),
               attention_samples(input, ordered_visible - partial, partial), stream,
               {.count = partial});
    device.run("32769 partial tile", input, ordered_visible + 1 - partial, partial_protected,
               attention_samples(input, ordered_visible + 1 - partial, partial), stream,
               {.count = partial});
    const int position = flash_visible - rows;
    const auto samples = attention_samples(input, position);
    const auto baseline = device.run("131072 visible", input, position, flash, samples, stream);
    device.run("131072 exact", input, position, exact, samples, stream, {.use_exact = true});
    device.run("131073 visible", input, position + 1, protected_rows,
               attention_samples(input, position + 1), stream);
    check_long_causality(input, device, position, baseline, stream);
    input.add_score_overflow(position);
    device.run("131072 Flash score overflow recovery", input, position, flash,
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
        device.run(label.c_str(), input, position, expected, samples, stream, {.count = count});
        if (count == 144) {
            device.run("144 rows unaligned key", input, position, ordered_dispatch(count),
                       samples, stream, {.count = count, .key_offset = 1});
            device.run("144 rows unaligned value", input, position, ordered_dispatch(count),
                       samples, stream, {.count = count, .value_offset = 1});
        }
    }
}

void check_partial_recovery(cudaStream_t stream) {
    constexpr int position = 61;
    constexpr int count = 193;
    constexpr int last = count - 1;
    Input input(position + count, count);
    DeviceInput device(input);
    const auto expected = flash_dispatch(count);
    const RunOptions options{.count = count};
    input.add_score_overflow(position, last);
    const auto baseline = device.run("partial final row score overflow", input, position,
                                     expected, attention_samples(input, position, count),
                                     stream, options);
    const auto changed_value = input.payload_offset(position + last, 0);
    input.values[changed_value] = 0x7f;
    input.queries[last * row_width + 2 * queries_per_kv * channels] =
        __ushort_as_bfloat16(0x7f80);
    const auto changed = device.run("partial final row nonfinite inputs", input, position,
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
    std::printf("MXFP8 prefill partial nonfinite causal_prefix=%zu nonfinite_outputs=%zu\n",
                last_offset, nonfinite);
}

}

void check_flash_prefill(cudaStream_t stream) {
    check_dispatch_errors(stream);
    check_partial_boundaries(stream);
    check_partial_recovery(stream);
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
        device.run("aligned exact", input, position, exact, samples, stream, {.use_exact = true});
        device.run("unaligned key", input, position, multirow, samples, stream, {.key_offset = 1});
        device.run("unaligned value", input, position, multirow, samples, stream,
                   {.value_offset = 1});
        input.add_score_overflow(position);
        device.run("Flash score overflow recovery", input, position, flash,
                   attention_samples(input, position), stream);
    }
    check_long_flash(stream);
}

}
