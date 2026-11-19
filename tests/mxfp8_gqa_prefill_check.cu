#include "sglang/decoder_cuda.cuh"
#include "sglang/mxfp8_gqa.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace gqa_test {

std::vector<__nv_bfloat16> make_queries(int start_token, int count);
double decode_value(std::uint8_t code, std::uint8_t scale);
void verify_ordered_prefill(const char* phase, sglang::Mxfp8KvCache cache,
                            sglang::Mxfp8PageTable pages, int first, int rows,
                            const __nv_bfloat16* queries, const std::vector<__nv_bfloat16>& actual,
                            cudaStream_t stream);

}

namespace {

using BFloat = __nv_bfloat16;
using Byte = std::uint8_t;
using gqa_test::decode_value;
using gqa_test::make_queries;
using gqa_test::verify_ordered_prefill;

constexpr int head_width = 256;
constexpr int kv_heads = 4;
constexpr int query_heads = 24;
constexpr int page_tokens = 128;
constexpr int physical_pages = 18;
constexpr int query_elements = query_heads * head_width;
constexpr std::size_t kv_elements_per_token = kv_heads * head_width;
constexpr std::size_t payload_bytes = physical_pages * sglang::mxfp8_payload_bytes_per_page;
constexpr std::size_t scale_bytes = physical_pages * sglang::mxfp8_scale_bytes_per_page;
constexpr Byte attention_sentinel = 0xff;

std::size_t row_index(int token, int head, int heads) {
    return static_cast<std::size_t>(token) * heads + head;
}

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + " failed with " +
                                 cudaGetErrorString(status));
    }
}

template <class Element>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count) : count_(count) {
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&data_), bytes()), "Allocate test buffer");
    }
    ~DeviceBuffer() { cudaFree(data_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    Element* get() const { return data_; }

    void upload(const Element* source, cudaStream_t stream) const {
        check_cuda(cudaMemcpyAsync(data_, source, bytes(), cudaMemcpyHostToDevice, stream),
                   "Upload test buffer");
    }

    void fill(Byte sentinel, cudaStream_t stream) const {
        check_cuda(cudaMemsetAsync(data_, sentinel, bytes(), stream), "Fill test buffer");
    }

    std::vector<Element> download(cudaStream_t stream) const {
        std::vector<Element> result(count_);
        check_cuda(cudaMemcpyAsync(result.data(), data_, bytes(), cudaMemcpyDeviceToHost, stream),
                   "Download test buffer");
        check_cuda(cudaStreamSynchronize(stream), "Complete test stream");
        return result;
    }

private:
    std::size_t bytes() const { return count_ * sizeof(Element); }

    std::size_t count_;
    Element* data_ = nullptr;
};

class Graph {
public:
    Graph() = default;
    Graph(const Graph&) = delete;
    Graph& operator=(const Graph&) = delete;

    ~Graph() {
        if (executable_) {
            cudaGraphExecDestroy(executable_);
        }
        if (graph_) {
            cudaGraphDestroy(graph_);
        }
    }

    template <class Submit>
    void capture(cudaStream_t stream, Submit submit) {
        check_cuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
                   "Begin attention capture");
        const auto launch_status = submit();
        const auto capture_status = cudaStreamEndCapture(stream, &graph_);
        check_cuda(launch_status, "Submit captured attention");
        check_cuda(capture_status, "End attention capture");
        check_cuda(cudaGraphInstantiate(&executable_, graph_, 0), "Instantiate attention graph");
    }

    void replay(cudaStream_t stream) const {
        check_cuda(cudaGraphLaunch(executable_, stream), "Replay attention graph");
    }

private:
    cudaGraph_t graph_ = nullptr;
    cudaGraphExec_t executable_ = nullptr;
};

struct Pages {
    const std::vector<std::int32_t> mapping;
    DeviceBuffer<std::int32_t> device{mapping.size()};

    sglang::Mxfp8PageTable view() const {
        return {device.get(), static_cast<int>(mapping.size())};
    }
};

struct RawCache {
    DeviceBuffer<Byte> keys{payload_bytes};
    DeviceBuffer<Byte> values{payload_bytes};
    DeviceBuffer<Byte> key_scales{scale_bytes};
    DeviceBuffer<Byte> value_scales{scale_bytes};

    sglang::Mxfp8KvCache view() const {
        return {keys.get(), values.get(), key_scales.get(), value_scales.get(), physical_pages};
    }
};

struct GdnBuffers {
    DeviceBuffer<BFloat> history{sglang::text_gdn_qkv_width * sglang::text_gdn_history_taps};
    DeviceBuffer<float> recurrent{sglang::text_gdn_value_width * sglang::text_gdn_head_dim};
    DeviceBuffer<BFloat> output{sglang::text_gdn_value_width};

    sglang::TextGdnState state() const {
        return {history.get(), recurrent.get()};
    }

    void initialize(cudaStream_t stream) {
        history.fill(0, stream);
        recurrent.fill(0, stream);
        output.fill(attention_sentinel, stream);
    }
};

template <class Element>
void require_same(const std::vector<Element>& actual, const std::vector<Element>& expected,
                  const char* message) {
    if (actual.size() != expected.size() ||
        std::memcmp(actual.data(), expected.data(), expected.size() * sizeof(Element)) != 0) {
        throw std::runtime_error(message);
    }
}

constexpr int prefill_guard = 8;

std::vector<BFloat> download_prefill(const DeviceBuffer<BFloat>& output, cudaStream_t stream) {
    const auto actual = output.download(stream);
    for (const auto* guard : {actual.data(), actual.data() + actual.size() - prefill_guard}) {
        const auto* bytes = reinterpret_cast<const Byte*>(guard);
        if (!std::all_of(bytes, bytes + prefill_guard * sizeof(BFloat),
                         [](Byte byte) { return byte == attention_sentinel; })) {
            throw std::runtime_error("Multirow prefill overwrote an output guard");
        }
    }
    return {actual.begin() + prefill_guard, actual.end() - prefill_guard};
}

void check_ordered_prefill_extent(cudaStream_t stream) {
    constexpr int limit = 8192;
    constexpr int page_count = limit / page_tokens + 1;
    constexpr int rows = 64;
    std::vector<std::int32_t> mapping(page_count);
    for (int index = 0; index < page_count; ++index) {
        mapping[index] = page_count - 1 - index;
    }
    const Pages pages{std::move(mapping)};
    pages.device.upload(pages.mapping.data(), stream);
    constexpr auto payload = page_count * sglang::mxfp8_payload_bytes_per_page;
    constexpr auto scales = page_count * sglang::mxfp8_scale_bytes_per_page;
    const DeviceBuffer<Byte> keys(payload), values(payload);
    const DeviceBuffer<Byte> key_scales(scales), value_scales(scales);
    keys.fill(0x38, stream);
    values.fill(0x38, stream);
    key_scales.fill(127, stream);
    value_scales.fill(127, stream);
    const sglang::Mxfp8KvCache cache{keys.get(), values.get(), key_scales.get(), value_scales.get(),
                                   page_count};
    const DeviceBuffer<BFloat> queries(rows * query_elements);
    for (const auto [first, count] : {std::pair{31, 63}, {31, 64}, {limit - rows, rows},
                                     {limit - rows + 1, rows}}) {
        const auto host_queries = make_queries(first, count);
        const DeviceBuffer<BFloat> output(count * query_elements + 2 * prefill_guard);
        queries.upload(host_queries.data(), stream);
        output.fill(attention_sentinel, stream);
        check_cuda(sglang::decoder::tensor_core_multirow_prefill_gqa(
                       cache, pages.view(), first, count, queries.get(),
                       output.get() + prefill_guard, stream), "Launch prefill at dispatch edge");
        const auto actual = download_prefill(output, stream);
        if (!std::all_of(actual.begin(), actual.end(), [](BFloat value) {
            return __bfloat162float(value) == 1.0f;
        })) {
            throw std::runtime_error("Constant-value prefill failed at dispatch edge");
        }
        verify_ordered_prefill("prefill dispatch edge", cache, pages.view(), first, count,
                               queries.get(), actual, stream);
        std::printf("Ordered prefill dispatch edge first=%d rows=%d\n", first, count);
    }
}

void check_ordered_prefill_precision(cudaStream_t stream) {
    constexpr int rows = page_tokens;
    constexpr auto elements = rows * query_elements;
    const Pages pages{{5}};
    pages.device.upload(pages.mapping.data(), stream);
    const RawCache cache;
    const auto view = cache.view();
    const DeviceBuffer<BFloat> queries(elements);
    const DeviceBuffer<BFloat> output(elements + 2 * prefill_guard);
    queries.fill(0, stream);
    check_cuda(cudaMemsetAsync(view.key_payload, 0, payload_bytes, stream), "Clear boundary keys");
    check_cuda(cudaMemsetAsync(view.key_scales, 127, scale_bytes, stream),
               "Set boundary key scales");
    const auto first_payload = view.value_payload
                               + pages.mapping.front() * page_tokens * kv_elements_per_token;
    for (const Byte scale : {0, 48, 127, 160, 252, 255}) {
        const bool sparse = scale == 255;
        const bool rounding = scale == 48 || scale == 127 || scale == 160;
        const Byte background = sparse ? 0 : 0x38;
        const Byte leading = sparse ? 0x01 : rounding ? 0x39 : background;
        check_cuda(cudaMemsetAsync(view.value_payload, background, payload_bytes, stream),
                   "Set boundary value payload");
        check_cuda(cudaMemsetAsync(view.value_scales, scale, scale_bytes, stream),
                   "Set boundary value scales");
        check_cuda(cudaMemsetAsync(first_payload, leading, kv_elements_per_token, stream),
                   "Set leading token payload");
        output.fill(attention_sentinel, stream);
        check_cuda(sglang::decoder::tensor_core_multirow_prefill_gqa(
                       view, pages.view(), 0, rows, queries.get(), output.get() + prefill_guard,
                       stream), "Launch precision-boundary prefill");
        const auto actual = download_prefill(output, stream);
        const double baseline = decode_value(background, scale);
        const double first_token = decode_value(leading, scale);
        for (int row = 0; row < rows; ++row) {
            const double mean = baseline + (first_token - baseline) / (row + 1);
            const float expected = __bfloat162float(__float2bfloat16_rn(static_cast<float>(mean)));
            const auto begin = actual.begin() + row * query_elements;
            if (!std::all_of(begin, begin + query_elements, [expected](BFloat value) {
                return __bfloat162float(value) == expected;
            })) {
                throw std::runtime_error("Multirow prefill precision mismatch at scale "
                                         + std::to_string(scale) + ", row " + std::to_string(row));
            }
        }
        if (scale != 252) {
            verify_ordered_prefill("prefill scale boundary", view, pages.view(), 0, rows,
                                   queries.get(), actual, stream);
        }
    }
    std::puts("Multirow prefill matches BF16 tie-to-even, extreme scales and overflow recovery");
}

void check_ordered_prefill_score_overflow(cudaStream_t stream) {
    constexpr int rows = 65;
    const Pages pages{{5}};
    pages.device.upload(pages.mapping.data(), stream);
    const RawCache cache;
    const auto view = cache.view();
    const auto offset = pages.mapping.front() * page_tokens * kv_elements_per_token;
    check_cuda(cudaMemsetAsync(view.key_payload, 0, payload_bytes, stream), "Clear overflow keys");
    check_cuda(cudaMemsetAsync(view.key_scales, 127, scale_bytes, stream),
               "Set overflow key scales");
    check_cuda(cudaMemsetAsync(view.key_payload + offset, 0x58, head_width, stream),
               "Set leading overflow key");
    check_cuda(cudaMemsetAsync(view.value_payload, 0x38, payload_bytes, stream),
               "Set overflow values");
    check_cuda(cudaMemsetAsync(view.value_scales, 127, scale_bytes, stream),
               "Set overflow value scales");
    check_cuda(cudaMemsetAsync(view.value_payload + offset, 0x40, head_width, stream),
               "Set leading overflow value");
    std::vector<BFloat> input(rows * query_elements, __float2bfloat16_rn(0.0f));
    for (int row = 0; row < rows; ++row) {
        std::fill_n(input.begin() + row * query_elements, head_width,
                    __float2bfloat16_rn(0x1p126f));
    }
    const DeviceBuffer<BFloat> queries(input.size());
    queries.upload(input.data(), stream);
    const DeviceBuffer<BFloat> output(rows * query_elements + 2 * prefill_guard);
    output.fill(attention_sentinel, stream);
    check_cuda(sglang::decoder::tensor_core_multirow_prefill_gqa(
                   view, pages.view(), 0, rows, queries.get(), output.get() + prefill_guard,
                   stream), "Launch score-overflow prefill");
    const auto actual = download_prefill(output, stream);
    for (int row = 0; row < rows; ++row) {
        const float mean = __bfloat162float(__double2bfloat16(1.0 + 1.0 / (row + 1)));
        for (int head = 0; head < query_heads; ++head) {
            const float ordinary = head < query_heads / kv_heads ? mean : 1.0f;
            const float expected = head == 0 ? 2.0f : ordinary;
            const auto begin = actual.begin() + row_index(row, head, query_heads) * head_width;
            if (!std::all_of(begin, begin + head_width, [expected](BFloat value) {
                return __bfloat162float(value) == expected;
            })) {
                throw std::runtime_error("Score overflow recovery mismatch at row "
                                         + std::to_string(row) + ", head "
                                         + std::to_string(head));
            }
        }
    }
    std::puts("Score overflow recovers in paired rows and odd tail");
}

}

namespace gqa_test {

void check_bound_gdn(cudaStream_t stream) {
    using namespace sglang;
    const DeviceBuffer<BFloat> convolution(text_gdn_qkv_width * (text_gdn_history_taps + 1));
    const DeviceBuffer<BFloat> head_weights(text_gdn_value_heads), norm(text_gdn_head_dim);
    const DeviceBuffer<BFloat> qkv(text_gdn_qkv_width), z(text_gdn_value_width);
    const DeviceBuffer<BFloat> gate(text_gdn_value_heads);
    const DeviceBuffer<BFloat> scratch(text_gdn_workspace_bytes(1) / sizeof(BFloat));
    const DeviceBuffer<BFloat> output(text_gdn_value_width);
    const DeviceBuffer<TextGdnState> binding(1);
    const std::vector<BFloat> convolution_values(
        text_gdn_qkv_width * (text_gdn_history_taps + 1), __float2bfloat16_rn(0.25f));
    const std::vector<BFloat> norm_values(text_gdn_head_dim, __float2bfloat16_rn(1.0f));
    const std::vector<BFloat> z_values(text_gdn_value_width, __float2bfloat16_rn(0.5f));
    convolution.upload(convolution_values.data(), stream);
    head_weights.fill(0, stream);
    gate.fill(0, stream);
    norm.upload(norm_values.data(), stream);
    z.upload(z_values.data(), stream);
    const TextGdnWeights weights{convolution.get(), head_weights.get(), head_weights.get(),
                                 norm.get()};
    GdnBuffers actual[2], eager[2];
    for (int lane = 0; lane < 2; ++lane) {
        actual[lane].initialize(stream);
        eager[lane].initialize(stream);
    }
    Graph graph;
    graph.capture(stream, [&] {
        return decoder::decode_gdn(qkv.get(), z.get(), gate.get(), gate.get(), weights,
                                   binding.get(), scratch.get(), output.get(), stream);
    });
    for (int trial = 0; trial < 6; ++trial) {
        const int lane = trial % 2;
        std::vector<BFloat> input(text_gdn_qkv_width);
        for (int index = 0; index < text_gdn_qkv_width; ++index) {
            input[index] = __float2bfloat16_rn(((index * 7 + trial * 13) % 31 - 15) / 16.0f);
        }
        const auto state = actual[lane].state();
        qkv.upload(input.data(), stream);
        binding.upload(&state, stream);
        graph.replay(stream);
        check_cuda(text_gdn_forward(1, TextGdnMode::decode, qkv.get(), z.get(), gate.get(),
                   gate.get(), weights, eager[lane].state(), scratch.get(),
                   text_gdn_workspace_bytes(1), eager[lane].output.get(), stream),
                   "Launch eager GDN oracle");
        require_same(output.download(stream), eager[lane].output.download(stream),
                     "Rebound GDN output differs from eager");
        for (int bank = 0; bank < 2; ++bank) {
            require_same(actual[bank].history.download(stream),
                         eager[bank].history.download(stream), "Wrong GDN history bank");
            require_same(actual[bank].recurrent.download(stream),
                         eager[bank].recurrent.download(stream), "Wrong GDN recurrent bank");
        }
        const auto emitted = output.download(stream);
        if (std::none_of(emitted.begin(), emitted.end(),
                         [](BFloat value) { return __bfloat162float(value) != 0.0f; })) {
            throw std::runtime_error("Degenerate GDN output");
        }
    }
    std::puts("Rebound GDN history and recurrent banks on six device graph replays");
}

void check_tiled_prefill_scales(cudaStream_t stream) {
    constexpr int rows = 2;
    constexpr int first = page_tokens - 1;
    constexpr auto page_payload = page_tokens * kv_elements_per_token;
    constexpr auto page_scales = scale_bytes / physical_pages;
    const Pages pages{{1, 0}};
    pages.device.upload(pages.mapping.data(), stream);
    const RawCache cache;
    const auto view = cache.view();
    const auto table = pages.view();
    const DeviceBuffer<BFloat> queries(rows * query_elements);
    const DeviceBuffer<BFloat> direct(rows * query_elements);
    const DeviceBuffer<BFloat> tiled(rows * query_elements);
    const auto host_queries = make_queries(first, rows);
    queries.upload(host_queries.data(), stream);
    check_cuda(cudaMemsetAsync(view.key_payload, 0, 2 * page_payload, stream), "Clear keys");
    check_cuda(cudaMemsetAsync(view.key_scales, 127, 2 * page_scales, stream), "Set key scales");
    for (const auto [exponent, payload] : {std::pair{0, 0x38}, std::pair{255, 0x01}}) {
        check_cuda(cudaMemsetAsync(view.value_payload, payload | 0x80, page_payload, stream),
                   "Set negative FP8 values");
        check_cuda(cudaMemsetAsync(view.value_payload + page_payload, payload, page_payload,
                                   stream), "Set positive FP8 values");
        check_cuda(cudaMemsetAsync(view.value_scales, exponent, 2 * page_scales, stream),
                   "Set value scales");
        check_cuda(sglang::mxfp8_gqa_attention(view, table, first + rows, first, rows,
                   queries.get(), direct.get(), stream), "Launch direct prefill oracle");
        check_cuda(sglang::decoder::tensor_core_prefill_gqa(view, table, first, rows,
                   queries.get(), tiled.get(), stream), "Launch tiled prefill");
        const auto expected = direct.download(stream);
        const auto actual = tiled.download(stream);
        const float first_value = __bfloat162float(expected.front());
        if (!std::isfinite(first_value) || first_value <= 0.0f ||
            std::memcmp(actual.data(), expected.data(), expected.size() * sizeof(BFloat)) != 0) {
            throw std::runtime_error("Tiled prefill scale mismatch at exponent " +
                                     std::to_string(exponent));
        }
    }
    std::puts("Tiled prefill scales match across a remapped page boundary");
}

void verify_ordered_prefill(const char* phase, sglang::Mxfp8KvCache cache,
                            sglang::Mxfp8PageTable pages, int first, int rows,
                            const BFloat* queries,
                            const std::vector<BFloat>& actual, cudaStream_t stream) {
    const DeviceBuffer<BFloat> ordered(rows * query_elements);
    check_cuda(sglang::decoder::tensor_core_prefill_gqa(
                   cache, pages, first, rows, queries, ordered.get(), stream), phase);
    const auto expected = ordered.download(stream);
    const auto different = std::mismatch(actual.begin(), actual.end(), expected.begin(),
        [](BFloat left, BFloat right) { return std::memcmp(&left, &right, sizeof(BFloat)) == 0; });
    if (different.first != actual.end()) {
        const auto index = different.first - actual.begin();
        std::fprintf(stderr, "%s row=%td head=%td channel=%td got %.9g ordered %.9g\n",
                     phase, index / query_elements, index / head_width % query_heads,
                     index % head_width, __bfloat162float(*different.first),
                     __bfloat162float(*different.second));
        throw std::runtime_error("Multirow prefill differs bitwise from ordered attention");
    }
}

void check_raw_multirow_prefill(cudaStream_t stream) {
    check_ordered_prefill_extent(stream);
    check_ordered_prefill_precision(stream);
    check_ordered_prefill_score_overflow(stream);
}

}
