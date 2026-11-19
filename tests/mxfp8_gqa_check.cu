#include "sglang/decoder_cuda.cuh"
#include "sglang/mxfp8_gqa.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace gqa_test {

std::vector<__nv_bfloat16> make_queries(int start_token, int count);
double decode_value(std::uint8_t code, std::uint8_t scale);
void check_bound_gdn(cudaStream_t stream);
void check_tiled_prefill_scales(cudaStream_t stream);
void check_raw_attention(cudaStream_t stream);
void verify_ordered_prefill(const char* phase, sglang::Mxfp8KvCache cache,
                            sglang::Mxfp8PageTable pages, int first, int rows,
                            const __nv_bfloat16* queries, const std::vector<__nv_bfloat16>& actual,
                            cudaStream_t stream);

}

namespace {

constexpr int head_width = 256;
constexpr int kv_heads = 4;
constexpr int query_heads = 24;
constexpr int group_width = 32;
constexpr int groups = head_width / group_width;
constexpr int page_tokens = 128;
constexpr int physical_pages = 18;
constexpr int prefix_tokens = page_tokens - 1;
constexpr int prefill_tokens = 3;
constexpr int prefill_end = page_tokens + 1;
constexpr int decode_end = prefill_end + 1;
constexpr double softmax_scale = 1.0 / 16.0;
constexpr double scale_amax_floor = 1e-30;
constexpr std::size_t kv_elements_per_token = kv_heads * head_width;
constexpr std::size_t payload_bytes = physical_pages * page_tokens * kv_elements_per_token;
constexpr std::size_t scale_bytes = physical_pages * kv_heads * page_tokens * groups;

using BFloat = __nv_bfloat16;
using Byte = std::uint8_t;
using gqa_test::decode_value;
using gqa_test::make_queries;

enum class KvKind { key, value };

std::size_t row_index(int token, int head, int heads) {
    return static_cast<std::size_t>(token) * heads + head;
}

std::size_t scale_index(int page, int head, int slot, int group) {
    const auto page_head = row_index(page, head, kv_heads);
    return ((page_head * group_width + slot % group_width) *
            (page_tokens / group_width) + slot / group_width) * groups + group;
}

double positive_e4m3(int code) {
    const int exponent = code / 8;
    const int fraction = code % 8;
    if (exponent == 0) {
        return std::ldexp(fraction, -9);
    }
    return std::ldexp(8 + fraction, exponent - 10);
}

Byte encode_e4m3(double value) {
    const double magnitude = std::abs(value);
    int lower = 0;
    int upper = 0x7e;
    while (upper - lower > 1) {
        const int middle = (lower + upper) / 2;
        if (positive_e4m3(middle) < magnitude) {
            lower = middle;
        } else {
            upper = middle;
        }
    }
    const double midpoint = (positive_e4m3(lower) + positive_e4m3(upper)) * 0.5;
    const int code = magnitude < midpoint || (magnitude == midpoint && lower % 2 == 0)
                         ? lower : upper;
    return static_cast<Byte>(code | (std::signbit(value) ? 0x80 : 0));
}

Byte encode_e8m0(double amax) {
    const double target = std::max(amax, scale_amax_floor);
    int lower = 0;
    int upper = 254;
    while (lower < upper) {
        const int middle = (lower + upper) / 2;
        if (std::ldexp(448.0, middle - 127) < target) {
            lower = middle + 1;
        } else {
            upper = middle;
        }
    }
    return static_cast<Byte>(lower);
}

}

double gqa_test::decode_value(Byte code, Byte scale) {
    const double magnitude = positive_e4m3(code & 0x7f);
    const double value = (code & 0x80) != 0 ? -magnitude : magnitude;
    return std::ldexp(value, static_cast<int>(scale) - 127);
}

namespace {

struct Encoded {
    std::vector<Byte> data;
    std::vector<Byte> scales;
    std::vector<double> decoded;
};

Encoded encode_rows(const std::vector<BFloat>& input) {
    const auto scale_count = input.size() / group_width;
    Encoded result{std::vector<Byte>(input.size()), std::vector<Byte>(scale_count),
                   std::vector<double>(input.size())};
    for (std::size_t group = 0; group < scale_count; ++group) {
        const auto start = group * group_width;
        double amax = 0;
        for (int lane = 0; lane < group_width; ++lane) {
            amax = std::max(amax, std::abs(static_cast<double>(
                                     __bfloat162float(input[start + lane]))));
        }
        const Byte scale = encode_e8m0(amax);
        result.scales[group] = scale;
        for (int lane = 0; lane < group_width; ++lane) {
            const auto offset = start + lane;
            const double value = __bfloat162float(input[offset]);
            const Byte code = encode_e4m3(std::ldexp(value, 127 - static_cast<int>(scale)));
            result.data[offset] = code;
            result.decoded[offset] = decode_value(code, scale);
        }
    }
    return result;
}

double kv_sample(KvKind kind, int token, int head, int group, int lane, int generation) {
    const int plane = kind == KvKind::key ? 0 : 1;
    if ((token + 3 * head + 5 * group + 7 * plane + generation) % 37 == 0) {
        return 0;
    }
    const int pattern = (token + head + group + plane + generation) % 3;
    const int peak = 446 + 2 * pattern;
    const int exponent = -9 + (token * 3 + head * 5 + group * 7 + plane + generation) % 4;
    const int scale_exponent = exponent + (peak > 448 ? 1 : 0);
    const int sign = (token + head + lane / 4 + plane + generation) % 2 == 0 ? 1 : -1;
    constexpr double ties[] = {0, 1.0625, 1.1875, -1.0625, -1.1875, 1.0 / 1024,
                               3.0 / 1024, -3.0 / 1024, -1.0 / 1024, -0.0};
    if (lane < static_cast<int>(sizeof(ties) / sizeof(ties[0]))) {
        return std::ldexp(ties[lane], scale_exponent);
    }
    if (lane == group_width - 1) {
        return std::ldexp(sign * peak, exponent);
    }
    const int amplitude = 16 + (token * 17 + head * 13 + group * 11 + lane * 7 +
                                plane * 19 + generation * 23) % 80;
    return std::ldexp(sign * amplitude, scale_exponent);
}

std::vector<BFloat> make_kv(KvKind kind, int tokens, int generation) {
    std::vector<BFloat> result(static_cast<std::size_t>(tokens) * kv_heads * head_width);
    for (int token = 0; token < tokens; ++token) {
        for (int head = 0; head < kv_heads; ++head) {
            const auto start = row_index(token, head, kv_heads) * head_width;
            for (int channel = 0; channel < head_width; ++channel) {
                const double value = kv_sample(kind, token, head, channel / group_width,
                                               channel % group_width, generation);
                result[start + channel] = __float2bfloat16_rn(static_cast<float>(value));
            }
        }
    }
    return result;
}

}

std::vector<BFloat> gqa_test::make_queries(int start_token, int count) {
    std::vector<BFloat> result(static_cast<std::size_t>(count) * query_heads * head_width);
    for (int token = 0; token < count; ++token) {
        for (int head = 0; head < query_heads; ++head) {
            const auto start = row_index(token, head, query_heads) * head_width;
            for (int channel = 0; channel < head_width; ++channel) {
                const int position = start_token + token;
                const int pattern = ((position + channel + 5) * (head + 3) + channel / 7) % 29;
                result[start + channel] = __float2bfloat16_rn((pattern - 14) / 16.0f);
            }
        }
    }
    return result;
}

namespace {

std::vector<double> reference_attention(const std::vector<BFloat>& queries,
                                        const Encoded& keys, const Encoded& values,
                                        int start_token, int count) {
    std::vector<double> result(queries.size());
    std::vector<double> scores(start_token + count);
    for (int query = 0; query < count; ++query) {
        const int visible = start_token + query + 1;
        for (int head = 0; head < query_heads; ++head) {
            const int kv_head = head / (query_heads / kv_heads);
            const auto query_row = row_index(query, head, query_heads) * head_width;
            for (int token = 0; token < visible; ++token) {
                const auto kv_row = row_index(token, kv_head, kv_heads) * head_width;
                double score = 0;
                for (int channel = 0; channel < head_width; ++channel) {
                    score += __bfloat162float(queries[query_row + channel]) *
                             keys.decoded[kv_row + channel];
                }
                scores[token] = score * softmax_scale;
            }
            const double maximum = *std::max_element(scores.begin(), scores.begin() + visible);
            double denominator = 0;
            for (int token = 0; token < visible; ++token) {
                scores[token] = std::exp(scores[token] - maximum);
                denominator += scores[token];
            }
            for (int token = 0; token < visible; ++token) {
                const auto kv_row = row_index(token, kv_head, kv_heads) * head_width;
                const double weight = scores[token] / denominator;
                for (int channel = 0; channel < head_width; ++channel) {
                    result[query_row + channel] += weight * values.decoded[kv_row + channel];
                }
            }
        }
    }
    return result;
}

double output_tolerance(double expected) {
    const double magnitude = std::abs(expected);
    const int exponent = magnitude < std::numeric_limits<float>::min()
                             ? -126 : std::ilogb(magnitude);
    const double half_bfloat_ulp = std::ldexp(1.0, exponent - 8);
    constexpr double fp32_accumulation_allowance = 2e-5;
    return half_bfloat_ulp + fp32_accumulation_allowance * std::max(1.0, magnitude);
}

void check_reference_codes() {
    if (encode_e8m0(0) != 19 || encode_e8m0(224) != 126 || encode_e8m0(446) != 127 ||
        encode_e8m0(448) != 127 || encode_e8m0(450) != 128) {
        throw std::runtime_error("Independent scale encoder failed a known-answer case");
    }
    if (encode_e4m3(1.0625) != 0x38 || encode_e4m3(1.1875) != 0x3a ||
        encode_e4m3(-1.1875) != 0xba || encode_e4m3(1.0 / 1024) != 0 ||
        encode_e4m3(3.0 / 1024) != 2 || encode_e4m3(-1.0 / 1024) != 0x80 ||
        encode_e4m3(448) != 0x7e || decode_value(0xba, 128) != -2.5) {
        throw std::runtime_error("Independent payload encoder failed a known-answer case");
    }
}

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + " failed with " +
                                 cudaGetErrorString(status));
    }
}

class Stream {
public:
    Stream() {
        check_cuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
                   "Create test stream");
    }
    ~Stream() { cudaStreamDestroy(stream_); }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;

    cudaStream_t get() const { return stream_; }

private:
    cudaStream_t stream_ = nullptr;
};

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

struct KvSequence {
    KvSequence(int tokens, int generation)
        : KvSequence(make_kv(KvKind::key, tokens, generation),
                     make_kv(KvKind::value, tokens, generation)) {}

    KvSequence(std::vector<BFloat> input_keys, std::vector<BFloat> input_values);

    const std::vector<BFloat> keys;
    const std::vector<BFloat> values;
    const Encoded encoded_keys;
    const Encoded encoded_values;
};

void check_bytes(const char* phase, const char* plane, const char* buffer,
                 const std::vector<Byte>& actual, const std::vector<Byte>& expected) {
    for (std::size_t offset = 0; offset < expected.size(); ++offset) {
        if (actual[offset] != expected[offset]) {
            std::fprintf(stderr, "%s %s %s byte %zu got 0x%02x expected 0x%02x\n",
                         phase, plane, buffer, offset, static_cast<unsigned>(actual[offset]),
                         static_cast<unsigned>(expected[offset]));
            throw std::runtime_error("Cache bytes differ from independent encoding");
        }
    }
}

class CachePlane {
public:
    explicit CachePlane(const char* name) : name_(name) {}
    Byte* data() const { return data_.get(); }
    Byte* scales() const { return scales_.get(); }
    void initialize(cudaStream_t stream) {
        std::fill(expected_data_.begin(), expected_data_.end(), payload_sentinel);
        std::fill(expected_scales_.begin(), expected_scales_.end(), scale_sentinel);
        data_.fill(payload_sentinel, stream);
        scales_.fill(scale_sentinel, stream);
    }

    void record_append(const Encoded& encoded, const Pages& pages, int first, int count) {
        for (int position = first; position < first + count; ++position) {
            const int page = pages.mapping[position / page_tokens];
            const int slot = position % page_tokens;
            const int physical_token = page * page_tokens + slot;
            for (int head = 0; head < kv_heads; ++head) {
                const auto logical_row = row_index(position, head, kv_heads);
                const auto physical_row = row_index(physical_token, head, kv_heads);
                std::copy_n(encoded.data.begin() + logical_row * head_width, head_width,
                            expected_data_.begin() + physical_row * head_width);
                for (int group = 0; group < groups; ++group) {
                    expected_scales_[scale_index(page, head, slot, group)] =
                        encoded.scales[logical_row * groups + group];
                }
            }
        }
    }

    void verify(const char* phase, cudaStream_t stream) const {
        check_bytes(phase, name_, "payload", data_.download(stream), expected_data_);
        check_bytes(phase, name_, "scale", scales_.download(stream), expected_scales_);
    }

private:
    static constexpr Byte payload_sentinel = 0xa5;
    static constexpr Byte scale_sentinel = 0xf0;
    const char* name_;
    DeviceBuffer<Byte> data_{payload_bytes};
    DeviceBuffer<Byte> scales_{scale_bytes};
    std::vector<Byte> expected_data_ = std::vector<Byte>(payload_bytes, payload_sentinel);
    std::vector<Byte> expected_scales_ = std::vector<Byte>(scale_bytes, scale_sentinel);
};

struct TestCache {
    CachePlane keys{"key"};
    CachePlane values{"value"};

    sglang::Mxfp8KvCache view() const {
        return {keys.data(), values.data(), keys.scales(), values.scales(), physical_pages};
    }
};

void check_append(const char* phase, TestCache& cache, const Pages& pages, const KvSequence& input,
                  int first, int count, cudaStream_t stream) {
    const auto offset = first * kv_elements_per_token;
    const auto elements = count * kv_elements_per_token;
    const DeviceBuffer<BFloat> keys(elements);
    const DeviceBuffer<BFloat> values(elements);
    keys.upload(input.keys.data() + offset, stream);
    values.upload(input.values.data() + offset, stream);
    check_cuda(sglang::mxfp8_kv_append(cache.view(), pages.view(), first, count, keys.get(),
                                      values.get(), stream), phase);
    cache.keys.record_append(input.encoded_keys, pages, first, count);
    cache.values.record_append(input.encoded_values, pages, first, count);
    cache.keys.verify(phase, stream);
    cache.values.verify(phase, stream);
    std::printf("%s verified interval [%d,%d) across %zu cache bytes\n",
                phase, first, first + count, 2 * (payload_bytes + scale_bytes));
}

void verify_attention(const char* phase, const std::vector<BFloat>& actual,
                      const std::vector<double>& expected) {
    double max_error = 0;
    for (std::size_t index = 0; index < expected.size(); ++index) {
        const double got = __bfloat162float(actual[index]);
        const double error = std::abs(got - expected[index]);
        const double tolerance = output_tolerance(expected[index]);
        if (!std::isfinite(got) || error > tolerance) {
            std::fprintf(stderr, "%s row %zu head %zu channel %zu got %.9g expected %.9g "
                                 "allowed error %.9g\n",
                         phase, index / (head_width * query_heads),
                         index / head_width % query_heads, index % head_width,
                         got, expected[index], tolerance);
            throw std::runtime_error("Attention differs from independent causal reference");
        }
        max_error = std::max(max_error, error);
    }
    std::printf("%s verified %zu BF16 outputs with maximum absolute error %.9g\n",
                phase, expected.size(), max_error);
}

void check_attention(const char* phase, const TestCache& cache, const Pages& pages,
                     const KvSequence& input, int valid, int first, int count,
                     cudaStream_t stream) {
    const auto queries = make_queries(first, count);
    const auto expected = reference_attention(queries, input.encoded_keys, input.encoded_values,
                                              first, count);
    const DeviceBuffer<BFloat> device_queries(queries.size());
    const DeviceBuffer<BFloat> output(queries.size());
    device_queries.upload(queries.data(), stream);
    output.fill(0xff, stream);
    check_cuda(sglang::mxfp8_gqa_attention(cache.view(), pages.view(), valid, first, count,
                                         device_queries.get(), output.get(), stream), phase);
    verify_attention(phase, output.download(stream), expected);
}

constexpr int query_elements = query_heads * head_width;
constexpr int long_position = 2048;
constexpr int score_capacity = physical_pages * page_tokens;
constexpr Byte attention_sentinel = 0xff;

void compare_decode(const char* phase, int position, const std::vector<BFloat>& actual,
                    const std::vector<BFloat>& direct) {
    const auto bytes = reinterpret_cast<const Byte*>(actual.data());
    for (const auto guard : {bytes, bytes + (query_elements + 1) * sizeof(BFloat)}) {
        if (!std::all_of(guard, guard + sizeof(BFloat),
                         [](Byte byte) { return byte == attention_sentinel; })) {
            throw std::runtime_error("Decode overwrote an output guard");
        }
    }
    for (int index = 0; index < query_elements; ++index) {
        if (std::memcmp(&actual[index + 1], &direct[index], sizeof(BFloat)) != 0) {
            std::fprintf(stderr, "%s position=%d head=%d channel=%d got %.9g expected %.9g\n",
                         phase, position, index / head_width, index % head_width,
                         __bfloat162float(actual[index + 1]), __bfloat162float(direct[index]));
            throw std::runtime_error("Decode differs bitwise from direct attention");
        }
    }
}

void check_score_tail(const DeviceBuffer<float>& scores, int position, cudaStream_t stream) {
    const auto host = scores.download(stream);
    const auto bytes = reinterpret_cast<const Byte*>(host.data());
    const auto check = [](const Byte* first, const Byte* end) {
        if (!std::all_of(first, end, [](Byte byte) { return byte == attention_sentinel; })) {
            throw std::runtime_error("Decode overwrote an inactive score slot");
        }
    };
    check(bytes, bytes + sizeof(float));
    for (int head = 0; head < query_heads; ++head) {
        const int row = head * score_capacity + 1;
        check(bytes + (row + position + 1) * sizeof(float),
              bytes + (row + score_capacity) * sizeof(float));
    }
    const auto end = bytes + host.size() * sizeof(float);
    check(end - sizeof(float), end);
}

void check_exact_decode(cudaStream_t stream) {
    TestCache cache;
    cache.keys.initialize(stream);
    cache.values.initialize(stream);
    std::vector<std::int32_t> mapping(physical_pages, std::numeric_limits<int>::max());
    for (int page = 0; page <= long_position / page_tokens; ++page) {
        mapping[page] = physical_pages - 1 - page;
    }
    const Pages pages{std::move(mapping)};
    pages.device.upload(pages.mapping.data(), stream);
    const KvSequence input(long_position + 1, 13);
    check_append("exact decode cache append", cache, pages, input, 0, long_position + 1, stream);

    const auto queries = make_queries(long_position, 1);
    const DeviceBuffer<BFloat> device_queries(query_elements);
    const DeviceBuffer<BFloat> direct(query_elements);
    const DeviceBuffer<BFloat> output(query_elements + 2);
    const DeviceBuffer<float> scores(query_heads * score_capacity + 2);
    const DeviceBuffer<int> position(1);
    const sglang::decoder::ExactGqaWorkspace workspace{scores.get() + 1, score_capacity};
    device_queries.upload(queries.data(), stream);
    const auto view = cache.view();
    const auto table = pages.view();
    const auto invalid = [](cudaError_t result) {
        if (result != cudaErrorInvalidValue) {
            throw std::runtime_error("Exact decode accepted invalid arguments");
        }
    };
    const auto eager = [&](int last, sglang::decoder::ExactGqaWorkspace scratch) {
        return sglang::decoder::exact_decode_gqa_attention(view, table, last,
            device_queries.get(), output.get() + 1, scratch, stream);
    };
    output.fill(attention_sentinel, stream);
    scores.fill(attention_sentinel, stream);
    invalid(eager(-1, workspace));
    invalid(eager(score_capacity, workspace));
    invalid(eager(0, {nullptr, score_capacity}));
    invalid(eager(0, {scores.get() + 1, 0}));
    invalid(eager(0, {scores.get() + 1, score_capacity + 1}));
    invalid(sglang::decoder::exact_decode_gqa_attention(view, table,
        static_cast<const int*>(nullptr), device_queries.get(), output.get() + 1,
        workspace, stream));
    const auto untouched = output.download(stream);
    const auto first = reinterpret_cast<const Byte*>(untouched.data());
    if (!std::all_of(first, first + untouched.size() * sizeof(BFloat),
                     [](Byte byte) { return byte == attention_sentinel; })) {
        throw std::runtime_error("Invalid decode overwrote output");
    }
    check_score_tail(scores, -1, stream);

    Graph graph;
    graph.capture(stream, [&] {
        return sglang::decoder::exact_decode_gqa_attention(view, table, position.get(),
            device_queries.get(), output.get() + 1, workspace, stream);
    });

    for (const int last : {0, 31, 32, 127, 128, long_position, 1024, 1}) {
        check_cuda(sglang::mxfp8_gqa_attention(view, table, last + 1, last, 1,
            device_queries.get(), direct.get(), stream), "Launch direct decode oracle");
        const auto expected = direct.download(stream);
        output.fill(attention_sentinel, stream);
        scores.fill(attention_sentinel, stream);
        check_cuda(eager(last, workspace), "Launch eager decode");
        compare_decode("eager decode", last, output.download(stream), expected);
        check_score_tail(scores, last, stream);
        output.fill(attention_sentinel, stream);
        scores.fill(attention_sentinel, stream);
        check_cuda(sglang::decoder::set_decode_position(position.get(), last, stream),
                   "Set device decode position");
        graph.replay(stream);
        compare_decode("graph decode", last, output.download(stream), expected);
        check_score_tail(scores, last, stream);
        if (position.download(stream).front() != last) {
            throw std::runtime_error("Attention changed the graph position");
        }
    }
    const auto actual_queries = device_queries.download(stream);
    if (std::memcmp(actual_queries.data(), queries.data(), query_elements * sizeof(BFloat)) != 0 ||
        pages.device.download(stream) != pages.mapping) {
        throw std::runtime_error("Decode changed its queries or page table");
    }
    cache.keys.verify("exact decode preserves cache", stream);
    cache.values.verify("exact decode preserves cache", stream);
    std::puts("Exact BF16 decode matched eager and repeated graph positions through 2048");
}

void check_device() {
    int device = 0;
    cudaDeviceProp properties{};
    check_cuda(cudaGetDevice(&device), "Select current CUDA device");
    check_cuda(cudaGetDeviceProperties(&properties, device), "Read CUDA device properties");
    if (properties.major != 10 || properties.minor != 0) {
        throw std::runtime_error("The MXFP8 GQA check requires an SM100 device");
    }
    std::printf("device=%s compute_capability=%d.%d\n", properties.name,
                properties.major, properties.minor);
}

void run_check(const Stream& stream) {
    check_reference_codes();
    check_device();
    TestCache cache;
    cache.keys.initialize(stream.get());
    cache.values.initialize(stream.get());

    const Pages pages{{3, 1}};
    pages.device.upload(pages.mapping.data(), stream.get());
    const KvSequence original(decode_end, 0);
    check_append("prefix append", cache, pages, original, 0, prefix_tokens, stream.get());
    check_append("cross-page append", cache, pages, original, prefix_tokens,
                 prefill_end - prefix_tokens, stream.get());
    check_attention("causal prefill", cache, pages, original, prefill_end,
                    prefill_end - prefill_tokens, prefill_tokens, stream.get());
    check_append("decode append", cache, pages, original, prefill_end, 1, stream.get());
    check_attention("decode", cache, pages, original, decode_end, prefill_end, 1, stream.get());

    const Pages reused{{1}};
    reused.device.upload(reused.mapping.data(), stream.get());
    const KvSequence replacement(1, 9);
    check_append("reused page append", cache, reused, replacement, 0, 1, stream.get());
    check_attention("reused page decode", cache, reused, replacement, 1, 0, 1, stream.get());
    check_exact_decode(stream.get());
}

void check_bound_cache(cudaStream_t stream) {
    const DeviceBuffer<sglang::Mxfp8KvCache> binding(1);
    const DeviceBuffer<sglang::Mxfp8PageTable> table(1);
    const DeviceBuffer<int> position(1);
    const DeviceBuffer<BFloat> keys(kv_elements_per_token), values(kv_elements_per_token);
    const DeviceBuffer<BFloat> queries(query_elements), output(query_elements + 2);
    const DeviceBuffer<BFloat> direct(query_elements);
    const DeviceBuffer<float> scores(query_heads * score_capacity + 2);
    const sglang::decoder::ExactGqaWorkspace workspace{scores.get() + 1, score_capacity};
    TestCache first, second;
    const Pages forward{{3, 1}}, reverse{{6, 2}};
    forward.device.upload(forward.mapping.data(), stream);
    reverse.device.upload(reverse.mapping.data(), stream);
    Graph graph;
    graph.capture(stream, [&] {
        const auto append = sglang::decoder::decode_kv_append(
            binding.get(), table.get(), position.get(), keys.get(), values.get(), stream);
        if (append != cudaSuccess) {
            return append;
        }
        return sglang::decoder::exact_decode_gqa_attention(
            binding.get(), table.get(), position.get(), queries.get(), output.get() + 1,
            workspace, stream);
    });
    for (int trial = 0; trial < 4; ++trial) {
        const int last = trial < 2 ? page_tokens - 1 + trial : 1 + trial;
        auto& cache = trial % 2 ? second : first;
        const auto& pages = trial % 2 ? reverse : forward;
        auto& inactive = trial % 2 ? first : second;
        for (auto* storage : {&first, &second}) {
            storage->keys.initialize(stream);
            storage->values.initialize(stream);
        }
        const KvSequence input(last + 1, 17 + trial);
        TestCache oracle;
        oracle.keys.initialize(stream);
        oracle.values.initialize(stream);
        check_append("bound independent prefix", cache, pages, input, 0, last, stream);
        check_append("bound eager oracle", oracle, pages, input, 0, last + 1, stream);
        const auto cache_view = cache.view();
        const auto pages_view = pages.view();
        const auto query = make_queries(last, 1);
        binding.upload(&cache_view, stream);
        table.upload(&pages_view, stream);
        position.upload(&last, stream);
        keys.upload(input.keys.data() + last * kv_elements_per_token, stream);
        values.upload(input.values.data() + last * kv_elements_per_token, stream);
        queries.upload(query.data(), stream);
        output.fill(attention_sentinel, stream);
        scores.fill(attention_sentinel, stream);
        graph.replay(stream);
        check_cuda(sglang::mxfp8_gqa_attention(oracle.view(), pages_view, last + 1, last, 1,
            queries.get(), direct.get(), stream), "Launch bound eager oracle");
        compare_decode("rebound cache", last, output.download(stream), direct.download(stream));
        check_score_tail(scores, last, stream);
        cache.keys.record_append(input.encoded_keys, pages, last, 1);
        cache.values.record_append(input.encoded_values, pages, last, 1);
        cache.keys.verify("rebound keys", stream);
        cache.values.verify("rebound values", stream);
        inactive.keys.verify("inactive keys", stream);
        inactive.values.verify("inactive values", stream);
    }
    std::puts("Rebound cache, page table and position on four device graph replays");
}

template <class Element>
void require_same(const std::vector<Element>& actual, const std::vector<Element>& expected,
                  const char* message) {
    if (actual.size() != expected.size() ||
        std::memcmp(actual.data(), expected.data(), expected.size() * sizeof(Element)) != 0) {
        throw std::runtime_error(message);
    }
}

constexpr int finite_range_tokens = 1536;
constexpr int finite_range_page = 1;

struct FiniteRangeCase {
    const char* name;
    float query;
    float key;
    float value;
};

std::vector<double> append_finite_range_kv(const FiniteRangeCase& test, TestCache& cache,
                                           const Pages& pages,
                                           const std::vector<BFloat>& queries,
                                           cudaStream_t stream) {
    const auto elements = finite_range_tokens * kv_elements_per_token;
    const auto page_elements = page_tokens * kv_elements_per_token;
    std::vector<BFloat> keys(elements, __float2bfloat16_rn(0.0f));
    std::vector<BFloat> values(elements, __float2bfloat16_rn(1.0f));
    std::fill_n(keys.begin() + finite_range_page * page_elements, page_elements,
                __float2bfloat16_rn(test.key));
    std::fill_n(values.begin() + finite_range_page * page_elements, page_elements,
                __float2bfloat16_rn(test.value));
    const auto encoded_keys = encode_rows(keys);
    const auto encoded_values = encode_rows(values);
    const DeviceBuffer<BFloat> device_keys(elements);
    const DeviceBuffer<BFloat> device_values(elements);
    device_keys.upload(keys.data(), stream);
    device_values.upload(values.data(), stream);
    check_cuda(sglang::mxfp8_kv_append(cache.view(), pages.view(), 0, finite_range_tokens,
                                      device_keys.get(), device_values.get(), stream), test.name);
    cache.keys.record_append(encoded_keys, pages, 0, finite_range_tokens);
    cache.values.record_append(encoded_values, pages, 0, finite_range_tokens);
    cache.keys.verify(test.name, stream);
    cache.values.verify(test.name, stream);
    return reference_attention(queries, encoded_keys, encoded_values, finite_range_tokens - 1, 1);
}

bool check_finite_output(const char* name, const char* route, const DeviceBuffer<BFloat>& output,
                         cudaStream_t stream) {
    const auto actual = output.download(stream);
    const auto incorrect = std::count_if(actual.begin(), actual.end(), [](BFloat value) {
        return __bfloat162float(value) != 1.0f;
    });
    const auto nonfinite = std::count_if(actual.begin(), actual.end(), [](BFloat value) {
        return !std::isfinite(__bfloat162float(value));
    });
    std::printf("%s %s incorrect=%td/%d nonfinite=%td\n",
                name, route, incorrect, query_elements, nonfinite);
    return incorrect == 0;
}

void check_finite_range_decode(cudaStream_t stream) {
    constexpr std::size_t workspace_bytes = 496640;
    const FiniteRangeCase cases[] = {
        {"finite QK score overflow", -0x1p120f, 0x1p20f, 16.0f},
        {"finite local value overflow", 1.0f, -8.0f, 0x1p125f},
    };
    TestCache cache;
    cache.keys.initialize(stream);
    cache.values.initialize(stream);
    const Pages pages{{11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0}};
    pages.device.upload(pages.mapping.data(), stream);
    const DeviceBuffer<BFloat> queries(query_elements);
    const DeviceBuffer<BFloat> output(query_elements);
    const DeviceBuffer<float> scores(workspace_bytes / sizeof(float));
    const DeviceBuffer<int> position(1);
    const sglang::decoder::ExactGqaWorkspace workspace{scores.get(), finite_range_tokens};
    const auto submit = [&](auto last, std::size_t bytes) {
        return sglang::decoder::exact_decode_gqa_attention(cache.view(), pages.view(), last,
            queries.get(), output.get(), workspace, stream, bytes);
    };
    const auto require_page_partial = [&] {
        if (scores.download(stream).front() != static_cast<float>(page_tokens)) {
            throw std::runtime_error("Finite-range fixture did not produce a split page partial");
        }
    };
    Graph ordered, split;
    ordered.capture(stream, [&] { return submit(position.get(), 0); });
    split.capture(stream, [&] { return submit(position.get(), workspace_bytes); });
    constexpr int last = finite_range_tokens - 1;
    check_cuda(sglang::decoder::set_decode_position(position.get(), last, stream),
               "Set finite-range position");
    bool passed = true;
    for (const auto& test : cases) {
        const std::vector<BFloat> host_queries(query_elements, __float2bfloat16_rn(test.query));
        const auto expected = append_finite_range_kv(test, cache, pages, host_queries, stream);
        if (!std::all_of(expected.begin(), expected.end(), [](double value) {
            return __bfloat162float(__float2bfloat16_rn(static_cast<float>(value))) == 1.0f;
        })) {
            throw std::runtime_error("Finite-range FP64 reference did not round to BF16 one");
        }
        queries.upload(host_queries.data(), stream);
        output.fill(attention_sentinel, stream);
        check_cuda(submit(last, 0), "Launch ordered finite-range attention");
        passed &= check_finite_output(test.name, "ordered eager", output, stream);
        output.fill(attention_sentinel, stream);
        ordered.replay(stream);
        passed &= check_finite_output(test.name, "ordered graph", output, stream);
        output.fill(attention_sentinel, stream);
        scores.fill(attention_sentinel, stream);
        check_cuda(submit(last, workspace_bytes), "Launch split finite-range attention");
        require_page_partial();
        passed &= check_finite_output(test.name, "split eager", output, stream);
        output.fill(attention_sentinel, stream);
        scores.fill(attention_sentinel, stream);
        split.replay(stream);
        require_page_partial();
        passed &= check_finite_output(test.name, "split graph", output, stream);
    }
    if (!passed) {
        throw std::runtime_error("Finite-range attention differs from its FP64 reference");
    }
}

KvSequence::KvSequence(std::vector<BFloat> input_keys, std::vector<BFloat> input_values)
    : keys(std::move(input_keys)),
      values(std::move(input_values)),
      encoded_keys(encode_rows(keys)),
      encoded_values(encode_rows(values)) {}

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

void check_ordered_multirow_prefill(cudaStream_t stream) {
    constexpr int first = group_width - 1;
    constexpr int rows = page_tokens + 1;
    constexpr int tokens = first + rows;
    constexpr int changed_token = page_tokens;
    constexpr int changed_head = 1;
    constexpr int changed_group = 2;
    constexpr int causal_rows = changed_token - first;
    constexpr auto changed_offset = (changed_token * kv_heads + changed_head) * head_width
                                    + changed_group * group_width;
    const Pages pages{{7, 2}};
    pages.device.upload(pages.mapping.data(), stream);
    TestCache cache;
    cache.keys.initialize(stream);
    cache.values.initialize(stream);
    const KvSequence original_kv(tokens, 0);
    auto values = original_kv.values;
    const auto host_queries = make_queries(first, rows);
    const DeviceBuffer<BFloat> queries(host_queries.size());
    const DeviceBuffer<BFloat> output(host_queries.size() + 2 * prefill_guard);
    queries.upload(host_queries.data(), stream);
    const auto submit = [&] {
        return sglang::decoder::tensor_core_multirow_prefill_gqa(
            cache.view(), pages.view(), first, rows, queries.get(), output.get() + prefill_guard,
            stream);
    };
    const auto verify = [&](const char* phase, const KvSequence& input) {
        const auto reference = reference_attention(host_queries, input.encoded_keys,
                                                   input.encoded_values, first, rows);
        const auto actual = download_prefill(output, stream);
        verify_attention(phase, actual, reference);
        gqa_test::verify_ordered_prefill(phase, cache.view(), pages.view(), first, rows,
                                         queries.get(), actual, stream);
        cache.keys.verify(phase, stream);
        cache.values.verify(phase, stream);
        require_same(queries.download(stream), host_queries, "Multirow prefill modified queries");
        require_same(pages.device.download(stream), pages.mapping,
                     "Multirow prefill modified page mapping");
        return actual;
    };

    check_append("ordered multirow KV", cache, pages, original_kv, 0, tokens, stream);
    output.fill(attention_sentinel, stream);
    check_cuda(submit(), "Launch ordered multirow prefill");
    const auto original = verify("ordered multirow prefill", original_kv);

    Graph graph;
    graph.capture(stream, submit);
    std::fill_n(values.begin() + changed_offset, group_width, __float2bfloat16_rn(8.0f));
    const KvSequence future(original_kv.keys, std::move(values));
    check_append("ordered future KV", cache, pages, future, changed_token, 1, stream);
    output.fill(attention_sentinel, stream);
    graph.replay(stream);
    const auto perturbed = verify("ordered multirow prefill graph", future);
    const auto causal_bytes = causal_rows * query_elements * sizeof(BFloat);
    if (std::memcmp(original.data(), perturbed.data(), causal_bytes) != 0 ||
        std::memcmp(original.data() + causal_rows * query_elements,
                    perturbed.data() + causal_rows * query_elements,
                    (rows - causal_rows) * query_elements * sizeof(BFloat)) == 0) {
        throw std::runtime_error("Future KV perturbation failed the prefill causal boundary");
    }
}

void check_ordered_prefill_masked_score_overflow(cudaStream_t stream) {
    constexpr int rows = 65;
    const Pages pages{{5}};
    pages.device.upload(pages.mapping.data(), stream);
    TestCache cache;
    cache.keys.initialize(stream);
    cache.values.initialize(stream);
    std::vector<BFloat> keys(rows * kv_elements_per_token, __float2bfloat16_rn(0.0f));
    std::vector<BFloat> values(keys.size(), __float2bfloat16_rn(1.0f));
    const auto second_key = keys.begin() + kv_elements_per_token;
    std::fill_n(second_key, group_width, __float2bfloat16_rn(-16.0f));
    std::fill_n(second_key + group_width, 4 * group_width, __float2bfloat16_rn(4.0f));
    std::fill_n(values.begin() + kv_elements_per_token, head_width, __float2bfloat16_rn(3.0f));
    const KvSequence input(std::move(keys), std::move(values));
    check_append("masked score overflow KV", cache, pages, input, 0, rows, stream);

    const std::vector<BFloat> host_queries(rows * query_elements, __float2bfloat16_rn(0x1p120f));
    const DeviceBuffer<BFloat> queries(host_queries.size());
    const DeviceBuffer<BFloat> output(host_queries.size() + 2 * prefill_guard);
    queries.upload(host_queries.data(), stream);
    output.fill(attention_sentinel, stream);
    check_cuda(sglang::decoder::tensor_core_multirow_prefill_gqa(
                   cache.view(), pages.view(), 0, rows, queries.get(),
                   output.get() + prefill_guard, stream), "Launch masked score overflow prefill");
    const auto actual = download_prefill(output, stream);
    bool passed = true;
    for (const int row : {1, rows - 1}) {
        const auto begin = actual.begin() + row * query_elements;
        const float expected = __bfloat162float(__double2bfloat16(1.0 + 2.0 / (row + 1)));
        const auto different = std::find_if(begin, begin + head_width, [expected](BFloat value) {
            return __bfloat162float(value) != expected;
        });
        if (different != begin + head_width) {
            std::fprintf(stderr, "Masked score overflow row=%d head=0 channel=%td "
                         "got %.9g expected %.9g\n", row, different - begin,
                         __bfloat162float(*different), expected);
            passed = false;
        }
    }
    if (!passed) {
        throw std::runtime_error("Masked score overflow recovery mismatch");
    }
    std::puts("Masked score overflow recovers in paired rows and odd tail");
}

}

int main() {
    try {
        const Stream stream;
        run_check(stream);
        check_bound_cache(stream.get());
        gqa_test::check_bound_gdn(stream.get());
        gqa_test::check_tiled_prefill_scales(stream.get());
        check_ordered_multirow_prefill(stream.get());
        gqa_test::check_raw_attention(stream.get());
        check_finite_range_decode(stream.get());
        check_ordered_prefill_masked_score_overflow(stream.get());
        std::puts("MXFP8 GQA check passed");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "MXFP8 GQA check failed. %s\n", error.what());
        return 1;
    }
}
