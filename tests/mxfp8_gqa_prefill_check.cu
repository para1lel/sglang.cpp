#include "sglang/decoder_cuda.cuh"
#include "sglang/mxfp8_gqa.h"

#include <algorithm>
#include <array>
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

    unsigned check_grid(int visible, unsigned width) const {
        std::size_t count = 0;
        check_cuda(cudaGraphGetNodes(graph_, nullptr, &count), "Count attention graph nodes");
        if (count != 1) {
            throw std::runtime_error("Expected one attention graph kernel");
        }
        cudaGraphNode_t node = nullptr;
        check_cuda(cudaGraphGetNodes(graph_, &node, &count), "Get attention graph node");
        cudaKernelNodeParams kernel{};
        check_cuda(cudaGraphKernelNodeGetParams(node, &kernel), "Get attention graph grid");
        if (kernel.gridDim.x != width || kernel.gridDim.y != query_heads / 2) {
            throw std::runtime_error("Unexpected prefill dispatch grid at "
                + std::to_string(visible) + " visible tokens: "
                + std::to_string(kernel.gridDim.x) + "x" + std::to_string(kernel.gridDim.y)
                + ", expected " + std::to_string(width) + "x" + std::to_string(query_heads / 2));
        }
        return kernel.gridDim.x;
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

using sglang::decoder::BoundPagedDecodeLane;
using sglang::decoder::ExactGqaWorkspace;
using sglang::decoder::PagedDecodeLane;
using sglang::decoder::paged_decode_gqa;

constexpr int paged_lanes = 4;
constexpr int paged_layers = 2;
constexpr int paged_pages = 131;
constexpr int paged_groups = 8;
constexpr int score_stride = paged_pages * page_tokens;
constexpr int split_guard = 16;
constexpr auto split_bytes = sglang::decoder::paged_decode_scratch_bytes;

bool paged_position(int position) {
    return position >= 8191 && position < 16384;
}

bool aliased_scores(int position) {
    return position == 127 || position == 128;
}

struct PagedBinding {
    sglang::Mxfp8KvCache cache;
    sglang::Mxfp8PageTable pages;
    int position;
    ExactGqaWorkspace scores;
};

struct PackedRows {
    DeviceBuffer<Byte> payload{paged_pages * sglang::mxfp8_payload_bytes_per_page};
    DeviceBuffer<Byte> scales{paged_pages * sglang::mxfp8_scale_bytes_per_page};
    std::vector<Byte> expected_payload = std::vector<Byte>(
        paged_pages * sglang::mxfp8_payload_bytes_per_page, attention_sentinel);
    std::vector<Byte> expected_scales = std::vector<Byte>(
        paged_pages * sglang::mxfp8_scale_bytes_per_page, attention_sentinel);
    std::vector<float> logical;

    void load(const std::vector<std::int32_t>& pages, int position, int seed, bool values,
              cudaStream_t stream) {
        std::fill(expected_payload.begin(), expected_payload.end(), attention_sentinel);
        std::fill(expected_scales.begin(), expected_scales.end(), attention_sentinel);
        logical.resize((position + 1) * kv_elements_per_token);
        for (int token = 0; token <= position; ++token) {
            const int page = pages[token / page_tokens];
            const int slot = token % page_tokens;
            const bool appended = token == position;
            for (int head = 0; head < kv_heads; ++head) {
                const auto logical_row = row_index(token, head, kv_heads) * head_width;
                const auto payload_row = row_index(page * page_tokens + slot, head, kv_heads)
                                         * head_width;
                for (int group = 0; group < paged_groups; ++group) {
                    const auto exponent = static_cast<Byte>(appended ? (values ? 123 : 121) :
                        124 + (token * 7 + head * 3 + group + seed) % 4);
                    const auto scale = ((((page * kv_heads + head) * 32 + slot % 32) * 4
                                        + slot / 32) * paged_groups + group);
                    expected_scales[scale] = exponent;
                    for (int index = 0; index < 32; ++index) {
                        const int channel = group * 32 + index;
                        const int sample = (token * 13 + head * 31 + channel * 5
                                            + seed * (index + 1)) % 23;
                        const auto code = static_cast<Byte>(appended ? 0x78 :
                            (0x30 + sample % 8) | (sample % 2 ? 0x80 : 0));
                        expected_payload[payload_row + channel] = code;
                        logical[logical_row + channel] =
                            static_cast<float>(decode_value(code, exponent));
                    }
                }
            }
        }
        const auto last_page = pages[position / page_tokens];
        const auto last_slot = position % page_tokens;
        const auto last_payload = (last_page * page_tokens + last_slot) * kv_elements_per_token;
        auto initial_payload = expected_payload;
        auto initial_scales = expected_scales;
        std::fill_n(initial_payload.begin() + last_payload, kv_elements_per_token,
                    attention_sentinel);
        for (int head = 0; head < kv_heads; ++head) {
            const auto scale = (((last_page * kv_heads + head) * 32 + last_slot % 32) * 4
                                + last_slot / 32) * paged_groups;
            std::fill_n(initial_scales.begin() + scale, paged_groups, attention_sentinel);
        }
        payload.upload(initial_payload.data(), stream);
        scales.upload(initial_scales.data(), stream);
        check_cuda(cudaStreamSynchronize(stream), "Complete paged prefix upload");
    }
};

struct PagedBank {
    PackedRows keys, values;
    DeviceBuffer<std::int32_t> page_table{paged_pages};
    DeviceBuffer<float> scores{query_heads * score_stride + 2};
    std::vector<std::int32_t> mapping = std::vector<std::int32_t>(paged_pages);
    std::vector<BFloat> queries;

    PagedBinding binding(int position) const {
        const int padding = aliased_scores(position) ? 0 : paged_pages % (1 + position % 29);
        return {{keys.payload.get(), values.payload.get(), keys.scales.get(),
                 values.scales.get(), paged_pages}, {page_table.get(), paged_pages}, position,
                {scores.get() + 1, position + 1 + padding}};
    }

    void prepare(int position, int seed, cudaStream_t stream) {
        std::fill(mapping.begin(), mapping.end(), std::numeric_limits<int>::max());
        for (int page = 0; page <= position / page_tokens; ++page) {
            mapping[page] = (page * (3 + 2 * (seed % 9)) + seed) % paged_pages;
        }
        page_table.upload(mapping.data(), stream);
        keys.load(mapping, position, seed, false, stream);
        values.load(mapping, position, seed + 17, true, stream);
        queries = make_queries(position + seed * 5, 1);
        scores.fill(attention_sentinel, stream);
    }
};

struct PagedSlot {
    DeviceBuffer<PagedBinding> device_binding{1};
    DeviceBuffer<BFloat> keys{kv_elements_per_token}, values{kv_elements_per_token};
    DeviceBuffer<BFloat> queries{query_elements}, output{query_elements + 2};
    DeviceBuffer<Byte> split{split_bytes + 2 * split_guard};

    BoundPagedDecodeLane lane() const {
        const auto* binding = device_binding.get();
        return {&binding->cache, &binding->pages, &binding->position, &binding->scores,
                {queries.get(), output.get() + 1, split.get() + split_guard}};
    }

    cudaError_t append(cudaStream_t stream) const {
        const auto binding = lane();
        return sglang::decoder::decode_kv_append(
            binding.cache, binding.pages, binding.position, keys.get(), values.get(), stream);
    }

    void prepare(const PagedBank& bank, PagedBinding& binding, cudaStream_t stream) {
        if (paged_position(binding.position)) {
            binding.scores = {};
        } else if (aliased_scores(binding.position)) {
            binding.scores.scores = reinterpret_cast<float*>(split.get() + split_guard);
        }
        const std::vector<BFloat> append_keys(kv_elements_per_token, __float2bfloat16_rn(4.0f));
        const std::vector<BFloat> append_values(kv_elements_per_token, __float2bfloat16_rn(16.0f));
        keys.upload(append_keys.data(), stream);
        values.upload(append_values.data(), stream);
        queries.upload(bank.queries.data(), stream);
        device_binding.upload(&binding, stream);
        output.fill(attention_sentinel, stream);
        split.fill(attention_sentinel, stream);
        check_cuda(cudaStreamSynchronize(stream), "Complete paged lane binding");
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

template <class Element>
bool poisoned(const std::vector<Element>& data, std::size_t first, std::size_t end) {
    const auto* bytes = reinterpret_cast<const Byte*>(data.data());
    return std::all_of(bytes + first * sizeof(Element), bytes + end * sizeof(Element),
                       [](Byte byte) { return byte == attention_sentinel; });
}

std::vector<double> paged_reference(const PagedBank& bank, int position) {
    std::vector<double> expected(query_elements), scores(position + 1);
    for (int head = 0; head < query_heads; ++head) {
        const int kv_head = head / (query_heads / kv_heads);
        const auto query = bank.queries.data() + head * head_width;
        for (int token = 0; token <= position; ++token) {
            const auto key = bank.keys.logical.data()
                             + row_index(token, kv_head, kv_heads) * head_width;
            double dot = 0;
            for (int channel = 0; channel < head_width; ++channel) {
                dot += __bfloat162float(query[channel]) * static_cast<double>(key[channel]);
            }
            scores[token] = dot / 16.0;
        }
        const double maximum = *std::max_element(scores.begin(), scores.end());
        double denominator = 0;
        for (int token = 0; token <= position; ++token) {
            const auto value = bank.values.logical.data()
                               + row_index(token, kv_head, kv_heads) * head_width;
            const double weight = std::exp(scores[token] - maximum);
            denominator += weight;
            for (int channel = 0; channel < head_width; ++channel) {
                expected[head * head_width + channel] += weight * value[channel];
            }
        }
        for (int channel = 0; channel < head_width; ++channel) {
            expected[head * head_width + channel] /= denominator;
        }
    }
    return expected;
}

void verify_paged_fp64(const PagedBank& bank, int position, const std::vector<BFloat>& actual) {
    const auto expected = paged_reference(bank, position);
    for (int index = 0; index < query_elements; ++index) {
        const double magnitude = std::abs(expected[index]);
        const int exponent = magnitude < std::numeric_limits<float>::min()
                                 ? -126 : std::ilogb(magnitude);
        const double tolerance = std::ldexp(1.0, exponent - 8) + 2e-5 * std::max(1.0, magnitude);
        const double output = __bfloat162float(actual[index]);
        if (!std::isfinite(output) || std::abs(output - expected[index]) > tolerance) {
            std::fprintf(stderr, "Paged FP64 position=%d head=%d channel=%d got %.9g "
                                 "expected %.9g tolerance %.9g\n", position, index / head_width,
                         index % head_width, output, expected[index], tolerance);
            throw std::runtime_error("Paged decode differs from independent causal reference");
        }
    }
}

std::vector<BFloat> paged_output(const PagedSlot& slot, cudaStream_t stream) {
    const auto result = slot.output.download(stream);
    if (!poisoned(result, 0, 1) || !poisoned(result, result.size() - 1, result.size())) {
        throw std::runtime_error("Paged decode overwrote an output guard");
    }
    return {result.begin() + 1, result.end() - 1};
}

template <class Element>
void check_ordered_score_tail(const std::vector<Element>& storage, std::size_t first,
                              int position, int stride) {
    constexpr auto factor = sizeof(float) / sizeof(Element);
    for (int head = 0; head < query_heads; ++head) {
        const auto row = first + head * stride * factor;
        if (!poisoned(storage, row + (position + 1) * factor, row + stride * factor)) {
            throw std::runtime_error("Ordered decode overwrote an inactive score tail");
        }
    }
    if (!poisoned(storage, first + query_heads * stride * factor, storage.size())) {
        throw std::runtime_error("Ordered decode overwrote inactive scratch");
    }
}

void check_paged_scratch(const PagedSlot& slot, const PagedBank& bank, int position,
                         cudaStream_t stream) {
    const auto split = slot.split.download(stream);
    const auto scores = bank.scores.download(stream);
    const bool paged = paged_position(position);
    const bool alias = aliased_scores(position);
    const int stride = bank.binding(position).scores.capacity;
    if (!poisoned(split, 0, split_guard) ||
        !poisoned(split, split_guard + split_bytes, split.size()) ||
        (paged || alias) == poisoned(split, split_guard, split_guard + split_bytes)) {
        throw std::runtime_error("Paged split scratch writes disagree with position or guards");
    }
    if (paged || alias) {
        if (!poisoned(scores, 0, scores.size())) {
            throw std::runtime_error("Paged lane wrote inactive ordered scratch");
        }
    } else {
        if (!poisoned(scores, 0, 1) || poisoned(scores, 1, position + 2)) {
            throw std::runtime_error("Ordered lane failed to write only active scratch");
        }
        check_ordered_score_tail(scores, 1, position, stride);
    }
    if (alias) {
        check_ordered_score_tail(split, split_guard, position, stride);
    }
}

void check_paged_eager(const PagedSlot& slot, PagedBinding binding,
                       const std::vector<BFloat>& expected, cudaStream_t stream) {
    const DeviceBuffer<BFloat> output(query_elements);
    if (paged_position(binding.position)) {
        const DeviceBuffer<Byte> scratch(split_bytes + 2 * split_guard);
        scratch.fill(attention_sentinel, stream);
        const PagedDecodeLane lane{
            binding.cache, binding.pages, binding.position,
            {slot.queries.get(), output.get(), scratch.get() + split_guard}};
        auto invalid = lane;
        invalid.buffers.scratch = scratch.get() + split_guard + 1;
        if (paged_decode_gqa(&invalid, 1, stream) != cudaErrorInvalidValue) {
            throw std::runtime_error("Paged decode accepted misaligned scratch");
        }
        invalid.buffers.scratch = nullptr;
        if (paged_decode_gqa(&invalid, 1, stream) != cudaErrorInvalidValue) {
            throw std::runtime_error("Paged decode accepted missing scratch");
        }
        invalid = lane;
        invalid.pages.page_count = binding.position / page_tokens;
        if (paged_decode_gqa(&invalid, 1, stream) != cudaErrorInvalidValue) {
            throw std::runtime_error("Paged decode accepted insufficient page capacity");
        }
        check_cuda(paged_decode_gqa(&lane, 1, stream), "Launch paged eager");
        const auto bytes = scratch.download(stream);
        if (!poisoned(bytes, 0, split_guard) ||
            !poisoned(bytes, split_guard + split_bytes, bytes.size())) {
            throw std::runtime_error("Paged eager overwrote a scratch guard");
        }
    } else {
        check_cuda(sglang::decoder::exact_decode_gqa_attention(
            binding.cache, binding.pages, binding.position, slot.queries.get(), output.get(),
            binding.scores, stream), "Launch ordered eager fallback");
    }
    require_same(output.download(stream), expected, "Bound and eager decode differ bitwise");
}

void check_paged_replay(cudaStream_t stream) {
    std::array<PagedSlot, paged_lanes * paged_layers> slots;
    std::array<PagedBank, paged_lanes * paged_layers> banks;
    Graph graph;
    graph.capture(stream, [&] {
        for (int layer = 0; layer < paged_layers; ++layer) {
            std::array<BoundPagedDecodeLane, paged_lanes> lanes{};
            for (int lane = 0; lane < paged_lanes; ++lane) {
                const auto& slot = slots[layer * paged_lanes + lane];
                const auto status = slot.append(stream);
                if (status != cudaSuccess) {
                    return status;
                }
                lanes[lane] = slot.lane();
            }
            const auto status = paged_decode_gqa(lanes.data(), paged_lanes, stream);
            if (status != cudaSuccess) {
                return status;
            }
        }
        return cudaSuccess;
    });
    constexpr int positions[][paged_lanes] = {
        {8190, 8191, 16383, 16384}, {8192, 8190, 16384, 16383}, {8319, 8320, 127, 128},
    };
    for (int trial = 0; trial < 3; ++trial) {
        const auto bank_index = [trial](int index) {
            return (index + trial * 3) % (paged_lanes * paged_layers);
        };
        std::array<PagedBinding, paged_lanes * paged_layers> bindings{};
        for (int index = 0; index < paged_lanes * paged_layers; ++index) {
            const int position = positions[trial][index % paged_lanes];
            auto& bank = banks[bank_index(index)];
            bank.prepare(position, 9 + index + trial * 19, stream);
            bindings[index] = bank.binding(position);
            slots[index].prepare(bank, bindings[index], stream);
        }
        graph.replay(stream);
        for (int index = 0; index < paged_lanes * paged_layers; ++index) {
            const int position = positions[trial][index % paged_lanes];
            const auto& bank = banks[bank_index(index)];
            const auto& slot = slots[index];
            const auto result = paged_output(slot, stream);
            check_paged_scratch(slot, bank, position, stream);
            verify_paged_fp64(bank, position, result);
            check_paged_eager(slot, bank.binding(position), result, stream);
            for (const auto* plane : {&bank.keys, &bank.values}) {
                require_same(plane->payload.download(stream), plane->expected_payload,
                             "Bound decode changed a packed payload outside its append");
                require_same(plane->scales.download(stream), plane->expected_scales,
                             "Bound decode changed packed scales outside its append");
            }
            require_same(bank.page_table.download(stream), bank.mapping,
                         "Bound decode changed the page table");
            require_same(slot.queries.download(stream), bank.queries,
                         "Bound decode changed a query");
        }
    }
    std::puts("Paged decode matched FP64 and eager across 24 rebound layer-lanes");
}

void check_paged_escape_markers(const PagedSlot& slot, int first_head, int escaped_heads,
                                cudaStream_t stream) {
    const auto scratch = slot.split.download(stream);
    const auto head_bytes = split_bytes / sglang::mxfp8_query_heads;
    for (int head = 0; head < sglang::mxfp8_query_heads; ++head) {
        float maximum;
        const auto offset =
            split_guard + head * head_bytes + sglang::mxfp8_head_dim * sizeof(float);
        std::memcpy(&maximum, scratch.data() + offset, sizeof(maximum));
        const bool escaped = head >= first_head && head < first_head + escaped_heads;
        if ((escaped && !std::isnan(maximum)) || (!escaped && !std::isfinite(maximum))) {
            throw std::runtime_error("Paged escape did not stay within the affected query heads");
        }
    }
}

void check_paged_finite_range(cudaStream_t stream) {
    constexpr int position = 8191;
    constexpr auto elements = (position + 1) * kv_elements_per_token;
    constexpr int hostile_kv_head = 1;
    constexpr int heads_per_kv = sglang::mxfp8_query_heads / sglang::mxfp8_kv_heads;
    constexpr int first_hostile = hostile_kv_head * heads_per_kv;
    const PagedBank bank;
    const std::array<PagedSlot, paged_lanes> slots;
    std::vector<std::int32_t> mapping(paged_pages, std::numeric_limits<int>::max());
    for (int page = 0; page <= position / page_tokens; ++page) {
        mapping[page] = paged_pages - 1 - page;
    }
    bank.page_table.upload(mapping.data(), stream);
    const auto binding = bank.binding(position);
    auto active = binding;
    active.scores = {};
    std::array<BoundPagedDecodeLane, paged_lanes> lanes;
    for (int index = 0; index < paged_lanes; ++index) {
        slots[index].device_binding.upload(&active, stream);
        lanes[index] = slots[index].lane();
    }
    Graph graph;
    graph.capture(stream, [&] { return paged_decode_gqa(lanes.data(), paged_lanes, stream); });
    const DeviceBuffer<BFloat> input_keys(elements), input_values(elements);
    const DeviceBuffer<BFloat> ordered(query_elements);
    const auto bf16 = [](float value) { return __float2bfloat16_rn(value); };
    struct FiniteCase {
        const char* name;
        float query, key, value;
        bool cancellation;
    };
    for (const auto& test : {
             FiniteCase{"Paged score overflow recovery mismatch", -0x1p120f, 0x1p20f, 16.0f,
                        false},
             FiniteCase{"Paged numerator overflow recovery mismatch", 1.0f, -8.0f, 0x1p125f,
                        false},
             FiniteCase{"Paged value cancellation mismatch", 1.0f, 0.0f, 0x1p125f, true},
         }) {
        std::vector<BFloat> host_keys(elements, bf16(0.0f)), host_values(elements, bf16(2.0f));
        const auto head_offset = [](int token) {
            return token * kv_elements_per_token + hostile_kv_head * sglang::mxfp8_head_dim;
        };
        for (int token = 0; token <= position; ++token) {
            std::fill_n(host_values.begin() + head_offset(token), sglang::mxfp8_head_dim,
                        bf16(1.0f));
        }
        if (test.cancellation) {
            std::fill_n(host_values.begin() + head_offset(0), sglang::mxfp8_head_dim,
                        bf16(test.value));
            std::fill_n(host_values.begin() + head_offset(1), sglang::mxfp8_head_dim,
                        bf16(-test.value));
        } else {
            for (int token = 1024; token < 1536; ++token) {
                std::fill_n(host_keys.begin() + head_offset(token), sglang::mxfp8_head_dim,
                            bf16(test.key));
                std::fill_n(host_values.begin() + head_offset(token), sglang::mxfp8_head_dim,
                            bf16(test.value));
            }
        }
        input_keys.upload(host_keys.data(), stream);
        input_values.upload(host_values.data(), stream);
        check_cuda(sglang::mxfp8_kv_append(binding.cache, binding.pages, 0, position + 1,
            input_keys.get(), input_values.get(), stream), "Append paged finite-range fixture");
        const bool score_overflow = test.query < 0.0f;
        std::array<std::vector<BFloat>, paged_lanes> queries;
        for (int index = 0; index < paged_lanes; ++index) {
            const auto& slot = slots[index];
            queries[index].assign(query_elements, bf16(1.0f));
            if (score_overflow) {
                std::fill_n(queries[index].begin() + (first_hostile + index) * head_width,
                            head_width, bf16(test.query));
            }
            slot.queries.upload(queries[index].data(), stream);
            slot.output.fill(attention_sentinel, stream);
            slot.split.fill(attention_sentinel, stream);
        }
        graph.replay(stream);
        for (int index = 0; index < paged_lanes; ++index) {
            const auto& slot = slots[index];
            const int first = first_hostile + (score_overflow ? index : 0);
            const int count = score_overflow ? 1 : heads_per_kv;
            std::vector<BFloat> expected(query_elements, bf16(2.0f));
            std::fill_n(expected.begin() + first_hostile * head_width, heads_per_kv * head_width,
                        bf16(score_overflow ? test.value : 1.0f));
            std::fill_n(expected.begin() + first * head_width, count * head_width, bf16(1.0f));
            const auto actual = paged_output(slot, stream);
            require_same(actual, expected, test.name);
            check_paged_escape_markers(slot, first, count, stream);
            check_cuda(sglang::decoder::exact_decode_gqa_attention(
                binding.cache, binding.pages, position, slot.queries.get(), ordered.get(),
                binding.scores, stream), "Launch ordered finite-range control");
            require_same(ordered.download(stream), expected,
                         "Ordered finite-range control mismatch");
            check_paged_eager(slot, binding, actual, stream);
        }
    }
    std::puts("Paged decode recovered from finite-input overflow and value cancellation");
}

constexpr int prefill_guard = 8;

std::vector<BFloat> download_prefill(const DeviceBuffer<BFloat>& output, cudaStream_t stream,
                                     int guard_elements = prefill_guard) {
    const auto actual = output.download(stream);
    for (const auto* guard : {actual.data(), actual.data() + actual.size() - guard_elements}) {
        const auto* bytes = reinterpret_cast<const Byte*>(guard);
        if (!std::all_of(bytes, bytes + guard_elements * sizeof(BFloat),
                         [](Byte byte) { return byte == attention_sentinel; })) {
            throw std::runtime_error("Multirow prefill overwrote an output guard");
        }
    }
    return {actual.begin() + guard_elements, actual.end() - guard_elements};
}

void check_ordered_prefill_geometry(cudaStream_t stream) {
    using sglang::decoder::tensor_core_multirow_prefill_gqa;
    check_cuda(tensor_core_multirow_prefill_gqa({}, {}, 0, 0, nullptr, nullptr, stream),
               "Launch empty multirow prefill");
    constexpr int last = std::numeric_limits<int>::max();
    for (const auto [first, rows] : {std::pair{-1, 1}, {0, -1}, {last, 1}}) {
        if (tensor_core_multirow_prefill_gqa({}, {}, first, rows, nullptr, nullptr, stream)
            != cudaErrorInvalidValue) {
            throw std::runtime_error("Multirow prefill accepted invalid row geometry");
        }
    }
}

void poison_unused_values(Byte* values, int page_count, int used_tokens, cudaStream_t stream) {
    constexpr auto bytes_per_page = sglang::mxfp8_payload_bytes_per_page;
    constexpr auto bytes_per_token = bytes_per_page / page_tokens;
    const int last_physical = page_count - 1 - (used_tokens - 1) / page_tokens;
    const int used_in_page = (used_tokens - 1) % page_tokens + 1;
    check_cuda(cudaMemsetAsync(values, 0x50, last_physical * bytes_per_page, stream),
               "Poison unused reverse-mapped pages");
    auto* tail = values + last_physical * bytes_per_page + used_in_page * bytes_per_token;
    check_cuda(cudaMemsetAsync(tail, 0x50, (page_tokens - used_in_page) * bytes_per_token,
                               stream), "Poison unused final page tokens");
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
    constexpr int masked_rows = 135;
    for (const auto [first, count] : {
             std::pair{128, 1}, {31, 63}, {31, 64}, {31, 66}, {31, 127}, {31, 128},
             {31, 129}, {31, 130}, {limit - rows, rows}, {limit - rows + 1, rows},
             {page_tokens - 1, masked_rows}}) {
        if (count == masked_rows) {
            poison_unused_values(values.get(), page_count, first + count, stream);
        }
        const auto host_queries = make_queries(first, count);
        const DeviceBuffer<BFloat> queries(host_queries.size());
        const DeviceBuffer<BFloat> output(count * query_elements + 2 * query_elements);
        queries.upload(host_queries.data(), stream);
        output.fill(attention_sentinel, stream);
        check_cuda(sglang::decoder::tensor_core_multirow_prefill_gqa(
                       cache, pages.view(), first, count, queries.get(),
                       output.get() + query_elements, stream), "Launch prefill at dispatch edge");
        const auto actual = download_prefill(output, stream, query_elements);
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
    constexpr std::array cases{
        std::pair{65, 33U},    std::pair{32768, 17U}, std::pair{32769, 2U},
        std::pair{131072, 2U}, std::pair{131073, 2U},
    };
    constexpr int page_count = (cases.back().first + page_tokens - 1) / page_tokens;
    constexpr auto payload = page_count * sglang::mxfp8_payload_bytes_per_page;
    constexpr auto scales = page_count * sglang::mxfp8_scale_bytes_per_page;
    std::vector<std::int32_t> mapping(page_count);
    std::generate(mapping.rbegin(), mapping.rend(), [page = 0]() mutable { return page++; });
    const Pages pages{std::move(mapping)};
    pages.device.upload(pages.mapping.data(), stream);
    const DeviceBuffer<Byte> keys(payload), values(payload), unaligned_keys(payload + 1);
    const DeviceBuffer<Byte> key_scales(scales), value_scales(scales);
    const sglang::Mxfp8KvCache view{keys.get(), values.get(), key_scales.get(), value_scales.get(),
                                  page_count};
    const auto offset = pages.mapping.front() * sglang::mxfp8_payload_bytes_per_page;
    keys.fill(0, stream);
    key_scales.fill(127, stream);
    values.fill(0x38, stream);
    value_scales.fill(127, stream);
    check_cuda(cudaMemsetAsync(view.key_payload + offset, 0x58, head_width, stream),
               "Set leading overflow key");
    check_cuda(cudaMemsetAsync(view.value_payload + offset, 0x40, head_width, stream),
               "Set leading overflow value");
    auto control = view;
    control.key_payload = unaligned_keys.get() + 1;
    check_cuda(cudaMemcpyAsync(control.key_payload, keys.get(), payload,
                               cudaMemcpyDeviceToDevice, stream), "Copy unaligned overflow keys");
    std::vector<BFloat> input(rows * query_elements, __float2bfloat16_rn(0.0f));
    for (int row = 0; row < rows; ++row) {
        std::fill_n(&input[row * query_elements], head_width, __float2bfloat16_rn(0x1p126f));
    }
    const DeviceBuffer<BFloat> queries(input.size());
    queries.upload(input.data(), stream);
    const DeviceBuffer<BFloat> output(rows * query_elements + 2 * prefill_guard);
    for (const auto [visible, expected_width] : cases) {
        const int first = visible - rows;
        const auto launch = [&](sglang::Mxfp8KvCache cache) {
            return sglang::decoder::tensor_core_multirow_prefill_gqa(
                cache, pages.view(), first, rows, queries.get(), output.get() + prefill_guard,
                stream);
        };
        Graph graph;
        graph.capture(stream, [&] { return launch(view); });
        const auto width = graph.check_grid(visible, expected_width);
        output.fill(attention_sentinel, stream);
        graph.replay(stream);
        const auto actual = download_prefill(output, stream);
        output.fill(attention_sentinel, stream);
        check_cuda(launch(control), "Launch unaligned overflow control");
        require_same(actual, download_prefill(output, stream), "Protected overflow parity");
        for (int row = 0; row < rows; ++row) {
            const float mean = __bfloat162float(__double2bfloat16(1.0 + 1.0 / (first + row + 1)));
            for (int head = 0; head < query_heads; ++head) {
                const float ordinary = head < query_heads / kv_heads ? mean : 1.0f;
                const float expected = head == 0 ? 2.0f : ordinary;
                const auto begin = actual.begin() + row_index(row, head, query_heads) * head_width;
                if (!std::all_of(begin, begin + head_width, [expected](BFloat value) {
                    return __bfloat162float(value) == expected;
                })) {
                    throw std::runtime_error("Score overflow at first=" + std::to_string(first)
                        + ", row=" + std::to_string(row) + ", head=" + std::to_string(head));
                }
            }
        }
        std::printf("Prefill overflow visible=%d grid=%u matched unaligned and FP64 reference\n",
                    visible, width);
    }
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

void check_raw_attention(cudaStream_t stream) {
    check_ordered_prefill_geometry(stream);
    check_ordered_prefill_extent(stream);
    check_ordered_prefill_precision(stream);
    check_ordered_prefill_score_overflow(stream);
    check_paged_replay(stream);
    check_paged_finite_range(stream);
}
}
