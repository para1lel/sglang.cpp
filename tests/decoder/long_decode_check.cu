#include "sglang/decoder_batch.h"
#include "sglang/decoder_cuda.cuh"
#include "sglang/text/gqa_decode_split.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <dlfcn.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {

using namespace sglang;
using namespace sglang::decoder;
using Bf16 = __nv_bfloat16;
constexpr int heads = mxfp8_query_heads;
constexpr int dim = mxfp8_head_dim;
constexpr int page_size = mxfp8_page_size;
constexpr int max_tokens = 262144;
constexpr int pages = max_tokens / page_size;
constexpr int width = decoder_batch_max_lanes;
constexpr int row_size = heads * dim;
constexpr int guard = 32;
constexpr std::size_t scratch_stride = paged_decode_scratch_bytes + 2 * guard;
constexpr unsigned char poison = 0xa5;

enum class Kind { split, combine, ordered, other };
enum class Mode { full, small, recovery, unaligned };
struct Launch {
    Kind kind;
    dim3 grid;
    dim3 block;
};
std::array<Launch, 2 * width> launches;
int observed = 0;
int fail_at = 0;
bool watching = false;

void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void cuda_check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

template <typename Value>
struct Buffer {
    explicit Buffer(std::size_t count) { cuda_check(cudaMalloc(&data, count * sizeof(Value))); }
    ~Buffer() { cudaFree(data); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    Value* data = nullptr;
};

int page_map(int lane, int logical, int physical_pages) {
    return (logical * 197 + lane * 37 + 11) % physical_pages;
}

__host__ __device__ int key_sign(int page, int slot, int kv_head) {
    return ((page + 3 * slot + kv_head) % 5) < 2 ? -1 : 1;
}

__host__ __device__ int value_sign(int page, int slot, int kv_head, int channel) {
    return ((page * 3 + slot * 5 + kv_head * 7 + channel * 11) % 17) < 8 ? -1 : 1;
}

__host__ __device__ float query_value(int lane, int head) {
    return 4.0f + static_cast<float>((lane + head * 3) % 9) * 0.5f;
}

__global__ void fill_cache(Mxfp8KvCache kv, int populated, int marker) {
    const auto linear = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear >= static_cast<std::size_t>(populated) * mxfp8_payload_bytes_per_page) return;
    const int page = linear / mxfp8_payload_bytes_per_page;
    const int slot = (linear / (mxfp8_kv_heads * dim)) % page_size;
    const int kv_head = (linear / dim) % mxfp8_kv_heads;
    const int channel = linear % dim;
    const bool recovery = page == marker && slot == page_size - 1 && kv_head == 0;
    const bool value_overflow = page == marker && kv_head == 1;
    kv.key_payload[linear] = channel == 0
        ? (recovery || key_sign(page, slot, kv_head) > 0 ? 0x38 : 0xb8) : 0;
    kv.value_payload[linear] = (recovery || value_overflow) && channel < 32
        ? (channel == 0 ? 0x38 : 0) : (value_sign(page, slot, kv_head, channel) > 0 ? 0x38 : 0xb8);
    if (channel % mxfp8_group_size == 0) {
        const auto scale =
            (((page * mxfp8_kv_heads + kv_head) * 32 + slot % 32) * 4 + slot / 32) * 8
            + channel / 32;
        const bool oversized = recovery && channel == 0;
        kv.key_scales[scale] = oversized ? 254 : 128 + (page + slot + kv_head) % 4;
        kv.value_scales[scale] = oversized || (value_overflow && channel == 0)
            ? 254 : 124 + (page + slot + channel / 32) % 4;
    }
}

double reference(int lane, int tokens, int head, int channel, int marker, int physical_pages) {
    const int kv_head = head / (heads / mxfp8_kv_heads);
    const double query = query_value(lane, head);
    double maximum = -std::numeric_limits<double>::infinity();
    double mass = 0;
    double numerator = 0;
    for (int token = 0; token < tokens; ++token) {
        const int page = page_map(lane, token / page_size, physical_pages);
        const int slot = token % page_size;
        const bool repair = page == marker && slot == page_size - 1 && kv_head == 0;
        const double key = repair ? std::ldexp(1.0, 127)
            : std::ldexp(double(key_sign(page, slot, kv_head)), 1 + (page + slot + kv_head) % 4);
        const bool large_value = repair || (page == marker && kv_head == 1);
        const double value = large_value && channel < 32 ? std::ldexp(double(channel == 0), 127)
            : std::ldexp(double(value_sign(page, slot, kv_head, channel)),
                         -3 + (page + slot + channel / 32) % 4);
        const double score = query * key / 16.0;
        const double next = std::max(maximum, score);
        const double prior = std::exp(maximum - next);
        const double weight = std::exp(score - next);
        mass = mass * prior + weight;
        numerator = numerator * prior + weight * value;
        maximum = next;
    }
    return numerator / mass;
}

void expect_launch(const Launch& launch, Kind kind, dim3 grid, unsigned int threads) {
    check(launch.kind == kind && launch.grid.x == grid.x && launch.grid.y == grid.y &&
              launch.grid.z == grid.z && launch.block.x == threads &&
              launch.block.y == 1 && launch.block.z == 1,
          "Unexpected grouped kernel or geometry");
}

void expect_launches(int count) {
    constexpr int group_width = text_gdn_max_verify_rows;
    check(observed == 2 * ((count + group_width - 1) / group_width),
          "Expected one split and combine/replay pair per target group");
    int index = 0;
    for (int offset = 0; offset < count; offset += group_width) {
        const int rows = std::min(group_width, count - offset);
        expect_launch(launches[index++], Kind::split, dim3(16, mxfp8_kv_heads, rows), 128);
        expect_launch(launches[index++], Kind::combine, dim3(heads, 1, rows), dim);
    }
    std::cout << "launches=" << observed << " width=" << count << '\n';
}

cudaError_t invoke_eager(const PagedDecodeLane* inputs, int count, cudaStream_t stream,
                         bool narrow = false) {
    observed = 0;
    fail_at = 0;
    watching = true;
    const auto result = narrow ? paged_decode_gqa(inputs, count, stream)
                              : target_split_gqa(inputs, count, max_tokens, stream);
    watching = false;
    return result;
}

cudaError_t invoke(const PagedDecodeLane* inputs, int count, cudaStream_t stream, int fault = 0) {
    observed = 0;
    fail_at = fault;
    watching = true;
    cudaError_t result = cudaSuccess;
    for (int offset = 0; offset < count && result == cudaSuccess;
         offset += target_split_eager_max_rows) {
        const int rows = std::min(target_split_eager_max_rows, count - offset);
        result = target_split_gqa(inputs + offset, rows, max_tokens, stream);
    }
    watching = false;
    return result;
}

void invalid(const PagedDecodeLane* inputs, int count, cudaStream_t stream) {
    check(invoke_eager(inputs, count, stream) == cudaErrorInvalidValue && observed == 0,
          "Invalid grouped input must fail before launch");
}

void check_launch_failure(cudaError_t result, int fault) {
    if (result != cudaErrorInvalidValue || observed != fault) {
        std::cerr << "fault=" << fault << " launches=" << observed
                  << " status=" << cudaGetErrorName(result) << '\n';
        check(false, "CUDA launch error changed or a later pass was submitted");
    }
}

void check_guards(const std::vector<unsigned char>& bytes, std::size_t stride, int count) {
    for (int lane = 0; lane < count; ++lane) {
        const auto begin = bytes.begin() + lane * stride;
        check(std::all_of(begin, begin + guard, [](auto value) { return value == poison; }) &&
                  std::all_of(begin + stride - guard, begin + stride,
                              [](auto value) { return value == poison; }),
              "Output or scratch guard changed");
    }
}

void check_errors(std::vector<PagedDecodeLane> inputs, unsigned char* scratch,
                  cudaStream_t stream) {
    invalid(nullptr, 1, stream);
    invalid(inputs.data(), 0, stream);
    invalid(inputs.data(), target_split_eager_max_rows + 1, stream);
    const auto original = inputs[1];
    inputs[1].position = target_split_first_position - 1;
    invalid(inputs.data(), 2, stream);
    inputs[1].position = max_tokens;
    invalid(inputs.data(), 2, stream);
    inputs[1] = original;
    --inputs[1].pages.page_count;
    invalid(inputs.data(), 2, stream);
    inputs[1] = original;
    inputs[1].cache.key_scales = nullptr;
    invalid(inputs.data(), 2, stream);
    inputs[1] = original;
    inputs[1].buffers.scratch = scratch + 1;
    invalid(inputs.data(), 2, stream);
    inputs[1] = original;
    inputs[1].buffers.output = nullptr;
    invalid(inputs.data(), 2, stream);
    inputs[1] = original;
    inputs[1].buffers.queries = nullptr;
    invalid(inputs.data(), 2, stream);
    inputs[1] = original;
    for (const int count : {2, 5}) {
        for (int fault = 1; fault <= 2; ++fault) {
            check_launch_failure(invoke(inputs.data(), count, stream, fault), fault);
            cuda_check(cudaStreamSynchronize(stream));
        }
    }
    inputs[0].position = paged_decode_min_visible - 1;
    inputs[1].position = paged_decode_min_visible - 2;
    check(invoke_eager(inputs.data(), 2, stream, true) == cudaErrorInvalidValue && observed == 0,
          "Narrow window accepted a short context");
    inputs[1].position = paged_decode_max_visible;
    check(invoke_eager(inputs.data(), 2, stream, true) == cudaErrorInvalidValue && observed == 0,
          "Narrow window accepted a long context");
    check(invoke_eager(inputs.data(), paged_decode_lanes + 1, stream, true) ==
              cudaErrorInvalidValue && observed == 0, "Narrow window accepted a wide batch");
    std::cout << "invalid=13 injected_launch_failures=4\n";
}

constexpr std::size_t output_stride = row_size * sizeof(Bf16) + 2 * guard;

void check_ordered(const PagedDecodeLane& input, const unsigned char* output,
                   cudaStream_t stream) {
    Buffer<Bf16> ordered(row_size);
    cuda_check(mxfp8_gqa_attention(input.cache, input.pages, input.position + 1,
                                  input.position, 1, input.buffers.queries, ordered.data, stream));
    std::vector<Bf16> decoded(row_size);
    cuda_check(cudaMemcpyAsync(decoded.data(), ordered.data, row_size * sizeof(Bf16),
                               cudaMemcpyDeviceToHost, stream));
    cuda_check(cudaStreamSynchronize(stream));
    std::vector<Bf16> grouped(row_size);
    std::memcpy(grouped.data(), output, row_size * sizeof(Bf16));
    double max_difference = 0;
    for (int index = 0; index < row_size; ++index) {
        const double expected = __bfloat162float(decoded[index]);
        const double actual = __bfloat162float(grouped[index]);
        check(std::isfinite(expected), "Nonfinite public ordered reference");
        max_difference = std::max(max_difference, std::abs(expected - actual));
    }
    check(max_difference <= 0.004, "Public ordered attention differs beyond tolerance");
    std::cout << "ordered_max_visibility_abs=" << max_difference << '\n';
}

void check_outputs(const std::vector<PagedDecodeLane>& inputs,
                   const std::vector<unsigned char>& output, int marker, int physical_pages) {
    std::vector<Bf16> decoded(row_size);
    int samples = 0;
    double worst = 0;
    for (std::size_t lane = 0; lane < inputs.size(); ++lane) {
        std::memcpy(decoded.data(), output.data() + lane * output_stride + guard,
                    row_size * sizeof(Bf16));
        for (int index = 0; index < row_size; ++index) {
            const auto value = __bfloat162float(decoded[index]);
            const bool nan_head = marker >= 0 && lane == 1 && index / dim == 12;
            check(nan_head ? std::isnan(value) : std::isfinite(value),
                  "Unexpected nonfinite attention output");
        }
        for (const int head : {0, 5, 6, 11, 12, 17, 18, 23}) {
            if (marker >= 0 && lane == 1 && head == 12) continue;
            for (const int channel : {0, 31, 32, 203, 255}) {
                const auto expected = reference(lane, inputs[lane].position + 1, head,
                                                channel, marker, physical_pages);
                const auto actual = double(__bfloat162float(decoded[head * dim + channel]));
                const double error = std::abs(actual - expected) /
                    std::max(1.0, std::abs(expected));
                worst = std::max(worst, error);
                ++samples;
            }
        }
    }
    check(worst <= 0.004, "Grouped attention diverged from FP64");
    std::cout << "count=" << inputs.size() << " recovery=" << (marker >= 0) << " samples="
              << samples << " max_scaled_error=" << worst << " guards=pass\n";
}

void check_recovery(const std::vector<unsigned char>& scratch, int count, int marker,
                    int physical_pages) {
    int replayed = 0;
    for (int lane = 0; lane < count; ++lane) {
        bool has_marker = false;
        for (int page = 0; page < 32768 / page_size; ++page) {
            has_marker |= page_map(lane, page, physical_pages) == marker;
        }
        for (int head = 0; head < heads; ++head) {
            const auto offset = lane * scratch_stride + guard +
                head * gqa_decode::max_splits * sizeof(gqa_decode::Partial) +
                offsetof(gqa_decode::Partial, maximum);
            float maximum;
            std::memcpy(&maximum, scratch.data() + offset, sizeof(maximum));
            const bool expected = (has_marker && head < 12) || (lane == 1 && head == 12);
            check(expected ? std::isnan(maximum) : std::isfinite(maximum),
                  "Unexpected ordered replay mask");
            replayed += expected;
        }
    }
    std::cout << "replayed_heads=" << replayed << " checked_heads=" << count * heads << '\n';
}

void expect_poison(const unsigned char* bytes, std::size_t size, const char* message) {
    check(std::all_of(bytes, bytes + size, [](auto value) { return value == poison; }), message);
}

void collect_outputs(unsigned char* output, unsigned char* scratch,
                     std::vector<unsigned char>& host_output,
                     std::vector<unsigned char>& host_scratch, cudaStream_t stream) {
    cuda_check(cudaMemcpyAsync(host_output.data(), output, host_output.size(),
                               cudaMemcpyDeviceToHost, stream));
    cuda_check(cudaMemcpyAsync(host_scratch.data(), scratch, host_scratch.size(),
                               cudaMemcpyDeviceToHost, stream));
    cuda_check(cudaStreamSynchronize(stream));
}

void check_unaligned(std::vector<PagedDecodeLane> inputs, int physical_pages,
                     unsigned char* output, unsigned char* scratch, cudaStream_t stream) {
    const auto payload_bytes = physical_pages * mxfp8_payload_bytes_per_page;
    const auto scale_bytes = physical_pages * mxfp8_scale_bytes_per_page;
    Buffer<std::uint8_t> shifted(payload_bytes + 1);
    Buffer<Bf16> shifted_queries(row_size + 1);
    const int count = inputs.size();
    auto& lane = inputs.back();
    const auto original = lane;
    std::vector<unsigned char> host_output(count * output_stride);
    std::vector<unsigned char> host_scratch(count * scratch_stride);
    const std::array pointers{&lane.cache.key_payload, &lane.cache.value_payload,
                              &lane.cache.key_scales, &lane.cache.value_scales};
    for (std::size_t index = 0; index <= pointers.size(); ++index) {
        if (index == pointers.size()) {
            cuda_check(cudaMemcpyAsync(shifted_queries.data + 1, original.buffers.queries,
                                       row_size * sizeof(Bf16), cudaMemcpyDeviceToDevice, stream));
            lane.buffers.queries = shifted_queries.data + 1;
        } else {
            auto* pointer = pointers[index];
            const auto bytes = index < 2 ? payload_bytes : scale_bytes;
            cuda_check(cudaMemcpyAsync(shifted.data + 1, *pointer, bytes,
                                       cudaMemcpyDeviceToDevice, stream));
            *pointer = shifted.data + 1;
        }
        cuda_check(cudaMemsetAsync(output, poison, host_output.size(), stream));
        cuda_check(cudaMemsetAsync(scratch, poison, host_scratch.size(), stream));
        cuda_check(invoke(inputs.data(), count, stream));
        expect_launches(count);
        collect_outputs(output, scratch, host_output, host_scratch, stream);
        check_guards(host_output, output_stride, count);
        check_guards(host_scratch, scratch_stride, count);
        expect_poison(host_scratch.data() + (count - 1) * scratch_stride, scratch_stride,
                      "Unaligned lane wrote split scratch");
        check(host_scratch[guard] != poison, "Aligned neighbor did not write split scratch");
        check_outputs(inputs, host_output, -1, physical_pages);
        lane = original;
    }
    std::cout << "unaligned_cases=5\n";
}

void check_narrow(std::vector<PagedDecodeLane> inputs, int physical_pages,
                  unsigned char* output, unsigned char* scratch, cudaStream_t stream) {
    constexpr std::array lengths{8192, 8193, 12345, 16384};
    inputs.resize(lengths.size());
    for (std::size_t index = 0; index < lengths.size(); ++index) {
        inputs[index].position = lengths[index] - 1;
        inputs[index].pages.page_count = (lengths[index] + page_size - 1) / page_size;
    }
    std::vector<unsigned char> host_output(inputs.size() * output_stride);
    std::vector<unsigned char> host_scratch(inputs.size() * scratch_stride);
    cuda_check(cudaMemsetAsync(output, poison, host_output.size(), stream));
    cuda_check(cudaMemsetAsync(scratch, poison, host_scratch.size(), stream));
    cuda_check(invoke(inputs.data(), inputs.size(), stream));
    expect_launches(inputs.size());
    collect_outputs(output, scratch, host_output, host_scratch, stream);
    check_guards(host_output, output_stride, inputs.size());
    check_guards(host_scratch, scratch_stride, inputs.size());
    check_outputs(inputs, host_output, -1, physical_pages);
}

void run(Mode mode) {
    const bool full = mode == Mode::full;
    const bool recovery = mode == Mode::recovery;
    const int count = full ? width : mode == Mode::small ? 1 : 5;
    const int physical_pages = full ? pages : 512;
    const int marker = recovery ? page_map(0, 32768 / page_size - 1, physical_pages) : -1;
    Buffer<std::uint8_t> keys(physical_pages * mxfp8_payload_bytes_per_page);
    Buffer<std::uint8_t> values(physical_pages * mxfp8_payload_bytes_per_page);
    Buffer<std::uint8_t> ks(physical_pages * mxfp8_scale_bytes_per_page);
    Buffer<std::uint8_t> vs(physical_pages * mxfp8_scale_bytes_per_page);
    Buffer<int> page_device(count * pages);
    Buffer<Bf16> queries(count * row_size);
    Buffer<unsigned char> output(count * output_stride);
    Buffer<unsigned char> scratch(count * scratch_stride);
    Mxfp8KvCache cache{keys.data, values.data, ks.data, vs.data, physical_pages};
    cudaStream_t stream = nullptr;
    cuda_check(cudaStreamCreate(&stream));
    std::vector<int> page_host(count * pages);
    std::vector<Bf16> query_host(count * row_size, __float2bfloat16(0.0f));
    std::vector<PagedDecodeLane> inputs;
    const std::array lengths{32768, 32769, 65537, 131073, 262144};
    for (int lane = 0; lane < count; ++lane) {
        for (int page = 0; page < pages; ++page) {
            page_host[lane * pages + page] = page_map(lane, page, physical_pages);
        }
        for (int head = 0; head < heads; ++head) {
            query_host[lane * row_size + head * dim] = __float2bfloat16(query_value(lane, head));
        }
        const int length = full && lane < int(lengths.size()) ? lengths[lane]
            : recovery ? 32769 : 32768 + lane;
        inputs.push_back({cache, {page_device.data + lane * pages, (length + 127) / 128},
                          length - 1, {queries.data + lane * row_size,
                          reinterpret_cast<Bf16*>(output.data + lane * output_stride + guard),
                          scratch.data + lane * scratch_stride + guard}});
    }
    if (recovery) {
        query_host[row_size + 12 * dim] =
            __float2bfloat16(std::numeric_limits<float>::quiet_NaN());
    }
    cuda_check(cudaMemcpyAsync(page_device.data, page_host.data(), page_host.size() * sizeof(int),
                               cudaMemcpyHostToDevice, stream));
    cuda_check(cudaMemcpyAsync(queries.data, query_host.data(), query_host.size() * sizeof(Bf16),
                               cudaMemcpyHostToDevice, stream));
    const auto slots = static_cast<std::size_t>(physical_pages) * mxfp8_payload_bytes_per_page;
    fill_cache<<<(slots + 255) / 256, 256, 0, stream>>>(cache, physical_pages, marker);
    cuda_check(cudaGetLastError());
    cuda_check(cudaMemsetAsync(output.data, poison, count * output_stride, stream));
    cuda_check(cudaMemsetAsync(scratch.data, poison, count * scratch_stride, stream));
    if (full) check_errors(inputs, scratch.data, stream);
    if (mode == Mode::unaligned) {
        check_unaligned(inputs, physical_pages, output.data, scratch.data, stream);
    }
    cuda_check(cudaMemsetAsync(output.data, poison, count * output_stride, stream));
    cuda_check(cudaMemsetAsync(scratch.data, poison, count * scratch_stride, stream));
    cuda_check(invoke(inputs.data(), count, stream));
    expect_launches(count);
    std::vector<unsigned char> host_output(count * output_stride);
    std::vector<unsigned char> host_scratch(count * scratch_stride);
    collect_outputs(output.data, scratch.data, host_output, host_scratch, stream);
    check_guards(host_output, output_stride, count);
    check_guards(host_scratch, scratch_stride, count);
    if (full) {
        constexpr int longest = 4;
        check_ordered(inputs[longest], host_output.data() + longest * output_stride + guard,
                      stream);
    }
    if (recovery) check_recovery(host_scratch, count, marker, physical_pages);
    check_outputs(inputs, host_output, marker, physical_pages);
    if (full) check_narrow(inputs, physical_pages, output.data, scratch.data, stream);
    cuda_check(cudaStreamDestroy(stream));
}

void check_serial() {
    constexpr int physical = 5;
    constexpr std::array positions{255, 511, 257};
    Buffer<std::uint8_t> keys(physical * mxfp8_payload_bytes_per_page);
    Buffer<std::uint8_t> values(physical * mxfp8_payload_bytes_per_page);
    Buffer<std::uint8_t> key_scales(physical * mxfp8_scale_bytes_per_page);
    Buffer<std::uint8_t> value_scales(physical * mxfp8_scale_bytes_per_page);
    Buffer<int> map(physical), position(1);
    Buffer<Bf16> query(row_size), direct(row_size), replay(row_size);
    const Mxfp8KvCache cache{keys.data, values.data, key_scales.data, value_scales.data, physical};
    const Mxfp8PageTable table{map.data, physical};
    std::array<int, physical> mapping{};
    std::array<Bf16, row_size> queries{}, expected{}, actual{};
    for (int row = 0; row < physical; ++row) mapping[row] = page_map(0, row, physical);
    for (int head = 0; head < heads; ++head) queries[head * dim] = Bf16(query_value(0, head));
    cudaStream_t stream{};
    cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    cuda_check(cudaMemcpyAsync(map.data, mapping.data(), sizeof(mapping), cudaMemcpyHostToDevice,
                               stream));
    cuda_check(cudaMemcpyAsync(query.data, queries.data(), sizeof(queries), cudaMemcpyHostToDevice,
                               stream));
    fill_cache<<<physical * mxfp8_payload_bytes_per_page / 256, 256, 0, stream>>>(cache, physical,
                                                                              -1);
    cuda_check(cudaGetLastError());
    cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    watching = true;
    observed = 0;
    cuda_check(decode_gqa_attention(cache, table, position.data, query.data, replay.data, stream));
    watching = false;
    cudaGraph_t graph{};
    cuda_check(cudaStreamEndCapture(stream, &graph));
    std::size_t nodes = 0;
    cuda_check(cudaGraphGetNodes(graph, nullptr, &nodes));
    check(nodes == 1 && observed == 1, "Serial GQA graph must contain one kernel");
    const bool ordered_graph = launches[0].kind == Kind::ordered;
    if (ordered_graph) expect_launch(launches[0], Kind::ordered, dim3(1, heads / 2), dim);
    bool ordered_direct = true;
    cudaGraphExec_t exec{};
    cuda_check(cudaGraphInstantiate(&exec, graph, 0));
    for (int last : positions) {
        cuda_check(cudaMemcpyAsync(position.data, &last, sizeof(last), cudaMemcpyHostToDevice,
                                   stream));
        observed = 0;
        watching = true;
        cuda_check(mxfp8_gqa_attention(cache, table, last + 1, last, 1, query.data, direct.data,
                                       stream));
        watching = false;
        check(observed == 1, "Scratch-free GQA must use one kernel");
        const bool ordered = launches[0].kind == Kind::ordered;
        if (ordered) expect_launch(launches[0], Kind::ordered, dim3(1, heads / 2), dim);
        ordered_direct &= ordered;
        cuda_check(cudaGraphLaunch(exec, stream));
        cuda_check(cudaMemcpyAsync(expected.data(), direct.data, sizeof(expected),
                                   cudaMemcpyDeviceToHost, stream));
        cuda_check(cudaMemcpyAsync(actual.data(), replay.data, sizeof(actual),
                                   cudaMemcpyDeviceToHost, stream));
        cuda_check(cudaStreamSynchronize(stream));
        check(std::memcmp(expected.data(), actual.data(), sizeof(actual)) == 0,
              "Graph did not use the current device position");
        for (int head = 0; head < heads; ++head) {
            const auto truth = reference(0, last + 1, head, 0, -1, physical);
            check(std::abs(float(actual[head * dim]) - truth) < 0.005,
                  "Scratch-free GQA disagrees with FP64");
        }
    }
    cuda_check(cudaGraphExecDestroy(exec));
    cuda_check(cudaGraphDestroy(graph));
    cuda_check(cudaStreamDestroy(stream));
    if (!ordered_graph) std::cerr << "Wrong reader for bound graph position\n";
    if (!ordered_direct) std::cerr << "Wrong reader for scratch-free host position\n";
    check(ordered_graph && ordered_direct, "Serial GQA did not use the scratch-free tiled reader");
}

void record_launch(std::string_view name, dim3 grid, dim3& block) {
    const auto type = name.find("split_kernel") != name.npos ? Kind::split
        : name.find("target_split_combine") != name.npos ? Kind::combine
        : name.find("tiled_prefill_attention") != name.npos ? Kind::ordered : Kind::other;
    check(observed < int(launches.size()), "Too many grouped launches");
    launches[observed++] = {type, grid, block};
    if (observed == fail_at) block.x = 1025;
}

}

extern "C" cudaError_t __cudaLaunchKernel(cudaKernel_t kernel, dim3 grid, dim3 block,
                                          void** args, std::size_t shared, cudaStream_t stream) {
    static const auto launch = reinterpret_cast<decltype(&__cudaLaunchKernel)>(
        dlsym(RTLD_NEXT, "__cudaLaunchKernel"));
    check(launch != nullptr, "Cannot resolve CUDA kernel launch");
    if (watching) {
        static void* const driver = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
        check(driver != nullptr, "Cannot load the CUDA driver");
        static const auto get_name = reinterpret_cast<decltype(&cuKernelGetName)>(
            dlsym(driver, "cuKernelGetName"));
        const char* name = nullptr;
        check(get_name && get_name(&name, kernel) == CUDA_SUCCESS,
              "Cannot identify compiler-generated CUDA launch");
        record_launch(name, grid, block);
    }
    return launch(kernel, grid, block, args, shared, stream);
}

extern "C" cudaError_t __real_cudaLaunchKernel(const void*, dim3, dim3, void**, std::size_t,
                                               cudaStream_t);
extern "C" cudaError_t __wrap_cudaLaunchKernel(const void* function, dim3 grid, dim3 block,
                                               void** args, std::size_t shared,
                                               cudaStream_t stream) {
    if (watching) {
        const char* name = nullptr;
        cuda_check(cudaFuncGetName(&name, function));
        record_launch(name, grid, block);
    }
    return __real_cudaLaunchKernel(function, grid, block, args, shared, stream);
}

int main(int argc, char** argv) try {
    check(argc <= 2, "Usage: long-decode-check [full|small|recovery|unaligned|serial]");
    const std::string_view mode = argc == 2 ? argv[1] : "full";
    if (mode == "full") run(Mode::full);
    else if (mode == "small") run(Mode::small);
    else if (mode == "recovery") run(Mode::recovery);
    else if (mode == "unaligned") run(Mode::unaligned);
    else if (mode == "serial") check_serial();
    else check(false, "Invalid operator test mode");
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
