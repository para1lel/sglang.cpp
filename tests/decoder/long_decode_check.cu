#include "sglang/decoder_batch.h"
#include "sglang/decoder_cuda.cuh"

#include <cuda_runtime.h>

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

enum class Kind { paired, tensor, merge, replay, other };
enum class Mode { full, paired, small, recovery, unaligned };
struct Launch {
    Kind kind;
    dim3 grid;
    dim3 block;
};
std::array<Launch, 4> launches;
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

void expect_launches(int count, Kind split) {
    check(observed == 3, "Expected split, merge and replay launches");
    const std::array kinds{split, Kind::merge, Kind::replay};
    for (int index = 0; index < observed; ++index) {
        const auto launch = launches[index];
        const bool tensor = kinds[index] == Kind::tensor;
        check(launch.kind == kinds[index] && launch.grid.x == (index == 0 ? 16 : 24)
                  && launch.grid.y == (tensor ? 4 : index == 0 ? 12 : 1)
                  && launch.grid.z == count && launch.block.x == (tensor ? 128 : dim)
                  && launch.block.y == 1 && launch.block.z == 1,
              "Unexpected grouped kernel or geometry");
    }
    std::cout << "split=" << (split == Kind::tensor ? "tensor" : "paired")
              << " grid=16x" << launches[0].grid.y << 'x' << count
              << " block=" << launches[0].block.x << '\n';
}

cudaError_t invoke(const PagedDecodeLane* inputs, int count, cudaStream_t stream,
                   int fault = 0) {
    observed = 0;
    fail_at = fault;
    watching = true;
    const auto result = long_paged_decode_gqa(inputs, count, stream);
    watching = false;
    return result;
}

void invalid(const PagedDecodeLane* inputs, int count, cudaStream_t stream) {
    check(invoke(inputs, count, stream) == cudaErrorInvalidValue && observed == 0,
          "Invalid grouped input must fail before launch");
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
    invalid(inputs.data(), width + 1, stream);
    const auto original = inputs[1];
    inputs[1].position = 32766;
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
    for (int count : {2, 5}) {
        for (int fault = 1; fault <= 3; ++fault) {
            check(invoke(inputs.data(), count, stream, fault) == cudaErrorLaunchFailure &&
                      observed == fault, "Launch error changed or a later pass was submitted");
            cuda_check(cudaStreamSynchronize(stream));
        }
    }
    std::cout << "invalid=10 injected_launch_failures=6\n";
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
                head * paged_decode_scratch_bytes / heads + dim * sizeof(float);
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

void check_unaligned(std::vector<PagedDecodeLane> inputs, std::size_t bytes,
                     cudaStream_t stream) {
    Buffer<std::uint8_t> shifted(bytes + 1);
    auto& cache = inputs.back().cache;
    for (auto payload : {&cache.key_payload, &cache.value_payload}) {
        const auto original = *payload;
        cuda_check(cudaMemcpyAsync(shifted.data + 1, original, bytes,
                                   cudaMemcpyDeviceToDevice, stream));
        *payload = shifted.data + 1;
        cuda_check(invoke(inputs.data(), inputs.size(), stream));
        expect_launches(inputs.size(), Kind::paired);
        cuda_check(cudaStreamSynchronize(stream));
        *payload = original;
    }
}

void run(Mode mode) {
    const bool full = mode == Mode::full || mode == Mode::paired;
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
    if (mode == Mode::unaligned) check_unaligned(inputs, slots, stream);
    cuda_check(invoke(inputs.data(), count, stream));
    expect_launches(count, mode == Mode::full || recovery || mode == Mode::unaligned
                              ? Kind::tensor : Kind::paired);
    std::vector<unsigned char> host_output(count * output_stride);
    std::vector<unsigned char> host_scratch(count * scratch_stride);
    cuda_check(cudaMemcpyAsync(host_output.data(), output.data, host_output.size(),
                               cudaMemcpyDeviceToHost, stream));
    cuda_check(cudaMemcpyAsync(host_scratch.data(), scratch.data, host_scratch.size(),
                               cudaMemcpyDeviceToHost, stream));
    cuda_check(cudaStreamSynchronize(stream));
    check_guards(host_output, output_stride, count);
    check_guards(host_scratch, scratch_stride, count);
    if (full) {
        constexpr int longest = 4;
        check_ordered(inputs[longest], host_output.data() + longest * output_stride + guard,
                      stream);
    }
    if (recovery) check_recovery(host_scratch, count, marker, physical_pages);
    check_outputs(inputs, host_output, marker, physical_pages);
    cuda_check(cudaStreamDestroy(stream));
}

}

extern "C" cudaError_t __real_cudaLaunchKernel(const void*, dim3, dim3, void**, std::size_t,
                                               cudaStream_t);
extern "C" cudaError_t __wrap_cudaLaunchKernel(const void* function, dim3 grid, dim3 block,
                                               void** args, std::size_t shared,
                                               cudaStream_t stream) {
    if (watching) {
        const char* name = nullptr;
        cuda_check(cudaFuncGetName(&name, function));
        const std::string_view symbol(name);
        const auto type = symbol.find("split_long_decode_attention") != symbol.npos ? Kind::paired
            : symbol.find("split_long_decode_tensor") != symbol.npos ? Kind::tensor
            : symbol.find("merge_decode_attention") != symbol.npos ? Kind::merge
            : symbol.find("replay_long_paged_decode") != symbol.npos ? Kind::replay : Kind::other;
        check(observed < int(launches.size()), "Too many grouped launches");
        launches[observed++] = {type, grid, block};
        if (observed == fail_at) return cudaErrorLaunchFailure;
    }
    return __real_cudaLaunchKernel(function, grid, block, args, shared, stream);
}

int main(int argc, char** argv) try {
    check(argc <= 2, "Usage: long-decode-check [full|paired|small|recovery|unaligned]");
    const std::string_view mode = argc == 2 ? argv[1] : "full";
    if (mode == "full") run(Mode::full);
    else if (mode == "paired") run(Mode::paired);
    else if (mode == "small") run(Mode::small);
    else if (mode == "recovery") run(Mode::recovery);
    else if (mode == "unaligned") run(Mode::unaligned);
    else check(false, "Invalid operator test mode");
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
