#include "sglang/text_ops.h"

#include "sglang/decoder_cuda.cuh"

#include <algorithm>
#include <cstddef>

#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <math_constants.h>

namespace sglang {
namespace {

constexpr int warp_width = 32;
constexpr int block_threads = 256;
constexpr int block_warps = block_threads / warp_width;
constexpr unsigned int warp_mask = 0xffffffffu;
constexpr float fp8_max = 448.0f;
constexpr float quant_epsilon = 1.0e-10f;
constexpr float rms_epsilon = 1.0e-6f;
constexpr int argmax_tile = 4 * block_threads;
static_assert(text_argmax_scratch_count(argmax_tile) == 1);
static_assert(text_argmax_scratch_count(argmax_tile + 1) == 2);

constexpr int divide_up(int count, int tile) {
    return count / tile + (count % tile != 0);
}

bool aligned(const void* pointer, std::size_t alignment) {
    return reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0;
}

__device__ float warp_sum(float value) {
    for (int offset = warp_width / 2; offset > 0; offset /= 2) {
        value += __shfl_down_sync(warp_mask, value, offset);
    }
    return value;
}

__device__ float warp_max(float value) {
    for (int offset = warp_width / 2; offset > 0; offset /= 2) {
        value = fmaxf(value, __shfl_down_sync(warp_mask, value, offset));
    }
    return value;
}

__global__ void embedding_kernel(const __nv_bfloat16* row, int width, __nv_bfloat16* output) {
    const auto col = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (col < static_cast<std::size_t>(width)) {
        output[col] = row[col];
    }
}

struct EmbeddingTokens {
    int ids[text_embedding_max_rows];
};

__global__ void embedding_rows_kernel(const __nv_bfloat16* weight, int width,
                                      const __grid_constant__ EmbeddingTokens tokens,
                                      __nv_bfloat16* output) {
    const auto col = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (col < static_cast<std::size_t>(width)) {
        output[static_cast<std::size_t>(blockIdx.y) * width + col] =
            weight[static_cast<std::size_t>(tokens.ids[blockIdx.y]) * width + col];
    }
}

__global__ void decode_embedding_kernel(const __nv_bfloat16* weight, int vocab_size, int width,
                                          const TextLogitCandidate* previous,
                                          __nv_bfloat16* output) {
    const auto column = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (column < static_cast<std::size_t>(width)) {
        const int token_id = previous->token_id;
        const int row = token_id >= 0 && token_id < vocab_size ? token_id : 0;
        output[column] = weight[static_cast<std::size_t>(row) * width + column];
    }
}

__global__ void set_decode_position_kernel(int* position, int value) {
    *position = value;
}

__global__ void advance_decode_position_kernel(int* position) {
    ++*position;
}

__device__ uint4 load_128(const void* address) {
    uint4 bits;
    const auto global = __cvta_generic_to_global(address);
    asm("ld.global.v4.b32 {%0, %1, %2, %3}, [%4];"
        : "=r"(bits.x), "=r"(bits.y), "=r"(bits.z), "=r"(bits.w)
        : "l"(global) : "memory");
    return bits;
}

__device__ float bf16x2_dot(unsigned int lhs, unsigned int rhs, float sum) {
    sum = fmaf(__uint_as_float(lhs << 16), __uint_as_float(rhs << 16), sum);
    return fmaf(__uint_as_float(lhs & 0xffff0000u), __uint_as_float(rhs & 0xffff0000u), sum);
}

template <bool Packed, bool Batched = false>
__global__ void bf16_gemv_kernel(const __nv_bfloat16* weight, const __nv_bfloat16* input,
                                 int out_width, int in_width, __nv_bfloat16* output) {
    if constexpr (Batched) {
        input += static_cast<std::size_t>(blockIdx.y) * in_width;
        output += static_cast<std::size_t>(blockIdx.y) * out_width;
    }
    const int lane = threadIdx.x % warp_width;
    const int row = blockIdx.x * block_warps + threadIdx.x / warp_width;
    if (row >= out_width) {
        return;
    }
    const auto* row_weight = weight + static_cast<std::size_t>(row) * in_width;
    const auto width = static_cast<std::size_t>(in_width);
    float sum = 0.0f;
    if constexpr (Packed) {
        constexpr int pack = 8;
        float second_sum = 0.0f;
        for (std::size_t col = lane * pack; col < width; col += warp_width * pack) {
            const uint4 rhs = load_128(input + col);
            const uint4 lhs = load_128(row_weight + col);
            sum = bf16x2_dot(lhs.x, rhs.x, sum);
            second_sum = bf16x2_dot(lhs.y, rhs.y, second_sum);
            sum = bf16x2_dot(lhs.z, rhs.z, sum);
            second_sum = bf16x2_dot(lhs.w, rhs.w, second_sum);
        }
        sum += second_sum;
    } else {
        for (std::size_t col = lane; col < width; col += warp_width) {
            sum = fmaf(__bfloat162float(row_weight[col]), __bfloat162float(input[col]), sum);
        }
    }
    sum = warp_sum(sum);
    if (lane == 0) {
        output[row] = __float2bfloat16_rn(sum);
    }
}

template <int Count>
struct Bf16Projections {
    TextBf16Projection entries[Count];
};

template <int Count>
__global__ void bf16_batch_packed_kernel(const __nv_bfloat16* weight, int out_width,
                                         int in_width, Bf16Projections<Count> batch) {
    const int lane = threadIdx.x % warp_width;
    const int row = blockIdx.x * block_warps + threadIdx.x / warp_width;
    if (row >= out_width) {
        return;
    }
    const auto* row_weight = weight + static_cast<std::size_t>(row) * in_width;
    constexpr int pack = 8;
    float sums[Count]{};
    float second_sums[Count]{};
    for (std::size_t col = lane * pack; col < static_cast<std::size_t>(in_width);
         col += warp_width * pack) {
        const uint4 lhs = load_128(row_weight + col);
#pragma unroll
        for (int index = 0; index < Count; ++index) {
            const uint4 rhs = load_128(batch.entries[index].input + col);
            sums[index] = bf16x2_dot(lhs.x, rhs.x, sums[index]);
            second_sums[index] = bf16x2_dot(lhs.y, rhs.y, second_sums[index]);
            sums[index] = bf16x2_dot(lhs.z, rhs.z, sums[index]);
            second_sums[index] = bf16x2_dot(lhs.w, rhs.w, second_sums[index]);
        }
    }
#pragma unroll
    for (int index = 0; index < Count; ++index) {
        const float sum = warp_sum(sums[index] + second_sums[index]);
        if (lane == 0) {
            batch.entries[index].output[row] = __float2bfloat16_rn(sum);
        }
    }
}

template <int Count>
cudaError_t launch_bf16_batch(const __nv_bfloat16* weight, int out_width, int in_width,
                              const TextBf16Projection* projections, cudaStream_t stream) {
    Bf16Projections<Count> batch;
    std::copy_n(projections, Count, batch.entries);
    void* args[] = {&weight, &out_width, &in_width, &batch};
    return cudaLaunchKernel(bf16_batch_packed_kernel<Count>, divide_up(out_width, block_warps),
                            block_threads, args, 0, stream);
}

template <bool Batched>
__global__ void quantize_fp8_kernel(const __nv_bfloat16* input, int width,
                                    TextFp8Activation output) {
    constexpr int group_warps = text_fp8_group_size / warp_width;
    __shared__ float warp_maxima[group_warps];
    __shared__ float reciprocal_scale;
    if constexpr (Batched) {
        const auto offset = static_cast<std::size_t>(blockIdx.y) * width;
        input += offset;
        output.values += offset;
        output.scales += static_cast<std::size_t>(blockIdx.y) * gridDim.x;
    }
    const int lane = threadIdx.x % warp_width;
    const int warp = threadIdx.x / warp_width;
    const auto col = static_cast<std::size_t>(blockIdx.x) * text_fp8_group_size + threadIdx.x;
    const bool valid = col < static_cast<std::size_t>(width);
    const float value = valid ? __bfloat162float(input[col]) : 0.0f;
    const float maximum = warp_max(fabsf(value));
    if (lane == 0) {
        warp_maxima[warp] = maximum;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float amax = quant_epsilon;
        for (int index = 0; index < group_warps; ++index) {
            amax = fmaxf(amax, warp_maxima[index]);
        }
        const float scale = __fdiv_rn(amax, fp8_max);
        output.scales[blockIdx.x] = scale;
        reciprocal_scale = __fdiv_rn(1.0f, scale);
    }
    __syncthreads();
    if (valid) {
        output.values[col] = __nv_cvt_float_to_fp8(value * reciprocal_scale, __NV_SATFINITE,
                                                   __NV_E4M3);
    }
}

__device__ float decode_fp8(std::uint8_t bits) {
    __nv_fp8_e4m3 value;
    value.__x = bits;
    return static_cast<float>(value);
}

__device__ float4 load_fp8x4(const std::uint8_t* address) {
    __nv_fp8x4_e4m3 value;
    const auto global = __cvta_generic_to_global(address);
    asm("ld.global.b32 %0, [%1];" : "=r"(value.__x) : "l"(global) : "memory");
    return static_cast<float4>(value);
}

template <bool Packed>
__global__ void fp8_gemv_kernel(const std::uint8_t* weight, const __nv_bfloat16* weight_scales,
                                TextFp8ActivationView input, int out_width, int in_width,
                                int groups, __nv_bfloat16* output) {
    const int lane = threadIdx.x % warp_width;
    const int row = blockIdx.x * block_warps + threadIdx.x / warp_width;
    if (row >= out_width) {
        return;
    }
    const auto* row_weight = weight + static_cast<std::size_t>(row) * in_width;
    const auto* row_scales =
        weight_scales + static_cast<std::size_t>(row / text_fp8_group_size) * groups;
    float sum = 0.0f;
    for (int group = 0; group < groups; ++group) {
        const auto start = static_cast<std::size_t>(group) * text_fp8_group_size;
        const int count = min(text_fp8_group_size, in_width - static_cast<int>(start));
        float dot = 0.0f;
        if constexpr (Packed) {
            const int col = lane * 4;
            if (col < count) {
                const float4 rhs = load_fp8x4(input.values + start + col);
                const float4 lhs = load_fp8x4(row_weight + start + col);
                dot = fmaf(lhs.x, rhs.x, dot);
                dot = fmaf(lhs.y, rhs.y, dot);
                dot = fmaf(lhs.z, rhs.z, dot);
                dot = fmaf(lhs.w, rhs.w, dot);
            }
        } else {
            for (int col = lane; col < count; col += warp_width) {
                dot = fmaf(decode_fp8(row_weight[start + col]),
                           decode_fp8(input.values[start + col]), dot);
            }
        }
        dot = warp_sum(dot);
        if (lane == 0) {
            const float scaled_dot = dot * input.scales[group];
            sum = fmaf(scaled_dot, __bfloat162float(row_scales[group]), sum);
        }
    }
    if (lane == 0) {
        output[row] = __float2bfloat16_rn(sum);
    }
}

__device__ float fp8x4_dot(unsigned int lhs_bits, unsigned int rhs_bits, float sum) {
#pragma unroll
    for (int pair = 0; pair < 2; ++pair) {
        const auto left = __nv_cvt_fp8x2_to_halfraw2(
            static_cast<std::uint16_t>(lhs_bits >> (16 * pair)), __NV_E4M3);
        const auto right = __nv_cvt_fp8x2_to_halfraw2(
            static_cast<std::uint16_t>(rhs_bits >> (16 * pair)), __NV_E4M3);
        asm("fma.rn.f32.f16 %0, %1, %2, %0;" : "+f"(sum) : "h"(left.x), "h"(right.x));
        asm("fma.rn.f32.f16 %0, %1, %2, %0;" : "+f"(sum) : "h"(left.y), "h"(right.y));
    }
    return sum;
}

constexpr int fp8_pack_bytes = 16;
constexpr int fp8_packed_warps = 4;
constexpr int fp8_groups_per_warp = warp_width * fp8_pack_bytes / text_fp8_group_size;

bool supports_wide_fp8(const std::uint8_t* weight, TextFp8ActivationView input, int in_width) {
    return in_width % (fp8_groups_per_warp * text_fp8_group_size) == 0 &&
           aligned(weight, fp8_pack_bytes) && aligned(input.values, fp8_pack_bytes);
}

__device__ uint4 load_fp8_weight(const void* address) {
    uint4 bits;
    const auto global = __cvta_generic_to_global(address);
    asm("ld.global.cs.v4.b32 {%0, %1, %2, %3}, [%4];"
        : "=r"(bits.x), "=r"(bits.y), "=r"(bits.z), "=r"(bits.w)
        : "l"(global) : "memory");
    return bits;
}

__global__ void fp8_gemv_packed_kernel(const std::uint8_t* weight,
                                       const __nv_bfloat16* weight_scales,
                                       TextFp8ActivationView input, int out_width, int in_width,
                                       int groups, __nv_bfloat16* output) {
    const int lane = threadIdx.x % warp_width;
    const int row = blockIdx.x * fp8_packed_warps + threadIdx.x / warp_width;
    if (row >= out_width) {
        return;
    }
    const auto* row_weight = weight + static_cast<std::size_t>(row) * in_width;
    const auto* row_scales =
        weight_scales + static_cast<std::size_t>(row / text_fp8_group_size) * groups;
    constexpr int lanes_per_group = warp_width / fp8_groups_per_warp;
    const int lane_group = lane / lanes_per_group;
    const int lane_col = lane * fp8_pack_bytes;
    float sum = 0.0f;
    for (int group = 0; group < groups; group += fp8_groups_per_warp) {
        const auto col = static_cast<std::size_t>(group) * text_fp8_group_size + lane_col;
        const uint4 rhs = load_128(input.values + col);
        const uint4 lhs = load_fp8_weight(row_weight + col);
        const float low = fp8x4_dot(lhs.y, rhs.y, fp8x4_dot(lhs.x, rhs.x, 0.0f));
        const float high = fp8x4_dot(lhs.w, rhs.w, fp8x4_dot(lhs.z, rhs.z, 0.0f));
        const int scale_index = group + lane_group;
        const float scaled = (low + high) * input.scales[scale_index];
        sum = fmaf(scaled, __bfloat162float(row_scales[scale_index]), sum);
    }
    sum = warp_sum(sum);
    if (lane == 0) {
        output[row] = __float2bfloat16_rn(sum);
    }
}

template <int Count>
struct Fp8Projections {
    TextFp8Projection entries[Count];
};

template <int Count>
__global__ void fp8_batch_packed_kernel(const std::uint8_t* weight,
                                        const __nv_bfloat16* weight_scales,
                                        int out_width, int in_width, int groups,
                                        Fp8Projections<Count> projections) {
    const int lane = threadIdx.x % warp_width;
    const int row = blockIdx.x * fp8_packed_warps + threadIdx.x / warp_width;
    if (row >= out_width) {
        return;
    }
    const auto* row_weight = weight + static_cast<std::size_t>(row) * in_width;
    const auto* row_scales =
        weight_scales + static_cast<std::size_t>(row / text_fp8_group_size) * groups;
    constexpr int lanes_per_group = warp_width / fp8_groups_per_warp;
    const int lane_group = lane / lanes_per_group;
    const int lane_col = lane * fp8_pack_bytes;
    float sums[Count]{};
    for (int group = 0; group < groups; group += fp8_groups_per_warp) {
        const auto col = static_cast<std::size_t>(group) * text_fp8_group_size + lane_col;
        const uint4 lhs = load_fp8_weight(row_weight + col);
        const int scale_index = group + lane_group;
        const float scale = __bfloat162float(row_scales[scale_index]);
#pragma unroll
        for (int index = 0; index < Count; ++index) {
            const auto input = projections.entries[index].input;
            const uint4 rhs = load_128(input.values + col);
            const float low = fp8x4_dot(lhs.y, rhs.y, fp8x4_dot(lhs.x, rhs.x, 0.0f));
            const float high = fp8x4_dot(lhs.w, rhs.w, fp8x4_dot(lhs.z, rhs.z, 0.0f));
            const float scaled = (low + high) * input.scales[scale_index];
            sums[index] = fmaf(scaled, scale, sums[index]);
        }
    }
#pragma unroll
    for (int index = 0; index < Count; ++index) {
        const float sum = warp_sum(sums[index]);
        if (lane == 0) {
            projections.entries[index].output[row] = __float2bfloat16_rn(sum);
        }
    }
}

template <int Count>
cudaError_t launch_fp8_batch(const std::uint8_t* weight, const __nv_bfloat16* weight_scales,
                             int out_width, int in_width, const TextFp8Projection* projections,
                             cudaStream_t stream) {
    Fp8Projections<Count> batch;
    for (int index = 0; index < Count; ++index) {
        batch.entries[index] = projections[index];
    }
    int groups = text_fp8_group_count(in_width);
    void* args[] = {&weight, &weight_scales, &out_width, &in_width, &groups, &batch};
    return cudaLaunchKernel(fp8_batch_packed_kernel<Count>,
                            divide_up(out_width, fp8_packed_warps),
                            fp8_packed_warps * warp_width, args, 0, stream);
}

__device__ float fp8_pack_dot(uint4 lhs, uint4 rhs) {
    const float low = fp8x4_dot(lhs.y, rhs.y, fp8x4_dot(lhs.x, rhs.x, 0.0f));
    const float high = fp8x4_dot(lhs.w, rhs.w, fp8x4_dot(lhs.z, rhs.z, 0.0f));
    return low + high;
}

template <int Count>
__global__ void fp8_prefill_tile_kernel(const std::uint8_t* weight,
                                        const __nv_bfloat16* weight_scales,
                                        TextFp8ActivationView input, int out_width, int in_width,
                                        int groups, int rows, int tiles, __nv_bfloat16* output) {
    constexpr int lanes_per_group = warp_width / fp8_groups_per_warp;
    const int lane = threadIdx.x % warp_width;
    const int row = (blockIdx.x / tiles) * fp8_packed_warps + threadIdx.x / warp_width;
    if (row >= out_width) {
        return;
    }
    const int first = (blockIdx.x % tiles) * Count;
    const auto* row_weight = weight + static_cast<std::size_t>(row) * in_width;
    const auto* row_scales =
        weight_scales + static_cast<std::size_t>(row / text_fp8_group_size) * groups;
    const auto* values = input.values + static_cast<std::size_t>(first) * in_width;
    const auto* scales = input.scales + static_cast<std::size_t>(first) * groups;
    float sums[Count]{};
    for (int group = 0; group < groups; group += fp8_groups_per_warp) {
        const auto col = static_cast<std::size_t>(group) * text_fp8_group_size +
                         lane * fp8_pack_bytes;
        const auto lhs = load_fp8_weight(row_weight + col);
        const int scale_index = group + lane / lanes_per_group;
        const float weight_scale = __bfloat162float(row_scales[scale_index]);
#pragma unroll
        for (int item = 0; item < Count; ++item) {
            if (first + item < rows) {
                const auto rhs = load_128(values + static_cast<std::size_t>(item) * in_width + col);
                const float scaled =
                    fp8_pack_dot(lhs, rhs) *
                    scales[static_cast<std::size_t>(item) * groups + scale_index];
                sums[item] = fmaf(scaled, weight_scale, sums[item]);
            }
        }
    }
#pragma unroll
    for (int item = 0; item < Count; ++item) {
        if (first + item < rows) {
            const float sum = warp_sum(sums[item]);
            if (lane == 0) {
                output[static_cast<std::size_t>(first + item) * out_width + row] =
                    __float2bfloat16_rn(sum);
            }
        }
    }
}

template <int Count>
cudaError_t launch_fp8_rows(const std::uint8_t* weight, const __nv_bfloat16* weight_scales,
                            TextFp8ActivationView input, int out_width, int in_width, int rows,
                            __nv_bfloat16* output, cudaStream_t stream) {
    int groups = text_fp8_group_count(in_width);
    int tiles = divide_up(rows, Count);
    const auto blocks = static_cast<std::uint64_t>(divide_up(out_width, fp8_packed_warps)) * tiles;
    if (blocks > 2147483647u) {
        return cudaErrorInvalidConfiguration;
    }
    void* args[] = {&weight, &weight_scales, &input, &out_width, &in_width,
                     &groups, &rows, &tiles, &output};
    return cudaLaunchKernel(fp8_prefill_tile_kernel<Count>, static_cast<unsigned int>(blocks),
                            fp8_packed_warps * warp_width, args, 0, stream);
}

__device__ float residual_sum(const __nv_bfloat16* branch, const __nv_bfloat16* residual,
                              std::size_t col) {
    const float value = __bfloat162float(branch[col]);
    return residual == nullptr ? value : value + __bfloat162float(residual[col]);
}

__device__ __forceinline__ void rms_norm_row(const __nv_bfloat16* branch,
                                             const __nv_bfloat16* residual,
                                             const __nv_bfloat16* weight, int width,
                                             __nv_bfloat16* output,
                                             __nv_bfloat16* residual_output) {
    __shared__ float warp_squares[block_warps];
    __shared__ float inverse_rms;
    const int lane = threadIdx.x % warp_width;
    const int warp = threadIdx.x / warp_width;
    const auto count = static_cast<std::size_t>(width);
    float squares = 0.0f;
    for (std::size_t col = threadIdx.x; col < count; col += block_threads) {
        const float value = residual_sum(branch, residual, col);
        squares = fmaf(value, value, squares);
    }
    squares = warp_sum(squares);
    if (lane == 0) {
        warp_squares[warp] = squares;
    }
    __syncthreads();
    if (warp == 0) {
        squares = warp_sum(lane < block_warps ? warp_squares[lane] : 0.0f);
        if (lane == 0) {
            inverse_rms = rsqrtf(squares / static_cast<float>(width) + rms_epsilon);
        }
    }
    __syncthreads();
    for (std::size_t col = threadIdx.x; col < count; col += block_threads) {
        const float value = residual_sum(branch, residual, col);
        const float gain = 1.0f + __bfloat162float(weight[col]);
        if (residual_output != nullptr) {
            residual_output[col] = __float2bfloat16_rn(value);
        }
        output[col] = __float2bfloat16_rn((value * inverse_rms) * gain);
    }
}

template <bool Batched>
__global__ void rms_norm_kernel(const __nv_bfloat16* branch, const __nv_bfloat16* residual,
                                const __nv_bfloat16* weight, int width, __nv_bfloat16* output,
                                __nv_bfloat16* residual_output) {
    const auto offset = Batched ? static_cast<std::size_t>(blockIdx.x) * width : 0;
    rms_norm_row(branch + offset, residual ? residual + offset : nullptr, weight, width,
                 output + offset, residual_output ? residual_output + offset : nullptr);
}

__global__ void rms_norm_pair_rows_kernel(const __nv_bfloat16* first, const __nv_bfloat16* second,
                                          const __nv_bfloat16* first_weight,
                                          const __nv_bfloat16* second_weight, int width,
                                          __nv_bfloat16* output) {
    const auto offset = static_cast<std::size_t>(blockIdx.x) * width;
    const bool second_half = blockIdx.y != 0;
    rms_norm_row((second_half ? second : first) + offset, nullptr,
                 second_half ? second_weight : first_weight, width,
                 output + 2 * offset + blockIdx.y * static_cast<std::size_t>(width), nullptr);
}

__global__ void swiglu_kernel(const __nv_bfloat16* gate, const __nv_bfloat16* up, int width,
                              __nv_bfloat16* output) {
    const auto col = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (col < static_cast<std::size_t>(width)) {
        const float value = __bfloat162float(gate[col]);
        const auto activated = __float2bfloat16_rn(value / (1.0f + expf(-value)));
        output[col] = __float2bfloat16_rn(__bfloat162float(activated) * __bfloat162float(up[col]));
    }
}

__device__ TextLogitCandidate best_candidate(TextLogitCandidate first,
                                             TextLogitCandidate second) {
    if (second.token_id >= 0 &&
        (first.token_id < 0 || second.score > first.score ||
         (second.score == first.score && second.token_id < first.token_id))) {
        return second;
    }
    return first;
}

__device__ TextLogitCandidate warp_best(TextLogitCandidate best) {
    for (int offset = warp_width / 2; offset > 0; offset /= 2) {
        const TextLogitCandidate other{
            __shfl_down_sync(warp_mask, best.score, offset),
            __shfl_down_sync(warp_mask, best.token_id, offset)};
        best = best_candidate(best, other);
    }
    return best;
}

__device__ void store_block_best(TextLogitCandidate best, TextLogitCandidate* output) {
    __shared__ TextLogitCandidate warp_candidates[block_warps];
    const int lane = threadIdx.x % warp_width;
    const int warp = threadIdx.x / warp_width;
    best = warp_best(best);
    if (lane == 0) {
        warp_candidates[warp] = best;
    }
    __syncthreads();
    if (warp == 0) {
        best = warp_best(lane < block_warps ? warp_candidates[lane]
                                           : TextLogitCandidate{-CUDART_INF_F, -1});
        if (lane == 0) {
            *output = best;
        }
    }
}

__global__ void argmax_tiles_kernel(const __nv_bfloat16* logits, int vocab_size,
                                    TextLogitCandidate* scratch) {
    const unsigned int start = blockIdx.x * argmax_tile;
    TextLogitCandidate best{-CUDART_INF_F, -1};
    for (unsigned int item = threadIdx.x; item < argmax_tile; item += block_threads) {
        const unsigned int token = start + item;
        if (token < static_cast<unsigned int>(vocab_size)) {
            const float score = __bfloat162float(logits[token]);
            if (!isnan(score)) {
                best = best_candidate(best, {score, static_cast<int>(token)});
            }
        }
    }
    store_block_best(best, scratch + blockIdx.x);
}

__global__ void argmax_finish_kernel(const TextLogitCandidate* scratch, int count,
                                     TextLogitCandidate* output) {
    TextLogitCandidate best{-CUDART_INF_F, -1};
    for (int item = threadIdx.x; item < count; item += block_threads) {
        best = best_candidate(best, scratch[item]);
    }
    store_block_best(best, output);
}

}

cudaError_t text_embedding(const __nv_bfloat16* weight, int vocab_size, int width, int token_id,
                           __nv_bfloat16* output, cudaStream_t stream) {
    if (weight == nullptr || output == nullptr || width <= 0 ||
        token_id < 0 || token_id >= vocab_size) {
        return cudaErrorInvalidValue;
    }
    const auto* row = weight + static_cast<std::size_t>(token_id) * width;
    void* args[] = {&row, &width, &output};
    return cudaLaunchKernel(embedding_kernel, divide_up(width, block_threads), block_threads,
                            args, 0, stream);
}

cudaError_t text_embedding_rows(const __nv_bfloat16* weight, int vocab_size, int width,
                                const int* host_token_ids, int rows, __nv_bfloat16* output,
                                cudaStream_t stream) {
    if (!weight || !host_token_ids || !output || vocab_size <= 0 || width <= 0 ||
        rows <= 0 || rows > text_embedding_max_rows) {
        return cudaErrorInvalidValue;
    }
    EmbeddingTokens tokens{};
    std::copy_n(host_token_ids, rows, tokens.ids);
    if (std::any_of(tokens.ids, tokens.ids + rows,
                    [vocab_size](int token) { return token < 0 || token >= vocab_size; })) {
        return cudaErrorInvalidValue;
    }
    void* args[] = {&weight, &width, &tokens, &output};
    return cudaLaunchKernel(embedding_rows_kernel, dim3(divide_up(width, block_threads), rows),
                            block_threads, args, 0, stream);
}

cudaError_t decoder::decode_embedding(const __nv_bfloat16* weight, int vocab_size, int width,
                                      const TextLogitCandidate* previous, __nv_bfloat16* output,
                                      cudaStream_t stream) {
    void* arguments[] = {&weight, &vocab_size, &width, &previous, &output};
    return cudaLaunchKernel(decode_embedding_kernel, divide_up(width, block_threads),
                            block_threads, arguments, 0, stream);
}

cudaError_t decoder::set_decode_position(int* position, int value, cudaStream_t stream) {
    void* arguments[] = {&position, &value};
    return cudaLaunchKernel(set_decode_position_kernel, 1, 1, arguments, 0, stream);
}

cudaError_t decoder::advance_decode_position(int* position, cudaStream_t stream) {
    void* arguments[] = {&position};
    return cudaLaunchKernel(advance_decode_position_kernel, 1, 1, arguments, 0, stream);
}

cudaError_t text_bf16_gemv(const __nv_bfloat16* weight, const __nv_bfloat16* input,
                           int out_width, int in_width, __nv_bfloat16* output,
                           cudaStream_t stream) {
    if (weight == nullptr || input == nullptr || output == nullptr ||
        out_width <= 0 || in_width <= 0) {
        return cudaErrorInvalidValue;
    }
    const bool packed = in_width % 8 == 0 && aligned(weight, 16) && aligned(input, 16);
    void* args[] = {&weight, &input, &out_width, &in_width, &output};
    return cudaLaunchKernel(packed ? bf16_gemv_kernel<true> : bf16_gemv_kernel<false>,
                            divide_up(out_width, block_warps), block_threads, args, 0, stream);
}

cudaError_t text_bf16_gemv_rows(const __nv_bfloat16* weight, const __nv_bfloat16* input,
                                int out_width, int in_width, int rows, __nv_bfloat16* output,
                                cudaStream_t stream) {
    if (!weight || !input || !output || out_width <= 0 || in_width <= 0 ||
        rows <= 0 || rows > 65535) {
        return cudaErrorInvalidValue;
    }
    const bool packed = in_width % 8 == 0 && aligned(weight, 16) && aligned(input, 16);
    void* args[] = {&weight, &input, &out_width, &in_width, &output};
    const auto kernel = packed ? bf16_gemv_kernel<true, true> : bf16_gemv_kernel<false, true>;
    return cudaLaunchKernel(kernel, dim3(divide_up(out_width, block_warps), rows),
                            block_threads, args, 0, stream);
}

cudaError_t text_bf16_gemm(const __nv_bfloat16* weight, int out_width, int in_width,
                           const TextBf16Projection* projections, int count, cudaStream_t stream) {
    if (!weight || !projections || out_width <= 0 || in_width <= 0 ||
        count < 1 || count > text_fp8_max_batch) {
        return cudaErrorInvalidValue;
    }
    bool packed = in_width % 8 == 0 && aligned(weight, 16);
    for (int index = 0; index < count; ++index) {
        if (!projections[index].input || !projections[index].output) {
            return cudaErrorInvalidValue;
        }
        packed = packed && aligned(projections[index].input, 16);
    }
    if (packed) {
        switch (count) {
        case 5:
            return launch_bf16_batch<5>(weight, out_width, in_width, projections, stream);
        case 4:
            return launch_bf16_batch<4>(weight, out_width, in_width, projections, stream);
        case 3:
            return launch_bf16_batch<3>(weight, out_width, in_width, projections, stream);
        case 2:
            return launch_bf16_batch<2>(weight, out_width, in_width, projections, stream);
        }
    }
    for (int index = 0; index < count; ++index) {
        const auto& entry = projections[index];
        const auto status = text_bf16_gemv(weight, entry.input, out_width, in_width,
                                          entry.output, stream);
        if (status != cudaSuccess) {
            return status;
        }
    }
    return cudaSuccess;
}

cudaError_t text_quantize_fp8(const __nv_bfloat16* input, int width, TextFp8Activation output,
                             cudaStream_t stream) {
    if (input == nullptr || output.values == nullptr || output.scales == nullptr || width <= 0) {
        return cudaErrorInvalidValue;
    }
    void* args[] = {&input, &width, &output};
    return cudaLaunchKernel(quantize_fp8_kernel<false>, text_fp8_group_count(width),
                            text_fp8_group_size, args, 0, stream);
}

cudaError_t text_quantize_fp8_rows(const __nv_bfloat16* input, int width, int rows,
                                  TextFp8Activation output, cudaStream_t stream) {
    if (!input || !output.values || !output.scales || width <= 0 || rows <= 0 || rows > 65535) {
        return cudaErrorInvalidValue;
    }
    void* args[] = {&input, &width, &output};
    return cudaLaunchKernel(quantize_fp8_kernel<true>, dim3(text_fp8_group_count(width), rows),
                            text_fp8_group_size, args, 0, stream);
}

cudaError_t text_fp8_gemv(const std::uint8_t* weight, const __nv_bfloat16* weight_scales,
                          TextFp8ActivationView input, int out_width, int in_width,
                          __nv_bfloat16* output, cudaStream_t stream) {
    if (weight == nullptr || weight_scales == nullptr || input.values == nullptr ||
        input.scales == nullptr || output == nullptr || out_width <= 0 || in_width <= 0) {
        return cudaErrorInvalidValue;
    }
    int groups = text_fp8_group_count(in_width);
    const bool wide = supports_wide_fp8(weight, input, in_width);
    const bool packed = in_width % 4 == 0 && aligned(weight, 4) && aligned(input.values, 4);
    void* args[] = {&weight, &weight_scales, &input, &out_width, &in_width, &groups, &output};
    const auto kernel = wide ? fp8_gemv_packed_kernel
                             : (packed ? fp8_gemv_kernel<true> : fp8_gemv_kernel<false>);
    const int rows_per_block = wide ? fp8_packed_warps : block_warps;
    return cudaLaunchKernel(kernel, divide_up(out_width, rows_per_block),
                            rows_per_block * warp_width, args, 0, stream);
}

cudaError_t text_fp8_gemm(const std::uint8_t* weight, const __nv_bfloat16* weight_scales,
                          int out_width, int in_width, const TextFp8Projection* projections,
                          int count, cudaStream_t stream) {
    if (!weight || !weight_scales || !projections || out_width <= 0 || in_width <= 0 ||
        count < 1 || count > text_fp8_max_batch) {
        return cudaErrorInvalidValue;
    }
    bool wide = true;
    for (int index = 0; index < count; ++index) {
        const auto& entry = projections[index];
        if (!entry.input.values || !entry.input.scales || !entry.output) {
            return cudaErrorInvalidValue;
        }
        wide = wide && supports_wide_fp8(weight, entry.input, in_width);
    }
    if (wide && count == 5) {
        return launch_fp8_batch<5>(weight, weight_scales, out_width, in_width, projections, stream);
    }
    if (wide && count == 4) {
        return launch_fp8_batch<4>(weight, weight_scales, out_width, in_width, projections, stream);
    }
    if (wide && count == 3) {
        return launch_fp8_batch<3>(weight, weight_scales, out_width, in_width, projections, stream);
    }
    if (wide && count == 2) {
        return launch_fp8_batch<2>(weight, weight_scales, out_width, in_width, projections, stream);
    }
    for (int index = 0; index < count; ++index) {
        const auto& entry = projections[index];
        const auto status = text_fp8_gemv(weight, weight_scales, entry.input, out_width, in_width,
                                           entry.output, stream);
        if (status != cudaSuccess) {
            return status;
        }
    }
    return cudaSuccess;
}

cudaError_t text_fp8_gemv_rows(const std::uint8_t* weight, const __nv_bfloat16* weight_scales,
                               TextFp8ActivationView input, int out_width, int in_width, int rows,
                               __nv_bfloat16* output, cudaStream_t stream) {
    if (!weight || !weight_scales || !input.values || !input.scales || !output ||
        out_width <= 0 || in_width <= 0 || rows <= 0 || rows > 65535) {
        return cudaErrorInvalidValue;
    }
    if (rows > 1 && supports_wide_fp8(weight, input, in_width)) {
        if (rows <= 8 || (rows <= 16 && (out_width <= 1024 || in_width >= 16384))) {
            return launch_fp8_rows<2>(weight, weight_scales, input, out_width, in_width, rows,
                                      output, stream);
        }
        if (rows < 64 || out_width <= 1024) {
            return launch_fp8_rows<4>(weight, weight_scales, input, out_width, in_width, rows,
                                      output, stream);
        }
        return launch_fp8_rows<8>(weight, weight_scales, input, out_width, in_width, rows,
                                  output, stream);
    }
    const int groups = text_fp8_group_count(in_width);
    for (int row = 0; row < rows; ++row) {
        const TextFp8ActivationView source{
            input.values + static_cast<std::size_t>(row) * in_width,
            input.scales + static_cast<std::size_t>(row) * groups};
        const auto status = text_fp8_gemv(weight, weight_scales, source, out_width, in_width,
                                          output + static_cast<std::size_t>(row) * out_width,
                                          stream);
        if (status != cudaSuccess) {
            return status;
        }
    }
    return cudaSuccess;
}

namespace {

constexpr int rms_hidden_width = 5120;

__device__ void stage_rms_vector(const __nv_bfloat16* input, __nv_bfloat16* shared) {
    const uint4 vector = load_128(input);
    const auto address = static_cast<unsigned int>(__cvta_generic_to_shared(shared));
    asm("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
        :: "r"(address), "r"(vector.x), "r"(vector.y), "r"(vector.z), "r"(vector.w)
        : "memory");
}

__device__ __forceinline__ float rms_inverse(float squares, int width) {
    __shared__ float warp_squares[block_warps];
    __shared__ float inverse_rms;
    const int lane = threadIdx.x % warp_width;
    const int warp = threadIdx.x / warp_width;
    squares = warp_sum(squares);
    if (lane == 0) {
        warp_squares[warp] = squares;
    }
    __syncthreads();
    if (warp == 0) {
        squares = warp_sum(lane < block_warps ? warp_squares[lane] : 0.0f);
        if (lane == 0) {
            inverse_rms = rsqrtf(squares / static_cast<float>(width) + rms_epsilon);
        }
    }
    __syncthreads();
    return inverse_rms;
}

__device__ void store_rms(float value, float inverse_rms, const __nv_bfloat16* weight,
                          std::size_t col, __nv_bfloat16* output,
                          __nv_bfloat16* residual_output) {
    const float gain = 1.0f + __bfloat162float(weight[col]);
    if (residual_output != nullptr) {
        residual_output[col] = __float2bfloat16_rn(value);
    }
    output[col] = __float2bfloat16_rn((value * inverse_rms) * gain);
}

template <bool HasResidual, bool StoresResidual>
__global__ void rms_norm_vector_kernel(const __nv_bfloat16* branch,
                                       const __nv_bfloat16* residual,
                                       const __nv_bfloat16* weight, __nv_bfloat16* output,
                                       __nv_bfloat16* residual_output) {
    __shared__ alignas(16) __nv_bfloat16 branch_values[rms_hidden_width];
    __shared__ alignas(16) __nv_bfloat16 residual_values[HasResidual ? rms_hidden_width : 1];
    __shared__ alignas(16) __nv_bfloat16 gains[rms_hidden_width];
    constexpr int vector_width = sizeof(uint4) / sizeof(__nv_bfloat16);
    static_assert(rms_hidden_width % block_threads == 0 && rms_hidden_width % vector_width == 0);
    for (int col = threadIdx.x * vector_width; col < rms_hidden_width;
         col += block_threads * vector_width) {
        stage_rms_vector(branch + col, branch_values + col);
        if constexpr (HasResidual) {
            stage_rms_vector(residual + col, residual_values + col);
        }
        stage_rms_vector(weight + col, gains + col);
    }
    __syncthreads();

    constexpr int values_per_thread = rms_hidden_width / block_threads;
    float values[values_per_thread];
    float squares = 0.0f;
#pragma unroll
    for (int index = 0; index < values_per_thread; ++index) {
        const int col = threadIdx.x + index * block_threads;
        const float value = residual_sum(branch_values, HasResidual ? residual_values : nullptr,
                                         col);
        values[index] = value;
        squares = fmaf(value, value, squares);
    }
    const float inverse_rms = rms_inverse(squares, rms_hidden_width);
#pragma unroll
    for (int index = 0; index < values_per_thread; ++index) {
        const int col = threadIdx.x + index * block_threads;
        store_rms(values[index], inverse_rms, gains, col, output,
                  StoresResidual ? residual_output : nullptr);
    }
}

}

cudaError_t text_rms_norm(const __nv_bfloat16* branch, const __nv_bfloat16* residual,
                         const __nv_bfloat16* weight, int width, __nv_bfloat16* output,
                         __nv_bfloat16* residual_output, cudaStream_t stream) {
    if (branch == nullptr || weight == nullptr || output == nullptr || width <= 0 ||
        output == residual_output) {
        return cudaErrorInvalidValue;
    }
    if (width == rms_hidden_width && aligned(branch, alignof(uint4)) &&
        aligned(residual, alignof(uint4)) && aligned(weight, alignof(uint4))) {
        void* args[] = {&branch, &residual, &weight, &output, &residual_output};
        const bool store_residual = residual_output != nullptr;
        const auto kernel = residual == nullptr
            ? (store_residual ? rms_norm_vector_kernel<false, true>
                              : rms_norm_vector_kernel<false, false>)
            : (store_residual ? rms_norm_vector_kernel<true, true>
                              : rms_norm_vector_kernel<true, false>);
        return cudaLaunchKernel(kernel, 1, block_threads, args, 0, stream);
    }
    void* args[] = {&branch, &residual, &weight, &width, &output, &residual_output};
    return cudaLaunchKernel(rms_norm_kernel<false>, 1, block_threads, args, 0, stream);
}

cudaError_t text_rms_norm_rows(const __nv_bfloat16* branch, const __nv_bfloat16* residual,
                              const __nv_bfloat16* weight, int width, int rows,
                              __nv_bfloat16* output, __nv_bfloat16* residual_output,
                              cudaStream_t stream) {
    if (!branch || !weight || !output || width <= 0 || rows <= 0 || output == residual_output) {
        return cudaErrorInvalidValue;
    }
    void* args[] = {&branch, &residual, &weight, &width, &output, &residual_output};
    return cudaLaunchKernel(rms_norm_kernel<true>, rows, block_threads, args, 0, stream);
}

cudaError_t text_rms_norm_pair_rows(const __nv_bfloat16* first, const __nv_bfloat16* second,
                                   const __nv_bfloat16* first_weight,
                                   const __nv_bfloat16* second_weight, int width, int rows,
                                   __nv_bfloat16* output, cudaStream_t stream) {
    if (!first || !second || !first_weight || !second_weight || !output ||
        width <= 0 || rows <= 0) {
        return cudaErrorInvalidValue;
    }
    void* args[] = {&first, &second, &first_weight, &second_weight, &width, &output};
    return cudaLaunchKernel(rms_norm_pair_rows_kernel, dim3(rows, 2), block_threads, args, 0,
                            stream);
}

cudaError_t text_swiglu(const __nv_bfloat16* gate, const __nv_bfloat16* up, int width,
                         __nv_bfloat16* output, cudaStream_t stream) {
    if (gate == nullptr || up == nullptr || output == nullptr || width <= 0) {
        return cudaErrorInvalidValue;
    }
    void* args[] = {&gate, &up, &width, &output};
    return cudaLaunchKernel(swiglu_kernel, divide_up(width, block_threads), block_threads,
                            args, 0, stream);
}

cudaError_t text_argmax(const __nv_bfloat16* logits, int vocab_size, TextLogitCandidate* scratch,
                        TextLogitCandidate* output, cudaStream_t stream) {
    if (logits == nullptr || scratch == nullptr || output == nullptr || scratch == output ||
        vocab_size <= 0) {
        return cudaErrorInvalidValue;
    }
    int count = text_argmax_scratch_count(vocab_size);
    void* tiles_args[] = {&logits, &vocab_size, &scratch};
    const cudaError_t status = cudaLaunchKernel(argmax_tiles_kernel, count, block_threads,
                                                tiles_args, 0, stream);
    if (status != cudaSuccess) {
        return status;
    }
    void* finish_args[] = {&scratch, &count, &output};
    return cudaLaunchKernel(argmax_finish_kernel, 1, block_threads, finish_args, 0, stream);
}

}
