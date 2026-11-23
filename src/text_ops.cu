#include "sglang/text_ops.h"

#include "sglang/decoder_cuda.cuh"
#include "sglang/decoder_internal.h"

#include "sglang/text/decode_proj.h"
#include "sglang/text/text_cuda_primitives.cuh"
#include "sglang/text/quantize_warp.cuh"

#include <algorithm>
#include <cstddef>

#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <math_constants.h>

namespace sglang {
namespace {

using namespace text_kernel;

constexpr int argmax_tile = 4 * block_threads;
constexpr int nonfinite_token = -2;
static_assert(text_argmax_scratch_count(argmax_tile) == 1);
static_assert(text_argmax_scratch_count(argmax_tile + 1) == 2);

constexpr int divide_up(int count, int tile) {
    return count / tile + (count % tile != 0);
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

__global__ void set_decode_position_kernel(int* position, int value, int* rotary,
                                          int rotary_value) {
    *position = value;
    if (rotary) *rotary = rotary_value;
}

__global__ void advance_decode_position_kernel(int* position, int* rotary) {
    ++*position;
    ++*rotary;
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

template <bool Batched, bool ApplySwiglu = false>
__global__ void quantize_fp8_kernel(const __nv_bfloat16* input, int width,
                                    TextFp8Activation output, const __nv_bfloat16* up) {
    constexpr int group_warps = text_fp8_group_size / warp_width;
    __shared__ float warp_maxima[group_warps];
    __shared__ float reciprocal_scale;
    if constexpr (Batched) {
        const auto offset = static_cast<std::size_t>(blockIdx.y) * width;
        input += offset;
        if constexpr (ApplySwiglu) {
            up += offset;
        }
        output.values += offset;
        output.scales += static_cast<std::size_t>(blockIdx.y) * gridDim.x;
    }
    const int lane = threadIdx.x % warp_width;
    const int warp = threadIdx.x / warp_width;
    const auto col = static_cast<std::size_t>(blockIdx.x) * text_fp8_group_size + threadIdx.x;
    const bool valid = col < static_cast<std::size_t>(width);
    float value = 0.0f;
    if (valid) {
        const auto element = ApplySwiglu ? swiglu(input[col], up[col]) : input[col];
        value = __bfloat162float(element);
    }
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

template <bool ApplySwiglu>
__global__ void prefill_quantize_warp(const __nv_bfloat16* input, const __nv_bfloat16* up,
                                    int width, TextFp8Activation output) {
    const auto offset = static_cast<std::size_t>(blockIdx.y) * width;
    input += offset;
    if constexpr (ApplySwiglu) {
        up += offset;
    }
    output.values += offset;
    output.scales += offset / text_fp8_group_size;
    quantize_warp<ApplySwiglu>(input, up, width, output);
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

template <int Count, int Channels = 1>
__global__ void fp8_batch_packed_kernel(const std::uint8_t* weight,
                                        const __nv_bfloat16* weight_scales,
                                        int out_width, int in_width, int groups,
                                        Fp8Projections<Count> projections) {
    static_assert(text_fp8_group_size % Channels == 0);
    const int lane = threadIdx.x % warp_width;
    const int row = Channels * (blockIdx.x * fp8_packed_warps + threadIdx.x / warp_width);
    if (row >= out_width) {
        return;
    }
    const auto* row_weight = weight + static_cast<std::size_t>(row) * in_width;
    const auto* row_scales =
        weight_scales + static_cast<std::size_t>(row / text_fp8_group_size) * groups;
    constexpr int lanes_per_group = warp_width / fp8_groups_per_warp;
    const int lane_group = lane / lanes_per_group;
    const int lane_col = lane * fp8_pack_bytes;
    float sums[Count][Channels]{};
    for (int group = 0; group < groups; group += fp8_groups_per_warp) {
        const auto col = static_cast<std::size_t>(group) * text_fp8_group_size + lane_col;
        uint4 lhs[Channels];
#pragma unroll
        for (int channel = 0; channel < Channels; ++channel) {
            lhs[channel] = load_fp8_weight(row_weight + col +
                                           static_cast<std::size_t>(channel) * in_width);
        }
        const int scale_index = group + lane_group;
        const float scale = __bfloat162float(row_scales[scale_index]);
#pragma unroll
        for (int index = 0; index < Count; ++index) {
            const auto input = projections.entries[index].input;
            const uint4 rhs = load_128(input.values + col);
            const float input_scale = input.scales[scale_index];
#pragma unroll
            for (int channel = 0; channel < Channels; ++channel) {
                const auto left = lhs[channel];
                const float low = fp8x4_dot(left.y, rhs.y, fp8x4_dot(left.x, rhs.x, 0.0f));
                const float high = fp8x4_dot(left.w, rhs.w, fp8x4_dot(left.z, rhs.z, 0.0f));
                const float scaled = (low + high) * input_scale;
                sums[index][channel] = fmaf(scaled, scale, sums[index][channel]);
            }
        }
    }
#pragma unroll
    for (int index = 0; index < Count; ++index) {
#pragma unroll
        for (int channel = 0; channel < Channels; ++channel) {
            const float sum = warp_sum(sums[index][channel]);
            if (lane == 0) {
                projections.entries[index].output[row + channel] = __float2bfloat16_rn(sum);
            }
        }
    }
}

template <int Count, int Channels = 1>
cudaError_t launch_fp8_batch(const std::uint8_t* weight, const __nv_bfloat16* weight_scales,
                             int out_width, int in_width, const TextFp8Projection* projections,
                             cudaStream_t stream) {
    Fp8Projections<Count> batch;
    for (int index = 0; index < Count; ++index) {
        batch.entries[index] = projections[index];
    }
    int groups = text_fp8_group_count(in_width);
    void* args[] = {&weight, &weight_scales, &out_width, &in_width, &groups, &batch};
    return cudaLaunchKernel(fp8_batch_packed_kernel<Count, Channels>,
                            divide_up(out_width, fp8_packed_warps * Channels),
                            fp8_packed_warps * warp_width, args, 0, stream);
}

#include "sglang/text/prefill_fp8_exact.cuh"
#include "sglang/text/fp8_mlp_verify.cuh"

__global__ void swiglu_kernel(const __nv_bfloat16* gate, const __nv_bfloat16* up, int width,
                              __nv_bfloat16* output) {
    const auto col = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (col < static_cast<std::size_t>(width)) {
        output[col] = swiglu(gate[col], up[col]);
    }
}

template <bool RequireFinite>
__device__ TextLogitCandidate best_candidate(TextLogitCandidate first,
                                             TextLogitCandidate second) {
    if constexpr (RequireFinite) {
        if (first.token_id == nonfinite_token || second.token_id == nonfinite_token) {
            return {-CUDART_INF_F, nonfinite_token};
        }
    }
    if (second.token_id >= 0 &&
        (first.token_id < 0 || second.score > first.score ||
         (second.score == first.score && second.token_id < first.token_id))) {
        return second;
    }
    return first;
}

template <bool RequireFinite>
__device__ TextLogitCandidate warp_best(TextLogitCandidate best) {
    for (int offset = warp_width / 2; offset > 0; offset /= 2) {
        const TextLogitCandidate other{
            __shfl_down_sync(warp_mask, best.score, offset),
            __shfl_down_sync(warp_mask, best.token_id, offset)};
        best = best_candidate<RequireFinite>(best, other);
    }
    return best;
}

template <bool RequireFinite>
__device__ void store_block_best(TextLogitCandidate best, TextLogitCandidate* output) {
    __shared__ TextLogitCandidate warp_candidates[block_warps];
    const int lane = threadIdx.x % warp_width;
    const int warp = threadIdx.x / warp_width;
    best = warp_best<RequireFinite>(best);
    if (lane == 0) {
        warp_candidates[warp] = best;
    }
    __syncthreads();
    if (warp == 0) {
        best = warp_best<RequireFinite>(lane < block_warps ? warp_candidates[lane]
                                           : TextLogitCandidate{-CUDART_INF_F, -1});
        if (lane == 0) {
            *output = best;
        }
    }
}

template <typename Score, bool RequireFinite>
__device__ void argmax_tile_best(const Score* logits, int vocab_size,
                                 TextLogitCandidate* scratch) {
    const unsigned int start = blockIdx.x * argmax_tile;
    TextLogitCandidate best{-CUDART_INF_F, -1};
    for (unsigned int item = threadIdx.x; item < argmax_tile; item += block_threads) {
        const unsigned int token = start + item;
        if (token < static_cast<unsigned int>(vocab_size)) {
            const float score = static_cast<float>(logits[token]);
            if constexpr (RequireFinite) {
                if (!isfinite(score)) {
                    best = {-CUDART_INF_F, nonfinite_token};
                    break;
                }
            }
            if (!isnan(score)) {
                best = best_candidate<RequireFinite>(best, {score, static_cast<int>(token)});
            }
        }
    }
    store_block_best<RequireFinite>(best, scratch + blockIdx.x);
}

template <bool RequireFinite>
__device__ void argmax_finish_best(const TextLogitCandidate* scratch, int count,
                                   TextLogitCandidate* output) {
    TextLogitCandidate best{-CUDART_INF_F, -1};
    for (int item = threadIdx.x; item < count; item += block_threads) {
        best = best_candidate<RequireFinite>(best, scratch[item]);
    }
    store_block_best<RequireFinite>(best, output);
}

template <typename Score, bool RequireFinite>
__global__ void argmax_tiles_kernel(const Score* logits, int vocab_size,
                                    TextLogitCandidate* scratch) {
    argmax_tile_best<Score, RequireFinite>(logits, vocab_size, scratch);
}

template <bool RequireFinite>
__global__ void argmax_finish_kernel(const TextLogitCandidate* scratch, int count,
                                     TextLogitCandidate* output) {
    argmax_finish_best<RequireFinite>(scratch, count, output);
}

template <bool RequireFinite>
__global__ void draft_argmax_tiles(const __grid_constant__ decoder::DraftArgmaxRows rows,
                                   int width) {
    const auto& row = rows.rows[blockIdx.y];
    argmax_tile_best<__nv_bfloat16, RequireFinite>(row.logits, width, row.scratch);
}

template <bool RequireFinite>
__global__ void draft_argmax_finish(const __grid_constant__ decoder::DraftArgmaxRows rows,
                                    int tiles) {
    const auto& row = rows.rows[blockIdx.x];
    argmax_finish_best<RequireFinite>(row.scratch, tiles, row.winner);
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

cudaError_t decoder::set_decode_position(int* position, int value, cudaStream_t stream,
                                         int* rotary, int rotary_value) {
    void* arguments[] = {&position, &value, &rotary, &rotary_value};
    return cudaLaunchKernel(set_decode_position_kernel, 1, 1, arguments, 0, stream);
}

cudaError_t decoder::advance_decode_position(int* position, int* rotary, cudaStream_t stream) {
    void* arguments[] = {&position, &rotary};
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
    const __nv_bfloat16* up = nullptr;
    void* args[] = {&input, &width, &output, &up};
    return cudaLaunchKernel(quantize_fp8_kernel<false>, text_fp8_group_count(width),
                            text_fp8_group_size, args, 0, stream);
}

cudaError_t text_quantize_fp8_rows(const __nv_bfloat16* input, int width, int rows,
                                  TextFp8Activation output, cudaStream_t stream) {
    if (!input || !output.values || !output.scales || width <= 0 || rows <= 0 || rows > 65535) {
        return cudaErrorInvalidValue;
    }
    const __nv_bfloat16* up = nullptr;
    if (rows == 1024 && width == text_gqa_q_width &&
        aligned(input, alignof(__nv_bfloat162)) &&
        aligned(output.values, alignof(__nv_fp8x4_storage_t))) {
        const dim3 grid(divide_up(width / text_fp8_group_size, block_warps), rows);
        void* warp_args[] = {&input, &up, &width, &output};
        return cudaLaunchKernel(prefill_quantize_warp<false>, grid, block_threads, warp_args,
                                0, stream);
    }
    void* args[] = {&input, &width, &output, &up};
    return cudaLaunchKernel(quantize_fp8_kernel<true>, dim3(text_fp8_group_count(width), rows),
                            text_fp8_group_size, args, 0, stream);
}

cudaError_t decoder::tensor_core_prefill_swiglu_quantize(
    const __nv_bfloat16* gate, const __nv_bfloat16* up, int width, int rows,
    TextFp8Activation activation, cudaStream_t stream) {
    void* args[] = {&gate, &width, &activation, &up};
    if (rows >= 256 && width % text_fp8_group_size == 0 &&
        aligned(gate, alignof(__nv_bfloat162)) && aligned(up, alignof(__nv_bfloat162)) &&
        aligned(activation.values, alignof(__nv_fp8x4_storage_t))) {
        const dim3 grid(divide_up(width / text_fp8_group_size, block_warps), rows);
        void* warp_args[] = {&gate, &up, &width, &activation};
        return cudaLaunchKernel(prefill_quantize_warp<true>, grid, block_threads,
                                warp_args, 0, stream);
    }
    const dim3 grid(text_fp8_group_count(width), rows);
    return cudaLaunchKernel(quantize_fp8_kernel<true, true>, grid,
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
    if (wide && count == mlp_verify_rows) {
        if (is_fp8_mlp_or_gdn_shape(out_width, in_width)) {
            return launch_fp8_mlp_staged(weight, weight_scales, out_width, in_width, projections,
                                        stream);
        }
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

cudaError_t decoder::mtp_seed_project_kv(const Fp8Weight& key, const Fp8Weight& value,
                                        TextFp8ActivationView input, int rows,
                                        __nv_bfloat16* keys, __nv_bfloat16* values,
                                        cudaStream_t stream) {
    constexpr int tile_rows = 4;
    constexpr int channels = 8;
    constexpr int threads = fp8_packed_warps * warp_width;
    static_assert(text_gqa_kv_width % (fp8_packed_warps * channels) == 0);
    static_assert(hidden_size % text_fp8_group_size == 0);
    if (rows < 2 || rows > text_attention_max_tokens || !input.values || !input.scales ||
        !key.values || !value.values || !key.scales || !value.scales || !keys || !values ||
        key.rows != text_gqa_kv_width || value.rows != text_gqa_kv_width ||
        key.columns != hidden_size || value.columns != hidden_size ||
        !supports_wide_fp8(key.values, input, hidden_size) ||
        !supports_wide_fp8(value.values, input, hidden_size)) {
        return cudaErrorInvalidValue;
    }
    int out_width = text_gqa_kv_width;
    int in_width = hidden_size;
    int groups = text_fp8_group_count(in_width);
    Fp8PrefillMatrix key_matrix{key.values, key.scales, keys};
    Fp8PrefillMatrix value_matrix{value.values, value.scales, values};
    void* args[] = {&key_matrix, &value_matrix, &input, &out_width, &in_width, &groups, &rows};
    const dim3 grid(divide_up(rows, tile_rows), out_width / (fp8_packed_warps * channels), 2);
    return cudaLaunchKernel(fp8_prefill_reuse_pair_kernel<tile_rows, channels>, grid, threads,
                            args, 0, stream);
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

template <bool RequireFinite = false, typename Score>
cudaError_t launch_argmax(const Score* logits, int vocab_size, TextLogitCandidate* scratch,
                          TextLogitCandidate* output, cudaStream_t stream) {
    if (logits == nullptr || scratch == nullptr || output == nullptr || scratch == output ||
        vocab_size <= 0) {
        return cudaErrorInvalidValue;
    }
    int count = text_argmax_scratch_count(vocab_size);
    void* tiles_args[] = {&logits, &vocab_size, &scratch};
    const cudaError_t status = cudaLaunchKernel(argmax_tiles_kernel<Score, RequireFinite>,
                                                count, block_threads, tiles_args, 0, stream);
    if (status != cudaSuccess) {
        return status;
    }
    void* finish_args[] = {&scratch, &count, &output};
    return cudaLaunchKernel(argmax_finish_kernel<RequireFinite>, 1, block_threads, finish_args,
                            0, stream);
}

cudaError_t text_argmax(const __nv_bfloat16* logits, int vocab_size, TextLogitCandidate* scratch,
                        TextLogitCandidate* output, cudaStream_t stream) {
    return launch_argmax(logits, vocab_size, scratch, output, stream);
}

cudaError_t text_argmax(const float* logits, int vocab_size, TextLogitCandidate* scratch,
                        TextLogitCandidate* output, cudaStream_t stream) {
    return launch_argmax(logits, vocab_size, scratch, output, stream);
}

cudaError_t text_argmax_finite(const __nv_bfloat16* logits, int vocab_size,
                               TextLogitCandidate* scratch, TextLogitCandidate* output,
                               cudaStream_t stream) {
    return launch_argmax<true>(logits, vocab_size, scratch, output, stream);
}

cudaError_t decoder::draft_argmax_wave(DraftArgmaxRows rows, int count, int width, bool finite,
                                       cudaStream_t stream) {
    if (count < 2 || count > draft_wave_rows || width <= 0) return cudaErrorInvalidValue;
    for (int index = 0; index < count; ++index) {
        const auto& row = rows.rows[index];
        if (!row.logits || !row.scratch || !row.winner || row.scratch == row.winner) {
            return cudaErrorInvalidValue;
        }
    }
    int tiles = text_argmax_scratch_count(width);
    void* tiles_args[] = {&rows, &width};
    const auto tile_kernel = finite ? draft_argmax_tiles<true> : draft_argmax_tiles<false>;
    const auto status = cudaLaunchKernel(tile_kernel, dim3(tiles, count), block_threads,
                                         tiles_args, 0, stream);
    if (status != cudaSuccess) return status;
    void* finish_args[] = {&rows, &tiles};
    const auto finish_kernel = finite ? draft_argmax_finish<true> : draft_argmax_finish<false>;
    return cudaLaunchKernel(finish_kernel, count, block_threads, finish_args, 0, stream);
}

cudaError_t text_fp8_decode_rows(const std::uint8_t* weight, const __nv_bfloat16* weight_scales,
                                 TextFp8ActivationView input, int out_width, int in_width,
                                 int rows, __nv_bfloat16* output, cudaStream_t stream) {
    if (!weight || !weight_scales || !input.values || !input.scales || !output ||
        out_width <= 0 || in_width <= 0 || rows <= 0 || rows > 65535) {
        return cudaErrorInvalidValue;
    }
    const TextFp8DecodeTarget target{weight, weight_scales, out_width, output};
    return text_fp8_decode_rows_fused(&target, 1, input, in_width, rows, stream);
}

namespace {

cudaError_t decode_fp8_rows(const TextFp8DecodeTarget* targets, int count,
                            TextFp8ActivationView input, int in_width, int rows,
                            cudaStream_t stream, const Fp8DecodeInputProducer* producer) {
    if (!targets || count < 1 || count > text_decode_max_targets || !input.values ||
        !input.scales || in_width <= 0 || rows <= 0 || rows > 65535) {
        return cudaErrorInvalidValue;
    }
    for (int index = 0; index < count; ++index) {
        const auto& target = targets[index];
        if (!target.weight || !target.weight_scales || !target.output || target.out_width <= 0) {
            return cudaErrorInvalidValue;
        }
    }
    if (const auto status = try_text_fp8_decode_sm100(targets, count, input, in_width, rows,
                                                      stream, producer)) {
        return *status;
    }
    if (producer) {
        const auto status = producer->launch(producer->context, false);
        if (status != cudaSuccess) {
            return status;
        }
    }
    for (int index = 0; index < count; ++index) {
        const auto& target = targets[index];
        const auto status = text_fp8_gemv_rows(target.weight, target.weight_scales, input,
                                               target.out_width, in_width, rows, target.output,
                                               stream);
        if (status != cudaSuccess) {
            return status;
        }
    }
    return cudaSuccess;
}

}

cudaError_t text_fp8_decode_rows_fused(const TextFp8DecodeTarget* targets, int count,
                                       TextFp8ActivationView input, int in_width, int rows,
                                       cudaStream_t stream) {
    return decode_fp8_rows(targets, count, input, in_width, rows, stream, nullptr);
}

cudaError_t decoder::tensor_core_decode_rms_project(
    const __nv_bfloat16* branch, __nv_bfloat16* residual, const __nv_bfloat16* weight,
    int width, int rows, __nv_bfloat16* normalized, TextFp8Activation activation,
    const TextFp8DecodeTarget* targets, int count, cudaStream_t stream) {
    const auto normalize = [&](bool prefetch_weights) {
        const auto launch = prefetch_weights ? RmsQuantizeLaunch::prefetch_decode_weights
                                             : RmsQuantizeLaunch::ordinary;
        return tensor_core_prefill_rms_quantize(branch, residual, weight, width, rows,
                                                normalized, activation, stream, launch);
    };
    const Fp8DecodeInputProducer producer{
        &normalize, [](const void* context, bool prefetch_weights) {
            return (*static_cast<const decltype(normalize)*>(context))(prefetch_weights);
        }};
    return decode_fp8_rows(targets, count, {activation.values, activation.scales}, width, rows,
                           stream, &producer);
}

cudaError_t text_bf16_decode_rows(const __nv_bfloat16* weight, const __nv_bfloat16* input,
                                  int out_width, int in_width, int rows, __nv_bfloat16* output,
                                  cudaStream_t stream) {
    if (!weight || !input || !output || out_width <= 0 || in_width <= 0 ||
        rows <= 0 || rows > 65535) {
        return cudaErrorInvalidValue;
    }
    if (out_width == text_gdn_value_heads && in_width == decoder::hidden_size) {
        if (const auto status = try_text_bf16_decode_gates(weight, nullptr, input, rows,
                                                          output, nullptr, stream)) {
            return *status;
        }
    }
    if (const auto status = try_text_bf16_decode_sm100(weight, input, out_width, in_width, rows,
                                                       output, stream)) {
        return *status;
    }
    return text_bf16_gemv_rows(weight, input, out_width, in_width, rows, output, stream);
}

cudaError_t text_bf16_decode_gates(const __nv_bfloat16* weight_b,
                                   const __nv_bfloat16* weight_a,
                                   const __nv_bfloat16* input, int rows,
                                   __nv_bfloat16* output_b, __nv_bfloat16* output_a,
                                   cudaStream_t stream) {
    if (!weight_b || !weight_a || !input || !output_b || !output_a ||
        rows <= 0 || rows > 64) {
        return cudaErrorInvalidValue;
    }
    if (const auto status = try_text_bf16_decode_gates(weight_b, weight_a, input, rows,
                                                       output_b, output_a, stream)) {
        return *status;
    }
    const auto status = text_bf16_decode_rows(weight_b, input, text_gdn_value_heads,
                                              decoder::hidden_size, rows, output_b, stream);
    return status == cudaSuccess
        ? text_bf16_decode_rows(weight_a, input, text_gdn_value_heads, decoder::hidden_size,
                                 rows, output_a, stream)
        : status;
}

}
