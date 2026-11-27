#include "sglang/text_ops.h"
#include "sglang/decoder_cuda.cuh"
#include "sglang/decoder_internal.h"
#include "sglang/decoder_mtp.h"
#include "sglang/text/decode_proj.h"
#include "sglang/text/text_cuda_primitives.cuh"
#include "sglang/text/quantization/quantize_warp.cuh"

#include <algorithm>
#include <cstddef>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <math_constants.h>

namespace sglang {
namespace {
using namespace text_kernel;
constexpr int nonfinite_token = -2;

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
    const unsigned int start = blockIdx.x * text_argmax_tile;
    TextLogitCandidate best{-CUDART_INF_F, -1};
    for (unsigned int item = threadIdx.x; item < text_argmax_tile; item += block_threads) {
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

__host__ __device__ __inline_hint__ int decoder::compact_gdn_row(int lane, int row) {
    constexpr int packet_lanes = GdnVerificationBatch::lanes;
    constexpr int packet_rows = packet_lanes * GdnVerificationBatch::rows;
    return (lane / packet_lanes) * packet_rows + row * packet_lanes + lane % packet_lanes;
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
    const bool exact = (count == decoder::wide_mtp_lanes || count == decoder::draft_projection_rows)
        && out_width == decoder::hidden_size && in_width == 2 * decoder::hidden_size;
    if (!weight || !projections || out_width <= 0 || in_width <= 0 ||
        count < 1 || (count > text_fp8_max_batch && !exact)) {
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
        case decoder::draft_projection_rows:
            return launch_bf16_batch<decoder::draft_projection_rows>(weight, out_width, in_width,
                                                                     projections, stream);
        case decoder::wide_mtp_lanes:
            return launch_bf16_batch<decoder::wide_mtp_lanes>(weight, out_width, in_width,
                                                             projections, stream);
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
    if (rows == 1024 &&
        (width == text_gqa_q_width || width == text_gqa_q_width / 2) &&
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

cudaError_t text_bf16_decode_rows(const __nv_bfloat16* weight, const __nv_bfloat16* input,
                                  int out_width, int in_width, int rows, __nv_bfloat16* output,
                                  cudaStream_t stream) {
    if (!weight || !input || !output || out_width <= 0 || in_width <= 0 ||
        rows <= 0 || rows > 65535) {
        return cudaErrorInvalidValue;
    }
    if ((out_width == text_gdn_value_heads || out_width == text_gdn_value_heads / 2) &&
        in_width == decoder::hidden_size) {
        if (const auto status = try_text_bf16_decode_gates(weight, nullptr, input, rows,
                                                          output, nullptr, out_width, stream)) {
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
                                                       output_b, output_a,
                                                       text_gdn_value_heads, stream)) {
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
