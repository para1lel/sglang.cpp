#include "sglang/text_attention.h"

#include "sglang/decoder_cuda.cuh"
#include "sglang/text/gqa_verify_batch.h"
#include "sglang/text/mtp/draft_batch_ops.h"
#include "sglang/text/quantize_warp.cuh"

#include <cuda_runtime.h>

namespace sglang {
namespace {

constexpr int warp_size = 32;
constexpr int rotary_dim = 64;
constexpr int rotary_half = rotary_dim / 2;
constexpr float rope_theta = 1.0e7f;
constexpr float norm_epsilon = 1.0e-6f;

static_assert(mxfp8_head_dim % warp_size == 0 && rotary_dim <= mxfp8_head_dim);

__device__ float gemma_norm(float value, float inverse_rms, __nv_bfloat16 weight) {
    return __bfloat162float(__float2bfloat16(value * inverse_rms
                                            * (1.0f + __bfloat162float(weight))));
}

template <typename Position>
__device__ int rotary_position(Position first_position, std::size_t row, int) {
    return decoder::position_value(first_position) + static_cast<int>(row);
}

__device__ int rotary_position(const MropePosition* positions, std::size_t row, int channel) {
    const auto& position = positions[row];
    if (channel % 3 == 1) {
        return position.height;
    }
    if (channel % 3 == 2 && channel < 30) {
        return position.width;
    }
    return position.temporal;
}

template <bool IncludeQueries, typename Position>
__device__ __forceinline__ void gqa_preprocess_head(
    Position first_position, std::size_t row, int head, const __nv_bfloat16* q_gate,
    const __nv_bfloat16* keys, const __nv_bfloat16* q_norm, const __nv_bfloat16* k_norm,
    __nv_bfloat16* rope_queries, __nv_bfloat16* rope_keys) {
    constexpr int query_heads = IncludeQueries ? mxfp8_query_heads : 0;
    const int channel = threadIdx.x;
    const bool is_query = head < query_heads;
    const int kv_head = head - query_heads;
    const __nv_bfloat16* const input = is_query
        ? q_gate + row * text_gqa_q_gate_width + head * 2 * mxfp8_head_dim
        : keys + row * text_gqa_kv_width + kv_head * mxfp8_head_dim;
    const __nv_bfloat16* const norm = is_query ? q_norm : k_norm;
    __nv_bfloat16* const output = is_query
        ? rope_queries + row * text_gqa_q_width + head * mxfp8_head_dim
        : rope_keys + row * text_gqa_kv_width + kv_head * mxfp8_head_dim;

    __shared__ float warp_squares[mxfp8_head_dim / warp_size];
    const float value = __bfloat162float(input[channel]);
    float squares = value * value;
    for (int offset = warp_size / 2; offset > 0; offset /= 2) {
        squares += __shfl_xor_sync(0xffffffffu, squares, offset);
    }
    if (channel % warp_size == 0) {
        warp_squares[channel / warp_size] = squares;
    }
    __syncthreads();
    float sum = 0.0f;
    #pragma unroll
    for (int warp = 0; warp < mxfp8_head_dim / warp_size; ++warp) {
        sum += warp_squares[warp];
    }
    const float inverse_rms = rsqrtf(sum / mxfp8_head_dim + norm_epsilon);
    const float normalized = gemma_norm(value, inverse_rms, norm[channel]);
    if (channel < rotary_half) {
        const int other = channel + rotary_half;
        const float paired = gemma_norm(__bfloat162float(input[other]), inverse_rms, norm[other]);
        const float power = static_cast<float>(channel) / rotary_half;
        const float inverse_frequency = 1.0f / powf(rope_theta, power);
        const float position = static_cast<float>(rotary_position(first_position, row, channel));
        float sine;
        float cosine;
        sincosf(position * inverse_frequency, &sine, &cosine);
        sine = __bfloat162float(__float2bfloat16(sine));
        cosine = __bfloat162float(__float2bfloat16(cosine));
        output[channel] = __float2bfloat16(fmaf(normalized, cosine, -paired * sine));
        output[other] = __float2bfloat16(fmaf(paired, cosine, normalized * sine));
    } else if (channel >= rotary_dim) {
        output[channel] = __float2bfloat16(normalized);
    }
}

template <typename Position, bool IncludeQueries = true>
__global__ void gqa_preprocess(Position first_position, const __nv_bfloat16* q_gate,
                               const __nv_bfloat16* keys, const __nv_bfloat16* q_norm,
                               const __nv_bfloat16* k_norm, __nv_bfloat16* rope_queries,
                               __nv_bfloat16* rope_keys) {
    constexpr int head_count = (IncludeQueries ? mxfp8_query_heads : 0) + mxfp8_kv_heads;
    gqa_preprocess_head<IncludeQueries>(first_position, blockIdx.x / head_count,
        blockIdx.x % head_count, q_gate, keys, q_norm, k_norm, rope_queries, rope_keys);
}

template <typename Position>
__global__ void gqa_preprocess_queries(Position position, const __nv_bfloat16* q_gate,
                                       const __nv_bfloat16* norm, __nv_bfloat16* queries) {
    gqa_preprocess_head<true>(position, blockIdx.x / mxfp8_query_heads,
        blockIdx.x % mxfp8_query_heads, q_gate, nullptr, norm, nullptr, queries, nullptr);
}

template <int Capacity, bool IncludeQueries = true>
__global__ void gqa_verify_preprocess_kernel(
    const __grid_constant__ decoder::GqaVerifyRows<decoder::GqaVerifyPreprocessRow, Capacity> batch,
    const __nv_bfloat16* q_norm, const __nv_bfloat16* k_norm) {
    const auto& row = batch.rows[blockIdx.x];
    gqa_preprocess_head<IncludeQueries>(row.position, 0, blockIdx.y, row.q_gate, row.keys,
                                       q_norm, k_norm, row.rope_queries, row.rope_keys);
}

__device__ __forceinline__ __nv_bfloat16 gated_attention(__nv_bfloat16 gate,
                                                         __nv_bfloat16 attention) {
    const float value = __bfloat162float(attention);
    const float sigmoid = 1.0f / (1.0f + expf(-__bfloat162float(gate)));
    return __float2bfloat16(value * sigmoid);
}

__device__ __forceinline__ void gqa_postgate_head(
    std::size_t row_head, const __nv_bfloat16* q_gate, const __nv_bfloat16* attention,
    __nv_bfloat16* output) {
    const int channel = threadIdx.x;
    const auto gate = q_gate[(row_head * 2 + 1) * mxfp8_head_dim + channel];
    const std::size_t output_index = row_head * mxfp8_head_dim + channel;
    output[output_index] = gated_attention(gate, attention[output_index]);
}

__global__ void gqa_postgate(const __nv_bfloat16* q_gate, const __nv_bfloat16* attention,
                             __nv_bfloat16* output) {
    gqa_postgate_head(blockIdx.x, q_gate, attention, output);
}

struct GatedAttentionPairs {
    __nv_bfloat162 first;
    __nv_bfloat162 second;
};

__device__ __forceinline__ GatedAttentionPairs gqa_postgate_pairs(
    const __nv_bfloat16* q_gate, const __nv_bfloat16* attention, int offset) {
    const int head = offset / mxfp8_head_dim;
    const auto* gates = reinterpret_cast<const __nv_bfloat162*>(
        q_gate + offset + (head + 1) * mxfp8_head_dim);
    const auto* inputs = reinterpret_cast<const __nv_bfloat162*>(attention + offset);
    const auto gate_first = gates[0];
    const auto gate_second = gates[1];
    const auto first = inputs[0];
    const auto second = inputs[1];
    return {{gated_attention(gate_first.x, first.x), gated_attention(gate_first.y, first.y)},
            {gated_attention(gate_second.x, second.x), gated_attention(gate_second.y, second.y)}};
}

__global__ void gqa_postgate_quantize(const __nv_bfloat16* q_gate, const __nv_bfloat16* attention,
                                     TextFp8Activation output) {
    using namespace text_kernel;
    static_assert(text_fp8_group_size == warp_width * 4);
    static_assert(mxfp8_head_dim % text_fp8_group_size == 0);
    constexpr int groups_per_row = text_gqa_q_width / text_fp8_group_size;
    static_assert(groups_per_row % block_warps == 0);
    const int group = blockIdx.x * block_warps + threadIdx.x / warp_width;
    const int lane = threadIdx.x % warp_width;
    const int offset = group * text_fp8_group_size + lane * 4;
    const auto row = static_cast<std::size_t>(blockIdx.y);
    const auto [first, second] = gqa_postgate_pairs(q_gate + row * text_gqa_q_gate_width,
                                                   attention + row * text_gqa_q_width, offset);
    output.values += row * text_gqa_q_width;
    output.scales += row * groups_per_row;
    quantize_warp_pairs(first, second, group, offset, lane, output);
}

template <int Capacity>
__global__ void gqa_verify_postgate_kernel(
    const __grid_constant__ decoder::GqaVerifyRows<decoder::GqaVerifyGateRow, Capacity> batch) {
    const auto& row = batch.rows[blockIdx.x];
    gqa_postgate_head(blockIdx.y, row.q_gate, row.attention, row.attention);
}

}

template <int Capacity>
cudaError_t decoder::gqa_verify_preprocess(
    const GqaVerifyRows<GqaVerifyPreprocessRow, Capacity>& batch, int count,
    const __nv_bfloat16* q_norm, const __nv_bfloat16* k_norm, cudaStream_t stream) {
    if (count <= 0 || count > Capacity || !q_norm || !k_norm) {
        return cudaErrorInvalidValue;
    }
    for (int index = 0; index < count; ++index) {
        const auto& row = batch.rows[index];
        if (row.position < 0 || row.position >= text_attention_max_tokens || !row.q_gate ||
            !row.keys || !row.rope_queries || !row.rope_keys) {
            return cudaErrorInvalidValue;
        }
    }
    void* arguments[] = {const_cast<GqaVerifyRows<GqaVerifyPreprocessRow, Capacity>*>(&batch),
                         &q_norm, &k_norm};
    return cudaLaunchKernel(gqa_verify_preprocess_kernel<Capacity>,
                            dim3(count, mxfp8_query_heads + mxfp8_kv_heads),
                            mxfp8_head_dim, arguments, 0, stream);
}

cudaError_t decoder::draft_verify_preprocess_keys(
    const GqaVerifyRows<GqaVerifyPreprocessRow>& batch, int count,
    const __nv_bfloat16* k_norm, cudaStream_t stream) {
    if (count <= 0 || count > target_split_eager_max_rows || !k_norm) {
        return cudaErrorInvalidValue;
    }
    for (int index = 0; index < count; ++index) {
        const auto& row = batch.rows[index];
        if (row.position < 0 || row.position >= text_attention_max_tokens ||
            !row.keys || !row.rope_keys) return cudaErrorInvalidValue;
    }
    const __nv_bfloat16* q_norm = nullptr;
    void* args[] = {const_cast<GqaVerifyRows<GqaVerifyPreprocessRow>*>(&batch), &q_norm, &k_norm};
    return cudaLaunchKernel(gqa_verify_preprocess_kernel<target_split_eager_max_rows, false>,
                            dim3(count, mxfp8_kv_heads), mxfp8_head_dim, args, 0, stream);
}

template <int Capacity>
cudaError_t decoder::gqa_verify_postgate(const GqaVerifyRows<GqaVerifyGateRow, Capacity>& batch,
                                         int count, cudaStream_t stream) {
    if (count <= 0 || count > Capacity) {
        return cudaErrorInvalidValue;
    }
    for (int index = 0; index < count; ++index) {
        const auto& row = batch.rows[index];
        if (!row.q_gate || !row.attention) {
            return cudaErrorInvalidValue;
        }
    }
    void* arguments[] = {const_cast<GqaVerifyRows<GqaVerifyGateRow, Capacity>*>(&batch)};
    return cudaLaunchKernel(gqa_verify_postgate_kernel<Capacity>, dim3(count, mxfp8_query_heads),
                            mxfp8_head_dim, arguments, 0, stream);
}

template cudaError_t decoder::gqa_verify_preprocess(
    const GqaVerifyRows<GqaVerifyPreprocessRow>&, int, const __nv_bfloat16*,
    const __nv_bfloat16*, cudaStream_t);
template cudaError_t decoder::gqa_verify_preprocess(
    const GqaVerifyRows<GqaVerifyPreprocessRow, max_wide_mtp_rows>&, int, const __nv_bfloat16*,
    const __nv_bfloat16*, cudaStream_t);
template cudaError_t decoder::gqa_verify_postgate(
    const GqaVerifyRows<GqaVerifyGateRow>&, int, cudaStream_t);
template cudaError_t decoder::gqa_verify_postgate(
    const GqaVerifyRows<GqaVerifyGateRow, max_wide_mtp_rows>&, int, cudaStream_t);

cudaError_t text_gqa_preprocess(int first_position, int token_count,
                                const __nv_bfloat16* q_gate, const __nv_bfloat16* keys,
                                const __nv_bfloat16* q_norm, const __nv_bfloat16* k_norm,
                                __nv_bfloat16* rope_queries, __nv_bfloat16* rope_keys,
                                cudaStream_t stream) {
    if (token_count == 0) {
        return cudaSuccess;
    }
    if (token_count < 0 || token_count > text_attention_max_tokens || first_position < 0
        || first_position > text_attention_max_tokens - token_count
        || !q_gate || !keys || !q_norm || !k_norm || !rope_queries || !rope_keys) {
        return cudaErrorInvalidValue;
    }
    const int blocks = token_count * (mxfp8_query_heads + mxfp8_kv_heads);
    void* arguments[] = {
        &first_position, &q_gate, &keys, &q_norm, &k_norm, &rope_queries, &rope_keys};
    return cudaLaunchKernel(gqa_preprocess<int>, dim3(blocks), dim3(mxfp8_head_dim),
                            arguments, 0, stream);
}

cudaError_t text_gqa_preprocess_keys(int first_position, int token_count,
                                    const __nv_bfloat16* keys, const __nv_bfloat16* k_norm,
                                    __nv_bfloat16* rope_keys, cudaStream_t stream) {
    if (token_count <= 0 || token_count > text_attention_max_tokens || first_position < 0 ||
        first_position > text_attention_max_tokens - token_count ||
        !keys || !k_norm || !rope_keys) {
        return cudaErrorInvalidValue;
    }
    const __nv_bfloat16* unused_input = nullptr;
    __nv_bfloat16* unused_output = nullptr;
    void* args[] = {&first_position, &unused_input, &keys, &unused_input, &k_norm,
                    &unused_output, &rope_keys};
    return cudaLaunchKernel(gqa_preprocess<int, false>, token_count * mxfp8_kv_heads,
                            mxfp8_head_dim, args, 0, stream);
}

cudaError_t decoder::decode_gqa_preprocess(const int* position, const __nv_bfloat16* q_gate,
                                           const __nv_bfloat16* keys,
                                           const __nv_bfloat16* q_norm,
                                           const __nv_bfloat16* k_norm,
                                           __nv_bfloat16* rope_queries,
                                           __nv_bfloat16* rope_keys, cudaStream_t stream) {
    constexpr int blocks = mxfp8_query_heads + mxfp8_kv_heads;
    void* arguments[] = {
        &position, &q_gate, &keys, &q_norm, &k_norm, &rope_queries, &rope_keys};
    return cudaLaunchKernel(gqa_preprocess<const int*>, dim3(blocks), dim3(mxfp8_head_dim),
                            arguments, 0, stream);
}

cudaError_t decoder::mrope_gqa_preprocess(
    const MropePosition* positions, int rows, const __nv_bfloat16* q_gate,
    const __nv_bfloat16* keys, const __nv_bfloat16* q_norm, const __nv_bfloat16* k_norm,
    __nv_bfloat16* rope_queries, __nv_bfloat16* rope_keys, cudaStream_t stream) {
    if (rows <= 0 || rows > text_attention_max_tokens || !positions || !q_gate || !keys ||
        !q_norm || !k_norm || !rope_queries || !rope_keys) {
        return cudaErrorInvalidValue;
    }
    const int blocks = rows * (mxfp8_query_heads + mxfp8_kv_heads);
    void* arguments[] = {
        &positions, &q_gate, &keys, &q_norm, &k_norm, &rope_queries, &rope_keys};
    return cudaLaunchKernel(gqa_preprocess<const MropePosition*>, blocks, mxfp8_head_dim,
                            arguments, 0, stream);
}

cudaError_t decoder::prefill_gqa_queries(int position, const MropePosition* positions, int rows,
                                        const __nv_bfloat16* q_gate, const __nv_bfloat16* norm,
                                        __nv_bfloat16* queries, cudaStream_t stream) {
    if (rows <= 0 || rows > text_attention_max_tokens || !q_gate || !norm || !queries ||
        (!positions && (position < 0 || position > text_attention_max_tokens - rows))) {
        return cudaErrorInvalidValue;
    }
    const auto launch = [&](auto first) {
        void* args[] = {&first, &q_gate, &norm, &queries};
        return cudaLaunchKernel(gqa_preprocess_queries<decltype(first)>, rows * mxfp8_query_heads,
                                mxfp8_head_dim, args, 0, stream);
    };
    return positions ? launch(positions) : launch(position);
}

cudaError_t decoder::mrope_gqa_preprocess_keys(
    const MropePosition* positions, int rows, const __nv_bfloat16* keys,
    const __nv_bfloat16* k_norm, __nv_bfloat16* rope_keys, cudaStream_t stream) {
    if (rows <= 0 || rows > text_attention_max_tokens || !positions || !keys || !k_norm ||
        !rope_keys) {
        return cudaErrorInvalidValue;
    }
    const __nv_bfloat16* unused_input = nullptr;
    __nv_bfloat16* unused_output = nullptr;
    void* arguments[] = {&positions, &unused_input, &keys, &unused_input, &k_norm,
                         &unused_output, &rope_keys};
    return cudaLaunchKernel(gqa_preprocess<const MropePosition*, false>,
                            rows * mxfp8_kv_heads, mxfp8_head_dim, arguments, 0, stream);
}

cudaError_t text_gqa_postgate(int token_count, const __nv_bfloat16* q_gate,
                              const __nv_bfloat16* attention, __nv_bfloat16* output,
                              cudaStream_t stream) {
    if (token_count == 0) {
        return cudaSuccess;
    }
    if (token_count < 0 || token_count > text_attention_max_tokens
        || !q_gate || !attention || !output) {
        return cudaErrorInvalidValue;
    }
    void* arguments[] = {&q_gate, &attention, &output};
    return cudaLaunchKernel(gqa_postgate, dim3(token_count * mxfp8_query_heads),
                            dim3(mxfp8_head_dim), arguments, 0, stream);
}

std::optional<cudaError_t> decoder::try_prefill_gqa_postgate_quantize(
    int rows, const __nv_bfloat16* q_gate, const __nv_bfloat16* attention,
    TextFp8Activation activation, cudaStream_t stream) {
    using namespace text_kernel;
    if (rows != 4096 || !q_gate || !attention || !activation.values || !activation.scales ||
        !aligned(q_gate, alignof(__nv_bfloat162)) || !aligned(attention, alignof(__nv_bfloat162)) ||
        !aligned(activation.values, alignof(__nv_fp8x4_storage_t)) ||
        !aligned(activation.scales, alignof(float))) {
        return std::nullopt;
    }
    const auto pending = cudaGetLastError();
    if (pending != cudaSuccess) return pending;
    constexpr int blocks = text_gqa_q_width / (text_fp8_group_size * block_warps);
    void* args[] = {&q_gate, &attention, &activation};
    const auto status = cudaLaunchKernel(gqa_postgate_quantize, dim3(blocks, rows), block_threads,
                                         args, 0, stream);
    const auto recorded = cudaGetLastError();
    return status == cudaSuccess ? recorded : status;
}

}
