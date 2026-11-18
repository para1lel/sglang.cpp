#include "sglang/text_attention.h"

#include "sglang/decoder_cuda.cuh"

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
__global__ void gqa_preprocess(Position first_position, const __nv_bfloat16* q_gate,
                               const __nv_bfloat16* keys, const __nv_bfloat16* q_norm,
                               const __nv_bfloat16* k_norm, __nv_bfloat16* rope_queries,
                               __nv_bfloat16* rope_keys) {
    constexpr int head_count = mxfp8_query_heads + mxfp8_kv_heads;
    const std::size_t row = blockIdx.x / head_count;
    const int head = blockIdx.x % head_count;
    const int channel = threadIdx.x;
    const bool is_query = head < mxfp8_query_heads;
    const int kv_head = head - mxfp8_query_heads;
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
        const float position =
            static_cast<float>(decoder::position_value(first_position) + static_cast<int>(row));
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

__global__ void gqa_postgate(const __nv_bfloat16* q_gate, const __nv_bfloat16* attention,
                             __nv_bfloat16* output) {
    const std::size_t row_head = blockIdx.x;
    const int channel = threadIdx.x;
    const float gate =
        __bfloat162float(q_gate[(row_head * 2 + 1) * mxfp8_head_dim + channel]);
    const std::size_t output_index = row_head * mxfp8_head_dim + channel;
    const float value = __bfloat162float(attention[output_index]);
    const float sigmoid = 1.0f / (1.0f + expf(-gate));
    output[output_index] = __float2bfloat16(value * sigmoid);
}

}

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

}
