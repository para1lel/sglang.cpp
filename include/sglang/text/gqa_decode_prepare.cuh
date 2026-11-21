#pragma once

#include "sglang/mxfp8_gqa.h"

#include <cstddef>
#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_fp8.h>

namespace sglang::gqa_decode {

inline constexpr int prepare_warp_size = 32;
inline constexpr int rotary_dim = 64;
inline constexpr int rotary_half = rotary_dim / 2;
inline constexpr float rope_theta = 1.0e7f;
inline constexpr float norm_epsilon = 1.0e-6f;

static_assert(mxfp8_head_dim % prepare_warp_size == 0 && rotary_dim <= mxfp8_head_dim);

__device__ inline float gemma_norm(float value, float inverse_rms, __nv_bfloat16 weight) {
    return __bfloat162float(__float2bfloat16(value * inverse_rms
                                            * (1.0f + __bfloat162float(weight))));
}

template <typename PositionOf>
__device__ inline void normalize_rotate(const __nv_bfloat16* input, const __nv_bfloat16* norm,
                                        __nv_bfloat16* output, PositionOf position_of) {
    __shared__ float warp_squares[mxfp8_head_dim / prepare_warp_size];
    const int channel = threadIdx.x;
    const float value = __bfloat162float(input[channel]);
    float squares = value * value;
    for (int offset = prepare_warp_size / 2; offset > 0; offset /= 2) {
        squares += __shfl_xor_sync(0xffffffffu, squares, offset);
    }
    if (channel % prepare_warp_size == 0) {
        warp_squares[channel / prepare_warp_size] = squares;
    }
    __syncthreads();
    float sum = 0.0f;
    #pragma unroll
    for (int warp = 0; warp < mxfp8_head_dim / prepare_warp_size; ++warp) {
        sum += warp_squares[warp];
    }
    const float inverse_rms = rsqrtf(sum / mxfp8_head_dim + norm_epsilon);
    const float normalized = gemma_norm(value, inverse_rms, norm[channel]);
    if (channel < rotary_half) {
        const int other = channel + rotary_half;
        const float paired = gemma_norm(__bfloat162float(input[other]), inverse_rms, norm[other]);
        const float power = static_cast<float>(channel) / rotary_half;
        const float inverse_frequency = 1.0f / powf(rope_theta, power);
        const float position = static_cast<float>(position_of(channel));
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

__device__ inline float gate_attention(__nv_bfloat16 gate_input, __nv_bfloat16 attention) {
    const float gate = __bfloat162float(gate_input);
    const float value = __bfloat162float(attention);
    const float sigmoid = 1.0f / (1.0f + expf(-gate));
    return value * sigmoid;
}

__device__ inline std::size_t append_payload_offset(int physical_page, int token, int head) {
    return static_cast<std::size_t>(physical_page) * mxfp8_payload_bytes_per_page
         + (static_cast<std::size_t>(token) * mxfp8_kv_heads + head) * mxfp8_head_dim;
}

__device__ inline std::size_t append_scale_offset(int physical_page, int token, int head,
                                                  int group) {
    constexpr int token_groups = mxfp8_page_size / mxfp8_group_size;
    const auto page_head = static_cast<std::size_t>(physical_page) * mxfp8_kv_heads + head;
    const auto token_lane = page_head * mxfp8_group_size + token % mxfp8_group_size;
    return (token_lane * token_groups + token / mxfp8_group_size) * mxfp8_groups + group;
}

__device__ inline int append_scale_exponent(float value) {
    float amax = fabsf(value);
    for (int delta = mxfp8_group_size / 2; delta > 0; delta /= 2) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, delta));
    }
    int binary_exponent = 0;
    const float fraction = frexpf(fmaxf(amax, 1e-30f), &binary_exponent);
    const int biased = binary_exponent - 9 + (fraction > 0.875f) + 127;
    return min(254, max(0, biased));
}

__device__ inline std::uint8_t append_encode(float value, int exponent) {
    return __nv_cvt_float_to_fp8(ldexpf(value, 127 - exponent), __NV_SATFINITE, __NV_E4M3);
}

// Appends one channel of one token's K and V; every thread of a 256-thread block must call it.
__device__ inline void append_channel(const Mxfp8KvCache& cache, int physical_page, int token,
                                      int head, int channel, float key, float value) {
    const auto payload = append_payload_offset(physical_page, token, head) + channel;
    const int key_exponent = append_scale_exponent(key);
    const int value_exponent = append_scale_exponent(value);
    cache.key_payload[payload] = append_encode(key, key_exponent);
    cache.value_payload[payload] = append_encode(value, value_exponent);
    if (channel % mxfp8_group_size == 0) {
        const auto scale =
            append_scale_offset(physical_page, token, head, channel / mxfp8_group_size);
        cache.key_scales[scale] = static_cast<std::uint8_t>(key_exponent);
        cache.value_scales[scale] = static_cast<std::uint8_t>(value_exponent);
    }
}

}
