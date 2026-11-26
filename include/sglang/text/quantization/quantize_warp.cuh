#pragma once

#include "sglang/text_ops.h"
#include "sglang/text/text_cuda_primitives.cuh"

#include <cuda_fp8.h>

namespace sglang::text_kernel {

__device__ inline __nv_bfloat16 swiglu(__nv_bfloat16 gate, __nv_bfloat16 up) {
    const float value = __bfloat162float(gate);
    const auto activated = __float2bfloat16_rn(value / (1.0f + expf(-value)));
    return __float2bfloat16_rn(__bfloat162float(activated) * __bfloat162float(up));
}

__device__ __forceinline__ void quantize_warp_pairs(__nv_bfloat162 first, __nv_bfloat162 second,
                                                   int group, int offset, int lane,
                                                   TextFp8Activation output) {
    const float4 values{
        __bfloat162float(first.x), __bfloat162float(first.y),
        __bfloat162float(second.x), __bfloat162float(second.y),
    };
    const float maximum = fmaxf(fmaxf(fabsf(values.x), fabsf(values.y)),
                                fmaxf(fabsf(values.z), fabsf(values.w)));
    const float warp_maximum = warp_max(maximum);
    float reciprocal = 0.0f;
    if (lane == 0) {
        const float scale = __fdiv_rn(fmaxf(quant_epsilon, warp_maximum), fp8_max);
        output.scales[group] = scale;
        reciprocal = __fdiv_rn(1.0f, scale);
    }
    reciprocal = __shfl_sync(warp_mask, reciprocal, 0);
    const float4 scaled{values.x * reciprocal, values.y * reciprocal,
                        values.z * reciprocal, values.w * reciprocal};
    *reinterpret_cast<__nv_fp8x4_storage_t*>(output.values + offset) = __nv_fp8x4_e4m3(scaled).__x;
}

template <bool ApplySwiglu>
__device__ __forceinline__ void quantize_warp(const __nv_bfloat16* input,
                                             const __nv_bfloat16* up, int width,
                                             TextFp8Activation output) {
    static_assert(text_fp8_group_size == warp_width * 4);
    const int group = blockIdx.x * block_warps + threadIdx.x / warp_width;
    if (group >= width / text_fp8_group_size) {
        return;
    }
    const int lane = threadIdx.x % warp_width;
    const auto offset = group * text_fp8_group_size + lane * 4;
    const auto* pairs = reinterpret_cast<const __nv_bfloat162*>(input + offset);
    auto first = pairs[0];
    auto second = pairs[1];
    if constexpr (ApplySwiglu) {
        const auto* up_pairs = reinterpret_cast<const __nv_bfloat162*>(up + offset);
        const auto up_first = up_pairs[0];
        const auto up_second = up_pairs[1];
        first = {swiglu(first.x, up_first.x), swiglu(first.y, up_first.y)};
        second = {swiglu(second.x, up_second.x), swiglu(second.y, up_second.y)};
    }
    quantize_warp_pairs(first, second, group, offset, lane, output);
}

}
