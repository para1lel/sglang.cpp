#include "sglang/text/quantization/rms_quantize.cuh"

#if defined(__CUDA_ARCH__) || !defined(SGLANG_TEXT_DEVICE_ONLY)

namespace sglang::text_kernel {

__device__ __inline_hint__ float residual_sum(const __nv_bfloat16* branch,
                                             const __nv_bfloat16* residual, std::size_t col) {
    const float value = __bfloat162float(branch[col]);
    return residual == nullptr ? value : value + __bfloat162float(residual[col]);
}

__device__ __inline_hint__ void stage_rms_vector_async(const __nv_bfloat16* input,
                                                      __nv_bfloat16* shared) {
    const auto address = static_cast<unsigned int>(__cvta_generic_to_shared(shared));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;"
                 :: "r"(address), "l"(input) : "memory");
}

__device__ __inline_hint__ float rms_inverse(float squares, int width) {
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

__device__ __inline_hint__ float4 rms_quant_values(uint2 bits) {
    return {__uint_as_float(bits.x << 16), __uint_as_float(bits.x & 0xffff0000u),
            __uint_as_float(bits.y << 16), __uint_as_float(bits.y & 0xffff0000u)};
}

__device__ __inline_hint__ float rms_quant_maximum(float4 values) {
    return fmaxf(fmaxf(fabsf(values.x), fabsf(values.y)),
                 fmaxf(fabsf(values.z), fabsf(values.w)));
}

__device__ __inline_hint__ __nv_fp8x4_storage_t rms_quant_pack(float4 values, float reciprocal) {
    const float4 scaled{values.x * reciprocal, values.y * reciprocal,
                        values.z * reciprocal, values.w * reciprocal};
    return __nv_fp8x4_e4m3(scaled).__x;
}

static __device__ __forceinline__ void rms_quantize_subwarps(const __nv_bfloat16* normalized,
                                      TextFp8Activation activation) {
    constexpr int lanes = 8;
    constexpr int half_group = text_fp8_group_size / 2;
    constexpr int groups = rms_hidden_width / text_fp8_group_size;
    constexpr int values_per_lane = half_group / lanes;
    constexpr int groups_per_block = block_threads / lanes;
    static_assert(values_per_lane == 8);
    static_assert(warp_width % lanes == 0 && groups % (warp_width / lanes) == 0);
    const int lane = threadIdx.x % lanes;
    for (int group = threadIdx.x / lanes; group < groups; group += groups_per_block) {
        const int col = group * text_fp8_group_size + lane * values_per_lane;
        const auto lower = *reinterpret_cast<const uint4*>(normalized + col);
        const auto upper = *reinterpret_cast<const uint4*>(normalized + col + half_group);
        const float4 first = rms_quant_values({lower.x, lower.y});
        const float4 second = rms_quant_values({lower.z, lower.w});
        const float4 third = rms_quant_values({upper.x, upper.y});
        const float4 fourth = rms_quant_values({upper.z, upper.w});
        float maximum = fmaxf(fmaxf(rms_quant_maximum(first), rms_quant_maximum(second)),
                              fmaxf(rms_quant_maximum(third), rms_quant_maximum(fourth)));
        for (int delta = lanes / 2; delta > 0; delta /= 2) {
            maximum = fmaxf(maximum, __shfl_down_sync(warp_mask, maximum, delta, lanes));
        }
        float reciprocal = 0.0f;
        if (lane == 0) {
            const float scale = __fdiv_rn(fmaxf(quant_epsilon, maximum), fp8_max);
            activation.scales[group] = scale;
            reciprocal = __fdiv_rn(1.0f, scale);
        }
        reciprocal = __shfl_sync(warp_mask, reciprocal, 0, lanes);
        auto* packed = reinterpret_cast<uint2*>(activation.values + col);
        *packed = {rms_quant_pack(first, reciprocal), rms_quant_pack(second, reciprocal)};
        packed[half_group / sizeof(uint2)] =
            {rms_quant_pack(third, reciprocal), rms_quant_pack(fourth, reciprocal)};
    }
}

template <bool HasResidual, bool PackedGroups>
__device__ __inline_hint__ void rms_quantize_staged(
    __nv_bfloat16* branch_values, const __nv_bfloat16* residual_values,
    const __nv_bfloat16* gains, __nv_bfloat16* residual, __nv_bfloat16* normalized,
    TextFp8Activation activation) {
    constexpr int values_per_thread = rms_hidden_width / block_threads;
    constexpr int groups = rms_hidden_width / text_fp8_group_size;
    constexpr int values_per_lane = text_fp8_group_size / warp_width;
    float values[values_per_thread];
    float squares = 0.0f;
#pragma unroll
    for (int index = 0; index < values_per_thread; ++index) {
        const int col = threadIdx.x + index * block_threads;
        const auto value = residual_sum(branch_values, HasResidual ? residual_values : nullptr,
                                        col);
        values[index] = value;
        squares = fmaf(value, value, squares);
    }
    const float inverse_rms = rms_inverse(squares, rms_hidden_width);
#pragma unroll
    for (int index = 0; index < values_per_thread; ++index) {
        const int col = threadIdx.x + index * block_threads;
        const float value = values[index];
        const float gain = 1.0f + __bfloat162float(gains[col]);
        const auto element = __float2bfloat16_rn((value * inverse_rms) * gain);
        normalized[col] = element;
        branch_values[col] = element;
        if constexpr (HasResidual) {
            residual[col] = __float2bfloat16_rn(value);
        }
    }
    __syncthreads();
    if constexpr (PackedGroups) {
        rms_quantize_subwarps(branch_values, activation);
        return;
    }
    const int lane = threadIdx.x % warp_width;
    const int warp = threadIdx.x / warp_width;
    for (int group = warp; group < groups; group += block_warps) {
        const int col = group * text_fp8_group_size + lane;
        float maximum = quant_epsilon;
        float elements[values_per_lane];
#pragma unroll
        for (int part = 0; part < values_per_lane; ++part) {
            elements[part] = __bfloat162float(branch_values[col + part * warp_width]);
            maximum = fmaxf(maximum, fabsf(elements[part]));
        }
        const float amax = __shfl_sync(warp_mask, warp_max(maximum), 0);
        const float scale = __fdiv_rn(amax, fp8_max);
        const float reciprocal = __fdiv_rn(1.0f, scale);
        if (lane == 0) {
            activation.scales[group] = scale;
        }
#pragma unroll
        for (int part = 0; part < values_per_lane; ++part) {
            activation.values[col + part * warp_width] =
                __nv_cvt_float_to_fp8(elements[part] * reciprocal, __NV_SATFINITE, __NV_E4M3);
        }
    }
}

template __device__ void rms_quantize_staged<false, false>(
    __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*,
    __nv_bfloat16*, TextFp8Activation);
template __device__ void rms_quantize_staged<false, true>(
    __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*,
    __nv_bfloat16*, TextFp8Activation);
template __device__ void rms_quantize_staged<true, false>(
    __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*,
    __nv_bfloat16*, TextFp8Activation);
template __device__ void rms_quantize_staged<true, true>(
    __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*,
    __nv_bfloat16*, TextFp8Activation);

}

#endif
