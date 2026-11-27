#pragma once

#include "sglang/text_ops.h"
#include "sglang/text/text_cuda_primitives.cuh"

#include <cuda_fp8.h>

namespace sglang::text_kernel {

constexpr float rms_epsilon = 1.0e-6f;
constexpr int rms_hidden_width = 5120;

__device__ float residual_sum(const __nv_bfloat16* branch, const __nv_bfloat16* residual,
                               std::size_t col);
__device__ void stage_rms_vector_async(const __nv_bfloat16* input, __nv_bfloat16* shared);
__device__ float rms_inverse(float squares, int width);
__device__ float4 rms_quant_values(uint2 bits);
__device__ float rms_quant_maximum(float4 values);
__device__ __nv_fp8x4_storage_t rms_quant_pack(float4 values, float reciprocal);

template <bool HasResidual, bool PackedGroups>
__device__ void rms_quantize_staged(
    __nv_bfloat16* branch_values, const __nv_bfloat16* residual_values,
    const __nv_bfloat16* gains, __nv_bfloat16* residual, __nv_bfloat16* normalized,
    TextFp8Activation activation);

}
