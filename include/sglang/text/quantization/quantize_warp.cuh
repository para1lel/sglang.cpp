#pragma once

#include "sglang/text_ops.h"
#include "sglang/text/text_cuda_primitives.cuh"

#include <cuda_fp8.h>

namespace sglang::text_kernel {

__device__ __nv_bfloat16 swiglu(__nv_bfloat16 gate, __nv_bfloat16 up);

__device__ void quantize_warp_pairs(__nv_bfloat162 first, __nv_bfloat162 second,
                                    int group, int offset, int lane, TextFp8Activation output);

template <bool ApplySwiglu>
__device__ void quantize_warp(const __nv_bfloat16* input, const __nv_bfloat16* up, int width,
                              TextFp8Activation output);

}
