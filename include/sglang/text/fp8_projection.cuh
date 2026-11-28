#pragma once

#include "sglang/text/text_cuda_primitives.cuh"
#include "sglang/text_ops.h"

namespace sglang::fp8_projection {

constexpr int fp8_pack_bytes = 16;
constexpr int fp8_packed_warps = 4;
constexpr int fp8_groups_per_warp = text_kernel::warp_width * fp8_pack_bytes / text_fp8_group_size;

bool supports_wide_fp8(const std::uint8_t* weight, TextFp8ActivationView input, int in_width);
__device__ uint4 load_fp8_weight(const void* address);
__device__ float fp8_pack_dot(uint4 lhs, uint4 rhs);

}
