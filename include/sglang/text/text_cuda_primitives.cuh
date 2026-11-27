#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace sglang::text_kernel {

constexpr int warp_width = 32;
constexpr int block_threads = 256;
constexpr int block_warps = block_threads / warp_width;
constexpr unsigned int warp_mask = 0xffffffffu;
constexpr float fp8_max = 448.0f;
constexpr float quant_epsilon = 1.0e-10f;

bool aligned(const void* pointer, std::size_t alignment);
int divide_up(int count, int tile);

__device__ float warp_sum(float value);
__device__ float warp_max(float value);
__device__ uint4 load_128(const void* address);

}
