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

inline bool aligned(const void* pointer, std::size_t alignment) {
    return reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0;
}

__device__ inline float warp_sum(float value) {
    for (int offset = warp_width / 2; offset > 0; offset /= 2) {
        value += __shfl_down_sync(warp_mask, value, offset);
    }
    return value;
}

__device__ inline float warp_max(float value) {
    for (int offset = warp_width / 2; offset > 0; offset /= 2) {
        value = fmaxf(value, __shfl_down_sync(warp_mask, value, offset));
    }
    return value;
}

__device__ inline uint4 load_128(const void* address) {
    uint4 bits;
    const auto global = __cvta_generic_to_global(address);
    asm("ld.global.v4.b32 {%0, %1, %2, %3}, [%4];"
        : "=r"(bits.x), "=r"(bits.y), "=r"(bits.z), "=r"(bits.w)
        : "l"(global) : "memory");
    return bits;
}

}
