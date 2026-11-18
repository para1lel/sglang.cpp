#include "sglang/decoder_cuda.cuh"
#include "sglang/decoder_internal.h"
#include "sglang/text_ops.h"

#include <cstddef>
#include <cstdint>

#include <cuda_fp8.h>
#include <cuda_runtime.h>

namespace sglang {
namespace {

constexpr int warp_width = 32;
constexpr int block_threads = 256;
constexpr int block_warps = block_threads / warp_width;
constexpr unsigned int warp_mask = 0xffffffffu;
constexpr float fp8_max = 448.0f;
constexpr float quant_epsilon = 1.0e-10f;

constexpr int divide_up(int count, int tile) {
    return count / tile + (count % tile != 0);
}

bool aligned(const void* pointer, std::size_t alignment) {
    return reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0;
}

__device__ float warp_sum(float value) {
    for (int offset = warp_width / 2; offset > 0; offset /= 2) {
        value += __shfl_down_sync(warp_mask, value, offset);
    }
    return value;
}

__device__ float warp_max(float value) {
    for (int offset = warp_width / 2; offset > 0; offset /= 2) {
        value = fmaxf(value, __shfl_down_sync(warp_mask, value, offset));
    }
    return value;
}

__device__ uint4 load_128(const void* address) {
    uint4 bits;
    const auto global = __cvta_generic_to_global(address);
    asm("ld.global.v4.b32 {%0, %1, %2, %3}, [%4];"
        : "=r"(bits.x), "=r"(bits.y), "=r"(bits.z), "=r"(bits.w)
        : "l"(global) : "memory");
    return bits;
}

__device__ float bf16x2_dot(unsigned int lhs, unsigned int rhs, float sum) {
    sum = fmaf(__uint_as_float(lhs << 16), __uint_as_float(rhs << 16), sum);
    return fmaf(__uint_as_float(lhs & 0xffff0000u), __uint_as_float(rhs & 0xffff0000u), sum);
}

struct Bf16Dot {
    float first = 0.0f;
    float second = 0.0f;

    __device__ void accumulate(uint4 lhs, uint4 rhs) {
        first = bf16x2_dot(lhs.x, rhs.x, first);
        second = bf16x2_dot(lhs.y, rhs.y, second);
        first = bf16x2_dot(lhs.z, rhs.z, first);
        second = bf16x2_dot(lhs.w, rhs.w, second);
    }

    __device__ float sum() const {
        return first + second;
    }
};

__device__ uint4 load_streaming_128(const void* address) {
    uint4 bits;
    const auto global = __cvta_generic_to_global(address);
    asm("ld.global.cs.v4.b32 {%0, %1, %2, %3}, [%4];"
        : "=r"(bits.x), "=r"(bits.y), "=r"(bits.z), "=r"(bits.w)
        : "l"(global) : "memory");
    return bits;
}

__global__ void bf16_gdn_gates_kernel(const __nv_bfloat16* weight_b,
                                     const __nv_bfloat16* weight_a,
                                     const __nv_bfloat16* input,
                                     __nv_bfloat16* output_b, __nv_bfloat16* output_a) {
    const bool gate_b = blockIdx.x < text_gdn_value_heads;
    const int row = blockIdx.x % text_gdn_value_heads;
    const auto* weight = (gate_b ? weight_b : weight_a) + row * decoder::hidden_size;
    Bf16Dot dot;
#pragma unroll 4
    for (int col = threadIdx.x * 8; col < decoder::hidden_size; col += warp_width * 8) {
        dot.accumulate(load_128(weight + col), load_128(input + col));
    }
    const auto result = warp_sum(dot.sum());
    if (threadIdx.x == 0) {
        (gate_b ? output_b : output_a)[row] = __float2bfloat16_rn(result);
    }
}

__global__ void bf16_lm_head_pair_kernel(const __nv_bfloat16* weight,
                                        const __nv_bfloat16* input, int rows,
                                        __nv_bfloat16* output) {
    const int lane = threadIdx.x % warp_width;
    const int row = 2 * (blockIdx.x * block_warps + threadIdx.x / warp_width);
    if (row >= rows) {
        return;
    }
    constexpr auto width = decoder::hidden_size;
    const auto* weights = weight + static_cast<std::size_t>(row) * width;
    Bf16Dot first;
    Bf16Dot second;
#pragma unroll 2
    for (int col = lane * 8; col < width; col += warp_width * 8) {
        const auto vector = load_128(input + col);
        first.accumulate(load_streaming_128(weights + col), vector);
        second.accumulate(load_streaming_128(weights + width + col), vector);
    }
    const auto first_result = warp_sum(first.sum());
    const auto second_result = warp_sum(second.sum());
    if (lane == 0) {
        output[row] = __float2bfloat16_rn(first_result);
        output[row + 1] = __float2bfloat16_rn(second_result);
    }
}

__global__ void quantize_fp8_warp(const __nv_bfloat16* input, TextFp8Activation output) {
    static_assert(text_fp8_group_size == warp_width * 4);
    const int lane = threadIdx.x;
    const auto index = static_cast<std::size_t>(blockIdx.x) * warp_width + lane;
    const auto bits = reinterpret_cast<const uint2*>(input)[index];
    const float first = __uint_as_float(bits.x << 16);
    const float second = __uint_as_float(bits.x & 0xffff0000u);
    const float third = __uint_as_float(bits.y << 16);
    const float fourth = __uint_as_float(bits.y & 0xffff0000u);
    const float maximum = warp_max(fmaxf(fmaxf(fabsf(first), fabsf(second)),
                                         fmaxf(fabsf(third), fabsf(fourth))));
    float reciprocal = 0.0f;
    if (lane == 0) {
        const float scale = __fdiv_rn(fmaxf(maximum, quant_epsilon), fp8_max);
        output.scales[blockIdx.x] = scale;
        reciprocal = __fdiv_rn(1.0f, scale);
    }
    reciprocal = __shfl_sync(warp_mask, reciprocal, 0);
    const auto lower = __nv_cvt_float2_to_fp8x2(
        make_float2(first * reciprocal, second * reciprocal), __NV_SATFINITE, __NV_E4M3);
    const auto upper = __nv_cvt_float2_to_fp8x2(
        make_float2(third * reciprocal, fourth * reciprocal), __NV_SATFINITE, __NV_E4M3);
    reinterpret_cast<std::uint32_t*>(output.values)[index] =
        lower | (std::uint32_t(upper) << 16);
}

}

cudaError_t decoder::decode_bf16_gdn_gates(const __nv_bfloat16* weight_b,
                                          const __nv_bfloat16* weight_a,
                                          const __nv_bfloat16* input, int rows, int width,
                                          __nv_bfloat16* output_b, __nv_bfloat16* output_a,
                                          cudaStream_t stream) {
    if (!weight_b || !weight_a || !input || !output_b || !output_a || rows <= 0 || width <= 0) {
        return cudaErrorInvalidValue;
    }
    if (rows != text_gdn_value_heads || width != hidden_size ||
        !aligned(weight_b, 16) || !aligned(weight_a, 16) || !aligned(input, 16)) {
        const auto status = text_bf16_gemv(weight_b, input, rows, width, output_b, stream);
        if (status != cudaSuccess) {
            return status;
        }
        return text_bf16_gemv(weight_a, input, rows, width, output_a, stream);
    }
    void* args[] = {&weight_b, &weight_a, &input, &output_b, &output_a};
    return cudaLaunchKernel(bf16_gdn_gates_kernel, 2 * text_gdn_value_heads, warp_width,
                            args, 0, stream);
}

cudaError_t decoder::decode_bf16_lm_head(const __nv_bfloat16* weight, const __nv_bfloat16* input,
                                       int rows, int width, __nv_bfloat16* output,
                                       cudaStream_t stream) {
    if (!weight || !input || !output || rows <= 0 || rows % 2 != 0 ||
        width != hidden_size || !aligned(weight, 16) || !aligned(input, 16)) {
        return text_bf16_gemv(weight, input, rows, width, output, stream);
    }
    void* args[] = {&weight, &input, &rows, &output};
    return cudaLaunchKernel(bf16_lm_head_pair_kernel, divide_up(rows, 2 * block_warps),
                            block_threads, args, 0, stream);
}

cudaError_t decoder::decode_quantize_fp8(const __nv_bfloat16* input, int width,
                                        TextFp8Activation output, cudaStream_t stream) {
    if (!input || !output.values || !output.scales || width <= 0 ||
        width % text_fp8_group_size != 0 || !aligned(input, alignof(uint2)) ||
        !aligned(output.values, alignof(std::uint32_t))) {
        return text_quantize_fp8(input, width, output, stream);
    }
    void* args[] = {&input, &output};
    return cudaLaunchKernel(quantize_fp8_warp, text_fp8_group_count(width), warp_width,
                            args, 0, stream);
}

}
