#include "sglang/decoder_cuda.cuh"
#include "sglang/decoder_internal.h"
#include "sglang/text/decode_proj.h"
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
constexpr int prefill_gate_tile_rows = 4;
constexpr int prefill_gate_warps = 4;
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

__device__ void accumulate_bf16_pair(float& first, float& second,
                                     unsigned lhs0, unsigned lhs1,
                                     unsigned rhs0, unsigned rhs1) {
    asm("{ .reg .b64 sums, left, right;\n"
        "mov.b64 sums, {%0,%1};\n"
        "mov.b64 left, {%2,%3};\n"
        "mov.b64 right, {%4,%5};\n"
        "fma.rn.f32x2 sums, left, right, sums;\n"
        "mov.b64 left, {%6,%7};\n"
        "mov.b64 right, {%8,%9};\n"
        "fma.rn.f32x2 sums, left, right, sums;\n"
        "mov.b64 {%0,%1}, sums; }"
        : "+f"(first), "+f"(second)
        : "r"(lhs0 << 16), "r"(lhs1 << 16), "r"(rhs0 << 16), "r"(rhs1 << 16),
          "r"(lhs0 & 0xffff0000u), "r"(lhs1 & 0xffff0000u),
          "r"(rhs0 & 0xffff0000u), "r"(rhs1 & 0xffff0000u));
}

__device__ void accumulate_packed_gate(Bf16Dot& dot, uint4 weight, uint4 input) {
    accumulate_bf16_pair(dot.first, dot.second, weight.x, weight.y, input.x, input.y);
    accumulate_bf16_pair(dot.first, dot.second, weight.z, weight.w, input.z, input.w);
}

__global__ void bf16_gdn_prefill_gates(const __nv_bfloat16* weight_b,
                                      const __nv_bfloat16* weight_a,
                                      const __nv_bfloat16* input,
                                      __nv_bfloat16* output_b, __nv_bfloat16* output_a) {
    constexpr int width = decoder::hidden_size;
    const int lane = threadIdx.x % warp_width;
    const int channel = blockIdx.x * prefill_gate_warps + threadIdx.x / warp_width;
    weight_b += channel * width;
    weight_a += channel * width;
    const int first_row = blockIdx.y * prefill_gate_tile_rows;
    Bf16Dot dots_b[prefill_gate_tile_rows];
    Bf16Dot dots_a[prefill_gate_tile_rows];
#pragma unroll 4
    for (int col = lane * 8; col < width; col += warp_width * 8) {
        const uint4 packed_b = load_128(weight_b + col);
        const uint4 packed_a = load_128(weight_a + col);
#pragma unroll
        for (int item = 0; item < prefill_gate_tile_rows; ++item) {
            const uint4 packed_input = load_128(input + (first_row + item) * width + col);
            accumulate_packed_gate(dots_b[item], packed_b, packed_input);
            accumulate_packed_gate(dots_a[item], packed_a, packed_input);
        }
    }
#pragma unroll
    for (int item = 0; item < prefill_gate_tile_rows; ++item) {
        const float result_b = warp_sum(dots_b[item].sum());
        const float result_a = warp_sum(dots_a[item].sum());
        if (lane == 0) {
            const int offset = (first_row + item) * text_gdn_value_heads + channel;
            output_b[offset] = __float2bfloat16_rn(result_b);
            output_a[offset] = __float2bfloat16_rn(result_a);
        }
    }
}

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

__global__ void bf16_gdn_decode_shared(const __nv_bfloat16* weight_b,
                                       const __nv_bfloat16* weight_a,
                                       const __nv_bfloat16* input, int rows,
                                       __nv_bfloat16* output_b, __nv_bfloat16* output_a) {
    constexpr int width = decoder::hidden_size;
    constexpr int packs = width / 8;
    __shared__ uint4 row_weight[packs];
    const bool gate_b = blockIdx.x < text_gdn_value_heads;
    const int channel = blockIdx.x % text_gdn_value_heads;
    const auto* weight = (gate_b ? weight_b : weight_a) + channel * width;
    auto* output = gate_b ? output_b : output_a;
    for (int pack = threadIdx.x; pack < packs; pack += blockDim.x) {
        row_weight[pack] = load_128(weight + pack * 8);
    }
    __syncthreads();
    const int lane = threadIdx.x % warp_width;
    for (int row = threadIdx.x / warp_width; row < rows; row += blockDim.x / warp_width) {
        Bf16Dot dot;
#pragma unroll 4
        for (int pack = lane; pack < packs; pack += warp_width) {
            dot.accumulate(row_weight[pack], load_128(input + row * width + pack * 8));
        }
        const auto result = warp_sum(dot.sum());
        if (lane == 0) {
            output[row * text_gdn_value_heads + channel] = __float2bfloat16_rn(result);
        }
    }
}

struct VerifyGdnGateArguments {
    decoder::VerifyGdnGateRow rows[text_fp8_max_batch];
};

__global__ void bf16_gdn_verify_gates(const __nv_bfloat16* weight_b,
                                     const __nv_bfloat16* weight_a,
                                     const __grid_constant__ VerifyGdnGateArguments arguments) {
    constexpr int packs = decoder::hidden_size / 8;
    __shared__ uint4 row_weight[packs];
    const bool gate_b = blockIdx.x < text_gdn_value_heads;
    const int channel = blockIdx.x % text_gdn_value_heads;
    const auto* weight = (gate_b ? weight_b : weight_a) + channel * decoder::hidden_size;
    for (int pack = threadIdx.x; pack < packs; pack += blockDim.x) {
        row_weight[pack] = load_128(weight + pack * 8);
    }
    __syncthreads();
    const int lane = threadIdx.x % warp_width;
    const auto row = arguments.rows[threadIdx.x / warp_width];
    Bf16Dot dot;
#pragma unroll 4
    for (int pack = lane; pack < packs; pack += warp_width) {
        accumulate_packed_gate(dot, row_weight[pack], load_128(row.input + pack * 8));
    }
    const float result = warp_sum(dot.sum());
    if (lane == 0) {
        (gate_b ? row.output_b : row.output_a)[channel] = __float2bfloat16_rn(result);
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

constexpr int head_batch_warps = 4;
constexpr int head_batch_lanes = std::tuple_size_v<decoder::DecodeHeadProjections>;

struct HeadBatchArguments {
    TextBf16Projection projections[head_batch_lanes];
};

__global__ void bf16_lm_head_batch_kernel(const __nv_bfloat16* weight, int rows,
                                         const __grid_constant__ HeadBatchArguments arguments) {
    const int lane = threadIdx.x % warp_width;
    const int row = 2 * (blockIdx.x * head_batch_warps + threadIdx.x / warp_width);
    if (row >= rows) {
        return;
    }
    constexpr int width = decoder::hidden_size;
    const auto* weights = weight + static_cast<std::size_t>(row) * width;
    Bf16Dot dots[2][head_batch_lanes];
#pragma unroll 2
    for (int column = lane * 8; column < width; column += warp_width * 8) {
        const auto first = load_streaming_128(weights + column);
        const auto second = load_streaming_128(weights + width + column);
#pragma unroll
        for (int request = 0; request < head_batch_lanes; ++request) {
            const auto input = load_128(arguments.projections[request].input + column);
            dots[0][request].accumulate(first, input);
            dots[1][request].accumulate(second, input);
        }
    }
#pragma unroll
    for (int offset = 0; offset < 2; ++offset) {
#pragma unroll
        for (int request = 0; request < head_batch_lanes; ++request) {
            const auto value = warp_sum(dots[offset][request].sum());
            if (lane == 0) {
                arguments.projections[request].output[row + offset] = __float2bfloat16_rn(value);
            }
        }
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

std::optional<cudaError_t> try_text_bf16_decode_gates(
    const __nv_bfloat16* weight_b, const __nv_bfloat16* weight_a,
    const __nv_bfloat16* input, int rows, __nv_bfloat16* output_b,
    __nv_bfloat16* output_a, cudaStream_t stream) {
    if (!aligned(weight_b, 16) || !aligned(weight_a, 16) || !aligned(input, 16)) {
        return std::nullopt;
    }
    const int channels = (weight_a ? 2 : 1) * text_gdn_value_heads;
    if (rows == 1) {
        void* args[] = {&weight_b, &weight_a, &input, &output_b, &output_a};
        return cudaLaunchKernel(bf16_gdn_gates_kernel, channels, warp_width, args, 0, stream);
    }
    void* args[] = {&weight_b, &weight_a, &input, &rows, &output_b, &output_a};
    const int row_warps = rows <= 8 ? 8 : (rows <= 16 ? 16 : 32);
    return cudaLaunchKernel(bf16_gdn_decode_shared, channels, row_warps * warp_width,
                            args, 0, stream);
}

std::optional<cudaError_t> decoder::try_prefill_bf16_gdn_gates(
    const __nv_bfloat16* weight_b, const __nv_bfloat16* weight_a, const __nv_bfloat16* input,
    int rows, __nv_bfloat16* output_b, __nv_bfloat16* output_a, cudaStream_t stream) {
    if (rows < 16 || rows > 65535 || rows % prefill_gate_tile_rows != 0 ||
        !weight_b || !weight_a || !input || !output_b || !output_a ||
        !aligned(weight_b, 16) || !aligned(weight_a, 16) || !aligned(input, 16)) {
        return std::nullopt;
    }
    void* args[] = {&weight_b, &weight_a, &input, &output_b, &output_a};
    static_assert(text_gdn_value_heads % prefill_gate_warps == 0);
    const dim3 grid(text_gdn_value_heads / prefill_gate_warps, rows / prefill_gate_tile_rows);
    return cudaLaunchKernel(bf16_gdn_prefill_gates, grid, prefill_gate_warps * warp_width,
                            args, 0, stream);
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

std::optional<cudaError_t> decoder::try_verify_bf16_gdn_gates(
    const __nv_bfloat16* weight_b, const __nv_bfloat16* weight_a,
    const VerifyGdnGateRow* rows, int count, cudaStream_t stream) {
    if (count < 2 || count > text_fp8_max_batch || !rows || !weight_b || !weight_a ||
        !aligned(weight_b, 16) || !aligned(weight_a, 16)) {
        return std::nullopt;
    }
    VerifyGdnGateArguments arguments{};
    for (int index = 0; index < count; ++index) {
        const auto& row = rows[index];
        if (!row.input || !row.output_b || !row.output_a || !aligned(row.input, 16)) {
            return std::nullopt;
        }
        arguments.rows[index] = row;
    }
    void* args[] = {&weight_b, &weight_a, &arguments};
    return cudaLaunchKernel(bf16_gdn_verify_gates, 2 * text_gdn_value_heads,
                            count * warp_width, args, 0, stream);
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

cudaError_t decoder::decode_bf16_lm_head_batch(const __nv_bfloat16* weight, int rows, int width,
                                             const DecodeHeadProjections& projections,
                                             cudaStream_t stream) {
    if (!weight || rows <= 0 || rows % 2 != 0 || width != hidden_size || !aligned(weight, 16)) {
        return text_bf16_gemm(weight, rows, width, projections.data(), projections.size(), stream);
    }
    HeadBatchArguments arguments;
    for (int request = 0; request < head_batch_lanes; ++request) {
        const auto& projection = projections[request];
        if (!projection.input || !projection.output || !aligned(projection.input, 16)) {
            return text_bf16_gemm(weight, rows, width, projections.data(), projections.size(),
                                  stream);
        }
        arguments.projections[request] = projection;
    }
    void* args[] = {&weight, &rows, &arguments};
    return cudaLaunchKernel(bf16_lm_head_batch_kernel, divide_up(rows, 2 * head_batch_warps),
                            warp_width * head_batch_warps, args, 0, stream);
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
