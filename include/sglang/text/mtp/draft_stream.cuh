#pragma once

#include "sglang/text/mtp/draft_stream.h"

namespace sglang::decoder::draft_stream {
namespace {

constexpr int width = hidden_size;
constexpr int fc_width = 2 * width;

template <int Count>
struct FcRows {
    TextBf16Projection entries[Count];
};

__device__ __forceinline__ void dot_pair(float& first, float& second,
                                         unsigned w0, unsigned w1, unsigned x0, unsigned x1) {
    asm("{ .reg .b64 sum, weight, input;\n"
        "mov.b64 sum, {%0,%1};\n"
        "mov.b64 weight, {%2,%3};\n"
        "mov.b64 input, {%4,%5};\n"
        "fma.rn.f32x2 sum, weight, input, sum;\n"
        "mov.b64 weight, {%6,%7};\n"
        "mov.b64 input, {%8,%9};\n"
        "fma.rn.f32x2 sum, weight, input, sum;\n"
        "mov.b64 {%0,%1}, sum; }"
        : "+f"(first), "+f"(second)
        : "r"(w0 << 16), "r"(w1 << 16), "r"(x0 << 16), "r"(x1 << 16),
          "r"(w0 & 0xffff0000u), "r"(w1 & 0xffff0000u),
          "r"(x0 & 0xffff0000u), "r"(x1 & 0xffff0000u));
}

template <int Count>
__global__ __launch_bounds__(block_threads) void stream_fc(
    const Bf16* weights, const __grid_constant__ FcRows<Count> rows) {
    constexpr int pack = sizeof(uint4) / sizeof(Bf16);
    const int lane = threadIdx.x % warp_width;
    const int row = (blockIdx.x * block_warps + threadIdx.x / warp_width) * Count;
    const auto* weight = weights + static_cast<std::size_t>(row) * fc_width;
    float first[Count][Count]{}, second[Count][Count]{};
#pragma unroll 4
    for (int column = lane * pack; column < fc_width; column += warp_width * pack) {
        uint4 values[Count];
#pragma unroll
        for (int index = 0; index < Count; ++index) {
            values[index] = load_128(rows.entries[index].input + column);
        }
#pragma unroll
        for (int channel = 0; channel < Count; ++channel) {
            const auto* source = weight + channel * fc_width + column;
            const auto packed = Count == 1 ? load_fp8_weight(source) : load_128(source);
#pragma unroll
            for (int index = 0; index < Count; ++index) {
                const auto input = values[index];
                dot_pair(first[index][channel], second[index][channel],
                         packed.x, packed.y, input.x, input.y);
                dot_pair(first[index][channel], second[index][channel],
                         packed.z, packed.w, input.z, input.w);
            }
        }
    }
#pragma unroll
    for (int index = 0; index < Count; ++index) {
#pragma unroll
        for (int channel = 0; channel < Count; ++channel) {
            const float sum = warp_sum(first[index][channel] + second[index][channel]);
            if (lane == 0) rows.entries[index].output[row + channel] = __float2bfloat16_rn(sum);
        }
    }
}

template <int Count>
cudaError_t launch_fc(const Bf16* weight, const TextBf16Projection* rows, cudaStream_t stream) {
    static_assert(width % (block_warps * Count) == 0);
    FcRows<Count> packed;
    std::copy_n(rows, Count, packed.entries);
    stream_fc<Count><<<width / (block_warps * Count), block_threads, 0, stream>>>(weight, packed);
    return cudaGetLastError();
}

template <int Count>
__global__ void stream_kv(const std::uint8_t* key, const Bf16* key_scales,
                          const std::uint8_t* value, const Bf16* value_scales,
                          const __grid_constant__ Fp8Projections<2 * Count> rows) {
    static_assert(width % text_fp8_group_size == 0);
    constexpr int groups = width / text_fp8_group_size;
    constexpr int lanes_per_group = warp_width / fp8_groups_per_warp;
    const int target = blockIdx.y;
    const auto* weight = target == 0 ? key : value;
    const auto* scales = target == 0 ? key_scales : value_scales;
    const int lane = threadIdx.x % warp_width;
    const int row = blockIdx.x * fp8_packed_warps + threadIdx.x / warp_width;
    weight += static_cast<std::size_t>(row) * width;
    scales += (row / text_fp8_group_size) * groups;
    float sums[Count]{};
#pragma unroll 2
    for (int group = 0; group < groups; group += fp8_groups_per_warp) {
        const int col = group * text_fp8_group_size + lane * fp8_pack_bytes;
        const int scale_index = group + lane / lanes_per_group;
        const float scale = __bfloat162float(scales[scale_index]);
        const uint4 lhs = load_fp8_weight(weight + col);
#pragma unroll
        for (int index = 0; index < Count; ++index) {
            const auto input = rows.entries[index].input;
            const uint4 rhs = load_128(input.values + col);
            const float low = fp8x4_dot(lhs.y, rhs.y, fp8x4_dot(lhs.x, rhs.x, 0.0f));
            const float high = fp8x4_dot(lhs.w, rhs.w, fp8x4_dot(lhs.z, rhs.z, 0.0f));
            sums[index] = fmaf((low + high) * input.scales[scale_index], scale, sums[index]);
        }
    }
#pragma unroll
    for (int index = 0; index < Count; ++index) {
        const float sum = warp_sum(sums[index]);
        if (lane == 0) rows.entries[target * Count + index].output[row] = __float2bfloat16_rn(sum);
    }
}

template <int Count>
cudaError_t launch_kv(const std::uint8_t* key, const Bf16* key_scales,
                      const std::uint8_t* value, const Bf16* value_scales,
                      const TextFp8Projection* rows, cudaStream_t stream) {
    static_assert(text_gqa_kv_width % fp8_packed_warps == 0);
    Fp8Projections<2 * Count> packed;
    std::copy_n(rows, 2 * Count, packed.entries);
    const dim3 grid(text_gqa_kv_width / fp8_packed_warps, 2);
    stream_kv<Count><<<grid, fp8_packed_warps * warp_width, 0, stream>>>(
        key, key_scales, value, value_scales, packed);
    return cudaGetLastError();
}

}

cudaError_t prepare() {
    const void* kernels[] = {
        reinterpret_cast<const void*>(stream_fc<1>), reinterpret_cast<const void*>(stream_fc<2>),
        reinterpret_cast<const void*>(stream_kv<1>), reinterpret_cast<const void*>(stream_kv<2>),
    };
    for (const auto* kernel : kernels) {
        cudaFuncAttributes attributes{};
        const auto status = cudaFuncGetAttributes(&attributes, kernel);
        if (status != cudaSuccess) return status;
        if (attributes.binaryVersion != 100 || attributes.localSizeBytes != 0) {
            return cudaErrorNotSupported;
        }
    }
    return cudaSuccess;
}

cudaError_t project_fc(const Bf16* weight, const TextBf16Projection* rows, int count,
                       cudaStream_t stream) {
    if (count == 1) return launch_fc<1>(weight, rows, stream);
    if (count == 2) return launch_fc<2>(weight, rows, stream);
    return cudaErrorInvalidValue;
}

cudaError_t project_kv(const std::uint8_t* key, const Bf16* key_scales, const std::uint8_t* value,
                       const Bf16* value_scales, const TextFp8Projection* rows, int count,
                       cudaStream_t stream) {
    if (count == 1) return launch_kv<1>(key, key_scales, value, value_scales, rows, stream);
    if (count == 2) return launch_kv<2>(key, key_scales, value, value_scales, rows, stream);
    return cudaErrorInvalidValue;
}

}
