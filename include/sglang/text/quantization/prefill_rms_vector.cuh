#pragma once

#include "sglang/text/quantization/rms_quantize.cuh"

namespace sglang::decoder::prefill_rms_detail {

using namespace text_kernel;
using Bf16 = __nv_bfloat16;
constexpr int vector_width = 8;
constexpr int group_width = text_fp8_group_size;
constexpr int group_lanes = 8;
constexpr int groups_per_block = block_threads / group_lanes;
constexpr int groups = rms_hidden_width / group_width;
static_assert(group_width == 2 * group_lanes * vector_width);
static_assert(groups % (warp_width / group_lanes) == 0);

struct alignas(16) PackedPair {
    uint2 lower, upper;
};

__device__ __forceinline__ uint2 round_bf16(float4 v) {
    const __nv_bfloat162_raw ab = __float22bfloat162_rn({v.x, v.y});
    const __nv_bfloat162_raw cd = __float22bfloat162_rn({v.z, v.w});
    return {static_cast<unsigned>(ab.x) | static_cast<unsigned>(ab.y) << 16,
            static_cast<unsigned>(cd.x) | static_cast<unsigned>(cd.y) << 16};
}

__device__ __forceinline__ uint4 prefill_bf16_sum(uint4 local, uint4 peer) {
    uint4 output;
    asm("{ .reg .b32 first, second, third, fourth;\n\t"
        "add.rn.bf16x2 first, %4, %8;\n\t"
        "add.rn.bf16x2 second, %5, %9;\n\t"
        "add.rn.bf16x2 third, %6, %10;\n\t"
        "add.rn.bf16x2 fourth, %7, %11;\n\t"
        "mov.b32 %0, first; mov.b32 %1, second;\n\t"
        "mov.b32 %2, third; mov.b32 %3, fourth; }"
        : "=r"(output.x), "=r"(output.y), "=r"(output.z), "=r"(output.w)
        : "r"(local.x), "r"(local.y), "r"(local.z), "r"(local.w),
          "r"(peer.x), "r"(peer.y), "r"(peer.z), "r"(peer.w));
    return output;
}

template <bool Residual>
__device__ __forceinline__ uint2 normalize4(uint2 branch, uint2 residual, uint2 gains,
                                             float inverse, uint2& updated) {
    auto value = rms_quant_values(branch);
    if constexpr (Residual) {
        const auto addend = rms_quant_values(residual);
        value = {value.x + addend.x, value.y + addend.y,
                 value.z + addend.z, value.w + addend.w};
        updated = round_bf16(value);
    }
    const auto gain = rms_quant_values(gains);
    return round_bf16({(value.x * inverse) * (1.0f + gain.x),
                       (value.y * inverse) * (1.0f + gain.y),
                       (value.z * inverse) * (1.0f + gain.z),
                       (value.w * inverse) * (1.0f + gain.w)});
}

template <bool Residual, bool SaveNormalized>
__device__ __forceinline__ float4 prepare_half(const Bf16* branch, const Bf16* residual_in,
                                                const Bf16* gains, Bf16* residual,
                                                Bf16* normalized, int col,
                                                float inverse, float4& second) {
    const auto input = reinterpret_cast<const PackedPair*>(branch + col)[0];
    const auto gain = reinterpret_cast<const PackedPair*>(gains + col)[0];
    PackedPair addend{};
    if constexpr (Residual) addend = reinterpret_cast<const PackedPair*>(residual_in + col)[0];
    PackedPair updated;
    const PackedPair result{normalize4<Residual>(input.lower, addend.lower, gain.lower,
                                                   inverse, updated.lower),
                            normalize4<Residual>(input.upper, addend.upper, gain.upper,
                                                   inverse, updated.upper)};
    if constexpr (Residual) reinterpret_cast<PackedPair*>(residual + col)[0] = updated;
    if constexpr (SaveNormalized) reinterpret_cast<PackedPair*>(normalized + col)[0] = result;
    second = rms_quant_values(result.upper);
    return rms_quant_values(result.lower);
}

template <bool Residual, bool SaveNormalized, bool Peer = false>
__global__ __launch_bounds__(block_threads) void vector_rms(const Bf16* branch, Bf16* residual,
                const Bf16* gain, Bf16* normalized, TextFp8Activation activation,
                const Bf16* remote) {
    const auto offset = static_cast<std::size_t>(blockIdx.x) * rms_hidden_width;
    branch += offset;
    if constexpr (Peer) remote += offset;
    if constexpr (Residual) residual += offset;
    if constexpr (SaveNormalized) normalized += offset;
    activation.values += offset;
    activation.scales += blockIdx.x * groups;

    __shared__ alignas(16) Bf16 staged_branch[rms_hidden_width];
    __shared__ alignas(16) Bf16 staged_residual[Residual ? rms_hidden_width : 1];
    for (int col = threadIdx.x * vector_width; col < rms_hidden_width;
         col += block_threads * vector_width) {
        if constexpr (Peer) {
            const auto local = reinterpret_cast<const uint4*>(branch + col)[0];
            const auto peer = reinterpret_cast<const uint4*>(remote + col)[0];
            reinterpret_cast<uint4*>(staged_branch + col)[0] = prefill_bf16_sum(local, peer);
        } else {
            stage_rms_vector_async(branch + col, staged_branch + col);
        }
        if constexpr (Residual) stage_rms_vector_async(residual + col, staged_residual + col);
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncthreads();
    float squares = 0.0f;
#pragma unroll
    for (int col = threadIdx.x; col < rms_hidden_width; col += block_threads) {
        const auto value = residual_sum(staged_branch, Residual ? staged_residual : nullptr, col);
        squares = fmaf(value, value, squares);
    }
    const float inverse = rms_inverse(squares, rms_hidden_width);
    const int lane = threadIdx.x % group_lanes;
    for (int group = threadIdx.x / group_lanes; group < groups; group += groups_per_block) {
        const int col = group * group_width + lane * vector_width;
        float4 second, fourth;
        const auto first = prepare_half<Residual, SaveNormalized>(
            staged_branch, staged_residual, gain, residual, normalized, col, inverse, second);
        const auto third = prepare_half<Residual, SaveNormalized>(
            staged_branch, staged_residual, gain, residual, normalized, col + group_width / 2,
            inverse, fourth);
        float maximum = fmaxf(fmaxf(rms_quant_maximum(first), rms_quant_maximum(second)),
                              fmaxf(rms_quant_maximum(third), rms_quant_maximum(fourth)));
#pragma unroll
        for (int delta = group_lanes / 2; delta > 0; delta /= 2) {
            maximum = fmaxf(maximum, __shfl_down_sync(warp_mask, maximum, delta, group_lanes));
        }
        float reciprocal = 0;
        if (lane == 0) {
            const float scale = __fdiv_rn(fmaxf(quant_epsilon, maximum), fp8_max);
            activation.scales[group] = scale;
            reciprocal = __fdiv_rn(1.0f, scale);
        }
        reciprocal = __shfl_sync(warp_mask, reciprocal, 0, group_lanes);
        auto* fp8 = reinterpret_cast<uint2*>(activation.values + col);
        fp8[0] = {rms_quant_pack(first, reciprocal), rms_quant_pack(second, reciprocal)};
        fp8[group_width / 2 / sizeof(uint2)] =
            {rms_quant_pack(third, reciprocal), rms_quant_pack(fourth, reciprocal)};
    }
}

}
