#include "sglang/text_ops.h"

#include "sglang/decoder_cuda.cuh"
#include "sglang/decoder_internal.h"
#include "sglang/decoder_mtp.h"
#include "sglang/text/decode_proj.h"
#include "sglang/text/fp8_projection.cuh"
#include "sglang/text/mtp/draft_stream.h"
#include "sglang/text/quantization/quantize_warp.cuh"
#include "sglang/text/text_cuda_primitives.cuh"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

namespace sglang {
namespace {

using namespace text_kernel;
using namespace fp8_projection;

__device__ float decode_fp8(std::uint8_t bits) {
    __nv_fp8_e4m3 value;
    value.__x = bits;
    return static_cast<float>(value);
}

__device__ float4 load_fp8x4(const std::uint8_t* address) {
    __nv_fp8x4_e4m3 value;
    const auto global = __cvta_generic_to_global(address);
    asm("ld.global.b32 %0, [%1];" : "=r"(value.__x) : "l"(global) : "memory");
    return static_cast<float4>(value);
}

template <bool Packed>
__global__ void fp8_gemv_kernel(const std::uint8_t* weight, const __nv_bfloat16* weight_scales,
                                TextFp8ActivationView input, int out_width, int in_width,
                                int groups, __nv_bfloat16* output) {
    const int lane = threadIdx.x % warp_width;
    const int row = blockIdx.x * block_warps + threadIdx.x / warp_width;
    if (row >= out_width) {
        return;
    }
    const auto* row_weight = weight + static_cast<std::size_t>(row) * in_width;
    const auto* row_scales =
        weight_scales + static_cast<std::size_t>(row / text_fp8_group_size) * groups;
    float sum = 0.0f;
    for (int group = 0; group < groups; ++group) {
        const auto start = static_cast<std::size_t>(group) * text_fp8_group_size;
        const int count = min(text_fp8_group_size, in_width - static_cast<int>(start));
        float dot = 0.0f;
        if constexpr (Packed) {
            const int col = lane * 4;
            if (col < count) {
                const float4 rhs = load_fp8x4(input.values + start + col);
                const float4 lhs = load_fp8x4(row_weight + start + col);
                dot = fmaf(lhs.x, rhs.x, dot);
                dot = fmaf(lhs.y, rhs.y, dot);
                dot = fmaf(lhs.z, rhs.z, dot);
                dot = fmaf(lhs.w, rhs.w, dot);
            }
        } else {
            for (int col = lane; col < count; col += warp_width) {
                dot = fmaf(decode_fp8(row_weight[start + col]),
                           decode_fp8(input.values[start + col]), dot);
            }
        }
        dot = warp_sum(dot);
        if (lane == 0) {
            const float scaled_dot = dot * input.scales[group];
            sum = fmaf(scaled_dot, __bfloat162float(row_scales[group]), sum);
        }
    }
    if (lane == 0) {
        output[row] = __float2bfloat16_rn(sum);
    }
}

__device__ float fp8_pair_dot(std::uint16_t bits, __half2_raw rhs, float sum) {
    const auto lhs = __nv_cvt_fp8x2_to_halfraw2(bits, __NV_E4M3);
    asm("fma.rn.f32.f16 %0, %1, %2, %0;" : "+f"(sum) : "h"(lhs.x), "h"(rhs.x));
    asm("fma.rn.f32.f16 %0, %1, %2, %0;" : "+f"(sum) : "h"(lhs.y), "h"(rhs.y));
    return sum;
}

__device__ float fp8x4_dot(unsigned lhs, unsigned rhs, float sum) {
#pragma unroll
    for (int pair = 0; pair < 2; ++pair) {
        const auto shift = static_cast<unsigned>(pair * 16);
        const auto right =
            __nv_cvt_fp8x2_to_halfraw2(static_cast<std::uint16_t>(rhs >> shift), __NV_E4M3);
        sum = fp8_pair_dot(static_cast<std::uint16_t>(lhs >> shift), right, sum);
    }
    return sum;
}

}

namespace fp8_projection {

using namespace text_kernel;

__device__ __inline_hint__ float fp8_pack_dot(uint4 lhs, uint4 rhs) {
    const float low = fp8x4_dot(lhs.y, rhs.y, fp8x4_dot(lhs.x, rhs.x, 0.0F));
    const float high = fp8x4_dot(lhs.w, rhs.w, fp8x4_dot(lhs.z, rhs.z, 0.0F));
    return low + high;
}

bool supports_wide_fp8(const std::uint8_t* weight, TextFp8ActivationView input, int in_width) {
    return in_width % (fp8_groups_per_warp * text_fp8_group_size) == 0 &&
           aligned(weight, fp8_pack_bytes) && aligned(input.values, fp8_pack_bytes);
}

__device__ __inline_hint__ uint4 load_fp8_weight(const void* address) {
    uint4 bits;
    const auto global = __cvta_generic_to_global(address);
    asm("ld.global.cs.v4.b32 {%0, %1, %2, %3}, [%4];"
        : "=r"(bits.x), "=r"(bits.y), "=r"(bits.z), "=r"(bits.w)
        : "l"(global) : "memory");
    return bits;
}

}

namespace {

using fp8_projection::fp8_pack_dot;

struct ActivationPack {
    __half2_raw pairs[8];
};

__device__ ActivationPack unpack_activation(uint4 bytes) {
    const unsigned words[]{bytes.x, bytes.y, bytes.z, bytes.w};
    ActivationPack pack{};
#pragma unroll
    for (std::size_t word = 0; word < 4; ++word) {
        pack.pairs[2 * word] =
            __nv_cvt_fp8x2_to_halfraw2(static_cast<std::uint16_t>(words[word]), __NV_E4M3);
        pack.pairs[(2 * word) + 1] =
            __nv_cvt_fp8x2_to_halfraw2(static_cast<std::uint16_t>(words[word] >> 16U), __NV_E4M3);
    }
    return pack;
}

__device__ float fp8_pack_dot(uint4 bytes, const ActivationPack& input) {
    const unsigned words[]{bytes.x, bytes.y, bytes.z, bytes.w};
    float low = 0.0F;
    float high = 0.0F;
#pragma unroll
    for (std::size_t word = 0; word < 2; ++word) {
        low = fp8_pair_dot(static_cast<std::uint16_t>(words[word]), input.pairs[2 * word], low);
        low = fp8_pair_dot(static_cast<std::uint16_t>(words[word] >> 16U),
                           input.pairs[(2 * word) + 1], low);
        high = fp8_pair_dot(static_cast<std::uint16_t>(words[word + 2]),
                            input.pairs[(2 * word) + 4], high);
        high = fp8_pair_dot(static_cast<std::uint16_t>(words[word + 2] >> 16U),
                            input.pairs[(2 * word) + 5], high);
    }
    return low + high;
}

using decoder::DecodeFp8Projection;

template <int Count> struct SingletonProjections {
    DecodeFp8Projection entries[Count];
};

template <int Width, int Count>
__global__ void fp8_decode_projection_kernel(TextFp8ActivationView input,
                                             SingletonProjections<Count> projections,
                                             int in_width) {
    const int width = Width == 0 ? in_width : Width;
    const int groups = width / text_fp8_group_size;
    const int lane = static_cast<int>(threadIdx.x % warp_width);
    int row = static_cast<int>((blockIdx.x * fp8_packed_warps) + (threadIdx.x / warp_width));
    auto matrix = projections.entries[0];
    if constexpr (Count == 3) {
        if (row >= matrix.rows) {
            row -= matrix.rows;
            matrix = projections.entries[1];
            if (row >= matrix.rows) {
                row -= matrix.rows;
                matrix = projections.entries[2];
            }
        }
    }
    if (row >= matrix.rows) {
        return;
    }
    constexpr int lanes_per_group = warp_width / fp8_groups_per_warp;
    const int lane_group = lane / lanes_per_group;
    const int lane_column = lane * fp8_pack_bytes;
    const auto offset = static_cast<std::size_t>(row) * width;
    const auto scale_offset = static_cast<std::size_t>(row / text_fp8_group_size) * groups;
    const auto* values = matrix.values + offset + lane_column;
    const auto* scales = matrix.scales + scale_offset + lane_group;
    float sum = 0.0F;
    constexpr int unroll_factor = Width == 0 ? 1 : 2;
#pragma unroll unroll_factor
    for (int group = 0; group < groups; group += fp8_groups_per_warp) {
        const int column = group * text_fp8_group_size;
        const auto rhs = load_128(input.values + column + lane_column);
        const auto lhs = load_fp8_weight(values + column);
        const float dot =
            Count == 3 ? fp8_pack_dot(lhs, unpack_activation(rhs)) : fp8_pack_dot(lhs, rhs);
        sum = fmaf(dot * input.scales[group + lane_group], __bfloat162float(scales[group]), sum);
    }
    sum = warp_sum(sum);
    if (lane == 0) {
        matrix.output[row] = __float2bfloat16_rn(sum);
    }
}

template <int Width, int Count>
cudaError_t launch_decode_projection(int width, TextFp8ActivationView input,
                                     SingletonProjections<Count> projections, int rows,
                                     cudaStream_t stream) {
    static_assert(Width % (text_fp8_group_size * fp8_groups_per_warp) == 0);
    void* args[] = {&input, &projections, &width};
    return cudaLaunchKernel(fp8_decode_projection_kernel<Width, Count>,
                            divide_up(rows, fp8_packed_warps), fp8_packed_warps * warp_width, args,
                            0, stream);
}

__global__ void fp8_decode_gate_up_kernel(TextFp8ActivationView input, DecodeFp8Projection gate,
                                          DecodeFp8Projection up) {
    const int lane = static_cast<int>(threadIdx.x % warp_width);
    const int row = static_cast<int>((blockIdx.x * fp8_packed_warps) + (threadIdx.x / warp_width));
    if (row >= gate.rows) {
        return;
    }
    constexpr int width = decoder::hidden_size;
    constexpr int groups = width / text_fp8_group_size;
    constexpr int lanes_per_group = warp_width / fp8_groups_per_warp;
    static_assert(groups % fp8_groups_per_warp == 0);
    const int lane_group = lane / lanes_per_group;
    const int lane_column = lane * fp8_pack_bytes;
    const auto row_column = (static_cast<std::size_t>(row) * width) + lane_column;
    const int scale_offset = ((row / text_fp8_group_size) * groups) + lane_group;
    float gate_sum = 0.0F;
    float up_sum = 0.0F;
#pragma unroll 1
    for (int group = 0; group < groups; group += fp8_groups_per_warp) {
        const int column = group * text_fp8_group_size;
        const auto rhs = unpack_activation(load_128(input.values + column + lane_column));
        const float scale = input.scales[group + lane_group];
        const auto gate_pack = load_fp8_weight(gate.values + row_column + column);
        gate_sum = fmaf(fp8_pack_dot(gate_pack, rhs) * scale,
                        __bfloat162float(gate.scales[scale_offset + group]), gate_sum);
        const auto up_pack = load_fp8_weight(up.values + row_column + column);
        up_sum = fmaf(fp8_pack_dot(up_pack, rhs) * scale,
                      __bfloat162float(up.scales[scale_offset + group]), up_sum);
    }
    gate_sum = warp_sum(gate_sum);
    up_sum = warp_sum(up_sum);
    if (lane == 0) {
        gate.output[row] = swiglu(__float2bfloat16_rn(gate_sum), __float2bfloat16_rn(up_sum));
    }
}

struct ByteSpan {
    std::uintptr_t begin;
    std::uintptr_t end;

    bool valid() const {
        return begin < end;
    }
    bool overlaps(ByteSpan other) const {
        return begin < other.end && other.begin < end;
    }
};

template <typename Element> ByteSpan byte_span(const Element* pointer, std::size_t count) {
    const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
    return {begin, begin + (count * sizeof(Element))};
}

template <std::size_t Count>
bool valid_singleton_buffers(TextFp8ActivationView input, int width,
                             const DecodeFp8Projection (&projections)[Count], bool alias = false) {
    static_assert(std::numeric_limits<std::uintptr_t>::digits >=
                  (2 * std::numeric_limits<int>::digits) + 1);
    if (width <= 0 || !input.values || !input.scales || !aligned(input.scales, alignof(float))) {
        return false;
    }
    const auto groups = static_cast<std::size_t>(text_fp8_group_count(width));
    const ByteSpan inputs[]{byte_span(input.values, width), byte_span(input.scales, groups)};
    if (!inputs[0].valid() || !inputs[1].valid()) {
        return false;
    }
    ByteSpan weights[Count][2];
    ByteSpan outputs[Count];
    for (std::size_t i = 0; i < Count; ++i) {
        const auto& [values, scales, output, rows] = projections[i];
        if (rows <= 0 || !values || !scales || !output ||
            !aligned(scales, alignof(__nv_bfloat16)) || !aligned(output, alignof(__nv_bfloat16))) {
            return false;
        }
        weights[i][0] = byte_span(values, std::size_t(rows) * width);
        weights[i][1] = byte_span(scales, text_fp8_group_count(rows) * groups);
        outputs[i] = byte_span(output, rows);
        if (!weights[i][0].valid() || !weights[i][1].valid() || !outputs[i].valid() ||
            inputs[0].overlaps(outputs[i]) || inputs[1].overlaps(outputs[i])) {
            return false;
        }
    }
    for (std::size_t i = 0; i < Count; ++i) {
        for (const auto& weight : weights) {
            if (weight[0].overlaps(outputs[i]) || weight[1].overlaps(outputs[i])) {
                return false;
            }
        }
        for (std::size_t j = 0; j < i; ++j) {
            const bool equal =
                outputs[j].begin == outputs[i].begin && outputs[j].end == outputs[i].end;
            if (outputs[j].overlaps(outputs[i]) && !(alias && equal)) {
                return false;
            }
        }
    }
    return true;
}

cudaError_t singleton_unfused(TextFp8ActivationView input, DecodeFp8Projection projection,
                              int width, cudaStream_t stream) {
    return text_fp8_gemv(projection.values, projection.scales, input, projection.rows, width,
                         projection.output, stream);
}

template <int Count>
struct Fp8Projections {
    TextFp8Projection entries[Count];
};

template <int Count, int Channels = 1>
__global__ void fp8_batch_packed_kernel(const std::uint8_t* weight,
                                        const __nv_bfloat16* weight_scales,
                                        int out_width, int in_width, int groups,
                                        Fp8Projections<Count> projections) {
    static_assert(text_fp8_group_size % Channels == 0);
    const int lane = threadIdx.x % warp_width;
    const int row = Channels * (blockIdx.x * fp8_packed_warps + threadIdx.x / warp_width);
    if (row >= out_width) {
        return;
    }
    const auto* row_weight = weight + static_cast<std::size_t>(row) * in_width;
    const auto* row_scales =
        weight_scales + static_cast<std::size_t>(row / text_fp8_group_size) * groups;
    constexpr int lanes_per_group = warp_width / fp8_groups_per_warp;
    const int lane_group = lane / lanes_per_group;
    const int lane_col = lane * fp8_pack_bytes;
    float sums[Count][Channels]{};
    for (int group = 0; group < groups; group += fp8_groups_per_warp) {
        const auto col = static_cast<std::size_t>(group) * text_fp8_group_size + lane_col;
        uint4 lhs[Channels];
#pragma unroll
        for (int channel = 0; channel < Channels; ++channel) {
            lhs[channel] = load_fp8_weight(row_weight + col +
                                           static_cast<std::size_t>(channel) * in_width);
        }
        const int scale_index = group + lane_group;
        const float scale = __bfloat162float(row_scales[scale_index]);
#pragma unroll
        for (int index = 0; index < Count; ++index) {
            const auto input = projections.entries[index].input;
            const uint4 rhs = load_128(input.values + col);
            const float input_scale = input.scales[scale_index];
#pragma unroll
            for (int channel = 0; channel < Channels; ++channel) {
                const float scaled = fp8_pack_dot(lhs[channel], rhs) * input_scale;
                sums[index][channel] = fmaf(scaled, scale, sums[index][channel]);
            }
        }
    }
#pragma unroll
    for (int index = 0; index < Count; ++index) {
#pragma unroll
        for (int channel = 0; channel < Channels; ++channel) {
            const float sum = warp_sum(sums[index][channel]);
            if (lane == 0) {
                projections.entries[index].output[row + channel] = __float2bfloat16_rn(sum);
            }
        }
    }
}

template <int Count, int Channels = 1>
cudaError_t launch_fp8_batch(const std::uint8_t* weight, const __nv_bfloat16* weight_scales,
                             int out_width, int in_width, const TextFp8Projection* projections,
                             cudaStream_t stream) {
    Fp8Projections<Count> batch;
    for (int index = 0; index < Count; ++index) {
        batch.entries[index] = projections[index];
    }
    int groups = text_fp8_group_count(in_width);
    void* args[] = {&weight, &weight_scales, &out_width, &in_width, &groups, &batch};
    return cudaLaunchKernel(fp8_batch_packed_kernel<Count, Channels>,
                            divide_up(out_width, fp8_packed_warps * Channels),
                            fp8_packed_warps * warp_width, args, 0, stream);
}

constexpr int mlp_verify_rows = 5;
constexpr int mlp_verify_channels = 4;
constexpr int mlp_verify_steps = 2;
constexpr int mlp_verify_threads = fp8_packed_warps * warp_width;
constexpr int mlp_verify_outputs = mlp_verify_channels * fp8_packed_warps;
constexpr int mlp_verify_groups = mlp_verify_steps * fp8_groups_per_warp;
static_assert(text_fp8_group_size % mlp_verify_channels == 0);
static_assert(decoder::hidden_size % mlp_verify_outputs == 0 &&
              decoder::intermediate_size % mlp_verify_outputs == 0);
static_assert(decoder::hidden_size % (mlp_verify_groups * text_fp8_group_size) == 0 &&
              decoder::intermediate_size % (mlp_verify_groups * text_fp8_group_size) == 0);
static_assert(text_gdn_qkv_width % mlp_verify_outputs == 0 &&
              text_gdn_value_width % mlp_verify_outputs == 0 &&
              text_gdn_value_width % (mlp_verify_groups * text_fp8_group_size) == 0);

bool is_fp8_mlp_or_gdn_shape(int out_width, int in_width) {
    if (in_width == decoder::hidden_size) {
        return out_width > 0 && out_width % mlp_verify_outputs == 0;
    }
    return out_width == decoder::hidden_size &&
           text_fp8_group_count(in_width) % mlp_verify_groups == 0;
}

struct Fp8MlpStage {
    uint4 weight[mlp_verify_channels][fp8_packed_warps][mlp_verify_steps][warp_width];
    uint4 activation[mlp_verify_rows * mlp_verify_steps * warp_width];
};

__device__ void copy_mlp_pack(uint4* target, const std::uint8_t* source) {
    const auto shared = static_cast<unsigned>(__cvta_generic_to_shared(target));
    const auto global = __cvta_generic_to_global(source);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::
                 "r"(shared), "l"(global) : "memory");
}

__device__ void load_mlp_stage(Fp8MlpStage& stage,
                               const std::uint8_t* weight, int in_width, int begin,
                               const Fp8Projections<mlp_verify_rows>& projections) {
    const int warp = threadIdx.x / warp_width;
    const int lane = threadIdx.x % warp_width;
    const auto weight_offset = static_cast<std::size_t>(
        mlp_verify_channels * (blockIdx.x * fp8_packed_warps + warp)) * in_width;
    const auto col = begin + lane * fp8_pack_bytes;
#pragma unroll
    for (int channel = 0; channel < mlp_verify_channels; ++channel) {
#pragma unroll
        for (int step = 0; step < mlp_verify_steps; ++step) {
            copy_mlp_pack(&stage.weight[channel][warp][step][lane],
                          weight + weight_offset + static_cast<std::size_t>(channel) * in_width +
                              col + step * warp_width * fp8_pack_bytes);
        }
    }
    constexpr int packs_per_row = mlp_verify_steps * warp_width;
    for (int index = threadIdx.x; index < mlp_verify_rows * packs_per_row;
         index += mlp_verify_threads) {
        const int item = index / packs_per_row;
        const int pack = index % packs_per_row;
        copy_mlp_pack(&stage.activation[index],
                      projections.entries[item].input.values + begin + pack * fp8_pack_bytes);
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
}

__global__ void fp8_mlp_staged_kernel(const std::uint8_t* weight,
                                      const __nv_bfloat16* weight_scales, int in_width, int groups,
                                      const __grid_constant__
                                          Fp8Projections<mlp_verify_rows> projections) {
    constexpr int lanes_per_group = warp_width / fp8_groups_per_warp;
    const int warp = threadIdx.x / warp_width;
    const int lane = threadIdx.x % warp_width;
    const int row = mlp_verify_channels * (blockIdx.x * fp8_packed_warps + warp);
    const auto* row_scales =
        weight_scales + static_cast<std::size_t>(row / text_fp8_group_size) * groups;
    __shared__ Fp8MlpStage stages[2];
    int current = 0;
    float sums[mlp_verify_rows][mlp_verify_channels]{};
    load_mlp_stage(stages[current], weight, in_width, 0, projections);
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncthreads();
    for (int group = 0; group < groups; group += mlp_verify_groups) {
        if (group + mlp_verify_groups < groups) {
            load_mlp_stage(stages[1 - current], weight, in_width,
                           (group + mlp_verify_groups) * text_fp8_group_size, projections);
        }
        const auto& stage = stages[current];
#pragma unroll
        for (int step = 0; step < mlp_verify_steps; ++step) {
            const int scale_index = group + step * fp8_groups_per_warp + lane / lanes_per_group;
            const float scale = __bfloat162float(row_scales[scale_index]);
            uint4 lhs[mlp_verify_channels];
#pragma unroll
            for (int channel = 0; channel < mlp_verify_channels; ++channel) {
                lhs[channel] = stage.weight[channel][warp][step][lane];
            }
#pragma unroll
            for (int item = 0; item < mlp_verify_rows; ++item) {
                const auto input = projections.entries[item].input;
                const auto rhs = stage.activation[(item * mlp_verify_steps + step) * warp_width +
                                                  lane];
                const float input_scale = input.scales[scale_index];
#pragma unroll
                for (int channel = 0; channel < mlp_verify_channels; ++channel) {
                    const auto left = lhs[channel];
                    const float low = fp8x4_dot(left.y, rhs.y, fp8x4_dot(left.x, rhs.x, 0.0f));
                    const float high = fp8x4_dot(left.w, rhs.w, fp8x4_dot(left.z, rhs.z, 0.0f));
                    const float scaled = (low + high) * input_scale;
                    sums[item][channel] = fmaf(scaled, scale, sums[item][channel]);
                }
            }
        }
        if (group + mlp_verify_groups < groups) {
            asm volatile("cp.async.wait_group 0;" ::: "memory");
            __syncthreads();
        }
        current = 1 - current;
    }
#pragma unroll
    for (int item = 0; item < mlp_verify_rows; ++item) {
#pragma unroll
        for (int channel = 0; channel < mlp_verify_channels; ++channel) {
            const float sum = warp_sum(sums[item][channel]);
            if (lane == 0) {
                projections.entries[item].output[row + channel] = __float2bfloat16_rn(sum);
            }
        }
    }
}

cudaError_t launch_fp8_mlp_staged(const std::uint8_t* weight, const __nv_bfloat16* weight_scales,
                                 int out_width, int in_width,
                                 const TextFp8Projection* projections, cudaStream_t stream) {
    Fp8Projections<mlp_verify_rows> batch;
    for (int index = 0; index < mlp_verify_rows; ++index) {
        batch.entries[index] = projections[index];
    }
    int groups = text_fp8_group_count(in_width);
    void* args[] = {&weight, &weight_scales, &in_width, &groups, &batch};
    return cudaLaunchKernel(fp8_mlp_staged_kernel, out_width / mlp_verify_outputs,
                            mlp_verify_threads, args, 0, stream);
}

}

cudaError_t text_fp8_gemv(const std::uint8_t* weight, const __nv_bfloat16* weight_scales,
                          TextFp8ActivationView input, int out_width, int in_width,
                          __nv_bfloat16* output, cudaStream_t stream) {
    if (weight == nullptr || weight_scales == nullptr || input.values == nullptr ||
        input.scales == nullptr || output == nullptr || out_width <= 0 || in_width <= 0) {
        return cudaErrorInvalidValue;
    }
    int groups = text_fp8_group_count(in_width);
    if (supports_wide_fp8(weight, input, in_width)) {
        return launch_decode_projection<0>(
            in_width, input, SingletonProjections<1>{{{weight, weight_scales, output, out_width}}},
            out_width, stream);
    }
    const bool packed = in_width % 4 == 0 && aligned(weight, 4) && aligned(input.values, 4);
    void* args[] = {&weight, &weight_scales, &input, &out_width, &in_width, &groups, &output};
    return cudaLaunchKernel(packed ? fp8_gemv_kernel<true> : fp8_gemv_kernel<false>,
                            divide_up(out_width, block_warps), block_threads, args, 0, stream);
}

cudaError_t decoder::decode_fp8_project(TextFp8ActivationView input, DecodeFp8Projection projection,
                                        int width, cudaStream_t stream) {
    if (!valid_singleton_buffers(input, width, {projection})) {
        return cudaErrorInvalidValue;
    }
    if (supports_wide_fp8(projection.values, input, width)) {
        const SingletonProjections<1> matrix{{projection}};
        switch (width) {
        case hidden_size:
            return launch_decode_projection<hidden_size>(width, input, matrix, projection.rows,
                                                         stream);
        case text_gqa_q_width:
            return launch_decode_projection<text_gqa_q_width>(width, input, matrix, projection.rows,
                                                              stream);
        case intermediate_size:
            return launch_decode_projection<intermediate_size>(width, input, matrix,
                                                               projection.rows, stream);
        case text_gqa_q_width / 2:
            return launch_decode_projection<text_gqa_q_width / 2>(width, input, matrix,
                                                                  projection.rows, stream);
        case intermediate_size / 2:
            return launch_decode_projection<intermediate_size / 2>(width, input, matrix,
                                                                   projection.rows, stream);
        default:
            break;
        }
    }
    return singleton_unfused(input, projection, width, stream);
}

cudaError_t decoder::decode_fp8_qkv(TextFp8ActivationView input, DecodeFp8Projection query,
                                    DecodeFp8Projection key, DecodeFp8Projection value, int width,
                                    cudaStream_t stream) {
    const SingletonProjections<3> matrices{{query, key, value}};
    if (!valid_singleton_buffers(input, width, matrices.entries)) {
        return cudaErrorInvalidValue;
    }
    const bool tp1 = query.rows == text_gqa_q_gate_width && key.rows == text_gqa_kv_width;
    const bool tp2 = query.rows == text_gqa_q_width && key.rows == text_gqa_kv_width / 2;
    if ((tp1 || tp2) && value.rows == key.rows && width == hidden_size &&
        aligned(input.values, fp8_pack_bytes) && aligned(query.values, fp8_pack_bytes) &&
        aligned(key.values, fp8_pack_bytes) && aligned(value.values, fp8_pack_bytes)) {
        return launch_decode_projection<hidden_size>(width, input, matrices,
                                                     query.rows + key.rows + value.rows, stream);
    }
    for (const auto& projection : matrices.entries) {
        if (const auto status = singleton_unfused(input, projection, width, stream);
            status != cudaSuccess) {
            return status;
        }
    }
    return cudaSuccess;
}

cudaError_t decoder::decode_fp8_gate_up(TextFp8ActivationView input, DecodeFp8Projection gate,
                                        DecodeFp8Projection up, int width, cudaStream_t stream) {
    const bool fused = width == hidden_size &&
                       (gate.rows == intermediate_size || gate.rows == intermediate_size / 2) &&
                       aligned(input.values, fp8_pack_bytes) &&
                       aligned(gate.values, fp8_pack_bytes) && aligned(up.values, fp8_pack_bytes);
    if (gate.rows != up.rows || !valid_singleton_buffers(input, width, {gate, up}, fused)) {
        return cudaErrorInvalidValue;
    }
    if (fused) {
        void* args[] = {&input, &gate, &up};
        return cudaLaunchKernel(fp8_decode_gate_up_kernel, divide_up(gate.rows, fp8_packed_warps),
                                fp8_packed_warps * warp_width, args, 0, stream);
    }
    for (const auto& projection : {gate, up}) {
        if (const auto status = singleton_unfused(input, projection, width, stream);
            status != cudaSuccess) {
            return status;
        }
    }
    return text_swiglu(gate.output, up.output, gate.rows, gate.output, stream);
}

cudaError_t text_fp8_gemm(const std::uint8_t* weight, const __nv_bfloat16* weight_scales,
                          int out_width, int in_width, const TextFp8Projection* projections,
                          int count, cudaStream_t stream) {
    const bool exact = (count == decoder::wide_mtp_lanes || count == decoder::draft_projection_rows)
        && out_width == text_gqa_kv_width && in_width == decoder::hidden_size;
    if (!weight || !weight_scales || !projections || out_width <= 0 || in_width <= 0 ||
        count < 1 || (count > text_fp8_max_batch && !exact)) {
        return cudaErrorInvalidValue;
    }
    bool wide = true;
    for (int index = 0; index < count; ++index) {
        const auto& entry = projections[index];
        if (!entry.input.values || !entry.input.scales || !entry.output) {
            return cudaErrorInvalidValue;
        }
        wide = wide && supports_wide_fp8(weight, entry.input, in_width);
    }
    switch (wide ? count : 0) {
    case decoder::draft_projection_rows:
        return launch_fp8_batch<decoder::draft_projection_rows>(
            weight, weight_scales, out_width, in_width, projections, stream);
    case decoder::wide_mtp_lanes:
        return launch_fp8_batch<decoder::wide_mtp_lanes>(
            weight, weight_scales, out_width, in_width, projections, stream);
    case mlp_verify_rows:
        if (is_fp8_mlp_or_gdn_shape(out_width, in_width)) {
            return launch_fp8_mlp_staged(weight, weight_scales, out_width, in_width,
                                         projections, stream);
        }
        return launch_fp8_batch<5>(weight, weight_scales, out_width, in_width, projections, stream);
    case 4:
        return launch_fp8_batch<4>(weight, weight_scales, out_width, in_width, projections, stream);
    case 3:
        return launch_fp8_batch<3>(weight, weight_scales, out_width, in_width, projections, stream);
    case 2:
        return launch_fp8_batch<2>(weight, weight_scales, out_width, in_width, projections, stream);
    }
    for (int index = 0; index < count; ++index) {
        const auto& entry = projections[index];
        const auto status = text_fp8_gemv(weight, weight_scales, entry.input, out_width, in_width,
                                           entry.output, stream);
        if (status != cudaSuccess) {
            return status;
        }
    }
    return cudaSuccess;
}

cudaError_t text_fp8_decode_rows(const std::uint8_t* weight, const __nv_bfloat16* weight_scales,
                                 TextFp8ActivationView input, int out_width, int in_width,
                                 int rows, __nv_bfloat16* output, cudaStream_t stream) {
    if (!weight || !weight_scales || !input.values || !input.scales || !output ||
        out_width <= 0 || in_width <= 0 || rows <= 0 || rows > 65535) {
        return cudaErrorInvalidValue;
    }
    const TextFp8DecodeTarget target{weight, weight_scales, out_width, output};
    return text_fp8_decode_rows_fused(&target, 1, input, in_width, rows, stream);
}

namespace {

cudaError_t decode_fp8_rows(const TextFp8DecodeTarget* targets, int count,
                            TextFp8ActivationView input, int in_width, int rows,
                            cudaStream_t stream, const Fp8DecodeInputProducer* producer) {
    if (!targets || count < 1 || count > text_decode_max_targets || !input.values ||
        !input.scales || in_width <= 0 || rows <= 0 || rows > 65535) {
        return cudaErrorInvalidValue;
    }
    for (int index = 0; index < count; ++index) {
        const auto& target = targets[index];
        if (!target.weight || !target.weight_scales || !target.output || target.out_width <= 0) {
            return cudaErrorInvalidValue;
        }
    }
    if (const auto status = try_text_fp8_decode_sm100(targets, count, input, in_width, rows,
                                                      stream, producer)) {
        return *status;
    }
    if (producer) {
        const auto status = producer->launch(producer->context, false);
        if (status != cudaSuccess) {
            return status;
        }
    }
    for (int index = 0; index < count; ++index) {
        const auto& target = targets[index];
        const auto status = text_fp8_gemv_rows(target.weight, target.weight_scales, input,
                                               target.out_width, in_width, rows, target.output,
                                               stream);
        if (status != cudaSuccess) {
            return status;
        }
    }
    return cudaSuccess;
}

}

cudaError_t text_fp8_decode_rows_fused(const TextFp8DecodeTarget* targets, int count,
                                       TextFp8ActivationView input, int in_width, int rows,
                                       cudaStream_t stream) {
    return decode_fp8_rows(targets, count, input, in_width, rows, stream, nullptr);
}

cudaError_t decoder::tensor_core_decode_rms_project(
    const __nv_bfloat16* branch, __nv_bfloat16* residual, const __nv_bfloat16* weight,
    int width, int rows, __nv_bfloat16* normalized, TextFp8Activation activation,
    const TextFp8DecodeTarget* targets, int count, cudaStream_t stream) {
    const auto normalize = [&](bool prefetch_weights) {
        const auto launch = prefetch_weights ? RmsQuantizeLaunch::prefetch_decode_weights
                                             : RmsQuantizeLaunch::ordinary;
        return tensor_core_prefill_rms_quantize(branch, residual, weight, width, rows,
                                                normalized, activation, stream, launch);
    };
    const Fp8DecodeInputProducer producer{
        &normalize, [](const void* context, bool prefetch_weights) {
            return (*static_cast<const decltype(normalize)*>(context))(prefetch_weights);
        }};
    return decode_fp8_rows(targets, count, {activation.values, activation.scales}, width, rows,
                           stream, &producer);
}

}

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
