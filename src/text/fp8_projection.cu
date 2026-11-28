#include "sglang/text_ops.h"

#include "sglang/decoder_cuda.cuh"
#include "sglang/decoder_internal.h"
#include "sglang/decoder_mtp.h"
#include "sglang/text/decode_proj.h"
#include "sglang/text/mtp/draft_stream.h"
#include "sglang/text/text_cuda_primitives.cuh"

#include <algorithm>
#include <cstddef>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

namespace sglang {
namespace {

using namespace text_kernel;

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

__device__ float fp8x4_dot(unsigned int lhs_bits, unsigned int rhs_bits, float sum) {
#pragma unroll
    for (int pair = 0; pair < 2; ++pair) {
        const auto left = __nv_cvt_fp8x2_to_halfraw2(
            static_cast<std::uint16_t>(lhs_bits >> (16 * pair)), __NV_E4M3);
        const auto right = __nv_cvt_fp8x2_to_halfraw2(
            static_cast<std::uint16_t>(rhs_bits >> (16 * pair)), __NV_E4M3);
        asm("fma.rn.f32.f16 %0, %1, %2, %0;" : "+f"(sum) : "h"(left.x), "h"(right.x));
        asm("fma.rn.f32.f16 %0, %1, %2, %0;" : "+f"(sum) : "h"(left.y), "h"(right.y));
    }
    return sum;
}

__device__ float fp8_pack_dot(uint4 lhs, uint4 rhs) {
    const float low = fp8x4_dot(lhs.y, rhs.y, fp8x4_dot(lhs.x, rhs.x, 0.0F));
    const float high = fp8x4_dot(lhs.w, rhs.w, fp8x4_dot(lhs.z, rhs.z, 0.0F));
    return low + high;
}

constexpr int fp8_pack_bytes = 16;
constexpr int fp8_packed_warps = 4;
constexpr int fp8_groups_per_warp = warp_width * fp8_pack_bytes / text_fp8_group_size;

bool supports_wide_fp8(const std::uint8_t* weight, TextFp8ActivationView input, int in_width) {
    return in_width % (fp8_groups_per_warp * text_fp8_group_size) == 0 &&
           aligned(weight, fp8_pack_bytes) && aligned(input.values, fp8_pack_bytes);
}

__device__ uint4 load_fp8_weight(const void* address) {
    uint4 bits;
    const auto global = __cvta_generic_to_global(address);
    asm("ld.global.cs.v4.b32 {%0, %1, %2, %3}, [%4];"
        : "=r"(bits.x), "=r"(bits.y), "=r"(bits.z), "=r"(bits.w)
        : "l"(global) : "memory");
    return bits;
}

__global__ void fp8_gemv_packed_kernel(const std::uint8_t* weight,
                                       const __nv_bfloat16* weight_scales,
                                       TextFp8ActivationView input, int out_width, int in_width,
                                       int groups, __nv_bfloat16* output) {
    const int lane = threadIdx.x % warp_width;
    const int row = blockIdx.x * fp8_packed_warps + threadIdx.x / warp_width;
    if (row >= out_width) {
        return;
    }
    const auto* row_weight = weight + static_cast<std::size_t>(row) * in_width;
    const auto* row_scales =
        weight_scales + static_cast<std::size_t>(row / text_fp8_group_size) * groups;
    constexpr int lanes_per_group = warp_width / fp8_groups_per_warp;
    const int lane_group = lane / lanes_per_group;
    const int lane_col = lane * fp8_pack_bytes;
    float sum = 0.0f;
    for (int group = 0; group < groups; group += fp8_groups_per_warp) {
        const auto col = static_cast<std::size_t>(group) * text_fp8_group_size + lane_col;
        const uint4 rhs = load_128(input.values + col);
        const uint4 lhs = load_fp8_weight(row_weight + col);
        const int scale_index = group + lane_group;
        const float scaled = fp8_pack_dot(lhs, rhs) * input.scales[scale_index];
        sum = fmaf(scaled, __bfloat162float(row_scales[scale_index]), sum);
    }
    sum = warp_sum(sum);
    if (lane == 0) {
        output[row] = __float2bfloat16_rn(sum);
    }
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

template <int Rows, int Channels = 1>
__global__ void fp8_prefill_tile_kernel(const std::uint8_t* weight,
                                       const __nv_bfloat16* weight_scales,
                                       TextFp8ActivationView input, int out_width, int in_width,
                                       int groups, int rows, int tiles, __nv_bfloat16* output) {
    static_assert(text_fp8_group_size % Channels == 0);
    constexpr int lanes_per_group = warp_width / fp8_groups_per_warp;
    const int lane = threadIdx.x % warp_width;
    const int block = Channels == 1 ? blockIdx.x / tiles : blockIdx.y;
    const int row = (block * fp8_packed_warps + threadIdx.x / warp_width) * Channels;
    if (row >= out_width) {
        return;
    }
    const int first = (Channels == 1 ? blockIdx.x % tiles : blockIdx.x) * Rows;
    const auto* row_weight = weight + static_cast<std::size_t>(row) * in_width;
    const auto* row_scales =
        weight_scales + static_cast<std::size_t>(row / text_fp8_group_size) * groups;
    const auto* values = input.values + static_cast<std::size_t>(first) * in_width;
    const auto* scales = input.scales + static_cast<std::size_t>(first) * groups;
    float sums[Rows][Channels]{};
    for (int group = 0; group < groups; group += fp8_groups_per_warp) {
        const auto col = static_cast<std::size_t>(group) * text_fp8_group_size +
                         lane * fp8_pack_bytes;
        uint4 lhs[Channels];
#pragma unroll
        for (int channel = 0; channel < Channels; ++channel) {
            lhs[channel] = load_fp8_weight(row_weight + col +
                                           static_cast<std::size_t>(channel) * in_width);
        }
        const int scale_index = group + lane / lanes_per_group;
        const float weight_scale = __bfloat162float(row_scales[scale_index]);
#pragma unroll
        for (int item = 0; item < Rows; ++item) {
            if (first + item < rows) {
                const auto rhs = load_128(values + static_cast<std::size_t>(item) * in_width + col);
                const float input_scale = scales[static_cast<std::size_t>(item) * groups +
                                                 scale_index];
#pragma unroll
                for (int channel = 0; channel < Channels; ++channel) {
                    const float scaled = fp8_pack_dot(lhs[channel], rhs) * input_scale;
                    sums[item][channel] = fmaf(scaled, weight_scale, sums[item][channel]);
                }
            }
        }
    }
#pragma unroll
    for (int item = 0; item < Rows; ++item) {
        if (first + item < rows) {
            auto* result = output + static_cast<std::size_t>(first + item) * out_width + row;
#pragma unroll
            for (int channel = 0; channel < Channels; ++channel) {
                const float sum = warp_sum(sums[item][channel]);
                if (lane == 0) {
                    result[channel] = __float2bfloat16_rn(sum);
                }
            }
        }
    }
}

__device__ __forceinline__ float accumulate_prefill_pair(__half2_raw lhs, __half2_raw rhs,
                                                        float sum) {
    asm("fma.rn.f32.f16 %0, %1, %2, %0;" : "+f"(sum) : "h"(lhs.x), "h"(rhs.x));
    asm("fma.rn.f32.f16 %0, %1, %2, %0;" : "+f"(sum) : "h"(lhs.y), "h"(rhs.y));
    return sum;
}

template <int Rows, int Channels>
__device__ __forceinline__ void accumulate_prefill_pairs(
    const unsigned int (&weights)[Channels], const unsigned int (&inputs)[Rows],
    float (&dots)[Rows][Channels]) {
#pragma unroll
    for (int pair = 0; pair < 2; ++pair) {
        __half2_raw decoded_weights[Channels];
        __half2_raw decoded_inputs[Rows];
#pragma unroll
        for (int channel = 0; channel < Channels; ++channel) {
            decoded_weights[channel] = __nv_cvt_fp8x2_to_halfraw2(
                static_cast<std::uint16_t>(weights[channel] >> (16 * pair)), __NV_E4M3);
        }
#pragma unroll
        for (int item = 0; item < Rows; ++item) {
            decoded_inputs[item] = __nv_cvt_fp8x2_to_halfraw2(
                static_cast<std::uint16_t>(inputs[item] >> (16 * pair)), __NV_E4M3);
        }
#pragma unroll
        for (int item = 0; item < Rows; ++item) {
#pragma unroll
            for (int channel = 0; channel < Channels; ++channel) {
                dots[item][channel] = accumulate_prefill_pair(
                    decoded_weights[channel], decoded_inputs[item], dots[item][channel]);
            }
        }
    }
}

template <unsigned int uint4::*Word, int Rows, int Channels>
__device__ __forceinline__ void accumulate_prefill_word(
    const uint4 (&weights)[Channels], const uint4 (&inputs)[Rows],
    float (&dots)[Rows][Channels]) {
    unsigned int weight_words[Channels];
    unsigned int input_words[Rows];
#pragma unroll
    for (int channel = 0; channel < Channels; ++channel) {
        weight_words[channel] = weights[channel].*Word;
    }
#pragma unroll
    for (int item = 0; item < Rows; ++item) {
        input_words[item] = inputs[item].*Word;
    }
    accumulate_prefill_pairs(weight_words, input_words, dots);
}

template <int Rows, int Channels>
__device__ __forceinline__ void fp8_prefill_reuse_tile(
    const std::uint8_t* weight, const __nv_bfloat16* weight_scales,
    TextFp8ActivationView input, int out_width, int in_width, int groups, int rows,
    __nv_bfloat16* output) {
    static_assert(text_fp8_group_size % Channels == 0);
    constexpr int lanes_per_group = warp_width / fp8_groups_per_warp;
    const int lane = threadIdx.x % warp_width;
    const int row = (blockIdx.y * fp8_packed_warps + threadIdx.x / warp_width) * Channels;
    if (row >= out_width) {
        return;
    }
    const int first = blockIdx.x * Rows;
    const auto* row_weight = weight + static_cast<std::size_t>(row) * in_width;
    const auto* row_scales =
        weight_scales + static_cast<std::size_t>(row / text_fp8_group_size) * groups;
    float sums[Rows][Channels]{};
    int input_rows[Rows];
#pragma unroll
    for (int item = 0; item < Rows; ++item) {
        input_rows[item] = min(first + item, rows - 1);
    }
    for (int group = 0; group < groups; group += fp8_groups_per_warp) {
        const auto col = static_cast<std::size_t>(group) * text_fp8_group_size +
                         lane * fp8_pack_bytes;
        uint4 lhs[Channels];
        uint4 rhs[Rows];
#pragma unroll
        for (int channel = 0; channel < Channels; ++channel) {
            lhs[channel] = load_fp8_weight(row_weight + col +
                                           static_cast<std::size_t>(channel) * in_width);
        }
#pragma unroll
        for (int item = 0; item < Rows; ++item) {
            rhs[item] = load_128(input.values +
                                 static_cast<std::size_t>(input_rows[item]) * in_width + col);
        }
        float low[Rows][Channels]{};
        float high[Rows][Channels]{};
        accumulate_prefill_word<&uint4::x>(lhs, rhs, low);
        accumulate_prefill_word<&uint4::y>(lhs, rhs, low);
        accumulate_prefill_word<&uint4::z>(lhs, rhs, high);
        accumulate_prefill_word<&uint4::w>(lhs, rhs, high);
        const int scale_index = group + lane / lanes_per_group;
        const float weight_scale = __bfloat162float(row_scales[scale_index]);
#pragma unroll
        for (int item = 0; item < Rows; ++item) {
            const float input_scale =
                input.scales[static_cast<std::size_t>(input_rows[item]) * groups + scale_index];
#pragma unroll
            for (int channel = 0; channel < Channels; ++channel) {
                const float scaled = (low[item][channel] + high[item][channel]) * input_scale;
                sums[item][channel] = fmaf(scaled, weight_scale, sums[item][channel]);
            }
        }
    }
#pragma unroll
    for (int item = 0; item < Rows; ++item) {
        if (first + item < rows) {
            auto* result = output + static_cast<std::size_t>(first + item) * out_width + row;
#pragma unroll
            for (int channel = 0; channel < Channels; ++channel) {
                const float sum = warp_sum(sums[item][channel]);
                if (lane == 0) {
                    result[channel] = __float2bfloat16_rn(sum);
                }
            }
        }
    }
}

template <int Rows, int Channels>
__global__ void fp8_prefill_reuse_kernel(const std::uint8_t* weight,
                                        const __nv_bfloat16* weight_scales,
                                        TextFp8ActivationView input, int out_width, int in_width,
                                        int groups, int rows, int, __nv_bfloat16* output) {
    fp8_prefill_reuse_tile<Rows, Channels>(weight, weight_scales, input, out_width, in_width,
                                          groups, rows, output);
}

struct Fp8PrefillMatrix {
    const std::uint8_t* weight;
    const __nv_bfloat16* scales;
    __nv_bfloat16* output;
};

template <int Rows, int Channels>
__global__ void fp8_prefill_reuse_pair_kernel(Fp8PrefillMatrix first, Fp8PrefillMatrix second,
                                             TextFp8ActivationView input, int out_width,
                                             int in_width, int groups, int rows) {
    const auto matrix = blockIdx.z == 0 ? first : second;
    fp8_prefill_reuse_tile<Rows, Channels>(matrix.weight, matrix.scales, input, out_width,
                                          in_width, groups, rows, matrix.output);
}

template <int Count>
cudaError_t launch_fp8_rows(const std::uint8_t* weight, const __nv_bfloat16* weight_scales,
                            TextFp8ActivationView input, int out_width, int in_width, int rows,
                            __nv_bfloat16* output, cudaStream_t stream) {
    int groups = text_fp8_group_count(in_width);
    int tiles = divide_up(rows, Count);
    void* args[] = {&weight, &weight_scales, &input, &out_width, &in_width,
                     &groups, &rows, &tiles, &output};
    if constexpr (Count == 8) {
        constexpr int tile_rows = 4;
        constexpr int channels = 4;
        if ((in_width == decoder::hidden_size || in_width == decoder::intermediate_size ||
             in_width == text_gqa_q_width || in_width == text_gqa_q_width / 2 ||
             in_width == decoder::intermediate_size / 2) &&
            out_width % channels == 0 && out_width <= 65535) {
            constexpr int reuse_channels = 8;
            if (rows >= 64 && rows <= 128 && out_width % reuse_channels == 0) {
                const dim3 grid(divide_up(rows, tile_rows),
                                divide_up(out_width, fp8_packed_warps * reuse_channels));
                return cudaLaunchKernel(fp8_prefill_reuse_kernel<tile_rows, reuse_channels>, grid,
                                        fp8_packed_warps * warp_width, args, 0, stream);
            }
            const dim3 grid(divide_up(rows, tile_rows),
                            divide_up(out_width, fp8_packed_warps * channels));
            return cudaLaunchKernel(fp8_prefill_tile_kernel<tile_rows, channels>, grid,
                                    fp8_packed_warps * warp_width, args, 0, stream);
        }
    }
    const auto blocks = static_cast<std::uint64_t>(divide_up(out_width, fp8_packed_warps)) * tiles;
    if (blocks > 2147483647u) {
        return cudaErrorInvalidConfiguration;
    }
    return cudaLaunchKernel(fp8_prefill_tile_kernel<Count>, static_cast<unsigned int>(blocks),
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
    const bool wide = supports_wide_fp8(weight, input, in_width);
    const bool packed = in_width % 4 == 0 && aligned(weight, 4) && aligned(input.values, 4);
    void* args[] = {&weight, &weight_scales, &input, &out_width, &in_width, &groups, &output};
    const auto kernel = wide ? fp8_gemv_packed_kernel
                             : (packed ? fp8_gemv_kernel<true> : fp8_gemv_kernel<false>);
    const int rows_per_block = wide ? fp8_packed_warps : block_warps;
    return cudaLaunchKernel(kernel, divide_up(out_width, rows_per_block),
                            rows_per_block * warp_width, args, 0, stream);
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
cudaError_t text_fp8_gemv_rows(const std::uint8_t* weight, const __nv_bfloat16* weight_scales,
                               TextFp8ActivationView input, int out_width, int in_width, int rows,
                               __nv_bfloat16* output, cudaStream_t stream) {
    if (!weight || !weight_scales || !input.values || !input.scales || !output ||
        out_width <= 0 || in_width <= 0 || rows <= 0 || rows > 65535) {
        return cudaErrorInvalidValue;
    }
    if (rows > 1 && supports_wide_fp8(weight, input, in_width)) {
        if (rows <= 8 || (rows <= 16 && (out_width <= 1024 || in_width >= 16384))) {
            return launch_fp8_rows<2>(weight, weight_scales, input, out_width, in_width, rows,
                                      output, stream);
        }
        if (rows < 64 || out_width <= 1024) {
            return launch_fp8_rows<4>(weight, weight_scales, input, out_width, in_width, rows,
                                      output, stream);
        }
        return launch_fp8_rows<8>(weight, weight_scales, input, out_width, in_width, rows,
                                  output, stream);
    }
    const int groups = text_fp8_group_count(in_width);
    for (int row = 0; row < rows; ++row) {
        const TextFp8ActivationView source{
            input.values + static_cast<std::size_t>(row) * in_width,
            input.scales + static_cast<std::size_t>(row) * groups};
        const auto status = text_fp8_gemv(weight, weight_scales, source, out_width, in_width,
                                          output + static_cast<std::size_t>(row) * out_width,
                                          stream);
        if (status != cudaSuccess) {
            return status;
        }
    }
    return cudaSuccess;
}

cudaError_t decoder::mtp_seed_project_kv(const Fp8Weight& key, const Fp8Weight& value,
                                        TextFp8ActivationView input, int rows,
                                        __nv_bfloat16* keys, __nv_bfloat16* values,
                                        cudaStream_t stream) {
    constexpr int tile_rows = 4;
    constexpr int channels = 8;
    constexpr int threads = fp8_packed_warps * warp_width;
    static_assert(text_gqa_kv_width % (fp8_packed_warps * channels) == 0);
    static_assert(hidden_size % text_fp8_group_size == 0);
    if (rows < 2 || rows > text_attention_max_tokens || !input.values || !input.scales ||
        !key.values || !value.values || !key.scales || !value.scales || !keys || !values ||
        key.rows != text_gqa_kv_width || value.rows != text_gqa_kv_width ||
        key.columns != hidden_size || value.columns != hidden_size ||
        !supports_wide_fp8(key.values, input, hidden_size) ||
        !supports_wide_fp8(value.values, input, hidden_size)) {
        return cudaErrorInvalidValue;
    }
    int out_width = text_gqa_kv_width;
    int in_width = hidden_size;
    int groups = text_fp8_group_count(in_width);
    Fp8PrefillMatrix key_matrix{key.values, key.scales, keys};
    Fp8PrefillMatrix value_matrix{value.values, value.scales, values};
    void* args[] = {&key_matrix, &value_matrix, &input, &out_width, &in_width, &groups, &rows};
    const dim3 grid(divide_up(rows, tile_rows), out_width / (fp8_packed_warps * channels), 2);
    return cudaLaunchKernel(fp8_prefill_reuse_pair_kernel<tile_rows, channels>, grid, threads,
                            args, 0, stream);
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
