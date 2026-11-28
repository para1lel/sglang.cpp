#include "sglang/text/fp8_projection.cuh"

#include "sglang/decoder_cuda.cuh"
#include "sglang/decoder_internal.h"
#include "sglang/runtime/model/qwen_layers.h"

#include <cstddef>
#include <cuda_fp8.h>

namespace sglang {
namespace {

using namespace fp8_projection;
using namespace text_kernel;

struct RowShape {
    int out_width;
    int in_width;
    int groups;
    int rows;
    int tiles;
};

template <int Rows, int Channels = 1>
__global__ void
fp8_prefill_tile_kernel(const std::uint8_t* weight, const __nv_bfloat16* weight_scales,
                        TextFp8ActivationView input, RowShape shape, __nv_bfloat16* output) {
    const auto [out_width, in_width, groups, rows, tiles] = shape;
    static_assert(text_fp8_group_size % Channels == 0);
    constexpr int lanes_per_group = warp_width / fp8_groups_per_warp;
    const int lane = static_cast<int>(threadIdx.x % warp_width);
    const int block = Channels == 1 ? blockIdx.x / tiles : blockIdx.y;
    const int row = ((block * fp8_packed_warps) + (threadIdx.x / warp_width)) * Channels;
    if (row >= out_width) {
        return;
    }
    const int first = (Channels == 1 ? blockIdx.x % tiles : blockIdx.x) * Rows;
    const auto* row_weight = weight + (static_cast<std::size_t>(row) * in_width);
    const auto* row_scales =
        weight_scales + (static_cast<std::size_t>(row / text_fp8_group_size) * groups);
    const auto* values = input.values + (static_cast<std::size_t>(first) * in_width);
    const auto* scales = input.scales + (static_cast<std::size_t>(first) * groups);
    float sums[Rows][Channels]{};
    for (int group = 0; group < groups; group += fp8_groups_per_warp) {
        const auto col = (static_cast<std::size_t>(group) * text_fp8_group_size) +
                         (static_cast<std::size_t>(lane) * fp8_pack_bytes);
        uint4 lhs[Channels];
#pragma unroll
        for (int channel = 0; channel < Channels; ++channel) {
            lhs[channel] =
                load_fp8_weight(row_weight + col + (static_cast<std::size_t>(channel) * in_width));
        }
        const int scale_index = group + (lane / lanes_per_group);
        const float weight_scale = __bfloat162float(row_scales[scale_index]);
#pragma unroll
        for (int item = 0; item < Rows; ++item) {
            if (first + item < rows) {
                const auto rhs =
                    load_128(values + (static_cast<std::size_t>(item) * in_width) + col);
                const float input_scale =
                    scales[(static_cast<std::size_t>(item) * groups) + scale_index];
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
            auto* result = output + (static_cast<std::size_t>(first + item) * out_width) + row;
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
__device__ __forceinline__ void accumulate_prefill_pairs(const unsigned int (&weights)[Channels],
                                                         const unsigned int (&inputs)[Rows],
                                                         float (&dots)[Rows][Channels]) {
#pragma unroll
    for (int pair = 0; pair < 2; ++pair) {
        const auto shift = static_cast<unsigned>(16 * pair);
        __half2_raw decoded_weights[Channels];
        __half2_raw decoded_inputs[Rows];
#pragma unroll
        for (int channel = 0; channel < Channels; ++channel) {
            decoded_weights[channel] = __nv_cvt_fp8x2_to_halfraw2(
                static_cast<std::uint16_t>(weights[channel] >> shift), __NV_E4M3);
        }
#pragma unroll
        for (int item = 0; item < Rows; ++item) {
            decoded_inputs[item] = __nv_cvt_fp8x2_to_halfraw2(
                static_cast<std::uint16_t>(inputs[item] >> shift), __NV_E4M3);
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

template <unsigned int uint4::* Word, int Rows, int Channels>
__device__ __forceinline__ void accumulate_prefill_word(const uint4 (&weights)[Channels],
                                                        const uint4 (&inputs)[Rows],
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
__device__ __forceinline__ void
fp8_prefill_reuse_tile(const std::uint8_t* weight, const __nv_bfloat16* weight_scales,
                       TextFp8ActivationView input, RowShape shape, __nv_bfloat16* output) {
    const int out_width = shape.out_width;
    const int in_width = shape.in_width;
    const int groups = shape.groups;
    const int rows = shape.rows;
    static_assert(text_fp8_group_size % Channels == 0);
    constexpr int lanes_per_group = warp_width / fp8_groups_per_warp;
    const int lane = static_cast<int>(threadIdx.x % warp_width);
    const int row = ((blockIdx.y * fp8_packed_warps) + (threadIdx.x / warp_width)) * Channels;
    if (row >= out_width) {
        return;
    }
    const int first = blockIdx.x * Rows;
    const auto* row_weight = weight + (static_cast<std::size_t>(row) * in_width);
    const auto* row_scales =
        weight_scales + (static_cast<std::size_t>(row / text_fp8_group_size) * groups);
    float sums[Rows][Channels]{};
    int input_rows[Rows];
#pragma unroll
    for (int item = 0; item < Rows; ++item) {
        input_rows[item] = min(first + item, rows - 1);
    }
    for (int group = 0; group < groups; group += fp8_groups_per_warp) {
        const auto col = (static_cast<std::size_t>(group) * text_fp8_group_size) +
                         (static_cast<std::size_t>(lane) * fp8_pack_bytes);
        uint4 lhs[Channels];
        uint4 rhs[Rows];
#pragma unroll
        for (int channel = 0; channel < Channels; ++channel) {
            lhs[channel] =
                load_fp8_weight(row_weight + col + (static_cast<std::size_t>(channel) * in_width));
        }
#pragma unroll
        for (int item = 0; item < Rows; ++item) {
            rhs[item] = load_128(input.values +
                                 (static_cast<std::size_t>(input_rows[item]) * in_width) + col);
        }
        float low[Rows][Channels]{};
        float high[Rows][Channels]{};
        accumulate_prefill_word<&uint4::x>(lhs, rhs, low);
        accumulate_prefill_word<&uint4::y>(lhs, rhs, low);
        accumulate_prefill_word<&uint4::z>(lhs, rhs, high);
        accumulate_prefill_word<&uint4::w>(lhs, rhs, high);
        const int scale_index = group + (lane / lanes_per_group);
        const float weight_scale = __bfloat162float(row_scales[scale_index]);
#pragma unroll
        for (int item = 0; item < Rows; ++item) {
            const float input_scale =
                input.scales[(static_cast<std::size_t>(input_rows[item]) * groups) + scale_index];
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
            auto* result = output + (static_cast<std::size_t>(first + item) * out_width) + row;
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
__global__ void
fp8_prefill_reuse_kernel(const std::uint8_t* weight, const __nv_bfloat16* weight_scales,
                         TextFp8ActivationView input, RowShape shape, __nv_bfloat16* output) {
    fp8_prefill_reuse_tile<Rows, Channels>(weight, weight_scales, input, shape, output);
}

struct Fp8PrefillMatrix {
    const std::uint8_t* weight;
    const __nv_bfloat16* scales;
    __nv_bfloat16* output;
};

template <int Rows, int Channels>
__global__ void fp8_prefill_reuse_pair_kernel(Fp8PrefillMatrix first, Fp8PrefillMatrix second,
                                              TextFp8ActivationView input, RowShape shape) {
    const auto matrix = blockIdx.z == 0 ? first : second;
    fp8_prefill_reuse_tile<Rows, Channels>(matrix.weight, matrix.scales, input, shape,
                                           matrix.output);
}

template <int Count>
cudaError_t launch_fp8_rows(const std::uint8_t* weight, const __nv_bfloat16* weight_scales,
                            TextFp8ActivationView input, int out_width, int in_width, int rows,
                            __nv_bfloat16* output, cudaStream_t stream) {
    RowShape shape{
        out_width, in_width, text_fp8_group_count(in_width), rows, divide_up(rows, Count),
    };
    void* args[] = {
        static_cast<void*>(&weight), static_cast<void*>(&weight_scales), &input, &shape,
        static_cast<void*>(&output),
    };
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
    const auto blocks =
        static_cast<std::uint64_t>(divide_up(out_width, fp8_packed_warps)) * shape.tiles;
    if (blocks > 2147483647U) {
        return cudaErrorInvalidConfiguration;
    }
    return cudaLaunchKernel(fp8_prefill_tile_kernel<Count>, static_cast<unsigned int>(blocks),
                            fp8_packed_warps * warp_width, args, 0, stream);
}

}

cudaError_t text_fp8_gemv_rows(const std::uint8_t* weight, const __nv_bfloat16* weight_scales,
                               TextFp8ActivationView input, int out_width, int in_width, int rows,
                               __nv_bfloat16* output, cudaStream_t stream) {
    if (weight == nullptr || weight_scales == nullptr || input.values == nullptr ||
        input.scales == nullptr || output == nullptr || out_width <= 0 || in_width <= 0 ||
        rows <= 0 || rows > 65535) {
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
        return launch_fp8_rows<8>(weight, weight_scales, input, out_width, in_width, rows, output,
                                  stream);
    }
    const int groups = text_fp8_group_count(in_width);
    for (int row = 0; row < rows; ++row) {
        const TextFp8ActivationView source{
            input.values + (static_cast<std::size_t>(row) * in_width),
            input.scales + (static_cast<std::size_t>(row) * groups),
        };
        const auto status =
            text_fp8_gemv(weight, weight_scales, source, out_width, in_width,
                          output + (static_cast<std::size_t>(row) * out_width), stream);
        if (status != cudaSuccess) {
            return status;
        }
    }
    return cudaSuccess;
}

cudaError_t decoder::mtp_seed_project_kv(const Fp8Weight& key, const Fp8Weight& value,
                                         TextFp8ActivationView input, int rows, __nv_bfloat16* keys,
                                         __nv_bfloat16* values, cudaStream_t stream) {
    constexpr int tile_rows = 4;
    constexpr int channels = 8;
    constexpr int threads = fp8_packed_warps * warp_width;
    static_assert(text_gqa_kv_width % (fp8_packed_warps * channels) == 0);
    static_assert(hidden_size % text_fp8_group_size == 0);
    if (rows < 2 || rows > text_attention_max_tokens || input.values == nullptr ||
        input.scales == nullptr || key.values == nullptr || value.values == nullptr ||
        key.scales == nullptr || value.scales == nullptr || keys == nullptr || values == nullptr ||
        key.rows != text_gqa_kv_width || value.rows != text_gqa_kv_width ||
        key.columns != hidden_size || value.columns != hidden_size ||
        !supports_wide_fp8(key.values, input, hidden_size) ||
        !supports_wide_fp8(value.values, input, hidden_size)) {
        return cudaErrorInvalidValue;
    }
    RowShape shape{text_gqa_kv_width, hidden_size, text_fp8_group_count(hidden_size), rows, 0};
    Fp8PrefillMatrix key_matrix{key.values, key.scales, keys};
    Fp8PrefillMatrix value_matrix{value.values, value.scales, values};
    void* args[] = {&key_matrix, &value_matrix, &input, &shape};
    const dim3 grid(divide_up(rows, tile_rows), shape.out_width / (fp8_packed_warps * channels), 2);
    return cudaLaunchKernel(fp8_prefill_reuse_pair_kernel<tile_rows, channels>, grid, threads, args,
                            0, stream);
}

}
