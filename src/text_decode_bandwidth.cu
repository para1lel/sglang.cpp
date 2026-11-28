#include "sglang/decoder_cuda.cuh"
#include "sglang/decoder_verify.h"
#include "sglang/text/decode_proj.h"
#include "sglang/text/text_cuda_primitives.cuh"
#include "sglang/text_ops.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <math_constants.h>

namespace sglang {
namespace {

using namespace text_kernel;

constexpr int prefill_gate_tile_rows = 4;
constexpr int prefill_gate_warps = 4;

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

template <int Channels, bool FullTile>
__global__ void bf16_gdn_prefill_gates(const __nv_bfloat16* weight_b,
                                      const __nv_bfloat16* weight_a,
                                      const __nv_bfloat16* input, int rows,
                                      __nv_bfloat16* output_b, __nv_bfloat16* output_a) {
    constexpr int width = decoder::hidden_size;
    const int lane = threadIdx.x % warp_width;
    const int channel = blockIdx.x * prefill_gate_warps + threadIdx.x / warp_width;
    weight_b += channel * width;
    weight_a += channel * width;
    const int first_row = blockIdx.y * prefill_gate_tile_rows;
    Bf16Dot dots_b[prefill_gate_tile_rows];
    Bf16Dot dots_a[prefill_gate_tile_rows];
    const int tile_rows =
        FullTile ? prefill_gate_tile_rows : min(prefill_gate_tile_rows, rows - first_row);
#pragma unroll 4
    for (int col = lane * 8; col < width; col += warp_width * 8) {
        const uint4 packed_b = load_128(weight_b + col);
        const uint4 packed_a = load_128(weight_a + col);
#pragma unroll
        for (int item = 0; item < prefill_gate_tile_rows; ++item) {
            if (!FullTile && item >= tile_rows) {
                break;
            }
            const auto offset = static_cast<std::ptrdiff_t>(first_row + item) * width;
            const uint4 packed_input = load_128(input + offset + col);
            accumulate_packed_gate(dots_b[item], packed_b, packed_input);
            accumulate_packed_gate(dots_a[item], packed_a, packed_input);
        }
    }
#pragma unroll
    for (int item = 0; item < prefill_gate_tile_rows; ++item) {
        if (!FullTile && item >= tile_rows) {
            break;
        }
        const float result_b = warp_sum(dots_b[item].sum());
        const float result_a = warp_sum(dots_a[item].sum());
        if (lane == 0) {
            const int offset = ((first_row + item) * Channels) + channel;
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

__device__ void quantize_draft_head_row(const __nv_bfloat16* source, std::uint8_t* values,
                                        float* maximum_out, int lane) {
    constexpr int width = decoder::hidden_size;
    auto* destination = reinterpret_cast<std::int8_t*>(values);
    float maximum = 0;
    bool invalid = false;
    for (int column = lane; column < width; column += warp_width) {
        const float value = __bfloat162float(source[column]);
        invalid |= !isfinite(value);
        maximum = fmaxf(maximum, fabsf(value));
    }
    invalid = __any_sync(warp_mask, invalid);
    maximum = __shfl_sync(warp_mask, warp_max(maximum), 0);
    const bool use_inverse = maximum >= 0x1p-120f && maximum <= 0x1p120f;
    const float inverse = use_inverse ? 127.0f / maximum : 0;
    for (int column = lane; column < width; column += warp_width) {
        int quantized = 0;
        if (!invalid && maximum != 0) {
            const float value = __bfloat162float(source[column]);
            const float scaled = use_inverse ? value * inverse : 127.0f * (value / maximum);
            quantized = __float2int_rn(scaled);
        }
        destination[column] = quantized;
    }
    if (lane == 0) {
        *maximum_out = invalid ? CUDART_NAN_F : maximum;
    }
}

__global__ void quantize_draft_head_rows(const __nv_bfloat16* input, int rows,
                                        std::uint8_t* values, float* maxima) {
    const int row = blockIdx.x * (blockDim.x / warp_width) + threadIdx.x / warp_width;
    if (row < rows) {
        const auto offset = static_cast<std::size_t>(row) * decoder::hidden_size;
        quantize_draft_head_row(input + offset, values + offset, maxima + row,
                                threadIdx.x % warp_width);
    }
}

template <int Count>
struct DraftHeadArguments {
    decoder::DraftInt8Projection rows[Count];
};

template <int Count>
__global__ void quantize_draft_head_batch(const __grid_constant__ DraftHeadArguments<Count> args) {
    const auto& row = args.rows[threadIdx.x / warp_width];
    quantize_draft_head_row(row.input, row.values, row.maximum, threadIdx.x % warp_width);
}

__device__ int dot_signed_bytes(uint4 weight, uint4 input, int sum) {
    sum = __dp4a(static_cast<int>(weight.x), static_cast<int>(input.x), sum);
    sum = __dp4a(static_cast<int>(weight.y), static_cast<int>(input.y), sum);
    sum = __dp4a(static_cast<int>(weight.z), static_cast<int>(input.z), sum);
    return __dp4a(static_cast<int>(weight.w), static_cast<int>(input.w), sum);
}

__device__ __nv_bfloat16 draft_head_logit(int sum, float weight_maximum, float input_maximum) {
    constexpr int quantization_denominator = 127 * 127;
    return __double2bfloat16(static_cast<double>(sum) * weight_maximum * input_maximum /
                            quantization_denominator);
}

__global__ void draft_head_int8_kernel(const std::uint8_t* weights, const float* weight_maxima,
                                      const std::uint8_t* input, const float* input_maximum,
                                      __nv_bfloat16* output) {
    constexpr int width = decoder::hidden_size;
    static_assert(width * 127 * 127 <= 2147483647);
    static_assert(decoder::vocab_size % (2 * block_warps) == 0);
    const int lane = threadIdx.x % warp_width;
    const int row = 2 * (blockIdx.x * block_warps + threadIdx.x / warp_width);
    const auto* weight = weights + static_cast<std::size_t>(row) * width;
    int first = 0;
    int second = 0;
    for (int column = lane * 16; column < width; column += warp_width * 16) {
        const auto vector = load_128(input + column);
        first = dot_signed_bytes(load_streaming_128(weight + column), vector, first);
        second = dot_signed_bytes(load_streaming_128(weight + width + column), vector, second);
    }
    first = __reduce_add_sync(warp_mask, first);
    second = __reduce_add_sync(warp_mask, second);
    if (lane == 0) {
        const float maximum = *input_maximum;
        output[row] = draft_head_logit(first, weight_maxima[row], maximum);
        output[row + 1] = draft_head_logit(second, weight_maxima[row + 1], maximum);
    }
}

template <int Count>
__global__ void draft_head_int8_batch_kernel(
    const std::uint8_t* weights, const float* weight_maxima,
    const __grid_constant__ DraftHeadArguments<Count> args) {
    constexpr int width = decoder::hidden_size;
    const int lane = threadIdx.x % warp_width;
    const int row = 2 * (blockIdx.x * block_warps + threadIdx.x / warp_width);
    const auto* weight = weights + static_cast<std::size_t>(row) * width;
    int first[Count]{};
    int second[Count]{};
    for (int column = lane * 16; column < width; column += warp_width * 16) {
        const auto lhs = load_streaming_128(weight + column);
        const auto rhs = load_streaming_128(weight + width + column);
        #pragma unroll
        for (int index = 0; index < Count; ++index) {
            const auto input = load_128(args.rows[index].values + column);
            first[index] = dot_signed_bytes(lhs, input, first[index]);
            second[index] = dot_signed_bytes(rhs, input, second[index]);
        }
    }
    #pragma unroll
    for (int index = 0; index < Count; ++index) {
        const int sum_first = __reduce_add_sync(warp_mask, first[index]);
        const int sum_second = __reduce_add_sync(warp_mask, second[index]);
        if (lane == 0) {
            const auto& projection = args.rows[index];
            const float maximum = *projection.maximum;
            projection.output[row] = draft_head_logit(sum_first, weight_maxima[row], maximum);
            projection.output[row + 1] =
                draft_head_logit(sum_second, weight_maxima[row + 1], maximum);
        }
    }
}

constexpr int draft_four_lane_threads = 128;
constexpr int draft_four_lane_rows_per_warp = 4;
constexpr int draft_four_lane_rows_per_block =
    draft_four_lane_rows_per_warp * (draft_four_lane_threads / warp_width);
static_assert(decoder::vocab_size % draft_four_lane_rows_per_block == 0);

__global__ __launch_bounds__(draft_four_lane_threads) void draft_head_int8_four_lane_kernel(
    const std::uint8_t* weights, const float* maxima,
    const __grid_constant__ DraftHeadArguments<4> projection) {
    constexpr int width = decoder::hidden_size;
    constexpr int rows = draft_four_lane_rows_per_warp;
    const int lane = threadIdx.x % warp_width;
    const int token = rows * (blockIdx.x * (draft_four_lane_threads / warp_width) +
                              threadIdx.x / warp_width);
    const auto* weight = weights + static_cast<std::size_t>(token) * width;
    int accumulators[4][rows]{};
    for (int column = lane * 16; column < width; column += warp_width * 16) {
        uint4 input[4];
        #pragma unroll
        for (int index = 0; index < 4; ++index) {
            input[index] = *reinterpret_cast<const uint4*>(projection.rows[index].values + column);
        }
        #pragma unroll
        for (int row = 0; row < rows; ++row) {
            const auto item = load_streaming_128(weight + row * width + column);
            #pragma unroll
            for (int index = 0; index < 4; ++index) {
                accumulators[index][row] = dot_signed_bytes(item, input[index],
                                                            accumulators[index][row]);
            }
        }
    }
    #pragma unroll
    for (int row = 0; row < rows; ++row) {
        #pragma unroll
        for (int index = 0; index < 4; ++index) {
            const int sum = __reduce_add_sync(warp_mask, accumulators[index][row]);
            if (lane == 0) {
                const auto& r = projection.rows[index];
                r.output[token + row] = draft_head_logit(sum, maxima[token + row], *r.maximum);
            }
        }
    }
}

template <int Count>
cudaError_t launch_draft_head_batch(const std::uint8_t* weights, const float* weight_maxima,
                                    const decoder::DraftInt8Projection* rows, cudaStream_t stream,
                                    int vocabulary_rows) {
    DraftHeadArguments<Count> batch;
    std::copy_n(rows, Count, batch.rows);
    void* quantize[] = {&batch};
    const auto status = cudaLaunchKernel(quantize_draft_head_batch<Count>, 1, Count * warp_width,
                                         quantize, 0, stream);
    if (status != cudaSuccess) {
        return status;
    }
    void* score[] = {&weights, &weight_maxima, &batch};
    if constexpr (Count == 4) {
        return cudaLaunchKernel(draft_head_int8_four_lane_kernel,
                                vocabulary_rows / draft_four_lane_rows_per_block,
                                draft_four_lane_threads, score, 0, stream);
    } else {
        return cudaLaunchKernel(draft_head_int8_batch_kernel<Count>,
                                vocabulary_rows / (2 * block_warps), block_threads, score, 0,
                                stream);
    }
}

template <int Channels>
__global__ void bf16_gdn_gates_kernel(const __nv_bfloat16* weight_b,
                                     const __nv_bfloat16* weight_a,
                                     const __nv_bfloat16* input,
                                     __nv_bfloat16* output_b, __nv_bfloat16* output_a) {
    const bool gate_b = blockIdx.x < Channels;
    const int row = blockIdx.x % Channels;
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

template <int Channels>
__global__ void bf16_gdn_decode_shared(const __nv_bfloat16* weight_b,
                                       const __nv_bfloat16* weight_a,
                                       const __nv_bfloat16* input, int rows,
                                       __nv_bfloat16* output_b, __nv_bfloat16* output_a) {
    constexpr int width = decoder::hidden_size;
    constexpr int packs = width / 8;
    __shared__ uint4 row_weight[packs];
    const bool gate_b = blockIdx.x < Channels;
    const int channel = blockIdx.x % Channels;
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
            output[row * Channels + channel] = __float2bfloat16_rn(result);
        }
    }
}

constexpr int wide_gdn_gate_rows = decoder::verification_packet_rows;
constexpr int max_gdn_gate_rows = decoder::max_wide_mtp_rows;

template <int capacity>
struct VerifyGdnGateArguments {
    decoder::VerifyGdnGateRow rows[capacity];
    int count;
};

template <int capacity, int Channels>
__global__ void bf16_gdn_verify_gates(
    const __nv_bfloat16* weight_b, const __nv_bfloat16* weight_a,
    const __grid_constant__ VerifyGdnGateArguments<capacity> args) {
    constexpr int packs = decoder::hidden_size / 8;
    __shared__ uint4 row_weight[packs];
    const bool gate_b = blockIdx.x < Channels;
    const int channel = blockIdx.x % Channels;
    const auto* weight = (gate_b ? weight_b : weight_a) + channel * decoder::hidden_size;
    for (int pack = threadIdx.x; pack < packs; pack += blockDim.x) {
        row_weight[pack] = load_128(weight + pack * 8);
    }
    __syncthreads();
    const int lane = threadIdx.x % warp_width;
    const auto row_index = (blockIdx.y * (blockDim.x / warp_width)) + (threadIdx.x / warp_width);
    if constexpr (capacity > wide_gdn_gate_rows) {
        if (row_index >= static_cast<unsigned int>(args.count)) {
            return;
        }
    }
    const auto row = args.rows[row_index];
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

namespace {

constexpr int local_gdn_gate_channels = text_gdn_value_heads / 2;

template <int Channels>
cudaError_t decode_bf16_gdn_gates_impl(const __nv_bfloat16* weight_b,
                                       const __nv_bfloat16* weight_a,
                                       const __nv_bfloat16* input, int rows, int width,
                                       __nv_bfloat16* output_b, __nv_bfloat16* output_a,
                                       cudaStream_t stream) {
    if (!weight_b || !weight_a || !input || !output_b || !output_a || rows <= 0 || width <= 0) {
        return cudaErrorInvalidValue;
    }
    if (rows != Channels || width != decoder::hidden_size ||
        !aligned(weight_b, 16) || !aligned(weight_a, 16) || !aligned(input, 16)) {
        const auto status = text_bf16_gemv(weight_b, input, rows, width, output_b, stream);
        if (status != cudaSuccess) {
            return status;
        }
        return text_bf16_gemv(weight_a, input, rows, width, output_a, stream);
    }
    void* args[] = {&weight_b, &weight_a, &input, &output_b, &output_a};
    return cudaLaunchKernel(bf16_gdn_gates_kernel<Channels>, 2 * Channels, warp_width,
                            args, 0, stream);
}

template <int Channels>
std::optional<cudaError_t> try_prefill_bf16_gdn_gates_impl(
    const __nv_bfloat16* weight_b, const __nv_bfloat16* weight_a, const __nv_bfloat16* input,
    int rows, __nv_bfloat16* output_b, __nv_bfloat16* output_a, cudaStream_t stream) {
    if (rows < 16 || rows > 65535 ||
        !weight_b || !weight_a || !input || !output_b || !output_a ||
        !aligned(weight_b, 16) || !aligned(weight_a, 16) || !aligned(input, 16)) {
        return std::nullopt;
    }
    void* args[] = {&weight_b, &weight_a, &input, &rows, &output_b, &output_a};
    static_assert(Channels % prefill_gate_warps == 0);
    const dim3 grid(Channels / prefill_gate_warps,
                    divide_up(rows, prefill_gate_tile_rows));
    const bool full_tiles = rows % prefill_gate_tile_rows == 0;
    const auto kernel = full_tiles ? bf16_gdn_prefill_gates<Channels, true>
                                   : bf16_gdn_prefill_gates<Channels, false>;
    return cudaLaunchKernel(kernel, grid, prefill_gate_warps * warp_width, args, 0, stream);
}

template <int Channels>
std::optional<cudaError_t> try_verify_bf16_gdn_gates_impl(
    const __nv_bfloat16* weight_b, const __nv_bfloat16* weight_a,
    const decoder::VerifyGdnGateRow* rows, int count, cudaStream_t stream) {
    if (count < 2 || count > max_gdn_gate_rows || !rows || !weight_b || !weight_a ||
        !aligned(weight_b, 16) || !aligned(weight_a, 16)) {
        return std::nullopt;
    }
    for (int index = 0; index < count; ++index) {
        const auto& row = rows[index];
        if (!row.input || !row.output_b || !row.output_a || !aligned(row.input, 16)) {
            return std::nullopt;
        }
    }
    const auto launch = [&](auto arguments) {
        arguments.count = count;
        for (int index = 0; index < count; ++index) {
            arguments.rows[index] = rows[index];
        }
        void* args[] = {&weight_b, &weight_a, &arguments};
        constexpr int capacity = sizeof(arguments.rows) / sizeof(*arguments.rows);
        const int packet = count <= wide_gdn_gate_rows ? count : wide_gdn_gate_rows;
        const dim3 grid(2 * Channels, divide_up(count, packet));
        return cudaLaunchKernel(bf16_gdn_verify_gates<capacity, Channels>, grid,
                                packet * warp_width, args, 0, stream);
    };
    if (count > wide_gdn_gate_rows) {
        return launch(VerifyGdnGateArguments<max_gdn_gate_rows>{});
    }
    return count > text_fp8_max_batch ? launch(VerifyGdnGateArguments<wide_gdn_gate_rows>{})
                                      : launch(VerifyGdnGateArguments<text_fp8_max_batch>{});
}

}

std::optional<cudaError_t> try_text_bf16_decode_gates(
    const __nv_bfloat16* weight_b, const __nv_bfloat16* weight_a,
    const __nv_bfloat16* input, int rows, __nv_bfloat16* output_b,
    __nv_bfloat16* output_a, int channels, cudaStream_t stream) {
    if (rows <= 0 || rows > 65535 || !weight_b || !input || !output_b ||
        (weight_a && !output_a) ||
        (channels != text_gdn_value_heads && channels != local_gdn_gate_channels) ||
        !aligned(weight_b, 16) || !aligned(weight_a, 16) || !aligned(input, 16)) {
        return std::nullopt;
    }
    const int outputs = (weight_a ? 2 : 1) * channels;
    if (rows == 1) {
        void* args[] = {&weight_b, &weight_a, &input, &output_b, &output_a};
        const auto single = [&](auto kernel) {
            return cudaLaunchKernel(kernel, outputs, warp_width, args, 0, stream);
        };
        if (channels == text_gdn_value_heads) {
            return single(bf16_gdn_gates_kernel<text_gdn_value_heads>);
        }
        if (channels == local_gdn_gate_channels) {
            return single(bf16_gdn_gates_kernel<local_gdn_gate_channels>);
        }
        return std::nullopt;
    }
    void* args[] = {&weight_b, &weight_a, &input, &rows, &output_b, &output_a};
    const int row_warps = rows <= 8 ? 8 : (rows <= 16 ? 16 : 32);
    const auto shared = [&](auto kernel) {
        return cudaLaunchKernel(kernel, (weight_a ? 2 : 1) * channels, row_warps * warp_width,
                                args, 0, stream);
    };
    if (channels == text_gdn_value_heads) {
        return shared(bf16_gdn_decode_shared<text_gdn_value_heads>);
    }
    if (channels == local_gdn_gate_channels) {
        return shared(bf16_gdn_decode_shared<local_gdn_gate_channels>);
    }
    return std::nullopt;
}

std::optional<cudaError_t> decoder::try_prefill_bf16_gdn_gates(
    const __nv_bfloat16* weight_b, const __nv_bfloat16* weight_a, const __nv_bfloat16* input,
    int rows, __nv_bfloat16* output_b, __nv_bfloat16* output_a, cudaStream_t stream) {
    return try_prefill_bf16_gdn_gates_impl<text_gdn_value_heads>(
        weight_b, weight_a, input, rows, output_b, output_a, stream);
}

std::optional<cudaError_t> decoder::try_prefill_bf16_gdn_gates_local(
    const __nv_bfloat16* weight_b, const __nv_bfloat16* weight_a, const __nv_bfloat16* input,
    int rows, __nv_bfloat16* output_b, __nv_bfloat16* output_a, cudaStream_t stream) {
    return try_prefill_bf16_gdn_gates_impl<local_gdn_gate_channels>(
        weight_b, weight_a, input, rows, output_b, output_a, stream);
}

cudaError_t decoder::decode_bf16_gdn_gates(const __nv_bfloat16* weight_b,
                                          const __nv_bfloat16* weight_a,
                                          const __nv_bfloat16* input, int rows, int width,
                                          __nv_bfloat16* output_b, __nv_bfloat16* output_a,
                                          cudaStream_t stream) {
    return decode_bf16_gdn_gates_impl<text_gdn_value_heads>(
        weight_b, weight_a, input, rows, width, output_b, output_a, stream);
}

cudaError_t decoder::decode_bf16_gdn_gates_local(const __nv_bfloat16* weight_b,
                                                const __nv_bfloat16* weight_a,
                                                const __nv_bfloat16* input, int rows, int width,
                                                __nv_bfloat16* output_b,
                                                __nv_bfloat16* output_a,
                                                cudaStream_t stream) {
    return decode_bf16_gdn_gates_impl<local_gdn_gate_channels>(
        weight_b, weight_a, input, rows, width, output_b, output_a, stream);
}

std::optional<cudaError_t> decoder::try_verify_bf16_gdn_gates(
    const __nv_bfloat16* weight_b, const __nv_bfloat16* weight_a,
    const VerifyGdnGateRow* rows, int count, cudaStream_t stream) {
    return try_verify_bf16_gdn_gates_impl<text_gdn_value_heads>(
        weight_b, weight_a, rows, count, stream);
}

std::optional<cudaError_t> decoder::try_verify_bf16_gdn_gates_local(
    const __nv_bfloat16* weight_b, const __nv_bfloat16* weight_a,
    const VerifyGdnGateRow* rows, int count, cudaStream_t stream) {
    return try_verify_bf16_gdn_gates_impl<local_gdn_gate_channels>(
        weight_b, weight_a, rows, count, stream);
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

cudaError_t decoder::quantize_draft_head_int8(const __nv_bfloat16* input, int rows,
                                             std::uint8_t* values, float* maxima,
                                             cudaStream_t stream) {
    if (!input || rows <= 0 || !values || !maxima) {
        return cudaErrorInvalidValue;
    }
    const int warps = rows == 1 ? 1 : block_warps;
    void* args[] = {&input, &rows, &values, &maxima};
    return cudaLaunchKernel(quantize_draft_head_rows, divide_up(rows, warps), warp_width * warps,
                            args, 0, stream);
}

cudaError_t decoder::decode_draft_head_int8(const std::uint8_t* weight, const float* weight_maxima,
                                            const __nv_bfloat16* input, std::uint8_t* input_values,
                                            float* input_maximum, __nv_bfloat16* output,
                                            cudaStream_t stream, int vocabulary_rows) {
    if (weight == nullptr || weight_maxima == nullptr || output == nullptr ||
        input_values == nullptr || vocabulary_rows <= 0 ||
        vocabulary_rows % (2 * block_warps) != 0 || !aligned(weight, 16) ||
        !aligned(input_values, 16)) {
        return cudaErrorInvalidValue;
    }
    const auto status = quantize_draft_head_int8(input, 1, input_values, input_maximum, stream);
    if (status != cudaSuccess) {
        return status;
    }
    void* args[] = {&weight, &weight_maxima, &input_values, &input_maximum, &output};
    return cudaLaunchKernel(draft_head_int8_kernel, vocabulary_rows / (2 * block_warps),
                            block_threads, args, 0, stream);
}

cudaError_t decoder::decode_draft_head_int8_batch(const std::uint8_t* weight,
                                                  const float* weight_maxima,
                                                  const DraftInt8Projection* rows, int count,
                                                  cudaStream_t stream, int vocabulary_rows) {
    if (weight == nullptr || weight_maxima == nullptr || rows == nullptr || count < 1 ||
        count > draft_wave_rows || !aligned(weight, 16) || vocabulary_rows <= 0 ||
        vocabulary_rows % (2 * block_warps) != 0 ||
        (count == 4 && vocabulary_rows % draft_four_lane_rows_per_block != 0)) {
        return cudaErrorInvalidValue;
    }
    for (int index = 0; index < count; ++index) {
        const auto& row = rows[index];
        if (row.input == nullptr || row.values == nullptr || row.maximum == nullptr ||
            row.output == nullptr || !aligned(row.values, 16)) {
            return cudaErrorInvalidValue;
        }
    }
    switch (count) {
    case 4:
        return launch_draft_head_batch<4>(weight, weight_maxima, rows, stream, vocabulary_rows);
    case 3:
        return launch_draft_head_batch<3>(weight, weight_maxima, rows, stream, vocabulary_rows);
    case 2:
        return launch_draft_head_batch<2>(weight, weight_maxima, rows, stream, vocabulary_rows);
    default:
        return decode_draft_head_int8(weight, weight_maxima, rows[0].input, rows[0].values,
                                      rows[0].maximum, rows[0].output, stream, vocabulary_rows);
    }
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
