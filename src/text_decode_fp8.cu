#include "sglang/decoder_cuda.cuh"

#include <cstddef>
#include <cstdint>
#include <limits>

#include <cuda_fp8.h>
#include <cuda_runtime.h>

namespace sglang::decoder {
namespace {

constexpr int warp_width = 32;
constexpr unsigned int warp_mask = 0xffffffffu;
constexpr int fp8_pack_bytes = 16;
constexpr int fp8_packed_warps = 4;
constexpr int fp8_groups_per_warp = warp_width * fp8_pack_bytes / text_fp8_group_size;
constexpr int decode_hidden_width = 5120;
constexpr int decode_attention_width = 6144;
constexpr int decode_intermediate_width = 17408;

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

__device__ uint4 load_128(const void* address) {
    uint4 bits;
    const auto global = __cvta_generic_to_global(address);
    asm("ld.global.v4.b32 {%0, %1, %2, %3}, [%4];"
        : "=r"(bits.x), "=r"(bits.y), "=r"(bits.z), "=r"(bits.w)
        : "l"(global) : "memory");
    return bits;
}

__device__ uint4 load_fp8_weight(const void* address) {
    uint4 bits;
    const auto global = __cvta_generic_to_global(address);
    asm("ld.global.cs.v4.b32 {%0, %1, %2, %3}, [%4];"
        : "=r"(bits.x), "=r"(bits.y), "=r"(bits.z), "=r"(bits.w)
        : "l"(global) : "memory");
    return bits;
}

struct DecodeActivationPack {
    __half2_raw pairs[8];
};

__device__ DecodeActivationPack decode_activation_pack(uint4 bytes) {
    const unsigned words[]{bytes.x, bytes.y, bytes.z, bytes.w};
    DecodeActivationPack pack;
#pragma unroll
    for (int word = 0; word < 4; ++word) {
        pack.pairs[2 * word] = __nv_cvt_fp8x2_to_halfraw2(
            static_cast<unsigned short>(words[word]), __NV_E4M3);
        pack.pairs[2 * word + 1] = __nv_cvt_fp8x2_to_halfraw2(
            static_cast<unsigned short>(words[word] >> 16), __NV_E4M3);
    }
    return pack;
}

__device__ float decode_fp8_pair_dot(unsigned short weight, const __half2_raw& input, float sum) {
    const auto decoded = __nv_cvt_fp8x2_to_halfraw2(weight, __NV_E4M3);
    asm("fma.rn.f32.f16 %0, %1, %2, %0;" : "+f"(sum) : "h"(decoded.x), "h"(input.x));
    asm("fma.rn.f32.f16 %0, %1, %2, %0;" : "+f"(sum) : "h"(decoded.y), "h"(input.y));
    return sum;
}

__device__ float decode_fp8_pack_dot(uint4 bytes, DecodeActivationPack input) {
    const unsigned words[]{bytes.x, bytes.y, bytes.z, bytes.w};
    float low = 0.0f;
    float high = 0.0f;
#pragma unroll
    for (int word = 0; word < 2; ++word) {
        low = decode_fp8_pair_dot(static_cast<unsigned short>(words[word]),
                                  input.pairs[2 * word], low);
        low = decode_fp8_pair_dot(static_cast<unsigned short>(words[word] >> 16),
                                  input.pairs[2 * word + 1], low);
        high = decode_fp8_pair_dot(static_cast<unsigned short>(words[word + 2]),
                                   input.pairs[2 * word + 4], high);
        high = decode_fp8_pair_dot(static_cast<unsigned short>(words[word + 2] >> 16),
                                   input.pairs[2 * word + 5], high);
    }
    return low + high;
}

struct DecodeFp8Projections {
    DecodeFp8Projection query;
    DecodeFp8Projection key;
    DecodeFp8Projection value;
};

template <int Width, bool Qkv>
__global__ void fp8_decode_projection_kernel(TextFp8ActivationView input,
                                              DecodeFp8Projections projections) {
    const int lane = threadIdx.x % warp_width;
    int row = blockIdx.x * fp8_packed_warps + threadIdx.x / warp_width;
    auto matrix = projections.query;
    if constexpr (Qkv) {
        if (row >= matrix.rows) {
            row -= matrix.rows;
            matrix = projections.key;
            if (row >= matrix.rows) {
                row -= matrix.rows;
                matrix = projections.value;
            }
        }
    }
    if (row >= matrix.rows) {
        return;
    }
    constexpr int groups = Width / text_fp8_group_size;
    constexpr int lanes_per_group = warp_width / fp8_groups_per_warp;
    const int lane_group = lane / lanes_per_group;
    const int lane_column = lane * fp8_pack_bytes;
    const auto* values = matrix.values + static_cast<std::size_t>(row) * Width + lane_column;
    const auto* scales =
        matrix.scales + static_cast<std::size_t>(row / text_fp8_group_size) * groups + lane_group;
    float sum = 0.0f;
#pragma unroll 2
    for (int group = 0; group < groups; group += fp8_groups_per_warp) {
        const int column = group * text_fp8_group_size;
        const auto rhs = decode_activation_pack(load_128(input.values + column + lane_column));
        const auto lhs = load_fp8_weight(values + column);
        const float scaled = decode_fp8_pack_dot(lhs, rhs) * input.scales[group + lane_group];
        sum = fmaf(scaled, __bfloat162float(scales[group]), sum);
    }
    sum = warp_sum(sum);
    if (lane == 0) {
        matrix.output[row] = __float2bfloat16_rn(sum);
    }
}

__device__ __nv_bfloat16 swiglu_value(__nv_bfloat16 gate, __nv_bfloat16 up) {
    const float value = __bfloat162float(gate);
    const auto activated = __float2bfloat16_rn(value / (1.0f + expf(-value)));
    return __float2bfloat16_rn(__bfloat162float(activated) * __bfloat162float(up));
}

__global__ void fp8_decode_gate_up_kernel(TextFp8ActivationView input, DecodeFp8Projection gate,
                                          DecodeFp8Projection up) {
    const int lane = threadIdx.x % warp_width;
    const int row = blockIdx.x * fp8_packed_warps + threadIdx.x / warp_width;
    if (row >= gate.rows) {
        return;
    }
    constexpr int groups = decode_hidden_width / text_fp8_group_size;
    constexpr int lanes_per_group = warp_width / fp8_groups_per_warp;
    const int lane_group = lane / lanes_per_group;
    const int lane_column = lane * fp8_pack_bytes;
    const auto row_column = static_cast<std::size_t>(row) * decode_hidden_width + lane_column;
    const int scale_offset = (row / text_fp8_group_size) * groups + lane_group;
    float gate_sum = 0.0f;
    float up_sum = 0.0f;
#pragma unroll 1
    for (int group = 0; group < groups; group += fp8_groups_per_warp) {
        const int column = group * text_fp8_group_size;
        const auto rhs = decode_activation_pack(load_128(input.values + column + lane_column));
        const float activation_scale = input.scales[group + lane_group];
        const auto gate_pack = load_fp8_weight(gate.values + row_column + column);
        gate_sum = fmaf(decode_fp8_pack_dot(gate_pack, rhs) * activation_scale,
                        __bfloat162float(gate.scales[scale_offset + group]), gate_sum);
        const auto up_pack = load_fp8_weight(up.values + row_column + column);
        up_sum = fmaf(decode_fp8_pack_dot(up_pack, rhs) * activation_scale,
                      __bfloat162float(up.scales[scale_offset + group]), up_sum);
    }
    gate_sum = warp_sum(gate_sum);
    up_sum = warp_sum(up_sum);
    if (lane == 0) {
        gate.output[row] = swiglu_value(__float2bfloat16_rn(gate_sum),
                                       __float2bfloat16_rn(up_sum));
    }
}

template <int Width, bool Qkv>
cudaError_t launch_decode_projection(TextFp8ActivationView input, DecodeFp8Projections projections,
                                     int rows, cudaStream_t stream) {
    void* args[] = {&input, &projections};
    return cudaLaunchKernel(fp8_decode_projection_kernel<Width, Qkv>,
                            divide_up(rows, fp8_packed_warps),
                            fp8_packed_warps * warp_width, args, 0, stream);
}

struct ByteSpan {
    std::uintptr_t begin;
    std::uintptr_t end;

    bool valid() const { return begin < end; }
    bool overlaps(ByteSpan other) const { return begin < other.end && other.begin < end; }
};

template <typename Element>
ByteSpan byte_span(const Element* pointer, std::size_t count) {
    const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
    return {begin, begin + count * sizeof(Element)};
}

struct Fp8ReadSpans {
    ByteSpan values;
    ByteSpan scales;

    bool valid() const { return values.valid() && scales.valid(); }
    bool overlaps(ByteSpan output) const {
        return values.overlaps(output) || scales.overlaps(output);
    }
};

template <std::size_t Count>
bool valid_buffers(TextFp8ActivationView input, int width,
                   const DecodeFp8Projection (&projections)[Count],
                   bool allow_equal_outputs = false) {
    static_assert(std::numeric_limits<std::uintptr_t>::digits >=
                  2 * std::numeric_limits<int>::digits + 1);
    if (width <= 0 || !input.values || !input.scales || !aligned(input.scales, alignof(float))) {
        return false;
    }
    const auto groups = static_cast<std::size_t>(text_fp8_group_count(width));
    const Fp8ReadSpans activation{byte_span(input.values, width), byte_span(input.scales, groups)};
    if (!activation.valid()) {
        return false;
    }
    Fp8ReadSpans weights[Count];
    ByteSpan outputs[Count];
    for (std::size_t index = 0; index < Count; ++index) {
        const auto& projection = projections[index];
        if (projection.rows <= 0 || !projection.values || !projection.scales ||
            !projection.output || !aligned(projection.scales, alignof(__nv_bfloat16)) ||
            !aligned(projection.output, alignof(__nv_bfloat16))) {
            return false;
        }
        const auto rows = static_cast<std::size_t>(projection.rows);
        weights[index] = {byte_span(projection.values, rows * width),
                          byte_span(projection.scales, text_fp8_group_count(projection.rows) *
                                                            groups)};
        outputs[index] = byte_span(projection.output, rows);
        if (!weights[index].valid() || !outputs[index].valid() ||
            activation.overlaps(outputs[index])) {
            return false;
        }
    }
    for (std::size_t index = 0; index < Count; ++index) {
        for (const auto& weight : weights) {
            if (weight.overlaps(outputs[index])) {
                return false;
            }
        }
        for (std::size_t earlier = 0; earlier < index; ++earlier) {
            const bool equal = outputs[earlier].begin == outputs[index].begin &&
                               outputs[earlier].end == outputs[index].end;
            if (outputs[earlier].overlaps(outputs[index]) && !(allow_equal_outputs && equal)) {
                return false;
            }
        }
    }
    return true;
}

cudaError_t unfused_projection(TextFp8ActivationView input, DecodeFp8Projection projection,
                               int width, cudaStream_t stream) {
    return text_fp8_gemv(projection.values, projection.scales, input, projection.rows, width,
                         projection.output, stream);
}

}

cudaError_t decode_fp8_project(TextFp8ActivationView input, DecodeFp8Projection projection,
                               int width, cudaStream_t stream) {
    const DecodeFp8Projection projections[]{projection};
    if (!valid_buffers(input, width, projections)) {
        return cudaErrorInvalidValue;
    }
    if (aligned(input.values, fp8_pack_bytes) && aligned(projection.values, fp8_pack_bytes)) {
        switch (width) {
            case decode_hidden_width:
                return launch_decode_projection<decode_hidden_width, false>(
                    input, {projection, {}, {}}, projection.rows, stream);
            case decode_attention_width:
                return launch_decode_projection<decode_attention_width, false>(
                    input, {projection, {}, {}}, projection.rows, stream);
            case decode_intermediate_width:
                return launch_decode_projection<decode_intermediate_width, false>(
                    input, {projection, {}, {}}, projection.rows, stream);
        }
    }
    return unfused_projection(input, projection, width, stream);
}

cudaError_t decode_fp8_qkv(TextFp8ActivationView input, DecodeFp8Projection query,
                           DecodeFp8Projection key, DecodeFp8Projection value, int width,
                           cudaStream_t stream) {
    const DecodeFp8Projection projections[]{query, key, value};
    if (!valid_buffers(input, width, projections)) {
        return cudaErrorInvalidValue;
    }
    if (width == decode_hidden_width && query.rows == text_gqa_q_gate_width &&
        key.rows == text_gqa_kv_width && value.rows == text_gqa_kv_width &&
        aligned(input.values, fp8_pack_bytes) && aligned(query.values, fp8_pack_bytes) &&
        aligned(key.values, fp8_pack_bytes) && aligned(value.values, fp8_pack_bytes)) {
        return launch_decode_projection<decode_hidden_width, true>(
            input, {query, key, value}, query.rows + key.rows + value.rows, stream);
    }
    for (const auto& projection : projections) {
        const auto status = unfused_projection(input, projection, width, stream);
        if (status != cudaSuccess) {
            return status;
        }
    }
    return cudaSuccess;
}

cudaError_t decode_fp8_gate_up(TextFp8ActivationView input, DecodeFp8Projection gate,
                               DecodeFp8Projection up, int width, cudaStream_t stream) {
    const bool fused = width == decode_hidden_width && gate.rows == decode_intermediate_width &&
        aligned(input.values, fp8_pack_bytes) && aligned(gate.values, fp8_pack_bytes) &&
        aligned(up.values, fp8_pack_bytes);
    const DecodeFp8Projection projections[]{gate, up};
    if (gate.rows != up.rows || !valid_buffers(input, width, projections, fused)) {
        return cudaErrorInvalidValue;
    }
    if (fused) {
        void* args[] = {&input, &gate, &up};
        return cudaLaunchKernel(fp8_decode_gate_up_kernel, divide_up(gate.rows, fp8_packed_warps),
                                fp8_packed_warps * warp_width, args, 0, stream);
    }
    auto status = unfused_projection(input, gate, width, stream);
    if (status == cudaSuccess) {
        status = unfused_projection(input, up, width, stream);
    }
    if (status != cudaSuccess) {
        return status;
    }
    return text_swiglu(gate.output, up.output, gate.rows, gate.output, stream);
}

}
