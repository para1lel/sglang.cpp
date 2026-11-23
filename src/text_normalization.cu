#include "sglang/text_ops.h"

#include "sglang/decoder_cuda.cuh"

#include "sglang/text/text_cuda_primitives.cuh"
#include "sglang/text/quantize_warp.cuh"

#include <cuda_fp8.h>

namespace sglang {
namespace {

using namespace text_kernel;

constexpr float rms_epsilon = 1.0e-6f;

__device__ float residual_sum(const __nv_bfloat16* branch, const __nv_bfloat16* residual,
                              std::size_t col) {
    const float value = __bfloat162float(branch[col]);
    return residual == nullptr ? value : value + __bfloat162float(residual[col]);
}

__device__ __forceinline__ void rms_norm_row(const __nv_bfloat16* branch,
                                             const __nv_bfloat16* residual,
                                             const __nv_bfloat16* weight, int width,
                                             __nv_bfloat16* output,
                                             __nv_bfloat16* residual_output) {
    __shared__ float warp_squares[block_warps];
    __shared__ float inverse_rms;
    const int lane = threadIdx.x % warp_width;
    const int warp = threadIdx.x / warp_width;
    const auto count = static_cast<std::size_t>(width);
    float squares = 0.0f;
    for (std::size_t col = threadIdx.x; col < count; col += block_threads) {
        const float value = residual_sum(branch, residual, col);
        squares = fmaf(value, value, squares);
    }
    squares = warp_sum(squares);
    if (lane == 0) {
        warp_squares[warp] = squares;
    }
    __syncthreads();
    if (warp == 0) {
        squares = warp_sum(lane < block_warps ? warp_squares[lane] : 0.0f);
        if (lane == 0) {
            inverse_rms = rsqrtf(squares / static_cast<float>(width) + rms_epsilon);
        }
    }
    __syncthreads();
    for (std::size_t col = threadIdx.x; col < count; col += block_threads) {
        const float value = residual_sum(branch, residual, col);
        const float gain = 1.0f + __bfloat162float(weight[col]);
        if (residual_output != nullptr) {
            residual_output[col] = __float2bfloat16_rn(value);
        }
        output[col] = __float2bfloat16_rn((value * inverse_rms) * gain);
    }
}

template <bool Batched>
__global__ void rms_norm_kernel(const __nv_bfloat16* branch, const __nv_bfloat16* residual,
                                const __nv_bfloat16* weight, int width, __nv_bfloat16* output,
                                __nv_bfloat16* residual_output) {
    const auto offset = Batched ? static_cast<std::size_t>(blockIdx.x) * width : 0;
    rms_norm_row(branch + offset, residual ? residual + offset : nullptr, weight, width,
                 output + offset, residual_output ? residual_output + offset : nullptr);
}

__global__ void rms_norm_pair_rows_kernel(const __nv_bfloat16* first, const __nv_bfloat16* second,
                                          const __nv_bfloat16* first_weight,
                                          const __nv_bfloat16* second_weight, int width,
                                          __nv_bfloat16* output) {
    const auto offset = static_cast<std::size_t>(blockIdx.x) * width;
    const bool second_half = blockIdx.y != 0;
    rms_norm_row((second_half ? second : first) + offset, nullptr,
                 second_half ? second_weight : first_weight, width,
                 output + 2 * offset + blockIdx.y * static_cast<std::size_t>(width), nullptr);
}

constexpr int rms_hidden_width = 5120;

__device__ void stage_rms_vector(const __nv_bfloat16* input, __nv_bfloat16* shared) {
    const uint4 vector = load_128(input);
    const auto address = static_cast<unsigned int>(__cvta_generic_to_shared(shared));
    asm("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
        :: "r"(address), "r"(vector.x), "r"(vector.y), "r"(vector.z), "r"(vector.w)
        : "memory");
}

__device__ void stage_rms_vector_async(const __nv_bfloat16* input, __nv_bfloat16* shared) {
    const auto address = static_cast<unsigned int>(__cvta_generic_to_shared(shared));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;"
                 :: "r"(address), "l"(input) : "memory");
}

__device__ __forceinline__ float rms_inverse(float squares, int width) {
    __shared__ float warp_squares[block_warps];
    __shared__ float inverse_rms;
    const int lane = threadIdx.x % warp_width;
    const int warp = threadIdx.x / warp_width;
    squares = warp_sum(squares);
    if (lane == 0) {
        warp_squares[warp] = squares;
    }
    __syncthreads();
    if (warp == 0) {
        squares = warp_sum(lane < block_warps ? warp_squares[lane] : 0.0f);
        if (lane == 0) {
            inverse_rms = rsqrtf(squares / static_cast<float>(width) + rms_epsilon);
        }
    }
    __syncthreads();
    return inverse_rms;
}

__device__ void store_rms(float value, float inverse_rms, const __nv_bfloat16* weight,
                          std::size_t col, __nv_bfloat16* output,
                          __nv_bfloat16* residual_output) {
    const float gain = 1.0f + __bfloat162float(weight[col]);
    if (residual_output != nullptr) {
        residual_output[col] = __float2bfloat16_rn(value);
    }
    output[col] = __float2bfloat16_rn((value * inverse_rms) * gain);
}

template <bool HasResidual, bool StoresResidual>
__global__ void rms_norm_vector_kernel(const __nv_bfloat16* branch,
                                       const __nv_bfloat16* residual,
                                       const __nv_bfloat16* weight, __nv_bfloat16* output,
                                       __nv_bfloat16* residual_output) {
    __shared__ alignas(16) __nv_bfloat16 branch_values[rms_hidden_width];
    __shared__ alignas(16) __nv_bfloat16 residual_values[HasResidual ? rms_hidden_width : 1];
    __shared__ alignas(16) __nv_bfloat16 gains[rms_hidden_width];
    constexpr int vector_width = sizeof(uint4) / sizeof(__nv_bfloat16);
    static_assert(rms_hidden_width % block_threads == 0 && rms_hidden_width % vector_width == 0);
    for (int col = threadIdx.x * vector_width; col < rms_hidden_width;
         col += block_threads * vector_width) {
        stage_rms_vector(branch + col, branch_values + col);
        if constexpr (HasResidual) {
            stage_rms_vector(residual + col, residual_values + col);
        }
        stage_rms_vector(weight + col, gains + col);
    }
    __syncthreads();

    constexpr int values_per_thread = rms_hidden_width / block_threads;
    float values[values_per_thread];
    float squares = 0.0f;
#pragma unroll
    for (int index = 0; index < values_per_thread; ++index) {
        const int col = threadIdx.x + index * block_threads;
        const float value = residual_sum(branch_values, HasResidual ? residual_values : nullptr,
                                         col);
        values[index] = value;
        squares = fmaf(value, value, squares);
    }
    const float inverse_rms = rms_inverse(squares, rms_hidden_width);
#pragma unroll
    for (int index = 0; index < values_per_thread; ++index) {
        const int col = threadIdx.x + index * block_threads;
        store_rms(values[index], inverse_rms, gains, col, output,
                  StoresResidual ? residual_output : nullptr);
    }
}

__device__ float4 rms_quant_values(uint2 bits) {
    return {__uint_as_float(bits.x << 16), __uint_as_float(bits.x & 0xffff0000u),
            __uint_as_float(bits.y << 16), __uint_as_float(bits.y & 0xffff0000u)};
}

__device__ float rms_quant_maximum(float4 values) {
    return fmaxf(fmaxf(fabsf(values.x), fabsf(values.y)),
                 fmaxf(fabsf(values.z), fabsf(values.w)));
}

__device__ __nv_fp8x4_storage_t rms_quant_pack(float4 values, float reciprocal) {
    const float4 scaled{values.x * reciprocal, values.y * reciprocal,
                        values.z * reciprocal, values.w * reciprocal};
    return __nv_fp8x4_e4m3(scaled).__x;
}

__device__ void rms_quantize_subwarps(const __nv_bfloat16* normalized,
                                      TextFp8Activation activation) {
    constexpr int lanes = 8;
    constexpr int half_group = text_fp8_group_size / 2;
    constexpr int groups = rms_hidden_width / text_fp8_group_size;
    constexpr int values_per_lane = half_group / lanes;
    constexpr int groups_per_block = block_threads / lanes;
    static_assert(values_per_lane == 8);
    static_assert(warp_width % lanes == 0 && groups % (warp_width / lanes) == 0);
    const int lane = threadIdx.x % lanes;
    for (int group = threadIdx.x / lanes; group < groups; group += groups_per_block) {
        const int col = group * text_fp8_group_size + lane * values_per_lane;
        const auto lower = *reinterpret_cast<const uint4*>(normalized + col);
        const auto upper = *reinterpret_cast<const uint4*>(normalized + col + half_group);
        const float4 first = rms_quant_values({lower.x, lower.y});
        const float4 second = rms_quant_values({lower.z, lower.w});
        const float4 third = rms_quant_values({upper.x, upper.y});
        const float4 fourth = rms_quant_values({upper.z, upper.w});
        float maximum = fmaxf(fmaxf(rms_quant_maximum(first), rms_quant_maximum(second)),
                              fmaxf(rms_quant_maximum(third), rms_quant_maximum(fourth)));
        for (int delta = lanes / 2; delta > 0; delta /= 2) {
            maximum = fmaxf(maximum, __shfl_down_sync(warp_mask, maximum, delta, lanes));
        }
        float reciprocal = 0.0f;
        if (lane == 0) {
            const float scale = __fdiv_rn(fmaxf(quant_epsilon, maximum), fp8_max);
            activation.scales[group] = scale;
            reciprocal = __fdiv_rn(1.0f, scale);
        }
        reciprocal = __shfl_sync(warp_mask, reciprocal, 0, lanes);
        auto* packed = reinterpret_cast<uint2*>(activation.values + col);
        *packed = {rms_quant_pack(first, reciprocal), rms_quant_pack(second, reciprocal)};
        packed[half_group / sizeof(uint2)] =
            {rms_quant_pack(third, reciprocal), rms_quant_pack(fourth, reciprocal)};
    }
}

template <bool HasResidual, bool PackedGroups = false, bool LaunchDependents = false,
          bool AsyncStage = PackedGroups>
__device__ __forceinline__ void rms_quantize(const __nv_bfloat16* branch, __nv_bfloat16* residual,
                                     const __nv_bfloat16* weight, __nv_bfloat16* normalized,
                                     TextFp8Activation activation) {
    constexpr int values_per_thread = rms_hidden_width / block_threads;
    constexpr int groups = rms_hidden_width / text_fp8_group_size;
    constexpr int values_per_lane = text_fp8_group_size / warp_width;
    constexpr int vector_width = sizeof(uint4) / sizeof(__nv_bfloat16);
    __shared__ alignas(16) __nv_bfloat16 branch_values[rms_hidden_width];
    __shared__ alignas(16) __nv_bfloat16 residual_values[HasResidual ? rms_hidden_width : 1];
    __shared__ alignas(16) __nv_bfloat16 gains[rms_hidden_width];
    for (int col = threadIdx.x * vector_width; col < rms_hidden_width;
         col += block_threads * vector_width) {
        if constexpr (AsyncStage) {
            stage_rms_vector_async(branch + col, branch_values + col);
            if constexpr (HasResidual) {
                stage_rms_vector_async(residual + col, residual_values + col);
            }
            stage_rms_vector_async(weight + col, gains + col);
        } else {
            stage_rms_vector(branch + col, branch_values + col);
            if constexpr (HasResidual) {
                stage_rms_vector(residual + col, residual_values + col);
            }
            stage_rms_vector(weight + col, gains + col);
        }
    }
    if constexpr (AsyncStage) {
        asm volatile("cp.async.commit_group;" ::: "memory");
        asm volatile("cp.async.wait_group 0;" ::: "memory");
    }
    __syncthreads();
    if constexpr (LaunchDependents) {
        asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    }
    float values[values_per_thread];
    float squares = 0.0f;
#pragma unroll
    for (int index = 0; index < values_per_thread; ++index) {
        const int col = threadIdx.x + index * block_threads;
        const auto value = residual_sum(branch_values, HasResidual ? residual_values : nullptr,
                                        col);
        values[index] = value;
        squares = fmaf(value, value, squares);
    }
    const float inverse_rms = rms_inverse(squares, rms_hidden_width);
#pragma unroll
    for (int index = 0; index < values_per_thread; ++index) {
        const int col = threadIdx.x + index * block_threads;
        const float value = values[index];
        const float gain = 1.0f + __bfloat162float(gains[col]);
        const auto element = __float2bfloat16_rn((value * inverse_rms) * gain);
        normalized[col] = element;
        branch_values[col] = element;
        if constexpr (HasResidual) {
            residual[col] = __float2bfloat16_rn(value);
        }
    }
    __syncthreads();
    if constexpr (PackedGroups) {
        rms_quantize_subwarps(branch_values, activation);
        return;
    }
    const int lane = threadIdx.x % warp_width;
    const int warp = threadIdx.x / warp_width;
    for (int group = warp; group < groups; group += block_warps) {
        const int col = group * text_fp8_group_size + lane;
        float maximum = quant_epsilon;
        float elements[values_per_lane];
#pragma unroll
        for (int part = 0; part < values_per_lane; ++part) {
            elements[part] = __bfloat162float(branch_values[col + part * warp_width]);
            maximum = fmaxf(maximum, fabsf(elements[part]));
        }
        const float amax = __shfl_sync(warp_mask, warp_max(maximum), 0);
        const float scale = __fdiv_rn(amax, fp8_max);
        const float reciprocal = __fdiv_rn(1.0f, scale);
        if (lane == 0) {
            activation.scales[group] = scale;
        }
#pragma unroll
        for (int part = 0; part < values_per_lane; ++part) {
            activation.values[col + part * warp_width] =
                __nv_cvt_float_to_fp8(elements[part] * reciprocal, __NV_SATFINITE, __NV_E4M3);
        }
    }
}

template <bool HasResidual, bool PackedGroups = false, bool LaunchDependents = false,
          bool AsyncStage = PackedGroups>
__global__ void prefill_rms_quantize(const __nv_bfloat16* branch, __nv_bfloat16* residual,
                                     const __nv_bfloat16* weight, __nv_bfloat16* normalized,
                                     TextFp8Activation activation) {
    const auto offset = static_cast<std::size_t>(blockIdx.x) * rms_hidden_width;
    activation.values += offset;
    activation.scales += offset / text_fp8_group_size;
    rms_quantize<HasResidual, PackedGroups, LaunchDependents, AsyncStage>(
        branch + offset, HasResidual ? residual + offset : nullptr, weight,
        normalized + offset, activation);
}

template <bool HasResidual, typename Rows>
__global__ void verify_rms_kernel(const __grid_constant__ Rows rows,
                                  const __nv_bfloat16* weight) {
    const auto& row = rows.entries[blockIdx.x];
    rms_quantize<HasResidual, true>(row.branch, row.residual, weight, row.normalized, row.output);
}

template <bool ApplySwiglu, typename Rows>
__global__ void verify_quantize_kernel(const __grid_constant__ Rows rows, int width) {
    const auto& row = rows.entries[blockIdx.y];
    quantize_warp<ApplySwiglu>(row.input, row.up, width, row.output);
}

template <typename Rows>
cudaError_t launch_verify_quantize(Rows rows, int count, int width, cudaStream_t stream) {
    const int groups = width / text_fp8_group_size;
    const dim3 grid((groups + block_warps - 1) / block_warps, count);
    void* args[] = {&rows, &width};
    const auto kernel = rows.entries[0].up ? verify_quantize_kernel<true, Rows>
                                           : verify_quantize_kernel<false, Rows>;
    return cudaLaunchKernel(kernel, grid, block_threads, args, 0, stream);
}
}

cudaError_t decoder::verify_quantize_fp8(VerifyQuantizeRows rows, int width, cudaStream_t stream) {
    return launch_verify_quantize(rows, fused_verify_rows, width, stream);
}

cudaError_t decoder::verify_quantize_fp8_wide(WideVerifyQuantizeRows rows, int count, int width,
                                             cudaStream_t stream) {
    if (count <= fused_verify_rows || count > WideVerifyQuantizeRows::capacity ||
        width <= 0 || width % text_fp8_group_size != 0) {
        return cudaErrorInvalidValue;
    }
    return launch_verify_quantize(rows, count, width, stream);
}

cudaError_t decoder::verify_rms_quantize(VerifyRmsQuantizeRows rows, const __nv_bfloat16* weight,
                                        cudaStream_t stream) {
    void* args[] = {&rows, &weight};
    const auto kernel = rows.entries[0].residual ? verify_rms_kernel<true, VerifyRmsQuantizeRows>
                                                 : verify_rms_kernel<false, VerifyRmsQuantizeRows>;
    return cudaLaunchKernel(kernel, fused_verify_rows, block_threads, args, 0, stream);
}

cudaError_t decoder::verify_rms_quantize_wide(WideVerifyRmsQuantizeRows rows, int count,
                                             const __nv_bfloat16* weight, cudaStream_t stream) {
    if (count <= fused_verify_rows || count > WideVerifyRmsQuantizeRows::capacity) {
        return cudaErrorInvalidValue;
    }
    void* args[] = {&rows, &weight};
    const auto kernel = rows.entries[0].residual
        ? verify_rms_kernel<true, WideVerifyRmsQuantizeRows>
        : verify_rms_kernel<false, WideVerifyRmsQuantizeRows>;
    return cudaLaunchKernel(kernel, count, block_threads, args, 0, stream);
}

cudaError_t decoder::tensor_core_prefill_rms_quantize(
    const __nv_bfloat16* branch, __nv_bfloat16* residual, const __nv_bfloat16* weight,
    int width, int rows, __nv_bfloat16* normalized, TextFp8Activation activation,
    cudaStream_t stream, RmsQuantizeLaunch launch) {
    if (width != rms_hidden_width || !aligned(branch, alignof(uint4)) ||
        !aligned(residual, alignof(uint4)) || !aligned(weight, alignof(uint4))) {
        const auto status = text_rms_norm_rows(branch, residual, weight, width, rows,
                                              normalized, residual, stream);
        return status == cudaSuccess
            ? text_quantize_fp8_rows(normalized, width, rows, activation, stream)
            : status;
    }
    void* args[] = {&branch, &residual, &weight, &normalized, &activation};
    const bool prefetch_weights = launch == RmsQuantizeLaunch::prefetch_decode_weights;
    if (launch == RmsQuantizeLaunch::full_group_prefill && rows == 4096 &&
        aligned(activation.values, alignof(uint2))) {
        const auto pending = cudaGetLastError();
        if (pending != cudaSuccess) return pending;
        const auto kernel = residual ? prefill_rms_quantize<true, true, false, false>
                                     : prefill_rms_quantize<false, true, false, false>;
        const auto status = cudaLaunchKernel(kernel, rows, block_threads, args, 0, stream);
        const auto recorded = cudaGetLastError();
        return status == cudaSuccess ? recorded : status;
    }
    if (residual && rows >= 2 && rows <= 32 && aligned(activation.values, alignof(uint2))) {
        const bool early = prefetch_weights ||
                           (launch == RmsQuantizeLaunch::packed_gate_up && rows == 32);
        const auto kernel = early
            ? prefill_rms_quantize<true, true, true> : prefill_rms_quantize<true, true>;
        return cudaLaunchKernel(kernel, rows, block_threads, args, 0, stream);
    }
    if (prefetch_weights) {
        const auto kernel = residual ? prefill_rms_quantize<true, false, true>
                                     : prefill_rms_quantize<false, false, true>;
        return cudaLaunchKernel(kernel, rows, block_threads, args, 0, stream);
    }
    const auto kernel = residual ? prefill_rms_quantize<true> : prefill_rms_quantize<false>;
    return cudaLaunchKernel(kernel, rows, block_threads, args, 0, stream);
}

cudaError_t text_rms_norm(const __nv_bfloat16* branch, const __nv_bfloat16* residual,
                         const __nv_bfloat16* weight, int width, __nv_bfloat16* output,
                         __nv_bfloat16* residual_output, cudaStream_t stream) {
    if (branch == nullptr || weight == nullptr || output == nullptr || width <= 0 ||
        output == residual_output) {
        return cudaErrorInvalidValue;
    }
    if (width == rms_hidden_width && aligned(branch, alignof(uint4)) &&
        aligned(residual, alignof(uint4)) && aligned(weight, alignof(uint4))) {
        void* args[] = {&branch, &residual, &weight, &output, &residual_output};
        const bool store_residual = residual_output != nullptr;
        const auto kernel = residual == nullptr
            ? (store_residual ? rms_norm_vector_kernel<false, true>
                              : rms_norm_vector_kernel<false, false>)
            : (store_residual ? rms_norm_vector_kernel<true, true>
                              : rms_norm_vector_kernel<true, false>);
        return cudaLaunchKernel(kernel, 1, block_threads, args, 0, stream);
    }
    void* args[] = {&branch, &residual, &weight, &width, &output, &residual_output};
    return cudaLaunchKernel(rms_norm_kernel<false>, 1, block_threads, args, 0, stream);
}

cudaError_t text_rms_norm_rows(const __nv_bfloat16* branch, const __nv_bfloat16* residual,
                              const __nv_bfloat16* weight, int width, int rows,
                              __nv_bfloat16* output, __nv_bfloat16* residual_output,
                              cudaStream_t stream) {
    if (!branch || !weight || !output || width <= 0 || rows <= 0 || output == residual_output) {
        return cudaErrorInvalidValue;
    }
    void* args[] = {&branch, &residual, &weight, &width, &output, &residual_output};
    return cudaLaunchKernel(rms_norm_kernel<true>, rows, block_threads, args, 0, stream);
}

cudaError_t text_rms_norm_pair_rows(const __nv_bfloat16* first, const __nv_bfloat16* second,
                                   const __nv_bfloat16* first_weight,
                                   const __nv_bfloat16* second_weight, int width, int rows,
                                   __nv_bfloat16* output, cudaStream_t stream) {
    if (!first || !second || !first_weight || !second_weight || !output ||
        width <= 0 || rows <= 0) {
        return cudaErrorInvalidValue;
    }
    void* args[] = {&first, &second, &first_weight, &second_weight, &width, &output};
    return cudaLaunchKernel(rms_norm_pair_rows_kernel, dim3(rows, 2), block_threads, args, 0,
                            stream);
}

}
