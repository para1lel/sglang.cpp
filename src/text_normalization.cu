#include "sglang/text_ops.h"

#include "sglang/decoder_cuda.cuh"

#include "sglang/text/text_cuda_primitives.cuh"

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

template <bool HasResidual>
__global__ void prefill_rms_quantize(const __nv_bfloat16* branch, __nv_bfloat16* residual,
                                     const __nv_bfloat16* weight, __nv_bfloat16* normalized,
                                     TextFp8Activation activation) {
    constexpr int values_per_thread = rms_hidden_width / block_threads;
    constexpr int groups = rms_hidden_width / text_fp8_group_size;
    constexpr int values_per_lane = text_fp8_group_size / warp_width;
    constexpr int vector_width = sizeof(uint4) / sizeof(__nv_bfloat16);
    __shared__ alignas(16) __nv_bfloat16 branch_values[rms_hidden_width];
    __shared__ alignas(16) __nv_bfloat16 residual_values[HasResidual ? rms_hidden_width : 1];
    __shared__ alignas(16) __nv_bfloat16 gains[rms_hidden_width];
    const auto offset = static_cast<std::size_t>(blockIdx.x) * rms_hidden_width;
    branch += offset;
    normalized += offset;
    if constexpr (HasResidual) {
        residual += offset;
    }
    for (int col = threadIdx.x * vector_width; col < rms_hidden_width;
         col += block_threads * vector_width) {
        stage_rms_vector(branch + col, branch_values + col);
        if constexpr (HasResidual) {
            stage_rms_vector(residual + col, residual_values + col);
        }
        stage_rms_vector(weight + col, gains + col);
    }
    __syncthreads();
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
            activation.scales[static_cast<std::size_t>(blockIdx.x) * groups + group] = scale;
        }
#pragma unroll
        for (int part = 0; part < values_per_lane; ++part) {
            activation.values[offset + col + part * warp_width] =
                __nv_cvt_float_to_fp8(elements[part] * reciprocal, __NV_SATFINITE, __NV_E4M3);
        }
    }
}

}

cudaError_t decoder::tensor_core_prefill_rms_quantize(
    const __nv_bfloat16* branch, __nv_bfloat16* residual, const __nv_bfloat16* weight,
    int width, int rows, __nv_bfloat16* normalized, TextFp8Activation activation,
    cudaStream_t stream) {
    if (width != rms_hidden_width || !aligned(branch, alignof(uint4)) ||
        !aligned(residual, alignof(uint4)) || !aligned(weight, alignof(uint4))) {
        const auto status = text_rms_norm_rows(branch, residual, weight, width, rows,
                                              normalized, residual, stream);
        return status == cudaSuccess
            ? text_quantize_fp8_rows(normalized, width, rows, activation, stream)
            : status;
    }
    void* args[] = {&branch, &residual, &weight, &normalized, &activation};
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
