#pragma once

#include <mma.h>

namespace sglang::gdn_chunk {

using Bf16 = __nv_bfloat16;
namespace mma = nvcuda::wmma;

constexpr int tile = 16;
constexpr int rows = 32;
constexpr int min_count = 128;
constexpr int dimension = text_gdn_head_dim;
constexpr int heads = text_gdn_value_heads;
constexpr int group = heads / text_gdn_qk_heads;
constexpr int warps = 4;
constexpr int threads = warps * 32;
constexpr int time_tiles = rows / tile;
constexpr int key_stride = dimension + tile / 2;
constexpr int vector_elements = sizeof(uint4) / sizeof(Bf16);
static_assert(dimension == warps * 2 * tile);

struct alignas(32) Matrices {
    Bf16 inverse[rows * rows];
    Bf16 scores[rows * rows];
    float initial[rows];
    float tail[rows];
    float beta[rows];
};

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
using ARow = mma::fragment<mma::matrix_a, tile, tile, tile, Bf16, mma::row_major>;
using BCol = mma::fragment<mma::matrix_b, tile, tile, tile, Bf16, mma::col_major>;
using Acc = mma::fragment<mma::accumulator, tile, tile, tile, float>;

__device__ __forceinline__ void stage_qk(Bf16* target, const Bf16* convolved,
                                        int first, int valid, int head) {
    constexpr int vectors_per_row = dimension / vector_elements;
    for (int vector = threadIdx.x; vector < 2 * rows * vectors_per_row; vector += threads) {
        const int row = vector / vectors_per_row;
        const int position = row % rows;
        const int component = vector % vectors_per_row * vector_elements;
        const int plane = row / rows;
        uint4 value{};
        if (position < valid) {
            const auto* source = convolved + static_cast<std::size_t>(first + position) *
                text_gdn_qkv_width + plane * text_gdn_qk_width + head / group * dimension;
            value = *reinterpret_cast<const uint4*>(source + component);
        }
        *reinterpret_cast<uint4*>(target + row * key_stride + component) = value;
    }
}

__global__ void prepare(int count, const Bf16* convolved, const float2* gates,
                        Matrices* matrices) {
    __shared__ alignas(32) Bf16 qk[2 * rows * key_stride];
    __shared__ alignas(32) float dots[2 * rows * rows];
    __shared__ float lower[rows * rows];
    __shared__ float decay[rows];
    __shared__ float beta[rows];
    const int head = blockIdx.x;
    const int first = blockIdx.y * rows;
    const int valid = min(count - first, rows);
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    auto& matrix = matrices[blockIdx.y * heads + head];

    stage_qk(qk, convolved, first, valid, head);
    if (threadIdx.x < rows) {
        const auto gate = threadIdx.x < valid
            ? gates[(first + threadIdx.x) * heads + head] : make_float2(1.0f, 0.0f);
        decay[threadIdx.x] = gate.x;
        beta[threadIdx.x] = gate.y;
        matrix.beta[threadIdx.x] = gate.y;
    }
    __syncthreads();

    for (int task = warp; task < 2 * time_tiles * time_tiles; task += warps) {
        const int plane = task / (time_tiles * time_tiles);
        const int target = task / time_tiles % time_tiles * tile;
        const int source = task % time_tiles * tile;
        if (source > target) {
            continue;
        }
        Acc result;
        mma::fill_fragment(result, 0.0f);
        #pragma unroll
        for (int key = 0; key < dimension; key += tile) {
            ARow lhs;
            BCol rhs;
            mma::load_matrix_sync(lhs, qk + (plane * rows + target) * key_stride + key,
                                  key_stride);
            mma::load_matrix_sync(rhs, qk + (rows + source) * key_stride + key, key_stride);
            mma::mma_sync(result, lhs, rhs, result);
        }
        mma::store_matrix_sync(dots + (plane * rows + target) * rows + source,
                               result, rows, mma::mem_row_major);
    }
    __syncthreads();

    if (threadIdx.x < rows) {
        const int target = threadIdx.x;
        float product = 1.0f;
        for (int source = target; source >= 0; --source) {
            const int index = target * rows + source;
            lower[source * rows + target] = source < target
                ? product * beta[target] * dots[rows * rows + index] : 0.0f;
            matrix.scores[index] = __float2bfloat16(product * dots[index]);
            if (target == rows - 1) {
                matrix.tail[source] = product;
            }
            product *= decay[source];
        }
        matrix.initial[target] = product;
    }
    for (int index = threadIdx.x; index < rows * rows; index += threads) {
        if (index / rows < index % rows) {
            matrix.scores[index] = __float2bfloat16(0.0f);
        }
    }
    __syncthreads();

    for (int column = warp; column < rows; column += warps) {
        float inverse = lane == column ? 1.0f : 0.0f;
        for (int pivot = column; pivot < rows - 1; ++pivot) {
            const float solved = __shfl_sync(0xffffffffu, inverse, pivot);
            if (lane > pivot) {
                inverse = fmaf(-lower[pivot * rows + lane], solved, inverse);
            }
        }
        matrix.inverse[lane * rows + column] = __float2bfloat16(inverse);
    }
}
#else
__global__ void prepare(int, const Bf16*, const float2*, Matrices*) {
    __trap();
}
#endif

}

#include "sglang/text/gdn_warp_mma.cuh"

namespace sglang::gdn_chunk {

inline std::size_t scratch_bytes(int count) {
    return static_cast<std::size_t>((count + rows - 1) / rows) * heads * sizeof(Matrices);
}

inline bool eligible(int count, const void* scratch, std::size_t bytes) {
    return count >= min_count && scratch && bytes >= scratch_bytes(count) &&
           reinterpret_cast<std::uintptr_t>(scratch) % alignof(Matrices) == 0;
}

inline cudaError_t configure(bool& enabled) {
    int device = 0, major = 0, minor = 0;
    auto status = cudaGetDevice(&device);
    if (status == cudaSuccess) {
        status = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
    }
    if (status == cudaSuccess) {
        status = cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
    }
    if (status != cudaSuccess) {
        return status;
    }
    enabled = major == 10 && minor == 0;
    if (!enabled) {
        return cudaSuccess;
    }
    return cudaFuncSetAttribute(warp_mma::recur_warp,
        cudaFuncAttributeMaxDynamicSharedMemorySize, warp_mma::dynamic_bytes);
}

inline cudaError_t launch(int count, const Bf16* convolved, const float2* gates,
                          void* scratch, float* recurrent, Bf16* observable, cudaStream_t stream) {
    auto* matrices = static_cast<Matrices*>(scratch);
    void* prepare_args[] = {&count, &convolved, &gates, &matrices};
    const auto status = cudaLaunchKernel(prepare, dim3(heads, (count + rows - 1) / rows),
                                         dim3(threads), prepare_args, 0, stream);
    if (status != cudaSuccess) {
        return status;
    }
    void* recur_args[] = {&count, &convolved, &matrices, &recurrent, &observable};
    return cudaLaunchKernel(warp_mma::recur_warp, dim3(heads, warp_mma::warp_value_tiles),
                            dim3(threads), recur_args, warp_mma::dynamic_bytes, stream);
}

}
