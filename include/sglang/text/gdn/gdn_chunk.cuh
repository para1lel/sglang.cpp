#pragma once

#include <mma.h>
#include <type_traits>

#include "sglang/text/gdn_local.h"

namespace sglang::gdn_chunk {

using Bf16 = __nv_bfloat16;
namespace mma = nvcuda::wmma;

constexpr int tile = 16;
constexpr int rows = 32;
constexpr int min_count = 128;
constexpr int dimension = text_gdn_head_dim;
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

struct Segment {
    int count;
    const Bf16* convolved;
    const float2* gates;
    Matrices* matrices;
    float* recurrent;
    Bf16* observable;
};

struct Segments {
    Segment values[decoder::max_gdn_prefill_segments];
};

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
using ARow = mma::fragment<mma::matrix_a, tile, tile, tile, Bf16, mma::row_major>;
using BCol = mma::fragment<mma::matrix_b, tile, tile, tile, Bf16, mma::col_major>;
using Acc = mma::fragment<mma::accumulator, tile, tile, tile, float>;

template <typename S, int PrepareThreads>
__device__ __forceinline__ void stage_qk(Bf16* target, const Bf16* convolved,
                                        int first, int valid, int qk_head) {
    constexpr int vectors_per_row = dimension / vector_elements;
    for (int vector = threadIdx.x; vector < 2 * rows * vectors_per_row;
         vector += PrepareThreads) {
        const int row = vector / vectors_per_row;
        const int position = row % rows;
        const int component = vector % vectors_per_row * vector_elements;
        const int plane = row / rows;
        uint4 value{};
        if (position < valid) {
            const auto* source = convolved + static_cast<std::size_t>(first + position) *
                S::qkv_width + plane * S::qk_width + qk_head * dimension;
            value = *reinterpret_cast<const uint4*>(source + component);
        }
        *reinterpret_cast<uint4*>(target + row * key_stride + component) = value;
    }
}

template<int Stride>
__device__ __forceinline__ void store_grouped_matrices(Matrices& matrix, const Bf16* solution,
                                                        const Bf16* scores, int thread) {
    __syncthreads();
    using Pair = __nv_bfloat162;
    static_assert(rows % 2 == 0 && Stride % 2 == 0);
    for (int index = thread * 2; index < rows * rows; index += 2 * threads) {
        const int offset = index / rows * Stride + index % rows;
        *reinterpret_cast<Pair*>(matrix.inverse + index) =
            *reinterpret_cast<const Pair*>(solution + offset);
        *reinterpret_cast<Pair*>(matrix.scores + index) =
            *reinterpret_cast<const Pair*>(scores + offset);
    }
}
template <int SharedHeads>
__device__ __forceinline__ void segment_qk_dots(const Bf16* qk, float* dots, int warp) {
    constexpr int prepare_warps = warps * SharedHeads;
    for (int task = warp; task < 2 * time_tiles * time_tiles; task += prepare_warps) {
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
}

template <typename S, int SharedHeads>
__device__ __forceinline__ void prepare_segment(int count, const Bf16* convolved,
                                                const float2* gates, Matrices* matrices) {
    const int first = blockIdx.y * rows;
    if (first >= count) {
        return;
    }
    __shared__ alignas(32) Bf16 qk[2 * rows * key_stride];
    __shared__ alignas(32) float dots[2 * rows * rows];
    __shared__ float lower[SharedHeads][rows * rows];
    __shared__ float decay[SharedHeads][rows];
    __shared__ float beta[SharedHeads][rows];
    const int first_head = blockIdx.x * SharedHeads;
    const int valid = min(count - first, rows);
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    auto* chunk = matrices + blockIdx.y * S::value_heads + first_head;

    stage_qk<S, threads * SharedHeads>(qk, convolved, first, valid,
                                       first_head / S::heads_per_qk);
    if (warp < SharedHeads) {
        const auto gate = lane < valid
            ? gates[(first + lane) * S::value_heads + first_head + warp]
            : make_float2(1.0f, 0.0f);
        decay[warp][lane] = gate.x;
        beta[warp][lane] = gate.y;
        chunk[warp].beta[lane] = gate.y;
    }
    __syncthreads();

    segment_qk_dots<SharedHeads>(qk, dots, warp);
    __syncthreads();

    constexpr bool grouped = SharedHeads > 1;
    constexpr int stride = rows + (grouped ? 2 : 0);
    constexpr int plane_size = rows * stride;
    static_assert(2 * SharedHeads * plane_size <= 2 * rows * key_stride);

    if (warp < SharedHeads) {
        auto& matrix = chunk[warp];
        auto* scores = grouped ? qk + warp * plane_size : matrix.scores;
        float product = 1.0f;
        for (int source = lane; source >= 0; --source) {
            const int index = lane * rows + source;
            lower[warp][source * rows + lane] = source < lane
                ? product * beta[warp][lane] * dots[rows * rows + index] : 0.0f;
            scores[lane * stride + source] = __float2bfloat16(product * dots[index]);
            if (lane == rows - 1) {
                matrix.tail[source] = product;
            }
            product *= decay[warp][source];
        }
        matrix.initial[lane] = product;
    }
    const int value_head = warp / warps;
    const int head_warp = warp % warps;
    auto& matrix = chunk[value_head];
    auto* scores = grouped ? qk + value_head * plane_size : matrix.scores;
    auto* solution = grouped ? qk + (SharedHeads + value_head) * plane_size : matrix.inverse;
    for (int index = head_warp * 32 + lane; index < rows * rows; index += threads) {
        if (index / rows < index % rows) {
            scores[index / rows * stride + index % rows] = __float2bfloat16(0.0f);
        }
    }
    __syncthreads();
    for (int column = head_warp; column < rows; column += warps) {
        float inverse = lane == column ? 1.0f : 0.0f;
        for (int pivot = column; pivot < rows - 1; ++pivot) {
            const float solved = __shfl_sync(0xffffffffu, inverse, pivot);
            if (lane > pivot) {
                inverse = fmaf(-lower[value_head][pivot * rows + lane], solved, inverse);
            }
        }
        solution[lane * stride + column] = __float2bfloat16(inverse);
    }
    if constexpr (grouped) {
        store_grouped_matrices<stride>(matrix, solution, scores, head_warp * 32 + lane);
    }
}

template <typename S = GdnFull, int SharedHeads = 1>
__global__ void prepare(int count, const Bf16* convolved, const float2* gates,
                        Matrices* matrices) {
    prepare_segment<S, SharedHeads>(count, convolved, gates, matrices);
}

template <typename S = GdnFull>
__global__ void prepare_segments(const __grid_constant__ Segments segments) {
    const auto& segment = segments.values[blockIdx.z];
    prepare_segment<S, S::heads_per_qk>(segment.count, segment.convolved, segment.gates,
                                        segment.matrices);
}
#else
template <typename S = GdnFull, int SharedHeads = 1>
__global__ void prepare(int, const Bf16*, const float2*, Matrices*) {
    __trap();
}
template <typename S = GdnFull>
__global__ void prepare_segments(Segments) {
    __trap();
}
#endif

}

#include "sglang/text/gdn/gdn_warp_mma.cuh"

namespace sglang::gdn_chunk {

template <typename S = GdnFull>
inline std::size_t scratch_bytes(int count) {
    return static_cast<std::size_t>((count + rows - 1) / rows) * S::value_heads *
           sizeof(Matrices);
}

template <typename S = GdnFull>
inline bool eligible(int count, const void* scratch, std::size_t bytes) {
    return count >= min_count && scratch && bytes >= scratch_bytes<S>(count) &&
           reinterpret_cast<std::uintptr_t>(scratch) % alignof(Matrices) == 0;
}

template <typename S>
inline constexpr int recurrence_warps = std::is_same_v<S, GdnLocal> ? 4 : warp_mma::block_warps;

template <typename S, bool StageValues = false, bool Pipelined = false>
inline constexpr auto recurrence_kernel =
    warp_mma::recur_warp<S, recurrence_warps<S>, std::is_same_v<S, GdnLocal>,
                        StageValues, Pipelined>;

template <typename S>
inline constexpr bool stage_values(int count) {
    return std::is_same_v<S, GdnLocal> && count >= 1024;
}

template <typename S = GdnFull>
inline cudaError_t configure(bool& enabled, bool grouped = false, bool* staged_values = nullptr,
                             bool* pipelined = nullptr) {
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
    if (grouped) {
        return cudaFuncSetAttribute(warp_mma::recur_segments<S>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, warp_mma::dynamic_bytes);
    }
    if constexpr (std::is_same_v<S, GdnLocal>) {
        if (staged_values) {
            int shared_limit = 0;
            status = cudaDeviceGetAttribute(&shared_limit, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                                            device);
            if (status != cudaSuccess) {
                return status;
            }
            if (shared_limit < warp_mma::local_shared_bytes) {
                enabled = false;
                return cudaSuccess;
            }
            *staged_values = *staged_values && shared_limit >= warp_mma::value_shared_bytes;
            if (pipelined) {
                *pipelined = *staged_values && shared_limit >= warp_mma::pipeline_shared_bytes;
                if (*pipelined) {
                    return cudaFuncSetAttribute(recurrence_kernel<S, true, true>,
                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                        warp_mma::pipeline_dynamic_bytes);
                }
            }
            if (*staged_values) {
                return cudaFuncSetAttribute(recurrence_kernel<S, true>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, warp_mma::value_dynamic_bytes);
            }
        }
    }
    return cudaFuncSetAttribute(recurrence_kernel<S>,
                                cudaFuncAttributeMaxDynamicSharedMemorySize,
                                warp_mma::dynamic_bytes);
}

template <typename S, bool StageValues, bool Pipelined = false>
inline cudaError_t launch_chunk(int count, const Bf16* convolved, const float2* gates,
                                void* scratch, float* recurrent, Bf16* observable,
                                cudaStream_t stream) {
    auto* matrices = static_cast<Matrices*>(scratch);
    void* prepare_args[] = {&count, &convolved, &gates, &matrices};
    constexpr int shared_heads = StageValues ? S::heads_per_qk : 1;
    const auto status = cudaLaunchKernel(
        prepare<S, shared_heads>, dim3(S::value_heads / shared_heads, (count + rows - 1) / rows),
        dim3(threads * shared_heads), prepare_args, 0, stream);
    if (status != cudaSuccess) {
        return status;
    }
    void* recur_args[] = {&count, &convolved, &matrices, &recurrent, &observable};
    constexpr int recurrence_threads = recurrence_warps<S> * (Pipelined ? 64 : 32);
    constexpr int shared = Pipelined ? warp_mma::pipeline_dynamic_bytes :
                           StageValues ? warp_mma::value_dynamic_bytes : warp_mma::dynamic_bytes;
    return cudaLaunchKernel(
        recurrence_kernel<S, StageValues, Pipelined>,
        dim3(S::value_heads, dimension / (warp_mma::columns * recurrence_warps<S>)),
        dim3(recurrence_threads), recur_args, shared, stream);
}

template <typename S = GdnFull>
inline cudaError_t launch(int count, const Bf16* convolved, const float2* gates,
                          void* scratch, float* recurrent, Bf16* observable,
                          cudaStream_t stream, bool staged_values, bool pipelined) {
    if constexpr (std::is_same_v<S, GdnLocal>) {
        if (pipelined) {
            return launch_chunk<S, true, true>(count, convolved, gates, scratch, recurrent,
                                               observable, stream);
        }
        if (staged_values) {
            return launch_chunk<S, true>(count, convolved, gates, scratch, recurrent, observable,
                                         stream);
        }
    }
    return launch_chunk<S, false>(count, convolved, gates, scratch, recurrent, observable, stream);
}

template <typename S = GdnFull>
inline cudaError_t launch_segments(Segments segments, int count, int max_rows,
                                   cudaStream_t stream) {
    void* args[] = {&segments};
    const auto chunks = (max_rows + rows - 1) / rows;
    const auto status = cudaLaunchKernel(prepare_segments<S>,
                                         dim3(S::qk_heads, chunks, count),
                                         dim3(threads * S::heads_per_qk), args, 0, stream);
    if (status != cudaSuccess) {
        return status;
    }
    return cudaLaunchKernel(warp_mma::recur_segments<S>,
                            dim3(S::value_heads, warp_mma::warp_value_tiles, count),
                            dim3(warp_mma::block_threads), args, warp_mma::dynamic_bytes,
                            stream);
}

}
