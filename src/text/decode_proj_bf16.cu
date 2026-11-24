#include "sglang/text/decode_proj.h"
#include "sglang/text/decode_proj_sm100.cuh"

#include <cuda_runtime.h>

namespace sglang {
namespace {

using namespace decode_proj;

constexpr int chunk_elements = 64;
template <int Columns>
__device__ void collect(Pipeline& pipe, int active, float (&result)[Columns]) {
    const int warp = threadIdx.x / warp_size;
    const unsigned lane_base = pipe.tmem + (static_cast<unsigned>(warp * warp_size) << 16);
    wait_barrier(&pipe.accum_ready[0], 0);
    fence_after_sync();
    if (active == 1) {
        result[0] = load_tmem1(lane_base);
        fence_before_sync();
        return;
    }
    #pragma unroll
    for (int block = 0; block < Columns; block += 16) {
        if (block < active) {
            unsigned partial[16];
            load_tmem16(lane_base + block, partial);
            #pragma unroll
            for (int col = 0; col < 16; ++col) {
                result[block + col] = __uint_as_float(partial[col]);
            }
        }
    }
    fence_before_sync();
}

template <int Columns>
__global__ __launch_bounds__(threads, 1)
void bf16_decode(const __grid_constant__ CUtensorMap weight_map,
                 const __grid_constant__ CUtensorMap activation_map, int out_width, int rows,
                 int chunks, __nv_bfloat16* output) {
    constexpr unsigned instruction = ((tile_rows / 16) << 24) | ((Columns / 8) << 17) |
                                     (1u << 10) | (1u << 7) | (1u << 4);
    alignas(1024) extern __shared__ unsigned char tiles[];
    __shared__ Pipeline pipe;
    const int splits = gridDim.x;
    const int first = blockIdx.x * chunks / splits;
    const int count = (blockIdx.x + 1) * chunks / splits - first;
    const int row_base = blockIdx.y * tile_rows;
    const int act_row = blockIdx.z * Columns;
    const int active = min(Columns, rows - act_row);
    const int warp = threadIdx.x / warp_size;
    if (threadIdx.x == 0) {
        init_pipeline(pipe);
    }
    if (warp == mma_warp) {
        allocate_tmem<Columns>(pipe);
    }
    __syncthreads();
    const unsigned shared = shared_address(tiles);
    float result[Columns] = {};
    if (warp < consumer_warps) {
        collect(pipe, active, result);
    } else if (warp == mma_warp && threadIdx.x % warp_size == 0) {
        multiply<instruction, false, Columns>(pipe, shared, count);
    } else if (warp == copy_warp && threadIdx.x % warp_size == 0) {
        produce<chunk_elements, Columns>(pipe, shared, &weight_map, &activation_map, row_base,
                                           act_row, first, count);
    }
    wait_dependencies();
    __syncthreads();
    if (warp == mma_warp) {
        free_tmem<Columns>(pipe);
    }
    store_result(result, reinterpret_cast<float*>(tiles), splits, active, row_base, out_width,
                 act_row, output);
}

}

std::optional<cudaError_t> try_text_bf16_decode_sm100(
    const __nv_bfloat16* weight, const __nv_bfloat16* input, int out_width, int in_width,
    int rows, __nv_bfloat16* output, cudaStream_t stream) {
    if (in_width % chunk_elements != 0 || !aligned16(weight) || !aligned16(input)) {
        return std::nullopt;
    }
    LaunchTarget target{};
    if (auto prepared = prepare_launch(stream, target); !prepared || *prepared != cudaSuccess) {
        return prepared;
    }
    int chunks = in_width / chunk_elements;
    const int row_blocks = (out_width + tile_rows - 1) / tile_rows;
    const void* full_kernel = reinterpret_cast<const void*>(bf16_decode<tile_cols>);
    SplitPlan plan{};
    auto status = plan_splits(full_kernel, 1, target, row_blocks, chunks,
                              stages * stage_bytes<tile_cols>, 0, plan);
    if (status != cudaSuccess) {
        return status;
    }
    if (plan.splits == 0) {
        return std::nullopt;
    }
    const int columns = rows <= 16 ? 16 : (rows <= 32 ? 32 : tile_cols);
    const void* kernel = full_kernel;
    if (columns != tile_cols) {
        kernel = columns == 16 ? reinterpret_cast<const void*>(bf16_decode<16>)
                               : reinterpret_cast<const void*>(bf16_decode<32>);
        int limit = 0;
        status = configure(kernel, columns == 16 ? 4 : 5, target.device, limit);
        if (status != cudaSuccess) {
            return status;
        }
    }
    CUtensorMap weight_map, activation_map;
    auto encoded = encode_map(target, weight_map, weight, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2,
                              out_width, in_width, tile_rows);
    if (encoded == CUDA_SUCCESS) {
        encoded = encode_map(target, activation_map, input, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2,
                             rows, in_width, columns);
    }
    switch (encoded) {
        case CUDA_SUCCESS: break;
        case CUDA_ERROR_NOT_SUPPORTED: return std::nullopt;
        case CUDA_ERROR_DEINITIALIZED: return cudaErrorCudartUnloading;
        case CUDA_ERROR_INVALID_CONTEXT: return cudaErrorIncompatibleDriverContext;
        case CUDA_ERROR_NOT_INITIALIZED: return cudaErrorInitializationError;
        case CUDA_ERROR_OUT_OF_MEMORY: return cudaErrorMemoryAllocation;
        default: return cudaErrorInvalidValue;
    }
    void* args[] = {&weight_map, &activation_map, &out_width, &rows, &chunks, &output};
    const dim3 grid(plan.splits, row_blocks, (rows - 1) / columns + 1);
    const int shared = stages * (weight_bytes + columns * 128);
    return launch_split(kernel, grid, shared, args, stream, grid.x);
}

}
