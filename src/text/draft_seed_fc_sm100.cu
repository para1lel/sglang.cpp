#include "sglang/text/decode_proj.h"
#include "sglang/runtime/backend/cuda_resources.h"

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cudaTypedefs.h>

#include <cstdint>
#include <optional>

namespace sglang::decoder {

namespace {

constexpr int input_width = 10240;
constexpr int output_width = 5120;
constexpr int columns = 128;
constexpr int rows = 32;
constexpr int depth_bytes = 128;
constexpr int stages = 5;
constexpr int tile_bytes = (columns + rows) * depth_bytes;
constexpr int groups = input_width * sizeof(__nv_bfloat16) / depth_bytes;
constexpr int threads = 192;

struct Pipeline {
    unsigned tmem;
    unsigned long long ready[stages];
    unsigned long long empty[stages];
    unsigned long long complete;
};

__device__ unsigned address(const void* pointer) {
    return static_cast<unsigned>(__cvta_generic_to_shared(pointer));
}

__device__ void init(unsigned long long* barrier) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(address(barrier)) : "memory");
}

__device__ void wait(unsigned long long* barrier, unsigned parity) {
    asm volatile("{ .reg .pred ready; AGAIN:\n"
                 "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 ready, [%0], %1;\n"
                 "@!ready bra.uni AGAIN; }"
                 :: "r"(address(barrier)), "r"(parity) : "memory");
}

__device__ void commit(unsigned long long* barrier) {
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
                 :: "r"(address(barrier)) : "memory");
}

__device__ void copy(const CUtensorMap* map, unsigned target, unsigned barrier,
                     int col, int row) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes "
                 "[%0], [%1, {%2, %3}], [%4];"
                 :: "r"(target), "l"(map), "r"(col), "r"(row), "r"(barrier) : "memory");
}

__device__ void stage(Pipeline& pipe, unsigned shared, const CUtensorMap* weight,
                      const CUtensorMap* input, int col, int row) {
    for (int group = 0; group < groups; ++group) {
        const int slot = group % stages;
        if (group >= stages) {
            wait(&pipe.empty[slot], (group / stages - 1) % 2);
        }
        const unsigned barrier = address(&pipe.ready[slot]);
        asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
                     :: "r"(barrier), "r"(tile_bytes) : "memory");
        const unsigned start = shared + slot * tile_bytes;
        copy(weight, start, barrier, group * depth_bytes, col);
        copy(input, start + columns * depth_bytes, barrier, group * depth_bytes, row);
    }
}

__device__ std::uint64_t descriptor(unsigned start) {
    constexpr std::uint64_t swizzle = std::uint64_t{2} << 61;
    constexpr std::uint64_t version = std::uint64_t{1} << 46;
    constexpr std::uint64_t stride = std::uint64_t{8 * depth_bytes / 16} << 32;
    constexpr std::uint64_t leading = std::uint64_t{1} << 16;
    return swizzle | version | stride | leading | ((start & 0x3ffffu) >> 4);
}

__device__ void multiply(Pipeline& pipe, unsigned shared) {
    constexpr unsigned instruction =
        ((columns / 16) << 24) | ((rows / 8) << 17) | (1 << 10) | (1 << 7) | (1 << 4);
    for (int group = 0; group < groups; ++group) {
        const int slot = group % stages;
        wait(&pipe.ready[slot], (group / stages) % 2);
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
        const unsigned start = shared + slot * tile_bytes;
        #pragma unroll
        for (int offset = 0; offset < depth_bytes; offset += 32) {
            const auto left = descriptor(start + offset);
            const auto right = descriptor(start + columns * depth_bytes + offset);
            asm volatile("{ .reg .pred accumulate;\n"
                         "setp.ne.u32 accumulate, %4, 0;\n"
                         "tcgen05.mma.cta_group::1.kind::f16 "
                         "[%0], %1, %2, %3, accumulate; }"
                         :: "r"(pipe.tmem), "l"(left), "l"(right), "r"(instruction),
                            "r"(group * depth_bytes + offset) : "memory");
        }
        commit(&pipe.empty[slot]);
    }
    commit(&pipe.complete);
}

__device__ void store_tile(unsigned tmem, int warp, int lane,
                           __nv_bfloat16 (&result)[rows][columns + 2]) {
    #pragma unroll
    for (int row = 0; row < rows; row += 8) {
        float values[8];
        const unsigned source = tmem + ((warp * 32) << 16) + row;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 "
                     "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8];\ntcgen05.wait::ld.sync.aligned;"
                     : "=f"(values[0]), "=f"(values[1]), "=f"(values[2]), "=f"(values[3]),
                       "=f"(values[4]), "=f"(values[5]), "=f"(values[6]), "=f"(values[7])
                     : "r"(source) : "memory");
        #pragma unroll
        for (int index = 0; index < 8; ++index) {
            result[row + index][warp * 32 + lane] = __float2bfloat16_rn(values[index]);
        }
    }
}

__global__ void seed_tc_kernel(const __grid_constant__ CUtensorMap weights,
                               const __grid_constant__ CUtensorMap input, int height,
                               __nv_bfloat16* output) {
    alignas(1024) extern __shared__ unsigned char tiles[];
    __shared__ Pipeline pipe;
    alignas(16) __shared__ __nv_bfloat16 result[rows][columns + 2];
    const int col = (blockIdx.x % (output_width / columns)) * columns;
    const int row = (blockIdx.x / (output_width / columns)) * rows;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    if (threadIdx.x == 0) {
        for (int slot = 0; slot < stages; ++slot) {
            init(&pipe.ready[slot]);
            init(&pipe.empty[slot]);
        }
        init(&pipe.complete);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    if (warp == 4) {
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                     :: "r"(address(&pipe.tmem)), "r"(32) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
    }
    __syncthreads();
    if (warp < 4) {
        wait(&pipe.complete, 0);
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
        store_tile(pipe.tmem, warp, lane, result);
        asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    } else if (warp == 4 && lane == 0) {
        multiply(pipe, address(tiles));
    } else if (warp == 5 && lane == 0) {
        stage(pipe, address(tiles), &weights, &input, col, row);
    }
    __syncthreads();
    if (warp == 4) {
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                     :: "r"(pipe.tmem), "r"(32) : "memory");
    }
    for (int index = threadIdx.x * 2; index < rows * columns; index += threads * 2) {
        const int offset = index / columns;
        const int channel = index % columns;
        if (row + offset < height) {
            auto* destination = output + (row + offset) * output_width + col + channel;
            *reinterpret_cast<unsigned*>(destination) =
                *reinterpret_cast<const unsigned*>(&result[offset][channel]);
        }
    }
}

CUresult encode(PFN_cuTensorMapEncodeTiled_v12000 function, CUtensorMap& map,
                const __nv_bfloat16* pointer, int height, int tile) {
    constexpr cuuint64_t stride[] = {input_width * sizeof(__nv_bfloat16)};
    constexpr cuuint32_t steps[] = {1, 1};
    const cuuint64_t dimensions[] = {stride[0], static_cast<cuuint64_t>(height)};
    const cuuint32_t box[] = {depth_bytes, static_cast<cuuint32_t>(tile)};
    return function(&map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<__nv_bfloat16*>(pointer),
                    dimensions, stride, box, steps, CU_TENSOR_MAP_INTERLEAVE_NONE,
                    CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
                    CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

}

static cudaError_t submit_status(cudaError_t status) {
    const auto recorded = cudaGetLastError();
    return status == cudaSuccess ? recorded : status;
}

std::optional<cudaError_t> try_seed_fc_sm100(const __nv_bfloat16* weight,
                                            const __nv_bfloat16* input, int height,
                                            __nv_bfloat16* output, cudaStream_t stream) {
    const auto pending = cudaGetLastError();
    if (pending != cudaSuccess) return pending;
    decode_proj::LaunchTarget target{};
    const auto prepared = decode_proj::prepare_launch(stream, target);
    if (!prepared) {
        const auto recorded = cudaGetLastError();
        if (recorded != cudaSuccess && recorded != cudaErrorNotSupported) return recorded;
        return std::nullopt;
    }
    if (*prepared != cudaSuccess) return submit_status(*prepared);
    constexpr int shared = stages * tile_bytes;
    cudaFuncAttributes attributes{};
    auto status = cudaFuncGetAttributes(&attributes, seed_tc_kernel);
    if (status == cudaErrorInvalidDeviceFunction || status == cudaErrorNoKernelImageForDevice) {
        const auto recorded = cudaGetLastError();
        if (recorded != cudaSuccess && recorded != status) return recorded;
        return std::nullopt;
    }
    if (status != cudaSuccess) return submit_status(status);
    int limit = 0;
    status = cudaDeviceGetAttribute(&limit, cudaDevAttrMaxSharedMemoryPerBlockOptin, target.device);
    if (status != cudaSuccess) return submit_status(status);
    if (static_cast<std::size_t>(limit) < shared + attributes.sharedSizeBytes) return std::nullopt;

    CUtensorMap weights{}, activation{};
    auto encoded = encode(target.encode, weights, weight, output_width, columns);
    if (encoded == CUDA_SUCCESS) {
        encoded = encode(target.encode, activation, input, height, rows);
    }
    if (encoded != CUDA_SUCCESS) {
        if (encoded == CUDA_ERROR_NOT_SUPPORTED) {
            return std::nullopt;
        }
        return CudaError::from_driver(encoded);
    }
    status = cudaFuncSetAttribute(seed_tc_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  shared);
    if (status != cudaSuccess) return submit_status(status);
    const auto blocks = (output_width / columns) * ((height + rows - 1) / rows);
    void* arguments[] = {&weights, &activation, &height, &output};
    return submit_status(cudaLaunchKernel(seed_tc_kernel, blocks, threads, arguments, shared,
                                          stream));
}

}
