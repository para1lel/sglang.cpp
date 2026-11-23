#include "sglang/text/mtp/draft_depth.h"
#include "sglang/decoder_internal.h"
#include "sglang/text/decode_proj_sm100.cuh"
#include <cuda_runtime.h>
#include <math_constants.h>

namespace sglang::decoder {
namespace {
using namespace sglang::decode_proj;
constexpr int K = hidden_size;
constexpr int V = vocab_size;
constexpr unsigned mask = 0xffffffff;
static_assert(K * 127 * 127 < 2147483647 && K % 128 == 0 && V % tile_rows == 0);

__device__ __nv_bfloat16 logit(int sum, float w, float a) {
    return __double2bfloat16(static_cast<double>(sum) * w * a / (127 * 127));
}

__global__ void quantize(const __grid_constant__ DraftDepthRows rows, int count,
                         std::int8_t* packed, float* maxima) {
    const int r = blockIdx.x;
    const int lane = threadIdx.x;
    if (r >= count) return;
    const auto* src = rows.row[r].input;
    auto* dst = packed + r * K;
    float maximum = 0;
    bool invalid = false;
    for (int k = lane; k < K; k += 32) {
        const float x = __bfloat162float(src[k]);
        invalid |= !isfinite(x);
        maximum = fmaxf(maximum, fabsf(x));
    }
    invalid = __any_sync(mask, invalid);
    for (int s = 16; s; s >>= 1) maximum = fmaxf(maximum, __shfl_down_sync(mask, maximum, s));
    maximum = __shfl_sync(mask, maximum, 0);
    const bool inverse_safe = maximum >= 0x1p-120f && maximum <= 0x1p120f;
    const float inverse = inverse_safe ? 127.f / maximum : 0;
    for (int k = lane; k < K; k += 32) {
        int q = 0;
        if (!invalid && maximum != 0) {
            const float x = __bfloat162float(src[k]);
            q = __float2int_rn(inverse_safe ? x * inverse : 127.f * (x / maximum));
        }
        dst[k] = q;
    }
    if (lane == 0) {
        maxima[r] = invalid ? CUDART_NAN_F : maximum;
        *rows.row[r].maximum = maxima[r];
    }
}

template<int Cols, int Stages>
__device__ void multiply_int(DecodePipeline<Stages>& pipe, unsigned shared) {
    constexpr unsigned instruction = (tile_rows / 16 << 24) | (Cols / 8 << 17) |
                                     (1u << 10) | (1u << 7) | (2u << 4);
    for (int k = 0; k < K / 128; ++k) {
        const int stage = k % Stages;
        wait_barrier(&pipe.copy_ready[stage], (k / Stages) & 1);
        fence_after_sync();
        const unsigned start = shared + stage * stage_bytes<Cols>;
        #pragma unroll
        for (int chunk = 0; chunk < 128; chunk += 32) {
            const auto w = matrix_desc(start + chunk);
            const auto a = matrix_desc(start + weight_bytes + chunk);
            asm volatile("{ .reg .pred p; setp.ne.u32 p, %4, 0; "
                         "tcgen05.mma.cta_group::1.kind::i8 [%0], %1, %2, %3, p; }"
                         :: "r"(pipe.tmem), "l"(w), "l"(a), "r"(instruction),
                            "r"(k | chunk) : "memory");
        }
        commit_mma(&pipe.copy_empty[stage]);
    }
    commit_mma(&pipe.accum_ready[0]);
}

template<int Cols, int Stages>
__global__ __launch_bounds__(threads, 2) void head_tc(
    const __grid_constant__ CUtensorMap weight, const __grid_constant__ CUtensorMap activation,
    const float* weight_max, const float* input_max, const __grid_constant__ DraftDepthRows rows,
    int count) {
    alignas(1024) extern __shared__ unsigned char storage[];
    __shared__ DecodePipeline<Stages> pipe;
    const int warp = threadIdx.x / 32;
    const int base = blockIdx.x * tile_rows;
    if (!threadIdx.x) init_pipeline(pipe);
    if (warp == mma_warp) {
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                     :: "r"(shared_address(&pipe.tmem)), "r"(32) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
    }
    __syncthreads();
    const unsigned shared = shared_address(storage);
    if (warp < consumer_warps) {
        wait_barrier(&pipe.accum_ready[0], 0);
        fence_after_sync();
        const int token = base + threadIdx.x;
        const auto src = pipe.tmem + (static_cast<unsigned>(warp * 32) << 16);
        const float maximum = weight_max[token];
        #pragma unroll
        for (int i = 0; i < Cols; i += 8) {
            unsigned v[8];
            load_tmem8(src + i, v);
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int r = i + j;
                if (r < count) {
                    rows.row[r].output[token] = logit(static_cast<int>(v[j]), maximum,
                                                      input_max[r]);
                }
            }
        }
        fence_before_sync();
    } else if (warp == mma_warp && threadIdx.x % 32 == 0) {
        multiply_int<Cols>(pipe, shared);
    } else if (warp == copy_warp && threadIdx.x % 32 == 0) {
        produce<128, Cols>(pipe, shared, &weight, &activation, base, 0, 0, K / 128);
    }
    __syncthreads();
    if (warp == mma_warp) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                     :: "r"(pipe.tmem), "r"(32) : "memory");
    }
}

template<int Cols>
std::optional<cudaError_t> configure() {
    const auto status = cudaFuncSetAttribute(head_tc<Cols, 2>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, 2 * stage_bytes<Cols>);
    const auto recorded = cudaGetLastError();
    if (recorded != cudaSuccess && recorded != status) return recorded;
    if (status == cudaErrorInvalidValue || status == cudaErrorNotSupported ||
        status == cudaErrorInvalidDeviceFunction || status == cudaErrorNoKernelImageForDevice) {
        return std::nullopt;
    }
    return status;
}

}

std::optional<cudaError_t> prepare_draft_depth(DraftDepthScratch& scratch,
                                             const std::uint8_t* weights, cudaStream_t stream) {
    const auto pending = cudaGetLastError();
    if (pending != cudaSuccess) return pending;
    LaunchTarget target{};
    const auto status = prepare_launch(stream, target);
    const auto recorded = cudaGetLastError();
    const auto expected = status.value_or(cudaErrorNotSupported);
    if (recorded != cudaSuccess && recorded != expected) return recorded;
    if (status != cudaSuccess) return status;
    const auto type = CU_TENSOR_MAP_DATA_TYPE_UINT8;
    constexpr int rows = decoder_batch_max_lanes;
    if (encode_map(target, scratch.weights, weights, type, 1, V, K, tile_rows) != CUDA_SUCCESS ||
        encode_map(target, scratch.activation16, scratch.activation, type, 1, rows, K, 16) !=
            CUDA_SUCCESS ||
        encode_map(target, scratch.activation32, scratch.activation, type, 1, rows, K, 32) !=
            CUDA_SUCCESS) {
        return std::nullopt;
    }
    const auto small = configure<16>();
    if (!small || *small != cudaSuccess) return small;
    return configure<32>();
}

cudaError_t project_draft_depth(const DraftDepthScratch& scratch, const float* maxima,
                                DraftDepthRows rows, int count, cudaStream_t stream) {
    if (count < 1 || count > decoder_batch_max_lanes) return cudaErrorInvalidValue;
    quantize<<<count, 32, 0, stream>>>(rows, count,
        reinterpret_cast<std::int8_t*>(scratch.activation), scratch.maxima);
    if (const auto status = cudaGetLastError(); status != cudaSuccess) return status;
    if (count <= 16) {
        head_tc<16, 2><<<V / tile_rows, threads, 2 * stage_bytes<16>, stream>>>(
            scratch.weights, scratch.activation16, maxima, scratch.maxima, rows, count);
    } else {
        head_tc<32, 2><<<V / tile_rows, threads, 2 * stage_bytes<32>, stream>>>(
            scratch.weights, scratch.activation32, maxima, scratch.maxima, rows, count);
    }
    return cudaGetLastError();
}
}
