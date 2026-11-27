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
constexpr int winner_tiles = V / tile_rows;
constexpr unsigned invalid_winner = 0xffffffffu;
static_assert(draft_depth_winner_tile_rows == tile_rows && tile_rows == 128 && V <= (1 << 18));

struct DraftDepthWinners {
    const unsigned* tiles;
    TextLogitCandidate* rows[draft_depth_winner_rows];
};

__device__ unsigned winner_key(__nv_bfloat16 score, int column) {
    const unsigned bits = __bfloat16_as_ushort(score);
    const unsigned magnitude = bits & 0x7fffu;
    if (magnitude >= 0x7f80u) return invalid_winner;
    const unsigned order = magnitude ? ((bits & 0x8000u) ? 0xffffu - bits : 0x8000u + bits)
                                     : 0x8000u;
    return (order << 8) | ((127u - column) << 1) | (bits >> 15);
}

__device__ unsigned long long winner_tile_key(unsigned key, int tile) {
    if (key == invalid_winner) return ~0ull;
    const unsigned token = tile * tile_rows + 127u - ((key >> 1) & 127u);
    return (static_cast<unsigned long long>(key >> 8) << 19) |
           (static_cast<unsigned long long>(V - 1 - token) << 1) | (key & 1u);
}

__device__ unsigned long long warp_winner(unsigned long long key) {
    const auto high = __reduce_max_sync(mask, static_cast<unsigned>(key >> 32));
    const auto low = __reduce_max_sync(mask, (key >> 32) == high ? static_cast<unsigned>(key) : 0u);
    return (static_cast<unsigned long long>(high) << 32) | low;
}

__global__ void finish_winners(DraftDepthWinners output) {
    __shared__ unsigned long long warp_keys[8];
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const auto* tiles = output.tiles + blockIdx.x * winner_tiles;
    unsigned long long key = 0;
    for (int tile = threadIdx.x; tile < winner_tiles; tile += blockDim.x) {
        key = max(key, winner_tile_key(tiles[tile], tile));
    }
    key = warp_winner(key);
    if (!lane) warp_keys[warp] = key;
    __syncthreads();
    if (warp) return;
    key = warp_winner(lane < 8 ? warp_keys[lane] : 0ull);
    if (lane) return;
    if (key == ~0ull) {
        *output.rows[blockIdx.x] = {-CUDART_INF_F, -2};
        return;
    }
    const unsigned order = key >> 19;
    const unsigned bits = order == 0x8000u ? (key & 1u) << 15 :
                          (order > 0x8000u ? order - 0x8000u : 0xffffu - order);
    *output.rows[blockIdx.x] = {__bfloat162float(__ushort_as_bfloat16(bits)),
                               V - 1 - static_cast<int>((key >> 1) & 0x3ffffu)};
}

__device__ __nv_bfloat16 logit(int sum, float w, float a) {
    return __double2bfloat16(static_cast<double>(sum) * w * a / (127 * 127));
}

template<int Threads>
__global__ void quantize(const __grid_constant__ DraftDepthRows rows, int count,
                         std::int8_t* packed, float* maxima) {
    const int r = blockIdx.x;
    const int lane = threadIdx.x % 32;
    if (r >= count) return;
    const auto* src = rows.row[r].input;
    auto* dst = packed + r * K;
    float maximum = 0;
    bool invalid = false;
    for (int k = threadIdx.x; k < K; k += Threads) {
        const float x = __bfloat162float(src[k]);
        invalid |= !isfinite(x);
        maximum = fmaxf(maximum, fabsf(x));
    }
    if constexpr (Threads == 32) {
        invalid = __any_sync(mask, invalid);
        for (int s = 16; s; s >>= 1) maximum = fmaxf(maximum, __shfl_down_sync(mask, maximum, s));
        maximum = __shfl_sync(mask, maximum, 0);
    } else {
        __shared__ unsigned warp_maxima[Threads / 32];
        const auto key = __reduce_max_sync(
            mask, invalid ? invalid_winner : __float_as_uint(maximum));
        if (!lane) warp_maxima[threadIdx.x / 32] = key;
        __syncthreads();
        unsigned bits = 0;
        for (const unsigned value : warp_maxima) bits = max(bits, value);
        invalid = bits == invalid_winner;
        maximum = invalid ? 0.f : __uint_as_float(bits);
    }
    const bool inverse_safe = maximum >= 0x1p-120f && maximum <= 0x1p120f;
    const float inverse = inverse_safe ? 127.f / maximum : 0;
    for (int k = threadIdx.x; k < K; k += Threads) {
        int q = 0;
        if (!invalid && maximum != 0) {
            const float x = __bfloat162float(src[k]);
            q = __float2int_rn(inverse_safe ? x * inverse : 127.f * (x / maximum));
        }
        dst[k] = q;
    }
    if (threadIdx.x == 0) {
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

template<int Cols, int Stages, bool Fused = false>
__global__ __launch_bounds__(threads, 2) void head_tc(
    const __grid_constant__ CUtensorMap weight, const __grid_constant__ CUtensorMap activation,
    const float* weight_max, const float* input_max, const __grid_constant__ DraftDepthRows rows,
    int count, unsigned* partial = nullptr) {
    alignas(1024) extern __shared__ unsigned char storage[];
    __shared__ DecodePipeline<Stages> pipe;
    __shared__ unsigned tiles[Fused ? consumer_warps : 1][Fused ? Cols : 1];
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
                    const auto score = logit(static_cast<int>(v[j]), maximum, input_max[r]);
                    if constexpr (Fused) {
                        const auto key = __reduce_max_sync(mask, winner_key(score, threadIdx.x));
                        if (threadIdx.x % 32 == 0) tiles[warp][r] = key;
                    } else {
                        rows.row[r].output[token] = score;
                    }
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
    if constexpr (Fused) {
        const int row = threadIdx.x;
        if (row < Cols) {
            auto key = tiles[0][row];
            #pragma unroll
            for (int warp = 1; warp < consumer_warps; ++warp) key = max(key, tiles[warp][row]);
            partial[row * winner_tiles + blockIdx.x] = key;
        }
    }
    if (warp == mma_warp) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                     :: "r"(pipe.tmem), "r"(32) : "memory");
    }
}

std::optional<cudaError_t> setup_status(std::optional<cudaError_t> direct) {
    const auto recorded = cudaGetLastError();
    const auto expected = direct.value_or(cudaErrorNotSupported);
    if (recorded != cudaSuccess && recorded != expected && !CudaError::terminal(expected)) {
        return recorded;
    }
    return direct;
}

std::optional<cudaError_t> tensor_map_status(CUresult status) {
    switch (status) {
        case CUDA_SUCCESS: return cudaSuccess;
        case CUDA_ERROR_NOT_SUPPORTED: return std::nullopt;
        case CUDA_ERROR_DEINITIALIZED: return cudaErrorCudartUnloading;
        case CUDA_ERROR_INVALID_CONTEXT: return cudaErrorIncompatibleDriverContext;
        case CUDA_ERROR_NOT_INITIALIZED: return cudaErrorInitializationError;
        case CUDA_ERROR_OUT_OF_MEMORY: return cudaErrorMemoryAllocation;
        case CUDA_ERROR_ILLEGAL_ADDRESS: return cudaErrorIllegalAddress;
        case CUDA_ERROR_LAUNCH_FAILED: return cudaErrorLaunchFailure;
        default: return cudaErrorInvalidValue;
    }
}

template<int Cols, int Stages = 2, bool Fused = false>
std::optional<cudaError_t> configure() {
    const auto status = cudaFuncSetAttribute(head_tc<Cols, Stages, Fused>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, Stages * stage_bytes<Cols>);
    const auto resolved = setup_status(status);
    if (resolved != status) return resolved;
    if (status == cudaErrorInvalidValue || status == cudaErrorNotSupported ||
        status == cudaErrorInvalidDeviceFunction || status == cudaErrorNoKernelImageForDevice) {
        return std::nullopt;
    }
    return status;
}

}

std::optional<cudaError_t> prepare_draft_depth(DraftDepthScratch& scratch,
                                             const std::uint8_t* weights, bool tensor_parallel,
                                             cudaStream_t stream) {
    const auto pending = cudaGetLastError();
    if (pending != cudaSuccess) return pending;
    LaunchTarget target{};
    const auto status = setup_status(prepare_launch(stream, target));
    if (status != cudaSuccess) return status;
    const auto type = CU_TENSOR_MAP_DATA_TYPE_UINT8;
    constexpr int rows = decoder_batch_max_lanes;
    const auto weight_map = setup_status(tensor_map_status(
        encode_map(target, scratch.weights, weights, type, 1, V, K, tile_rows)));
    if (weight_map != cudaSuccess) return weight_map;
    const auto small_map = setup_status(tensor_map_status(
        encode_map(target, scratch.activation16, scratch.activation, type, 1, rows, K, 16)));
    if (small_map != cudaSuccess) return small_map;
    const auto large_map = setup_status(tensor_map_status(
        encode_map(target, scratch.activation32, scratch.activation, type, 1, rows, K, 32)));
    if (large_map != cudaSuccess) return large_map;
    const auto small = configure<16>();
    if (!small || *small != cudaSuccess) return small;
    const auto large = configure<32>();
    if (large != cudaSuccess) return large;
    scratch.fused_winner = false;
    if (tensor_parallel) {
        const auto fused = configure<draft_depth_winner_rows, 4, true>();
        if (fused && *fused != cudaSuccess) return fused;
        scratch.fused_winner = fused.has_value();
    }
    return cudaSuccess;
}

cudaError_t project_draft_depth(const DraftDepthScratch& scratch, const float* maxima,
                                DraftDepthRows rows, int count, cudaStream_t stream) {
    if (count < 1 || count > decoder_batch_max_lanes) return cudaErrorInvalidValue;
    quantize<32><<<count, 32, 0, stream>>>(rows, count,
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

cudaError_t score_draft_depth16(const DraftDepthScratch& scratch, const float* maxima,
                                 DraftDepthRows rows, TextLogitCandidate* const* winners,
                                 cudaStream_t stream) {
    constexpr int count = draft_depth_winner_rows;
    constexpr int stages = 4;
    auto* partial = reinterpret_cast<unsigned*>(scratch.maxima + decoder_batch_max_lanes);
    quantize<128><<<count, 128, 0, stream>>>(rows, count,
        reinterpret_cast<std::int8_t*>(scratch.activation), scratch.maxima);
    if (const auto status = cudaGetLastError(); status != cudaSuccess) return status;
    head_tc<count, stages, true><<<winner_tiles, threads, stages * stage_bytes<count>, stream>>>(
        scratch.weights, scratch.activation16, maxima, scratch.maxima, rows, count, partial);
    if (const auto status = cudaGetLastError(); status != cudaSuccess) return status;
    DraftDepthWinners output{partial, {}};
    for (int row = 0; row < count; ++row) output.rows[row] = winners[row];
    finish_winners<<<count, 256, 0, stream>>>(output);
    return cudaGetLastError();
}
}
