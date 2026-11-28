#include "sglang/decoder_cuda.cuh"
#include "sglang/decoder_internal.h"
#include "sglang/prefill_fp8.h"
#include "sglang/text/decode_proj.h"
#include "sglang/text/decode_proj_sm100.cuh"

#include <cuda_runtime.h>

namespace sglang::decoder {
namespace {

using namespace decode_proj;

constexpr int min_rows = 512;
constexpr int max_rows = grouped_prefill_max_rows;
constexpr int columns = 32;
constexpr int copy_stages = 3;
constexpr int splits = 10;
constexpr int chunks_per_split = hidden_size * sizeof(Bf16) / 128 / splits;
constexpr int reduction_threads = 256;
static_assert(hidden_size == 5120);
static_assert((hidden_size * sizeof(Bf16) / 128) % splits == 0);

__device__ void copy_gate_tiles(DecodePipeline<copy_stages>& pipe, unsigned tiles,
                                const CUtensorMap* weight_b, const CUtensorMap* weight_a,
                                const CUtensorMap* input, int row, int split) {
    for (int chunk = 0; chunk < chunks_per_split; ++chunk) {
        const int stage = chunk % copy_stages;
        auto* ready = &pipe.copy_ready[stage];
        if (chunk >= copy_stages) {
            wait_barrier(&pipe.copy_empty[stage], (chunk / copy_stages - 1) & 1);
        }
        expect_bytes(ready, stage_bytes<columns>);
        const unsigned start = tiles + stage * stage_bytes<columns>;
        const int k = (split * chunks_per_split + chunk) * 128;
        copy_tile(weight_b, start, ready, k, 0);
        copy_tile(weight_a, start + weight_bytes / 2, ready, k, 0);
        copy_tile(input, start + weight_bytes, ready, k, row);
    }
}

template <int Channels>
__global__ __launch_bounds__(threads) void project_gdn_gate_partials(
    const __grid_constant__ CUtensorMap weight_b, const __grid_constant__ CUtensorMap weight_a,
    const __grid_constant__ CUtensorMap input, int rows, float* partials) {
    constexpr unsigned instruction = bf16_instruction<columns>;
    constexpr int channels = Channels;
    constexpr int gates = 2 * channels;

    alignas(1024) extern __shared__ unsigned char tiles[];
    __shared__ DecodePipeline<copy_stages> pipe;
    alignas(16) __shared__ float results[columns][tile_rows + 2];
    const int row = blockIdx.x * columns;
    const int split = blockIdx.y;
    const int warp = threadIdx.x / warp_size;
    const int lane = threadIdx.x % warp_size;
    if (threadIdx.x == 0) {
        init_pipeline(pipe);
    }
    if (warp == mma_warp) {
        allocate_tmem(&pipe.tmem, tmem_columns<columns, 1>);
    }
    __syncthreads();
    if (warp == copy_warp && lane == 0) {
        copy_gate_tiles(pipe, shared_address(tiles), &weight_b, &weight_a, &input, row, split);
    } else if (warp == mma_warp && lane == 0) {
        multiply<instruction, false, columns>(pipe, shared_address(tiles), chunks_per_split);
    } else if (warp < consumer_warps) {
        wait_barrier(&pipe.accum_ready[0], 0);
        fence_after_sync();
        const unsigned source = pipe.tmem + (static_cast<unsigned>(warp * warp_size) << 16);
        #pragma unroll
        for (int offset = 0; offset < columns; offset += 8) {
            unsigned values[8];
            load_tmem8(source + offset, values);
            #pragma unroll
            for (int item = 0; item < 8; ++item) {
                results[offset + item][warp * warp_size + lane] = __uint_as_float(values[item]);
            }
        }
        fence_before_sync();
    }
    __syncthreads();
    if (warp == mma_warp) {
        fence_after_sync();
        free_tmem(&pipe.tmem, tmem_columns<columns, 1>);
    }
    for (int index = threadIdx.x * 2; index < columns * gates; index += threads * 2) {
        const int local_row = index / gates;
        if (row + local_row >= rows) break;
        const int gate = index % gates;
        const auto* source = &results[local_row][gate / channels * (tile_rows / 2) +
                                                  gate % channels];
        auto* dest = partials + split * rows * gates + row * gates + index;
        *reinterpret_cast<float2*>(dest) = *reinterpret_cast<const float2*>(source);
    }
}

template <int Channels>
__device__ Bf16* gate_output(Bf16* b, Bf16* a, int row, int gate) {
    return (gate < Channels ? b : a) + row * Channels + gate % Channels;
}

template <int Channels>
__global__ void round_gdn_gates(const float* partials, int elements, Bf16* b, Bf16* a) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    constexpr int gates = 2 * Channels;
    float sum = partials[index];
    #pragma unroll
    for (int split = 1; split < splits; ++split) {
        sum += partials[split * elements + index];
    }
    *gate_output<Channels>(b, a, index / gates, index % gates) = __float2bfloat16_rn(sum);
}

CUresult encode_gate_map(const LaunchTarget& target, CUtensorMap& map, const Bf16* data,
                         int height, int tile) {
    return encode_tiled_map(
        target.encode, map, data,
        {CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, height, hidden_size * sizeof(Bf16), tile},
        CU_TENSOR_MAP_L2_PROMOTION_NONE);
}

cudaError_t take_launch_status(cudaError_t direct) {
    const auto recorded = cudaGetLastError();
    if (direct == cudaSuccess || (!CudaError::terminal(direct) && CudaError::terminal(recorded))) {
        return recorded;
    }
    return direct;
}

std::optional<cudaError_t> descriptor_error(CUresult result) {
    if (result == CUDA_ERROR_NOT_SUPPORTED) {
        const auto recorded = cudaGetLastError();
        if (recorded == cudaSuccess || recorded == cudaErrorNotSupported) {
            return std::nullopt;
        }
        return recorded;
    }
    return take_launch_status(CudaError::from_driver(result));
}

template <int Channels>
std::optional<cudaError_t> prefill_bf16_gdn_gates_sm100(
    const Bf16* weight_b, const Bf16* weight_a, const Bf16* input, int rows, Bf16* b, Bf16* a,
    void* scratch, std::size_t scratch_bytes, cudaStream_t stream) {
    static_assert(Channels == text_gdn_value_heads || Channels == text_gdn_value_heads / 2);
    constexpr int gates = 2 * Channels;
    if (rows < min_rows || rows > max_rows) return std::nullopt;
    if (const auto pending = cudaGetLastError(); pending != cudaSuccess) return pending;
    int elements = rows * gates;
    const auto scratch_size = std::size_t(splits * elements) * sizeof(float);
    if (!weight_b || !weight_a || !input || !b || !a || !scratch || scratch_bytes < scratch_size ||
        !aligned16(weight_b) || !aligned16(weight_a) || !aligned16(input) || !aligned16(scratch)) {
        return std::nullopt;
    }
    LaunchTarget target{};
    const auto prepared = prepare_launch(stream, target);
    if (!prepared) {
        const auto recorded = cudaGetLastError();
        if (recorded == cudaSuccess || recorded == cudaErrorNotSupported) return std::nullopt;
        return recorded;
    }
    if (*prepared != cudaSuccess) return take_launch_status(*prepared);
    constexpr int shared_bytes = copy_stages * stage_bytes<columns>;
    int shared_limit = 0;
    const void* kernel = reinterpret_cast<const void*>(project_gdn_gate_partials<Channels>);
    const auto configured = configure(kernel, target.device, shared_limit);
    if (configured != cudaSuccess) return take_launch_status(configured);
    if (shared_limit < shared_bytes) return std::nullopt;
    CUtensorMap maps[3];
    auto encoded = encode_gate_map(target, maps[0], weight_b, Channels, tile_rows / 2);
    if (encoded == CUDA_SUCCESS) {
        encoded = encode_gate_map(target, maps[1], weight_a, Channels, tile_rows / 2);
    }
    if (encoded == CUDA_SUCCESS) {
        encoded = encode_gate_map(target, maps[2], input, rows, columns);
    }
    if (encoded != CUDA_SUCCESS) return descriptor_error(encoded);
    auto* partials = static_cast<float*>(scratch);
    void* project_args[] = {&maps[0], &maps[1], &maps[2], &rows, &partials};
    const auto status = take_launch_status(cudaLaunchKernel(project_gdn_gate_partials<Channels>,
        dim3((rows + columns - 1) / columns, splits), threads, project_args, shared_bytes, stream));
    if (status != cudaSuccess) return status;
    void* round_args[] = {&partials, &elements, &b, &a};
    return take_launch_status(cudaLaunchKernel(round_gdn_gates<Channels>,
        (elements + reduction_threads - 1) / reduction_threads, reduction_threads,
        round_args, 0, stream));
}
}

std::optional<cudaError_t> try_prefill_bf16_gdn_gates_sm100(
    const Bf16* weight_b, const Bf16* weight_a, const Bf16* input, int rows, Bf16* b, Bf16* a,
    void* scratch, std::size_t scratch_bytes, cudaStream_t stream) {
    return prefill_bf16_gdn_gates_sm100<text_gdn_value_heads>(
        weight_b, weight_a, input, rows, b, a, scratch, scratch_bytes, stream);
}

std::optional<cudaError_t> try_prefill_bf16_gdn_gates_sm100_local(
    const Bf16* weight_b, const Bf16* weight_a, const Bf16* input, int rows, Bf16* b, Bf16* a,
    void* scratch, std::size_t scratch_bytes, cudaStream_t stream) {
    return prefill_bf16_gdn_gates_sm100<text_gdn_value_heads / 2>(
        weight_b, weight_a, input, rows, b, a, scratch, scratch_bytes, stream);
}

}
