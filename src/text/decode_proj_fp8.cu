#include "sglang/text/decode_proj.h"
#include "sglang/text/decode_proj_sm100.cuh"
#include "sglang/text/target_fp8.h"

#include <cuda_runtime.h>
#include <type_traits>

namespace sglang {
namespace {

using namespace decode_proj;

constexpr int group_size = text_fp8_group_size;
constexpr int max_targets = text_decode_max_targets;
constexpr int mlp_slots = 4;
constexpr int hidden_width = 5120;
constexpr int intermediate_width = 17408;
template <int Columns>
constexpr int scale_bytes = (Columns + 1) * int(sizeof(float));

struct Fp8Job {
    CUtensorMap weight[max_targets];
    CUtensorMap activation;
    const float* act_scales;
    const __nv_bfloat16* weight_scales[max_targets];
    __nv_bfloat16* output[max_targets];
    int out_width[max_targets];
    int first_block[max_targets];
    int targets;
    int groups;
    int rows;
};

template <int Capacity>
struct Fp8RowPointers {
    __nv_bfloat16* output[Capacity];

    __device__ __forceinline__ __nv_bfloat16* operator()(int col) const {
        return output[col];
    }
};

template <int Capacity>
struct Fp8RowJob : Fp8Job {
    Fp8RowPointers<Capacity> lanes[max_targets];
    const float* scales[Capacity];
};

template <int Columns, typename Job>
__device__ void load_scales(const Job& job, float* scales,
                            int act_row, int first, int count, int active) {
    constexpr int fragment = Columns == 8 ? 8 : 16;
    const int padded = active == 1 ? 1 : (active + fragment - 1) / fragment * fragment;
    for (int index = threadIdx.x; index < count * padded; index += tile_rows) {
        const int col = index / count;
        const int group = index % count;
        float value = 0.0f;
        if (col < active) {
            if constexpr (!std::is_same_v<Job, Fp8Job>) {
                value = job.scales[col][first + group];
            } else {
                value = job.act_scales[std::size_t(act_row + col) * job.groups + first + group];
            }
        }
        scales[group * Columns + col] = value;
    }
    consumer_sync();
}

template <int Fragment>
__device__ void promote(const unsigned (&partial)[Fragment], const float* scales, float scale,
                        float* result) {
    #pragma unroll
    for (int col = 0; col < Fragment; ++col) {
        const float scaled = __fmul_rn(__uint_as_float(partial[col]), scales[col]);
        result[col] = __fmaf_rn(scaled, scale, result[col]);
    }
}

template <int Columns, int Slots>
__device__ void consume_single(DecodePipeline<Slots>& pipe, const float* scales,
                               const float* weight_scales,
                               int count, unsigned lane_base, float& result) {
    for (int index = 0; index < count; ++index) {
        const int slot = index % Slots;
        const float scale = weight_scales[index];
        wait_barrier(&pipe.accum_ready[slot], (index / Slots) & 1);
        fence_after_sync();
        const float partial = load_tmem1(lane_base + slot * Columns);
        const float scaled = __fmul_rn(partial, scales[index * Columns]);
        result = __fmaf_rn(scaled, scale, result);
        release_slot(&pipe.accum_empty[slot]);
    }
}

template <int Columns, int Slots>
__device__ void consume(DecodePipeline<Slots>& pipe, const float* scales,
                        const float* weight_scales,
                        int count, int active, float (&result)[Columns]) {
    const int warp = threadIdx.x / warp_size;
    const unsigned lane_base = pipe.tmem + (static_cast<unsigned>(warp * warp_size) << 16);
    if (active == 1) {
        consume_single<Columns>(pipe, scales, weight_scales, count, lane_base, result[0]);
        return;
    }
    constexpr int fragment = Columns == 8 ? 8 : 16;
    for (int index = 0; index < count; ++index) {
        const int slot = index % Slots;
        const float scale = weight_scales[index];
        wait_barrier(&pipe.accum_ready[slot], (index / Slots) & 1);
        fence_after_sync();
        #pragma unroll
        for (int block = 0; block < Columns; block += fragment) {
            if (block < active) {
                unsigned partial[fragment];
                if constexpr (fragment == 8) {
                    load_tmem8(lane_base + slot * Columns + block, partial);
                } else {
                    load_tmem16(lane_base + slot * Columns + block, partial);
                }
                promote(partial, scales + index * Columns + block, scale, result + block);
            }
        }
        release_slot(&pipe.accum_empty[slot]);
    }
}

template <int Columns, int Slots = stages, typename Job = Fp8Job>
__global__ __launch_bounds__(threads, 1)
void fp8_decode(const __grid_constant__ Job job) {
    constexpr unsigned instruction =
        ((tile_rows / 16) << 24) | ((Columns / 8) << 17) | (1u << 4);
    alignas(1024) extern __shared__ unsigned char tiles[];
    __shared__ DecodePipeline<Slots> pipe;
    const int splits = gridDim.x;
    const int first = blockIdx.x * job.groups / splits;
    const int count = (blockIdx.x + 1) * job.groups / splits - first;
    const int act_row = blockIdx.z * Columns;
    const int active = min(Columns, job.rows - act_row);
    const int warp = threadIdx.x / warp_size;
    int target = 0;
    for (int index = 1; index < job.targets; ++index) {
        if (blockIdx.y >= unsigned(job.first_block[index])) {
            target = index;
        }
    }
    const int row_block = blockIdx.y - job.first_block[target];
    const int row_base = row_block * tile_rows;
    auto* scales = reinterpret_cast<float*>(tiles + Slots * stage_bytes<Columns>);
    auto* weight_scales = scales + count * Columns;
    if (threadIdx.x == 0) {
        init_pipeline<true>(pipe);
    }
    if (warp == mma_warp) {
        allocate_tmem<Columns>(pipe);
    }
    __syncthreads();
    const unsigned shared = shared_address(tiles);
    float result[Columns] = {};
    if (warp < consumer_warps) {
        const auto* source = job.weight_scales[target] + std::size_t(row_block) * job.groups;
        for (int index = threadIdx.x; index < count; index += tile_rows) {
            weight_scales[index] = __bfloat162float(source[first + index]);
        }
        wait_dependencies();
        load_scales<Columns>(job, scales, act_row, first, count, active);
        consume(pipe, scales, weight_scales, count, active, result);
    } else if (warp == mma_warp && threadIdx.x % warp_size == 0) {
        multiply<instruction, true, Columns>(pipe, shared, count);
    } else if (warp == copy_warp && threadIdx.x % warp_size == 0) {
        produce<group_size, Columns, true>(pipe, shared, &job.weight[target], &job.activation,
                                            row_base, act_row, first, count);
    }
    wait_dependencies();
    __syncthreads();
    if (warp == mma_warp) {
        free_tmem<Columns>(pipe);
    }
    if constexpr (!std::is_same_v<Job, Fp8Job>) {
        store_result(result, reinterpret_cast<float*>(tiles), splits, active, row_base,
                     job.out_width[target], job.lanes[target]);
    } else {
        store_result(result, reinterpret_cast<float*>(tiles), splits, active, row_base,
                     job.out_width[target], act_row, job.output[target]);
    }
}

std::optional<cudaError_t> encode_job(const LaunchTarget& target,
                                      const TextFp8DecodeTarget* targets, int count,
                                      TextFp8ActivationView input, int in_width, int rows,
                                      int columns,
                                      Fp8Job& job, int& blocks) {
    job.targets = count;
    job.groups = in_width / group_size;
    job.rows = rows;
    job.act_scales = input.scales;
    auto encoded = encode_map(target, job.activation, input.values,
                              CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, rows, in_width, columns);
    for (int index = 0; index < count && encoded == CUDA_SUCCESS; ++index) {
        const auto& projection = targets[index];
        job.first_block[index] = blocks;
        encoded = encode_map(target, job.weight[index], projection.weight,
                             CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, projection.out_width,
                             in_width, tile_rows);
        job.weight_scales[index] = projection.weight_scales;
        job.output[index] = projection.output;
        job.out_width[index] = projection.out_width;
        blocks += text_fp8_group_count(projection.out_width);
    }
    if (encoded != CUDA_SUCCESS) {
        return encoded == CUDA_ERROR_NOT_SUPPORTED ? std::optional<cudaError_t>{}
                                                   : std::optional{cudaErrorInvalidValue};
    }
    return cudaSuccess;
}

bool supported_targets(const TextFp8DecodeTarget* targets, int count, int in_width,
                       TextFp8ActivationView input) {
    if (count < 1 || count > max_targets || in_width % group_size != 0 ||
        !aligned16(input.values)) {
        return false;
    }
    for (int index = 0; index < count; ++index) {
        if (!aligned16(targets[index].weight)) {
            return false;
        }
    }
    return true;
}

bool use_mlp_pipeline(const TextFp8DecodeTarget* targets, int count, int in_width, int rows) {
    return rows >= 16 && rows <= 32 && in_width == hidden_width && count == 2 &&
           targets[0].out_width == intermediate_width &&
           targets[1].out_width == intermediate_width;
}

cudaError_t plan_fp8_splits(const LaunchTarget& target, int blocks, int groups, SplitPlan& plan) {
    return plan_splits(reinterpret_cast<const void*>(fp8_decode<tile_cols>), 0, target,
                       blocks, groups, stages * stage_bytes<tile_cols>,
                       scale_bytes<tile_cols>, plan);
}

template <int Columns, int Slots, int Capacity>
std::optional<cudaError_t> launch_rows(
    const LaunchTarget& target, const TextFp8DecodeTarget* projections, int target_count,
    TextFp8ActivationView input, int in_width, const TextFp8Projection* rows, int count,
    int slot, cudaStream_t stream) {
    static_assert(Capacity <= Columns);
    using Job = Fp8RowJob<Capacity>;
    Job job{};
    int blocks = 0;
    if (auto encoded = encode_job(target, projections, target_count, input, in_width, count,
                                  Columns, job, blocks);
        !encoded || *encoded != cudaSuccess) {
        return encoded;
    }
    for (int row = 0; row < count; ++row) {
        job.scales[row] = rows[row].input.scales;
        for (int index = 0; index < target_count; ++index) {
            job.lanes[index].output[row] = rows[index * count + row].output;
        }
    }
    SplitPlan plan{};
    auto status = plan_fp8_splits(target, blocks, job.groups, plan);
    if (status != cudaSuccess) {
        return status;
    }
    if (plan.splits == 0) {
        return std::nullopt;
    }
    const auto* kernel = reinterpret_cast<const void*>(fp8_decode<Columns, Slots, Job>);
    int limit = 0;
    status = configure(kernel, slot, target.device, limit);
    if (status != cudaSuccess) {
        return status;
    }
    const int shared = Slots * stage_bytes<Columns> +
                       ((job.groups - 1) / plan.splits + 1) * scale_bytes<Columns>;
    if (shared > limit) {
        return std::nullopt;
    }
    const dim3 grid(plan.splits, blocks, 1);
    void* args[] = {&job};
    return launch_split(kernel, grid, shared, args, stream, grid.x);
}

}

std::optional<cudaError_t> try_text_fp8_decode_sm100_rows(
    const TextFp8DecodeTarget* projections, int target_count, const std::uint8_t* packed_values,
    int in_width, const TextFp8Projection* rows, int count, cudaStream_t stream) {
    const TextFp8ActivationView input{packed_values, nullptr};
    if (count < 2 || (count > text_target_fp8_max_rows && count != text_target_fp8_m40_rows) ||
        !supported_targets(projections, target_count, in_width, input)) {
        return std::nullopt;
    }
    LaunchTarget target{};
    if (auto prepared = prepare_launch(stream, target); !prepared || *prepared != cudaSuccess) {
        return prepared;
    }
    if (count == text_target_fp8_m40_rows) {
        if (in_width == hidden_width && target_count == 2 &&
            projections[0].out_width == intermediate_width &&
            projections[1].out_width == intermediate_width) {
            return launch_rows<64, mlp_slots, text_target_fp8_m40_rows>(
                target, projections, target_count, input, in_width, rows, count, 15, stream);
        }
        return launch_rows<64, stages, text_target_fp8_m40_rows>(
            target, projections, target_count, input, in_width, rows, count, 16, stream);
    }
    if (count <= text_fp8_max_batch) {
        return launch_rows<8, stages, text_fp8_max_batch>(
            target, projections, target_count, input, in_width, rows, count, 9, stream);
    }
    if (count <= 8) {
        return launch_rows<8, stages, 8>(
            target, projections, target_count, input, in_width, rows, count, 11, stream);
    }
    if (count <= 16) {
        return launch_rows<16, stages, 16>(
            target, projections, target_count, input, in_width, rows, count, 12, stream);
    }
    if (use_mlp_pipeline(projections, target_count, in_width, count)) {
        return launch_rows<32, mlp_slots, text_target_fp8_max_rows>(
            target, projections, target_count, input, in_width, rows, count, 13, stream);
    }
    return launch_rows<32, stages, text_target_fp8_max_rows>(
        target, projections, target_count, input, in_width, rows, count, 14, stream);
}

std::optional<cudaError_t> try_text_fp8_decode_sm100(const TextFp8DecodeTarget* targets,
                                                     int count, TextFp8ActivationView input,
                                                     int in_width, int rows,
                                                     cudaStream_t stream,
                                                     const Fp8DecodeInputProducer* producer) {
    if (!supported_targets(targets, count, in_width, input)) {
        return std::nullopt;
    }
    LaunchTarget target{};
    if (auto prepared = prepare_launch(stream, target); !prepared || *prepared != cudaSuccess) {
        return prepared;
    }
    Fp8Job job{};
    int blocks = 0;
    const int columns = rows <= 8 ? 8 : (rows <= 16 ? 16 : (rows <= 32 ? 32 : tile_cols));
    if (auto encoded = encode_job(target, targets, count, input, in_width, rows, columns,
                                   job, blocks);
        !encoded || *encoded != cudaSuccess) {
        return encoded;
    }
    const void* full_kernel = reinterpret_cast<const void*>(fp8_decode<tile_cols>);
    SplitPlan plan{};
    auto status = plan_fp8_splits(target, blocks, job.groups, plan);
    if (status != cudaSuccess) {
        return status;
    }
    if (plan.splits == 0) {
        return std::nullopt;
    }
    const void* kernel = full_kernel;
    int slot = 0;
    const bool short_pipeline = use_mlp_pipeline(targets, count, in_width, rows);
    if (short_pipeline) {
        kernel = columns == 16 ? reinterpret_cast<const void*>(fp8_decode<16, mlp_slots>)
                               : reinterpret_cast<const void*>(fp8_decode<32, mlp_slots>);
        slot = columns == 16 ? 7 : 8;
    } else if (columns == 8) {
        kernel = reinterpret_cast<const void*>(fp8_decode<8>);
        slot = 6;
    } else if (columns != tile_cols) {
        kernel = columns == 16 ? reinterpret_cast<const void*>(fp8_decode<16>)
                               : reinterpret_cast<const void*>(fp8_decode<32>);
        slot = columns == 16 ? 2 : 3;
    }
    if (slot != 0) {
        int limit = 0;
        status = configure(kernel, slot, target.device, limit);
        if (status != cudaSuccess) {
            return status;
        }
    }
    const int slots = short_pipeline ? mlp_slots : stages;
    const int shared = slots * (weight_bytes + columns * 128) +
                       ((job.groups - 1) / plan.splits + 1) * (columns + 1) * sizeof(float);
    const dim3 grid(plan.splits, blocks, (rows - 1) / columns + 1);
    void* args[] = {&job};
    if (producer) {
        status = producer->launch(producer->context, true);
        if (status != cudaSuccess) {
            return status;
        }
    }
    return launch_split(kernel, grid, shared, args, stream, grid.x);
}

}
