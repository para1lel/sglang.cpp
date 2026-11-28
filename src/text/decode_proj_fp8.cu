#include "sglang/text/decode_proj.h"
#include "sglang/text/decode_proj_sm100.cuh"
#include "sglang/text/quantization/quantize_warp.cuh"
#include "sglang/text/target_fp8.h"

#include <algorithm>
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
constexpr int local_intermediate_width = intermediate_width / 2;
constexpr int single_tile_rows = text_target_fp8_single_tile_rows;
constexpr int m10_rows = text_target_fp8_m10_rows;
constexpr int m40_rows = text_target_fp8_m40_rows;
constexpr int m40_columns = 48;
constexpr int m10_columns = 16;
constexpr int gate_cluster = 2;
constexpr int m80_rows = text_target_fp8_m80_rows;
constexpr int m80_half = m80_rows / 2;
constexpr int m80_consumers = 2 * consumer_warps;
constexpr int m80_threads = (m80_consumers + 2) * warp_size;

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

struct Fp8M80Job : Fp8RowJob<m80_rows> {};

struct Fp8GateUpJob : Fp8Job {
    const float* scales[m10_rows];
    TextFp8Activation quantized[m10_rows];
};

template <int Columns, int Workers = tile_rows, typename Job>
__device__ void load_scales(const Job& job, float* scales,
                            int act_row, int first, int count, int active) {
    constexpr int fragment = Columns == 8 ? 8 : 16;
    const int padded = active == 1 ? 1 : (active + fragment - 1) / fragment * fragment;
    for (int index = threadIdx.x; index < count * padded; index += Workers) {
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
    if constexpr (Workers == tile_rows) {
        consumer_sync();
    } else {
        asm volatile("bar.sync 1, %0;" :: "n"(Workers) : "memory");
    }
}

template <bool Packed, int Fragment>
__device__ __forceinline__ void promote(const unsigned (&partial)[Fragment], const float* scales,
                                        float scale, float* result) {
    if constexpr (Packed) {
        #pragma unroll
        for (int col = 0; col < Fragment; col += 2) {
            asm("{ .reg .b64 values, factors, sum;\n"
                "mov.b64 values, {%2,%3};\n"
                "mov.b64 factors, {%4,%5};\n"
                "mul.rn.f32x2 values, values, factors;\n"
                "mov.b64 factors, {%6,%6};\n"
                "mov.b64 sum, {%0,%1};\n"
                "fma.rn.f32x2 sum, values, factors, sum;\n"
                "mov.b64 {%0,%1}, sum; }"
                : "+f"(result[col]), "+f"(result[col + 1])
                : "r"(partial[col]), "r"(partial[col + 1]), "f"(scales[col]), "f"(scales[col + 1]),
                  "f"(scale));
        }
    } else {
        #pragma unroll
        for (int col = 0; col < Fragment; ++col) {
            const float scaled = __fmul_rn(__uint_as_float(partial[col]), scales[col]);
            result[col] = __fmaf_rn(scaled, scale, result[col]);
        }
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

__device__ __forceinline__ void load_tmem10(unsigned source, unsigned (&out)[m10_rows]) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                 : "=r"(out[0]), "=r"(out[1]), "=r"(out[2]), "=r"(out[3]),
                   "=r"(out[4]), "=r"(out[5]), "=r"(out[6]), "=r"(out[7])
                 : "r"(source) : "memory");
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x2.b32 {%0,%1}, [%2];"
                 "\ntcgen05.wait::ld.sync.aligned;"
                 : "=r"(out[8]), "=r"(out[9]) : "r"(source + 8) : "memory");
}

template <int Fragment>
__device__ __forceinline__ void load_fragment(unsigned source, unsigned (&partial)[Fragment]) {
    if constexpr (Fragment == m10_rows) {
        load_tmem10(source, partial);
    } else if constexpr (Fragment == 8) {
        load_tmem8(source, partial);
    } else {
        load_tmem16(source, partial);
    }
}

template <int Columns, bool Halves, int ResultRows, int Slots>
__device__ __forceinline__ void consume(DecodePipeline<Slots>& pipe, int count, const float* scales,
                                        const float* weight_scales, int active,
                                        float (&result)[ResultRows]) {
    constexpr bool packed_m40 = Columns == m40_columns && ResultRows == m40_rows;
    constexpr bool compact = packed_m40 || Columns == 80 || Columns == 160;
    constexpr bool packed_m10 = ResultRows == m10_rows;
    const int warp = threadIdx.x / warp_size;
    const int half = Halves ? (warp / consumer_warps) * ResultRows : 0;
    const unsigned first_lane = (Halves ? warp % consumer_warps : warp) * warp_size;
    const unsigned lane_base = pipe.tmem + (first_lane << 16) + half;
    if constexpr (!compact && !packed_m10) {
        if (active == 1) {
            consume_single<Columns>(pipe, scales, weight_scales, count, lane_base, result[0]);
            return;
        }
    }
    constexpr int fragment = packed_m10 ? m10_rows : Columns == 8 ? 8 : 16;
    constexpr bool tail = packed_m40 || Halves;
    constexpr int last_full_fragment = tail ? ResultRows - 8 : ResultRows;
    for (int index = 0; index < count; ++index) {
        const int slot = index % Slots;
        const float scale = weight_scales[index];
        const auto* row_scales = scales + (static_cast<std::ptrdiff_t>(index) * Columns) + half;
        wait_barrier(&pipe.accum_ready[slot], (index / Slots) & 1);
        fence_after_sync();
#pragma unroll
        for (int block = 0; block < last_full_fragment; block += fragment) {
            if (compact || packed_m10 || block < active) {
                unsigned partial[fragment];
                load_fragment(lane_base + (slot * Columns) + block, partial);
                promote<compact || packed_m10>(partial, row_scales + block, scale, result + block);
            }
        }
        if constexpr (tail) {
            unsigned partial[8];
            load_tmem8(lane_base + slot * Columns + last_full_fragment, partial);
            promote<true>(partial, row_scales + last_full_fragment, scale,
                          result + last_full_fragment);
        }
        release_slot(&pipe.accum_empty[slot]);
    }
}

__device__ __forceinline__ float4 load_cluster4(const float* local, unsigned rank) {
    if (rank == blockIdx.x) return *reinterpret_cast<const float4*>(local);
    unsigned remote;
    float4 value;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                 : "=r"(remote) : "r"(shared_address(local)), "r"(rank));
    asm volatile("ld.shared::cluster.v4.f32 {%0,%1,%2,%3}, [%4];"
                 : "=f"(value.x), "=f"(value.y), "=f"(value.z), "=f"(value.w)
                 : "r"(remote) : "memory");
    return value;
}

template <int Columns, typename Output, int Workers = threads>
__device__ __noinline__ void reduce_compact(const float* partials, int splits, int row_base,
                                           int out_width, const Output* output) {
    if (out_width % 4 != 0) {
        reduce_store(partials, splits, Columns, row_base, out_width, output->output);
        return;
    }
    constexpr int vectors = Columns * tile_rows / 4;
    const int first = blockIdx.x * vectors / splits;
    const int last = (blockIdx.x + 1) * vectors / splits;
    for (int vector = first + threadIdx.x; vector < last; vector += Workers) {
        const int index = vector * 4;
        const int row = row_base + index % tile_rows;
        if (row >= out_width) continue;
        const auto* values = partials + index;
        auto sum = load_cluster4(values, 0);
        for (int source = 1; source < splits; ++source) {
            const auto other = load_cluster4(values, source);
            sum.x += other.x;
            sum.y += other.y;
            sum.z += other.z;
            sum.w += other.w;
        }
        auto* out = (*output)(index / tile_rows) + row;
        uint2 packed;
        asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(packed.x) : "f"(sum.x), "f"(sum.y));
        asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(packed.y) : "f"(sum.z), "f"(sum.w));
        if (reinterpret_cast<std::uintptr_t>(out) % 8 == 0) {
            *reinterpret_cast<uint2*>(out) = packed;
        } else {
            out[0] = __nv_bfloat16_raw{static_cast<unsigned short>(packed.x)};
            out[1] = __nv_bfloat16_raw{static_cast<unsigned short>(packed.x >> 16)};
            out[2] = __nv_bfloat16_raw{static_cast<unsigned short>(packed.y)};
            out[3] = __nv_bfloat16_raw{static_cast<unsigned short>(packed.y >> 16)};
        }
    }
}

__device__ void store_m80_halves(const float (&result)[m80_half], float* partials, int splits,
                                 int row_base, int out_width,
                                 const Fp8RowPointers<m80_rows>& output) {
    const int thread = threadIdx.x;
    const int row = thread % tile_rows;
    const int half = (thread / tile_rows) * m80_half;
    if (thread < m80_consumers * warp_size) {
        if (splits == 1) {
            if (row_base + row < out_width) {
                #pragma unroll
                for (int col = 0; col < m80_half; ++col) {
                    output(half + col)[row_base + row] = __float2bfloat16_rn(result[col]);
                }
            }
        } else {
            #pragma unroll
            for (int col = 0; col < m80_half; ++col) {
                partials[(half + col) * tile_rows + row] = result[col];
            }
        }
    }
    if (splits > 1) {
        cluster_sync();
        reduce_compact<m80_rows, Fp8RowPointers<m80_rows>, m80_threads>(
            partials, splits, row_base, out_width, &output);
        cluster_sync();
    }
}

template <int Columns, typename Output>
__device__ __forceinline__ void store_compact(const float (&result)[Columns], float* partials,
                                               int splits, int row_base, int out_width,
                                               const Output& output) {
    if (splits == 1) {
        store_result(result, partials, 1, Columns, row_base, out_width, output.output);
        return;
    }
    if (threadIdx.x < tile_rows) {
        #pragma unroll
        for (int col = 0; col < Columns; ++col) {
            partials[col * tile_rows + threadIdx.x] = result[col];
        }
    }
    cluster_sync();
    reduce_compact<Columns>(partials, splits, row_base, out_width, &output);
    cluster_sync();
}

__device__ __forceinline__ uint2 load_cluster_bf16x4(const __nv_bfloat16* local, unsigned rank) {
    if (rank == blockIdx.x) return *reinterpret_cast<const uint2*>(local);
    unsigned remote;
    uint2 value;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                 : "=r"(remote) : "r"(shared_address(local)), "r"(rank));
    asm volatile("ld.shared::cluster.v2.u32 {%0,%1}, [%2];"
                 : "=r"(value.x), "=r"(value.y) : "r"(remote) : "memory");
    return value;
}

__device__ __forceinline__ __nv_bfloat162 swiglu_pair(unsigned gate_bits, unsigned up_bits) {
    const __nv_bfloat16 gate_lo = __nv_bfloat16_raw{static_cast<unsigned short>(gate_bits)};
    const __nv_bfloat16 up_lo = __nv_bfloat16_raw{static_cast<unsigned short>(up_bits)};
    const __nv_bfloat16 gate_hi = __nv_bfloat16_raw{static_cast<unsigned short>(gate_bits >> 16)};
    const __nv_bfloat16 up_hi = __nv_bfloat16_raw{static_cast<unsigned short>(up_bits >> 16)};
    return {text_kernel::swiglu(gate_lo, up_lo), text_kernel::swiglu(gate_hi, up_hi)};
}

__device__ void store_gate_up(const float (&result)[m10_columns], __nv_bfloat16* tile,
                              int target, int row_block, int warp, const Fp8GateUpJob& job) {
    const int row_base = row_block * tile_rows;
    if (threadIdx.x < tile_rows) {
        #pragma unroll
        for (int row = 0; row < m10_rows; ++row) {
            tile[row * tile_rows + threadIdx.x] = __float2bfloat16_rn(result[row]);
        }
    }
    cluster_sync();
    constexpr int rows_per_cta = m10_rows / gate_cluster;
    if (warp < rows_per_cta) {
        const int row = target * rows_per_cta + warp;
        const int lane = threadIdx.x % warp_size;
        const auto* values = tile + row * tile_rows + lane * 4;
        const auto gate = load_cluster_bf16x4(values, 0);
        const auto up = load_cluster_bf16x4(values, 1);
        text_kernel::quantize_warp_pairs(swiglu_pair(gate.x, up.x), swiglu_pair(gate.y, up.y),
                                         row_block, row_base + lane * 4, lane, job.quantized[row]);
    }
    cluster_sync();
}

template <int Slots>
__device__ void init_m80_pipeline(DecodePipeline<Slots>& pipe) {
    for (int slot = 0; slot < Slots; ++slot) {
        init_barrier(&pipe.copy_ready[slot], 1);
        init_barrier(&pipe.accum_ready[slot], 1);
        init_barrier(&pipe.accum_empty[slot], m80_consumers);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}

template <int Columns, int Slots = stages, typename Job = Fp8Job>
__global__ __launch_bounds__((std::is_same_v<Job, Fp8M80Job> ? m80_threads : threads), 1)
void fp8_decode(const __grid_constant__ Job job) {
    constexpr unsigned instruction = fp8_instruction<Columns>;
    alignas(1024) extern __shared__ unsigned char tiles[];
    __shared__ DecodePipeline<Slots> pipe;
    constexpr bool gate_up = std::is_same_v<Job, Fp8GateUpJob>;
    constexpr bool packed_m10 = std::is_same_v<Job, Fp8RowJob<m10_rows>>;
    constexpr bool packed_m40 = Columns == m40_columns && std::is_same_v<Job, Fp8RowJob<m40_rows>>;
    const int splits = gate_up ? 1 : gridDim.x;
    const int split = gate_up ? 0 : blockIdx.x;
    const int first = split * job.groups / splits;
    const int count = (split + 1) * job.groups / splits - first;
    const int act_row = blockIdx.z * Columns;
    constexpr bool compact = Columns == 80 || Columns == 160;
    const int active = gate_up || packed_m10 ? m10_rows
        : packed_m40 ? m40_rows : compact ? Columns : min(Columns, job.rows - act_row);
    const int warp = threadIdx.x / warp_size;
    constexpr bool halves = std::is_same_v<Job, Fp8M80Job>;
    constexpr int mma = halves ? m80_consumers : mma_warp;
    constexpr int copy = mma + 1;
    constexpr int scale_workers = mma * warp_size;
    int target = blockIdx.x;
    int row_block = blockIdx.y;
    if constexpr (!gate_up) {
        target = 0;
        for (int index = 1; index < job.targets; ++index) {
            if (blockIdx.y >= unsigned(job.first_block[index])) {
                target = index;
            }
        }
        row_block -= job.first_block[target];
    }
    const int row_base = row_block * tile_rows;
    auto* scales = reinterpret_cast<float*>(tiles + Slots * stage_bytes<Columns>);
    auto* weight_scales = scales + count * Columns;
    if (threadIdx.x == 0) {
        if constexpr (halves) {
            init_m80_pipeline(pipe);
        } else {
            init_pipeline<true>(pipe);
        }
    }
    if (warp == mma) {
        allocate_tmem(&pipe.tmem, tmem_columns<Columns, Slots>);
    }
    __syncthreads();
    const unsigned shared = shared_address(tiles);
    float result[halves ? m80_half : packed_m10 ? m10_rows : packed_m40 ? m40_rows : Columns] = {};
    if (warp < mma) {
        const auto* source = job.weight_scales[target] + std::size_t(row_block) * job.groups;
        for (int index = threadIdx.x; index < count; index += scale_workers) {
            weight_scales[index] = __bfloat162float(source[first + index]);
        }
        wait_dependencies();
        load_scales<Columns, scale_workers>(job, scales, act_row, first, count, active);
        consume<Columns, halves>(pipe, count, scales, weight_scales, active, result);
    } else if (warp == mma && threadIdx.x % warp_size == 0) {
        multiply<instruction, true, Columns>(pipe, shared, count);
    } else if (warp == copy && threadIdx.x % warp_size == 0) {
        produce<group_size, Columns, true>(pipe, shared, &job.weight[target], &job.activation,
                                            row_base, act_row, first, count);
    }
    wait_dependencies();
    __syncthreads();
    if (warp == mma) {
        free_tmem(&pipe.tmem, tmem_columns<Columns, Slots>);
    }
    if constexpr (halves) {
        store_m80_halves(result, reinterpret_cast<float*>(tiles), splits, row_base,
                         job.out_width[target], job.lanes[target]);
    } else if constexpr (gate_up) {
        store_gate_up(result, reinterpret_cast<__nv_bfloat16*>(tiles),
                      target, row_block, warp, job);
    } else if constexpr (compact || packed_m10 || packed_m40) {
        store_compact(result, reinterpret_cast<float*>(tiles), splits, row_base,
                      job.out_width[target], job.lanes[target]);
    } else if constexpr (!std::is_same_v<Job, Fp8Job>) {
        store_result(result, reinterpret_cast<float*>(tiles), splits, active, row_base,
                     job.out_width[target], job.lanes[target].output);
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

bool tp2_gate_up_shape(const TextFp8DecodeTarget* targets, int count, int in_width) {
    return count == 2 && in_width == hidden_width &&
           targets[0].out_width == local_intermediate_width &&
           targets[1].out_width == local_intermediate_width;
}

bool use_mlp_pipeline(const TextFp8DecodeTarget* targets, int count, int in_width, int rows) {
    return rows >= 16 && rows <= 32 && in_width == hidden_width && count == 2 &&
           targets[0].out_width == targets[1].out_width &&
           (targets[0].out_width == local_intermediate_width ||
            targets[0].out_width == intermediate_width);
}

bool tp2_m80_shape(const TextFp8DecodeTarget* targets, int count, int in_width) {
    constexpr int attention_width = 3072;
    constexpr int gdn_qkv_width = 5120;
    constexpr int gqa_query_gate_width = 6144;
    constexpr int gqa_kv_width = 512;
    if (count == 1) {
        return targets[0].out_width == hidden_width &&
               (in_width == attention_width || in_width == local_intermediate_width);
    }
    if (in_width != hidden_width) return false;
    if (count == 2) {
        return tp2_gate_up_shape(targets, count, in_width) ||
               (targets[0].out_width == gdn_qkv_width && targets[1].out_width == attention_width);
    }
    return count == 3 && targets[0].out_width == gqa_query_gate_width &&
           targets[1].out_width == gqa_kv_width && targets[2].out_width == gqa_kv_width;
}

cudaError_t plan_fp8_splits(const LaunchTarget& target, int blocks, int groups, SplitPlan& plan) {
    return plan_splits(reinterpret_cast<const void*>(fp8_decode<tile_cols>), target, blocks, groups,
                       stages * stage_bytes<tile_cols>, scale_bytes<tile_cols>, plan);
}

template <int Columns, int Slots, int Capacity, typename Job = Fp8RowJob<Capacity>>
std::optional<cudaError_t>
launch_rows(const LaunchTarget& target, const TextFp8DecodeTarget* projections, int target_count,
            TextFp8ActivationView input, int in_width, const TextFp8Projection* rows, int count,
            cudaStream_t stream) {
    static_assert(Capacity <= Columns);
    if (count > Capacity || target_count > max_targets)
        return std::nullopt;
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
    status = configure(kernel, target.device, limit);
    if (status != cudaSuccess) {
        return status;
    }
    const int stages_and_scales = Slots * stage_bytes<Columns> +
                                  ((job.groups - 1) / plan.splits + 1) * scale_bytes<Columns>;
    const int partials = plan.splits > 1 ? Columns * tile_rows * int(sizeof(float)) : 0;
    const int shared = std::max(stages_and_scales, partials);
    if (shared > limit) {
        return std::nullopt;
    }
    const dim3 grid(plan.splits, blocks, 1);
    void* args[] = {&job};
    if constexpr (std::is_same_v<Job, Fp8M80Job>) {
        return launch_split(kernel, grid, shared, args, stream, grid.x, m80_threads);
    }
    return launch_split(kernel, grid, shared, args, stream, grid.x);
}

std::optional<cudaError_t> compact_status(std::optional<cudaError_t> status) {
    const auto recorded = cudaGetLastError();
    if ((!status || *status == cudaSuccess) && recorded != cudaSuccess) return recorded;
    return status;
}

}

std::optional<cudaError_t> try_text_fp8_decode_sm100_rows(
    const TextFp8DecodeTarget* projections, int target_count, const std::uint8_t* packed_values,
    int in_width, const TextFp8Projection* rows, int count, cudaStream_t stream) {
    const bool compact = count == text_target_fp8_m80_rows || count == text_target_fp8_m160_rows;
    if (compact) {
        const auto pending = cudaGetLastError();
        if (pending != cudaSuccess) return pending;
    }
    const TextFp8ActivationView input{packed_values, nullptr};
    if (count < 2 || (count > text_target_fp8_max_rows && !target_fp8_wide_rows(count)) ||
        !supported_targets(projections, target_count, in_width, input)) {
        return std::nullopt;
    }
    LaunchTarget target{};
    if (auto prepared = prepare_launch(stream, target); !prepared || *prepared != cudaSuccess) {
        if (!compact) return prepared;
        const auto status = compact_status(prepared);
        return !prepared && status == cudaErrorNotSupported ? std::nullopt : status;
    }
    if (count == text_target_fp8_m80_rows) {
        if (tp2_m80_shape(projections, target_count, in_width)) {
            if (auto packed = compact_status(launch_rows<m80_rows, 4, m80_rows, Fp8M80Job>(
                    target, projections, target_count, input, in_width, rows, count, stream))) {
                return packed;
            }
        }
        return compact_status(launch_rows<text_target_fp8_m80_rows, 4, text_target_fp8_m80_rows>(
            target, projections, target_count, input, in_width, rows, count, stream));
    }
    if (count == text_target_fp8_m160_rows) {
        return compact_status(launch_rows<text_target_fp8_m160_rows, 3, text_target_fp8_m160_rows>(
            target, projections, target_count, input, in_width, rows, count, stream));
    }
    if (count == text_target_fp8_m40_rows) {
        if (in_width == hidden_width && target_count == 2 &&
            projections[0].out_width == intermediate_width &&
            projections[1].out_width == intermediate_width) {
            return launch_rows<64, mlp_slots, text_target_fp8_m40_rows>(
                target, projections, target_count, input, in_width, rows, count, stream);
        }
        if (!tp2_gate_up_shape(projections, target_count, in_width)) {
            if (const auto pending = cudaGetLastError(); pending != cudaSuccess) return pending;
            if (auto packed = compact_status(launch_rows<m40_columns, 4, m40_rows>(
                    target, projections, target_count, input, in_width, rows, count, stream))) {
                return packed;
            }
        }
        return launch_rows<64, stages, text_target_fp8_m40_rows>(
            target, projections, target_count, input, in_width, rows, count, stream);
    }
    if (count <= text_fp8_max_batch) {
        return launch_rows<single_tile_rows, stages, text_fp8_max_batch>(
            target, projections, target_count, input, in_width, rows, count, stream);
    }
    if (count <= single_tile_rows) {
        return launch_rows<single_tile_rows, stages, single_tile_rows>(
            target, projections, target_count, input, in_width, rows, count, stream);
    }
    if (count == m10_rows && !tp2_gate_up_shape(projections, target_count, in_width) &&
        std::all_of(projections, projections + target_count, [](const auto& projection) {
            return projection.out_width > 0 && projection.out_width % group_size == 0;
        })) {
        if (auto packed = launch_rows<m10_columns, stages, m10_rows>(
                target, projections, target_count, input, in_width, rows, count, stream)) {
            return packed;
        }
    }
    if (count <= 16) {
        return launch_rows<16, stages, 16>(target, projections, target_count, input, in_width, rows,
                                           count, stream);
    }
    if (use_mlp_pipeline(projections, target_count, in_width, count)) {
        return launch_rows<32, mlp_slots, text_target_fp8_max_rows>(
            target, projections, target_count, input, in_width, rows, count, stream);
    }
    return launch_rows<32, stages, text_target_fp8_max_rows>(target, projections, target_count,
                                                             input, in_width, rows, count, stream);
}

std::optional<cudaError_t> try_target_fp8_gate_up(
    const TextFp8DecodeTarget (&targets)[2], int in_width,
    const TextFp8Activation (&inputs)[m10_rows],
    const TextFp8Activation (&quantized)[m10_rows], cudaStream_t stream) {
    if (!tp2_gate_up_shape(targets, 2, in_width)) return std::nullopt;
    const TextFp8ActivationView input{inputs[0].values, nullptr};
    if (!input.values || !supported_targets(targets, 2, in_width, input)) return std::nullopt;
    for (const auto& target : targets) {
        if (!target.weight || !target.weight_scales) return std::nullopt;
    }
    Fp8GateUpJob job{};
    for (int row = 0; row < m10_rows; ++row) {
        const auto& activation = inputs[row];
        if (activation.values != input.values + row * in_width || !activation.scales ||
            !quantized[row].values || !quantized[row].scales ||
            !aligned16(quantized[row].values)) return std::nullopt;
        job.scales[row] = activation.scales;
        job.quantized[row] = quantized[row];
    }
    LaunchTarget target{};
    if (auto ready = prepare_launch(stream, target); !ready || *ready != cudaSuccess) return ready;
    constexpr int tiles = local_intermediate_width / tile_rows;
    constexpr int groups = hidden_width / group_size;
    SplitPlan plan{};
    auto status = plan_fp8_splits(target, gate_cluster * tiles, groups, plan);
    if (status != cudaSuccess) return status;
    if (plan.splits != 1) return std::nullopt;
    const auto* kernel = reinterpret_cast<const void*>(
        fp8_decode<m10_columns, stages, Fp8GateUpJob>);
    constexpr int shared = stages * stage_bytes<m10_columns> + groups * scale_bytes<m10_columns>;
    if (const auto support = prepare_gate_cluster(kernel, target, tiles, shared);
        !support || *support != cudaSuccess) return support;
    int blocks = 0;
    if (auto encoded = encode_job(target, targets, 2, input, in_width, m10_rows, m10_columns,
                                  job, blocks); !encoded || *encoded != cudaSuccess) return encoded;
    void* args[] = {&job};
    return launch_split(kernel, dim3(gate_cluster, tiles, 1), shared, args, stream, gate_cluster);
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
    int columns = tile_cols;
    if (rows <= single_tile_rows) {
        columns = single_tile_rows;
    } else if (rows <= 16) {
        columns = 16;
    } else if (rows <= 32) {
        columns = 32;
    }
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
    const bool short_pipeline = use_mlp_pipeline(targets, count, in_width, rows);
    if (short_pipeline) {
        kernel = columns == 16 ? reinterpret_cast<const void*>(fp8_decode<16, mlp_slots>)
                               : reinterpret_cast<const void*>(fp8_decode<32, mlp_slots>);
    } else if (columns == single_tile_rows) {
        kernel = reinterpret_cast<const void*>(fp8_decode<single_tile_rows>);
    } else if (columns != tile_cols) {
        kernel = columns == 16 ? reinterpret_cast<const void*>(fp8_decode<16>)
                               : reinterpret_cast<const void*>(fp8_decode<32>);
    }
    if (kernel != full_kernel) {
        int limit = 0;
        status = configure(kernel, target.device, limit);
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
