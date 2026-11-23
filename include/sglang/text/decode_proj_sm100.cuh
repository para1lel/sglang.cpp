#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda.h>
#include <cuda_bf16.h>

namespace sglang::decode_proj {

inline constexpr int warp_size = 32;
inline constexpr int tile_rows = 128;
inline constexpr int tile_cols = 64;
inline constexpr int stages = 8;
inline constexpr int consumer_warps = tile_rows / warp_size;
inline constexpr int mma_warp = consumer_warps;
inline constexpr int copy_warp = consumer_warps + 1;
inline constexpr int threads = (consumer_warps + 2) * warp_size;
inline constexpr int max_splits = 8;
inline constexpr int weight_bytes = tile_rows * 128;

template <int Columns>
inline constexpr int stage_bytes = weight_bytes + Columns * 128;

template <int Slots>
struct DecodePipeline {
    unsigned long long copy_ready[Slots];
    unsigned long long copy_empty[Slots];
    unsigned long long accum_ready[Slots];
    unsigned long long accum_empty[Slots];
    unsigned tmem;
};

using Pipeline = DecodePipeline<stages>;

__device__ __forceinline__ unsigned shared_address(const void* pointer) {
    return static_cast<unsigned>(__cvta_generic_to_shared(pointer));
}

__device__ __forceinline__ void init_barrier(unsigned long long* barrier, int arrivals) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
                 :: "r"(shared_address(barrier)), "r"(arrivals) : "memory");
}

__device__ __forceinline__ void wait_barrier(const unsigned long long* barrier, unsigned phase) {
    asm volatile("{ .reg .pred ready;\n"
                 "WAIT: mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 ready, [%0], %1;\n"
                 "@!ready bra.uni WAIT; }"
                 :: "r"(shared_address(barrier)), "r"(phase) : "memory");
}

__device__ __forceinline__ void arrive_barrier(unsigned long long* barrier) {
    asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];"
                 :: "r"(shared_address(barrier)) : "memory");
}

__device__ __forceinline__ void expect_bytes(unsigned long long* barrier, unsigned bytes) {
    asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
                 :: "r"(shared_address(barrier)), "r"(bytes) : "memory");
}

__device__ __forceinline__ void commit_mma(unsigned long long* barrier) {
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
                 :: "r"(shared_address(barrier)) : "memory");
}

__device__ __forceinline__ void copy_tile(const CUtensorMap* map, unsigned shared,
                                          unsigned long long* barrier, int col, int row) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes "
                 "[%0], [%1, {%2, %3}], [%4];"
                 :: "r"(shared), "l"(map), "r"(col), "r"(row), "r"(shared_address(barrier))
                 : "memory");
}

__device__ __forceinline__ std::uint64_t matrix_desc(unsigned start) {
    constexpr std::uint64_t stride = 8 * 128 / 16;
    constexpr std::uint64_t swizzle_128b = std::uint64_t{2} << 61;
    constexpr std::uint64_t descriptor_version = std::uint64_t{1} << 46;
    constexpr std::uint64_t leading_offset = std::uint64_t{1} << 16;
    return swizzle_128b | descriptor_version | (stride << 32) | leading_offset |
           ((start & 0x3ffffu) >> 4);
}

__device__ __forceinline__ void fence_after_sync() {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

__device__ __forceinline__ void fence_before_sync() {
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
}

template <int Columns, int Slots>
inline constexpr int tmem_columns = [] {
    int columns = 32;
    while (columns < Slots * Columns) columns *= 2;
    return columns;
}();

template <int Columns, int Accumulators = 0, int Slots>
__device__ __forceinline__ void allocate_tmem(DecodePipeline<Slots>& pipe) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                 :: "r"(shared_address(&pipe.tmem)),
                    "r"(tmem_columns<Columns, Accumulators ? Accumulators : Slots>) : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
}

template <int Columns, int Accumulators = 0, int Slots>
__device__ __forceinline__ void free_tmem(const DecodePipeline<Slots>& pipe) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                 :: "r"(pipe.tmem),
                    "r"(tmem_columns<Columns, Accumulators ? Accumulators : Slots>) : "memory");
}

template <bool PerStage = false, int Slots>
__device__ __forceinline__ void init_pipeline(DecodePipeline<Slots>& pipe) {
    for (int slot = 0; slot < Slots; ++slot) {
        init_barrier(&pipe.copy_ready[slot], 1);
        if constexpr (!PerStage) {
            init_barrier(&pipe.copy_empty[slot], 1);
        }
    }
    for (int slot = 0; slot < Slots; ++slot) {
        init_barrier(&pipe.accum_ready[slot], 1);
        init_barrier(&pipe.accum_empty[slot], consumer_warps);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}

__device__ __forceinline__ void load_tmem16(unsigned source, unsigned* out) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,"
                 "%10,%11,%12,%13,%14,%15}, [%16];\ntcgen05.wait::ld.sync.aligned;"
                 : "=r"(out[0]), "=r"(out[1]), "=r"(out[2]), "=r"(out[3]), "=r"(out[4]),
                   "=r"(out[5]), "=r"(out[6]), "=r"(out[7]), "=r"(out[8]), "=r"(out[9]),
                   "=r"(out[10]), "=r"(out[11]), "=r"(out[12]), "=r"(out[13]), "=r"(out[14]),
                   "=r"(out[15])
                 : "r"(source) : "memory");
}

__device__ __forceinline__ void load_tmem8(unsigned source, unsigned* out) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                 "\ntcgen05.wait::ld.sync.aligned;"
                 : "=r"(out[0]), "=r"(out[1]), "=r"(out[2]), "=r"(out[3]),
                   "=r"(out[4]), "=r"(out[5]), "=r"(out[6]), "=r"(out[7])
                 : "r"(source) : "memory");
}

__device__ __forceinline__ float load_tmem1(unsigned source) {
    unsigned value;
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0}, [%1];"
                 "\ntcgen05.wait::ld.sync.aligned;"
                 : "=r"(value) : "r"(source) : "memory");
    return __uint_as_float(value);
}

__device__ __forceinline__ void wait_dependencies() {
    asm volatile("griddepcontrol.wait;" ::: "memory");
}

__device__ __forceinline__ void launch_dependents() {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}

__device__ __forceinline__ void consumer_sync() {
    asm volatile("bar.sync 1, %0;" :: "n"(consumer_warps * warp_size) : "memory");
}

__device__ __forceinline__ void release_slot(unsigned long long* barrier) {
    fence_before_sync();
    __syncwarp();
    if (threadIdx.x % warp_size == 0) {
        arrive_barrier(barrier);
    }
}

template <int Elements, int Columns, bool PerStage = false, int Slots>
__device__ void produce(DecodePipeline<Slots>& pipe, unsigned shared, const CUtensorMap* weight,
                        const CUtensorMap* activation, int row_base, int act_row, int first,
                        int count) {
    const int prefetched = count < Slots ? count : Slots;
    for (int index = 0; index < prefetched; ++index) {
        expect_bytes(&pipe.copy_ready[index], stage_bytes<Columns>);
        copy_tile(weight, shared + index * stage_bytes<Columns>, &pipe.copy_ready[index],
                  (first + index) * Elements, row_base);
    }
    wait_dependencies();
    const auto* mma_done = PerStage ? pipe.accum_ready : pipe.copy_empty;
    for (int index = 0; index < count; ++index) {
        const int stage = index % Slots;
        const unsigned start = shared + stage * stage_bytes<Columns>;
        const int col = (first + index) * Elements;
        if (index >= Slots) {
            wait_barrier(&mma_done[stage], (index / Slots - 1) & 1);
            expect_bytes(&pipe.copy_ready[stage], stage_bytes<Columns>);
            copy_tile(weight, start, &pipe.copy_ready[stage], col, row_base);
        }
        copy_tile(activation, start + weight_bytes, &pipe.copy_ready[stage], col, act_row);
    }
    launch_dependents();
}

template <unsigned Instruction, bool PerStage, int Columns, int Slots>
__device__ void multiply(DecodePipeline<Slots>& pipe, unsigned shared, int count) {
    for (int index = 0; index < count; ++index) {
        const int stage = index % Slots;
        const int slot = PerStage ? stage : 0;
        if (PerStage && index >= Slots) {
            wait_barrier(&pipe.accum_empty[slot], (index / Slots - 1) & 1);
        }
        wait_barrier(&pipe.copy_ready[stage], (index / Slots) & 1);
        fence_after_sync();
        const unsigned start = shared + stage * stage_bytes<Columns>;
        const unsigned destination = pipe.tmem + slot * Columns;
        #pragma unroll
        for (int chunk = 0; chunk < 128; chunk += 32) {
            const auto weight_desc = matrix_desc(start + chunk);
            const auto input_desc = matrix_desc(start + weight_bytes + chunk);
            const unsigned accumulate = PerStage ? chunk : (index | chunk);
            if constexpr (((Instruction >> 7) & 7) != 0) {
                asm volatile("{ .reg .pred accumulate;\n"
                             "setp.ne.u32 accumulate, %4, 0;\n"
                             "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, accumulate; }"
                             :: "r"(destination), "l"(weight_desc), "l"(input_desc),
                                "r"(Instruction), "r"(accumulate) : "memory");
            } else {
                asm volatile("{ .reg .pred accumulate;\n"
                             "setp.ne.u32 accumulate, %4, 0;\n"
                             "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, "
                             "accumulate; }"
                             :: "r"(destination), "l"(weight_desc), "l"(input_desc),
                                "r"(Instruction), "r"(accumulate) : "memory");
            }
        }
        if constexpr (!PerStage) {
            commit_mma(&pipe.copy_empty[stage]);
        }
        if (PerStage || index == count - 1) {
            commit_mma(&pipe.accum_ready[slot]);
        }
    }
}

__device__ __forceinline__ void cluster_sync() {
    asm volatile("barrier.cluster.arrive.release.aligned;\n"
                 "barrier.cluster.wait.acquire.aligned;" ::: "memory");
}

__device__ __forceinline__ float load_cluster(const float* local, unsigned rank) {
    unsigned remote = 0;
    float value = 0.0f;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                 : "=r"(remote) : "r"(shared_address(local)), "r"(rank));
    asm volatile("ld.shared::cluster.f32 %0, [%1];" : "=f"(value) : "r"(remote) : "memory");
    return value;
}

struct ContiguousOutput {
    __nv_bfloat16* data;
    int width;
    int first;

    __device__ __forceinline__ __nv_bfloat16* operator()(int col) const {
        return data + std::size_t(first + col) * width;
    }
};

template <typename Output>
__device__ __forceinline__ void reduce_store(const float* partials, int splits, int active,
                                             int row_base, int out_width, Output output) {
    const int elements = active * tile_rows;
    const int rank = blockIdx.x;
    const int first = rank * elements / splits;
    const int last = (rank + 1) * elements / splits;
    for (int index = first + threadIdx.x; index < last; index += threads) {
        float sum = load_cluster(partials + index, 0);
        for (int source = 1; source < splits; ++source) {
            sum += load_cluster(partials + index, source);
        }
        const int col = index / tile_rows;
        const int row = row_base + index % tile_rows;
        if (row < out_width) {
            output(col)[row] = __float2bfloat16_rn(sum);
        }
    }
}

template <int Columns, typename Output>
__device__ __forceinline__ void store_result(const float (&result)[Columns], float* partials,
                                             int splits, int active, int row_base,
                                             int out_width, Output output) {
    const int warp = threadIdx.x / warp_size;
    const int local_row = threadIdx.x % tile_rows;
    if (splits == 1) {
        const int row = row_base + local_row;
        if (warp < consumer_warps && row < out_width) {
            #pragma unroll
            for (int col = 0; col < Columns; ++col) {
                if (col < active) {
                    output(col)[row] = __float2bfloat16_rn(result[col]);
                }
            }
        }
        return;
    }
    if (warp < consumer_warps) {
        #pragma unroll
        for (int col = 0; col < Columns; ++col) {
            if (col < active) {
                partials[col * tile_rows + local_row] = result[col];
            }
        }
    }
    cluster_sync();
    reduce_store(partials, splits, active, row_base, out_width, output);
    cluster_sync();
}

template <int Columns>
__device__ __forceinline__ void store_result(const float (&result)[Columns], float* partials,
                                             int splits, int active, int row_base,
                                             int out_width, int act_row,
                                             __nv_bfloat16* output) {
    store_result(result, partials, splits, active, row_base, out_width,
                 ContiguousOutput{output, out_width, act_row});
}

}
