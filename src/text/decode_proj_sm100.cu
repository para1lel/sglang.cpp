#include "sglang/text/decode_proj_sm100.cuh"

#include <cuda/ptx>

namespace sglang::decode_proj {

__device__ __inline_hint__ unsigned shared_address(const void* pointer) {
    return static_cast<unsigned>(__cvta_generic_to_shared(pointer));
}

__device__ __inline_hint__ void init_barrier(unsigned long long* barrier, int arrivals) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
                 :: "r"(shared_address(barrier)), "r"(arrivals) : "memory");
}

__device__ __inline_hint__ void wait_barrier(const unsigned long long* barrier, unsigned phase) {
    asm volatile("{ .reg .pred ready;\n"
                 "WAIT: mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 ready, [%0], %1;\n"
                 "@!ready bra.uni WAIT; }"
                 :: "r"(shared_address(barrier)), "r"(phase) : "memory");
}

static __device__ __forceinline__ void arrive_barrier(unsigned long long* barrier) {
    asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];"
                 :: "r"(shared_address(barrier)) : "memory");
}

__device__ __inline_hint__ void expect_bytes(unsigned long long* barrier, unsigned bytes) {
    asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
                 :: "r"(shared_address(barrier)), "r"(bytes) : "memory");
}

__device__ __inline_hint__ void commit_mma(unsigned long long* barrier) {
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
                 :: "r"(shared_address(barrier)) : "memory");
}

__device__ __inline_hint__ void copy_tile(const CUtensorMap* map, unsigned shared,
                                          unsigned long long* barrier, int col, int row) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes "
                 "[%0], [%1, {%2, %3}], [%4];"
                 :: "r"(shared), "l"(map), "r"(col), "r"(row), "r"(shared_address(barrier))
                 : "memory");
}

__device__ __inline_hint__ std::uint64_t matrix_desc(unsigned start) {
    constexpr std::uint64_t stride = 8 * 128 / 16;
    constexpr std::uint64_t swizzle_128b = std::uint64_t{2} << 61;
    constexpr std::uint64_t descriptor_version = std::uint64_t{1} << 46;
    constexpr std::uint64_t leading_offset = std::uint64_t{1} << 16;
    return swizzle_128b | descriptor_version | (stride << 32) | leading_offset |
           ((start & 0x3ffffu) >> 4);
}

__device__ __inline_hint__ void fence_after_sync() {
    cuda::ptx::tcgen05_fence_after_thread_sync();
}

__device__ __inline_hint__ void fence_before_sync() {
    cuda::ptx::tcgen05_fence_before_thread_sync();
}

__device__ __inline_hint__ void allocate_tmem(unsigned* address, int columns) {
    cuda::ptx::tcgen05_alloc(cuda::ptx::cta_group_1, address, columns);
    cuda::ptx::tcgen05_relinquish_alloc_permit(cuda::ptx::cta_group_1);
}

__device__ __inline_hint__ void free_tmem(const unsigned* address, int columns) {
    cuda::ptx::tcgen05_dealloc(cuda::ptx::cta_group_1, *address, columns);
}

template <int Columns, int Accumulators, int Slots>
__device__ __inline_hint__ void allocate_tmem(DecodePipeline<Slots>& pipe) {
    constexpr int count = Accumulators ? Accumulators : Slots;
    allocate_tmem(&pipe.tmem, tmem_columns<Columns, count>);
}

template <int Columns, int Accumulators, int Slots>
__device__ __inline_hint__ void free_tmem(const DecodePipeline<Slots>& pipe) {
    constexpr int count = Accumulators ? Accumulators : Slots;
    free_tmem(&pipe.tmem, tmem_columns<Columns, count>);
}

template <bool PerStage, int Slots>
__device__ __inline_hint__ void init_pipeline(DecodePipeline<Slots>& pipe) {
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
    cuda::ptx::fence_mbarrier_init(cuda::ptx::sem_release, cuda::ptx::scope_cluster);
}

__device__ __inline_hint__ void load_tmem16(unsigned source, unsigned* out) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,"
                 "%10,%11,%12,%13,%14,%15}, [%16];\ntcgen05.wait::ld.sync.aligned;"
                 : "=r"(out[0]), "=r"(out[1]), "=r"(out[2]), "=r"(out[3]), "=r"(out[4]),
                   "=r"(out[5]), "=r"(out[6]), "=r"(out[7]), "=r"(out[8]), "=r"(out[9]),
                   "=r"(out[10]), "=r"(out[11]), "=r"(out[12]), "=r"(out[13]), "=r"(out[14]),
                   "=r"(out[15])
                 : "r"(source) : "memory");
}

__device__ __inline_hint__ void load_tmem8(unsigned source, unsigned* out) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                 "\ntcgen05.wait::ld.sync.aligned;"
                 : "=r"(out[0]), "=r"(out[1]), "=r"(out[2]), "=r"(out[3]),
                   "=r"(out[4]), "=r"(out[5]), "=r"(out[6]), "=r"(out[7])
                 : "r"(source) : "memory");
}

__device__ __inline_hint__ float load_tmem1(unsigned source) {
    unsigned value;
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0}, [%1];"
                 "\ntcgen05.wait::ld.sync.aligned;"
                 : "=r"(value) : "r"(source) : "memory");
    return __uint_as_float(value);
}

__device__ __inline_hint__ void wait_dependencies() {
    asm volatile("griddepcontrol.wait;" ::: "memory");
}

static __device__ __forceinline__ void launch_dependents() {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}

__device__ __inline_hint__ void consumer_sync() {
    asm volatile("bar.sync 1, %0;" :: "n"(consumer_warps * warp_size) : "memory");
}

__device__ __inline_hint__ void release_slot(unsigned long long* barrier) {
    fence_before_sync();
    __syncwarp();
    if (threadIdx.x % warp_size == 0) {
        arrive_barrier(barrier);
    }
}

template <int Elements, int Columns, bool PerStage, int Slots>
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

template <int Columns>
__device__ void collect_bf16(Pipeline& pipe, int active, float (&result)[Columns]) {
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
__device__ __inline_hint__ void bf16_tile(Pipeline& pipe, unsigned char* tiles,
                                        const CUtensorMap* weight, const CUtensorMap* input,
                                        int token, int row, int first, int count, int active,
                                        float (&result)[Columns]) {
    const int warp = threadIdx.x / warp_size;
    if (threadIdx.x == 0) init_pipeline(pipe);
    if (warp == mma_warp) allocate_tmem<Columns>(pipe);
    __syncthreads();
    const unsigned shared = shared_address(tiles);
    if (warp < consumer_warps) {
        collect_bf16(pipe, active, result);
    } else if (warp == mma_warp && threadIdx.x % warp_size == 0) {
        multiply<bf16_instruction<Columns>, false, Columns>(pipe, shared, count);
    } else if (warp == copy_warp && threadIdx.x % warp_size == 0) {
        produce<bf16_chunk_elements, Columns>(pipe, shared, weight, input, token, row,
                                              first, count);
    }
    wait_dependencies();
    __syncthreads();
    if (warp == mma_warp) free_tmem<Columns>(pipe);
}
__device__ __inline_hint__ void cluster_sync() {
    asm volatile("barrier.cluster.arrive.release.aligned;\n"
                 "barrier.cluster.wait.acquire.aligned;" ::: "memory");
}

static __device__ __forceinline__ float load_cluster(const float* local, unsigned rank) {
    unsigned remote = 0;
    float value = 0.0f;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                 : "=r"(remote) : "r"(shared_address(local)), "r"(rank));
    asm volatile("ld.shared::cluster.f32 %0, [%1];" : "=f"(value) : "r"(remote) : "memory");
    return value;
}
namespace {

struct RowPointers {
    __nv_bfloat16* const* data;

    __device__ __forceinline__ __nv_bfloat16* operator()(int col) const {
        return data[col];
    }
};

struct ContiguousOutput {
    __nv_bfloat16* data;
    int width;
    int first;

    __device__ __forceinline__ __nv_bfloat16* operator()(int col) const {
        return data + std::size_t(first + col) * width;
    }
};

template <typename Output>
__device__ __forceinline__ void reduce_store_output(const float* partials, int splits, int active,
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
__device__ __forceinline__ void store_result_output(const float (&result)[Columns], float* partials,
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
    reduce_store_output(partials, splits, active, row_base, out_width, output);
    cluster_sync();
}

}

__device__ __inline_hint__ void reduce_store(const float* partials, int splits, int active,
                                             int row_base, int out_width,
                                             __nv_bfloat16* const* output) {
    reduce_store_output(partials, splits, active, row_base, out_width, RowPointers{output});
}

template <int Columns>
__device__ __inline_hint__ void store_result(const float (&result)[Columns], float* partials,
                              int splits, int active, int row_base, int out_width,
                              __nv_bfloat16* const* output) {
    store_result_output(result, partials, splits, active, row_base, out_width, RowPointers{output});
}

template <int Columns>
__device__ __inline_hint__ void store_result(const float (&result)[Columns], float* partials,
                                             int splits, int active, int row_base,
                                             int out_width, int act_row,
                                             __nv_bfloat16* output) {
    store_result_output(result, partials, splits, active, row_base, out_width,
                 ContiguousOutput{output, out_width, act_row});
}

template __device__ void init_pipeline<false, 2>(DecodePipeline<2>&);
template __device__ void init_pipeline<false, 3>(DecodePipeline<3>&);
template __device__ void init_pipeline<false, 4>(DecodePipeline<4>&);
template __device__ void init_pipeline<true, 3>(DecodePipeline<3>&);
template __device__ void init_pipeline<true, 4>(DecodePipeline<4>&);
template __device__ void init_pipeline<true, 8>(DecodePipeline<8>&);

template __device__ void allocate_tmem<16, 1, 4>(DecodePipeline<4>&);
template __device__ void free_tmem<16, 1, 4>(const DecodePipeline<4>&);
template __device__ void produce<128, 8, true, 8>(DecodePipeline<8>&, unsigned,
    const CUtensorMap*, const CUtensorMap*, int, int, int, int);
template __device__ void produce<128, 16, true, 8>(DecodePipeline<8>&, unsigned,
    const CUtensorMap*, const CUtensorMap*, int, int, int, int);
template __device__ void produce<128, 32, true, 8>(DecodePipeline<8>&, unsigned,
    const CUtensorMap*, const CUtensorMap*, int, int, int, int);
template __device__ void produce<128, 64, true, 8>(DecodePipeline<8>&, unsigned,
    const CUtensorMap*, const CUtensorMap*, int, int, int, int);
template __device__ void produce<128, 16, true, 4>(DecodePipeline<4>&, unsigned,
    const CUtensorMap*, const CUtensorMap*, int, int, int, int);
template __device__ void produce<128, 32, true, 4>(DecodePipeline<4>&, unsigned,
    const CUtensorMap*, const CUtensorMap*, int, int, int, int);
template __device__ void produce<128, 48, true, 4>(DecodePipeline<4>&, unsigned,
    const CUtensorMap*, const CUtensorMap*, int, int, int, int);
template __device__ void produce<128, 64, true, 4>(DecodePipeline<4>&, unsigned,
    const CUtensorMap*, const CUtensorMap*, int, int, int, int);
template __device__ void produce<128, 80, true, 4>(DecodePipeline<4>&, unsigned,
    const CUtensorMap*, const CUtensorMap*, int, int, int, int);
template __device__ void produce<128, 160, true, 3>(DecodePipeline<3>&, unsigned,
    const CUtensorMap*, const CUtensorMap*, int, int, int, int);
template __device__ void produce<128, 16, false, 2>(DecodePipeline<2>&, unsigned,
    const CUtensorMap*, const CUtensorMap*, int, int, int, int);
template __device__ void produce<128, 32, false, 2>(DecodePipeline<2>&, unsigned,
    const CUtensorMap*, const CUtensorMap*, int, int, int, int);
template __device__ void produce<128, 16, false, 4>(DecodePipeline<4>&, unsigned,
    const CUtensorMap*, const CUtensorMap*, int, int, int, int);
template __device__ void produce<128, 32, false, 4>(DecodePipeline<4>&, unsigned,
    const CUtensorMap*, const CUtensorMap*, int, int, int, int);
template __device__ void produce<128, 64, false, 4>(DecodePipeline<4>&, unsigned,
    const CUtensorMap*, const CUtensorMap*, int, int, int, int);
template __device__ void produce<128, 80, false, 4>(DecodePipeline<4>&, unsigned,
                                                    const CUtensorMap*, const CUtensorMap*, int,
                                                    int, int, int);
template __device__ void produce<128, 16, false, 8>(DecodePipeline<8>&, unsigned,
    const CUtensorMap*, const CUtensorMap*, int, int, int, int);

template __device__ void multiply<fp8_instruction<8>, true, 8, 8>(
    DecodePipeline<8>&, unsigned, int);
template __device__ void multiply<fp8_instruction<16>, true, 16, 8>(
    DecodePipeline<8>&, unsigned, int);
template __device__ void multiply<fp8_instruction<32>, true, 32, 8>(
    DecodePipeline<8>&, unsigned, int);
template __device__ void multiply<fp8_instruction<64>, true, 64, 8>(
    DecodePipeline<8>&, unsigned, int);
template __device__ void multiply<fp8_instruction<16>, true, 16, 4>(
    DecodePipeline<4>&, unsigned, int);
template __device__ void multiply<fp8_instruction<32>, true, 32, 4>(
    DecodePipeline<4>&, unsigned, int);
template __device__ void multiply<fp8_instruction<48>, true, 48, 4>(
    DecodePipeline<4>&, unsigned, int);
template __device__ void multiply<fp8_instruction<64>, true, 64, 4>(
    DecodePipeline<4>&, unsigned, int);
template __device__ void multiply<fp8_instruction<80>, true, 80, 4>(
    DecodePipeline<4>&, unsigned, int);
template __device__ void multiply<fp8_instruction<160>, true, 160, 3>(
    DecodePipeline<3>&, unsigned, int);
template __device__ void multiply<bf16_instruction<32>, false, 32, 3>(
    DecodePipeline<3>&, unsigned, int);

template __device__ void bf16_tile<16>(Pipeline&, unsigned char*, const CUtensorMap*,
    const CUtensorMap*, int, int, int, int, int, float (&)[16]);
template __device__ void bf16_tile<32>(Pipeline&, unsigned char*, const CUtensorMap*,
    const CUtensorMap*, int, int, int, int, int, float (&)[32]);
template __device__ void bf16_tile<64>(Pipeline&, unsigned char*, const CUtensorMap*,
    const CUtensorMap*, int, int, int, int, int, float (&)[64]);

template __device__ void store_result<8>(const float (&)[8], float*, int, int, int, int,
    __nv_bfloat16* const*);
template __device__ void store_result<10>(const float (&)[10], float*, int, int, int, int,
    __nv_bfloat16* const*);
template __device__ void store_result<16>(const float (&)[16], float*, int, int, int, int,
    __nv_bfloat16* const*);
template __device__ void store_result<32>(const float (&)[32], float*, int, int, int, int,
    __nv_bfloat16* const*);
template __device__ void store_result<40>(const float (&)[40], float*, int, int, int, int,
    __nv_bfloat16* const*);
template __device__ void store_result<64>(const float (&)[64], float*, int, int, int, int,
    __nv_bfloat16* const*);
template __device__ void store_result<80>(const float (&)[80], float*, int, int, int, int,
    __nv_bfloat16* const*);
template __device__ void store_result<160>(const float (&)[160], float*, int, int, int, int,
    __nv_bfloat16* const*);

template __device__ void store_result<8>(const float (&)[8], float*, int, int, int, int,
    int, __nv_bfloat16*);
template __device__ void store_result<16>(const float (&)[16], float*, int, int, int, int,
    int, __nv_bfloat16*);
template __device__ void store_result<32>(const float (&)[32], float*, int, int, int, int,
    int, __nv_bfloat16*);
template __device__ void store_result<64>(const float (&)[64], float*, int, int, int, int,
    int, __nv_bfloat16*);

}
