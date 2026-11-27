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

inline constexpr int bf16_chunk_elements = 64;

inline constexpr int bf16_small_columns = 16;

int bf16_columns(int rows);

template <int Slots>
struct DecodePipeline {
    unsigned long long copy_ready[Slots];
    unsigned long long copy_empty[Slots];
    unsigned long long accum_ready[Slots];
    unsigned long long accum_empty[Slots];
    unsigned tmem;
};

using Pipeline = DecodePipeline<stages>;

__device__ unsigned shared_address(const void* pointer);

__device__ void init_barrier(unsigned long long* barrier, int arrivals);

__device__ void wait_barrier(const unsigned long long* barrier, unsigned phase);

__device__ void expect_bytes(unsigned long long* barrier, unsigned bytes);

__device__ void commit_mma(unsigned long long* barrier);

__device__ void copy_tile(const CUtensorMap* map, unsigned shared,
                          unsigned long long* barrier, int col, int row);

__device__ std::uint64_t matrix_desc(unsigned start);

__device__ void fence_after_sync();

__device__ void fence_before_sync();

template <int Required, int Capacity, bool Fits = (Capacity >= Required)>
inline constexpr int tmem_capacity = tmem_capacity<Required, Capacity * 2>;

template <int Required, int Capacity>
inline constexpr int tmem_capacity<Required, Capacity, true> = Capacity;

template <int Columns, int Slots>
inline constexpr int tmem_columns = tmem_capacity<Columns * Slots, 32>;

template <int Columns>
inline constexpr unsigned fp8_instruction =
    ((tile_rows / 16) << 24) | ((Columns / 8) << 17) | (1u << 4);

template <int Columns>
inline constexpr unsigned bf16_instruction = fp8_instruction<Columns> | (1u << 10) | (1u << 7);

template <int Columns, int Accumulators = 0, int Slots>
__device__ void allocate_tmem(DecodePipeline<Slots>& pipe);

template <int Columns, int Accumulators = 0, int Slots>
__device__ void free_tmem(const DecodePipeline<Slots>& pipe);

template <bool PerStage = false, int Slots>
__device__ void init_pipeline(DecodePipeline<Slots>& pipe);

__device__ void load_tmem16(unsigned source, unsigned* out);

__device__ void load_tmem8(unsigned source, unsigned* out);

__device__ float load_tmem1(unsigned source);

__device__ void wait_dependencies();

__device__ void consumer_sync();

__device__ void release_slot(unsigned long long* barrier);

template <int Elements, int Columns, bool PerStage = false, int Slots>
__device__ void produce(DecodePipeline<Slots>& pipe, unsigned shared, const CUtensorMap* weight,
                        const CUtensorMap* activation, int row_base, int act_row, int first,
                        int count);

template <unsigned Instruction, bool PerStage, int Columns, int Slots>
__device__ void multiply(DecodePipeline<Slots>& pipe, unsigned shared, int count);

template <int Columns>
__device__ void bf16_tile(Pipeline& pipe, unsigned char* tiles,
                          const CUtensorMap* weight, const CUtensorMap* input,
                          int token, int row, int first, int count, int active,
                          float (&result)[Columns]);

__device__ void cluster_sync();

__device__ void reduce_store(const float* partials, int splits, int active, int row_base,
                             int out_width, __nv_bfloat16* const* output);

template <int Columns>
__device__ void store_result(const float (&result)[Columns], float* partials,
                              int splits, int active, int row_base, int out_width,
                              __nv_bfloat16* const* output);

template <int Columns>
__device__ void store_result(const float (&result)[Columns], float* partials,
                              int splits, int active, int row_base,
                              int out_width, int act_row,
                              __nv_bfloat16* output);

}
