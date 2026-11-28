#include "sglang/text_ops.h"

#include "sglang/decoder_internal.h"
#include "sglang/text/text_cuda_primitives.cuh"

#include <cstdint>
#include <mma.h>
#include <optional>

namespace sglang {
namespace decoder {

std::optional<cudaError_t> try_seed_fc_sm100(const __nv_bfloat16* weight,
                                             const __nv_bfloat16* input, int height, int width,
                                             __nv_bfloat16* output, cudaStream_t stream);
}

namespace {

namespace wmma = nvcuda::wmma;
using text_kernel::warp_width;
constexpr int seed_mma_edge = 16;
constexpr int seed_tile_rows = 32;
constexpr int seed_tile_columns = 64;
constexpr int seed_tile_k = 64;
constexpr int seed_vector_width = sizeof(uint4) / sizeof(__nv_bfloat16);
constexpr int seed_stride = seed_tile_k + seed_vector_width;
constexpr int seed_threads = seed_tile_rows * seed_tile_columns /
                             (seed_mma_edge * seed_mma_edge) * warp_width;
constexpr int seed_inner = 2 * decoder::hidden_size;

__device__ void seed_copy_vector(__nv_bfloat16* destination, const __nv_bfloat16* source,
                                 bool valid) {
    const auto shared_address = static_cast<std::uint32_t>(__cvta_generic_to_shared(destination));
    const int source_bytes = valid ? sizeof(uint4) : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
                 :: "r"(shared_address), "l"(source), "r"(source_bytes) : "memory");
}

struct SeedOperands {
    __nv_bfloat16 lhs[seed_tile_rows][seed_stride];
    __nv_bfloat16 rhs[seed_tile_columns][seed_stride];
};

__device__ void seed_stage_operands(SeedOperands& operands, const __nv_bfloat16* input,
                                    const __nv_bfloat16* weight, int rows, int first_row,
                                    int first_column, int offset) {
    constexpr int vectors_per_row = seed_tile_k / seed_vector_width;
    for (int vector = threadIdx.x; vector < seed_tile_rows * vectors_per_row;
         vector += seed_threads) {
        const int row = vector / vectors_per_row;
        const int column = vector % vectors_per_row * seed_vector_width;
        const bool valid = first_row + row < rows;
        const auto* source = valid ? input + (first_row + row) * seed_inner + offset + column
                                   : input;
        seed_copy_vector(&operands.lhs[row][column], source, valid);
    }
    for (int vector = threadIdx.x; vector < seed_tile_columns * vectors_per_row;
         vector += seed_threads) {
        const int row = vector / vectors_per_row;
        const int column = vector % vectors_per_row * seed_vector_width;
        seed_copy_vector(&operands.rhs[row][column],
                         weight + (first_column + row) * seed_inner + offset + column, true);
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
}

__global__ __launch_bounds__(seed_threads) void bf16_seed_fc_kernel(const __nv_bfloat16* weight,
                                                                    const __nv_bfloat16* input,
                                                                    int rows, __nv_bfloat16* output,
                                                                    int width) {
    union alignas(32) SeedTile {
        SeedOperands operands[2];
        float result[seed_tile_rows][seed_tile_columns];
    };
    __shared__ SeedTile tile;
    const int first_row = blockIdx.y * seed_tile_rows;
    const int first_column = blockIdx.x * seed_tile_columns;
    const int warp = threadIdx.x / warp_width;
    const int row_tile = warp / (seed_tile_columns / seed_mma_edge) * seed_mma_edge;
    const int column_tile = warp % (seed_tile_columns / seed_mma_edge) * seed_mma_edge;
    wmma::fragment<wmma::accumulator, seed_mma_edge, seed_mma_edge, seed_mma_edge, float>
        accumulator;
    wmma::fill_fragment(accumulator, 0.0f);
    seed_stage_operands(tile.operands[0], input, weight, rows, first_row, first_column, 0);
    for (int offset = 0; offset < seed_inner; offset += seed_tile_k) {
        const int stage = (offset / seed_tile_k) % 2;
        if (offset + seed_tile_k < seed_inner) {
            seed_stage_operands(tile.operands[1 - stage], input, weight, rows, first_row,
                                first_column, offset + seed_tile_k);
            asm volatile("cp.async.wait_group 1;" ::: "memory");
        } else {
            asm volatile("cp.async.wait_group 0;" ::: "memory");
        }
        __syncthreads();
        #pragma unroll
        for (int depth = 0; depth < seed_tile_k; depth += seed_mma_edge) {
            wmma::fragment<wmma::matrix_a, seed_mma_edge, seed_mma_edge, seed_mma_edge,
                           __nv_bfloat16, wmma::row_major> left;
            wmma::fragment<wmma::matrix_b, seed_mma_edge, seed_mma_edge, seed_mma_edge,
                           __nv_bfloat16, wmma::col_major> right;
            wmma::load_matrix_sync(right, &tile.operands[stage].rhs[column_tile][depth],
                                   seed_stride);
            wmma::load_matrix_sync(left, &tile.operands[stage].lhs[row_tile][depth], seed_stride);
            wmma::mma_sync(accumulator, left, right, accumulator);
        }
        __syncthreads();
    }
    wmma::store_matrix_sync(&tile.result[row_tile][column_tile], accumulator, seed_tile_columns,
                            wmma::mem_row_major);
    __syncthreads();
    for (int element = threadIdx.x; element < seed_tile_rows * seed_tile_columns;
         element += seed_threads) {
        const int row = element / seed_tile_columns;
        const int column = element % seed_tile_columns;
        if (first_row + row < rows) {
            output[((first_row + row) * width) + first_column + column] =
                __float2bfloat16_rn(tile.result[row][column]);
        }
    }
}
}

cudaError_t text_bf16_seed_fc(const __nv_bfloat16* weight, const __nv_bfloat16* input,
                             int out_width, int in_width, int rows, __nv_bfloat16* output,
                             cudaStream_t stream) {
    if (weight == nullptr || input == nullptr || output == nullptr || out_width <= 0 ||
        out_width % seed_tile_columns != 0 || in_width != seed_inner ||
        rows < decoder::wmma_seed_min_rows || rows > decoder::wmma_seed_max_rows ||
        reinterpret_cast<std::uintptr_t>(weight) % sizeof(uint4) != 0 ||
        reinterpret_cast<std::uintptr_t>(input) % sizeof(uint4) != 0) {
        return text_bf16_gemv_rows(weight, input, out_width, in_width, rows, output, stream);
    }
    static_assert(decoder::hidden_size % seed_tile_columns == 0 &&
                  seed_inner % seed_tile_k == 0);
    if (reinterpret_cast<std::uintptr_t>(output) % alignof(unsigned) == 0) {
        if (const auto status =
                decoder::try_seed_fc_sm100(weight, input, rows, out_width, output, stream)) {
            return *status;
        }
    }
    const auto tiles = (rows + seed_tile_rows - 1) / seed_tile_rows;
    bf16_seed_fc_kernel<<<dim3(out_width / seed_tile_columns, tiles), seed_threads, 0, stream>>>(
        weight, input, rows, output, out_width);
    return cudaGetLastError();
}

}
