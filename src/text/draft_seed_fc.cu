#include "sglang/text_ops.h"

#include "sglang/decoder_internal.h"
#include "sglang/text/text_cuda_primitives.cuh"

#include <cstdint>
#include <mma.h>

namespace sglang {
namespace {

namespace wmma = nvcuda::wmma;
using text_kernel::warp_width;
constexpr int seed_mma_edge = 16;
constexpr int seed_tile_rows = 32;
constexpr int seed_tile_columns = 64;
constexpr int seed_tile_k = 64;
constexpr int seed_stride = seed_tile_k + 8;

__global__ void bf16_seed_fc_kernel(const __nv_bfloat16* weight, const __nv_bfloat16* input,
                                     int rows, int inner, int columns, __nv_bfloat16* output) {
    __shared__ alignas(32) __nv_bfloat16 lhs[seed_tile_rows][seed_stride];
    __shared__ alignas(32) __nv_bfloat16 rhs[seed_tile_columns][seed_stride];
    __shared__ alignas(32) float result[seed_tile_rows][seed_tile_columns];
    const int first_row = blockIdx.y * seed_tile_rows;
    const int first_column = blockIdx.x * seed_tile_columns;
    const int warp = threadIdx.x / warp_width;
    const int row_tile = warp / (seed_tile_columns / seed_mma_edge) * seed_mma_edge;
    const int column_tile = warp % (seed_tile_columns / seed_mma_edge) * seed_mma_edge;
    wmma::fragment<wmma::accumulator, seed_mma_edge, seed_mma_edge, seed_mma_edge, float>
        accumulator;
    wmma::fill_fragment(accumulator, 0.0f);
    constexpr int vector_width = sizeof(uint4) / sizeof(__nv_bfloat16);
    constexpr int vectors_per_row = seed_tile_k / vector_width;
    for (int offset = 0; offset < inner; offset += seed_tile_k) {
        for (int vector = threadIdx.x; vector < seed_tile_rows * vectors_per_row;
             vector += blockDim.x) {
            const int row = vector / vectors_per_row;
            const int column = vector % vectors_per_row * vector_width;
            auto* destination = reinterpret_cast<uint4*>(&lhs[row][column]);
            *destination = first_row + row < rows
                ? *reinterpret_cast<const uint4*>(input + (first_row + row) * inner +
                                                  offset + column)
                : uint4{};
        }
        for (int vector = threadIdx.x; vector < seed_tile_columns * vectors_per_row;
             vector += blockDim.x) {
            const int row = vector / vectors_per_row;
            const int column = vector % vectors_per_row * vector_width;
            auto* destination = reinterpret_cast<uint4*>(&rhs[row][column]);
            *destination = *reinterpret_cast<const uint4*>(
                weight + (first_column + row) * inner + offset + column);
        }
        __syncthreads();
        #pragma unroll
        for (int depth = 0; depth < seed_tile_k; depth += seed_mma_edge) {
            wmma::fragment<wmma::matrix_a, seed_mma_edge, seed_mma_edge, seed_mma_edge,
                           __nv_bfloat16, wmma::row_major> left;
            wmma::fragment<wmma::matrix_b, seed_mma_edge, seed_mma_edge, seed_mma_edge,
                           __nv_bfloat16, wmma::col_major> right;
            wmma::load_matrix_sync(left, &lhs[row_tile][depth], seed_stride);
            wmma::load_matrix_sync(right, &rhs[column_tile][depth], seed_stride);
            wmma::mma_sync(accumulator, left, right, accumulator);
        }
        __syncthreads();
    }
    wmma::store_matrix_sync(&result[row_tile][column_tile], accumulator, seed_tile_columns,
                            wmma::mem_row_major);
    __syncthreads();
    for (int element = threadIdx.x; element < seed_tile_rows * seed_tile_columns;
         element += blockDim.x) {
        const int row = element / seed_tile_columns;
        const int column = element % seed_tile_columns;
        if (first_row + row < rows) {
            output[(first_row + row) * columns + first_column + column] =
                __float2bfloat16_rn(result[row][column]);
        }
    }
}

}

cudaError_t text_bf16_seed_fc(const __nv_bfloat16* weight, const __nv_bfloat16* input,
                             int out_width, int in_width, int rows, __nv_bfloat16* output,
                             cudaStream_t stream) {
    if (!weight || !input || !output || out_width != decoder::hidden_size ||
        in_width != 2 * decoder::hidden_size || rows < decoder::wmma_seed_min_rows ||
        rows > decoder::wmma_seed_max_rows ||
        reinterpret_cast<std::uintptr_t>(weight) % sizeof(uint4) != 0 ||
        reinterpret_cast<std::uintptr_t>(input) % sizeof(uint4) != 0) {
        return text_bf16_gemv_rows(weight, input, out_width, in_width, rows, output, stream);
    }
    static_assert(decoder::hidden_size % seed_tile_columns == 0 &&
                  2 * decoder::hidden_size % seed_tile_k == 0);
    const auto tiles = (rows + seed_tile_rows - 1) / seed_tile_rows;
    constexpr int threads = seed_tile_rows * seed_tile_columns /
                            (seed_mma_edge * seed_mma_edge) * warp_width;
    bf16_seed_fc_kernel<<<dim3(out_width / seed_tile_columns, tiles), threads, 0, stream>>>(
        weight, input, rows, in_width, out_width, output);
    return cudaGetLastError();
}

}
