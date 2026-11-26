#pragma once

#include "sglang/text/decode_proj_sm100.cuh"
#include "sglang/text/mtp/tc_vocab_certificate.cuh"

namespace sglang::decoder::tc_vocab {
namespace proj = sglang::decode_proj;

constexpr int score_stages = 4;
constexpr int score_chunk = 128;
template <int Columns>
constexpr int score_shared_bytes = score_stages * proj::stage_bytes<Columns>;

template <int Columns>
__device__ void multiply_scores(proj::DecodePipeline<score_stages>& pipe, unsigned shared) {
    using namespace decode_proj;
    constexpr unsigned instruction = (tile_rows / 16 << 24) | (Columns / 8 << 17) |
                                     (1u << 10) | (1u << 7) | (2u << 4);
    for (int index = 0; index < width / score_chunk; ++index) {
        const int stage = index % score_stages;
        wait_barrier(&pipe.copy_ready[stage], (index / score_stages) & 1);
        fence_after_sync();
        const unsigned start = shared + stage * stage_bytes<Columns>;
        #pragma unroll
        for (int chunk = 0; chunk < score_chunk; chunk += 32) {
            const auto weight = matrix_desc(start + chunk);
            const auto input = matrix_desc(start + weight_bytes + chunk);
            const unsigned accumulate = index | chunk;
            asm volatile("{ .reg .pred accumulate;\n"
                         "setp.ne.u32 accumulate, %4, 0;\n"
                         "tcgen05.mma.cta_group::1.kind::i8 [%0], %1, %2, %3, accumulate; }"
                         :: "r"(pipe.tmem), "l"(weight), "l"(input),
                            "r"(instruction), "r"(accumulate) : "memory");
        }
        commit_mma(&pipe.copy_empty[stage]);
    }
    commit_mma(&pipe.accum_ready[0]);
}

constexpr int short_rows = tc_vocab_short_rows;
constexpr int short_columns = proj::bf16_columns(short_rows);
constexpr int short_tiles = vocabulary / proj::tile_rows;
constexpr int short_bf16_shared = proj::stages * proj::stage_bytes<short_columns>;
static_assert(short_rows > 8 && short_rows <= 16);
static_assert(vocabulary % proj::tile_rows == 0 && width % proj::bf16_chunk_elements == 0);

struct ShortCandidates {
    int tile_count;
    int row_count[short_rows];
    int tiles[short_tiles];
};

__device__ inline float warp_max(float value) {
    for (int offset = 16; offset > 0; offset /= 2) {
        value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
    }
    return value;
}

__device__ inline Winner warp_best(Winner winner) {
    for (int offset = 16; offset > 0; offset /= 2) {
        winner = best(winner, {__shfl_down_sync(0xffffffff, winner.score, offset),
                               __shfl_down_sync(0xffffffff, winner.token, offset)});
    }
    return winner;
}

__device__ inline Interval packed_interval(int raw, const Stats& weight, const Stats& input) {

    if (!isfinite(weight.value_norm) || !isfinite(input.value_norm)) {
        return {-CUDART_INF_F, CUDART_INF_F};
    }

    return interval(raw, weight, input, 0x1p-9f);
}

__global__ __launch_bounds__(proj::threads, 2) void screen_short_scores(
    const __grid_constant__ CUtensorMap weight, const __grid_constant__ CUtensorMap input,
    const Stats* weight_stats, const Stats* input_stats, int* upper, float* lower) {
    alignas(1024) extern __shared__ unsigned char shared_tiles[];
    __shared__ proj::DecodePipeline<score_stages> pipe;
    __shared__ Stats activation_stats[short_rows];
    __shared__ float warp_lower[short_rows][proj::consumer_warps];
    const int lane = threadIdx.x % proj::warp_size;
    const int warp = threadIdx.x / proj::warp_size;
    const int first = blockIdx.x * proj::tile_rows;
    if (threadIdx.x == 0) proj::init_pipeline(pipe);
    if (threadIdx.x < short_rows) activation_stats[threadIdx.x] = input_stats[threadIdx.x];
    if (warp == proj::mma_warp) proj::allocate_tmem<short_columns, 1>(pipe);
    __syncthreads();
    const unsigned shared = proj::shared_address(shared_tiles);
    if (warp < proj::consumer_warps) {
        const int token = first + threadIdx.x;
        proj::wait_barrier(&pipe.accum_ready[0], 0);
        proj::fence_after_sync();
        unsigned raw[short_columns];
        proj::load_tmem16(pipe.tmem + (static_cast<unsigned>(warp * proj::warp_size) << 16), raw);
        proj::fence_before_sync();
        const Stats token_stats = weight_stats[token];
        #pragma unroll
        for (int row = 0; row < short_rows; ++row) {
            const auto bounds = packed_interval(static_cast<int>(raw[row]), token_stats,
                                                activation_stats[row]);
            upper[row * vocabulary + token] = __float_as_int(bounds.upper);
            const float maximum = warp_max(bounds.lower);
            if (lane == 0) warp_lower[row][warp] = maximum;
        }
    } else if (warp == proj::mma_warp && lane == 0) {
        multiply_scores<short_columns>(pipe, shared);
    } else if (warp == proj::copy_warp && lane == 0) {
        proj::produce<score_chunk, short_columns>(pipe, shared, &weight, &input, first, 0, 0,
                                                  width / score_chunk);
    }
    __syncthreads();
    if (warp == proj::mma_warp) proj::free_tmem<short_columns, 1>(pipe);
    if (threadIdx.x < short_rows) {
        float maximum = -CUDART_INF_F;
        for (float value : warp_lower[threadIdx.x]) maximum = fmaxf(maximum, value);
        lower[threadIdx.x * short_tiles + blockIdx.x] = maximum;
    }
}

__global__ void select_short_seeds(const float* lower, float* threshold, int* seeds) {
    const auto* row = lower + blockIdx.x * short_tiles;
    Winner seed{-CUDART_INF_F, -1};
    for (int tile = threadIdx.x; tile < short_tiles; tile += blockDim.x) {
        seed = best(seed, {row[tile], tile});
    }
    seed = block_best(warp_best(seed));
    if (threadIdx.x == 0) {
        threshold[blockIdx.x] = seed.score;
        seeds[blockIdx.x] = seed.token;
    }
}

__global__ __launch_bounds__(proj::threads, 1) void improve_short_thresholds(
    const __grid_constant__ CUtensorMap weight, const __grid_constant__ CUtensorMap input,
    const int* seeds, float* threshold) {
    const int row = blockIdx.x;
    if (threshold[row] == -CUDART_INF_F) return;
    alignas(1024) extern __shared__ unsigned char shared_tiles[];
    __shared__ proj::Pipeline pipe;
    __shared__ float maxima[proj::consumer_warps];
    float result[short_columns];
    proj::bf16_tile(pipe, shared_tiles, &weight, &input, seeds[row] * proj::tile_rows, 0, 0,
                    width / proj::bf16_chunk_elements, short_rows, result);
    const int warp = threadIdx.x / proj::warp_size;
    if (warp < proj::consumer_warps) {
        const float score = __bfloat162float(__float2bfloat16_rn(result[row]));
        const float maximum = warp_max(isnan(score) ? -CUDART_INF_F : score);
        if (threadIdx.x % proj::warp_size == 0) maxima[warp] = maximum;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float bound = threshold[row];
        for (float maximum : maxima) bound = fmaxf(bound, maximum);
        threshold[row] = bound;
    }
}

__global__ void collect_short_tiles(const int* upper, const float* threshold,
                                    ShortCandidates* candidates) {
    const int token = blockIdx.x * proj::tile_rows + threadIdx.x;
    const int lane = threadIdx.x % proj::warp_size;
    bool tile_selected = false;
    #pragma unroll
    for (int row = 0; row < short_rows; ++row) {

        const bool selected = __int_as_float(upper[row * vocabulary + token]) >= threshold[row];
        const unsigned mask = __ballot_sync(0xffffffff, selected);
        if (lane == 0 && mask) atomicAdd(candidates->row_count + row, __popc(mask));
        tile_selected |= selected;
    }
    if (__syncthreads_or(tile_selected) && threadIdx.x == 0) {
        const int slot = atomicAdd(&candidates->tile_count, 1);
        candidates->tiles[slot] = blockIdx.x;
    }
}

__device__ inline bool replay_full_head(const ShortCandidates& candidates) {
    bool full = false;
    #pragma unroll
    for (int row = 0; row < short_rows; ++row) {
        const int count = candidates.row_count[row];
        full |= count <= 0 || count > slots;
    }
    return full;
}

__global__ __launch_bounds__(proj::threads, 1) void replay_short_tiles(
    const __grid_constant__ CUtensorMap weight, const __grid_constant__ CUtensorMap input,
    const int* upper, const float* threshold, const ShortCandidates* candidates, Winner* output) {
    const bool full = replay_full_head(*candidates);
    if (!full && blockIdx.x >= candidates->tile_count) return;
    const int tile = full ? blockIdx.x : candidates->tiles[blockIdx.x];
    const int first = tile * proj::tile_rows;
    alignas(1024) extern __shared__ unsigned char shared_tiles[];
    __shared__ proj::Pipeline pipe;
    __shared__ Winner partial[short_rows][proj::consumer_warps];
    float result[short_columns];
    proj::bf16_tile(pipe, shared_tiles, &weight, &input, first, 0, 0,
                    width / proj::bf16_chunk_elements, short_rows, result);
    const int warp = threadIdx.x / proj::warp_size;
    if (warp < proj::consumer_warps) {
        const int token = first + threadIdx.x;
        #pragma unroll
        for (int row = 0; row < short_rows; ++row) {
            const bool selected = full ||
                __int_as_float(upper[row * vocabulary + token]) >= threshold[row];
            const float score = __bfloat162float(__float2bfloat16_rn(result[row]));
            const auto winner = warp_best(selected && !isnan(score) ? Winner{score, token}
                                                                   : Winner{-CUDART_INF_F, -1});
            if (threadIdx.x % proj::warp_size == 0) partial[row][warp] = winner;
        }
    }
    __syncthreads();
    if (threadIdx.x < short_rows) {
        Winner winner{-CUDART_INF_F, -1};
        for (const auto other : partial[threadIdx.x]) winner = best(winner, other);
        output[threadIdx.x * short_tiles + blockIdx.x] = winner;
    }
}

__global__ void finish_short_rows(const ShortCandidates* candidates, const Winner* partial,
                                  Outputs winners) {
    const int count = replay_full_head(*candidates) ? short_tiles : candidates->tile_count;
    const auto* row = partial + blockIdx.x * short_tiles;
    Winner winner{-CUDART_INF_F, -1};
    for (int tile = threadIdx.x; tile < count; tile += blockDim.x) {
        winner = best(winner, row[tile]);
    }
    winner = block_best(warp_best(winner));
    if (threadIdx.x == 0) *winners.rows[blockIdx.x] = {winner.score, winner.token};
}

}
