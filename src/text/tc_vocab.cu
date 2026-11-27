#include "sglang/text/mtp/tc_vocab_launch.h"

#include <math_constants.h>

namespace sglang::decoder::tc_vocab {

constexpr int threads = 256;

__device__ double sum_up(double value) {
    for (int offset = 16; offset; offset /= 2) {
        value = __dadd_ru(value, __shfl_down_sync(0xffffffff, value, offset));
    }
    return value;
}

__device__ float norm_up(double sum, double* totals) {
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    sum = sum_up(sum);
    if (lane == 0) {
        totals[warp] = sum;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int index = 1; index < threads / 32; ++index) {
            sum = __dadd_ru(sum, totals[index]);
        }
        totals[0] = __dsqrt_ru(sum);
    }
    __syncthreads();
    const float norm = __double2float_ru(totals[0]);
    __syncthreads();
    return norm;
}

__global__ void pack_inputs(Inputs inputs, Bf16* output) {
    const int row = blockIdx.x;
    auto* destination = output + row * width;
    for (int column = threadIdx.x; column < width; column += threads) {
        destination[column] = inputs.rows[row][column];
    }
}

__global__ void shared_stats(const Bf16* original, const std::uint8_t* quantized,
                             const float* maxima, Stats* output) {
    const int row = blockIdx.x;
    const auto offset = static_cast<std::size_t>(row) * width;
    const auto* bytes = reinterpret_cast<const std::int8_t*>(quantized) + offset;
    const float maximum = maxima[row];
    const float scale = maximum == 0 ? 1.0f : maximum / 127.0f;
    if (!(scale > 0 && maximum <= 0x1p32f)) {
        if (threadIdx.x == 0) {
            output[row] = {};
        }
        return;
    }
    __shared__ double limits[threads / 32];
    double approx_norm = 0, error_norm = 0, value_norm = 0;
    for (int column = threadIdx.x; column < width; column += threads) {
        const double value = __bfloat162float(original[offset + column]);
        const double scaled = bytes[column] * static_cast<double>(scale);
        const double error = fmax(__dsub_ru(value, scaled), __dsub_ru(scaled, value));
        approx_norm = __dadd_ru(approx_norm, __dmul_ru(scaled, scaled));
        error_norm = __dadd_ru(error_norm, __dmul_ru(error, error));
        value_norm = __dadd_ru(value_norm, __dmul_ru(value, value));
    }
    const float approx = norm_up(approx_norm, limits);
    const float error = norm_up(error_norm, limits);
    const float value = norm_up(value_norm, limits);
    if (threadIdx.x == 0) {
        output[row] = {scale, approx, error, value};
    }
}

struct Interval {
    float lower;
    float upper;
};

__device__ Interval interval(int raw, const Stats& weight, const Stats& input,
                             float rounding_error = 0x1p-16f) {
    if (!(weight.scale > 0 && input.scale > 0)) {
        return {-CUDART_INF_F, CUDART_INF_F};
    }
    const double product = static_cast<double>(weight.scale) * input.scale;
    const float weight_error = __fmul_ru(weight.error_norm, input.value_norm);
    const float input_error = __fmul_ru(weight.approx_norm, input.error_norm);
    const float norm_product = __fmul_ru(weight.value_norm, input.value_norm);
    const float rounding = __fmaf_ru(norm_product, rounding_error, 0x1p-100f);
    const float error = __fadd_ru(__fadd_ru(weight_error, input_error), rounding);
    const float lower = __double2float_rd(__dsub_rd(__dmul_rd(raw, product), error));
    const float upper = __double2float_ru(__dadd_ru(__dmul_ru(raw, product), error));
    return {__bfloat162float(__float2bfloat16_rn(lower)),
            __bfloat162float(__float2bfloat16_rn(upper))};
}

__device__ float block_max(float score) {
    __shared__ float maxima[threads / 32];
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    for (int offset = 16; offset; offset /= 2) {
        score = fmaxf(score, __shfl_down_sync(0xffffffff, score, offset));
    }
    if (lane == 0) {
        maxima[warp] = score;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int index = 1; index < threads / 32; ++index) {
            score = fmaxf(score, maxima[index]);
        }
    }
    return score;
}

__global__ void lower_bounds(int* scores, const Stats* weight, const Stats* inputs, float* lower) {
    const int row = blockIdx.y;
    const auto offset = static_cast<std::size_t>(row) * vocabulary;
    float maximum = -CUDART_INF_F;
    for (int index = threadIdx.x; index < tile; index += threads) {
        const int token = blockIdx.x * tile + index;
        if (token < vocabulary) {
            const auto value = interval(scores[offset + token], weight[token], inputs[row]);
            maximum = fmaxf(maximum, value.lower);
            scores[offset + token] = __float_as_int(value.upper);
        }
    }
    maximum = block_max(maximum);
    if (threadIdx.x == 0) {
        lower[row * tiles + blockIdx.x] = maximum;
    }
}

__global__ void thresholds(const float* lower, float* threshold) {
    const int row = blockIdx.x;
    const int item = threadIdx.x;
    const float value = block_max(item < tiles ? lower[row * tiles + item] : -CUDART_INF_F);
    if (item == 0) {
        threshold[row] = value;
    }
}

__global__ void candidates(const int* bounds, const float* threshold, int* count, int* ids) {
    const int row = blockIdx.y;
    const int token = blockIdx.x * blockDim.x + threadIdx.x;
    const auto offset = static_cast<std::size_t>(row) * vocabulary;
    const bool selected = token < vocabulary &&
        __int_as_float(bounds[offset + token]) >= threshold[row];
    const unsigned mask = __ballot_sync(0xffffffff, selected);
    if (selected) {
        const int leader = __ffs(mask) - 1;
        const int lane = threadIdx.x % 32;
        const int first = lane == leader ? atomicAdd(count + row, __popc(mask)) : 0;
        const int slot = __shfl_sync(mask, first, leader) + __popc(mask & ((1u << lane) - 1u));
        if (slot < slots) {
            ids[row * slots + slot] = token;
        }
    }
}

__device__ Winner best(Winner first, Winner second) {
    if (second.token >= 0 &&
        (first.token < 0 || second.score > first.score ||
         (second.score == first.score && second.token < first.token))) {
        return second;
    }
    return first;
}

__device__ Winner block_best(Winner winner) {
    __shared__ Winner partial[threads / 32];
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    if (lane == 0) {
        partial[warp] = winner;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int index = 1; index < threads / 32; ++index) {
            winner = best(winner, partial[index]);
        }
    }
    return winner;
}

__device__ float dot_pair(unsigned left, unsigned right, float sum) {
    sum = fmaf(__uint_as_float(left << 16), __uint_as_float(right << 16), sum);
    return fmaf(__uint_as_float(left & 0xffff0000u),
                __uint_as_float(right & 0xffff0000u), sum);
}

__device__ float score_exact(const Bf16* weight, const Bf16* input) {
    const int lane = threadIdx.x % 32;
    float first = 0.0f, second = 0.0f;
    for (int col = lane * 8; col < width; col += 256) {
        const auto lhs = *reinterpret_cast<const uint4*>(weight + col);
        const auto rhs = *reinterpret_cast<const uint4*>(input + col);
        first = dot_pair(lhs.x, rhs.x, first);
        second = dot_pair(lhs.y, rhs.y, second);
        first = dot_pair(lhs.z, rhs.z, first);
        second = dot_pair(lhs.w, rhs.w, second);
    }
    float sum = first + second;
    for (int offset = 16; offset; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    return __bfloat162float(__float2bfloat16_rn(sum));
}

__global__ void refine(const Bf16* weight, Inputs inputs, const int* count, const int* ids,
                       Winner* results) {
    const int row = blockIdx.y;
    const int warp = threadIdx.x / 32;
    const bool fallback = count[row] <= 0 || count[row] > slots;
    const int start = blockIdx.x * (fallback ? tile : refine_items);
    const int end = fallback ? min(vocabulary - start, tile)
                             : min(count[row] - start, refine_items);
    auto winner = Winner{-CUDART_INF_F, -1};
    if (end > 0) {
        for (int index = warp; index < end; index += threads / 32) {
            const int token = fallback ? start + index : ids[row * slots + start + index];
            const float score = score_exact(weight + static_cast<std::size_t>(token) * width,
                                            inputs.rows[row]);
            if (!isnan(score)) {
                winner = best(winner, {score, token});
            }
        }
    }
    winner = block_best(winner);
    if (threadIdx.x == 0) {
        results[row * refine_tiles + blockIdx.x] = winner;
    }
}

__global__ void finish(const Winner* tiles_results, Outputs winners) {
    const int index = threadIdx.x;
    auto winner = tiles_results[blockIdx.x * refine_tiles + index];
    for (int offset = 16; offset; offset /= 2) {
        winner = best(winner, {__shfl_down_sync(0xffffffff, winner.score, offset),
                               __shfl_down_sync(0xffffffff, winner.token, offset)});
    }
    winner = block_best(winner);
    if (index == 0) {
        *winners.rows[blockIdx.x] = {winner.score, winner.token};
    }
}

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

constexpr int short_bf16_shared = proj::stages * proj::stage_bytes<short_columns>;
static_assert(vocabulary % proj::tile_rows == 0 && width % proj::bf16_chunk_elements == 0);

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

namespace sglang::decoder {
namespace {

using namespace tc_vocab;

static_assert(vocabulary % decode_proj::tile_rows == 0 && width % score_chunk == 0);
static_assert(tc_vocab_rows <= score_columns && tc_vocab_rows > 16);
static_assert(tc_vocab_wide_rows <= wide_columns && tc_vocab_wide_rows > 32);
static_assert(tc_vocab_narrow_rows <= 8);

template <int Columns>
__device__ void store_scores(const decode_proj::DecodePipeline<score_stages>& pipe,
                             int first_token, int* scores, int count) {
    using namespace decode_proj;
    wait_barrier(&pipe.accum_ready[0], 0);
    fence_after_sync();
    const int token = first_token + threadIdx.x;
    const unsigned source = pipe.tmem + (static_cast<unsigned>(threadIdx.x / warp_size *
                                                               warp_size) << 16);
    if constexpr (Columns == narrow_columns) {
        unsigned values[8];
        load_tmem8(source, values);
        for (int row = 0; row < count; ++row) {
            scores[row * vocabulary + token] = static_cast<int>(values[row]);
        }
    } else if constexpr (Columns == score_columns) {
        unsigned values[24];
        load_tmem16(source, values);
        load_tmem8(source + 16, values + 16);
        #pragma unroll
        for (int row = 0; row < tc_vocab_rows; ++row) {
            scores[row * vocabulary + token] = static_cast<int>(values[row]);
        }
    } else {
        unsigned values[40];
        load_tmem16(source, values);
        load_tmem16(source + 16, values + 16);
        load_tmem8(source + 32, values + 32);
        #pragma unroll
        for (int row = 0; row < tc_vocab_wide_rows; ++row) {
            scores[row * vocabulary + token] = static_cast<int>(values[row]);
        }
    }
    fence_before_sync();
}

template <int Columns>
__global__ __launch_bounds__(decode_proj::threads, 2) void vocabulary_scores(
    const __grid_constant__ CUtensorMap weight, const __grid_constant__ CUtensorMap input,
    int* scores, int count) {
    using namespace decode_proj;
    alignas(1024) extern __shared__ unsigned char tiles[];
    __shared__ DecodePipeline<score_stages> pipe;
    const int warp = threadIdx.x / warp_size;
    const int first_token = blockIdx.x * tile_rows;
    if (threadIdx.x == 0) {
        init_pipeline(pipe);
    }
    if (warp == mma_warp) {
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                     :: "r"(shared_address(&pipe.tmem)), "r"(tmem_columns<Columns, 1>) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
    }
    __syncthreads();
    const auto shared = shared_address(tiles);
    if (warp < consumer_warps) {
        store_scores<Columns>(pipe, first_token, scores, count);
    } else if (warp == mma_warp && threadIdx.x % warp_size == 0) {
        multiply_scores<Columns>(pipe, shared);
    } else if (warp == copy_warp && threadIdx.x % warp_size == 0) {
        produce<score_chunk, Columns>(pipe, shared, &weight, &input, first_token, 0, 0,
                                      width / score_chunk);
    }
    __syncthreads();
    if (warp == mma_warp) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                     :: "r"(pipe.tmem), "r"(tmem_columns<Columns, 1>) : "memory");
    }
}

}

namespace tc_vocab {

cudaError_t configure_packet_scores(int count) {
    if (count <= tc_vocab_narrow_rows) {
        return cudaFuncSetAttribute(vocabulary_scores<narrow_columns>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, score_shared_bytes<narrow_columns>);
    }
    if (count == tc_vocab_wide_rows) {
        return cudaFuncSetAttribute(vocabulary_scores<wide_columns>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, score_shared_bytes<wide_columns>);
    }
    return cudaFuncSetAttribute(vocabulary_scores<score_columns>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, score_shared_bytes<score_columns>);
}

cudaError_t configure_short_screen() {
    return cudaFuncSetAttribute(screen_short_scores,
        cudaFuncAttributeMaxDynamicSharedMemorySize, score_shared_bytes<short_columns>);
}

cudaError_t configure_short_seeds() {
    return cudaFuncSetAttribute(improve_short_thresholds,
        cudaFuncAttributeMaxDynamicSharedMemorySize, short_bf16_shared);
}

cudaError_t configure_short_replay() {
    return cudaFuncSetAttribute(replay_short_tiles,
        cudaFuncAttributeMaxDynamicSharedMemorySize, short_bf16_shared);
}

cudaError_t launch_pack_inputs(int count, const Inputs& inputs, Bf16* output, cudaStream_t stream) {
    pack_inputs<<<count, threads, 0, stream>>>(inputs, output);
    return cudaGetLastError();
}

cudaError_t launch_shared_stats(int count, const Bf16* original, const std::uint8_t* quantized,
                               const float* maxima, Stats* output, cudaStream_t stream) {
    shared_stats<<<count, threads, 0, stream>>>(original, quantized, maxima, output);
    return cudaGetLastError();
}

cudaError_t launch_vocabulary_scores(const CUtensorMap& weight, const CUtensorMap& input,
                                    int* scores, int count, cudaStream_t stream) {
    constexpr int blocks = vocabulary / decode_proj::tile_rows;
    if (count <= tc_vocab_narrow_rows) {
        vocabulary_scores<narrow_columns>
            <<<blocks, decode_proj::threads, score_shared_bytes<narrow_columns>, stream>>>(
                weight, input, scores, count);
    } else if (count == tc_vocab_wide_rows) {
        vocabulary_scores<wide_columns>
            <<<blocks, decode_proj::threads, score_shared_bytes<wide_columns>, stream>>>(
                weight, input, scores, count);
    } else {
        vocabulary_scores<score_columns>
            <<<blocks, decode_proj::threads, score_shared_bytes<score_columns>, stream>>>(
                weight, input, scores, count);
    }
    return cudaGetLastError();
}

cudaError_t launch_lower_bounds(int count, int* scores, const Stats* weight, const Stats* inputs,
                               float* lower, cudaStream_t stream) {
    lower_bounds<<<dim3(tiles, count), threads, 0, stream>>>(scores, weight, inputs, lower);
    return cudaGetLastError();
}

cudaError_t launch_thresholds(int count, const float* lower, float* threshold,
                              cudaStream_t stream) {
    thresholds<<<count, threads, 0, stream>>>(lower, threshold);
    return cudaGetLastError();
}

cudaError_t launch_candidates(int count, const int* bounds, const float* threshold,
                              int* count_per_row, int* ids, cudaStream_t stream) {
    candidates<<<dim3((vocabulary + threads - 1) / threads, count), threads, 0, stream>>>(
        bounds, threshold, count_per_row, ids);
    return cudaGetLastError();
}

cudaError_t launch_refine(int count, const Bf16* weight, const Inputs& inputs,
                          const int* count_per_row, const int* ids, Winner* partial,
                          cudaStream_t stream) {
    refine<<<dim3(refine_tiles, count), threads, 0, stream>>>(
        weight, inputs, count_per_row, ids, partial);
    return cudaGetLastError();
}

cudaError_t launch_finish(int count, const Winner* partial, const Outputs& outputs,
                          cudaStream_t stream) {
    finish<<<count, threads, 0, stream>>>(partial, outputs);
    return cudaGetLastError();
}

cudaError_t launch_screen_short(const CUtensorMap& weight, const CUtensorMap& input,
                               const Stats* weight_stats, const Stats* input_stats,
                               int* upper, float* lower, cudaStream_t stream) {
    screen_short_scores<<<short_tiles, decode_proj::threads,
                          score_shared_bytes<short_columns>, stream>>>(
        weight, input, weight_stats, input_stats, upper, lower);
    return cudaGetLastError();
}

cudaError_t launch_select_short(const float* lower, float* threshold, int* seeds,
                                cudaStream_t stream) {
    select_short_seeds<<<short_rows, threads, 0, stream>>>(lower, threshold, seeds);
    return cudaGetLastError();
}

cudaError_t launch_improve_short(const CUtensorMap& weight, const CUtensorMap& input,
                                 const int* seeds, float* threshold, cudaStream_t stream) {
    improve_short_thresholds<<<short_rows, decode_proj::threads, short_bf16_shared, stream>>>(
        weight, input, seeds, threshold);
    return cudaGetLastError();
}

cudaError_t launch_collect_short(const int* upper, const float* threshold,
                                 ShortCandidates* candidates, cudaStream_t stream) {
    collect_short_tiles<<<short_tiles, decode_proj::tile_rows, 0, stream>>>(
        upper, threshold, candidates);
    return cudaGetLastError();
}

cudaError_t launch_replay_short(const CUtensorMap& weight, const CUtensorMap& input,
                                const int* upper, const float* threshold,
                                const ShortCandidates* candidates, Winner* partial,
                                cudaStream_t stream) {
    replay_short_tiles<<<short_tiles, decode_proj::threads, short_bf16_shared, stream>>>(
        weight, input, upper, threshold, candidates, partial);
    return cudaGetLastError();
}

cudaError_t launch_finish_short(const ShortCandidates* candidates, const Winner* partial,
                               const Outputs& outputs, cudaStream_t stream) {
    finish_short_rows<<<short_rows, threads, 0, stream>>>(candidates, partial, outputs);
    return cudaGetLastError();
}

}

}
