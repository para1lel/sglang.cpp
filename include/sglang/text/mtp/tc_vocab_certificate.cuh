#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include "tc_vocab.h"

namespace sglang::decoder::tc_vocab {

using Bf16 = __nv_bfloat16;
constexpr int vocabulary = 248320;
constexpr int width = 5120;
constexpr int rows = tc_vocab_rows;
constexpr int tile = 1024;
constexpr int tiles = (vocabulary + tile - 1) / tile;
constexpr int threads = 256;
constexpr int refine_items = 64;
constexpr int refine_tiles = 256;
constexpr int slots = refine_items * refine_tiles;
static_assert(width * 128 * 128 < (1LL << 31));

struct Stats {
    float scale;
    float approx_norm;
    float error_norm;
    float value_norm;
};

struct Winner {
    float score;
    int token;
};

struct Inputs {
    const Bf16* rows[tc_vocab::rows];
};

struct Outputs {
    TextLogitCandidate* rows[tc_vocab::rows];
};

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

__device__ Interval interval(int raw, const Stats& weight, const Stats& input) {
    if (!(weight.scale > 0 && input.scale > 0)) {
        return {-CUDART_INF_F, CUDART_INF_F};
    }
    const double product = static_cast<double>(weight.scale) * input.scale;
    const float weight_error = __fmul_ru(weight.error_norm, input.value_norm);
    const float input_error = __fmul_ru(weight.approx_norm, input.error_norm);
    const float norm_product = __fmul_ru(weight.value_norm, input.value_norm);
    const float legacy_error = __fmaf_ru(norm_product, 0x1p-16f, 0x1p-100f);
    const float error = __fadd_ru(__fadd_ru(weight_error, input_error), legacy_error);
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

}
