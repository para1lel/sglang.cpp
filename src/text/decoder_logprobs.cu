#include "sglang/text/decoder_logprobs.h"

#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_reduce.cuh>
#include <math_constants.h>

#include <cfloat>
#include <cstdint>

namespace sglang::decoder {
namespace {

constexpr int threads = 256;
constexpr int items = 4;
constexpr int tile_size = threads * items;
constexpr int max_vocabulary = threads * tile_size;
constexpr int token_bits = 18;
constexpr unsigned token_mask = (1u << token_bits) - 1;
static_assert(max_vocabulary == token_mask + 1);
using Key = std::uint64_t;
using Sort = cub::BlockRadixSort<Key, threads, items>;
using FloatReduce = cub::BlockReduce<float, threads>;
using KeyReduce = cub::BlockReduce<Key, threads>;

struct Tile {
    float maximum;
    float mass;
    Key ranked[decoder_max_generated_top_logprobs];
};

struct Scratch {
    Tile tiles[threads];
    GeneratedLogprobReadback result;
};

__device__ Key rank_key(unsigned bits, unsigned token) {
    if ((bits & 0x7fffu) == 0) {
        bits = 0;
    }
    const unsigned ordered = bits ^ (bits & 0x8000u ? 0xffffu : 0x8000u);
    return (Key{ordered} << token_bits) | (token_mask - token);
}

__device__ int token_id(Key key) {
    return token_mask - (key & token_mask);
}

__global__ void measure_tiles(const __nv_bfloat16* logits, int vocabulary, int top, Tile* tiles) {
    __shared__ union {
        FloatReduce::TempStorage reduce;
        Sort::TempStorage sort;
    } temporary;
    __shared__ float maximum;
    __shared__ int invalid;
    float scores[items];
    Key keys[items];
    float local_max = -CUDART_INF_F;
    bool bad = false;
    const int base = blockIdx.x * tile_size + threadIdx.x;
    for (int item = 0; item < items; ++item) {
        const int token = base + item * threads;
        const auto raw = token < vocabulary ? logits[token] : __float2bfloat16(-CUDART_INF_F);
        const float score = __bfloat162float(raw);
        scores[item] = score;
        if (top) {
            keys[item] = token < vocabulary ? rank_key(__bfloat16_as_ushort(raw), token) : 0;
        }
        bad |= isnan(score) || score == CUDART_INF_F;
        local_max = fmaxf(local_max, score);
    }
    const int bad_count = __syncthreads_count(bad);
    const float best = FloatReduce(temporary.reduce).Reduce(local_max, cuda::maximum<>{});
    if (threadIdx.x == 0) {
        maximum = best;
        invalid = bad_count;
    }
    __syncthreads();
    float mass = 0;
    if (isfinite(maximum)) {
        for (float score : scores) {
            mass += expf(score - maximum);
        }
    }
    const float tile_mass = FloatReduce(temporary.reduce).Sum(mass);
    auto& tile = tiles[blockIdx.x];
    if (threadIdx.x == 0) {
        tile.maximum = invalid ? CUDART_INF_F : maximum;
        tile.mass = tile_mass;
    }
    if (top) {
        __syncthreads();
        Sort(temporary.sort).SortDescendingBlockedToStriped(keys, 0, token_bits + 16);
        if (threadIdx.x < top) {
            tile.ranked[threadIdx.x] = keys[0];
        }
    }
}

__device__ float log_probability(float score, float maximum, float log_mass) {
    return fmaxf(-FLT_MAX, (score - maximum) - log_mass);
}

__global__ void finish_row(const __nv_bfloat16* logits, int vocabulary,
                           const TextLogitCandidate* winner, int count, int top, Scratch* scratch) {
    __shared__ union {
        FloatReduce::TempStorage reduce;
        KeyReduce::TempStorage rank;
    } temporary;
    __shared__ float maximum;
    __shared__ float log_mass;
    __shared__ Key best;
    const int index = threadIdx.x;
    const auto* tile = index < count ? scratch->tiles + index : nullptr;
    const float tile_max = tile ? tile->maximum : -CUDART_INF_F;
    const float row_max = FloatReduce(temporary.reduce).Reduce(tile_max, cuda::maximum<>{});
    if (index == 0) {
        maximum = row_max;
        scratch->result.selected = {-1, 0.0f};
    }
    __syncthreads();
    const int chosen = winner->token_id;
    if (!isfinite(maximum) || chosen < 0 || chosen >= vocabulary) {
        return;
    }
    const float mass = isfinite(tile_max) ? tile->mass * expf(tile_max - maximum) : 0.0f;
    const float total = FloatReduce(temporary.reduce).Sum(mass);
    if (index == 0) {
        log_mass = logf(total);
        scratch->result.selected = {
            chosen, log_probability(__bfloat162float(logits[chosen]), maximum, log_mass)};
    }
    __syncthreads();
    int cursor = 0;
    for (int rank = 0; rank < top; ++rank) {
        const Key candidate = tile ? tile->ranked[cursor] : 0;
        const Key key = KeyReduce(temporary.rank).Reduce(candidate, cuda::maximum<>{});
        if (index == 0) {
            best = key;
            const int token = token_id(key);
            scratch->result.alternatives[rank] = {
                token, log_probability(__bfloat162float(logits[token]), maximum, log_mass)};
        }
        __syncthreads();
        cursor += candidate == best;
        __syncthreads();
    }
}

}

std::size_t generated_logprob_workspace_bytes() { return sizeof(Scratch); }

cudaError_t enqueue_generated_logprobs(const __nv_bfloat16* logits, int vocabulary,
                                       const TextLogitCandidate* winner, int top, void* workspace,
                                       GeneratedLogprobReadback* host, cudaStream_t stream) {
    if (!logits || !winner || !workspace || !host || vocabulary < 1 ||
        vocabulary > max_vocabulary || top < 0 || top > decoder_max_generated_top_logprobs ||
        top > vocabulary) {
        return cudaErrorInvalidValue;
    }
    auto* scratch = static_cast<Scratch*>(workspace);
    const int count = (vocabulary + tile_size - 1) / tile_size;
    auto status = cudaMemsetAsync(&scratch->result, 0, sizeof(scratch->result), stream);
    if (status != cudaSuccess) return status;
    measure_tiles<<<count, threads, 0, stream>>>(logits, vocabulary, top, scratch->tiles);
    status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    finish_row<<<1, threads, 0, stream>>>(logits, vocabulary, winner, count, top, scratch);
    status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    return cudaMemcpyAsync(host, &scratch->result, offsetof(GeneratedLogprobReadback, alternatives)
        + top * sizeof(DecoderLogprob), cudaMemcpyDeviceToHost, stream);
}

}
