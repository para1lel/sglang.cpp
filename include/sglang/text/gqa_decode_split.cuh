#pragma once

#include "sglang/decoder_cuda.cuh"
#include "sglang/mxfp8_gqa.h"

#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>
#include <math_constants.h>

namespace sglang::gqa_decode {

/// Packed decode lanes accepted by one split-KV launch. Independent of the graph chunk size.
inline constexpr int max_lanes = 64;
/// Upper bound on KV splits per lane; each split owns one partial per query head.
inline constexpr int max_splits = 16;
/// Tokens per pipeline chunk; splits own whole chunks.
inline constexpr int split_chunk_tokens = 64;
/// Minimum chunks per split before a lane uses another split.
inline constexpr int min_split_chunks = 4;

static_assert(mxfp8_page_size % split_chunk_tokens == 0);

/// Unnormalized split result for one query head. `maximum` is in the log2 domain.
/// A nonfinite `maximum` at split 0 after the combine requests an FP64 replay of the head.
struct alignas(16) Partial {
    float value[mxfp8_head_dim];
    float maximum;
    float mass;
};

static_assert(sizeof(Partial) * max_splits * mxfp8_query_heads <=
              decoder::paged_decode_scratch_bytes);

__host__ __device__ inline Partial* partials(void* scratch, int head) {
    return static_cast<Partial*>(scratch) + head * max_splits;
}

/// Number of splits used for a lane reading positions [0, position].
__host__ __device__ inline int split_count(int position) {
    const int chunks = position / split_chunk_tokens + 1;
    const int splits = (chunks + min_split_chunks - 1) / min_split_chunks;
    return splits < max_splits ? splits : max_splits;
}

/// Device-side lane check. Unreadable lanes are skipped and their output is left untouched.
__host__ __device__ inline bool readable(const decoder::PagedDecodeLane& lane) {
    const int last_page = lane.position / mxfp8_page_size;
    return lane.position >= 0 && lane.position < text_attention_max_tokens &&
           last_page < lane.cache.page_count && last_page < lane.pages.page_count &&
           lane.pages.device_pages && lane.cache.key_payload && lane.cache.value_payload &&
           lane.cache.key_scales && lane.cache.value_scales;
}

__device__ inline bool tensor_aligned(const decoder::PagedDecodeLane& lane) {
    const auto addresses = reinterpret_cast<std::uintptr_t>(lane.buffers.queries) |
        reinterpret_cast<std::uintptr_t>(lane.cache.key_payload) |
        reinterpret_cast<std::uintptr_t>(lane.cache.value_payload) |
        reinterpret_cast<std::uintptr_t>(lane.cache.key_scales) |
        reinterpret_cast<std::uintptr_t>(lane.cache.value_scales);
    return addresses % 16 == 0;
}

inline bool valid_buffers(const decoder::PagedDecodeBuffers& buffers) {
    const auto scratch = reinterpret_cast<std::uintptr_t>(buffers.scratch);
    return buffers.queries && buffers.output && scratch && scratch % alignof(Partial) == 0;
}

struct Batch {
    decoder::BoundPagedDecodeLane lanes[max_lanes];
};

struct TargetLane {
    decoder::PagedDecodeLane eager;
    decoder::BoundPagedDecodeLane bound;
};

/// Copies `count` lanes into `batch` after checking buffer pointers and alignment.
inline cudaError_t make_batch(const decoder::BoundPagedDecodeLane* lanes, int count,
                              Batch& batch) {
    if (!lanes || count < 1 || count > max_lanes) {
        return cudaErrorInvalidValue;
    }
    for (int index = 0; index < count; ++index) {
        const auto& lane = lanes[index];
        if (!valid_buffers(lane.buffers) || !lane.cache || !lane.pages || !lane.position) {
            return cudaErrorInvalidValue;
        }
        batch.lanes[index] = lane;
    }
    return cudaSuccess;
}

/// Launches the split-KV pass. Each lane reads positions [0, position] of its cache and
/// writes one Partial per query head and split; combine_head finishes the head.
cudaError_t launch(const Batch& batch, int count, cudaStream_t stream);
cudaError_t launch(TargetLane lane, cudaStream_t stream);

/// Split-KV attention, combine, FP64 replay of flagged heads, and optional output gates.
/// Null gates leave the attention values ungated.
cudaError_t decode(const decoder::BoundPagedDecodeLane* lanes, const __nv_bfloat16* const* gates,
                   int count, cudaStream_t stream);

#ifdef __CUDACC__
__device__ inline decoder::PagedDecodeLane resolve(TargetLane lane) {
    const auto& bound = lane.bound;
    return bound.position ? decoder::PagedDecodeLane{*bound.cache, *bound.pages, *bound.position,
                                                      bound.buffers} : lane.eager;
}

/// Combines the splits of one head with 256 threads (one channel each) and writes the output.
/// Returns false (block-uniform) and marks split 0 nonfinite when the head needs a replay.
__device__ inline bool combine_head(const decoder::PagedDecodeLane& lane, int head) {
    __shared__ float weights[max_splits];
    __shared__ float total;
    const int channel = static_cast<int>(threadIdx.x);
    const int splits = split_count(lane.position);
    auto* part = partials(lane.buffers.scratch, head);
    const float maximum = channel < splits ? part[channel].maximum : -CUDART_INF_F;
    bool invalid = __syncthreads_or(channel < splits && !isfinite(maximum));
    if (!invalid) {
        if (channel < 32) {
            float global = maximum;
            for (int offset = 16; offset > 0; offset /= 2) {
                global = fmaxf(global, __shfl_xor_sync(0xffffffffu, global, offset));
            }
            const float delta = maximum - global;
            const float weight = channel >= splits ? 0.0f :
                delta < -126.0f ? CUDART_NAN_F : exp2f(delta);
            float contribution = channel < splits ? weight * part[channel].mass : 0.0f;
            for (int offset = 16; offset > 0; offset /= 2) {
                contribution += __shfl_xor_sync(0xffffffffu, contribution, offset);
            }
            if (channel < max_splits) {
                weights[channel] = weight;
            }
            if (channel == 0) {
                total = contribution;
            }
        }
        __syncthreads();
        float numerator = 0.0f;
        for (int split = 0; split < splits; ++split) {
            numerator = fmaf(weights[split], part[split].value[channel], numerator);
        }
        const auto result = __float2bfloat16_rn(numerator / total);
        invalid = __syncthreads_or(!(total > 0.0f) || !isfinite(total) || !isfinite(numerator) ||
                                   !isfinite(__bfloat162float(result)));
        if (!invalid) {
            lane.buffers.output[head * mxfp8_head_dim + channel] = result;
            return true;
        }
    }
    if (channel == 0) {
        part[0].maximum = CUDART_NAN_F;
    }
    return false;
}
#endif

}
