#include "sglang/text/gqa_decode_step.h"

#include "sglang/text/gqa_decode_prepare.cuh"
#include "sglang/text/gqa_decode_split.cuh"

#include <cuda_runtime.h>

namespace sglang {
namespace {

using decoder::GqaDecodeStepLane;

static_assert(decoder::gqa_decode_step_max_lanes == gqa_decode::max_lanes);

struct StepBatch {
    GqaDecodeStepLane lanes[decoder::gqa_decode_step_max_lanes];
};

__device__ bool append_target(const GqaDecodeStepLane& lane, int position, Mxfp8KvCache* cache,
                              int* physical_page) {
    if (position < 0 || position >= text_attention_max_tokens) {
        return false;
    }
    *cache = *lane.read.cache;
    const Mxfp8PageTable pages = *lane.read.pages;
    const int page_index = position / mxfp8_page_size;
    if (page_index >= pages.page_count || !pages.device_pages) {
        return false;
    }
    *physical_page = pages.device_pages[page_index];
    return *physical_page >= 0 && *physical_page < cache->page_count &&
           cache->key_payload && cache->value_payload && cache->key_scales && cache->value_scales;
}

__global__ void __launch_bounds__(mxfp8_head_dim)
prepare_kernel(const __grid_constant__ StepBatch batch, const __nv_bfloat16* q_norm,
               const __nv_bfloat16* k_norm) {
    const GqaDecodeStepLane& lane = batch.lanes[blockIdx.y];
    const int position = *lane.read.position;
    if (position < 0 || position >= text_attention_max_tokens) {
        return;
    }
    const auto position_of = [position](int) { return position; };
    const int head = blockIdx.x;
    if (head < mxfp8_query_heads) {
        gqa_decode::normalize_rotate(lane.q_gate + head * 2 * mxfp8_head_dim, q_norm,
                                     lane.rope_queries + head * mxfp8_head_dim,
                                     position_of);
        return;
    }
    const int kv_head = head - mxfp8_query_heads;
    __nv_bfloat16* const rope_keys = lane.rope_keys + kv_head * mxfp8_head_dim;
    gqa_decode::normalize_rotate(lane.keys + kv_head * mxfp8_head_dim, k_norm, rope_keys,
                                 position_of);
    __syncthreads();
    Mxfp8KvCache cache;
    int physical_page = 0;
    if (!append_target(lane, position, &cache, &physical_page)) {
        return;
    }
    const int channel = threadIdx.x;
    gqa_decode::append_channel(
        cache, physical_page, position % mxfp8_page_size, kv_head, channel,
        __bfloat162float(rope_keys[channel]),
        __bfloat162float(lane.values[kv_head * mxfp8_head_dim + channel]));
}

bool step_lane_valid(const GqaDecodeStepLane& lane) {
    return lane.read.cache && lane.read.pages && lane.read.position && lane.q_gate && lane.keys
        && lane.values && lane.rope_keys && lane.rope_queries
        && lane.read.buffers.queries == lane.rope_queries
        && gqa_decode::valid_buffers(lane.read.buffers);
}

}

cudaError_t decoder::decode_gqa_step(const GqaDecodeStepLane* lanes, int count,
                                     const __nv_bfloat16* q_norm,
                                     const __nv_bfloat16* k_norm, cudaStream_t stream) {
    if (!lanes || count < 1 || count > gqa_decode_step_max_lanes || !q_norm || !k_norm) {
        return cudaErrorInvalidValue;
    }
    StepBatch batch{};
    BoundPagedDecodeLane reads[gqa_decode_step_max_lanes];
    const __nv_bfloat16* gates[gqa_decode_step_max_lanes];
    for (int index = 0; index < count; ++index) {
        if (!step_lane_valid(lanes[index])) {
            return cudaErrorInvalidValue;
        }
        batch.lanes[index] = lanes[index];
        reads[index] = lanes[index].read;
        gates[index] = lanes[index].q_gate;
    }
    constexpr int prepare_heads = mxfp8_query_heads + mxfp8_kv_heads;
    prepare_kernel<<<dim3(prepare_heads, count), mxfp8_head_dim, 0, stream>>>(
        batch, q_norm, k_norm);
    if (const cudaError_t error = cudaGetLastError(); error != cudaSuccess) {
        return error;
    }
    return gqa_decode::decode(reads, gates, count, stream);
}

}
