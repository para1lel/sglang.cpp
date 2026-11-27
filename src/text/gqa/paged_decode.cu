#include "sglang/text/gqa/decode_kernels.cuh"
#include "sglang/text/gqa_verify_batch.h"
#include <algorithm>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <math_constants.h>

namespace sglang {
namespace {

using namespace gqa_device;

constexpr int decode_splits = decoder::paged_decode_partitions;
constexpr int split_tokens = decoder::paged_decode_max_visible / decode_splits;
constexpr int split_warps = 8;
constexpr int max_split_value_exponent = 160;

struct alignas(16) DecodePartial {
    float value[mxfp8_head_dim];
    float maximum;
    float mass;
};

static_assert(sizeof(DecodePartial) * decode_splits * mxfp8_query_heads ==
              decoder::paged_decode_scratch_bytes);

template <typename Lane>
struct PagedDecodeBatch {
    Lane lanes[decoder::paged_decode_lanes];
};

__device__ decoder::PagedDecodeLane resolve_lane(decoder::PagedDecodeLane lane) { return lane; }

__device__ decoder::PagedDecodeLane resolve_lane(decoder::BoundPagedDecodeLane lane) {
    return {*lane.cache, *lane.pages, *lane.position, lane.buffers};
}

__device__ DecodePartial* decode_partials(decoder::PagedDecodeBuffers buffers, int head) {
    return static_cast<DecodePartial*>(buffers.scratch) + head * decode_splits;
}

template <typename Lane>
__global__ void split_decode_attention(PagedDecodeBatch<Lane> batch) {
    const auto input = resolve_lane(batch.lanes[blockIdx.z]);
    if (!decoder::paged_decode_position(input.position)) {
        return;
    }
    __shared__ float weights[split_tokens];
    __shared__ float maxima[split_warps];
    __shared__ float masses[split_warps];
    __shared__ float values[split_warps][mxfp8_head_dim];
    __shared__ float maximum;
    const int head = blockIdx.y;
    if (head >= gqa_query_heads(input.cache.kv_heads)) return;
    const int thread = threadIdx.x;
    const int warp = thread / mxfp8_group_size;
    const int lane = thread % mxfp8_group_size;
    const int kv_head = head / queries_per_kv_head;
    const int visible = input.position + 1;
    const int pages = (visible + mxfp8_page_size - 1) / mxfp8_page_size;
    const int first = blockIdx.x * pages / decode_splits * mxfp8_page_size;
    const int end = min(visible, (blockIdx.x + 1) * pages / decode_splits * mxfp8_page_size);
    const int count = end - first;
    auto& partial = decode_partials(input.buffers, head)[blockIdx.x];
    float query[mxfp8_groups];
    load_query(gqa_query_heads(input.cache.kv_heads), input.buffers.queries, head, lane, query);
    float local_max = -CUDART_INF_F;
    bool invalid = false;
    for (int index = warp; index < count; index += split_warps) {
        const int token = first + index;
        const int page = input.pages.device_pages[token / mxfp8_page_size];
        const float score =
            attention_score(input.cache, page, token % mxfp8_page_size, kv_head, lane, query);
        if (lane == 0) {
            weights[index] = score;
            local_max = fmaxf(local_max, score);
            invalid |= !isfinite(score);
        }
    }
    if (lane == 0) {
        maxima[warp] = local_max;
    }
    if (__syncthreads_or(invalid)) {
        if (thread == 0) {
            partial.maximum = CUDART_NAN_F;
        }
        return;
    }
    if (warp == 0) {
        const float reduced = warp_max(lane < split_warps ? maxima[lane] : -CUDART_INF_F);
        if (lane == 0) {
            maximum = reduced;
        }
    }
    __syncthreads();
    float sum[mxfp8_groups]{};
    float mass = 0.0f;
    for (int index = warp; index < count; index += split_warps) {
        const int token = first + index;
        const int slot = token % mxfp8_page_size;
        const int page = input.pages.device_pages[token / mxfp8_page_size];
        const auto data = payload_offset(input.cache.kv_heads, page, slot, kv_head) + lane;
        const auto scale = scale_offset(input.cache.kv_heads, page, slot, kv_head, 0);
        const float weight = expf(weights[index] - maximum);
        mass += weight;
        #pragma unroll
        for (int group = 0; group < mxfp8_groups; ++group) {
            const auto exponent = input.cache.value_scales[scale + group];
            const auto payload = input.cache.value_payload[data + group * mxfp8_group_size];
            invalid |= exponent > max_split_value_exponent;
            sum[group] = fmaf(weight, decode_e8m0(payload, exponent), sum[group]);
        }
    }
    #pragma unroll
    for (int group = 0; group < mxfp8_groups; ++group) {
        values[warp][group * mxfp8_group_size + lane] = sum[group];
        invalid |= !isfinite(sum[group]);
    }
    if (lane == 0) {
        masses[warp] = mass;
    }
    if (__syncthreads_or(invalid)) {
        if (thread == 0) {
            partial.maximum = CUDART_NAN_F;
        }
        return;
    }
    float value = 0.0f;
    #pragma unroll
    for (int index = 0; index < split_warps; ++index) {
        value += values[index][thread];
    }
    partial.value[thread] = value;
    if (warp == 0) {
        const float total = warp_sum(lane < split_warps ? masses[lane] : 0.0f);
        if (lane == 0) {
            partial.maximum = maximum;
            partial.mass = total;
        }
    }
}

template <typename Lane>
__global__ void merge_decode_attention(PagedDecodeBatch<Lane> batch) {
    const auto input = resolve_lane(batch.lanes[blockIdx.z]);
    if (!decoder::paged_decode_position(input.position)) {
        return;
    }
    __shared__ float weights[decode_splits];
    __shared__ float mass;
    const int head = blockIdx.x;
    if (head >= gqa_query_heads(input.cache.kv_heads)) {
        return;
    }
    const int channel = threadIdx.x;
    auto* partials = decode_partials(input.buffers, head);
    const float maximum = channel < decode_splits ? partials[channel].maximum : -CUDART_INF_F;
    if (__syncthreads_or(channel < decode_splits && !isfinite(maximum))) {
        if (channel == 0) {
            partials[0].maximum = CUDART_NAN_F;
        }
        return;
    }
    if (channel < mxfp8_group_size) {
        const float global_max = __shfl_sync(warp_mask, warp_max(maximum), 0);
        float contribution = 0.0f;
        if (channel < decode_splits) {
            const float weight = expf(maximum - global_max);
            weights[channel] = weight;
            contribution = weight * partials[channel].mass;
        }
        const float total = warp_sum(contribution);
        if (channel == 0) {
            mass = total;
        }
    }
    __syncthreads();
    float numerator = 0.0f;
    #pragma unroll
    for (int partition = 0; partition < decode_splits; ++partition) {
        numerator = fmaf(weights[partition], partials[partition].value[channel], numerator);
    }
    const auto result = __float2bfloat16_rn(numerator / mass);
    const bool invalid = !(mass > 0.0f) || !isfinite(mass) || !isfinite(numerator) ||
                         !isfinite(__bfloat162float(result));
    if (__syncthreads_or(invalid)) {
        if (channel == 0) {
            partials[0].maximum = CUDART_NAN_F;
        }
        return;
    }
    input.buffers.output[head * mxfp8_head_dim + channel] = result;
}

}

namespace gqa_device {

__device__ void wide_decode_head(decoder::PagedDecodeLane input, int head) {
    constexpr int tile = mxfp8_group_size;
    __shared__ double scores[tile];
    __shared__ double weights[tile];
    __shared__ double history;
    __shared__ double mass;
    const int thread = threadIdx.x;
    const int warp = thread / tile;
    const int lane = thread % tile;
    const int kv_head = head / queries_per_kv_head;
    float query[mxfp8_groups];
    load_query(gqa_query_heads(input.cache.kv_heads), input.buffers.queries, head, lane, query);
    double maximum = -CUDART_INF;
    double value = 0.0;
    double denominator = 0.0;
    for (int first = 0; first <= input.position; first += tile) {
        const int count = min(tile, input.position + 1 - first);
        const int page = input.pages.device_pages[first / mxfp8_page_size];
        const int slot = first % mxfp8_page_size;
        for (int index = warp; index < count; index += split_warps) {
            double dot = 0.0;
            const auto data =
                payload_offset(input.cache.kv_heads, page, slot + index, kv_head) + lane;
            const auto scale = scale_offset(input.cache.kv_heads, page, slot + index, kv_head, 0);
            #pragma unroll
            for (int group = 0; group < mxfp8_groups; ++group) {
                const auto exponent = input.cache.key_scales[scale + group];
                const auto payload = input.cache.key_payload[data + group * tile];
                dot += double(query[group]) * decode_e8m0(payload, exponent);
            }
            for (int delta = tile / 2; delta > 0; delta /= 2) {
                dot += __shfl_down_sync(warp_mask, dot, delta);
            }
            if (lane == 0) {
                scores[index] = dot * softmax_scale;
            }
        }
        __syncthreads();
        if (thread == 0) {
            double next = maximum;
            for (int index = 0; index < count; ++index) {
                next = fmax(next, scores[index]);
            }
            history = exp(maximum - next);
            denominator *= history;
            for (int index = 0; index < count; ++index) {
                weights[index] = exp(scores[index] - next);
                denominator += weights[index];
            }
            mass = denominator;
            maximum = next;
        }
        __syncthreads();
        value *= history;
        for (int index = 0; index < count; ++index) {
            const auto data =
                payload_offset(input.cache.kv_heads, page, slot + index, kv_head) + thread;
            const auto scale =
                scale_offset(input.cache.kv_heads, page, slot + index, kv_head, warp);
            const float decoded =
                decode_e8m0(input.cache.value_payload[data], input.cache.value_scales[scale]);
            value = fma(weights[index], double(decoded), value);
        }
        __syncthreads();
    }
    input.buffers.output[head * mxfp8_head_dim + thread] = __double2bfloat16(value / mass);
}

}

namespace {

using namespace gqa_device;

__global__ void bound_exact_decode_scores(PagedDecodeBatch<decoder::BoundPagedDecodeLane> batch) {
    const auto binding = batch.lanes[blockIdx.z];
    const int position = *binding.position;
    if (!decoder::paged_decode_position(position)) {
        exact_scores(*binding.cache, binding.pages->device_pages, position,
                     binding.buffers.queries, *binding.scores);
    }
}

template <typename Lane>
__global__ void replay_paged_decode(PagedDecodeBatch<Lane> batch) {
    const auto binding = batch.lanes[blockIdx.z];
    const auto input = resolve_lane(binding);
    const int head = blockIdx.x;
    if (head >= gqa_query_heads(input.cache.kv_heads)) {
        return;
    }
    if (decoder::paged_decode_position(input.position)) {
        if (blockIdx.y == 0 && !isfinite(decode_partials(input.buffers, head)[0].maximum)) {
            wide_decode_head(input, head);
        }
    } else if constexpr (std::is_same_v<Lane, decoder::BoundPagedDecodeLane>) {
        const auto workspace = *binding.scores;
        ordered_replay<false>(input.cache, input.pages.device_pages, input.position,
                             workspace.scores + head * workspace.capacity, nullptr,
                             input.buffers.output);
    }
}
constexpr int split_query_heads = 2;
constexpr int score_lanes = 8;
constexpr int groups_per_warp = mxfp8_group_size / score_lanes;
constexpr int score_passes = mxfp8_groups / groups_per_warp;
static_assert(queries_per_kv_head % split_query_heads == 0);
static_assert(groups_per_warp == 4 && mxfp8_groups % groups_per_warp == 0);

struct PairedSplit {
    float weights[split_tokens];
    float maxima[split_warps];
    float masses[split_warps];
    float values[split_warps][mxfp8_head_dim];
    float maximum;
};

using PairedQuery = float[split_query_heads][score_passes][groups_per_warp];

__device__ void load_paired_query(int query_heads, const __nv_bfloat16* queries, int head,
                                  int lane, PairedQuery& query) {
    #pragma unroll
    for (int pair = 0; pair < split_query_heads; ++pair) {
        #pragma unroll
        for (int pass = 0; pass < score_passes; ++pass) {
            const int group = pass * groups_per_warp + lane / score_lanes;
            const auto row = row_offset(0, query_heads, head + pair);
            #pragma unroll
            for (int value = 0; value < groups_per_warp; ++value) {
                const auto offset = group * mxfp8_group_size + lane % score_lanes;
                query[pair][pass][value] =
                    __bfloat162float(queries[row + offset + value * score_lanes]);
            }
        }
    }
}

__device__ float reduce_group_score(const float (&products)[groups_per_warp]) {
    float sum = __fadd_rn(__fadd_rn(products[0], products[2]),
                         __fadd_rn(products[1], products[3]));
    #pragma unroll
    for (int delta = score_lanes / 2; delta > 0; delta /= 2) {
        sum += __shfl_down_sync(warp_mask, sum, delta, score_lanes);
    }
    return sum;
}

__device__ void paired_decode_scores(Mxfp8KvCache cache, int page, int slot, int kv_head,
                                     int lane, const PairedQuery& query,
                                     float (&scores)[split_query_heads]) {
    const auto data = payload_offset(cache.kv_heads, page, slot, kv_head) + lane % score_lanes;
    const auto scale = scale_offset(cache.kv_heads, page, slot, kv_head, 0);
    #pragma unroll
    for (int pass = 0; pass < score_passes; ++pass) {
        const int group = pass * groups_per_warp + lane / score_lanes;
        const auto exponent = cache.key_scales[scale + group];
        float keys[groups_per_warp];
        #pragma unroll
        for (int value = 0; value < groups_per_warp; ++value) {
            const auto element = data + group * mxfp8_group_size + value * score_lanes;
            keys[value] = decode_e8m0(cache.key_payload[element], exponent);
        }
        #pragma unroll
        for (int pair = 0; pair < split_query_heads; ++pair) {
            float products[groups_per_warp];
            #pragma unroll
            for (int value = 0; value < groups_per_warp; ++value) {
                products[value] = __fmul_rn(query[pair][pass][value], keys[value]);
            }
            const float sum = reduce_group_score(products);
            #pragma unroll
            for (int subgroup = 0; subgroup < groups_per_warp; ++subgroup) {
                const float ordered = __shfl_sync(warp_mask, sum, subgroup * score_lanes);
                if (lane == 0) {
                    scores[pair] += ordered;
                }
            }
        }
    }
}

__device__ void paired_decode_values(decoder::PagedDecodeLane input,
                                     PairedSplit (&state)[split_query_heads],
                                     int first, int count, int kv_head, int warp, int lane,
                                     bool (&invalid)[split_query_heads]) {
    float sum[split_query_heads][mxfp8_groups]{};
    float mass[split_query_heads]{};
    bool oversized = false;
    for (int index = warp; index < count; index += split_warps) {
        const int token = first + index;
        const int slot = token % mxfp8_page_size;
        const int page = input.pages.device_pages[token / mxfp8_page_size];
        const auto data = payload_offset(input.cache.kv_heads, page, slot, kv_head) + lane;
        const auto scale = scale_offset(input.cache.kv_heads, page, slot, kv_head, 0);
        float weights[split_query_heads];
        #pragma unroll
        for (int pair = 0; pair < split_query_heads; ++pair) {
            weights[pair] = expf(state[pair].weights[index] - state[pair].maximum);
            mass[pair] += weights[pair];
        }
        #pragma unroll
        for (int group = 0; group < mxfp8_groups; ++group) {
            const auto exponent = input.cache.value_scales[scale + group];
            const auto payload = input.cache.value_payload[data + group * mxfp8_group_size];
            oversized |= exponent > max_split_value_exponent;
            const float value = decode_e8m0(payload, exponent);
            #pragma unroll
            for (int pair = 0; pair < split_query_heads; ++pair) {
                sum[pair][group] = fmaf(weights[pair], value, sum[pair][group]);
            }
        }
    }
    #pragma unroll
    for (int pair = 0; pair < split_query_heads; ++pair) {
        invalid[pair] |= oversized;
        #pragma unroll
        for (int group = 0; group < mxfp8_groups; ++group) {
            state[pair].values[warp][group * mxfp8_group_size + lane] = sum[pair][group];
            invalid[pair] |= !isfinite(sum[pair][group]);
        }
        if (lane == 0) {
            state[pair].masses[warp] = mass[pair];
        }
    }
}

__device__ void store_paired_partial(DecodePartial& partial, const PairedSplit& state,
                                     bool invalid, int thread, int warp, int lane) {
    if (__syncthreads_or(invalid)) {
        if (thread == 0) {
            partial.maximum = CUDART_NAN_F;
        }
        return;
    }
    float value = 0.0f;
    #pragma unroll
    for (int index = 0; index < split_warps; ++index) {
        value += state.values[index][thread];
    }
    partial.value[thread] = value;
    if (warp == 0) {
        const float total = warp_sum(lane < split_warps ? state.masses[lane] : 0.0f);
        if (lane == 0) {
            partial.maximum = state.maximum;
            partial.mass = total;
        }
    }
}

template <typename Lane>
__global__ void split_decode_attention_paired(PagedDecodeBatch<Lane> batch) {
    const auto input = resolve_lane(batch.lanes[blockIdx.z]);
    if (!decoder::paged_decode_position(input.position)) {
        return;
    }
    __shared__ PairedSplit state[split_query_heads];
    const int head = blockIdx.y * split_query_heads;
    if (head >= gqa_query_heads(input.cache.kv_heads)) {
        return;
    }
    const int thread = threadIdx.x;
    const int warp = thread / mxfp8_group_size;
    const int lane = thread % mxfp8_group_size;
    const int kv_head = head / queries_per_kv_head;
    const int visible = input.position + 1;
    const int pages = (visible + mxfp8_page_size - 1) / mxfp8_page_size;
    const int first = blockIdx.x * pages / decode_splits * mxfp8_page_size;
    const int end = min(visible, (blockIdx.x + 1) * pages / decode_splits * mxfp8_page_size);
    const int count = end - first;
    PairedQuery query;
    float local_max[split_query_heads] = {-CUDART_INF_F, -CUDART_INF_F};
    bool invalid[split_query_heads]{};
    load_paired_query(gqa_query_heads(input.cache.kv_heads), input.buffers.queries, head, lane,
                      query);
    for (int index = warp; index < count; index += split_warps) {
        const int token = first + index;
        const int page = input.pages.device_pages[token / mxfp8_page_size];
        float scores[split_query_heads]{};
        paired_decode_scores(input.cache, page, token % mxfp8_page_size, kv_head, lane,
                             query, scores);
        if (lane == 0) {
            #pragma unroll
            for (int pair = 0; pair < split_query_heads; ++pair) {
                const float score = scores[pair] * softmax_scale;
                state[pair].weights[index] = score;
                local_max[pair] = fmaxf(local_max[pair], score);
                invalid[pair] |= !isfinite(score);
            }
        }
    }
    if (lane == 0) {
        #pragma unroll
        for (int pair = 0; pair < split_query_heads; ++pair) {
            state[pair].maxima[warp] = local_max[pair];
        }
    }
    __syncthreads();
    if (warp == 0) {
        #pragma unroll
        for (int pair = 0; pair < split_query_heads; ++pair) {
            const float maximum = warp_max(lane < split_warps ? state[pair].maxima[lane] :
                                                                -CUDART_INF_F);
            if (lane == 0) {
                state[pair].maximum = maximum;
            }
        }
    }
    __syncthreads();
    paired_decode_values(input, state, first, count, kv_head, warp, lane, invalid);
    #pragma unroll
    for (int pair = 0; pair < split_query_heads; ++pair) {
        auto& partial = decode_partials(input.buffers, head + pair)[blockIdx.x];
        store_paired_partial(partial, state[pair], invalid[pair], thread, warp, lane);
    }
}

template <typename Lane>
int lane_query_heads(const Lane& lane) {
    if constexpr (std::is_same_v<Lane, decoder::BoundPagedDecodeLane>) {
        return mxfp8_query_heads;
    } else {
        return gqa_query_heads(lane.cache.kv_heads);
    }
}

template <typename Lane>
void launch_decode_split(const PagedDecodeBatch<Lane>& batch, int count, cudaStream_t stream) {
    const int query_heads = lane_query_heads(batch.lanes[0]);
    if (count == decoder::paged_decode_lanes) {
        split_decode_attention_paired
            <<<dim3(decode_splits, query_heads / split_query_heads, count),
               mxfp8_head_dim, 0, stream>>>(batch);
    } else {
        split_decode_attention<<<dim3(decode_splits, query_heads, count),
                                 mxfp8_head_dim, 0, stream>>>(batch);
    }
}

template <typename Lane>
cudaError_t launch_paged_decode(const Lane* lanes, int count, cudaStream_t stream) {
    if (!lanes || count < 1 || count > decoder::paged_decode_lanes) {
        return cudaErrorInvalidValue;
    }
    PagedDecodeBatch<Lane> batch{};
    const int query_heads = lane_query_heads(lanes[0]);
    for (int index = 0; index < count; ++index) {
        const auto buffers = lanes[index].buffers;
        const auto address = reinterpret_cast<std::uintptr_t>(buffers.scratch);
        bool missing_cache = false;
        if constexpr (std::is_same_v<Lane, decoder::BoundPagedDecodeLane>) {
            missing_cache = !lanes[index].cache;
        }
        if (!buffers.queries || !buffers.output || !address || address % alignof(DecodePartial)
            || missing_cache || lane_query_heads(lanes[index]) != query_heads) {
            return cudaErrorInvalidValue;
        }
        batch.lanes[index] = lanes[index];
    }
    launch_decode_split(batch, count, stream);
    auto status = cudaGetLastError();
    if (status != cudaSuccess) {
        return status;
    }
    merge_decode_attention<<<dim3(query_heads, 1, count),
                             mxfp8_head_dim, 0, stream>>>(batch);
    status = cudaGetLastError();
    if (status != cudaSuccess) {
        return status;
    }
    if constexpr (std::is_same_v<Lane, decoder::BoundPagedDecodeLane>) {
        bound_exact_decode_scores<<<dim3(32, query_heads, count),
                                    mxfp8_head_dim, 0, stream>>>(batch);
        status = cudaGetLastError();
        if (status != cudaSuccess) {
            return status;
        }
    }
    replay_paged_decode<<<dim3(query_heads, mxfp8_head_dim / replay_channels, count),
                          mxfp8_head_dim, 0, stream>>>(batch);
    return cudaGetLastError();
}

}

cudaError_t decoder::paged_decode_gqa(const PagedDecodeLane* lanes, int count,
                                     cudaStream_t stream) {
    if (!lanes || count < 1 || count > paged_decode_lanes) {
        return cudaErrorInvalidValue;
    }
    for (int index = 0; index < count; ++index) {
        const auto& lane = lanes[index];
        if (!paged_decode_position(lane.position) ||
            !cache_covers(lane.cache, lane.pages, lane.position + 1)) {
            return cudaErrorInvalidValue;
        }
    }
    return launch_paged_decode(lanes, count, stream);
}

cudaError_t decoder::paged_decode_gqa(
    const BoundPagedDecodeLane* lanes, int count, cudaStream_t stream) {
    return launch_paged_decode(lanes, count, stream);
}

}
