#pragma once

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

__device__ void load_paired_query(const __nv_bfloat16* queries, int head, int lane,
                                  PairedQuery& query) {
    #pragma unroll
    for (int pair = 0; pair < split_query_heads; ++pair) {
        #pragma unroll
        for (int pass = 0; pass < score_passes; ++pass) {
            const int group = pass * groups_per_warp + lane / score_lanes;
            const auto row = row_offset(0, mxfp8_query_heads, head + pair);
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
    const auto data = payload_offset(page, slot, kv_head) + lane % score_lanes;
    const auto scale = scale_offset(page, slot, kv_head, 0);
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
        const auto data = payload_offset(page, slot, kv_head) + lane;
        const auto scale = scale_offset(page, slot, kv_head, 0);
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
    load_paired_query(input.buffers.queries, head, lane, query);
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
void launch_decode_split(const PagedDecodeBatch<Lane>& batch, int count, cudaStream_t stream) {
    if (count == decoder::paged_decode_lanes) {
        split_decode_attention_paired
            <<<dim3(decode_splits, mxfp8_query_heads / split_query_heads, count),
               mxfp8_head_dim, 0, stream>>>(batch);
    } else {
        split_decode_attention<<<dim3(decode_splits, mxfp8_query_heads, count),
                                 mxfp8_head_dim, 0, stream>>>(batch);
    }
}

struct LongSplit {
    float history[split_query_heads];
    PairedSplit paired[split_query_heads];
};

__device__ void score_long_decode_tile(decoder::PagedDecodeLane input, LongSplit& state,
                                       const PairedQuery& query, int first, int count,
                                       int kv_head, int warp, int lane,
                                       bool (&invalid)[split_query_heads]) {
    float local_max[split_query_heads] = {-CUDART_INF_F, -CUDART_INF_F};
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
                state.paired[pair].weights[index] = score;
                local_max[pair] = fmaxf(local_max[pair], score);
                invalid[pair] |= !isfinite(score);
            }
        }
    }
    if (lane == 0) {
        #pragma unroll
        for (int pair = 0; pair < split_query_heads; ++pair) {
            state.paired[pair].maxima[warp] = local_max[pair];
        }
    }
    __syncthreads();
    if (warp == 0) {
        #pragma unroll
        for (int pair = 0; pair < split_query_heads; ++pair) {
            auto& split = state.paired[pair];
            const float maximum = warp_max(lane < split_warps ? split.maxima[lane] :
                                                                -CUDART_INF_F);
            if (lane == 0) {
                const float next = fmaxf(split.maximum, maximum);
                state.history[pair] = expf(split.maximum - next);
                split.maximum = next;
            }
        }
    }
    __syncthreads();
}

__global__ void split_long_decode_attention(const __grid_constant__ LongDecodeBatch batch) {
    __shared__ LongSplit state;
    const auto input = batch.lanes[blockIdx.z];
    const int head = blockIdx.y * split_query_heads;
    const int thread = threadIdx.x;
    const int warp = thread / mxfp8_group_size;
    const int lane = thread % mxfp8_group_size;
    const int kv_head = head / queries_per_kv_head;
    const int visible = input.position + 1;
    const int pages = (visible + mxfp8_page_size - 1) / mxfp8_page_size;
    const int first = blockIdx.x * pages / decode_splits * mxfp8_page_size;
    const int end = min(visible, (blockIdx.x + 1) * pages / decode_splits * mxfp8_page_size);
    PairedQuery query;
    load_paired_query(input.buffers.queries, head, lane, query);
    float sums[split_query_heads][mxfp8_groups]{};
    float masses[split_query_heads]{};
    bool invalid[split_query_heads]{};
    if (thread == 0) {
        #pragma unroll
        for (int pair = 0; pair < split_query_heads; ++pair) {
            state.paired[pair].maximum = -CUDART_INF_F;
        }
    }
    for (int tile = first; tile < end; tile += split_tokens) {
        const int count = min(split_tokens, end - tile);
        score_long_decode_tile(input, state, query, tile, count, kv_head, warp, lane, invalid);
        paired_decode_values(input, state.paired, tile, count, kv_head, warp, lane, invalid);
        #pragma unroll
        for (int pair = 0; pair < split_query_heads; ++pair) {
            const auto& split = state.paired[pair];
            const float history = state.history[pair];
            #pragma unroll
            for (int group = 0; group < mxfp8_groups; ++group) {
                const int channel = group * mxfp8_group_size + lane;
                sums[pair][group] = fmaf(history, sums[pair][group], split.values[warp][channel]);
            }
            if (lane == 0) {
                masses[pair] = fmaf(history, masses[pair], split.masses[warp]);
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int pair = 0; pair < split_query_heads; ++pair) {
        auto& split = state.paired[pair];
        #pragma unroll
        for (int group = 0; group < mxfp8_groups; ++group) {
            split.values[warp][group * mxfp8_group_size + lane] = sums[pair][group];
            invalid[pair] |= !isfinite(sums[pair][group]);
        }
        if (lane == 0) {
            split.masses[warp] = masses[pair];
        }
        auto& partial = decode_partials(input.buffers, head + pair)[blockIdx.x];
        store_paired_partial(partial, split, invalid[pair], thread, warp, lane);
    }
}

__global__ void replay_long_paged_decode(const __grid_constant__ LongDecodeBatch batch) {
    const auto input = batch.lanes[blockIdx.z];
    const int head = blockIdx.x;
    if (!isfinite(decode_partials(input.buffers, head)[0].maximum)) {
        wide_decode_head(input, head);
    }
}

template <typename Lane>
cudaError_t launch_paged_decode(const Lane* lanes, int count, cudaStream_t stream) {
    if (!lanes || count < 1 || count > decoder::paged_decode_lanes) {
        return cudaErrorInvalidValue;
    }
    PagedDecodeBatch<Lane> batch{};
    for (int index = 0; index < count; ++index) {
        const auto buffers = lanes[index].buffers;
        const auto address = reinterpret_cast<std::uintptr_t>(buffers.scratch);
        if (!buffers.queries || !buffers.output || !address || address % alignof(DecodePartial)) {
            return cudaErrorInvalidValue;
        }
        batch.lanes[index] = lanes[index];
    }
    launch_decode_split(batch, count, stream);
    auto status = cudaGetLastError();
    if (status != cudaSuccess) {
        return status;
    }
    merge_decode_attention<<<dim3(mxfp8_query_heads, 1, count),
                             mxfp8_head_dim, 0, stream>>>(batch);
    status = cudaGetLastError();
    if (status != cudaSuccess) {
        return status;
    }
    if constexpr (std::is_same_v<Lane, decoder::BoundPagedDecodeLane>) {
        bound_exact_decode_scores<<<dim3(32, mxfp8_query_heads, count),
                                    mxfp8_head_dim, 0, stream>>>(batch);
        status = cudaGetLastError();
        if (status != cudaSuccess) {
            return status;
        }
    }
    replay_paged_decode<<<dim3(mxfp8_query_heads, mxfp8_head_dim / replay_channels, count),
                          mxfp8_head_dim, 0, stream>>>(batch);
    return cudaGetLastError();
}

#include "sglang/text/mxfp8_gqa_wide_tc.cuh"

cudaError_t launch_long_paged_decode(const decoder::PagedDecodeLane* lanes, int count,
                                    cudaStream_t stream) {
    if (!lanes || count < 1 || count > decoder_batch_max_lanes) {
        return cudaErrorInvalidValue;
    }
    LongDecodeBatch batch{};
    bool tensor = count > decoder::paged_decode_lanes;
    for (int index = 0; index < count; ++index) {
        const auto& lane = lanes[index];
        const auto& buffers = lane.buffers;
        const auto address = reinterpret_cast<std::uintptr_t>(buffers.scratch);
        if (!decoder::long_paged_decode_position(lane.position) ||
            !cache_covers(lane.cache, lane.pages, lane.position + 1) ||
            !buffers.queries || !buffers.output || !address || address % alignof(DecodePartial)) {
            return cudaErrorInvalidValue;
        }
        const auto key = reinterpret_cast<std::uintptr_t>(lane.cache.key_payload);
        const auto value = reinterpret_cast<std::uintptr_t>(lane.cache.value_payload);
        tensor &= (key | value) % alignof(uint4) == 0;
        batch.lanes[index] = lane;
    }
    auto status = launch_gqa_kernel(
        tensor ? split_long_decode_tensor : split_long_decode_attention,
        dim3(decode_splits, tensor ? mxfp8_kv_heads : mxfp8_query_heads / split_query_heads, count),
        tensor ? wide_warps * 32 : mxfp8_head_dim, stream, batch);
    if (status != cudaSuccess) {
        return status;
    }
    const dim3 grid(mxfp8_query_heads, 1, count);
    status = launch_gqa_kernel(
        merge_decode_attention<decoder::PagedDecodeLane, decoder_batch_max_lanes>,
        grid, mxfp8_head_dim, stream, batch);
    if (status != cudaSuccess) {
        return status;
    }
    return launch_gqa_kernel(replay_long_paged_decode, grid, mxfp8_head_dim, stream, batch);
}
