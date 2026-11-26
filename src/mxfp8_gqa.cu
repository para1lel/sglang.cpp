#include "sglang/mxfp8_gqa.h"

#include "sglang/decoder_batch.h"
#include "sglang/decoder_cuda.cuh"
#include "sglang/text/gqa/gqa_decode_long.cuh"
#include "sglang/text/gqa_decode_split.cuh"
#include "sglang/text/gqa_verify_batch.h"
#include "sglang/runtime/execution/tp_verify_graph.h"

#include <algorithm>
#include <type_traits>

#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <math_constants.h>

namespace sglang {
namespace {

using decoder::cache_covers;

constexpr unsigned int warp_mask = 0xffffffffu;
constexpr int queries_per_kv_head = mxfp8_query_heads / mxfp8_kv_heads;
constexpr float softmax_scale = 1.0f / 16.0f;

__device__ std::size_t row_offset(std::size_t token, int heads, int head) {
    return (token * heads + head) * mxfp8_head_dim;
}

__device__ std::size_t payload_offset(int kv_heads, int physical_page, int token, int head) {
    return static_cast<std::size_t>(physical_page) * gqa_payload_bytes_per_page(kv_heads)
         + row_offset(token, kv_heads, head);
}

__device__ std::size_t scale_offset(int kv_heads, int physical_page, int token, int head,
                                    int group) {
    constexpr int token_groups = mxfp8_page_size / mxfp8_group_size;
    const auto page_head = static_cast<std::size_t>(physical_page) * kv_heads + head;
    const auto token_lane = page_head * mxfp8_group_size + token % mxfp8_group_size;
    return (token_lane * token_groups + token / mxfp8_group_size) * mxfp8_groups + group;
}

__device__ int scale_exponent(float value) {
    float amax = fabsf(value);
    for (int delta = mxfp8_group_size / 2; delta > 0; delta /= 2) {
        amax = fmaxf(amax, __shfl_xor_sync(warp_mask, amax, delta));
    }
    int binary_exponent = 0;
    const float fraction = frexpf(fmaxf(amax, 1e-30f), &binary_exponent);
    const int biased = binary_exponent - 9 + (fraction > 0.875f) + 127;
    return min(254, max(0, biased));
}

__device__ std::uint8_t encode(float value, int exponent) {
    return __nv_cvt_float_to_fp8(ldexpf(value, 127 - exponent), __NV_SATFINITE, __NV_E4M3);
}

__device__ float decode(std::uint8_t payload, std::uint8_t exponent) {
    __nv_fp8_e4m3 value;
    value.__x = payload;
    return ldexpf(static_cast<float>(value), static_cast<int>(exponent) - 127);
}

__device__ Mxfp8KvCache cache_value(Mxfp8KvCache cache) { return cache; }
__device__ Mxfp8KvCache cache_value(const Mxfp8KvCache* cache) { return *cache; }

__device__ const std::int32_t* page_values(const std::int32_t* pages) { return pages; }
__device__ const std::int32_t* page_values(const Mxfp8PageTable* pages) {
    return pages->device_pages;
}

__device__ __forceinline__ void append_head(
    Mxfp8KvCache cache, const std::int32_t* pages, int position, std::size_t input_row,
    const __nv_bfloat16* keys, const __nv_bfloat16* values) {
    const int token = position % mxfp8_page_size;
    const int physical_page = pages[position / mxfp8_page_size];
    const int head = blockIdx.y;
    if (head >= cache.kv_heads) {
        return;
    }
    const int channel = threadIdx.x;
    const auto input = row_offset(input_row, cache.kv_heads, head) + channel;
    const auto payload = payload_offset(cache.kv_heads, physical_page, token, head) + channel;
    const float key = __bfloat162float(keys[input]);
    const float value = __bfloat162float(values[input]);
    const int key_exponent = scale_exponent(key);
    const int value_exponent = scale_exponent(value);
    cache.key_payload[payload] = encode(key, key_exponent);
    cache.value_payload[payload] = encode(value, value_exponent);
    if (channel % mxfp8_group_size == 0) {
        const auto scale =
            scale_offset(cache.kv_heads, physical_page, token, head, channel / mxfp8_group_size);
        cache.key_scales[scale] = static_cast<std::uint8_t>(key_exponent);
        cache.value_scales[scale] = static_cast<std::uint8_t>(value_exponent);
    }
}

template <typename Position, typename Cache, typename Pages>
__global__ void append_kernel(Cache cache_binding, Pages page_binding,
                              Position first_position,
                              const __nv_bfloat16* keys, const __nv_bfloat16* values) {
    const int position = decoder::position_value(first_position) + static_cast<int>(blockIdx.x);
    append_head(cache_value(cache_binding), page_values(page_binding), position, blockIdx.x,
                keys, values);
}

__device__ float warp_sum(float value) {
    for (int delta = mxfp8_group_size / 2; delta > 0; delta /= 2) {
        value += __shfl_down_sync(warp_mask, value, delta);
    }
    return value;
}

__global__ void tiny_attention_kernel(Mxfp8KvCache cache, const std::int32_t* pages,
                                      int first_query_position, const __nv_bfloat16* queries,
                                      __nv_bfloat16* output) {
    __shared__ float warp_dots[mxfp8_groups];
    __shared__ float history_weight;
    __shared__ float token_weight;
    __shared__ float denominator;

    const int query_head = blockIdx.y;
    const int kv_head = query_head / queries_per_kv_head;
    const int channel = threadIdx.x;
    const int group = channel / mxfp8_group_size;
    const int query_heads = gqa_query_heads(cache.kv_heads);
    const auto row = row_offset(blockIdx.x, query_heads, query_head) + channel;
    const int last_position = first_query_position + static_cast<int>(blockIdx.x);
    const float query = __bfloat162float(queries[row]);
    float max_score = -CUDART_INF_F;
    float weight_sum = 0.0f;
    float weighted_value = 0.0f;

    for (int position = 0; position <= last_position; ++position) {
        const int physical_page = pages[position / mxfp8_page_size];
        const int token = position % mxfp8_page_size;
        const auto payload =
            payload_offset(cache.kv_heads, physical_page, token, kv_head) + channel;
        const auto scale = scale_offset(cache.kv_heads, physical_page, token, kv_head, group);
        const float key = decode(cache.key_payload[payload], cache.key_scales[scale]);
        const float product = warp_sum(query * key);
        if (channel % mxfp8_group_size == 0) {
            warp_dots[group] = product;
        }
        __syncthreads();

        if (channel == 0) {
            float dot = 0.0f;
            for (int warp = 0; warp < mxfp8_groups; ++warp) {
                dot += warp_dots[warp];
            }
            const float score = dot * softmax_scale;
            const float next_max = fmaxf(max_score, score);
            history_weight = expf(max_score - next_max);
            token_weight = expf(score - next_max);
            weight_sum = fmaf(history_weight, weight_sum, token_weight);
            max_score = next_max;
        }
        const float value = decode(cache.value_payload[payload], cache.value_scales[scale]);
        __syncthreads();
        weighted_value = fmaf(token_weight, value, history_weight * weighted_value);
    }

    if (channel == 0) {
        denominator = weight_sum;
    }
    __syncthreads();
    output[row] = __float2bfloat16_rn(weighted_value / denominator);
}

constexpr int score_warps = 8;
constexpr int replay_tile = 32;
constexpr int replay_channels = 2 * mxfp8_group_size;
constexpr int replay_threads = mxfp8_head_dim;
static_assert(mxfp8_page_size % replay_tile == 0 && replay_tile == mxfp8_group_size);
static_assert(replay_threads % replay_channels == 0 && mxfp8_head_dim % replay_channels == 0);

__device__ float decode_e8m0(std::uint8_t payload, std::uint8_t exponent) {
    if (exponent == 255) {
        return decode(payload, exponent);
    }
    __nv_fp8_e4m3 value;
    value.__x = payload;
    const auto scale_bits = exponent == 0 ? 0x00400000u :
                                           static_cast<unsigned int>(exponent) << 23;
    return __fmul_rn(static_cast<float>(value), __uint_as_float(scale_bits));
}

__device__ decoder::ExactGqaWorkspace score_workspace(decoder::ExactGqaWorkspace workspace) {
    return workspace;
}

__device__ decoder::ExactGqaWorkspace score_workspace(const decoder::ExactGqaWorkspace* workspace) {
    return *workspace;
}

constexpr int narrow_min_tokens = 1536;
constexpr int narrow_max_tokens = 2560;
constexpr int partial_stride = mxfp8_head_dim + 2;
constexpr int max_partitions = narrow_max_tokens / mxfp8_page_size;
constexpr std::size_t narrow_workspace_bytes(int query_heads) {
    return static_cast<std::size_t>(query_heads) * max_partitions * partial_stride * sizeof(float);
}
static_assert(max_partitions <= mxfp8_group_size);
static_assert(score_warps * mxfp8_group_size == mxfp8_head_dim);
static_assert(replay_channels >= mxfp8_group_size);

__host__ __device__ bool narrow_position(int position) {
    return position >= narrow_min_tokens - 1 && position < narrow_max_tokens;
}

__device__ float warp_max(float value) {
    for (int delta = mxfp8_group_size / 2; delta > 0; delta /= 2) {
        value = fmaxf(value, __shfl_down_sync(warp_mask, value, delta));
    }
    return value;
}

__device__ void load_query(int query_heads, const __nv_bfloat16* queries, int head, int lane,
                           float (&query)[mxfp8_groups]) {
    #pragma unroll
    for (int group = 0; group < mxfp8_groups; ++group) {
        const int channel = group * mxfp8_group_size + lane;
        query[group] = __bfloat162float(queries[row_offset(0, query_heads, head) + channel]);
    }
}

__device__ float attention_score(Mxfp8KvCache cache, int physical_page, int token, int kv_head,
                                 int lane, const float (&query)[mxfp8_groups]) {
    const auto payload = payload_offset(cache.kv_heads, physical_page, token, kv_head) + lane;
    const auto scale = scale_offset(cache.kv_heads, physical_page, token, kv_head, 0);
    float dot = 0.0f;
    #pragma unroll
    for (int group = 0; group < mxfp8_groups; ++group) {
        const float key = decode_e8m0(cache.key_payload[payload + group * mxfp8_group_size],
                                      cache.key_scales[scale + group]);
        const float product = warp_sum(query[group] * key);
        if (lane == 0) {
            dot += product;
        }
    }
    return dot * softmax_scale;
}

__device__ void partition_attention(Mxfp8KvCache cache, const std::int32_t* pages, int last,
                                   const __nv_bfloat16* queries, float* workspace) {
    __shared__ float weights[mxfp8_page_size];
    __shared__ float warp_stats[score_warps];
    __shared__ float maximum;
    __shared__ float numerators[mxfp8_head_dim];
    const int partition = blockIdx.x;
    const int partitions = last / mxfp8_page_size + 1;
    if (partition >= partitions) {
        return;
    }
    const int head = blockIdx.y;
    if (head >= gqa_query_heads(cache.kv_heads)) {
        return;
    }
    const int kv_head = head / queries_per_kv_head;
    const int thread = threadIdx.x;
    const int warp = thread / mxfp8_group_size;
    const int lane = thread % mxfp8_group_size;
    const int count = min(mxfp8_page_size, last - partition * mxfp8_page_size + 1);
    const int page = pages[partition];
    auto* partial = workspace + (head * partitions + partition) * partial_stride;
    float query[mxfp8_groups];
    load_query(gqa_query_heads(cache.kv_heads), queries, head, lane, query);
    for (int token = warp; token < count; token += score_warps) {
        const float score = attention_score(cache, page, token, kv_head, lane, query);
        if (lane == 0) {
            weights[token] = score;
        }
    }
    __syncthreads();

    const float score = thread < count ? weights[thread] : -CUDART_INF_F;
    const float local_max =
        warp_max(thread < count && !isfinite(score) ? CUDART_INF_F : score);
    if (lane == 0) {
        warp_stats[warp] = local_max;
    }
    __syncthreads();
    if (warp == 0) {
        const float page_max = warp_max(lane < score_warps ? warp_stats[lane] : -CUDART_INF_F);
        if (lane == 0) {
            maximum = page_max;
            partial[mxfp8_head_dim] = page_max;
        }
    }
    __syncthreads();
    const float weight = thread < count ? expf(score - maximum) : 0.0f;
    if (thread < count) {
        weights[thread] = weight;
    }
    const float local_sum = warp_sum(weight);
    if (lane == 0) {
        warp_stats[warp] = local_sum;
    }
    __syncthreads();
    if (warp == 0) {
        const float denominator = warp_sum(lane < score_warps ? warp_stats[lane] : 0.0f);
        if (lane == 0) {
            partial[mxfp8_head_dim + 1] = denominator;
        }
    }

    constexpr int half_page = mxfp8_page_size / 2;
    const int channel = (thread % (mxfp8_head_dim / 2)) * 2;
    const int first = (thread / (mxfp8_head_dim / 2)) * half_page;
    float even = 0.0f;
    float odd = 0.0f;
    for (int token = first; token < min(count, first + half_page); ++token) {
        const auto payload = payload_offset(cache.kv_heads, page, token, kv_head) + channel;
        const auto scale =
            scale_offset(cache.kv_heads, page, token, kv_head, channel / mxfp8_group_size);
        const auto exponent = cache.value_scales[scale];
        even = fmaf(weights[token], decode_e8m0(cache.value_payload[payload], exponent), even);
        odd = fmaf(weights[token], decode_e8m0(cache.value_payload[payload + 1], exponent), odd);
    }
    if (first == 0) {
        numerators[channel] = even;
        numerators[channel + 1] = odd;
    }
    __syncthreads();
    if (first != 0) {
        partial[channel] = numerators[channel] + even;
        partial[channel + 1] = numerators[channel + 1] + odd;
    }
}

__device__ bool merge_attention(const float* workspace, int last, __nv_bfloat16* output,
                                int query_heads) {
    __shared__ float rescale[max_partitions];
    __shared__ float denominator;
    const int partitions = last / mxfp8_page_size + 1;
    const int head = blockIdx.x;
    const int thread = threadIdx.x;
    const auto* head_partials = workspace + head * partitions * partial_stride;
    if (thread < mxfp8_group_size) {
        const auto* partial = head_partials + min(thread, partitions - 1) * partial_stride;
        const float maximum = thread < partitions ? partial[mxfp8_head_dim] : -CUDART_INF_F;
        const float global_max = __shfl_sync(warp_mask, warp_max(maximum), 0);
        const float weight = thread < partitions ? expf(maximum - global_max) : 0.0f;
        if (thread < partitions) {
            rescale[thread] = weight;
        }
        const float sum = warp_sum(thread < partitions ? weight * partial[mxfp8_head_dim + 1]
                                                        : 0.0f);
        if (thread == 0) {
            denominator = sum;
        }
    }
    __syncthreads();
    float numerator = 0.0f;
    if (thread < replay_channels) {
        const int channel = blockIdx.y * replay_channels + thread;
        for (int partition = 0; partition < partitions; ++partition) {
            numerator = fmaf(rescale[partition],
                             head_partials[partition * partial_stride + channel], numerator);
        }
    }
    if (__syncthreads_or(!isfinite(numerator) || !isfinite(denominator) || denominator <= 0.0f)) {
        return false;
    }
    if (thread < replay_channels) {
        const int channel = blockIdx.y * replay_channels + thread;
        output[row_offset(0, query_heads, head) + channel] =
            __float2bfloat16_rn(numerator / denominator);
    }
    return true;
}

__device__ void exact_scores(Mxfp8KvCache cache, const std::int32_t* pages, int last,
                              const __nv_bfloat16* queries,
                              decoder::ExactGqaWorkspace workspace) {
    const int warp = threadIdx.x / mxfp8_group_size;
    const int lane = threadIdx.x % mxfp8_group_size;
    const int query_head = blockIdx.y;
    if (query_head >= gqa_query_heads(cache.kv_heads)) {
        return;
    }
    const int kv_head = query_head / queries_per_kv_head;
    const int first = blockIdx.x * score_warps + warp;
    if (first > last) {
        return;
    }
    float query[mxfp8_groups];
    load_query(gqa_query_heads(cache.kv_heads), queries, query_head, lane, query);
    auto* scores = workspace.scores + query_head * workspace.capacity;
    const int stride = gridDim.x * score_warps;
    for (int index = first; index <= last; index += stride) {
        const int physical_page = pages[index / mxfp8_page_size];
        const int token = index % mxfp8_page_size;
        const float score = attention_score(cache, physical_page, token, kv_head, lane, query);
        if (lane == 0) {
            scores[index] = score;
        }
    }
}

template <bool Narrow, bool SkipTargetSplit, typename Position, typename Cache, typename Pages,
          typename Scores>
__global__ void exact_scores_kernel(
    Cache cache_binding, Pages page_binding, Position position,
    const __nv_bfloat16* queries, Scores score_binding, int split_end) {
    const int last = decoder::position_value(position);
    if constexpr (SkipTargetSplit) {
        if (decoder::target_split_position(last, split_end)) {
            return;
        }
    }
    const auto cache = cache_value(cache_binding);
    const auto* pages = page_values(page_binding);
    const auto workspace = score_workspace(score_binding);
    if constexpr (Narrow) {
        if (narrow_position(last)) {
            partition_attention(cache, pages, last, queries, workspace.scores);
            return;
        }
    }
    exact_scores(cache, pages, last, queries, workspace);
}

#include "sglang/text/gqa/ordered_gqa_vector.cuh"

template <bool Narrow, typename Position, typename Cache, typename Pages, typename Scores>
__global__ void exact_replay_kernel(
    Cache cache_binding, Pages page_binding, Position position,
    Scores score_binding, const __nv_bfloat16* queries, __nv_bfloat16* output) {
    const int last = decoder::position_value(position);
    if (static_cast<int>(blockIdx.x) >=
        gqa_query_heads(cache_value(cache_binding).kv_heads)) {
        return;
    }
    const auto workspace = score_workspace(score_binding);
    if constexpr (Narrow) {
        if (narrow_position(last)) {
            if (!merge_attention(workspace.scores, last, output,
                                 gqa_query_heads(cache_value(cache_binding).kv_heads))) {
                ordered_replay<true>(cache_value(cache_binding), page_values(page_binding), last,
                                     nullptr, queries, output);
            }
            return;
        }
    }
    const auto* scores = workspace.scores + blockIdx.x * workspace.capacity;
    ordered_replay<false>(cache_value(cache_binding), page_values(page_binding), last,
                          scores, nullptr, output);
}

#include "sglang/text/gqa_prefill_replay.cuh"

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

__device__ __noinline__ void wide_decode_head(decoder::PagedDecodeLane input, int head) {
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
#include "sglang/text/mxfp8_gqa_decode.cuh"

int score_grid_capacity(decoder::ExactGqaWorkspace workspace) {
    return workspace.scores ? workspace.capacity : 0;
}

int score_grid_capacity(const decoder::ExactGqaWorkspace* workspace) {
    return workspace ? text_attention_max_tokens : 0;
}

#include "sglang/text/gqa_ordered_launch.cuh"

}

cudaError_t mxfp8_kv_append(Mxfp8KvCache cache, Mxfp8PageTable pages, int first_position,
                           int token_count, const __nv_bfloat16* keys,
                           const __nv_bfloat16* values, cudaStream_t stream,
                           GqaGeometry geometry) {
    if (token_count == 0) {
        return cudaSuccess;
    }
    const auto end = static_cast<std::int64_t>(first_position) + token_count;
    if (first_position < 0 || token_count < 0 || !keys || !values
        || !gqa_geometry_matches(geometry, cache) || !cache_covers(cache, pages, end)) {
        return cudaErrorInvalidValue;
    }
    const auto kernel = append_kernel<int, Mxfp8KvCache, const std::int32_t*>;
    return launch_gqa_kernel(kernel, dim3(token_count, cache.kv_heads), mxfp8_head_dim, stream,
                             cache, pages.device_pages, first_position, keys, values);
}

cudaError_t mxfp8_gqa_attention(Mxfp8KvCache cache, Mxfp8PageTable pages, int valid_tokens,
                               int first_query_position, int query_count,
                               const __nv_bfloat16* queries, __nv_bfloat16* output,
                               cudaStream_t stream, GqaGeometry geometry) {
    if (query_count == 0) {
        return cudaSuccess;
    }
    const auto end = static_cast<std::int64_t>(first_query_position) + query_count;
    if (first_query_position < 0 || query_count < 0 || end > valid_tokens || !queries || !output
        || !gqa_geometry_matches(geometry, cache)
        || !cache_covers(cache, pages, valid_tokens)) {
        return cudaErrorInvalidValue;
    }
    constexpr int min_tiled_tokens = 8;
    if (end >= min_tiled_tokens) {
        return decoder::tensor_core_prefill_gqa(cache, pages, first_query_position, query_count,
                                                queries, output, stream);
    }
    return launch_gqa_kernel(tiny_attention_kernel,
                             dim3(query_count, gqa_query_heads(cache.kv_heads)),
                             mxfp8_head_dim, stream, cache, pages.device_pages,
                             first_query_position, queries, output);
}

cudaError_t decoder::decode_kv_append(Mxfp8KvCache cache, Mxfp8PageTable pages,
                                      const int* position, const __nv_bfloat16* keys,
                                      const __nv_bfloat16* values, cudaStream_t stream) {
    if (!gqa_geometry_valid({gqa_query_heads(cache.kv_heads), cache.kv_heads})) {
        return cudaErrorInvalidValue;
    }
    const auto kernel = append_kernel<const int*, Mxfp8KvCache, const std::int32_t*>;
    return launch_gqa_kernel(kernel, dim3(1, cache.kv_heads), mxfp8_head_dim, stream,
                             cache, pages.device_pages, position, keys, values);
}

cudaError_t decoder::decode_gqa_attention(Mxfp8KvCache cache, Mxfp8PageTable pages,
                                          const int* position, const __nv_bfloat16* queries,
                                          __nv_bfloat16* output, cudaStream_t stream) {
    return tensor_core_prefill_gqa(cache, pages, 0, 1, queries, output, stream, position);
}

cudaError_t decoder::exact_decode_gqa_attention(
    Mxfp8KvCache cache, Mxfp8PageTable pages, int position, const __nv_bfloat16* queries,
    __nv_bfloat16* output, ExactGqaWorkspace workspace, cudaStream_t stream,
    std::size_t workspace_bytes, bool prefill) {
    if (position < 0 || position >= workspace.capacity) {
        return cudaErrorInvalidValue;
    }
    if (!cache_covers(cache, pages, workspace.capacity)) {
        return cudaErrorInvalidValue;
    }
    if (workspace_bytes >= narrow_workspace_bytes(gqa_query_heads(cache.kv_heads))
        && narrow_position(position)) {
        return launch_exact_attention<true>(cache, pages.device_pages, position, queries, output,
                                            workspace, stream);
    }
    if (prefill && position >= 8191 &&
        reinterpret_cast<std::uintptr_t>(cache.value_payload) % 16 == 0 &&
        reinterpret_cast<std::uintptr_t>(cache.value_scales) % 8 == 0) {
        return launch_exact_attention<false, true>(cache, pages.device_pages, position, queries,
                                                   output, workspace, stream);
    }
    return launch_exact_attention(cache, pages.device_pages, position, queries, output,
                                  workspace, stream);
}

cudaError_t decoder::exact_decode_gqa_attention(
    Mxfp8KvCache cache, Mxfp8PageTable pages, const int* position, const __nv_bfloat16* queries,
    __nv_bfloat16* output, ExactGqaWorkspace workspace, cudaStream_t stream,
    std::size_t workspace_bytes) {
    if (!position) {
        return cudaErrorInvalidValue;
    }
    if (!cache_covers(cache, pages, workspace.capacity)) {
        return cudaErrorInvalidValue;
    }
    if (workspace_bytes >= narrow_workspace_bytes(gqa_query_heads(cache.kv_heads))
        && workspace.capacity >= narrow_min_tokens) {
        return launch_exact_attention<true>(cache, pages.device_pages, position, queries, output,
                                            workspace, stream);
    }
    return launch_exact_attention(cache, pages.device_pages, position, queries, output,
                                  workspace, stream);
}

cudaError_t decoder::decode_kv_append(const Mxfp8KvCache* cache, const Mxfp8PageTable* pages,
                                      const int* position, const __nv_bfloat16* keys,
                                      const __nv_bfloat16* values, cudaStream_t stream) {
    if (!cache) {
        return cudaErrorInvalidValue;
    }
    const auto kernel = append_kernel<const int*, const Mxfp8KvCache*, const Mxfp8PageTable*>;
    return launch_gqa_kernel(kernel, dim3(1, mxfp8_kv_heads), mxfp8_head_dim, stream,
                             cache, pages, position, keys, values);
}

cudaError_t decoder::exact_decode_gqa_attention(
    const Mxfp8KvCache* cache, const Mxfp8PageTable* pages, const int* position,
    const __nv_bfloat16* queries, __nv_bfloat16* output, ExactGqaWorkspace workspace,
    cudaStream_t stream) {
    return launch_exact_attention(cache, pages, position, queries, output, workspace, stream);
}

cudaError_t decoder::exact_decode_gqa_attention(
    const Mxfp8KvCache* cache, const Mxfp8PageTable* pages, const int* position,
    const __nv_bfloat16* queries, __nv_bfloat16* output, const ExactGqaWorkspace* workspace,
    cudaStream_t stream, std::size_t workspace_bytes) {
    if (!cache || !pages || !position) {
        return cudaErrorInvalidValue;
    }
    return workspace_bytes >= narrow_workspace_bytes(mxfp8_query_heads)
        ? launch_exact_attention<true>(cache, pages, position, queries, output, workspace, stream)
        : launch_exact_attention(cache, pages, position, queries, output, workspace, stream);
}

cudaError_t decoder::vector_decode_gqa_attention(
    Mxfp8KvCache cache, Mxfp8PageTable pages, int position, const __nv_bfloat16* queries,
    __nv_bfloat16* output, ExactGqaWorkspace workspace, cudaStream_t stream) {
    if (position < 0 || position >= workspace.capacity ||
        !cache_covers(cache, pages, workspace.capacity)) {
        return cudaErrorInvalidValue;
    }
    return launch_exact_attention<false, false, true>(
        cache, pages.device_pages, position, queries, output, workspace, stream);
}

cudaError_t decoder::vector_decode_gqa_attention(
    const Mxfp8KvCache* cache, const Mxfp8PageTable* pages, const int* position,
    const __nv_bfloat16* queries, __nv_bfloat16* output, const ExactGqaWorkspace* workspace,
    cudaStream_t stream) {
    if (!cache || !pages || !position) {
        return cudaErrorInvalidValue;
    }
    return launch_exact_attention<false, false, true>(
        cache, pages, position, queries, output, workspace, stream);
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

#include "sglang/text/gqa_target_split.cuh"
#include "sglang/text/mtp/gqa_attention.cuh"
#include "sglang/text/gqa/short_verify.cuh"

}
