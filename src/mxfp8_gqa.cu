#include "sglang/text/gqa/decode_kernels.cuh"
#include "sglang/text/gqa_verify_batch.h"
#include "sglang/runtime/execution/tp_verify_graph.h"
#include <algorithm>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <math_constants.h>

namespace sglang {
namespace gqa_device {

__device__ __inline_hint__ float decode(std::uint8_t payload, std::uint8_t exponent) {
    __nv_fp8_e4m3 value;
    value.__x = payload;
    return ldexpf(static_cast<float>(value), static_cast<int>(exponent) - 127);
}

__device__ __inline_hint__ float warp_sum(float value) {
    for (int delta = mxfp8_group_size / 2; delta > 0; delta /= 2) {
        value += __shfl_down_sync(warp_mask, value, delta);
    }
    return value;
}

__device__ __inline_hint__ float decode_e8m0(std::uint8_t payload, std::uint8_t exponent) {
    if (exponent == 255) {
        return decode(payload, exponent);
    }
    __nv_fp8_e4m3 value;
    value.__x = payload;
    const auto scale_bits = exponent == 0 ? 0x00400000u :
                                           static_cast<unsigned int>(exponent) << 23;
    return __fmul_rn(static_cast<float>(value), __uint_as_float(scale_bits));
}

__device__ __inline_hint__ float warp_max(float value) {
    for (int delta = mxfp8_group_size / 2; delta > 0; delta /= 2) {
        value = fmaxf(value, __shfl_down_sync(warp_mask, value, delta));
    }
    return value;
}

__device__ __inline_hint__ void load_query(int query_heads, const __nv_bfloat16* queries,
                                           int head, int lane, float (&query)[mxfp8_groups]) {
    #pragma unroll
    for (int group = 0; group < mxfp8_groups; ++group) {
        const int channel = group * mxfp8_group_size + lane;
        query[group] = __bfloat162float(queries[row_offset(0, query_heads, head) + channel]);
    }
}

__device__ __inline_hint__ float attention_score(Mxfp8KvCache cache, int physical_page,
                                                int token, int kv_head, int lane,
                                                const float (&query)[mxfp8_groups]) {
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

__device__ __inline_hint__ void exact_scores(Mxfp8KvCache cache, const std::int32_t* pages,
                                             int last, const __nv_bfloat16* queries,
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

__device__ __inline_hint__ bool replay_values_aligned(Mxfp8KvCache cache) {
    return reinterpret_cast<std::uintptr_t>(cache.value_payload) % alignof(std::uint32_t) == 0;
}

__device__ __forceinline__ std::uint32_t replay_value_word(
    const std::uint8_t* source, bool aligned_values) {
    if (aligned_values) {
        return *reinterpret_cast<const std::uint32_t*>(source);
    }
    return static_cast<std::uint32_t>(source[0]) |
           (static_cast<std::uint32_t>(source[1]) << 8) |
           (static_cast<std::uint32_t>(source[2]) << 16) |
           (static_cast<std::uint32_t>(source[3]) << 24);
}

__device__ __inline_hint__ void load_replay_values(
    Mxfp8KvCache cache, int page, int token, int kv_head, int group, int count,
    bool aligned_values, float (&values)[replay_tile][vector_replay_channels]) {
    constexpr int vectors = vector_replay_channels / 4;
    const int payload_stride = cache.kv_heads * mxfp8_head_dim;
    constexpr int scale_stride = (mxfp8_page_size / mxfp8_group_size) * mxfp8_groups;
    const auto payload = payload_offset(cache.kv_heads, page, token, kv_head)
                       + group * vector_replay_channels;
    const auto scale = scale_offset(cache.kv_heads, page, token, kv_head, group);
    const int vector = threadIdx.x % vectors;
    #pragma unroll
    for (int index = threadIdx.x / vectors; index < replay_tile;
         index += vector_replay_threads / vectors) {
        if (index < count) {
            const auto* source = cache.value_payload + payload + index * payload_stride +
                                 vector * 4;
            const auto packed = replay_value_word(source, aligned_values);
            const auto exponent = cache.value_scales[scale + index * scale_stride];
            *reinterpret_cast<float4*>(&values[index][vector * 4]) =
                make_float4(decode_e8m0(packed, exponent), decode_e8m0(packed >> 8, exponent),
                            decode_e8m0(packed >> 16, exponent),
                            decode_e8m0(packed >> 24, exponent));
        }
    }
}

template <bool Unit>
__device__ __inline_hint__ float replay_vector_channel(
                                        const float (&values)[replay_tile][vector_replay_channels],
                                        const float2* weights, int count, int lane,
                                        float weighted) {
    #pragma unroll 8
    for (int index = 0; index < count; ++index) {
        const auto factor = weights[index];
        weighted = fmaf(factor.y, values[index][lane], Unit ? weighted : factor.x * weighted);
    }
    return weighted;
}

__device__ void vector_replay_body(Mxfp8KvCache cache, const std::int32_t* pages,
                                    int last, decoder::ExactGqaWorkspace workspace,
                                    __nv_bfloat16* output, int head, int group) {
    __shared__ alignas(16) float values[replay_tile][vector_replay_channels];
    __shared__ float2 weights[replay_tile];
    __shared__ float denominator;
    __shared__ bool unit_history;
    const int query_heads = gqa_query_heads(cache.kv_heads);
    const int kv_head = head / queries_per_kv_head;
    const int thread = threadIdx.x;
    const int lane = thread % mxfp8_group_size;
    const int column = group * vector_replay_channels;
    const auto* scores = workspace.scores + head * workspace.capacity;
    float maximum = -CUDART_INF_F;
    float mass = 0.0f;
    float weighted = 0.0f;
    const bool aligned_values = replay_values_aligned(cache);
    for (int first = 0; first <= last; first += replay_tile) {
        const int count = min(replay_tile, last - first + 1);
        const int physical_page = pages[first / mxfp8_page_size];
        const int token = first % mxfp8_page_size;
        load_replay_values(cache, physical_page, token, kv_head, group, count, aligned_values,
                           values);
        if (thread < mxfp8_group_size) {
            const float score = lane < count ? scores[first + lane] : -CUDART_INF_F;
            const bool unit = isfinite(maximum) && __all_sync(warp_mask, score <= maximum);
            if (lane == 0) {
                unit_history = unit;
            }
            float prefix = fmaxf(maximum, score);
            if (unit) {
                if (lane < count) {
                    weights[lane] = {1.0f, expf(score - prefix)};
                }
            } else {
                #pragma unroll
                for (int offset = 1; offset < replay_tile; offset *= 2) {
                    const float earlier = __shfl_up_sync(warp_mask, prefix, offset);
                    if (lane >= offset) {
                        prefix = fmaxf(earlier, prefix);
                    }
                }
                const float previous = __shfl_up_sync(warp_mask, prefix, 1);
                if (lane < count) {
                    weights[lane] = {expf((lane == 0 ? maximum : previous) - prefix),
                                     expf(score - prefix)};
                }
                maximum = __shfl_sync(warp_mask, prefix, replay_tile - 1);
            }
        }
        __syncthreads();
        if (thread < mxfp8_group_size) {
            weighted = unit_history
                ? replay_vector_channel<true>(values, weights, count, lane, weighted)
                : replay_vector_channel<false>(values, weights, count, lane, weighted);
        }
        if (thread == mxfp8_group_size) {
            #pragma unroll 8
            for (int index = 0; index < count; ++index) {
                const auto factor = weights[index];
                mass = fmaf(factor.x, mass, factor.y);
            }
            denominator = mass;
        }
        __syncthreads();
    }
    if (thread < mxfp8_group_size) {
        output[row_offset(0, query_heads, head) + column + lane] =
            __float2bfloat16_rn(weighted / denominator);
    }
}

template __device__ float replay_vector_channel<true>(
    const float (&)[replay_tile][vector_replay_channels], const float2*, int, int, float);
template __device__ float replay_vector_channel<false>(
    const float (&)[replay_tile][vector_replay_channels], const float2*, int, int, float);

template <bool RecomputeScores>
__device__ void ordered_replay(
    Mxfp8KvCache cache, const std::int32_t* pages, int last, const float* scores,
    const __nv_bfloat16* queries, __nv_bfloat16* output) {
    __shared__ float history_weight[replay_tile];
    __shared__ float token_weight[replay_tile];
    __shared__ float denominator;
    __shared__ float values[replay_tile][replay_channels];
    __shared__ float tile_scores[RecomputeScores ? replay_tile : 1];
    const int query_head = blockIdx.x;
    const int kv_head = query_head / queries_per_kv_head;
    const int query_heads = gqa_query_heads(cache.kv_heads);
    if (query_head >= query_heads) return;
    const int thread = threadIdx.x;
    const int column = thread % replay_channels;
    const int channel = blockIdx.y * replay_channels + column;
    const int group = channel / mxfp8_group_size;
    const int warp = thread / mxfp8_group_size;
    const int lane = thread % mxfp8_group_size;
    float query[mxfp8_groups];
    if constexpr (RecomputeScores)
        load_query(query_heads, queries, query_head, lane, query);
    float maximum = -CUDART_INF_F;
    float weight_sum = 0.0f;
    float weighted_value = 0.0f;
    constexpr int tokens_per_load = replay_threads / replay_channels;
    const int payload_stride = cache.kv_heads * mxfp8_head_dim;
    constexpr int scale_stride = (mxfp8_page_size / mxfp8_group_size) * mxfp8_groups;
    for (int first = 0; first <= last; first += replay_tile) {
        const int count = min(replay_tile, last - first + 1);
        const int physical_page = pages[first / mxfp8_page_size];
        const int token = first % mxfp8_page_size;
        const auto payload =
            payload_offset(cache.kv_heads, physical_page, token, kv_head) + channel;
        const auto scale = scale_offset(cache.kv_heads, physical_page, token, kv_head, group);
        if constexpr (RecomputeScores) {
            for (int index = warp; index < count; index += score_warps) {
                const float score =
                    attention_score(cache, physical_page, token + index, kv_head, lane, query);
                if (lane == 0) {
                    tile_scores[index] = score;
                }
            }
            __syncthreads();
        }
        #pragma unroll
        for (int index = thread / replay_channels; index < replay_tile; index += tokens_per_load) {
            if (index < count) {
                values[index][column] = decode_e8m0(
                    cache.value_payload[payload + index * payload_stride],
                    cache.value_scales[scale + index * scale_stride]);
            }
        }
        if (thread < mxfp8_group_size) {
            float score = -CUDART_INF_F;
            if (thread < count) {
                if constexpr (RecomputeScores) {
                    score = tile_scores[thread];
                } else {
                    score = scores[first + thread];
                }
            }
            float prefix = fmaxf(maximum, score);
            #pragma unroll
            for (int offset = 1; offset < replay_tile; offset *= 2) {
                const float earlier = __shfl_up_sync(warp_mask, prefix, offset);
                if (thread >= offset) {
                    prefix = fmaxf(earlier, prefix);
                }
            }
            const float previous = __shfl_up_sync(warp_mask, prefix, 1);
            if (thread < count) {
                history_weight[thread] = expf((thread == 0 ? maximum : previous) - prefix);
                token_weight[thread] = expf(score - prefix);
            }
            maximum = __shfl_sync(warp_mask, prefix, replay_tile - 1);
        }
        __syncthreads();

        if (thread < replay_channels) {
            #pragma unroll 4
            for (int index = 0; index < count; ++index) {
                weighted_value = fmaf(token_weight[index], values[index][column],
                                      history_weight[index] * weighted_value);
            }
        }
        if (thread == 0) {
            for (int index = 0; index < count; ++index) {
                weight_sum = fmaf(history_weight[index], weight_sum, token_weight[index]);
            }
            denominator = weight_sum;
        }
        __syncthreads();
    }
    if (thread < replay_channels) {
        output[row_offset(0, query_heads, query_head) + channel] =
            __float2bfloat16_rn(weighted_value / denominator);
    }
}

template __device__ void ordered_replay<true>(
    Mxfp8KvCache, const std::int32_t*, int, const float*, const __nv_bfloat16*, __nv_bfloat16*);
template __device__ void ordered_replay<false>(
    Mxfp8KvCache, const std::int32_t*, int, const float*, const __nv_bfloat16*, __nv_bfloat16*);

template <typename... Parameters>
cudaError_t launch_gqa_kernel(void (*kernel)(Parameters...), dim3 grid, dim3 block,
                              cudaStream_t stream,
                              std::enable_if_t<std::is_trivially_copyable_v<Parameters>,
                                               Parameters>... arguments) {
    void* packed[] = {&arguments...};
    return cudaLaunchKernel(kernel, grid, block, packed, 0, stream);
}

}

namespace {

using namespace gqa_device;
using decoder::cache_covers;

__device__ Mxfp8KvCache cache_value(Mxfp8KvCache cache) { return cache; }
__device__ Mxfp8KvCache cache_value(const Mxfp8KvCache* cache) { return *cache; }

__device__ const std::int32_t* page_values(const std::int32_t* pages) { return pages; }
__device__ const std::int32_t* page_values(const Mxfp8PageTable* pages) {
    return pages->device_pages;
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

template <bool Narrow, bool SkipTargetSplit, typename Position, typename Cache, typename Pages,
          typename Scores>
__global__ void exact_scores_kernel(
    Cache cache_binding, Pages page_binding, Position position,
    const __nv_bfloat16* queries, Scores score_binding, int split_end) {
    const int last = gqa_device::position_value(position);
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

template <bool SkipTargetSplit, typename Position, typename Cache, typename Pages, typename Scores>
__global__ void vector_replay_kernel(Cache cache_binding, Pages page_binding, Position position,
                                     Scores score_binding, __nv_bfloat16* output, int split_end) {
    const auto cache = cache_value(cache_binding);
    const auto* pages = page_values(page_binding);
    const int last = gqa_device::position_value(position);
    if constexpr (SkipTargetSplit) {
        if (decoder::target_split_position(last, split_end)) {
            return;
        }
    }
    const auto workspace = score_workspace(score_binding);
    const int head = blockIdx.x;
    if (head >= gqa_query_heads(cache.kv_heads)) {
        return;
    }
    vector_replay_body(cache, pages, last, workspace, output, head, blockIdx.y);
}

template <bool Narrow, typename Position, typename Cache, typename Pages, typename Scores>
__global__ void exact_replay_kernel(
    Cache cache_binding, Pages page_binding, Position position,
    Scores score_binding, const __nv_bfloat16* queries, __nv_bfloat16* output) {
    const int last = gqa_device::position_value(position);
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

constexpr int prefill_replay_threads = 128;

struct PrefillReplayTile {
    alignas(16) std::uint8_t values[replay_tile][replay_channels];
    alignas(8) std::uint8_t exponents[replay_tile][mxfp8_groups];
};

__device__ void load_prefill_replay(PrefillReplayTile& tile, Mxfp8KvCache cache,
                                    const std::int32_t* pages, int first, int last,
                                    int kv_head, int first_channel) {
    constexpr int vectors = replay_channels / 16;
    const int token = threadIdx.x / vectors;
    const int column = threadIdx.x % vectors * 16;
    const int page = pages[first / mxfp8_page_size];
    const int slot = (first + token) % mxfp8_page_size;
    const bool visible = first + token <= last;
    const auto payload =
        payload_offset(cache.kv_heads, page, slot, kv_head) + first_channel + column;
    const auto values =
        static_cast<unsigned>(__cvta_generic_to_shared(&tile.values[token][column]));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
                 :: "r"(values), "l"(cache.value_payload + payload), "r"(visible ? 16 : 0)
                 : "memory");
    if (column == 0) {
        const auto scale = scale_offset(cache.kv_heads, page, slot, kv_head, 0);
        const auto exponents =
            static_cast<unsigned>(__cvta_generic_to_shared(tile.exponents[token]));
        asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;"
                     :: "r"(exponents), "l"(cache.value_scales + scale), "r"(visible ? 8 : 0)
                     : "memory");
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
}

__global__ void prefill_replay_kernel(Mxfp8KvCache cache, const std::int32_t* pages, int last,
                                      decoder::ExactGqaWorkspace workspace,
                                      __nv_bfloat16* output) {
    __shared__ PrefillReplayTile tiles[2];
    __shared__ float history_weight[replay_tile], token_weight[replay_tile], denominator;
    __shared__ bool unit_history;
    const int head = blockIdx.x;
    const int thread = threadIdx.x;
    const int first_channel = blockIdx.y * replay_channels;
    const int group = (first_channel + thread) / mxfp8_group_size;
    const int kv_head = head / queries_per_kv_head;
    const int query_heads = gqa_query_heads(cache.kv_heads);
    if (head >= query_heads) {
        return;
    }
    const auto* scores = workspace.scores + head * workspace.capacity;
    float maximum = -CUDART_INF_F;
    float mass = 0.0f;
    float weighted = 0.0f;
    load_prefill_replay(tiles[0], cache, pages, 0, last, kv_head, first_channel);
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncthreads();
    for (int first = 0; first <= last; first += replay_tile) {
        const int count = min(replay_tile, last - first + 1);
        const int stage = first / replay_tile % 2;
        if (first + replay_tile <= last) {
            load_prefill_replay(tiles[1 - stage], cache, pages, first + replay_tile,
                                last, kv_head, first_channel);
        }
        if (thread < replay_tile) {
            const float score = thread < count ? scores[first + thread] : -CUDART_INF_F;
            const bool unit = isfinite(maximum) && __all_sync(warp_mask, score <= maximum);
            if (thread == 0) {
                unit_history = unit;
            }
            float prefix = fmaxf(maximum, score);
            if (!unit) {
                #pragma unroll
                for (int offset = 1; offset < replay_tile; offset *= 2) {
                    const float earlier = __shfl_up_sync(warp_mask, prefix, offset);
                    if (thread >= offset) {
                        prefix = fmaxf(earlier, prefix);
                    }
                }
                const float previous = __shfl_up_sync(warp_mask, prefix, 1);
                history_weight[thread] = expf((thread == 0 ? maximum : previous) - prefix);
                maximum = __shfl_sync(warp_mask, prefix, replay_tile - 1);
            }
            token_weight[thread] = expf(score - prefix);
        }
        __syncthreads();
        if (thread < replay_channels) {
            const auto& tile = tiles[stage];
            if (unit_history) {
                #pragma unroll 4
                for (int token = 0; token < count; ++token) {
                    const float value = decode_e8m0(tile.values[token][thread],
                                                   tile.exponents[token][group]);
                    weighted = fmaf(token_weight[token], value, weighted);
                }
            } else {
                #pragma unroll 4
                for (int token = 0; token < count; ++token) {
                    const float value = decode_e8m0(tile.values[token][thread],
                                                   tile.exponents[token][group]);
                    weighted = fmaf(token_weight[token], value, history_weight[token] * weighted);
                }
            }
        }
        if (thread == 0) {
            for (int token = 0; token < count; ++token) {
                mass = unit_history ? mass + token_weight[token] :
                    fmaf(history_weight[token], mass, token_weight[token]);
            }
        }
        asm volatile("cp.async.wait_group 0;" ::: "memory");
        __syncthreads();
    }
    if (thread == 0) {
        denominator = mass;
    }
    __syncthreads();
    if (thread < replay_channels) {
        output[row_offset(0, query_heads, head) + first_channel + thread] =
            __float2bfloat16_rn(weighted / denominator);
    }
}

int score_grid_capacity(decoder::ExactGqaWorkspace workspace) {
    return workspace.scores ? workspace.capacity : 0;
}

int score_grid_capacity(const decoder::ExactGqaWorkspace* workspace) {
    return workspace ? text_attention_max_tokens : 0;
}

template <bool Narrow = false, bool Prefill = false, bool Vector = false,
          bool SkipTargetSplit = false, typename Position, typename Cache, typename Pages,
          typename Scores>
cudaError_t launch_exact_attention(
    Cache cache, Pages pages, Position position, const __nv_bfloat16* queries,
    __nv_bfloat16* output, Scores workspace, cudaStream_t stream, int split_end = 0) {
    const int capacity = score_grid_capacity(workspace);
    if (!queries || !output || capacity <= 0 || capacity > text_attention_max_tokens) {
        return cudaErrorInvalidValue;
    }
    const int query_heads = [&] {
        if constexpr (std::is_pointer_v<Cache>) {
            return mxfp8_query_heads;
        } else {
            return gqa_query_heads(cache.kv_heads);
        }
    }();
    const int blocks = std::min(32, (capacity + score_warps - 1) / score_warps);
    const auto status = launch_gqa_kernel(
        exact_scores_kernel<Narrow, SkipTargetSplit, Position, Cache, Pages, Scores>,
        dim3(blocks, query_heads), score_warps * mxfp8_group_size, stream,
        cache, pages, position, queries, workspace, split_end);
    if (status != cudaSuccess) {
        return status;
    }
    if constexpr (Prefill) {
        return launch_gqa_kernel(prefill_replay_kernel,
            dim3(query_heads, mxfp8_head_dim / replay_channels),
            prefill_replay_threads, stream, cache, pages, position, workspace, output);
    } else if constexpr (Vector) {
        return launch_gqa_kernel(
            vector_replay_kernel<SkipTargetSplit, Position, Cache, Pages, Scores>,
            dim3(query_heads, mxfp8_head_dim / vector_replay_channels),
            vector_replay_threads, stream, cache, pages, position, workspace, output, split_end);
    } else {
        return launch_gqa_kernel(
            exact_replay_kernel<Narrow, Position, Cache, Pages, Scores>,
            dim3(query_heads, mxfp8_head_dim / replay_channels),
            replay_threads, stream, cache, pages, position, workspace, queries, output);
    }
}

}

namespace gqa_device {

cudaError_t launch_target_fallback(decoder::BoundPagedDecodeLane lane, int end,
                                   cudaStream_t stream) {
    return launch_exact_attention<false, false, true, true>(
        lane.cache, lane.pages, lane.position, lane.buffers.queries, lane.buffers.output,
        lane.scores, stream, end);
}

cudaError_t launch_draft_range_scores(Mxfp8KvCache cache, const std::int32_t* pages, int count,
                                      const __nv_bfloat16* queries,
                                      decoder::ExactGqaWorkspace workspace, cudaStream_t stream) {
    const int blocks = std::min(32, (count + score_warps - 1) / score_warps);
    const auto scores = exact_scores_kernel<
        false, false, int, Mxfp8KvCache, const std::int32_t*, decoder::ExactGqaWorkspace>;
    return launch_gqa_kernel(scores, dim3(blocks, gqa_query_heads(cache.kv_heads)),
        score_warps * mxfp8_group_size, stream, cache, pages, count - 1, queries, workspace, 0);
}

cudaError_t launch_draft_vector_attention(Mxfp8KvCache cache, const std::int32_t* pages,
                                          int position, const __nv_bfloat16* queries,
                                          __nv_bfloat16* output,
                                          decoder::ExactGqaWorkspace workspace,
                                          cudaStream_t stream) {
    return launch_exact_attention<false, false, true>(
        cache, pages, position, queries, output, workspace, stream);
}

using Bf16 = __nv_bfloat16;

template cudaError_t launch_gqa_kernel(
    void (*)(Mxfp8KvCache, const std::int32_t*, int, const Bf16*, const Bf16*),
    dim3, dim3, cudaStream_t, Mxfp8KvCache, const std::int32_t*, int, const Bf16*, const Bf16*);
template cudaError_t launch_gqa_kernel(
    void (*)(Mxfp8KvCache, const std::int32_t*, const int*, const Bf16*, const Bf16*),
    dim3, dim3, cudaStream_t, Mxfp8KvCache, const std::int32_t*, const int*, const Bf16*,
    const Bf16*);
template cudaError_t launch_gqa_kernel(
    void (*)(const Mxfp8KvCache*, const Mxfp8PageTable*, const int*, const Bf16*, const Bf16*),
    dim3, dim3, cudaStream_t, const Mxfp8KvCache*, const Mxfp8PageTable*, const int*, const Bf16*,
    const Bf16*);
template cudaError_t launch_gqa_kernel(
    void (*)(Mxfp8KvCache, const std::int32_t*, int, const Bf16*, Bf16*),
    dim3, dim3, cudaStream_t, Mxfp8KvCache, const std::int32_t*, int, const Bf16*, Bf16*);
template cudaError_t launch_gqa_kernel(
    void (*)(Mxfp8KvCache, const std::int32_t*, int, const float*, float*, bool, bool, Bf16*),
    dim3, dim3, cudaStream_t, Mxfp8KvCache, const std::int32_t*, int, const float*, float*,
    bool, bool, Bf16*);
template cudaError_t launch_gqa_kernel(
    void (*)(const decoder::tp_verify_graph::GqaBinding*), dim3, dim3, cudaStream_t,
    const decoder::tp_verify_graph::GqaBinding*);

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

}
