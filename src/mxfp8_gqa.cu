#include "sglang/mxfp8_gqa.h"

#include "sglang/decoder_cuda.cuh"

#include <algorithm>
#include <limits>

#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <math_constants.h>

namespace sglang {
namespace {

constexpr unsigned int warp_mask = 0xffffffffu;
constexpr int queries_per_kv_head = mxfp8_query_heads / mxfp8_kv_heads;
constexpr float softmax_scale = 1.0f / 16.0f;

__device__ std::size_t row_offset(std::size_t token, int heads, int head) {
    return (token * heads + head) * mxfp8_head_dim;
}

__device__ std::size_t payload_offset(int physical_page, int token, int head) {
    return static_cast<std::size_t>(physical_page) * mxfp8_payload_bytes_per_page
         + row_offset(token, mxfp8_kv_heads, head);
}

__device__ std::size_t scale_offset(int physical_page, int token, int head, int group) {
    constexpr int token_groups = mxfp8_page_size / mxfp8_group_size;
    const auto page_head = static_cast<std::size_t>(physical_page) * mxfp8_kv_heads + head;
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

template <typename Position, typename Cache, typename Pages>
__global__ void append_kernel(Cache cache_binding, Pages page_binding,
                              Position first_position,
                              const __nv_bfloat16* keys, const __nv_bfloat16* values) {
    const auto cache = cache_value(cache_binding);
    const auto* pages = page_values(page_binding);
    const int position = decoder::position_value(first_position) + static_cast<int>(blockIdx.x);
    const int token = position % mxfp8_page_size;
    const int physical_page = pages[position / mxfp8_page_size];
    const int head = blockIdx.y;
    const int channel = threadIdx.x;
    const auto input = row_offset(blockIdx.x, mxfp8_kv_heads, head) + channel;
    const auto payload = payload_offset(physical_page, token, head) + channel;
    const float key = __bfloat162float(keys[input]);
    const float value = __bfloat162float(values[input]);
    const int key_exponent = scale_exponent(key);
    const int value_exponent = scale_exponent(value);
    cache.key_payload[payload] = encode(key, key_exponent);
    cache.value_payload[payload] = encode(value, value_exponent);
    if (channel % mxfp8_group_size == 0) {
        const auto scale = scale_offset(physical_page, token, head, channel / mxfp8_group_size);
        cache.key_scales[scale] = static_cast<std::uint8_t>(key_exponent);
        cache.value_scales[scale] = static_cast<std::uint8_t>(value_exponent);
    }
}

__device__ float warp_sum(float value) {
    for (int delta = mxfp8_group_size / 2; delta > 0; delta /= 2) {
        value += __shfl_down_sync(warp_mask, value, delta);
    }
    return value;
}

template <typename Position>
__global__ void attention_kernel(Mxfp8KvCache cache, const std::int32_t* pages,
                                 Position first_query_position, const __nv_bfloat16* queries,
                                 __nv_bfloat16* output) {
    __shared__ float warp_dots[mxfp8_groups];
    __shared__ float history_weight;
    __shared__ float token_weight;
    __shared__ float denominator;

    const int query_head = blockIdx.y;
    const int kv_head = query_head / queries_per_kv_head;
    const int channel = threadIdx.x;
    const int group = channel / mxfp8_group_size;
    const auto row = row_offset(blockIdx.x, mxfp8_query_heads, query_head) + channel;
    const int last_position =
        decoder::position_value(first_query_position) + static_cast<int>(blockIdx.x);
    const float query = __bfloat162float(queries[row]);
    float max_score = -CUDART_INF_F;
    float weight_sum = 0.0f;
    float weighted_value = 0.0f;

    for (int position = 0; position <= last_position; ++position) {
        const int physical_page = pages[position / mxfp8_page_size];
        const int token = position % mxfp8_page_size;
        const auto payload = payload_offset(physical_page, token, kv_head) + channel;
        const auto scale = scale_offset(physical_page, token, kv_head, group);
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

bool cache_covers(Mxfp8KvCache cache, Mxfp8PageTable pages, std::int64_t token_count) {
    if (token_count <= 0 || token_count > std::numeric_limits<int>::max()) {
        return false;
    }
    const auto last_page = (token_count - 1) / mxfp8_page_size;
    return last_page < cache.page_count && last_page < pages.page_count && pages.device_pages
        && cache.key_payload && cache.value_payload && cache.key_scales && cache.value_scales;
}

constexpr int score_warps = 8;
constexpr int replay_tile = 32;
static_assert(mxfp8_page_size % replay_tile == 0 && replay_tile == mxfp8_group_size);

template <typename Position, typename Cache, typename Pages>
__global__ void exact_scores_kernel(
    Cache cache_binding, Pages page_binding, Position position,
    const __nv_bfloat16* queries, decoder::ExactGqaWorkspace workspace) {
    const int warp = threadIdx.x / mxfp8_group_size;
    const int lane = threadIdx.x % mxfp8_group_size;
    const int query_head = blockIdx.y;
    const int kv_head = query_head / queries_per_kv_head;
    const int last = decoder::position_value(position);
    const int first = blockIdx.x * score_warps + warp;
    if (first > last) {
        return;
    }
    const auto cache = cache_value(cache_binding);
    const auto* pages = page_values(page_binding);
    float query[mxfp8_groups];
    #pragma unroll
    for (int group = 0; group < mxfp8_groups; ++group) {
        const int channel = group * mxfp8_group_size + lane;
        query[group] =
            __bfloat162float(queries[row_offset(0, mxfp8_query_heads, query_head) + channel]);
    }
    auto* scores = workspace.scores + query_head * workspace.capacity;
    const int stride = gridDim.x * score_warps;
    for (int index = first; index <= last; index += stride) {
        const int physical_page = pages[index / mxfp8_page_size];
        const int token = index % mxfp8_page_size;
        const auto payload = payload_offset(physical_page, token, kv_head) + lane;
        const auto scale = scale_offset(physical_page, token, kv_head, 0);
        float dot = 0.0f;
        #pragma unroll
        for (int group = 0; group < mxfp8_groups; ++group) {
            const float key = decode(cache.key_payload[payload + group * mxfp8_group_size],
                                     cache.key_scales[scale + group]);
            const float product = warp_sum(query[group] * key);
            if (lane == 0) {
                dot += product;
            }
        }
        if (lane == 0) {
            scores[index] = dot * softmax_scale;
        }
    }
}

template <typename Position, typename Cache, typename Pages>
__global__ void exact_replay_kernel(
    Cache cache_binding, Pages page_binding, Position position,
    decoder::ExactGqaWorkspace workspace, __nv_bfloat16* output) {
    __shared__ float prefix_maximum[replay_tile + 1];
    __shared__ float history_weight[replay_tile];
    __shared__ float token_weight[replay_tile];
    __shared__ float denominator;

    const int query_head = blockIdx.x;
    const int kv_head = query_head / queries_per_kv_head;
    const int channel = threadIdx.x;
    const int group = channel / mxfp8_group_size;
    const int last = decoder::position_value(position);
    const auto cache = cache_value(cache_binding);
    const auto* pages = page_values(page_binding);
    const auto* scores = workspace.scores + query_head * workspace.capacity;
    float maximum = -CUDART_INF_F;
    float weight_sum = 0.0f;
    float weighted_value = 0.0f;

    for (int first = 0; first <= last; first += replay_tile) {
        const int count = min(replay_tile, last - first + 1);
        if (channel == 0) {
            prefix_maximum[0] = maximum;
            for (int index = 0; index < count; ++index) {
                maximum = fmaxf(maximum, scores[first + index]);
                prefix_maximum[index + 1] = maximum;
            }
        }
        const int physical_page = pages[first / mxfp8_page_size];
        const int token = first % mxfp8_page_size;
        const auto payload = payload_offset(physical_page, token, kv_head) + channel;
        const auto scale = scale_offset(physical_page, token, kv_head, group);
        __syncthreads();

        if (channel < count) {
            const float next_maximum = prefix_maximum[channel + 1];
            history_weight[channel] = expf(prefix_maximum[channel] - next_maximum);
            token_weight[channel] = expf(scores[first + channel] - next_maximum);
        }
        __syncthreads();

        #pragma unroll 4
        for (int index = 0; index < count; ++index) {
            const auto payload_index = payload + index * mxfp8_kv_heads * mxfp8_head_dim;
            const auto scale_index = scale + index * (mxfp8_page_size / mxfp8_group_size)
                                                  * mxfp8_groups;
            const float value =
                decode(cache.value_payload[payload_index], cache.value_scales[scale_index]);
            weighted_value =
                fmaf(token_weight[index], value, history_weight[index] * weighted_value);
        }
        if (channel == 0) {
            for (int index = 0; index < count; ++index) {
                weight_sum = fmaf(history_weight[index], weight_sum, token_weight[index]);
            }
            denominator = weight_sum;
        }
        __syncthreads();
    }
    output[row_offset(0, mxfp8_query_heads, query_head) + channel] =
        __float2bfloat16_rn(weighted_value / denominator);
}

template <typename Position, typename Cache, typename Pages>
cudaError_t launch_exact_attention(
    Cache cache, Pages pages, Position position, const __nv_bfloat16* queries,
    __nv_bfloat16* output, decoder::ExactGqaWorkspace workspace, cudaStream_t stream) {
    if (!queries || !output || !workspace.scores ||
        workspace.capacity <= 0 || workspace.capacity > text_attention_max_tokens) {
        return cudaErrorInvalidValue;
    }
    constexpr int max_score_blocks = 32;
    const int blocks =
        std::min(max_score_blocks, (workspace.capacity + score_warps - 1) / score_warps);
    void* score_arguments[] = {&cache, &pages, &position, &queries, &workspace};
    const auto status = cudaLaunchKernel(exact_scores_kernel<Position, Cache, Pages>,
                                         dim3(blocks, mxfp8_query_heads),
                                         score_warps * mxfp8_group_size, score_arguments,
                                         0, stream);
    if (status != cudaSuccess) {
        return status;
    }
    void* replay_arguments[] = {&cache, &pages, &position, &workspace, &output};
    return cudaLaunchKernel(exact_replay_kernel<Position, Cache, Pages>,
                            mxfp8_query_heads, mxfp8_head_dim,
                            replay_arguments, 0, stream);
}

}

cudaError_t mxfp8_kv_append(Mxfp8KvCache cache, Mxfp8PageTable pages, int first_position,
                           int token_count, const __nv_bfloat16* keys,
                           const __nv_bfloat16* values, cudaStream_t stream) {
    if (token_count == 0) {
        return cudaSuccess;
    }
    const auto end = static_cast<std::int64_t>(first_position) + token_count;
    if (first_position < 0 || token_count < 0 || !keys || !values
        || !cache_covers(cache, pages, end)) {
        return cudaErrorInvalidValue;
    }
    void* arguments[] = {&cache, &pages.device_pages, &first_position, &keys, &values};
    const auto kernel = append_kernel<int, Mxfp8KvCache, const std::int32_t*>;
    return cudaLaunchKernel(kernel, dim3(token_count, mxfp8_kv_heads),
                            dim3(mxfp8_head_dim), arguments, 0, stream);
}

cudaError_t mxfp8_gqa_attention(Mxfp8KvCache cache, Mxfp8PageTable pages, int valid_tokens,
                               int first_query_position, int query_count,
                               const __nv_bfloat16* queries, __nv_bfloat16* output,
                               cudaStream_t stream) {
    if (query_count == 0) {
        return cudaSuccess;
    }
    const auto end = static_cast<std::int64_t>(first_query_position) + query_count;
    if (first_query_position < 0 || query_count < 0 || end > valid_tokens || !queries || !output
        || !cache_covers(cache, pages, valid_tokens)) {
        return cudaErrorInvalidValue;
    }
    void* arguments[] = {&cache, &pages.device_pages, &first_query_position, &queries, &output};
    return cudaLaunchKernel(attention_kernel<int>, dim3(query_count, mxfp8_query_heads),
                            dim3(mxfp8_head_dim), arguments, 0, stream);
}

cudaError_t decoder::decode_kv_append(Mxfp8KvCache cache, Mxfp8PageTable pages,
                                      const int* position, const __nv_bfloat16* keys,
                                      const __nv_bfloat16* values, cudaStream_t stream) {
    void* arguments[] = {&cache, &pages.device_pages, &position, &keys, &values};
    const auto kernel = append_kernel<const int*, Mxfp8KvCache, const std::int32_t*>;
    return cudaLaunchKernel(kernel, dim3(1, mxfp8_kv_heads),
                            dim3(mxfp8_head_dim), arguments, 0, stream);
}

cudaError_t decoder::decode_gqa_attention(Mxfp8KvCache cache, Mxfp8PageTable pages,
                                          const int* position, const __nv_bfloat16* queries,
                                          __nv_bfloat16* output, cudaStream_t stream) {
    void* arguments[] = {&cache, &pages.device_pages, &position, &queries, &output};
    return cudaLaunchKernel(attention_kernel<const int*>, dim3(1, mxfp8_query_heads),
                            dim3(mxfp8_head_dim), arguments, 0, stream);
}

cudaError_t decoder::exact_decode_gqa_attention(
    Mxfp8KvCache cache, Mxfp8PageTable pages, int position, const __nv_bfloat16* queries,
    __nv_bfloat16* output, ExactGqaWorkspace workspace, cudaStream_t stream) {
    if (position < 0 || position >= workspace.capacity) {
        return cudaErrorInvalidValue;
    }
    if (!cache_covers(cache, pages, workspace.capacity)) {
        return cudaErrorInvalidValue;
    }
    return launch_exact_attention(cache, pages.device_pages, position, queries, output,
                                  workspace, stream);
}

cudaError_t decoder::exact_decode_gqa_attention(
    Mxfp8KvCache cache, Mxfp8PageTable pages, const int* position, const __nv_bfloat16* queries,
    __nv_bfloat16* output, ExactGqaWorkspace workspace, cudaStream_t stream) {
    if (!position) {
        return cudaErrorInvalidValue;
    }
    if (!cache_covers(cache, pages, workspace.capacity)) {
        return cudaErrorInvalidValue;
    }
    return launch_exact_attention(cache, pages.device_pages, position, queries, output,
                                  workspace, stream);
}

cudaError_t decoder::decode_kv_append(const Mxfp8KvCache* cache, const Mxfp8PageTable* pages,
                                      const int* position, const __nv_bfloat16* keys,
                                      const __nv_bfloat16* values, cudaStream_t stream) {
    void* arguments[] = {&cache, &pages, &position, &keys, &values};
    const auto kernel = append_kernel<const int*, const Mxfp8KvCache*, const Mxfp8PageTable*>;
    return cudaLaunchKernel(kernel, dim3(1, mxfp8_kv_heads), mxfp8_head_dim, arguments, 0, stream);
}

cudaError_t decoder::exact_decode_gqa_attention(
    const Mxfp8KvCache* cache, const Mxfp8PageTable* pages, const int* position,
    const __nv_bfloat16* queries, __nv_bfloat16* output, ExactGqaWorkspace workspace,
    cudaStream_t stream) {
    return launch_exact_attention(cache, pages, position, queries, output, workspace, stream);
}

}
