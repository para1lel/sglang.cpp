#include "sglang/text/gqa/decode_kernels.cuh"
#include "sglang/text/gqa_verify_batch.h"
#include <algorithm>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <math_constants.h>

namespace sglang {
namespace {

using namespace gqa_device;
using decoder::cache_covers;

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
    const int position = gqa_device::position_value(first_position) + static_cast<int>(blockIdx.x);
    append_head(cache_value(cache_binding), page_values(page_binding), position, blockIdx.x,
                keys, values);
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

namespace {

template <int Capacity>
__global__ void gqa_verify_append_kernel(
    const __grid_constant__ decoder::GqaVerifyRows<decoder::GqaVerifyAppendRow, Capacity> batch) {
    const auto& row = batch.rows[blockIdx.x];
    append_head(row.cache, row.pages.device_pages, row.position, 0, row.keys, row.values);
}

__global__ void
live_append(const __grid_constant__
                decoder::GqaVerifyRows<decoder::GqaVerifyAppendRow, decoder::max_wide_mtp_rows>
                    batch,
            const decoder::PagedDecodeLane* reads) {
    const auto& row = batch.rows[blockIdx.x];
    const auto& read = reads[blockIdx.x];
    append_head(read.cache, read.pages.device_pages, read.position, 0, row.keys, row.values);
}
}

template <int Capacity>
cudaError_t decoder::gqa_verify_append(const GqaVerifyRows<GqaVerifyAppendRow, Capacity>& batch,
                                       int count, cudaStream_t stream) {
    if (count <= 0 || count > Capacity) {
        return cudaErrorInvalidValue;
    }
    for (int index = 0; index < count; ++index) {
        const auto& row = batch.rows[index];
        if (row.position < 0 || !row.keys || !row.values ||
            row.cache.kv_heads != batch.rows[0].cache.kv_heads ||
            !cache_covers(row.cache, row.pages, static_cast<std::int64_t>(row.position) + 1)) {
            return cudaErrorInvalidValue;
        }
    }
    void* arguments[] = {const_cast<GqaVerifyRows<GqaVerifyAppendRow, Capacity>*>(&batch)};
    return cudaLaunchKernel(gqa_verify_append_kernel<Capacity>,
                            dim3(count, batch.rows[0].cache.kv_heads),
                            mxfp8_head_dim, arguments, 0, stream);
}

cudaError_t
decoder::gqa_verify_append_bound(const GqaVerifyRows<GqaVerifyAppendRow, max_wide_mtp_rows>& batch,
                                 const PagedDecodeLane* reads, int rows, int kv_heads,
                                 cudaStream_t stream) {
    live_append<<<dim3(rows, kv_heads), mxfp8_head_dim, 0, stream>>>(batch, reads);
    return cudaPeekAtLastError();
}

template cudaError_t decoder::gqa_verify_append(
    const GqaVerifyRows<GqaVerifyAppendRow>&, int, cudaStream_t);
template cudaError_t decoder::gqa_verify_append(
    const GqaVerifyRows<GqaVerifyAppendRow, max_wide_mtp_rows>&, int, cudaStream_t);

}
