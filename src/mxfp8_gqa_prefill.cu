#include "sglang/mxfp8_gqa.h"

#include "sglang/decoder_cuda.cuh"

#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <climits>
#include <math_constants.h>

namespace sglang {
namespace {

using Bf16 = __nv_bfloat16;
using Word = unsigned int;
constexpr int channels = mxfp8_head_dim;
constexpr int ratio = mxfp8_query_heads / mxfp8_kv_heads;
constexpr int ordered_heads = 2;
constexpr int keys_per_tile = 32;
constexpr int max_multirow_visible_tokens = 32768;
constexpr Word warp_mask = 0xffffffffu;
constexpr float softmax_scale = 1.0f / 16.0f;

static_assert(channels == 256 && ratio == 6 && mxfp8_page_size % keys_per_tile == 0);
static_assert(keys_per_tile == mxfp8_group_size && ratio % ordered_heads == 0);

template <typename Real, int Heads = ordered_heads>
struct OrderedTile {
    Real scores[Heads][keys_per_tile];
    Real history_weight[Heads][keys_per_tile];
    Real token_weight[Heads][keys_per_tile];
    Real denominator[Heads];
};

struct OrderedScratch {
    OrderedTile<float> ordered;
    OrderedTile<double> wide;
};

struct Attention {
    Mxfp8KvCache cache;
    const std::int32_t* pages;
    int position;
    const Bf16* queries;
    Bf16* output;
};

__device__ std::size_t row_offset(int row, int head) {
    return (static_cast<std::size_t>(row) * mxfp8_query_heads + head) * channels;
}

__device__ std::size_t payload_offset(int page, int token, int head) {
    return static_cast<std::size_t>(page) * mxfp8_payload_bytes_per_page
        + (token * mxfp8_kv_heads + head) * channels;
}

__device__ std::size_t scale_offset(int page, int token, int head) {
    constexpr int token_groups = mxfp8_page_size / mxfp8_group_size;
    const auto page_head = static_cast<std::size_t>(page) * mxfp8_kv_heads + head;
    const auto token_lane = page_head * mxfp8_group_size + token % mxfp8_group_size;
    return (token_lane * token_groups + token / mxfp8_group_size) * mxfp8_groups;
}

__device__ float ordered_decode(std::uint8_t payload, std::uint8_t exponent) {
    __nv_fp8_e4m3 value;
    value.__x = payload;
    if (exponent == 255) {
        return ldexpf(static_cast<float>(value), static_cast<int>(exponent) - 127);
    }
    const auto scale_bits = exponent == 0 ? 0x00400000u :
                                           static_cast<Word>(exponent) << 23;
    return __fmul_rn(static_cast<float>(value), __uint_as_float(scale_bits));
}

template <typename Real>
__device__ Real warp_sum(Real value) {
    for (int delta = mxfp8_group_size / 2; delta > 0; delta /= 2) {
        value += __shfl_down_sync(warp_mask, value, delta);
    }
    return value;
}

template <typename Real>
__device__ __forceinline__ Real ordered_prefix(Real prefix, int lane) {
    #pragma unroll
    for (int offset = 1; offset < keys_per_tile; offset *= 2) {
        const Real earlier = __shfl_up_sync(warp_mask, prefix, offset);
        if (lane >= offset) {
            prefix = fmax(earlier, prefix);
        }
    }
    return prefix;
}

__device__ int store_ordered(Attention request, std::size_t row, int channel,
                             const float (&values)[ordered_heads],
                             const float (&denominators)[ordered_heads], int) {
    int invalid = 0;
    #pragma unroll
    for (int head = 0; head < ordered_heads; ++head) {
        const float output = values[head] / denominators[head];
        request.output[row + head * channels + channel] = __float2bfloat16_rn(output);
        invalid |= static_cast<int>(!isfinite(output)) << head;
    }
    return invalid;
}

__device__ int store_ordered(Attention request, std::size_t row, int channel,
                             const double (&values)[ordered_heads],
                             const double (&denominators)[ordered_heads], int invalid) {
    #pragma unroll
    for (int head = 0; head < ordered_heads; ++head) {
        const double output = values[head] / denominators[head];
        if ((invalid & (1 << head)) && isfinite(output)) {
            request.output[row + head * channels + channel] = __double2bfloat16(output);
        }
    }
    return 0;
}

template <int Heads>
__device__ int score_recovery_mask(int warp, int lane, bool invalid_score) {
    __shared__ int failed[Heads];
    if (warp < Heads) {
        const int failed_head = __any_sync(warp_mask, invalid_score);
        if (lane == 0) {
            failed[warp] = failed_head << warp;
        }
    }
    __syncthreads();
    int mask = 0;
    #pragma unroll
    for (int head = 0; head < Heads; ++head) {
        mask |= failed[head];
    }
    return mask;
}

template <typename Real, bool recover_scores = false>
__device__ int ordered_attention(OrderedTile<Real>& shared, Attention request, int row_index,
                                 int query_head, int invalid = 0) {
    const int channel = threadIdx.x;
    const int warp = channel / mxfp8_group_size;
    const int lane = channel % mxfp8_group_size;
    const int kv_head = query_head / ratio;
    const auto row = row_offset(row_index, query_head);
    const int last = request.position + row_index;
    const auto cache = request.cache;
    float query[ordered_heads][mxfp8_groups];
    #pragma unroll
    for (int head = 0; head < ordered_heads; ++head) {
        #pragma unroll
        for (int group = 0; group < mxfp8_groups; ++group) {
            const auto offset = row + head * channels + group * mxfp8_group_size + lane;
            query[head][group] = __bfloat162float(request.queries[offset]);
        }
    }
    Real maximum = -CUDART_INF_F;
    Real mass = 0;
    Real weighted_value[ordered_heads]{};
    bool invalid_score = false;
    for (int first = 0; first <= last; first += keys_per_tile) {
        const int count = min(keys_per_tile, last - first + 1);
        const int page = request.pages[first / mxfp8_page_size];
        const int token = first % mxfp8_page_size;
        const auto payload = payload_offset(page, token, kv_head);
        const auto scale = scale_offset(page, token, kv_head);
        constexpr int payload_stride = mxfp8_kv_heads * channels;
        constexpr int scale_stride = (mxfp8_page_size / mxfp8_group_size) * mxfp8_groups;
        for (int index = warp; index < count; index += channels / mxfp8_group_size) {
            Real dot[ordered_heads]{};
            #pragma unroll
            for (int group = 0; group < mxfp8_groups; ++group) {
                const auto key_index =
                    payload + index * payload_stride + group * mxfp8_group_size;
                const auto key_exponent = cache.key_scales[scale + index * scale_stride + group];
                const Real key = ordered_decode(cache.key_payload[key_index + lane], key_exponent);
                #pragma unroll
                for (int head = 0; head < ordered_heads; ++head) {
                    const Real product = warp_sum(query[head][group] * key);
                    if (lane == 0) {
                        dot[head] += product;
                    }
                }
            }
            if (lane == 0) {
                #pragma unroll
                for (int head = 0; head < ordered_heads; ++head) {
                    shared.scores[head][index] = dot[head] * softmax_scale;
                }
            }
        }
        __syncthreads();
        if (warp < ordered_heads) {
            const Real prior_maximum = maximum;
            const Real score = lane < count ? shared.scores[warp][lane] : -CUDART_INF_F;
            const Real prefix = ordered_prefix(fmax(prior_maximum, score), lane);
            const Real previous = __shfl_up_sync(warp_mask, prefix, 1);
            maximum = __shfl_sync(warp_mask, prefix, count - 1);
            if (lane < count) {
                if constexpr (recover_scores) {
                    invalid_score |= !isfinite(score);
                }
                shared.history_weight[warp][lane] =
                    exp((lane == 0 ? prior_maximum : previous) - prefix);
                shared.token_weight[warp][lane] = exp(score - prefix);
            }
        }
        __syncthreads();
        #pragma unroll 4
        for (int index = 0; index < count; ++index) {
            const Real value = ordered_decode(
                cache.value_payload[payload + index * payload_stride + channel],
                cache.value_scales[scale + index * scale_stride + warp]);
            #pragma unroll
            for (int head = 0; head < ordered_heads; ++head) {
                weighted_value[head] =
                    fma(shared.token_weight[head][index], value,
                        shared.history_weight[head][index] * weighted_value[head]);
            }
        }
        if (warp < ordered_heads && lane == 0) {
            for (int index = 0; index < count; ++index) {
                mass = fma(shared.history_weight[warp][index], mass,
                           shared.token_weight[warp][index]);
            }
            shared.denominator[warp] = mass;
        }
        __syncthreads();
    }
    int failed = store_ordered(request, row, channel, weighted_value, shared.denominator, invalid);
    if constexpr (recover_scores) {
        failed |= score_recovery_mask<ordered_heads>(warp, lane, invalid_score);
    }
    return failed;
}

__device__ __noinline__ void recover_wide(
    OrderedTile<double>& shared, Attention request, int row, int head, int invalid) {
    ordered_attention(shared, request, row, head, invalid);
}

__device__ __forceinline__ void protected_attention(
    OrderedScratch& shared, Attention request, int row, int head) {
    const int invalid = ordered_attention<float, true>(shared.ordered, request, row, head);
    if (__syncthreads_or(invalid != 0)) {
        recover_wide(shared.wide, request, row, head, invalid);
    }
}

__global__ void tiled_prefill_attention(Attention request) {
    __shared__ OrderedTile<float> shared;
    ordered_attention(shared, request, blockIdx.x, blockIdx.y * ordered_heads);
}

__launch_bounds__(channels, 4)
__global__ void protected_prefill_attention(Attention request) {
    __shared__ OrderedScratch shared;
    protected_attention(shared, request, blockIdx.x, blockIdx.y * ordered_heads);
}

constexpr int wide_rows = 4;
constexpr int paired_rows = 2;
constexpr int score_warps = channels / mxfp8_group_size;
constexpr int payload_stride = mxfp8_kv_heads * channels;
constexpr int scale_stride = (mxfp8_page_size / mxfp8_group_size) * mxfp8_groups;
constexpr int score_lanes = 8;
constexpr int groups_per_warp = mxfp8_group_size / score_lanes;
constexpr int group_passes = mxfp8_groups / groups_per_warp;
static_assert(groups_per_warp == 4 && mxfp8_groups % groups_per_warp == 0);

template <int Heads>
__device__ void tile_scores(OrderedTile<float, Heads>& shared, Mxfp8KvCache cache,
                            std::size_t payload, std::size_t scale, int end, int warp, int lane,
                            const float4 (&queries)[group_passes][Heads]) {
    const int group_in_warp = lane / score_lanes;
    for (int token = warp; token < end; token += score_warps) {
        float dots[Heads]{};
        #pragma unroll
        for (int pass = 0; pass < group_passes; ++pass) {
            const int group = pass * groups_per_warp + group_in_warp;
            const auto offset = payload + token * payload_stride + group * mxfp8_group_size
                + lane % score_lanes;
            const auto exponent = cache.key_scales[scale + token * scale_stride + group];
            const float key0 = ordered_decode(cache.key_payload[offset], exponent);
            const float key1 = ordered_decode(cache.key_payload[offset + score_lanes], exponent);
            const float key2 = ordered_decode(
                cache.key_payload[offset + 2 * score_lanes], exponent);
            const float key3 = ordered_decode(
                cache.key_payload[offset + 3 * score_lanes], exponent);
            #pragma unroll
            for (int head = 0; head < Heads; ++head) {
                const auto query = queries[pass][head];
                const float low = __fadd_rn(__fmul_rn(query.x, key0), __fmul_rn(query.z, key2));
                const float high = __fadd_rn(__fmul_rn(query.y, key1), __fmul_rn(query.w, key3));
                float value = low + high;
                #pragma unroll
                for (int delta = score_lanes / 2; delta > 0; delta /= 2) {
                    value += __shfl_down_sync(warp_mask, value, delta, score_lanes);
                }
                dots[head] += value;
                #pragma unroll
                for (int group_lane = score_lanes; group_lane < mxfp8_group_size;
                     group_lane += score_lanes) {
                    dots[head] += __shfl_sync(warp_mask, value, group_lane);
                }
            }
        }
        if (lane == 0) {
            #pragma unroll
            for (int head = 0; head < Heads; ++head) {
                shared.scores[head][token] = dots[head] * softmax_scale;
            }
        }
    }
}

template <int Heads>
__device__ void tile_weights(OrderedTile<float, Heads>& shared, int count, int warp, int lane,
                             float& maximum,
                             bool& invalid_score) {
    if (warp >= Heads || count == 0) {
        return;
    }
    const float prior = maximum;
    const float score = lane < count ? shared.scores[warp][lane] : -CUDART_INF_F;
    const float prefix = ordered_prefix(fmaxf(prior, score), lane);
    const float previous = __shfl_up_sync(warp_mask, prefix, 1);
    maximum = __shfl_sync(warp_mask, prefix, count - 1);
    if (lane < count) {
        invalid_score |= !isfinite(score);
        shared.history_weight[warp][lane] = expf((lane == 0 ? prior : previous) - prefix);
        shared.token_weight[warp][lane] = expf(score - prefix);
    }
}

template <int Heads>
__device__ void tile_values(OrderedTile<float, Heads>& shared, Mxfp8KvCache cache,
                            std::size_t payload, std::size_t scale, int first_visible, int end,
                            int channel, int warp, float (&weighted)[Heads]) {
    const int count = max(0, min(keys_per_tile, first_visible));
    #pragma unroll 4
    for (int token = 0; token < count; ++token) {
        const float value = ordered_decode(cache.value_payload[payload + token * payload_stride
                                                                + channel],
                                          cache.value_scales[scale + token * scale_stride + warp]);
        #pragma unroll
        for (int head = 0; head < Heads; ++head) {
            weighted[head] = fmaf(shared.token_weight[head][token], value,
                                  shared.history_weight[head][token] * weighted[head]);
        }
    }
    for (int token = count; token < end; ++token) {
        const float value = ordered_decode(cache.value_payload[payload + token * payload_stride
                                                                + channel],
                                          cache.value_scales[scale + token * scale_stride + warp]);
        #pragma unroll
        for (int head = ordered_heads; head < Heads; ++head) {
            if (token < first_visible + head / ordered_heads) {
                weighted[head] = fmaf(shared.token_weight[head][token], value,
                                      shared.history_weight[head][token] * weighted[head]);
            }
        }
    }
}

template <int Heads>
__device__ int tile_attention(OrderedTile<float, Heads>& shared, Attention request, int row,
                             int head) {
    static_assert(Heads <= score_warps);
    constexpr int tile_rows = Heads / ordered_heads;
    const int channel = threadIdx.x;
    const int warp = channel / mxfp8_group_size;
    const int lane = channel % mxfp8_group_size;
    const int kv_head = head / ratio;
    const int last = request.position + row;
    float4 query[group_passes][Heads];
    float weighted[Heads]{};
    #pragma unroll
    for (int member = 0; member < Heads; ++member) {
        const auto offset = row_offset(row + member / ordered_heads,
                                       head + member % ordered_heads);
        #pragma unroll
        for (int pass = 0; pass < group_passes; ++pass) {
            const int group = pass * groups_per_warp + lane / score_lanes;
            const auto index = offset + group * mxfp8_group_size + lane % score_lanes;
            query[pass][member] = {__bfloat162float(request.queries[index]),
                                   __bfloat162float(request.queries[index + score_lanes]),
                                   __bfloat162float(request.queries[index + 2 * score_lanes]),
                                   __bfloat162float(request.queries[index + 3 * score_lanes])};
        }
    }
    float mass = 0.0f;
    float maximum = -CUDART_INF_F;
    bool invalid_score = false;
    for (int first = 0; first < last + tile_rows; first += keys_per_tile) {
        const int first_visible = last - first + 1;
        const int end = min(keys_per_tile, last - first + tile_rows);
        const int visible = max(0, min(keys_per_tile, first_visible + warp / ordered_heads));
        const int page = request.pages[first / mxfp8_page_size];
        const int token = first % mxfp8_page_size;
        const auto payload = payload_offset(page, token, kv_head);
        const auto scale = scale_offset(page, token, kv_head);
        tile_scores(shared, request.cache, payload, scale, end, warp, lane, query);
        __syncthreads();
        tile_weights(shared, visible, warp, lane, maximum, invalid_score);
        __syncthreads();
        tile_values(shared, request.cache, payload, scale, first_visible, end, channel, warp,
                    weighted);
        if (warp < Heads && lane == 0) {
            for (int index = 0; index < visible; ++index) {
                mass = fmaf(shared.history_weight[warp][index], mass,
                            shared.token_weight[warp][index]);
            }
            shared.denominator[warp] = mass;
        }
        __syncthreads();
    }
    int invalid = score_recovery_mask<Heads>(warp, lane, invalid_score);
    #pragma unroll
    for (int member = 0; member < Heads; ++member) {
        const auto offset = row_offset(row + member / ordered_heads,
                                       head + member % ordered_heads);
        const float result = weighted[member] / shared.denominator[member];
        request.output[offset + channel] = __float2bfloat16_rn(result);
        invalid |= static_cast<int>(!isfinite(result)) << member;
    }
    return invalid;
}

template <int Rows, int Blocks>
__launch_bounds__(channels, Blocks)
__global__ void multirow_prefill_attention(Attention request, int rows) {
    __shared__ OrderedTile<float, Rows * ordered_heads> shared;
    __shared__ OrderedScratch fallback;
    const int tile = blockIdx.x;
    const int full_tiles = rows / Rows;
    const int head = blockIdx.y * ordered_heads;
    if (tile >= full_tiles) {
        protected_attention(fallback, request, full_tiles * Rows + tile - full_tiles, head);
        return;
    }
    const int row = tile * Rows;
    const int invalid = tile_attention(shared, request, row, head);
    if (__syncthreads_or(invalid != 0)) {
        constexpr int mask = (1 << ordered_heads) - 1;
        for (int offset = 0; offset < Rows; ++offset) {
            const int failed = (invalid >> (offset * ordered_heads)) & mask;
            if (__syncthreads_or(failed != 0)) {
                recover_wide(fallback.wide, request, row + offset, head, failed);
            }
        }
    }
}

}

cudaError_t decoder::tensor_core_prefill_gqa(
    Mxfp8KvCache cache, Mxfp8PageTable pages, int position, int rows,
    const Bf16* queries, Bf16* output, cudaStream_t stream) {
    Attention request{cache, pages.device_pages, position, queries, output};
    void* args[] = {&request};
    const dim3 blocks(rows, mxfp8_query_heads / ordered_heads);
    return cudaLaunchKernel(tiled_prefill_attention, blocks, channels, args, 0, stream);
}

cudaError_t decoder::tensor_core_multirow_prefill_gqa(
    Mxfp8KvCache cache, Mxfp8PageTable pages, int position, int rows,
    const Bf16* queries, Bf16* output, cudaStream_t stream) {
    if (rows == 0) {
        return cudaSuccess;
    }
    if (position < 0 || rows < 0 || position > INT_MAX - rows) {
        return cudaErrorInvalidValue;
    }
    Attention request{cache, pages.device_pages, position, queries, output};
    void* args[] = {&request, &rows};
    constexpr int min_paired_rows = 64;
    if (rows < min_paired_rows || position + rows > max_multirow_visible_tokens) {
        const dim3 blocks(rows, mxfp8_query_heads / ordered_heads);
        return cudaLaunchKernel(protected_prefill_attention, blocks, channels, args, 0, stream);
    }
    constexpr int min_wide_rows = 128;
    const int tile_rows = rows >= min_wide_rows ? wide_rows : paired_rows;
    const dim3 blocks(rows / tile_rows + rows % tile_rows, mxfp8_query_heads / ordered_heads);
    const auto kernel = tile_rows == wide_rows ? multirow_prefill_attention<wide_rows, 2> :
                                                multirow_prefill_attention<paired_rows, 3>;
    return cudaLaunchKernel(kernel, blocks, channels, args, 0, stream);
}

}
