#pragma once

#include "sglang/mxfp8_gqa.h"

#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <optional>

namespace sglang {

std::optional<cudaError_t> launch_tmem_prefill(
    Mxfp8KvCache cache, Mxfp8PageTable pages, int position,
    const __nv_bfloat16* queries, __nv_bfloat16* output, cudaStream_t stream, bool ordered);

std::optional<cudaError_t> launch_paired_tmem_prefill(
    Mxfp8KvCache cache, Mxfp8PageTable pages, int position,
    const __nv_bfloat16* queries, __nv_bfloat16* output, cudaStream_t stream);

namespace {

using Bf16 = __nv_bfloat16;
using Word = unsigned int;
constexpr int channels = mxfp8_head_dim;
constexpr int ratio = mxfp8_query_heads / mxfp8_kv_heads;
constexpr int ordered_heads = 2;
constexpr int keys_per_tile = 32;
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

template <bool recover_scores, typename Real>
__device__ __forceinline__ void ordered_weights(OrderedTile<Real>& shared, int warp, int lane,
                                               int count, Real& maximum, bool& invalid_score) {
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
    for (int first = 0;; first += keys_per_tile) {
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
            ordered_weights<recover_scores>(shared, warp, lane, count, maximum, invalid_score);
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
        if (last - first < keys_per_tile) {
            break;
        }
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

constexpr int flash_rows = 64;
constexpr int flash_queries = flash_rows * ordered_heads;
constexpr int flash_keys = 64;
constexpr int flash_chunk = 1024;
static_assert(mxfp8_page_size % flash_keys == 0);

__device__ __forceinline__ unsigned flash_decode_half(unsigned bits, int scale, bool& bad) {
    const int exponent = (bits >> 10) & 31;
    const int scaled = exponent + scale - 15;
    const unsigned sign = bits & 0x8000u;
    const unsigned fraction = (bits >> 3) & 127u;
    const bool invalid = exponent == 31 ||
        (exponent != 0 && (scaled < 1 || scaled > 157 || (scaled == 157 && fraction)));
    bad |= invalid;
    return exponent == 0 || invalid ? sign : sign | (scaled << 7) | fraction;
}

template <bool CommonScale>
__device__ __forceinline__ Word flash_decode_pair(Word packed, int scale, bool& bad) {
    Word halves;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(halves) : "h"(static_cast<short>(packed)));
    if constexpr (CommonScale) {
        constexpr Word sign = 0x80008000u;
        const Word magnitude = halves & ~sign;
        const Word nonzero = (magnitude + 0x7fff7fffu) & sign;
        const Word nonfinite = (magnitude + 0x04000400u) & sign;
        const Word keep = ((nonzero ^ nonfinite) >> 15) * 0xffffu;
        const Word bias = static_cast<Word>(scale - 15) * 0x00800080u;
        bad |= nonfinite != 0;
        const Word bfloat = (magnitude & 0x7ff87ff8u) >> 3;
        return (halves & sign) | ((bfloat + bias) & keep);
    } else {
        const auto lo = flash_decode_half(halves & 0xffffu, scale, bad);
        const auto hi = flash_decode_half(halves >> 16, scale, bad);
        return lo | (hi << 16);
    }
}

template <bool CommonScale>
__device__ __forceinline__ void flash_store_vector(std::uint16_t* destination, uint4 payload,
                                                  int scale, bool& bad) {
    auto* output = reinterpret_cast<uint4*>(destination);
    output[0] = make_uint4(flash_decode_pair<CommonScale>(payload.x, scale, bad),
                           flash_decode_pair<CommonScale>(payload.x >> 16, scale, bad),
                           flash_decode_pair<CommonScale>(payload.y, scale, bad),
                           flash_decode_pair<CommonScale>(payload.y >> 16, scale, bad));
    output[1] = make_uint4(flash_decode_pair<CommonScale>(payload.z, scale, bad),
                           flash_decode_pair<CommonScale>(payload.z >> 16, scale, bad),
                           flash_decode_pair<CommonScale>(payload.w, scale, bad),
                           flash_decode_pair<CommonScale>(payload.w >> 16, scale, bad));
}

struct FlashSoftmax {
    float maximum;
    float mass;
};

}
}
