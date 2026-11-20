#pragma once

constexpr int wide_warps = 4;
constexpr int wide_keys = 16;
constexpr int wide_heads = queries_per_kv_head;
constexpr int wide_stride = mxfp8_head_dim + 8;
constexpr int wide_prob_stride = wide_keys + 8;
static_assert(wide_heads == 6 && mxfp8_page_size % wide_keys == 0);

struct WideTile {
    alignas(16) std::uint16_t query[16][wide_stride];
    alignas(16) std::uint16_t kv[wide_warps][wide_keys][wide_stride];
    alignas(16) std::uint16_t probability[wide_warps][16][wide_prob_stride];
};

struct WideMerge {
    float value[wide_warps][wide_heads][mxfp8_head_dim];
    float maximum[wide_warps][wide_heads];
    float mass[wide_warps][wide_heads];
    bool bad[wide_warps][wide_heads];
};

struct WideLeft { unsigned int words[4]; };
struct WideRight { unsigned int words[2]; };
struct WideAccumulator { float value[4]; };

__device__ __forceinline__ WideLeft wide_load_left(const std::uint16_t* address) {
    WideLeft result;
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
        : "=r"(result.words[0]), "=r"(result.words[1]),
          "=r"(result.words[2]), "=r"(result.words[3])
        : "r"(static_cast<unsigned int>(__cvta_generic_to_shared(address))) : "memory");
    return result;
}

template <bool Transpose>
__device__ __forceinline__ WideRight wide_load_right(const std::uint16_t* address) {
    const auto shared = static_cast<unsigned int>(__cvta_generic_to_shared(address));
    WideRight result;
    if constexpr (Transpose) {
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];"
            : "=r"(result.words[0]), "=r"(result.words[1]) : "r"(shared) : "memory");
    } else {
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
            : "=r"(result.words[0]), "=r"(result.words[1]) : "r"(shared) : "memory");
    }
    return result;
}

__device__ __forceinline__ void wide_mma(WideAccumulator& sum, WideLeft lhs, WideRight rhs) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(sum.value[0]), "+f"(sum.value[1]), "+f"(sum.value[2]), "+f"(sum.value[3])
        : "r"(lhs.words[0]), "r"(lhs.words[1]), "r"(lhs.words[2]), "r"(lhs.words[3]),
          "r"(rhs.words[0]), "r"(rhs.words[1]));
}

__device__ __forceinline__ unsigned int wide_decode_half(unsigned int bits, int scale,
                                                         bool& bad) {
    const int exponent = (bits >> 10) & 31;
    const int scaled = exponent + scale - 15;
    const auto sign = bits & 0x8000u;
    const auto fraction = (bits >> 3) & 127u;
    const bool invalid = exponent == 31 ||
        (exponent && (scaled < 1 || scaled > 157 || (scaled == 157 && fraction)));
    bad |= invalid;
    return exponent == 0 || invalid ? sign : sign | (scaled << 7) | fraction;
}

template <bool CommonScale>
__device__ __forceinline__ unsigned int wide_decode_pair(unsigned int packed, int scale,
                                                         bool& bad) {
    unsigned int halves;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(halves) : "h"(static_cast<short>(packed)));
    if constexpr (CommonScale) {
        constexpr unsigned int sign = 0x80008000u;
        const auto magnitude = halves & ~sign;
        const auto nonzero = (magnitude + 0x7fff7fffu) & sign;
        const auto nonfinite = (magnitude + 0x04000400u) & sign;
        const auto keep = ((nonzero ^ nonfinite) >> 15) * 0xffffu;
        const auto bias = static_cast<unsigned int>(scale - 15) * 0x00800080u;
        bad |= nonfinite != 0;
        return (halves & sign) | ((((magnitude & 0x7ff87ff8u) >> 3) + bias) & keep);
    } else {
        const auto lo = wide_decode_half(halves & 0xffffu, scale, bad);
        const auto hi = wide_decode_half(halves >> 16, scale, bad);
        return lo | (hi << 16);
    }
}

template <bool CommonScale>
__device__ __forceinline__ void wide_store_vector(std::uint16_t* destination, uint4 payload,
                                                  int scale, bool& bad) {
    auto* output = reinterpret_cast<uint4*>(destination);
    output[0] = make_uint4(wide_decode_pair<CommonScale>(payload.x, scale, bad),
                           wide_decode_pair<CommonScale>(payload.x >> 16, scale, bad),
                           wide_decode_pair<CommonScale>(payload.y, scale, bad),
                           wide_decode_pair<CommonScale>(payload.y >> 16, scale, bad));
    output[1] = make_uint4(wide_decode_pair<CommonScale>(payload.z, scale, bad),
                           wide_decode_pair<CommonScale>(payload.z >> 16, scale, bad),
                           wide_decode_pair<CommonScale>(payload.w, scale, bad),
                           wide_decode_pair<CommonScale>(payload.w >> 16, scale, bad));
}

template <bool Value>
__device__ __forceinline__ void wide_stage(WideTile& tile, decoder::PagedDecodeLane input,
                                           int warp, int lane, int first, int count, bool& bad) {
    const int channel = (lane % 16) * 16;
    const int page = input.pages.device_pages[first / mxfp8_page_size];
    const auto* payload = Value ? input.cache.value_payload : input.cache.key_payload;
    const auto* scales = Value ? input.cache.value_scales : input.cache.key_scales;
    #pragma unroll 2
    for (int index = lane / 16; index < wide_keys; index += 2) {
        auto* destination = &tile.kv[warp][index][channel];
        if (index < count) {
            const int slot = (first + index) % mxfp8_page_size;
            const auto data = payload_offset(page, slot, blockIdx.y) + channel;
            const auto scale = scales[scale_offset(page, slot, blockIdx.y, channel / 32)];
            const auto words = *reinterpret_cast<const uint4*>(payload + data);
            if (scale >= 15 && scale <= 148) {
                wide_store_vector<true>(destination, words, scale, bad);
            } else {
                wide_store_vector<false>(destination, words, scale, bad);
            }
        } else {
            auto* zeros = reinterpret_cast<uint4*>(destination);
            zeros[0] = zeros[1] = make_uint4(0, 0, 0, 0);
        }
    }
}

__device__ __forceinline__ void wide_write_partials(
    WideMerge& merge, const decoder::PagedDecodeLane& input, int first_head, int warp,
    int row, int column, float maximum, float mass, bool bad,
    const bool (&bad_query)[wide_heads],
    const WideAccumulator (&numerator)[mxfp8_head_dim / 8]) {
    if (row < wide_heads) {
        if (column == 0) {
            merge.maximum[warp][row] = maximum;
            merge.mass[warp][row] = mass;
            merge.bad[warp][row] = bad || bad_query[row];
        }
        #pragma unroll
        for (int col = 0; col < mxfp8_head_dim; col += 8) {
            merge.value[warp][row][col + column] = numerator[col / 8].value[0];
            merge.value[warp][row][col + column + 1] = numerator[col / 8].value[1];
        }
    }
    __syncthreads();
    for (int head = 0; head < wide_heads; ++head) {
        auto& partial = decode_partials(input.buffers, first_head + head)[blockIdx.x];
        float peak = -CUDART_INF_F;
        bool invalid = false;
        #pragma unroll
        for (int peer = 0; peer < wide_warps; ++peer) {
            peak = fmaxf(peak, merge.maximum[peer][head]);
            invalid |= merge.bad[peer][head];
        }
        float denominator = 0.0f;
        float lo = 0.0f;
        float hi = 0.0f;
        #pragma unroll
        for (int peer = 0; peer < wide_warps; ++peer) {
            const float scale = __expf(merge.maximum[peer][head] - peak);
            denominator = fmaf(scale, merge.mass[peer][head], denominator);
            lo = fmaf(scale, merge.value[peer][head][threadIdx.x], lo);
            hi = fmaf(scale, merge.value[peer][head][threadIdx.x + 128], hi);
        }
        if (threadIdx.x == 0) {
            partial.maximum = invalid ? CUDART_NAN_F : peak;
            partial.mass = denominator;
        }
        partial.value[threadIdx.x] = lo;
        partial.value[threadIdx.x + 128] = hi;
    }
}

__global__ void split_long_decode_tensor(const __grid_constant__ LongDecodeBatch batch) {
    __shared__ union {
        WideTile tile;
        WideMerge merge;
    } shared;
    auto& tile = shared.tile;
    const auto input = batch.lanes[blockIdx.z];
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int first_head = blockIdx.y * wide_heads;
    const int visible = input.position + 1;
    const int pages = (visible + mxfp8_page_size - 1) / mxfp8_page_size;
    const int first = blockIdx.x * pages / decode_splits * mxfp8_page_size;
    const int end = min(visible, (blockIdx.x + 1) * pages / decode_splits * mxfp8_page_size);
    __shared__ bool bad_query[wide_heads];
    for (int row = warp; row < 16; row += wide_warps) {
        bool bad = false;
        #pragma unroll
        for (int channel = lane; channel < mxfp8_head_dim; channel += 32) {
            const auto bits = row < wide_heads ? __bfloat16_as_ushort(
                input.buffers.queries[(first_head + row) * mxfp8_head_dim + channel]) : 0;
            const auto magnitude = bits & 0x7fffu;
            const bool invalid = magnitude > 0x4e80u || (magnitude && magnitude < 0x80u);
            tile.query[row][channel] = invalid ? 0 : bits;
            bad |= invalid;
        }
        const auto any_bad = __any_sync(warp_mask, bad);
        if (row < wide_heads && lane == 0) {
            bad_query[row] = any_bad;
        }
    }
    __syncthreads();

    const int row = lane / 4;
    const int column = 2 * (lane % 4);
    float maximum = -CUDART_INF_F;
    float mass = 0.0f;
    WideAccumulator numerator[mxfp8_head_dim / 8]{};
    bool bad_cache = false;
    for (int offset = first + warp * wide_keys; offset < end; offset += wide_warps * wide_keys) {
        const int count = min(wide_keys, end - offset);
        wide_stage<false>(tile, input, warp, lane, offset, count, bad_cache);
        __syncwarp();
        WideAccumulator scores[2]{};
        #pragma unroll 4
        for (int dim = 0; dim < mxfp8_head_dim; dim += 16) {
            const auto query = wide_load_left(&tile.query[lane % 16][dim + (lane / 16) * 8]);
            const int channel = dim + (lane & 8);
            wide_mma(scores[0], query, wide_load_right<false>(&tile.kv[warp][lane % 8][channel]));
            wide_mma(scores[1], query,
                     wide_load_right<false>(&tile.kv[warp][lane % 8 + 8][channel]));
        }
        __syncwarp();
        wide_stage<true>(tile, input, warp, lane, offset, count, bad_cache);
        float next = maximum;
        #pragma unroll
        for (int key = 0; key < wide_keys; key += 8) {
            #pragma unroll
            for (int item = 0; item < 2; ++item) {
                auto& score = scores[key / 8].value[item];
                score = key + column + item < count ? score * softmax_scale : -CUDART_INF_F;
                next = fmaxf(next, score);
            }
        }
        next = fmaxf(next, __shfl_xor_sync(warp_mask, next, 1, 4));
        next = fmaxf(next, __shfl_xor_sync(warp_mask, next, 2, 4));
        const float history = __expf(maximum - next);
        float tile_mass = 0.0f;
        #pragma unroll
        for (int key = 0; key < wide_keys; key += 8) {
            #pragma unroll
            for (int item = 0; item < 2; ++item) {
                const float weight = __expf(scores[key / 8].value[item] - next);
                tile.probability[warp][row][key + column + item] =
                    __bfloat16_as_ushort(__float2bfloat16_rn(weight));
                tile.probability[warp][row + 8][key + column + item] = 0;
                tile_mass += weight;
            }
        }
        tile_mass += __shfl_xor_sync(warp_mask, tile_mass, 1, 4);
        tile_mass += __shfl_xor_sync(warp_mask, tile_mass, 2, 4);
        mass = fmaf(history, mass, tile_mass);
        maximum = next;
        __syncwarp();
        const auto probability = wide_load_left(&tile.probability[warp][lane % 16][lane / 16 * 8]);
        #pragma unroll
        for (int channel = 0; channel < mxfp8_head_dim; channel += 8) {
            auto& sum = numerator[channel / 8];
            sum.value[0] *= history;
            sum.value[1] *= history;
            wide_mma(sum, probability, wide_load_right<true>(&tile.kv[warp][lane % 16][channel]));
        }
        __syncwarp();
    }
    const bool bad = __any_sync(warp_mask, bad_cache);
    __syncthreads();
    wide_write_partials(shared.merge, input, first_head, warp, row, column, maximum, mass, bad,
                        bad_query, numerator);
}
