#include "sglang/text/mxfp8_gqa_prefill_common.cuh"

#include "sglang/decoder_cuda.cuh"

#include <climits>
#include <cstdlib>
#include <string_view>

namespace sglang {
namespace {

constexpr int max_multirow_visible_tokens = 32768;
constexpr int max_flash_visible_tokens = 131072;

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

template <int Heads, bool Staged>
__device__ void tile_scores(OrderedTile<float, Heads>& shared, Mxfp8KvCache cache,
                            std::size_t payload, std::size_t scale, int end, int warp, int lane,
                            const float4 (&queries)[group_passes][Heads],
                            const PrefillPayload<keys_per_tile>* raw) {
    const int group_in_warp = lane / score_lanes;
    for (int token = warp; token < end; token += score_warps) {
        float dots[Heads]{};
        #pragma unroll
        for (int pass = 0; pass < group_passes; ++pass) {
            const int group = pass * groups_per_warp + group_in_warp;
            const int channel = group * mxfp8_group_size + lane % score_lanes;
            const auto* key = Staged ? &raw->key[token][channel] :
                cache.key_payload + payload + token * payload_stride + channel;
            const auto exponent = Staged ? raw->key_scale[token][group] :
                cache.key_scales[scale + token * scale_stride + group];
            const float key0 = ordered_decode(key[0], exponent);
            const float key1 = ordered_decode(key[score_lanes], exponent);
            const float key2 = ordered_decode(key[2 * score_lanes], exponent);
            const float key3 = ordered_decode(key[3 * score_lanes], exponent);
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

template <int Heads, bool SkipUnitHistory>
__device__ bool tile_weights(OrderedTile<float, Heads>& shared, int count, int warp, int lane,
                             float& maximum, bool& invalid_score) {
    if (warp >= Heads || count == 0) {
        return true;
    }
    const float prior = maximum;
    const float score = lane < count ? shared.scores[warp][lane] : -CUDART_INF_F;
    if constexpr (SkipUnitHistory) {
        if (isfinite(prior) && __all_sync(warp_mask, score <= prior)) {
            if (lane < count) {
                invalid_score |= !isfinite(score);
                shared.history_weight[warp][lane] = 1.0f;
                shared.token_weight[warp][lane] = expf(score - prior);
            }
            return true;
        }
    }
    const float prefix = ordered_prefix(fmaxf(prior, score), lane);
    const float previous = __shfl_up_sync(warp_mask, prefix, 1);
    maximum = __shfl_sync(warp_mask, prefix, count - 1);
    if (lane < count) {
        invalid_score |= !isfinite(score);
        shared.history_weight[warp][lane] = expf((lane == 0 ? prior : previous) - prefix);
        shared.token_weight[warp][lane] = expf(score - prefix);
    }
    return false;
}

template <bool Staged>
__device__ __forceinline__ float tile_value(Mxfp8KvCache cache, std::size_t payload,
                                            std::size_t scale, int token, int channel, int warp,
                                            const PrefillPayload<keys_per_tile>* raw) {
    return Staged ? ordered_decode(raw->value[token][channel], raw->value_scale[token][warp]) :
        ordered_decode(cache.value_payload[payload + token * payload_stride + channel],
                       cache.value_scales[scale + token * scale_stride + warp]);
}

template <bool UnitHistory, int Heads>
__device__ __forceinline__ void ordered_value_step(OrderedTile<float, Heads>& shared, int head,
                                                   int token, float value, float& weighted) {
    const float history = UnitHistory ? weighted : shared.history_weight[head][token] * weighted;
    weighted = fmaf(shared.token_weight[head][token], value, history);
}

template <int Heads, bool Staged, bool UnitHistory>
__device__ void tile_values(OrderedTile<float, Heads>& shared, Mxfp8KvCache cache,
                            std::size_t payload, std::size_t scale, int first_visible, int end,
                            int channel, int warp, float (&weighted)[Heads],
                            const PrefillPayload<keys_per_tile>* raw) {
    const int count = max(0, min(keys_per_tile, first_visible));
    #pragma unroll 4
    for (int token = 0; token < count; ++token) {
        const float value = tile_value<Staged>(cache, payload, scale, token, channel, warp, raw);
        #pragma unroll
        for (int head = 0; head < Heads; ++head) {
            ordered_value_step<UnitHistory>(shared, head, token, value, weighted[head]);
        }
    }
    for (int token = count; token < end; ++token) {
        const float value = tile_value<Staged>(cache, payload, scale, token, channel, warp, raw);
        #pragma unroll
        for (int head = ordered_heads; head < Heads; ++head) {
            if (token < first_visible + head / ordered_heads) {
                ordered_value_step<UnitHistory>(shared, head, token, value, weighted[head]);
            }
        }
    }
}

template <int Heads, bool Staged>
__device__ int tile_attention(OrderedTile<float, Heads>& shared, Attention request, int row,
                             int head, PrefillPayload<keys_per_tile>* stages) {
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
    const int total = last + tile_rows;
    if constexpr (Staged) {
        prefetch_payload<keys_per_tile, channels>(stages[0], request, 0,
                                                  min(keys_per_tile, total), kv_head);
        asm volatile("cp.async.wait_group 0;" ::: "memory");
        __syncthreads();
    }
    for (int first = 0; first < total; first += keys_per_tile) {
        const int first_visible = last - first + 1;
        const int end = min(keys_per_tile, last - first + tile_rows);
        const int visible = max(0, min(keys_per_tile, first_visible + warp / ordered_heads));
        const int stage = (first / keys_per_tile) % 2;
        if constexpr (Staged) {
            const int next = first + keys_per_tile;
            if (next < total) {
                prefetch_payload<keys_per_tile, channels>(stages[1 - stage], request, next,
                    min(keys_per_tile, total - next), kv_head);
            }
        }
        std::size_t payload = 0, scale = 0;
        if constexpr (!Staged) {
            const int page = request.pages[first / mxfp8_page_size];
            const int token = first % mxfp8_page_size;
            payload = payload_offset(page, token, kv_head);
            scale = scale_offset(page, token, kv_head);
        }
        const auto* raw = Staged ? stages + stage : nullptr;
        tile_scores<Heads, Staged>(shared, request.cache, payload, scale, end, warp, lane,
                                  query, raw);
        __syncthreads();
        const bool unit_history = tile_weights<Heads, Staged>(
            shared, visible, warp, lane, maximum, invalid_score);
        bool all_unit_history = false;
        if constexpr (Staged) {
            all_unit_history = __syncthreads_and(unit_history);
        } else {
            __syncthreads();
        }
        if (all_unit_history) {
            tile_values<Heads, Staged, true>(shared, request.cache, payload, scale, first_visible,
                                            end, channel, warp, weighted, raw);
        } else {
            tile_values<Heads, Staged, false>(shared, request.cache, payload, scale, first_visible,
                                             end, channel, warp, weighted, raw);
        }
        if (warp < Heads && lane == 0) {
            for (int index = 0; index < visible; ++index) {
                const float weight = shared.token_weight[warp][index];
                mass = all_unit_history ? mass + weight :
                    fmaf(shared.history_weight[warp][index], mass, weight);
            }
        }
        if constexpr (Staged) {
            asm volatile("cp.async.wait_group 0;" ::: "memory");
        }
        __syncthreads();
    }
    if (warp < Heads && lane == 0) {
        shared.denominator[warp] = mass;
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

template <int Rows, int Blocks, bool Staged = false>
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
    PrefillPayload<keys_per_tile>* stages = nullptr;
    if constexpr (Staged) {
        __shared__ PrefillPayload<keys_per_tile> storage[2];
        stages = storage;
    }
    const int invalid = tile_attention<Rows * ordered_heads, Staged>(
        shared, request, row, head, stages);
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

constexpr int flash_mma_rows = 16;
constexpr int flash_stride = channels + 8;

template <int Rows>
constexpr int flash_math_threads = Rows * ordered_heads / flash_mma_rows * 32;

template <int Rows>
constexpr int flash_threads = flash_math_threads<Rows> < channels
    ? channels : flash_math_threads<Rows>;

template <int Rows>
constexpr bool flash_paired_fragments = Rows != flash_rows;

template <int Rows>
struct FlashTile {
    static_assert(Rows * ordered_heads % flash_mma_rows == 0);
    alignas(16) std::uint16_t query[Rows * ordered_heads][flash_stride];
    alignas(16) std::uint16_t key[flash_keys][flash_stride];
    alignas(16) std::uint16_t value[flash_keys][flash_stride];
    int bad_query[Rows * ordered_heads];
    int bad_key;
};

using FlashPayload = PrefillPayload<flash_keys>;

struct FragmentA {
    Word words[4];
};

struct FragmentB {
    Word words[2];
};

struct FragmentPairB {
    FragmentB first;
    FragmentB second;
};

struct Accumulator {
    float values[4];
};

__device__ __forceinline__ FragmentA load_flash_a(const std::uint16_t* address) {
    FragmentA result;
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
        : "=r"(result.words[0]), "=r"(result.words[1]),
          "=r"(result.words[2]), "=r"(result.words[3])
        : "r"(static_cast<Word>(__cvta_generic_to_shared(address))) : "memory");
    return result;
}

template <bool Transpose>
__device__ __forceinline__ FragmentB load_flash_b(const std::uint16_t* address) {
    FragmentB result;
    const auto shared = static_cast<Word>(__cvta_generic_to_shared(address));
    if constexpr (Transpose) {
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];"
            : "=r"(result.words[0]), "=r"(result.words[1]) : "r"(shared) : "memory");
    } else {
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
            : "=r"(result.words[0]), "=r"(result.words[1]) : "r"(shared) : "memory");
    }
    return result;
}

template <bool Transpose>
__device__ __forceinline__ FragmentPairB load_flash_b_pair(const std::uint16_t* address) {
    FragmentPairB result;
    const auto shared = static_cast<Word>(__cvta_generic_to_shared(address));
    if constexpr (Transpose) {
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
            : "=r"(result.first.words[0]), "=r"(result.first.words[1]),
              "=r"(result.second.words[0]), "=r"(result.second.words[1])
            : "r"(shared) : "memory");
    } else {
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
            : "=r"(result.first.words[0]), "=r"(result.first.words[1]),
              "=r"(result.second.words[0]), "=r"(result.second.words[1])
            : "r"(shared) : "memory");
    }
    return result;
}

__device__ __forceinline__ void flash_mma(Accumulator& sum, FragmentA lhs, FragmentB rhs) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(sum.values[0]), "+f"(sum.values[1]), "+f"(sum.values[2]), "+f"(sum.values[3])
        : "r"(lhs.words[0]), "r"(lhs.words[1]), "r"(lhs.words[2]), "r"(lhs.words[3]),
          "r"(rhs.words[0]), "r"(rhs.words[1]));
}

__device__ __forceinline__ void flash_decode_vector(std::uint16_t* destination,
                                                    const std::uint8_t* source,
                                                    int scale, bool& bad) {
    const auto payload = *reinterpret_cast<const uint4*>(source);
    if (scale >= 15 && scale <= 148) {
        flash_store_vector<true>(destination, payload, scale, bad);
    } else {
        flash_store_vector<false>(destination, payload, scale, bad);
    }
}

template <int Rows>
__device__ __forceinline__ void flash_prefetch(FlashPayload& raw, Attention request, int first,
                                               int visible, int kv_head) {
    prefetch_payload<flash_keys, flash_threads<Rows>>(raw, request, first, visible, kv_head);
}

template <bool Async, int Rows>
__device__ void flash_stage_kv(FlashTile<Rows>& tile, const FlashPayload* raw, Attention request,
                               int first, int visible, int kv_head) {
    const int channel = (threadIdx.x % flash_token_lanes) * flash_vector_elements;
    const int group = channel / mxfp8_group_size;
    const int page = Async ? 0 : request.pages[first / mxfp8_page_size];
    const auto cache = request.cache;
    for (int index = threadIdx.x / flash_token_lanes; index < flash_keys;
         index += flash_threads<Rows> / flash_token_lanes) {
        bool bad = false;
        if constexpr (Async) {
            flash_decode_vector(&tile.key[index][channel], &raw->key[index][channel],
                                raw->key_scale[index][group], bad);
            flash_decode_vector(&tile.value[index][channel], &raw->value[index][channel],
                                raw->value_scale[index][group], bad);
        } else if (index < visible) {
            const int token = (first + index) % mxfp8_page_size;
            const auto payload = payload_offset(page, token, kv_head) + channel;
            const auto scale = scale_offset(page, token, kv_head) + group;
            flash_decode_vector(&tile.key[index][channel], cache.key_payload + payload,
                                cache.key_scales[scale], bad);
            flash_decode_vector(&tile.value[index][channel], cache.value_payload + payload,
                                cache.value_scales[scale], bad);
        } else {
            const auto zeros = make_uint4(0, 0, 0, 0);
            auto* key = reinterpret_cast<uint4*>(&tile.key[index][channel]);
            auto* value = reinterpret_cast<uint4*>(&tile.value[index][channel]);
            key[0] = key[1] = value[0] = value[1] = zeros;
        }
        const auto invalid = __ballot_sync(warp_mask, bad);
        if (channel == 0 && ((invalid >> (threadIdx.x & 16)) & 0xffffu)) {
            atomicMin(&tile.bad_key, first + index);
        }
    }
}

template <bool Partial, int Rows>
__device__ void flash_stage_query(FlashTile<Rows>& tile, Attention request, int row, int head,
                                   int active_rows) {
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    if (threadIdx.x == 0) {
        tile.bad_key = INT_MAX;
    }
    for (int member = warp; member < Rows * ordered_heads; member += flash_threads<Rows> / 32) {
        const bool active = !Partial || member / ordered_heads < active_rows;
        const auto offset = row_offset(row + member / ordered_heads, head + member % ordered_heads);
        bool bad = false;
        #pragma unroll
        for (int channel = lane; channel < channels; channel += 32) {
            const auto bits = active ? __bfloat16_as_ushort(request.queries[offset + channel]) : 0;
            const unsigned magnitude = bits & 0x7fffu;
            const bool invalid = magnitude > 0x4e80u || (magnitude && magnitude < 0x80u);
            tile.query[member][channel] = invalid ? 0 : bits;
            bad |= invalid;
        }
        const auto invalid = __any_sync(warp_mask, bad);
        if (lane == 0) {
            tile.bad_query[member] = invalid;
        }
    }
}

constexpr int flash_score_fragments = flash_keys / 8;
constexpr int flash_value_fragments = flash_keys / 16;

template <int Rows>
__device__ __forceinline__ void flash_scores(const FlashTile<Rows>& tile, int query, int lane,
                                            Accumulator (&scores)[flash_score_fragments]) {
    #pragma unroll 4
    for (int dim = 0; dim < channels; dim += 16) {
        const auto q = load_flash_a(&tile.query[query + lane % 16][dim + (lane / 16) * 8]);
        if constexpr (flash_paired_fragments<Rows>) {
            const auto* keys = &tile.key[lane % 8 + (lane / 16) * 8][dim + (lane & 8)];
            #pragma unroll
            for (int group = 0; group < flash_score_fragments; group += 2) {
                const auto pair = load_flash_b_pair<false>(keys + group * 8 * flash_stride);
                flash_mma(scores[group], q, pair.first);
                flash_mma(scores[group + 1], q, pair.second);
            }
        } else {
            const auto* keys = &tile.key[lane % 8][dim + (lane & 8)];
            #pragma unroll
            for (int group = 0; group < flash_score_fragments; ++group) {
                flash_mma(scores[group], q, load_flash_b<false>(keys + group * 8 * flash_stride));
            }
        }
    }
}


template <int Half, bool FastPrefix>
__device__ __forceinline__ float flash_softmax(
    int column, int count, bool full, Accumulator (&scores)[flash_score_fragments],
    FragmentA (&probabilities)[flash_value_fragments], FlashSoftmax& state) {
    float maximum = state.maximum;
    if (FastPrefix && full) {
        #pragma unroll
        for (int group = 0; group < flash_score_fragments; ++group) {
            #pragma unroll
            for (int item = 0; item < 2; ++item) {
                auto& score = scores[group].values[Half * 2 + item];
                score *= softmax_scale;
                maximum = fmaxf(maximum, score);
            }
        }
    } else {
        #pragma unroll
        for (int group = 0; group < flash_score_fragments; ++group) {
            #pragma unroll
            for (int item = 0; item < 2; ++item) {
                const int token = group * 8 + column + item;
                auto& score = scores[group].values[Half * 2 + item];
                score = token < count ? score * softmax_scale : -CUDART_INF_F;
                maximum = fmaxf(maximum, score);
            }
        }
    }
    maximum = fmaxf(maximum, __shfl_xor_sync(warp_mask, maximum, 2, 4));
    maximum = fmaxf(maximum, __shfl_xor_sync(warp_mask, maximum, 1, 4));
    const float history = state.maximum == maximum ? 1.0f : __expf(state.maximum - maximum);
    float mass = 0.0f;
    #pragma unroll
    for (int group = 0; group < flash_score_fragments; ++group) {
        float pair[2];
        #pragma unroll
        for (int item = 0; item < 2; ++item) {
            const float score = scores[group].values[Half * 2 + item];
            pair[item] = maximum == -CUDART_INF_F ? 0.0f : __expf(score - maximum);
            mass += pair[item];
        }
        auto& word = probabilities[group / 2].words[(group % 2) * 2 + Half];
        asm("cvt.rn.bf16x2.f32 %0, %2, %1;"
            : "=r"(word) : "f"(pair[0]), "f"(pair[1]));
    }
    mass += __shfl_xor_sync(warp_mask, mass, 2, 4);
    mass += __shfl_xor_sync(warp_mask, mass, 1, 4);
    state.mass = fmaf(state.mass, history, mass);
    state.maximum = maximum;
    return history;
}

template <int Rows>
__device__ __forceinline__ void flash_values(
    const FlashTile<Rows>& tile, int lane, float top, float bottom,
    const FragmentA (&probabilities)[flash_value_fragments],
    Accumulator (&numerator)[channels / 8]) {
    if (!flash_paired_fragments<Rows> || __any_sync(warp_mask, top != 1.0f || bottom != 1.0f)) {
        #pragma unroll
        for (auto& accumulator : numerator) {
            accumulator.values[0] *= top;
            accumulator.values[1] *= top;
            accumulator.values[2] *= bottom;
            accumulator.values[3] *= bottom;
        }
    }
    #pragma unroll
    for (int fragment = 0; fragment < flash_value_fragments; ++fragment) {
        const int key = fragment * 16;
        if constexpr (flash_paired_fragments<Rows>) {
            const auto* values = &tile.value[key + lane % 16][(lane / 16) * 8];
            #pragma unroll
            for (int column = 0; column < channels; column += 16) {
                const auto pair = load_flash_b_pair<true>(values + column);
                flash_mma(numerator[column / 8], probabilities[fragment], pair.first);
                flash_mma(numerator[column / 8 + 1], probabilities[fragment], pair.second);
            }
        } else {
            #pragma unroll
            for (int column = 0; column < channels; column += 8) {
                const auto v = load_flash_b<true>(&tile.value[key + lane % 16][column]);
                flash_mma(numerator[column / 8], probabilities[fragment], v);
            }
        }
    }
}

template <bool Partial>
__device__ void flash_store(Attention request, int row, int head, int active_rows,
                            int query, int lane, const FlashSoftmax (&state)[2],
                            const Accumulator (&numerator)[channels / 8]) {
    #pragma unroll
    for (int half = 0; half < 2; ++half) {
        const int member = query + lane / 4 + half * 8;
        if constexpr (Partial) {
            if (member / ordered_heads >= active_rows) {
                continue;
            }
        }
        const auto output = row_offset(row + member / ordered_heads, head + member % ordered_heads);
        const float inverse_mass = __frcp_rn(state[half].mass);
        #pragma unroll
        for (int column = 0; column < channels; column += 8) {
            const int channel = column + (lane % 4) * 2;
            const auto& sum = numerator[column / 8];
            const float left = sum.values[half * 2] * inverse_mass;
            const float right = sum.values[half * 2 + 1] * inverse_mass;
            request.output[output + channel] = __float2bfloat16_rn(left);
            request.output[output + channel + 1] = __float2bfloat16_rn(right);
        }
    }
}

template <int Rows, bool Partial, bool Async>
__device__ __forceinline__ void flash_attention_tile(Attention request, int rows, int row,
                                                    int head, FlashTile<Rows>& tile,
                                                    OrderedScratch& fallback) {
    auto* raw = Async ? reinterpret_cast<FlashPayload*>(&tile + 1) : nullptr;
    const int position = request.position + row;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int query = warp * flash_mma_rows;
    const int member = query + lane / 4;
    const int column = (lane % 4) * 2;
    const int active_rows = Partial ? min(Rows, rows - row) : Rows;
    const int last = position + active_rows;
    Accumulator numerator[channels / 8]{};
    FlashSoftmax state[2]{{-CUDART_INF_F, 0.0f}, {-CUDART_INF_F, 0.0f}};
    if constexpr (Async) {
        flash_prefetch<Rows>(*raw, request, 0, min(flash_keys, last), head / ratio);
    }
    flash_stage_query<Partial>(tile, request, row, head, active_rows);
    if constexpr (Async) {
        asm volatile("cp.async.wait_group 0;" ::: "memory");
    }
    __syncthreads();
    for (int first = 0; first < last; first += flash_keys) {
        flash_stage_kv<Async>(tile, raw, request, first, min(flash_keys, last - first),
                              head / ratio);
        __syncthreads();
        const int next = first + flash_keys;
        if constexpr (Async) {
            if (next < last) {
                flash_prefetch<Rows>(*raw, request, next, min(flash_keys, last - next),
                                     head / ratio);
            }
        }
        if (threadIdx.x < flash_math_threads<Rows>) {
            Accumulator scores[flash_score_fragments]{};
            flash_scores(tile, query, lane, scores);
            const int visible = position + member / ordered_heads - first + 1;
            const int top_visible = !Partial || member / ordered_heads < active_rows ? visible : 0;
            const int bottom_visible = !Partial || (member + 8) / ordered_heads < active_rows
                ? visible + 8 / ordered_heads : 0;
            const bool full = next <= position + query / ordered_heads + 1 &&
                              (!Partial || query + flash_mma_rows <= active_rows * ordered_heads);
            FragmentA probabilities[flash_value_fragments];
            constexpr bool fast = flash_paired_fragments<Rows>;
            const auto top = flash_softmax<0, fast>(column, top_visible, full, scores,
                                                    probabilities, state[0]);
            const auto bottom = flash_softmax<1, fast>(column, bottom_visible, full, scores,
                                                       probabilities, state[1]);
            flash_values(tile, lane, top, bottom, probabilities, numerator);
        }
        if constexpr (Async) {
            asm volatile("cp.async.wait_group 0;" ::: "memory");
        }
        __syncthreads();
    }
    if (threadIdx.x < flash_math_threads<Rows>) {
        flash_store<Partial>(request, row, head, active_rows, query, lane, state, numerator);
    }
    if (__syncthreads_or(threadIdx.x < Rows * ordered_heads && tile.bad_query[threadIdx.x]) ||
        tile.bad_key < last) {
        if constexpr (Rows > flash_rows) {
            if (threadIdx.x >= channels) {
                return;
            }
        }
        for (int offset = 0; offset < active_rows; ++offset) {
            if (position + offset >= tile.bad_key ||
                tile.bad_query[offset * ordered_heads] ||
                tile.bad_query[offset * ordered_heads + 1]) {
                protected_attention(fallback, request, row + offset, head);
            }
        }
    }
}

template <int Rows, bool Partial, bool Async>
__launch_bounds__(flash_threads<Rows>, 1)
__global__ void flash_prefill_attention(Attention request, int rows) {
    alignas(16) extern __shared__ unsigned char shared[];
    __shared__ OrderedScratch fallback;
    flash_attention_tile<Rows, Partial, Async>(
        request, rows, blockIdx.x * Rows, blockIdx.y * ordered_heads,
        *reinterpret_cast<FlashTile<Rows>*>(shared), fallback);
}

constexpr int min_long_prefix = 4096;
constexpr int min_flash_rows = 144;

template <int Rows, bool Partial>
__launch_bounds__(flash_threads<Rows>, 1)
__global__ void batched_paged_attention(const __grid_constant__ PagedAttentionBatch batch) {
    alignas(16) extern __shared__ unsigned char shared[];
    __shared__ OrderedScratch fallback;
    const int tile = blockIdx.x;
    const int index = batch.locate(tile);
    flash_attention_tile<Rows, Partial, true>(
        batch.requests[index], batch.rows[index], batch.row(tile, index, Rows),
        blockIdx.y * ordered_heads,
        *reinterpret_cast<FlashTile<Rows>*>(shared), fallback);
}

template <int Rows, bool Partial>
std::optional<cudaError_t> launch_paged_batch(PagedAttentionBatch& batch, int shared_limit,
                                             cudaStream_t stream) {
    const auto kernel = batched_paged_attention<Rows, Partial>;
    constexpr auto bytes = sizeof(FlashTile<Rows>) + sizeof(FlashPayload);
    cudaFuncAttributes attributes{};
    auto status = cudaFuncGetAttributes(&attributes, kernel);
    if (status != cudaSuccess) {
        return status;
    }
    if (static_cast<std::size_t>(shared_limit) < bytes + attributes.sharedSizeBytes) {
        return std::nullopt;
    }
    status = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
    if (status != cudaSuccess) {
        return status;
    }
    int tiles = 0;
    for (int index = 0; index < batch.count; ++index) {
        tiles += (batch.rows[index] + Rows - 1) / Rows;
        batch.tile_end[index] = tiles;
    }
    void* args[] = {&batch};
    return cudaLaunchKernel(kernel, dim3(tiles, head_pairs), flash_threads<Rows>, args,
                            bytes, stream);
}

std::optional<cudaError_t> launch_flash_prefill(Attention request, int rows,
                                               cudaStream_t stream) {
    int device = 0;
    auto status = cudaGetDevice(&device);
    if (status != cudaSuccess) {
        return status;
    }
    int shared_limit = 0;
    status = cudaDeviceGetAttribute(&shared_limit, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (status != cudaSuccess) {
        return status;
    }
    void* args[] = {&request, &rows};
    const bool partial = rows % flash_rows != 0;
    const bool aligned_scales =
        (reinterpret_cast<std::uintptr_t>(request.cache.key_scales) |
         reinterpret_cast<std::uintptr_t>(request.cache.value_scales)) % 8 == 0;
    const auto launch = [&](auto kernel, int tile_rows, int threads,
                            std::size_t bytes) -> std::optional<cudaError_t> {
        cudaFuncAttributes attributes{};
        status = cudaFuncGetAttributes(&attributes, kernel);
        if (status != cudaSuccess) {
            return status;
        }
        if (static_cast<std::size_t>(shared_limit) < bytes + attributes.sharedSizeBytes) {
            return std::nullopt;
        }
        status = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
        if (status != cudaSuccess) {
            return status;
        }
        const dim3 blocks((rows + tile_rows - 1) / tile_rows, mxfp8_query_heads / ordered_heads);
        return cudaLaunchKernel(kernel, blocks, threads, args, bytes, stream);
    };
    if (aligned_scales && rows == flash_chunk && request.position >= flash_chunk) {
        if (const auto wide = launch(flash_prefill_attention<flash_wide_rows, true, true>,
                                     flash_wide_rows, flash_threads<flash_wide_rows>,
                                     sizeof(FlashTile<flash_wide_rows>) + sizeof(FlashPayload))) {
            return wide;
        }
    }
    if (aligned_scales && rows < flash_chunk && request.position >= min_long_prefix) {
        int multiprocessors = 0;
        status = cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, device);
        if (status != cudaSuccess) {
            return status;
        }
        const int tile_rows = long_flash_tile_rows(rows, multiprocessors / head_pairs);
        if (tile_rows == flash_short_rows) {
            if (const auto short_tile = launch(
                    flash_prefill_attention<flash_short_rows, true, true>, flash_short_rows,
                    flash_threads<flash_short_rows>,
                    sizeof(FlashTile<flash_short_rows>) + sizeof(FlashPayload))) {
                return short_tile;
            }
        } else if (tile_rows == flash_medium_rows) {
            if (const auto medium_tile = launch(
                    flash_prefill_attention<flash_medium_rows, true, true>, flash_medium_rows,
                    flash_threads<flash_medium_rows>,
                    sizeof(FlashTile<flash_medium_rows>) + sizeof(FlashPayload))) {
                return medium_tile;
            }
        } else if (tile_rows == flash_wide_rows) {
            if (const auto wide = launch(
                    flash_prefill_attention<flash_wide_rows, true, true>, flash_wide_rows,
                    flash_threads<flash_wide_rows>,
                    sizeof(FlashTile<flash_wide_rows>) + sizeof(FlashPayload))) {
                return wide;
            }
        }
    }
    for (bool async : {true, false}) {
        if (async && !aligned_scales) {
            continue;
        }
        const auto kernel = async
            ? (partial ? flash_prefill_attention<flash_rows, true, true>
                       : flash_prefill_attention<flash_rows, false, true>)
            : (partial ? flash_prefill_attention<flash_rows, true, false>
                       : flash_prefill_attention<flash_rows, false, false>);
        const auto bytes = sizeof(FlashTile<flash_rows>) + (async ? sizeof(FlashPayload) : 0);
        if (const auto regular = launch(kernel, flash_rows, flash_threads<flash_rows>, bytes)) {
            return regular;
        }
    }
    return std::nullopt;
}

}

std::optional<cudaError_t> decoder::try_batched_paged_prefill(
    const PagedPrefillSequence* sequences, int count, cudaStream_t stream) {
    if (!sequences || count < 2 || count > paged_batch_sequences ||
        std::getenv("SGLANG_PREFILL_ATTN") || std::getenv("SGLANG_EXPERIMENTAL_TMEM_PREFILL")) {
        return std::nullopt;
    }
    int device = 0;
    auto status = cudaGetDevice(&device);
    if (status != cudaSuccess) {
        return status;
    }
    int multiprocessors = 0;
    status = cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, device);
    if (status != cudaSuccess) {
        return status;
    }
    PagedAttentionBatch batch{};
    batch.count = count;
    int tile_rows = 0;
    bool partial = false;
    const auto& first = sequences[0];
    const bool ragged = default_ragged_prefill(first.cache, first.position, first.rows);
    for (int index = 0; index < count; ++index) {
        const auto& sequence = sequences[index];
        const auto& cache = sequence.cache;
        if (sequence.rows < min_flash_rows || sequence.rows >= flash_chunk ||
            sequence.position < 0 ||
            sequence.position > max_multirow_visible_tokens - sequence.rows ||
            !sequence.pages.device_pages || !sequence.queries || !sequence.output ||
            !cache.key_payload || !cache.value_payload ||
            !cache.key_scales || !cache.value_scales ||
            reinterpret_cast<std::uintptr_t>(cache.key_payload) % alignof(uint4) ||
            reinterpret_cast<std::uintptr_t>(cache.value_payload) % alignof(uint4) ||
            reinterpret_cast<std::uintptr_t>(cache.key_scales) % alignof(std::uint64_t) ||
            reinterpret_cast<std::uintptr_t>(cache.value_scales) % alignof(std::uint64_t)) {
            return std::nullopt;
        }
        const int last_page = (sequence.position + sequence.rows - 1) / mxfp8_page_size;
        if (last_page >= sequence.pages.page_count || last_page >= cache.page_count) {
            return std::nullopt;
        }
        const bool lane_ragged = default_ragged_prefill(cache, sequence.position, sequence.rows);
        if (lane_ragged != ragged || (ragged && sequence.rows > 384)) {
            return std::nullopt;
        }
        if (ragged) {
            continue;
        }
        const int shape = sequence.position >= min_long_prefix
            ? long_flash_tile_rows(sequence.rows, multiprocessors / head_pairs) : flash_rows;
        const bool tail = shape != flash_rows || sequence.rows % flash_rows != 0;
        if ((shape != flash_rows && shape != flash_short_rows) ||
            (index && (tile_rows != shape || partial != tail))) {
            return std::nullopt;
        }
        tile_rows = shape;
        partial = tail;
        batch.requests[index] = {cache, sequence.pages.device_pages, sequence.position,
                                 sequence.queries, sequence.output};
        batch.rows[index] = sequence.rows;
    }
    if (ragged) {
        return launch_batched_ragged_tmem(sequences, count, stream);
    }
    int shared_limit = 0;
    status = cudaDeviceGetAttribute(&shared_limit, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (status != cudaSuccess) {
        return status;
    }
    if (tile_rows == flash_short_rows) {
        return launch_paged_batch<flash_short_rows, true>(batch, shared_limit, stream);
    }
    return partial ? launch_paged_batch<flash_rows, true>(batch, shared_limit, stream)
                   : launch_paged_batch<flash_rows, false>(batch, shared_limit, stream);
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
    return tensor_core_multirow_prefill_gqa(cache, pages, position, rows, queries, output,
                                           stream, PairedPrefillScratch{});
}

cudaError_t decoder::tensor_core_multirow_prefill_gqa(
    Mxfp8KvCache cache, Mxfp8PageTable pages, int position, int rows,
    const Bf16* queries, Bf16* output, cudaStream_t stream, PairedPrefillScratch scratch) {
    if (rows == 0) {
        return cudaSuccess;
    }
    if (position < 0 || rows < 0 || position > INT_MAX - rows) {
        return cudaErrorInvalidValue;
    }
    Attention request{cache, pages.device_pages, position, queries, output};
    const int visible = position + rows;
    const int flash_limit = rows == flash_chunk ? max_flash_visible_tokens :
                                                  max_multirow_visible_tokens;
    const auto* setting = std::getenv("SGLANG_EXPERIMENTAL_TMEM_PREFILL");
    const std::string_view mode = setting ? setting : "";
    const bool default_ragged = !setting && default_ragged_prefill(cache, position, rows);
    if ((default_ragged || (mode == "ragged" && position >= 4096)) &&
        rows >= 256 && rows <= 512 &&
        visible <= max_multirow_visible_tokens) {
        if (const auto status = launch_tmem_prefill(cache, pages, position, queries, output,
                                                   stream, true, rows, default_ragged)) {
            return *status;
        }
    }
    if (rows >= min_flash_rows && rows <= flash_chunk && visible <= flash_limit &&
        (reinterpret_cast<std::uintptr_t>(cache.key_payload) % alignof(uint4) == 0) &&
        (reinterpret_cast<std::uintptr_t>(cache.value_payload) % alignof(uint4) == 0)) {
        if (rows == flash_chunk && position >= 8192) {
            constexpr int max_default_paired_position = 31744;
            const bool default_paired = (!setting || mode == "ragged") &&
                position <= max_default_paired_position &&
                (position % flash_chunk == 0 || position % mxfp8_page_size != 0);
            if (default_paired && scratch.keys && scratch.values) {
                if (const auto status = launch_staged_tmem_prefill(cache, pages, position,
                                                                  queries, output, stream,
                                                                  scratch)) {
                    return *status;
                }
            }
            if (default_paired || mode == "paired") {
                if (const auto status = launch_paired_tmem_prefill(cache, pages, position, queries,
                                                                  output, stream)) {
                    return *status;
                }
            }
            const bool ordered = mode == "ordered";
            if (position % flash_keys == 0 &&
                (ordered || (mode == "1" && position >= max_multirow_visible_tokens))) {
                if (const auto status = launch_tmem_prefill(cache, pages, position, queries,
                                                           output, stream, ordered)) {
                    return *status;
                }
            }
        }
        if (const auto status = launch_flash_prefill(request, rows, stream)) {
            return *status;
        }
    }
    void* args[] = {&request, &rows};
    constexpr int min_paired_rows = 64;
    constexpr int min_staged_visible = 8192;
    constexpr int min_wide_tail_rows = 32;
    constexpr int min_paired_tail_rows = 12;
    if (rows >= 2 && rows < min_flash_rows &&
        visible >= min_staged_visible && visible <= max_multirow_visible_tokens &&
        reinterpret_cast<std::uintptr_t>(cache.key_payload) % alignof(uint4) == 0 &&
        reinterpret_cast<std::uintptr_t>(cache.value_payload) % alignof(uint4) == 0 &&
        reinterpret_cast<std::uintptr_t>(cache.key_scales) % alignof(std::uint64_t) == 0 &&
        reinterpret_cast<std::uintptr_t>(cache.value_scales) % alignof(std::uint64_t) == 0) {
        const int tile_rows = rows >= min_wide_tail_rows ? wide_rows :
            rows >= min_paired_tail_rows && rows % paired_rows == 0 ? paired_rows : 1;
        const dim3 blocks(rows / tile_rows + rows % tile_rows, mxfp8_query_heads / ordered_heads);
        const auto kernel = tile_rows == wide_rows ?
            multirow_prefill_attention<wide_rows, 2, true> : tile_rows == paired_rows ?
            multirow_prefill_attention<paired_rows, 3, true> :
            multirow_prefill_attention<1, 4, true>;
        return cudaLaunchKernel(kernel, blocks, channels, args, 0, stream);
    }
    if (rows < min_paired_rows || visible > max_multirow_visible_tokens) {
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
