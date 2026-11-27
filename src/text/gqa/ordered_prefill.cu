#include "sglang/text/mxfp8_gqa_prefill_common.cuh"
#include "sglang/decoder_cuda.cuh"

#include <climits>

namespace sglang {
namespace {

using namespace gqa_prefill;

__global__ void tiled_prefill_attention(Attention request, const int* position) {
    __shared__ OrderedTile<float> shared;
    if (position) request.position = *position;
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
                cache.key_payload + payload + token * (cache.kv_heads * channels) + channel;
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
        ordered_decode(cache.value_payload[payload + token * (cache.kv_heads * channels)
                                           + channel],
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
    const int query_heads = gqa_query_heads(request.cache.kv_heads);
    float4 query[group_passes][Heads];
    float weighted[Heads]{};
    #pragma unroll
    for (int member = 0; member < Heads; ++member) {
        const auto offset =
            row_offset(query_heads, row + member / ordered_heads, head + member % ordered_heads);
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
            payload = payload_offset(request.cache.kv_heads, page, token, kv_head);
            scale = scale_offset(request.cache.kv_heads, page, token, kv_head);
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
    if (warp < Heads && lane == 0) shared.denominator[warp] = mass;
    int invalid = score_recovery_mask<Heads>(warp, lane, invalid_score);
    #pragma unroll
    for (int member = 0; member < Heads; ++member) {
        const auto offset =
            row_offset(query_heads, row + member / ordered_heads, head + member % ordered_heads);
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

}

cudaError_t decoder::tensor_core_multirow_prefill_gqa(
    Mxfp8KvCache cache, Mxfp8PageTable pages, int position, int rows,
    const Bf16* queries, Bf16* output, cudaStream_t stream, GqaGeometry geometry) {
    if (rows == 0) return cudaSuccess;
    if (!gqa_geometry_matches(geometry, cache)) return cudaErrorInvalidValue;
    return tensor_core_multirow_prefill_gqa(cache, pages, position, rows, queries, output,
                                           stream, PairedPrefillScratch{});
}

cudaError_t decoder::tensor_core_multirow_prefill_gqa(
    Mxfp8KvCache cache, Mxfp8PageTable pages, int position, int rows,
    const Bf16* queries, Bf16* output, cudaStream_t stream, PairedPrefillScratch scratch) {
    if (rows == 0) return cudaSuccess;
    if (position < 0 || rows < 0 || position > INT_MAX - rows) return cudaErrorInvalidValue;
    Attention request{cache, pages.device_pages, position, queries, output};
    const int visible = position + rows;
    const bool ragged = default_ragged_prefill(cache, position, rows) ||
        default_ragged_prefill_tail({cache, pages, position, rows, queries, output},
                                   max_multirow_visible_tokens);
    if ((ragged || raw_pipeline_before_flash(cache, rows, visible)) &&
        visible <= max_multirow_visible_tokens) {
        if (const auto status = launch_tmem_prefill(cache, pages, position, queries, output,
                                                   stream, true, rows, true)) {
            return *status;
        }
    }
    if (rows >= min_flash_rows && rows <= flash_chunk && visible <= max_flash_visible_tokens &&
        (reinterpret_cast<std::uintptr_t>(cache.key_payload) % alignof(uint4) == 0) &&
        (reinterpret_cast<std::uintptr_t>(cache.value_payload) % alignof(uint4) == 0)) {
        if (rows == flash_chunk &&
            (position >= 8192 || position % flash_chunk == 0)) {
            constexpr int max_default_paired_position = 31744;
            const bool default_paired = position <= max_default_paired_position &&
                (position % flash_chunk == 0 || position % mxfp8_page_size != 0);
            if (default_paired && scratch.keys && scratch.values) {
                if (const auto status = launch_staged_tmem_prefill(cache, pages, position,
                                                                  queries, output, stream,
                                                                  scratch)) {
                    return *status;
                }
            }
            if (default_paired || (position % flash_chunk == 0 &&
                                   position >= max_multirow_visible_tokens)) {
                if (const auto status = launch_paired_tmem_prefill(cache, pages, position, queries,
                                                                  output, stream)) {
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
        visible >= min_staged_visible && visible <= max_flash_visible_tokens &&
        reinterpret_cast<std::uintptr_t>(cache.key_payload) % alignof(uint4) == 0 &&
        reinterpret_cast<std::uintptr_t>(cache.value_payload) % alignof(uint4) == 0 &&
        reinterpret_cast<std::uintptr_t>(cache.key_scales) % alignof(std::uint64_t) == 0 &&
        reinterpret_cast<std::uintptr_t>(cache.value_scales) % alignof(std::uint64_t) == 0) {
        const int tile_rows = rows >= min_wide_tail_rows ? wide_rows :
            rows >= min_paired_tail_rows && rows % paired_rows == 0 ? paired_rows : 1;
        const dim3 blocks(rows / tile_rows + rows % tile_rows, gqa_head_pairs(cache));
        const auto kernel = tile_rows == wide_rows ?
            multirow_prefill_attention<wide_rows, 2, true> : tile_rows == paired_rows ?
            multirow_prefill_attention<paired_rows, 3, true> :
            multirow_prefill_attention<1, 4, true>;
        return cudaLaunchKernel(kernel, blocks, channels, args, 0, stream);
    }
    if (rows < min_paired_rows || visible > max_multirow_visible_tokens) {
        const dim3 blocks(rows, gqa_head_pairs(cache));
        return cudaLaunchKernel(protected_prefill_attention, blocks, channels, args, 0, stream);
    }
    constexpr int min_wide_rows = 128;
    const int tile_rows = rows >= min_wide_rows ? wide_rows : paired_rows;
    const dim3 blocks(rows / tile_rows + rows % tile_rows, gqa_head_pairs(cache));
    const auto kernel = tile_rows == wide_rows ? multirow_prefill_attention<wide_rows, 2> :
                                                multirow_prefill_attention<paired_rows, 3>;
    return cudaLaunchKernel(kernel, blocks, channels, args, 0, stream);
}

cudaError_t decoder::tensor_core_prefill_gqa(
    Mxfp8KvCache cache, Mxfp8PageTable pages, int position, int rows,
    const Bf16* queries, Bf16* output, cudaStream_t stream, const int* device_position) {
    Attention request{cache, pages.device_pages, position, queries, output};
    void* args[] = {&request, &device_position};
    return cudaLaunchKernel(tiled_prefill_attention, dim3(rows, gqa_head_pairs(cache)),
                            channels, args, 0, stream);
}

cudaError_t decoder::tensor_core_model_prefill_gqa(
    Mxfp8KvCache cache, Mxfp8PageTable pages, int position, int rows,
    const __nv_bfloat16* queries, __nv_bfloat16* output, cudaStream_t stream,
    PairedPrefillScratch scratch) {
    return tensor_core_multirow_prefill_gqa(cache, pages, position, rows, queries, output,
                                           stream, scratch);
}

}
