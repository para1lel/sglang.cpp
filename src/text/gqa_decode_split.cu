#include "sglang/text/gqa_decode_split.cuh"

#include <cuda_bf16.h>
#include <math_constants.h>

#include <cstddef>
#include <cstdint>

namespace sglang::gqa_decode {
namespace {

constexpr int group_heads = mxfp8_query_heads / mxfp8_kv_heads;
constexpr int warps = 4;
constexpr int threads = warps * 32;
constexpr int tile_tokens = 16;
constexpr int chunk_tokens = split_chunk_tokens;
constexpr int chunks_per_page = mxfp8_page_size / chunk_tokens;
constexpr int stages = 2;
constexpr int key_stride = mxfp8_head_dim + 32;
constexpr int value_stride = mxfp8_head_dim + 16;
constexpr int scale_row = 2 * mxfp8_groups;
constexpr int value_tiles = mxfp8_head_dim / 16;
constexpr int copies_per_token = mxfp8_head_dim / 16;
constexpr int max_lane_chunks = text_attention_max_tokens / chunk_tokens;
constexpr int max_split_chunks = (max_lane_chunks + max_splits - 1) / max_splits;
constexpr int max_split_pages = (max_split_chunks + chunks_per_page - 1) / chunks_per_page + 1;
constexpr unsigned int full_mask = 0xffffffffu;
constexpr unsigned int scale_limit = 0xa0a0a0a0u;
constexpr unsigned int key_bias = 0x1e1e1e1eu;
constexpr int value_bias = 142;
constexpr int fp32_min_exponent = -126;
constexpr int e4m3_min_exponent = -9;
constexpr int fp32_exponent_bias = 127;
constexpr float query_min = 0x1p-100f;
constexpr float query_limit = 0x1p30f;
constexpr float score_scale = 0x1p-34f * 1.44269504088896341f;
constexpr float output_unscale = 0x1p-71f;

static_assert(chunk_tokens == warps * tile_tokens && group_heads == 6);
static_assert(max_splits <= 32);

struct Stage {
    alignas(16) std::uint8_t keys[chunk_tokens * key_stride];
    alignas(16) std::uint8_t values[chunk_tokens * value_stride];
    alignas(16) std::uint8_t key_scales[chunk_tokens * mxfp8_groups];
    alignas(16) std::uint8_t value_scales[chunk_tokens * mxfp8_groups];
};

/// Per-token scale factors of one warp tile, laid out [group][token].
struct WarpScales {
    float key[mxfp8_groups][tile_tokens];
    float value[mxfp8_groups][tile_tokens];
};

struct Query {
    uint4 words[mxfp8_groups][32];
    bool flag[32];
};

template <int StageCount>
struct Shared {
    Stage stage[StageCount];
    WarpScales scales[warps];
    int pages[max_split_pages];
    Query query;
};

struct Merge {
    float value[warps][group_heads][mxfp8_head_dim];
    float maximum[warps][group_heads];
    float mass[warps][group_heads];
    int flag[warps][group_heads];
    float weight[warps][group_heads];
};

static_assert(sizeof(Merge) <= sizeof(Stage) && sizeof(Stage) % 16 == 0);
static_assert(2 * (sizeof(Shared<stages>) + 1024) <= 233472);
static_assert(3 * (sizeof(Shared<1>) + 1024) <= 233472);

/// Online state of one warp. `out` holds sum(p * v) * 2^(142 - R_g) for group g, where
/// R_g is the largest value-scale byte seen in that group (packed bytes in `reference`).
struct Running {
    float out[value_tiles][4];
    float maximum;
    float mass;
    unsigned int reference[2];
    bool flag;
};

template <typename Target>
__device__ decoder::PagedDecodeLane resolve_input(const Target& input) {
    auto lane = resolve(input);
    if (!decoder::target_split_position(lane.position, input.end)) {
        lane.position = -1;
    }
    return lane;
}

__device__ __forceinline__ void copy_16(void* shared, const void* global, bool valid) {
    const auto address = static_cast<unsigned int>(__cvta_generic_to_shared(shared));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
                 :: "r"(address), "l"(global), "r"(valid ? 16 : 0) : "memory");
}

__device__ __forceinline__ void commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }

template <int Pending>
__device__ __forceinline__ void wait_pending() {
    asm volatile("cp.async.wait_group %0;" :: "n"(Pending) : "memory");
}

__device__ __forceinline__ void mma(float (&sum)[4], unsigned int a0, unsigned int a1,
                                    unsigned int a2, unsigned int a3, unsigned int b0,
                                    unsigned int b1) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(sum[0]), "+f"(sum[1]), "+f"(sum[2]), "+f"(sum[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

/// Converts four E4M3 bytes to two exact bf16x2 words: bytes 0-1 to `low`, bytes 2-3 to `high`.
__device__ __forceinline__ void expand(unsigned int bytes, unsigned int& low, unsigned int& high) {
    asm("{\n.reg .b16 l, h;\nmov.b32 {l, h}, %2;\n"
        "cvt.rn.bf16x2.e4m3x2 %0, l;\ncvt.rn.bf16x2.e4m3x2 %1, h;\n}"
        : "=r"(low), "=r"(high) : "r"(bytes));
}

__device__ __forceinline__ unsigned int pack(float low, float high) {
    const auto pair = __floats2bfloat162_rn(low, high);
    return *reinterpret_cast<const unsigned int*>(&pair);
}

__device__ __forceinline__ int byte_of(unsigned int word, int index) {
    return static_cast<int>(word >> (8 * index) & 0xffu);
}

__device__ __forceinline__ float power_of_two(int biased) {
    return __uint_as_float(static_cast<unsigned int>(biased) << 23);
}

__device__ void issue_warp_tile(const int* pages, Stage& stage,
                               const decoder::PagedDecodeLane& lane, int kv_head, int chunk,
                               int first_page, int warp, int lane_id) {
    const int half = chunk % chunks_per_page;
    const auto physical = static_cast<std::size_t>(pages[chunk / chunks_per_page - first_page]);
    const int first = chunk * chunk_tokens;
    const std::size_t payload =
        physical * gqa_payload_bytes_per_page(lane.cache.kv_heads) +
        (static_cast<std::size_t>(half * chunk_tokens) * lane.cache.kv_heads + kv_head) *
            mxfp8_head_dim;
    #pragma unroll
    for (int step = 0; step < tile_tokens * copies_per_token / 32; ++step) {
        const int index = lane_id + step * 32;
        const int token = warp * tile_tokens + index / copies_per_token;
        const int column = index % copies_per_token * 16;
        const bool valid = first + token <= lane.position;
        const auto source = payload + token * lane.cache.kv_heads * mxfp8_head_dim + column;
        copy_16(stage.keys + token * key_stride + column, lane.cache.key_payload + source, valid);
        copy_16(stage.values + token * value_stride + column, lane.cache.value_payload + source,
                valid);
    }
    static_assert(mxfp8_groups == 8 && tile_tokens == 16);
    const int token = warp * tile_tokens + lane_id % tile_tokens;
    const int offset = token % mxfp8_group_size * scale_row +
                       token / mxfp8_group_size * mxfp8_groups;
    const bool value = lane_id >= tile_tokens;
    const std::size_t source =
        (physical * lane.cache.kv_heads + kv_head) * mxfp8_page_size * mxfp8_groups +
        token % mxfp8_group_size * (mxfp8_page_size / mxfp8_group_size) * mxfp8_groups +
        half * scale_row + token / mxfp8_group_size * mxfp8_groups;
    const auto dest = __cvta_generic_to_shared(
        (value ? stage.value_scales : stage.key_scales) + offset);
    const auto* source_ptr = (value ? lane.cache.value_scales : lane.cache.key_scales) + source;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;"
                 :: "r"(static_cast<unsigned int>(dest)), "l"(source_ptr) : "memory");
}

/// Loads one BF16 query head per MMA row and flags operands outside the FP32 score range.
__device__ void load_query(Query& query, const __nv_bfloat16* queries, int kv_head, int lane) {
    const int row = lane / 4;
    const int quad = lane % 4;
    const bool active = row < group_heads;
    const auto* base = queries + (kv_head * group_heads + row) * mxfp8_head_dim + quad * 8;
    bool bad = false;
    #pragma unroll
    for (int group = 0; group < mxfp8_groups; ++group) {
        const uint4 raw = active ? *reinterpret_cast<const uint4*>(base + group * 32)
                                 : make_uint4(0, 0, 0, 0);
        query.words[group][lane] = raw;
        const unsigned int pairs[] = {raw.x, raw.y, raw.z, raw.w};
        #pragma unroll
        for (int index = 0; index < 4; ++index) {
            const auto bits = pairs[index];
            const float low = fabsf(__uint_as_float(bits << 16));
            const float high = fabsf(__uint_as_float(bits & 0xffff0000u));
            bad |= !(low < query_limit && high < query_limit);
            bad |= (low != 0.0f && low < query_min) || (high != 0.0f && high < query_min);
        }
    }
    query.flag[lane] = active && bad;
}

__device__ int slot_token(int warp, int quad, int slot) {
    return warp * tile_tokens + slot / 2 * 8 + 2 * quad + slot % 2;
}

__device__ void raise_reference(Running& run, const unsigned int (&next)[2]) {
    if (next[0] == run.reference[0] && next[1] == run.reference[1]) {
        return;
    }
    #pragma unroll
    for (int group = 0; group < mxfp8_groups; ++group) {
        const int change =
            byte_of(run.reference[group / 4], group % 4) - byte_of(next[group / 4], group % 4);
        run.flag |= change < -126 && run.maximum != -CUDART_INF_F;
        const float factor = power_of_two(max(127 + change, 0));
        #pragma unroll
        for (int index = 0; index < 4; ++index) {
            run.out[2 * group][index] *= factor;
            run.out[2 * group + 1][index] *= factor;
        }
    }
    run.reference[0] = next[0];
    run.reference[1] = next[1];
}

/// Lane l decodes the scales of token l / 2, groups 4 * (l % 2) .. +3, into `scales`.
/// Key factors are 2^(e - 97); value factors are 2^(e - R_g + 15) and zero for dead tokens.
__device__ int prepare_scales(const Stage& stage, WarpScales& scales, Running& run, int first,
                              int position, int warp, int lane) {
    const int local = lane / 2;
    const int half = lane % 2;
    const int token = warp * tile_tokens + local;
    const bool live = first + token <= position;
    const int offset = token % 32 * scale_row + token / 32 * mxfp8_groups + 4 * half;
    const auto key = *reinterpret_cast<const unsigned int*>(stage.key_scales + offset);
    const auto value = *reinterpret_cast<const unsigned int*>(stage.value_scales + offset);
    const bool over =
        live && (__vcmpgtu4(key, scale_limit) | __vcmpgtu4(value, scale_limit)) != 0u;
    run.flag |= __any_sync(full_mask, over);
    const unsigned int key_bytes = __vminu4(key, scale_limit) + key_bias;
    const unsigned int value_bytes = live ? __vminu4(value, scale_limit) : 0u;
    unsigned int tile = value_bytes;
    #pragma unroll
    for (int step = 2; step < 32; step *= 2) {
        tile = __vmaxu4(tile, __shfl_xor_sync(full_mask, tile, step));
    }
    const unsigned int next[2] = {__vmaxu4(run.reference[0], __shfl_sync(full_mask, tile, 0)),
                                  __vmaxu4(run.reference[1], __shfl_sync(full_mask, tile, 1))};
    raise_reference(run, next);
    const unsigned int reference = half ? run.reference[1] : run.reference[0];
    int min_exponent = value_bias;
    __syncwarp();
    #pragma unroll
    for (int index = 0; index < 4; ++index) {
        const int exponent =
            byte_of(value_bytes, index) + value_bias - byte_of(reference, index);
        min_exponent = min(min_exponent, exponent);
        scales.key[4 * half + index][local] = power_of_two(byte_of(key_bytes, index));
        scales.value[4 * half + index][local] = live ? power_of_two(max(exponent, 0)) : 0.0f;
    }
    __syncwarp();
    return __reduce_min_sync(full_mask, live ? min_exponent : value_bias);
}

template <bool SharedTail = false>
__device__ void tile_scores(const Stage& stage, const WarpScales& scales, const Query& query,
                            int warp, int row, int quad, float (&score)[4], int last = 0) {
    #pragma unroll
    for (int slot = 0; slot < 4; ++slot) {
        score[slot] = 0.0f;
    }
    #pragma unroll
    for (int group = 0; group < mxfp8_groups; ++group) {
        const auto words = query.words[group][4 * row + quad];
        #pragma unroll
        for (int tile = 0; tile < 2; ++tile) {
            const int token = warp * tile_tokens + tile * 8 + row;
            const auto bits = SharedTail && token > last ? make_uint2(0, 0) :
                *reinterpret_cast<const uint2*>(
                    stage.keys + token * key_stride + group * mxfp8_group_size + quad * 8);
            unsigned int k0, k1, k2, k3;
            expand(bits.x, k0, k1);
            expand(bits.y, k2, k3);
            float sum[4] = {};
            mma(sum, words.x, 0u, words.y, 0u, k0, k1);
            mma(sum, words.z, 0u, words.w, 0u, k2, k3);
            const auto factor =
                *reinterpret_cast<const float2*>(&scales.key[group][tile * 8 + 2 * quad]);
            score[2 * tile] = fmaf(sum[0], factor.x, score[2 * tile]);
            score[2 * tile + 1] = fmaf(sum[1], factor.y, score[2 * tile + 1]);
        }
    }
}

// Use the 16 MMA rows for keys and the eight columns for query heads.
// Reduce K32 groups in the original order before restoring the score lane layout.
__device__ void transposed_scores(const Stage& stage, const WarpScales& scales,
                                  const Query& query, int warp, int row, int quad,
                                  float (&score)[4]) {
    float transposed[4] = {};
    #pragma unroll
    for (int group = 0; group < mxfp8_groups; ++group) {
        const auto words = query.words[group][4 * row + quad];
        const auto* keys = stage.keys + (warp * tile_tokens + row) * key_stride +
                           group * mxfp8_group_size + quad * 8;
        const auto low = *reinterpret_cast<const uint2*>(keys);
        const auto high = *reinterpret_cast<const uint2*>(keys + 8 * key_stride);
        unsigned int lo[4], hi[4];
        expand(low.x, lo[0], lo[1]);
        expand(low.y, lo[2], lo[3]);
        expand(high.x, hi[0], hi[1]);
        expand(high.y, hi[2], hi[3]);
        float sum[4] = {};
        mma(sum, lo[0], hi[0], lo[1], hi[1], words.x, words.y);
        mma(sum, lo[2], hi[2], lo[3], hi[3], words.z, words.w);
        const float factor_low = scales.key[group][row];
        const float factor_high = scales.key[group][row + 8];
        transposed[0] = fmaf(sum[0], factor_low, transposed[0]);
        transposed[1] = fmaf(sum[1], factor_low, transposed[1]);
        transposed[2] = fmaf(sum[2], factor_high, transposed[2]);
        transposed[3] = fmaf(sum[3], factor_high, transposed[3]);
    }
    const int lane0 = 8 * quad + row / 2;
    const int lane1 = lane0 + 4;
    const float even0 = __shfl_sync(full_mask, transposed[0], lane0);
    const float even1 = __shfl_sync(full_mask, transposed[0], lane1);
    const float odd0 = __shfl_sync(full_mask, transposed[1], lane0);
    const float odd1 = __shfl_sync(full_mask, transposed[1], lane1);
    const float even2 = __shfl_sync(full_mask, transposed[2], lane0);
    const float even3 = __shfl_sync(full_mask, transposed[2], lane1);
    const float odd2 = __shfl_sync(full_mask, transposed[3], lane0);
    const float odd3 = __shfl_sync(full_mask, transposed[3], lane1);
    score[0] = row % 2 ? odd0 : even0;
    score[1] = row % 2 ? odd1 : even1;
    score[2] = row % 2 ? odd2 : even2;
    score[3] = row % 2 ? odd3 : even3;
}

__device__ void update(Running& run, const float (&score)[4], float min_delta,
                       unsigned int valid, int quad, float (&weight)[4]) {
    float logit[4];
    float tile = -CUDART_INF_F;
    #pragma unroll
    for (int slot = 0; slot < 4; ++slot) {
        const bool live = valid >> slot & 1u;
        logit[slot] = live ? score[slot] * score_scale : -CUDART_INF_F;
        run.flag |= live && !isfinite(logit[slot]);
        tile = fmaxf(tile, logit[slot]);
    }
    tile = fmaxf(tile, __shfl_xor_sync(full_mask, tile, 1));
    tile = fmaxf(tile, __shfl_xor_sync(full_mask, tile, 2));
    const float next = fmaxf(run.maximum, tile);
    const float base = next == -CUDART_INF_F ? 0.0f : next;
    const float prior = run.maximum - base;
    const float alpha = exp2f(prior);
    run.flag |= run.maximum != -CUDART_INF_F && prior < -126.0f;
    float sum = 0.0f;
    #pragma unroll
    for (int slot = 0; slot < 4; ++slot) {
        const float delta = logit[slot] - base;
        run.flag |= (valid >> slot & 1u) && delta < min_delta;
        weight[slot] = exp2f(delta);
        sum += weight[slot];
    }
    run.mass = fmaf(run.mass, alpha, sum);
    run.maximum = next;
    if (!__all_sync(full_mask, alpha == 1.0f)) {
        const float even = __shfl_sync(full_mask, alpha, 8 * quad);
        const float odd = __shfl_sync(full_mask, alpha, 8 * quad + 4);
        #pragma unroll
        for (int tile_index = 0; tile_index < value_tiles; ++tile_index) {
            run.out[tile_index][0] *= even;
            run.out[tile_index][1] *= odd;
            run.out[tile_index][2] *= even;
            run.out[tile_index][3] *= odd;
        }
    }
}

/// O^T += V^T P'. Rows of a value tile are channels 32g + 4r (+1) and 32g + 4r + 2 (+3).
template <bool SharedTail = false>
__device__ void accumulate(const Stage& stage, const WarpScales& scales, const float (&weight)[4],
                           int warp, int row, int quad, Running& run, int last = 0) {
    const int token = warp * tile_tokens + 2 * quad;
    const auto* base = stage.values + token * value_stride + 4 * row;
    #pragma unroll
    for (int group = 0; group < mxfp8_groups; ++group) {
        const auto low = *reinterpret_cast<const float2*>(&scales.value[group][2 * quad]);
        const auto high = *reinterpret_cast<const float2*>(&scales.value[group][8 + 2 * quad]);
        const auto b0 = pack(weight[0] * low.x, weight[1] * low.y);
        const auto b1 = pack(weight[2] * high.x, weight[3] * high.y);
        const auto* column = base + group * mxfp8_group_size;
        const auto wa = SharedTail && token > last ? 0u :
            *reinterpret_cast<const unsigned int*>(column);
        const auto wb = SharedTail && token + 1 > last ? 0u :
            *reinterpret_cast<const unsigned int*>(column + value_stride);
        const auto wc = SharedTail && token + 8 > last ? 0u :
            *reinterpret_cast<const unsigned int*>(column + 8 * value_stride);
        const auto wd = SharedTail && token + 9 > last ? 0u :
            *reinterpret_cast<const unsigned int*>(column + 9 * value_stride);
        unsigned int ab[4], cd[4];
        expand(__byte_perm(wa, wb, 0x5140u), ab[0], ab[1]);
        expand(__byte_perm(wa, wb, 0x7362u), ab[2], ab[3]);
        expand(__byte_perm(wc, wd, 0x5140u), cd[0], cd[1]);
        expand(__byte_perm(wc, wd, 0x7362u), cd[2], cd[3]);
        mma(run.out[2 * group], ab[0], ab[1], cd[0], cd[1], b0, b1);
        mma(run.out[2 * group + 1], ab[2], ab[3], cd[2], cd[3], b0, b1);
    }
}

template <bool SharedTail = false, bool Transposed = false>
__device__ void process_tile(const Stage& stage, WarpScales& scales, const Query& query,
                             Running& run, int chunk, int position, int warp, int lane) {
    const int first = chunk * chunk_tokens;
    if (first + warp * tile_tokens > position) {
        return;
    }
    const int row = lane / 4;
    const int quad = lane % 4;
    unsigned int valid = 0;
    #pragma unroll
    for (int slot = 0; slot < 4; ++slot) {
        valid |= (first + slot_token(warp, quad, slot) <= position ? 1u : 0u) << slot;
    }
    const int min_exponent = prepare_scales(stage, scales, run, first, position, warp, lane);
    const int min_value_delta =
        fp32_min_exponent - e4m3_min_exponent - (min_exponent - fp32_exponent_bias);
    const float min_delta = static_cast<float>(max(fp32_min_exponent, min_value_delta));
    float score[4];
    if constexpr (Transposed) {
        transposed_scores(stage, scales, query, warp, row, quad, score);
    } else {
        tile_scores<SharedTail>(stage, scales, query, warp, row, quad, score, position - first);
    }
    float weight[4];
    update(run, score, min_delta, valid, quad, weight);
    accumulate<SharedTail>(stage, scales, weight, warp, row, quad, run, position - first);
}

/// Writes the warp state to `merge` at true scale: out * 2^(R_g - 142).
__device__ void store_warp(Merge& merge, const Running& run, int warp, int row, int quad) {
    float mass = run.mass;
    mass += __shfl_xor_sync(full_mask, mass, 1);
    mass += __shfl_xor_sync(full_mask, mass, 2);
    int flag = run.flag ? 1 : 0;
    flag |= __shfl_xor_sync(full_mask, flag, 1);
    flag |= __shfl_xor_sync(full_mask, flag, 2);
    if (quad == 0 && row < group_heads) {
        merge.maximum[warp][row] = run.maximum;
        merge.mass[warp][row] = mass;
        merge.flag[warp][row] = flag;
    }
    if (2 * quad + 1 < group_heads) {
        #pragma unroll
        for (int tile = 0; tile < value_tiles; ++tile) {
            const int group = tile / 2;
            const float unscale =
                output_unscale * power_of_two(byte_of(run.reference[group / 4], group % 4) + 56);
            const int channel = group * mxfp8_group_size + 4 * row + 2 * (tile % 2);
            merge.value[warp][2 * quad][channel] = run.out[tile][0] * unscale;
            merge.value[warp][2 * quad + 1][channel] = run.out[tile][1] * unscale;
            merge.value[warp][2 * quad][channel + 1] = run.out[tile][2] * unscale;
            merge.value[warp][2 * quad + 1][channel + 1] = run.out[tile][3] * unscale;
        }
    }
}

__device__ void write_partials(const Merge& merge, void* scratch, int kv_head, int split) {
    for (int item = threadIdx.x; item < group_heads * mxfp8_head_dim; item += threads) {
        const int head = item / mxfp8_head_dim;
        const int channel = item % mxfp8_head_dim;
        float maximum = -CUDART_INF_F;
        int flag = 0;
        #pragma unroll
        for (int warp = 0; warp < warps; ++warp) {
            maximum = fmaxf(maximum, merge.maximum[warp][head]);
            flag |= merge.flag[warp][head];
        }
        float mass = 0.0f;
        float value = 0.0f;
        #pragma unroll
        for (int warp = 0; warp < warps; ++warp) {
            const float previous = merge.maximum[warp][head];
            const float delta = previous - maximum;
            flag |= previous != -CUDART_INF_F && delta < -126.0f;
            const float weight = previous == -CUDART_INF_F ? 0.0f : exp2f(delta);
            mass = fmaf(weight, merge.mass[warp][head], mass);
            value = fmaf(weight, merge.value[warp][head][channel], value);
        }
        auto& out = partials(scratch, kv_head * group_heads + head)[split];
        out.value[channel] = value;
        if (channel == 0) {
            out.maximum = flag ? CUDART_NAN_F : maximum;
            out.mass = mass;
        }
    }
}

__device__ void write_weighted_partials(Merge& merge, void* scratch, int kv_head, int split,
                                       int thread = threadIdx.x) {
    if (thread < group_heads) {
        float maximum = -CUDART_INF_F;
        int flag = 0;
        #pragma unroll
        for (int warp = 0; warp < warps; ++warp) {
            maximum = fmaxf(maximum, merge.maximum[warp][thread]);
            flag |= merge.flag[warp][thread];
        }
        float mass = 0.0f;
        #pragma unroll
        for (int warp = 0; warp < warps; ++warp) {
            const float previous = merge.maximum[warp][thread];
            const float delta = previous - maximum;
            flag |= previous != -CUDART_INF_F && delta < -126.0f;
            const float weight = previous == -CUDART_INF_F ? 0.0f : exp2f(delta);
            merge.weight[warp][thread] = weight;
            mass = fmaf(weight, merge.mass[warp][thread], mass);
        }
        auto& out = partials(scratch, kv_head * group_heads + thread)[split];
        out.maximum = flag ? CUDART_NAN_F : maximum;
        out.mass = mass;
    }
    __syncthreads();
    for (int item = thread; item < group_heads * mxfp8_head_dim; item += threads) {
        const int head = item / mxfp8_head_dim;
        const int channel = item % mxfp8_head_dim;
        float value = 0.0f;
        #pragma unroll
        for (int warp = 0; warp < warps; ++warp) {
            value = fmaf(merge.weight[warp][head], merge.value[warp][head][channel], value);
        }
        partials(scratch, kv_head * group_heads + head)[split].value[channel] = value;
    }
}

template <typename Input, bool SingleStage = false>
__global__ void __launch_bounds__(threads, SingleStage ? 3 : 2)
split_kernel(const __grid_constant__ Input input) {
    alignas(16) extern __shared__ unsigned char shared_raw[];
    constexpr int stage_count = SingleStage ? 1 : stages;
    auto& shared = *reinterpret_cast<Shared<stage_count>*>(shared_raw);
    const auto lane = resolve_input(input);
    if (!readable(lane) || !tensor_aligned(lane)) {
        return;
    }
    const int chunks = lane.position / chunk_tokens + 1;
    const int splits = split_count(lane.position);
    const int split = static_cast<int>(blockIdx.x);
    if (split >= splits) {
        return;
    }
    const int kv_head = static_cast<int>(blockIdx.y);
    if (kv_head >= lane.cache.kv_heads) {
        return;
    }
    const int first = split * chunks / splits;
    const int last = (split + 1) * chunks / splits;
    const int first_page = first / chunks_per_page;
    const int page_total = (last - 1) / chunks_per_page - first_page + 1;
    for (int index = threadIdx.x; index < page_total; index += threads) {
        shared.pages[index] = lane.pages.device_pages[first_page + index];
    }
    const int warp = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    if (warp == 0) {
        load_query(shared.query, lane.buffers.queries, kv_head, lane_id);
    }
    __syncthreads();
    Running run{};
    run.maximum = -CUDART_INF_F;
    run.flag = shared.query.flag[lane_id];
    if constexpr (SingleStage) {
        for (int chunk = first; chunk < last; ++chunk) {
            issue_warp_tile(shared.pages, shared.stage[0], lane, kv_head, chunk, first_page, warp,
                            lane_id);
            commit();
            wait_pending<0>();
            __syncwarp();
            process_tile<false, true>(shared.stage[0], shared.scales[warp], shared.query,
                                     run, chunk, lane.position, warp, lane_id);
            __syncwarp();
        }
    } else {
        #pragma unroll
        for (int index = 0; index < stages - 1; ++index) {
            if (first + index < last) {
                issue_warp_tile(shared.pages, shared.stage[index], lane, kv_head, first + index,
                                first_page, warp, lane_id);
            }
            commit();
        }
        for (int chunk = first; chunk < last; ++chunk) {
            wait_pending<stages - 2>();
            __syncwarp();
            const int next = chunk + stages - 1;
            if (next < last) {
                issue_warp_tile(shared.pages, shared.stage[(next - first) % stages], lane, kv_head,
                                next, first_page, warp, lane_id);
            }
            commit();
            process_tile(shared.stage[(chunk - first) % stages], shared.scales[warp], shared.query,
                         run, chunk, lane.position, warp, lane_id);
        }
        wait_pending<0>();
    }
    __syncthreads();
    auto& merge = *reinterpret_cast<Merge*>(shared_raw);
    store_warp(merge, run, warp, lane_id / 4, lane_id % 4);
    __syncthreads();
    if constexpr (SingleStage) {
        write_weighted_partials(merge, lane.buffers.scratch, kv_head, split);
    } else {
        write_partials(merge, lane.buffers.scratch, kv_head, split);
    }
}

/// Rank-local KV-head count of a split launch input: TP2 lanes carry half the
/// heads of the full-geometry cache. Bound descriptors keep their cache on the
/// device, so the host cannot read its head count; size their grid for the
/// maximum and let the kernel's per-lane guard skip the surplus blocks.
int input_kv_heads(const TargetLane& lane) {
    return lane.bound.position ? mxfp8_kv_heads : lane.eager.cache.kv_heads;
}

int input_kv_heads(const TargetBatch&) {
    return mxfp8_kv_heads;
}

int input_kv_heads(const EagerTargetBatch& batch) {
    return batch.lanes[0].cache.kv_heads;
}

template <bool SingleStage = false, typename Input>
cudaError_t launch_split(const Input& input, int count, cudaStream_t stream) {
    constexpr std::size_t shared_bytes = sizeof(Shared<SingleStage ? 1 : stages>);
    const auto status = cudaFuncSetAttribute(split_kernel<Input, SingleStage>,
                                            cudaFuncAttributeMaxDynamicSharedMemorySize,
                                            static_cast<int>(shared_bytes));
    if (status != cudaSuccess) {
        return status;
    }
    split_kernel<Input, SingleStage>
        <<<dim3(max_splits, input_kv_heads(input), count), threads, shared_bytes, stream>>>(input);
    return cudaGetLastError();
}

#include "sglang/text/gqa/gqa_decode_reuse.cuh"

}

std::optional<cudaError_t> try_launch_reuse(const EagerTargetBatch& batch, int count,
                                           cudaStream_t stream) {
    return launch_reuse(batch, count, stream);
}

cudaError_t launch(TargetLane lane, cudaStream_t stream) {
    return launch_split(lane, 1, stream);
}

cudaError_t launch(const TargetBatch& batch, int count, cudaStream_t stream) {
    return launch_split(batch, count, stream);
}

cudaError_t launch(const EagerTargetBatch& batch, int count, cudaStream_t stream) {
    if (count >= 2 * text_gdn_max_verify_rows && batch.end == text_attention_max_tokens) {
        return launch_split<true>(batch, count, stream);
    }
    return launch_split(batch, count, stream);
}

}
