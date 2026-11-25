#pragma once

#include "sglang/text/mxfp8_gqa_prefill_common.cuh"

#include <cstddef>

namespace sglang {
namespace {

constexpr int tmem_swizzle = 64;
constexpr int tmem_producers = 256;
constexpr int pair_key_rows = flash_keys / 2;
constexpr int pair_value_channels = channels / 2;
constexpr int pair_value_swizzle = 128;
constexpr int pair_page_halves = mxfp8_page_size / flash_keys;
constexpr int pair_scale_bytes = mxfp8_page_size * mxfp8_groups;

struct alignas(128) PairPayload {
    uint4 key[pair_page_halves][pair_key_rows][channels / sizeof(uint4)];
    uint4 value[pair_page_halves][flash_keys][pair_value_channels / sizeof(uint4)];
    std::uint8_t key_scales[pair_scale_bytes];
    std::uint8_t value_scales[pair_scale_bytes];
};

struct alignas(1024) PairTile {
    std::uint16_t query[channels / tmem_swizzle][flash_queries][tmem_swizzle];
    std::uint16_t key[channels / tmem_swizzle][pair_key_rows][tmem_swizzle];
    std::uint16_t value[2][pair_value_channels / tmem_swizzle][flash_keys][tmem_swizzle];
    std::uint16_t probability[2][flash_queries][flash_keys];
    PairPayload raw[2];
    int bad_query[flash_queries];
    int bad_key;
    Word tmem;
    alignas(8) unsigned long long copy_ready;
    alignas(8) unsigned long long scores_ready[2];
    alignas(8) unsigned long long output_ready[2];
    alignas(8) unsigned long long operands_ready[2];
    alignas(8) unsigned long long key_ready;
};

static_assert(mxfp8_page_size == 2 * flash_keys && mxfp8_page_size == 2 * flash_rows);
static_assert(flash_chunk % mxfp8_page_size == 0);
static_assert(sizeof(PairPayload) ==
              pair_page_halves * (pair_key_rows * channels + flash_keys * pair_value_channels) +
                  2 * pair_scale_bytes);
static_assert(pair_value_channels == pair_value_swizzle && sizeof(PairPayload) % 1024 == 0 &&
              offsetof(PairPayload, value) % 1024 == 0 && offsetof(PairTile, raw) % 1024 == 0);
static_assert(offsetof(PairPayload, key) % 128 == 0 &&
              offsetof(PairPayload, value) % 128 == 0 &&
              offsetof(PairPayload, key_scales) % 128 == 0 &&
              offsetof(PairPayload, value_scales) % 128 == 0);
static_assert(offsetof(PairTile, raw) % alignof(PairPayload) == 0 &&
              offsetof(PairTile, query) % 1024 == 0 && offsetof(PairTile, key) % 1024 == 0 &&
              offsetof(PairTile, value) % 1024 == 0 &&
              offsetof(PairTile, probability) % 1024 == 0);

__device__ __forceinline__ int tmem_swizzled(int row, int channel) {
    return channel ^ ((row & 7) * 8);
}

__device__ __forceinline__ Word tmem_fp8_pair(Word packed, Word scale) {
    Word result;
    asm("{ .reg .b32 pair;\n"
        "cvt.rn.bf16x2.e4m3x2 pair, %1;\n"
        "mul.rn.bf16x2 %0, pair, %2; }"
        : "=r"(result) : "h"(static_cast<short>(packed)), "r"(scale));
    return result;
}

__device__ __forceinline__ Word tmem_fp8_nan(Word packed) {
    return ((packed & 0x7f7f7f7fu) + 0x01010101u) & 0x80808080u;
}

__device__ __forceinline__ void tmem_decode(std::uint16_t* destination, uint4 payload,
                                             int scale, bool& bad) {
    const Word nan = tmem_fp8_nan(payload.x) | tmem_fp8_nan(payload.y) |
                     tmem_fp8_nan(payload.z) | tmem_fp8_nan(payload.w);
    if (scale < 15 || scale > 148 || nan) {
        flash_store_vector<false>(destination, payload, scale, bad);
        return;
    }
    const Word gain = static_cast<Word>(scale) * 0x00800080u;
    auto* output = reinterpret_cast<uint4*>(destination);
    output[0] = make_uint4(tmem_fp8_pair(payload.x, gain), tmem_fp8_pair(payload.x >> 16, gain),
                           tmem_fp8_pair(payload.y, gain), tmem_fp8_pair(payload.y >> 16, gain));
    output[1] = make_uint4(tmem_fp8_pair(payload.z, gain), tmem_fp8_pair(payload.z >> 16, gain),
                           tmem_fp8_pair(payload.w, gain), tmem_fp8_pair(payload.w >> 16, gain));
}

template <bool Key>
__device__ __forceinline__ void pair_decode_tail(
    std::uint16_t* output, const Mxfp8KvCache& cache, int page, int token, int kv_head,
    int channel, bool& bad) {
    const auto* payload = Key ? cache.key_payload : cache.value_payload;
    const auto* scales = Key ? cache.key_scales : cache.value_scales;
    const auto offset = payload_offset(cache.kv_heads, page, token, kv_head) + channel;
    const auto scale = scale_offset(cache.kv_heads, page, token, kv_head)
        + channel / mxfp8_group_size;
    tmem_decode(output, *reinterpret_cast<const uint4*>(payload + offset), scales[scale], bad);
}

template <bool Key, bool PartialPage>
__device__ void pair_stage_kv(PairTile& tile, const PairPayload& payload, int first, int rank,
                             const Attention& request, int kv_head, int full_end, int end,
                             int stage = 0) {
    constexpr int vector_elements = 16;
    constexpr int width = Key ? channels : pair_value_channels;
    constexpr int row_count = Key ? pair_key_rows : flash_keys;
    constexpr int token_lanes = width / vector_elements;
    constexpr int vectors_per_subtile = tmem_swizzle / vector_elements;
    constexpr int paired_row_bits = vectors_per_subtile | 1;
    constexpr int value_swizzle_rows = pair_value_swizzle / sizeof(uint4);
    const int producer = threadIdx.x - flash_queries;
    const int vector_lane = producer % token_lanes;
    const int row = producer / token_lanes;
    const int first_row = Key ? row : row ^ ((vector_lane / vectors_per_subtile) * paired_row_bits);
    const int vector = Key ? vector_lane :
        vector_lane % vectors_per_subtile + (row & 1) * vectors_per_subtile;
    const int channel = vector * vector_elements;
    const int scale_channel = channel + (Key ? 0 : rank * pair_value_channels);
    const int half = (first / flash_keys) % pair_page_halves;
    const auto* scales = Key ? payload.key_scales : payload.value_scales;
    const bool tail = PartialPage && first >= full_end;
    const int page = tail ? request.pages[first / mxfp8_page_size] : 0;
    for (int index = first_row; index < row_count;
         index += tmem_producers / token_lanes) {
        const int sub = channel / tmem_swizzle;
        const int column = channel % tmem_swizzle;
        const int low = tmem_swizzled(index, column);
        const int high = tmem_swizzled(index, column + 8);
        const int logical = first + index + (Key ? rank * pair_key_rows : 0);
        const int token = logical % mxfp8_page_size;
        bool bad = false;
        uint4 packed[2];
        auto* decoded = reinterpret_cast<std::uint16_t*>(packed);
        if (tail) {
            if (logical < end) {
                pair_decode_tail<Key>(decoded, request.cache, page, token, kv_head,
                                      scale_channel, bad);
            } else {
                packed[0] = {};
                packed[1] = {};
            }
        } else {
            const int scale = (token % mxfp8_group_size) * (pair_scale_bytes / mxfp8_group_size)
                             + (token / mxfp8_group_size) * mxfp8_groups
                             + scale_channel / mxfp8_group_size;
            const int value_column = vector ^ (index % value_swizzle_rows);
            const uint4 data = Key ? payload.key[half][index][vector] :
                                     payload.value[half][index][value_column];
            tmem_decode(decoded, data, scales[scale], bad);
        }
        if constexpr (Key) {
            *reinterpret_cast<uint4*>(&tile.key[sub][index][low]) = packed[0];
            *reinterpret_cast<uint4*>(&tile.key[sub][index][high]) = packed[1];
            const Word active = __ballot_sync(warp_mask, bad);
            const int lane_base = (producer % 32) & ~(token_lanes - 1);
            if (vector == 0 && ((active >> lane_base) & ((1u << token_lanes) - 1))) {
                atomicMin(&tile.bad_key, logical);
            }
        } else {
            *reinterpret_cast<uint4*>(&tile.value[stage][sub][index][low]) = packed[0];
            *reinterpret_cast<uint4*>(&tile.value[stage][sub][index][high]) = packed[1];
            if (bad) {
                atomicMin(&tile.bad_key, logical);
            }
        }
    }
}

}
}
