#pragma once

#include "sglang/runtime/cache/mxfp8_cache_geometry.h"
#include "sglang/runtime/model/qwen_layers.h"
#include "sglang/text_attention.h"

#include <array>
#include <cstddef>
#include <string>

namespace sglang::decoder {

struct PrefixCacheLayout {
    int world_size = 1;
    int rank = 0;
    int kv_heads = mxfp8_kv_heads;
    int gdn_key_heads = text_gdn_qk_heads;
    int gdn_value_heads = text_gdn_value_heads;
    bool draft_lane = true;
};

inline constexpr PrefixCacheLayout tp1_prefix_cache_layout{};

PrefixCacheLayout tp2_prefix_cache_layout(int rank);

std::size_t prefix_kv_payload_bytes_per_page(const PrefixCacheLayout& layout);

std::size_t prefix_kv_scale_bytes_per_page(const PrefixCacheLayout& layout);
std::array<std::size_t, 4> prefix_gdn_state_bytes(const PrefixCacheLayout& layout);


std::string prefix_cache_domain(const PrefixCacheLayout& layout);

}
