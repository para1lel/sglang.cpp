#pragma once

#include "sglang/runtime/cache/mxfp8_cache_geometry.h"
#include "sglang/runtime/model/qwen_layers.h"
#include "sglang/text_attention.h"

#include <array>
#include <cstddef>
#include <string>

namespace sglang::decoder {

/// Rank-local geometry of one decoder lane's prefix cache payload. A TP1 lane
/// stores every KV and GDN head; a TP2 lane stores half of the target GQA KV
/// heads and half of the GDN state, while the MTP draft cache stays full width
/// on the single draft lane. The hidden and residual rows are replicated and
/// therefore keep the full width on every lane.
struct PrefixCacheLayout {
    int world_size = 1;
    int rank = 0;
    int kv_heads = mxfp8_kv_heads;
    int gdn_key_heads = text_gdn_qk_heads;
    int gdn_value_heads = text_gdn_value_heads;
    bool draft_lane = true;
};

inline constexpr PrefixCacheLayout tp1_prefix_cache_layout{};

/// TP2 splits the target geometry evenly per rank; only rank 0 drafts.
constexpr PrefixCacheLayout tp2_prefix_cache_layout(int rank) {
    return {2, rank, mxfp8_kv_heads / 2, text_gdn_qk_heads / 2,
            text_gdn_value_heads / 2, rank == 0};
}

/// K and V payload bytes of one page of one layer for this lane's target cache.
inline std::size_t prefix_kv_payload_bytes_per_page(const PrefixCacheLayout& layout) {
    return static_cast<std::size_t>(mxfp8_page_size) * layout.kv_heads * mxfp8_head_dim;
}

/// K and V scale bytes of one page of one layer for this lane's target cache.
inline std::size_t prefix_kv_scale_bytes_per_page(const PrefixCacheLayout& layout) {
    return static_cast<std::size_t>(mxfp8_page_size) * layout.kv_heads * mxfp8_groups;
}
/// Rank-local byte sizes of the four GDN endpoint planes: per-layer QKV
/// history, per-layer recurrent state, and the replicated hidden and residual
/// rows of the checkpoint position.
inline std::array<std::size_t, 4> prefix_gdn_state_bytes(const PrefixCacheLayout& layout) {
    const auto qkv_width = static_cast<std::size_t>(layout.gdn_key_heads) * 2 +
                           layout.gdn_value_heads;
    return {static_cast<std::size_t>(gdn_layers) * qkv_width * text_gdn_head_dim *
                text_gdn_history_taps * sizeof(Bf16),
            static_cast<std::size_t>(gdn_layers) * layout.gdn_value_heads *
                text_gdn_head_dim * text_gdn_head_dim * sizeof(float),
            static_cast<std::size_t>(hidden_size) * sizeof(Bf16),
            static_cast<std::size_t>(hidden_size) * sizeof(Bf16)};
}


/// Layout-qualified key domain prefix. The default layout keeps the historical
/// TP1 domain byte-identical so existing records stay addressable; a TP2 lane
/// appends its own layout identity so records of different lanes or worlds are
/// never interchangeable.
std::string prefix_cache_domain(const PrefixCacheLayout& layout);

}  // namespace sglang::decoder
