#include "sglang/runtime/cache/prefix_cache_layout.h"

namespace sglang::decoder {

PrefixCacheLayout tp2_prefix_cache_layout(int rank) {
    return {
        2, rank, mxfp8_kv_heads / 2, text_gdn_qk_heads / 2, text_gdn_value_heads / 2, true,
    };
}

std::size_t prefix_kv_payload_bytes_per_page(const PrefixCacheLayout& layout) {
    return static_cast<std::size_t>(mxfp8_page_size) * layout.kv_heads * mxfp8_head_dim;
}

std::size_t prefix_kv_scale_bytes_per_page(const PrefixCacheLayout& layout) {
    return static_cast<std::size_t>(mxfp8_page_size) * layout.kv_heads * mxfp8_groups;
}

std::array<std::size_t, 4> prefix_gdn_state_bytes(const PrefixCacheLayout& layout) {
    const auto qkv_width = static_cast<std::size_t>(layout.gdn_key_heads) * 2 +
                           layout.gdn_value_heads;
    return {static_cast<std::size_t>(gdn_layers) * qkv_width * text_gdn_head_dim *
                text_gdn_history_taps * sizeof(Bf16),
            static_cast<std::size_t>(gdn_layers) * layout.gdn_value_heads *
                text_gdn_head_dim * text_gdn_head_dim * sizeof(float),
            static_cast<std::size_t>(hidden_size) * sizeof(Bf16),
            static_cast<std::size_t>(hidden_size) * sizeof(Bf16)};
}

}
