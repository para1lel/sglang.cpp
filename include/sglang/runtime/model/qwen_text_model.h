#pragma once

#include <string_view>

namespace sglang::decoder {

struct QwenTextModel {
    static constexpr std::string_view architecture = "Qwen3_5ForConditionalGeneration";
    static constexpr std::string_view model_type = "qwen3_5";
    static constexpr std::string_view text_model_type = "qwen3_5_text";

    static constexpr int query_heads = 24;
    static constexpr int kv_heads = 4;
    static constexpr int attention_head_dim = 256;
    static constexpr int gdn_key_heads = 16;
    static constexpr int gdn_value_heads = 48;
    static constexpr int gdn_head_dim = 128;
    static constexpr int gdn_conv_kernel_dim = 4;
    static constexpr int max_position_embeddings = 262144;
    static constexpr int fp8_weight_block_size = 128;

    int hidden_size = 5120;
    int intermediate_size = 17408;
    int vocab_size = 248320;
    int layer_count = 64;
    int full_attention_interval = 4;

    struct LayerMapping {
        bool full_attention;
        int bank;
    };

    constexpr bool is_full_attention(int layer_index) const {
        return (layer_index + 1) % full_attention_interval == 0;
    }

    constexpr int full_attention_layers() const { return layer_count / full_attention_interval; }
    constexpr int gdn_layers() const { return layer_count - full_attention_layers(); }

    constexpr LayerMapping layer_mapping(int layer_index) const {
        const int full_attention_before = layer_index / full_attention_interval;
        const bool full_attention = is_full_attention(layer_index);
        return {full_attention, full_attention ? full_attention_before
                                              : layer_index - full_attention_before};
    }
};

inline constexpr QwenTextModel qwen_text_model{};

}
