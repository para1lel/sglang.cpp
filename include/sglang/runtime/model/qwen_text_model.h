#pragma once

namespace sglang::decoder {

struct QwenTextModel {
    int hidden_size = 5120;
    int intermediate_size = 17408;
    int vocab_size = 248320;
    int layer_count = 64;
    int full_attention_interval = 4;

    constexpr bool is_full_attention(int layer_index) const {
        return (layer_index + 1) % full_attention_interval == 0;
    }
};

inline constexpr QwenTextModel qwen_text_model{};

}
