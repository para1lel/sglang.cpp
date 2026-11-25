#pragma once

#include "sglang/runtime/model/qwen_text_model.h"
#include "sglang/text_attention.h"
#include "sglang/text_ops.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace sglang::decoder {

using Bf16 = __nv_bfloat16;

inline constexpr int hidden_size = qwen_text_model.hidden_size;
inline constexpr int intermediate_size = qwen_text_model.intermediate_size;
inline constexpr int vocab_size = qwen_text_model.vocab_size;
inline constexpr int layer_count = qwen_text_model.layer_count;
inline constexpr int full_attention_layers = qwen_text_model.full_attention_layers();
inline constexpr int gdn_layers = qwen_text_model.gdn_layers();
inline constexpr std::size_t gdn_history_elements = text_gdn_qkv_width * text_gdn_history_taps;
inline constexpr std::size_t gdn_state_elements =
    text_gdn_value_heads * text_gdn_head_dim * text_gdn_head_dim;
// Per-layer GDN state geometry for a rank-local shard under tensor parallel
// size 2: half the QKV width and half the value heads, so state banks pack
// contiguous rank-local planes at these strides (prefix cache layout).
inline constexpr std::size_t gdn_local_history_elements =
    text_gdn_qkv_width / 2 * text_gdn_history_taps;
inline constexpr std::size_t gdn_local_state_elements =
    text_gdn_value_heads / 2 * text_gdn_head_dim * text_gdn_head_dim;
static_assert(gdn_local_history_elements * 2 == gdn_history_elements);
static_assert(gdn_local_state_elements * 2 == gdn_state_elements);

static_assert(mxfp8_query_heads == qwen_text_model.query_heads);
static_assert(mxfp8_kv_heads == qwen_text_model.kv_heads);
static_assert(mxfp8_head_dim == qwen_text_model.attention_head_dim);
static_assert(text_gdn_qk_heads == qwen_text_model.gdn_key_heads);
static_assert(text_gdn_value_heads == qwen_text_model.gdn_value_heads);
static_assert(text_gdn_head_dim == qwen_text_model.gdn_head_dim);
static_assert(text_gdn_history_taps + 1 == qwen_text_model.gdn_conv_kernel_dim);
static_assert(text_attention_max_tokens == qwen_text_model.max_position_embeddings);
static_assert(text_fp8_group_size == qwen_text_model.fp8_weight_block_size);

struct Fp8Weight {
    const std::uint8_t* values = nullptr;
    const Bf16* scales = nullptr;
    int rows = 0;
    int columns = 0;
};

struct Bf16Weight {
    const Bf16* values = nullptr;
    int rows = 0;
    int columns = 0;
};

struct GdnLayer {
    Fp8Weight qkv;
    Fp8Weight z;
    Bf16Weight b;
    Bf16Weight a;
    TextGdnWeights recurrent;
    Fp8Weight output;
};

struct GqaLayer {
    Fp8Weight q_gate;
    Fp8Weight key;
    Fp8Weight value;
    const Bf16* query_norm;
    const Bf16* key_norm;
    Fp8Weight output;
};

struct Layer {
    const Bf16* input_norm;
    const Bf16* attention_norm;
    Fp8Weight mlp_gate;
    Fp8Weight mlp_up;
    Fp8Weight mlp_down;
    GdnLayer gdn;
    GqaLayer gqa;
};

struct LayerResidual {
    const Bf16* branch;
    Bf16* residual;
};

template <typename Normalize, typename Gqa, typename Gdn, typename Mlp>
void execute_layers_with_attention_input(const std::array<Layer, layer_count>& layers,
                                         Normalize normalize, Gqa gqa, Gdn gdn, Mlp mlp) {
    for (int index = 0; index < layer_count; ++index) {
        const auto& layer = layers[index];
        const auto input = [first = index == 0](const auto& work) {
            return first ? LayerResidual{work.residual, nullptr}
                         : LayerResidual{work.hidden, work.residual};
        };
        const auto [full_attention, bank] = qwen_text_model.layer_mapping(index);
        if (full_attention) {
            gqa(layer, bank, input);
        } else {
            gdn(layer, bank, input);
        }
        normalize(layer.attention_norm, "Normalize post-attention residual", [](const auto& work) {
            return LayerResidual{work.projected_attention, work.residual};
        });
        mlp(layer);
    }
}

template <typename Normalize, typename Gqa, typename Gdn, typename Mlp>
void execute_layers(const std::array<Layer, layer_count>& layers, Normalize normalize,
                    Gqa gqa, Gdn gdn, Mlp mlp) {
    execute_layers_with_attention_input(
        layers, normalize,
        [&](const Layer& layer, int bank, const auto& input) {
            normalize(layer.input_norm, "Normalize layer input", input);
            gqa(layer.gqa, bank);
        },
        [&](const Layer& layer, int bank, const auto& input) {
            normalize(layer.input_norm, "Normalize layer input", input);
            gdn(layer.gdn, bank);
        }, mlp);
}

}
