#include "sglang/decoder_internal.h"

#include "sglang/decoder_prefill.h"
#include "sglang/runtime/model/qwen_weight_shard.h"
#include "sglang/runtime/model/qwen_weight_upload.h"
#include "sglang/text/mtp/tc_vocab.h"

#include <string>
#include <utility>

namespace sglang {
namespace {

using namespace decoder;

GdnLayer load_gdn(QwenWeightUpload& weights, const QwenTextModel& model,
                  const WeightSharding& shard, const std::string& prefix) {
    // The checkpoint stores Q, K and V in that channel order inside one
    // in_proj_qkv matrix; each rank packs its contiguous halves of the three
    // sections into one local qkv matrix, followed by its half of z.
    const int rank = shard.rank;
    const int qk_width = shard.local_gdn_qk_width();
    const int value_width = shard.local_gdn_value_width();
    const int value_heads = shard.local_gdn_value_heads();
    const auto bank = weights.packed_fp8(
        {
            Fp8PackSlice{prefix + "in_proj_qkv", text_gdn_qkv_width, rank * qk_width, qk_width},
            Fp8PackSlice{prefix + "in_proj_qkv", text_gdn_qkv_width,
                         text_gdn_qk_width + rank * qk_width, qk_width, true},
            Fp8PackSlice{prefix + "in_proj_qkv", text_gdn_qkv_width,
                         2 * text_gdn_qk_width + rank * value_width, value_width, true},
            Fp8PackSlice{prefix + "in_proj_z", text_gdn_value_width, rank * value_width,
                         value_width},
        },
        model.hidden_size);
    return {
        bank[0], bank[1],
        weights.bf16_matrix_rows(prefix + "in_proj_b.weight", text_gdn_value_heads,
                                 model.hidden_size, rank * value_heads, value_heads),
        weights.bf16_matrix_rows(prefix + "in_proj_a.weight", text_gdn_value_heads,
                                 model.hidden_size, rank * value_heads, value_heads),
        {
            weights.bf16_packed_rows(prefix + "conv1d.weight",
                                     {text_gdn_qkv_width, 1, text_gdn_history_taps + 1},
                                     {{rank * qk_width, qk_width},
                                      {text_gdn_qk_width + rank * qk_width, qk_width},
                                      {2 * text_gdn_qk_width + rank * value_width, value_width}}),
            weights.bf16_packed_rows(prefix + "A_log", {text_gdn_value_heads},
                                     {{rank * value_heads, value_heads}}),
            weights.bf16_packed_rows(prefix + "dt_bias", {text_gdn_value_heads},
                                     {{rank * value_heads, value_heads}}),
            weights.bf16(prefix + "norm.weight", {text_gdn_head_dim}),
        },
        weights.fp8_matrix_columns(prefix + "out_proj", model.hidden_size,
                                   text_gdn_value_width, rank * value_width, value_width),
    };
}

GqaLayer load_gqa(QwenWeightUpload& weights, const QwenTextModel& model,
                  const WeightSharding& shard, const std::string& prefix) {
    // q_proj interleaves q and gate per head, so the rank's contiguous block
    // of heads is one contiguous row range; K and V split by kv head.
    const int rank = shard.rank;
    const int q_gate_width = shard.local_q_gate_width();
    const int kv_width = shard.local_gqa_kv_width();
    const auto bank = weights.packed_fp8(
        {
            Fp8PackSlice{prefix + "q_proj", text_gqa_q_gate_width, rank * q_gate_width,
                         q_gate_width},
            Fp8PackSlice{prefix + "k_proj", text_gqa_kv_width, rank * kv_width, kv_width},
            Fp8PackSlice{prefix + "v_proj", text_gqa_kv_width, rank * kv_width, kv_width},
        },
        model.hidden_size);
    return {
        bank[0], bank[1], bank[2],
        weights.bf16(prefix + "q_norm.weight", {mxfp8_head_dim}),
        weights.bf16(prefix + "k_norm.weight", {mxfp8_head_dim}),
        weights.fp8_matrix_columns(prefix + "o_proj", model.hidden_size, text_gqa_q_width,
                                   rank * shard.local_q_width(), shard.local_q_width()),
    };
}

void load_layer(QwenWeightUpload& weights, const QwenTextModel& model,
                const WeightSharding& shard, Layer& layer, int index, const std::string& root) {
    const auto prefix = root + "layers." + std::to_string(index) + ".";
    layer.input_norm = weights.bf16(prefix + "input_layernorm.weight", {hidden_size});
    layer.attention_norm = weights.bf16(prefix + "post_attention_layernorm.weight", {hidden_size});
    if (model.is_full_attention(index)) {
        layer.gqa = load_gqa(weights, model, shard, prefix + "self_attn.");
    } else {
        layer.gdn = load_gdn(weights, model, shard, prefix + "linear_attn.");
    }
    const int rank = shard.rank;
    const int local_ffn = shard.local_intermediate_size();
    const auto bank = weights.packed_fp8(
        {
            Fp8PackSlice{prefix + "mlp.gate_proj", intermediate_size, rank * local_ffn,
                         local_ffn},
            Fp8PackSlice{prefix + "mlp.up_proj", intermediate_size, rank * local_ffn,
                         local_ffn},
        },
        model.hidden_size);
    layer.mlp_gate = bank[0];
    layer.mlp_up = bank[1];
    layer.mlp_down = weights.fp8_matrix_columns(prefix + "mlp.down_proj", model.hidden_size,
                                                model.intermediate_size, rank * local_ffn,
                                                local_ffn);
}

}

Decoder::Model::Model(const std::filesystem::path& directory, int cuda_device,
                      const decoder::WeightSharding& sharding)
    : device(cuda_device), memory(std::make_shared<Allocations>(cuda_device)) {
    sharding.validate();
    DeviceScope selected_device(device);
    auto selected_backend = read_cuda_backend_device(device);
    backend = selected_backend.policy;
    accelerator_name = std::move(selected_backend.name);
    execution = DecoderExecutionPolicy::for_backend(
        backend, {1, batch_prefill_tile_rows, prefill_chunk_rows});
    auto& checkpoint = memory->transfers().checkpoint;
    checkpoint = std::make_unique<Checkpoint>(directory);
    const auto& model = checkpoint->text_model();
    memory->open_stream();
    QwenWeightUpload weights(*checkpoint, *memory, memory->stream());
    const std::string root = "model.language_model.";
    const auto model_vocab = static_cast<std::uint64_t>(model.vocab_size);
    const auto model_hidden = static_cast<std::uint64_t>(model.hidden_size);
    // The embedding is replicated so every rank embeds its tokens locally; the
    // LM head stays whole only on the rank that scores logits.
    embedding = weights.bf16(root + "embed_tokens.weight", {model_vocab, model_hidden});
    final_norm = weights.bf16(root + "norm.weight", {model_hidden});
    if (sharding.primary_rank()) {
        lm_head = weights.bf16_matrix("lm_head.weight", model.vocab_size, model.hidden_size);
    }
    for (int index = 0; index < model.layer_count; ++index) {
        load_layer(weights, model, sharding, layers[index], index, root);
    }
    memory->close_stream();
    selected_device.restore();
}

}
