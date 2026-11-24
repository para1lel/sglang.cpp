#include "sglang/runtime/model/qwen_checkpoint.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace sglang::decoder {
namespace {

using Path = std::filesystem::path;

[[noreturn]] void invalid(const Path& path, const std::string& reason) {
    throw std::runtime_error(path.string() + ": " + reason);
}

const json::Object& require_object(const json::Object& parent, const char* name, const Path& path) {
    const auto* value = json::get(parent, name);
    const auto* object = value ? value->object() : nullptr;
    if (!object) {
        invalid(path, std::string("expected a JSON object at ") + name);
    }
    return *object;
}

bool matches_config(const json::Value& actual, const json::Value& expected) {
    if (const auto* expected_array = expected.array()) {
        const auto* actual_array = actual.array();
        return actual_array && std::equal(actual_array->begin(), actual_array->end(),
                                         expected_array->begin(), expected_array->end(),
                                         matches_config);
    }
    if (const auto text = expected.string()) {
        return actual.string() == text;
    }
    if (const auto flag = expected.boolean()) {
        return actual.boolean() == flag;
    }
    const auto number = expected.number();
    return number && actual.number() == number;
}

void check_fields(const json::Object& actual,
                  std::initializer_list<std::pair<const char*, json::Value>> expected,
                  const Path& path, const std::string& prefix, const char* model = "text") {
    for (const auto& [name, expected_value] : expected) {
        const auto* value = json::get(actual, name);
        if (!value || !matches_config(*value, expected_value)) {
            invalid(path, prefix + name + " differs from the pinned " + model + " model");
        }
    }
}

}

const QwenTextModel& validate_qwen_text_config(const json::Object& config, const Path& path) {
    const auto& model = qwen_text_model;
    check_fields(config, {
        {"architectures", json::Array{std::string(model.architecture)}},
        {"model_type", std::string(model.model_type)},
        {"tie_word_embeddings", false},
    }, path, "");

    const auto& text = require_object(config, "text_config", path);
    json::Array layers;
    layers.reserve(model.layer_count);
    for (int layer = 0; layer < model.layer_count; ++layer) {
        layers.emplace_back(model.is_full_attention(layer) ? "full_attention" : "linear_attention");
    }
    check_fields(text, {
        {"model_type", std::string(model.text_model_type)},
        {"dtype", "bfloat16"},
        {"hidden_act", "silu"},
        {"hidden_size", model.hidden_size},
        {"intermediate_size", model.intermediate_size},
        {"vocab_size", model.vocab_size},
        {"tie_word_embeddings", false},
        {"num_hidden_layers", model.layer_count},
        {"full_attention_interval", model.full_attention_interval},
        {"layer_types", std::move(layers)},
        {"rms_norm_eps", 1e-6},
        {"num_attention_heads", model.query_heads},
        {"num_key_value_heads", model.kv_heads},
        {"head_dim", model.attention_head_dim},
        {"attention_bias", false},
        {"attention_dropout", 0.0},
        {"attn_output_gate", true},
        {"linear_conv_kernel_dim", model.gdn_conv_kernel_dim},
        {"linear_key_head_dim", model.gdn_head_dim},
        {"linear_value_head_dim", model.gdn_head_dim},
        {"linear_num_key_heads", model.gdn_key_heads},
        {"linear_num_value_heads", model.gdn_value_heads},
        {"output_gate_type", "swish"},
        {"mamba_ssm_dtype", "float32"},
        {"max_position_embeddings", model.max_position_embeddings},
        {"partial_rotary_factor", 0.25},
    }, path, "text_config.");

    const auto& rope = require_object(text, "rope_parameters", path);
    check_fields(rope, {
        {"rope_type", "default"},
        {"rope_theta", 10000000},
        {"partial_rotary_factor", 0.25},
        {"mrope_interleaved", true},
        {"mrope_section", json::Array{11, 11, 10}},
    }, path, "text_config.rope_parameters.");

    const auto& quantization = require_object(config, "quantization_config", path);
    check_fields(quantization, {
        {"quant_method", "fp8"},
        {"fmt", "e4m3"},
        {"activation_scheme", "dynamic"},
        {"weight_block_size", json::Array{model.fp8_weight_block_size,
                                         model.fp8_weight_block_size}},
    }, path, "quantization_config.");
    return model;
}

void validate_qwen_vision_config(const json::Object& config, const Path& path,
                                 const QwenTextModel& model) {
    const auto& vision = require_object(config, "vision_config", path);
    check_fields(vision, {
        {"model_type", std::string(model.model_type)},
        {"depth", 27},
        {"hidden_size", 1152},
        {"intermediate_size", 4304},
        {"out_hidden_size", model.hidden_size},
        {"num_heads", 16},
        {"num_position_embeddings", 2304},
        {"in_channels", 3},
        {"patch_size", 16},
        {"temporal_patch_size", 2},
        {"spatial_merge_size", 2},
        {"hidden_act", "gelu_pytorch_tanh"},
        {"deepstack_visual_indexes", json::Array{}},
    }, path, "vision_config.", "vision");
}

}
