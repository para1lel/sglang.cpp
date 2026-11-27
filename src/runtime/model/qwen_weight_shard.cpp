#include "sglang/runtime/model/qwen_weight_shard.h"
#include <stdexcept>

namespace sglang::decoder {

void WeightSharding::validate() const {
    if (world_size != 1 && world_size != 2) {
        throw std::invalid_argument("Weight sharding supports world sizes 1 and 2");
    }
    if (rank < 0 || rank >= world_size) {
        throw std::invalid_argument("Weight sharding rank outside its world");
    }
    if (qwen_text_model.query_heads % world_size != 0 ||
        qwen_text_model.kv_heads % world_size != 0 ||
        qwen_text_model.gdn_key_heads % world_size != 0 ||
        qwen_text_model.gdn_value_heads % world_size != 0 ||
        qwen_text_model.intermediate_size % world_size != 0) {
        throw std::invalid_argument("Weight sharding leaves a remainder on a parallel axis");
    }
    if (local_q_gate_width() % text_fp8_group_size != 0 ||
        local_gqa_kv_width() % text_fp8_group_size != 0 ||
        local_gdn_qkv_width() % text_fp8_group_size != 0 ||
        local_gdn_value_width() % text_fp8_group_size != 0 ||
        local_intermediate_size() % text_fp8_group_size != 0) {
        throw std::invalid_argument("Weight sharding is not aligned to FP8 scale blocks");
    }
}

bool WeightSharding::single_rank() const { return world_size == 1; }

bool WeightSharding::primary_rank() const { return rank == 0; }

int WeightSharding::local_q_width() const { return local_query_heads() * mxfp8_head_dim; }

int WeightSharding::local_query_heads() const { return qwen_text_model.query_heads / world_size; }

int WeightSharding::local_kv_heads() const { return qwen_text_model.kv_heads / world_size; }

int WeightSharding::local_gdn_key_heads() const {
    return qwen_text_model.gdn_key_heads / world_size;
}

int WeightSharding::local_gdn_value_heads() const {
    return qwen_text_model.gdn_value_heads / world_size;
}

int WeightSharding::local_intermediate_size() const {
    return qwen_text_model.intermediate_size / world_size;
}

int WeightSharding::local_q_gate_width() const {
    return 2 * local_query_heads() * mxfp8_head_dim;
}

int WeightSharding::local_gqa_kv_width() const { return local_kv_heads() * mxfp8_head_dim; }

int WeightSharding::local_gdn_qk_width() const {
    return local_gdn_key_heads() * text_gdn_head_dim;
}

int WeightSharding::local_gdn_value_width() const {
    return local_gdn_value_heads() * text_gdn_head_dim;
}

int WeightSharding::local_gdn_qkv_width() const {
    return 2 * local_gdn_qk_width() + local_gdn_value_width();
}

}
