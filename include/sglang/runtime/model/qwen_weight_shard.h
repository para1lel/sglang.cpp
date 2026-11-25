#pragma once

#include "sglang/runtime/model/qwen_text_model.h"
#include "sglang/text_attention.h"
#include "sglang/text_ops.h"

#include <stdexcept>

namespace sglang::decoder {

/// Rank identity for native tensor-parallel weight upload.
///
/// world_size 1 keeps the historical single-rank layout: every helper below
/// returns the full checkpoint width. world_size 2 splits every parallel axis
/// of the text model in half; rank r owns slice [r*n,(r+1)*n) of each logical
/// axis of local length n (docs/research/tp2.md). The embedding is replicated
/// on both ranks; the LM head and the MTP draft belong to the primary rank.
struct WeightSharding {
    int rank = 0;
    int world_size = 1;

    constexpr WeightSharding() = default;
    constexpr WeightSharding(int rank_, int world_size_) : rank(rank_), world_size(world_size_) {}

    /// Throws unless the sharding is one this loader implements.
    void validate() const {
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

    /// True when every axis keeps its full checkpoint width.
    constexpr bool single_rank() const { return world_size == 1; }

    /// The rank that owns the LM head and the MTP draft weights.
    constexpr bool primary_rank() const { return rank == 0; }
    /// Width of the rank-local query projection input (o_proj column slice).
    constexpr int local_q_width() const { return local_query_heads() * mxfp8_head_dim; }

    constexpr int local_query_heads() const { return qwen_text_model.query_heads / world_size; }
    constexpr int local_kv_heads() const { return qwen_text_model.kv_heads / world_size; }
    constexpr int local_gdn_key_heads() const {
        return qwen_text_model.gdn_key_heads / world_size;
    }
    constexpr int local_gdn_value_heads() const {
        return qwen_text_model.gdn_value_heads / world_size;
    }
    constexpr int local_intermediate_size() const {
        return qwen_text_model.intermediate_size / world_size;
    }

    constexpr int local_q_gate_width() const {
        return 2 * local_query_heads() * mxfp8_head_dim;
    }
    constexpr int local_gqa_kv_width() const { return local_kv_heads() * mxfp8_head_dim; }
    constexpr int local_gdn_qk_width() const {
        return local_gdn_key_heads() * text_gdn_head_dim;
    }
    constexpr int local_gdn_value_width() const {
        return local_gdn_value_heads() * text_gdn_head_dim;
    }
    constexpr int local_gdn_qkv_width() const {
        return 2 * local_gdn_qk_width() + local_gdn_value_width();
    }
};

}  // namespace sglang::decoder
