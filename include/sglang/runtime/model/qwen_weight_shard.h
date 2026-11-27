#pragma once

#include "sglang/runtime/model/qwen_text_model.h"
#include "sglang/text_attention.h"
#include "sglang/text_ops.h"


namespace sglang::decoder {

struct WeightSharding {
    int rank = 0;
    int world_size = 1;

    void validate() const;

    bool single_rank() const;

    bool primary_rank() const;
    int local_q_width() const;

    int local_query_heads() const;
    int local_kv_heads() const;
    int local_gdn_key_heads() const;
    int local_gdn_value_heads() const;
    int local_intermediate_size() const;

    int local_q_gate_width() const;
    int local_gqa_kv_width() const;
    int local_gdn_qk_width() const;
    int local_gdn_value_width() const;
    int local_gdn_qkv_width() const;
};

}
