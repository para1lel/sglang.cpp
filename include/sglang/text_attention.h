#pragma once

#include <cstddef>

#include "sglang/mxfp8_gqa.h"

namespace sglang {

inline constexpr int text_attention_max_tokens = 262144;
inline constexpr int text_gdn_qk_heads = 16;
inline constexpr int text_gdn_value_heads = 48;
inline constexpr int text_gdn_head_dim = 128;
inline constexpr int text_gdn_qk_width = text_gdn_qk_heads * text_gdn_head_dim;
inline constexpr int text_gdn_value_width = text_gdn_value_heads * text_gdn_head_dim;
inline constexpr int text_gdn_qkv_width = 2 * text_gdn_qk_width + text_gdn_value_width;
inline constexpr int text_gdn_history_taps = 3;
inline constexpr int text_gqa_q_width = mxfp8_query_heads * mxfp8_head_dim;
inline constexpr int text_gqa_q_gate_width = 2 * text_gqa_q_width;
inline constexpr int text_gqa_kv_width = mxfp8_kv_heads * mxfp8_head_dim;

enum class TextGdnMode { prefill, decode };

struct TextGdnWeights {
    const __nv_bfloat16* convolution;
    const __nv_bfloat16* a_log;
    const __nv_bfloat16* dt_bias;
    const __nv_bfloat16* norm;
};

struct TextGdnState {
    __nv_bfloat16* history;
    float* recurrent;
};

inline constexpr int text_gdn_max_verify_rows = 5;

struct TextGdnReplayRows {
    __nv_bfloat16* qkv;
    __nv_bfloat16* convolved;
    __nv_bfloat16* b;
    __nv_bfloat16* a;
};

std::size_t text_gdn_workspace_bytes(int token_count);

cudaError_t text_gdn_forward(int token_count, TextGdnMode mode, const __nv_bfloat16* qkv,
                             const __nv_bfloat16* z, const __nv_bfloat16* b,
                             const __nv_bfloat16* a, TextGdnWeights weights,
                             TextGdnState state, __nv_bfloat16* workspace,
                             std::size_t workspace_bytes, __nv_bfloat16* output,
                             cudaStream_t stream);

cudaError_t text_gdn_verify_tail(int token_count, TextGdnReplayRows rows,
                                const __nv_bfloat16* z, TextGdnWeights weights,
                                TextGdnState initial, TextGdnState tail,
                                __nv_bfloat16* observable, __nv_bfloat16* output,
                                cudaStream_t stream);
cudaError_t text_gdn_replay_prefix(int token_count, TextGdnReplayRows rows, TextGdnWeights weights,
                                  TextGdnState initial, TextGdnState result, cudaStream_t stream);

cudaError_t text_gqa_preprocess(int first_position, int token_count,
                                const __nv_bfloat16* q_gate, const __nv_bfloat16* keys,
                                const __nv_bfloat16* q_norm, const __nv_bfloat16* k_norm,
                                __nv_bfloat16* rope_queries, __nv_bfloat16* rope_keys,
                                cudaStream_t stream);

cudaError_t text_gqa_preprocess_keys(int first_position, int token_count,
                                    const __nv_bfloat16* keys, const __nv_bfloat16* k_norm,
                                    __nv_bfloat16* rope_keys, cudaStream_t stream);

cudaError_t text_gqa_postgate(int token_count, const __nv_bfloat16* q_gate,
                              const __nv_bfloat16* attention, __nv_bfloat16* output,
                              cudaStream_t stream);

}
