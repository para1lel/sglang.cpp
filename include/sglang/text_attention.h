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
    const __nv_bfloat16* convolution;  ///< [10240,4], oldest-to-current taps.
    const __nv_bfloat16* a_log;        ///< [48].
    const __nv_bfloat16* dt_bias;      ///< [48].
    const __nv_bfloat16* norm;         ///< [128], direct gated RMS weight.
};

struct TextGdnState {
    __nv_bfloat16* history;  ///< [10240,3], per-channel raw QKV, oldest first.
    float* recurrent;        ///< [48,128,128], indexed [head,value,key].
};

/// All arrays are contiguous device arrays. Keep inputs, outputs, state, and
/// workspace alive and ordered on the supplied stream through completion.
/// Arrays must be disjoint except for the exact postgate alias allowed below.
/// Inputs and intermediate arithmetic must be finite. Token counts are in
/// [0,text_attention_max_tokens]. Empty calls return success without touching arrays.
/// Functions enqueue work and return launch status. On error, observe stream
/// completion before reusing buffers. Earlier launches may have mutated state.

/// Returns the size of the BF16 workspace in bytes, or 0 for an invalid count.
constexpr std::size_t text_gdn_workspace_bytes(int token_count) {
    return token_count > 0 && token_count <= text_attention_max_tokens
               ? static_cast<std::size_t>(token_count)
                     * (text_gdn_qkv_width + text_gdn_value_width) * sizeof(__nv_bfloat16)
               : 0;
}

/// Runs one request in row order and advances its history and recurrent state.
/// Zero both state arrays before the first row of a fresh request.
/// qkv is BF16 [tokens,10240] with consecutive Q, K, V at widths 2048, 2048,
/// 6144. z and output are [tokens,6144]. b and a are each [tokens,48].
/// Prefill rounds normalized Q/K to BF16. Decode retains FP32 normalized Q/K.
/// output is ready for out_proj. workspace holds text_gdn_workspace_bytes()
/// bytes and is aligned for BF16.
cudaError_t text_gdn_forward(int token_count, TextGdnMode mode, const __nv_bfloat16* qkv,
                             const __nv_bfloat16* z, const __nv_bfloat16* b,
                             const __nv_bfloat16* a, TextGdnWeights weights,
                             TextGdnState state, __nv_bfloat16* workspace,
                             std::size_t workspace_bytes, __nv_bfloat16* output,
                             cudaStream_t stream);

/// q_gate is [tokens,24,2,256] with Q then gate within each head. keys and
/// rope_keys are [tokens,4,256]. rope_queries is [tokens,24,256].
/// Norms are [256] Gemma weights; the BF16 normalized Q/K receive text RoPE
/// at first_position + row. Pass rope_keys and the original V projection
/// directly to mxfp8_kv_append, then rope_queries to mxfp8_gqa_attention.
/// The position interval must fit the native text context. No scratch required.
cudaError_t text_gqa_preprocess(int first_position, int token_count,
                                const __nv_bfloat16* q_gate, const __nv_bfloat16* keys,
                                const __nv_bfloat16* q_norm, const __nv_bfloat16* k_norm,
                                __nv_bfloat16* rope_queries, __nv_bfloat16* rope_keys,
                                cudaStream_t stream);

/// Multiplies BF16 attention by sigmoid of the original q_gate gate in FP32.
/// attention and output are [tokens,24,256]. output may equal attention.
/// output is BF16 and ready for o_proj. No scratch required.
cudaError_t text_gqa_postgate(int token_count, const __nv_bfloat16* q_gate,
                              const __nv_bfloat16* attention, __nv_bfloat16* output,
                              cudaStream_t stream);

}
