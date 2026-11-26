#pragma once

// Rank-local GDN kernels for TP2. Every rank runs the full GDN recurrence on
// its own half of the key heads (8) and value heads (24); the packed local
// QKV row is [Q_r(1024), K_r(1024), V_r(3072)] sliced from the full
// [10240,5120] projection, and the local value-head range drives z, gates,
// out_proj, history and recurrent state slicing. All shapes derive from
// GdnShape below; the GdnFull instantiation reproduces the TP1 16/48 kernels.

#include "sglang/decoder_cuda.cuh"

namespace sglang {

/// Compile-time GDN geometry. qk_width is the width of each of the Q and K
/// planes; qkv_width is 2 * qk_width + value_width.
template <int QkHeads, int ValueHeads>
struct GdnShape {
    static constexpr int qk_heads = QkHeads;
    static constexpr int value_heads = ValueHeads;
    static constexpr int head_dim = text_gdn_head_dim;
    static constexpr int history_taps = text_gdn_history_taps;
    static constexpr int qk_width = qk_heads * head_dim;
    static constexpr int value_width = value_heads * head_dim;
    static constexpr int qkv_width = 2 * qk_width + value_width;
    static constexpr int heads_per_qk = value_heads / qk_heads;

    static_assert(qk_heads > 0 && value_heads > 0);
    static_assert(value_heads % qk_heads == 0);
    static_assert(qk_width % head_dim == 0 && value_width % head_dim == 0);
    // Finish kernels process 4 value heads per block; verify packs 8 heads.
    static_assert(value_heads % 4 == 0 && value_heads % 8 == 0);
};

/// Rank-local chunked-prefill scratch (gdn_chunk Matrices), 32-byte aligned:
/// ceil(rows/32) * 24 * sizeof(gdn_chunk::Matrices). Size grouped-prefill
/// scratch by the largest segment's rows.
std::size_t text_gdn_local_chunk_scratch_bytes(int rows);

using GdnFull = GdnShape<text_gdn_qk_heads, text_gdn_value_heads>;
// TP2 halves both head counts; three adjacent value heads still share one
// local key head, exactly as in the full shape.
using GdnLocal = GdnShape<text_gdn_qk_heads / 2, text_gdn_value_heads / 2>;
static_assert(GdnLocal::heads_per_qk == GdnFull::heads_per_qk);

/// Rank-local per-layer state sizes: BF16 history [qkv_width,3] per channel
/// plus FP32 recurrent [value_heads,128,128] indexed [head,value,key].
inline constexpr std::size_t text_gdn_local_history_bytes() {
    return sizeof(__nv_bfloat16) * static_cast<std::size_t>(GdnLocal::qkv_width) *
           GdnLocal::history_taps;
}
inline constexpr std::size_t text_gdn_local_recurrent_bytes() {
    return sizeof(float) * static_cast<std::size_t>(GdnLocal::value_heads) * GdnLocal::head_dim *
           GdnLocal::head_dim;
}
/// Rank-local prefill/decode workspace: rows * (qkv_width + value_width) BF16.
inline constexpr std::size_t text_gdn_local_workspace_bytes(int rows) {
    return rows > 0 && rows <= text_attention_max_tokens
               ? sizeof(__nv_bfloat16) * static_cast<std::size_t>(rows) *
                     (GdnLocal::qkv_width + GdnLocal::value_width)
               : 0;
}

// Rank-local twins of the GDN entry points. Pointer-only argument structs
// (TextGdnWeights, TextGdnState, TextGdnReplayRows, GdnPrefillSegment(s),
// GdnVerificationBatch, CompactGdnVerificationBatch, CompactGdnFp8Rows) are
// the TP1 types; every buffer behind them is sized for GdnLocal instead:
// qkv [tokens,5120], z [tokens,3072], b/a [tokens,24], gates float2
// [tokens,24], history [5120,3], recurrent [24,128,128], workspace
// text_gdn_local_workspace_bytes(tokens), output [tokens,3072]. The contracts
// documented on the TP1 entry points in sglang/text_attention.h and
// sglang/decoder_cuda.cuh (aliasing, ordering, finiteness, error handling)
// carry over unchanged.

cudaError_t text_gdn_forward_local(int token_count, TextGdnMode mode,
                                   const __nv_bfloat16* qkv, const __nv_bfloat16* z,
                                   const __nv_bfloat16* b, const __nv_bfloat16* a,
                                   TextGdnWeights weights, TextGdnState state,
                                   __nv_bfloat16* workspace, std::size_t workspace_bytes,
                                   __nv_bfloat16* output, cudaStream_t stream);
cudaError_t text_gdn_verify_tail_local(int token_count, TextGdnReplayRows rows,
                                       const __nv_bfloat16* z, TextGdnWeights weights,
                                       TextGdnState initial, TextGdnState tail,
                                       __nv_bfloat16* observable, __nv_bfloat16* output,
                                       cudaStream_t stream);
cudaError_t text_gdn_replay_prefix_local(int token_count, TextGdnReplayRows rows,
                                         TextGdnWeights weights, TextGdnState initial,
                                         TextGdnState result, cudaStream_t stream);

namespace decoder {

std::optional<cudaError_t> try_prefill_bf16_gdn_gates_sm100_local(
    const __nv_bfloat16* weight_b, const __nv_bfloat16* weight_a, const __nv_bfloat16* input,
    int rows, __nv_bfloat16* output_b, __nv_bfloat16* output_a, void* scratch,
    std::size_t scratch_bytes, cudaStream_t stream);

cudaError_t tensor_core_prefill_gdn_local(int rows, const __nv_bfloat16* qkv,
                                          const __nv_bfloat16* z, const __nv_bfloat16* b,
                                          const __nv_bfloat16* a, TextGdnWeights weights,
                                          TextGdnState state, __nv_bfloat16* workspace,
                                          float2* gates, __nv_bfloat16* output,
                                          cudaStream_t stream, void* chunk_scratch = nullptr,
                                          std::size_t chunk_bytes = 0);
cudaError_t tensor_core_prefill_gdn_local(int rows, const __nv_bfloat16* qkv,
                                          const __nv_bfloat16* z, const __nv_bfloat16* b,
                                          const __nv_bfloat16* a, TextGdnWeights weights,
                                          TextGdnState state, __nv_bfloat16* workspace,
                                          float2* gates, TextFp8Activation output,
                                          cudaStream_t stream, void* chunk_scratch = nullptr,
                                          std::size_t chunk_bytes = 0);

std::optional<cudaError_t> tensor_core_prefill_gdn_group_local(
    const GdnPrefillSegment* segments, int count, TextGdnWeights weights, cudaStream_t stream);

cudaError_t decode_gdn_local(const __nv_bfloat16* qkv, const __nv_bfloat16* z,
                             const __nv_bfloat16* b, const __nv_bfloat16* a,
                             TextGdnWeights weights, const TextGdnState* state,
                             __nv_bfloat16* workspace, __nv_bfloat16* output,
                             cudaStream_t stream);

cudaError_t decode_verify_gdn_local(int rows, TextGdnReplayRows inputs,
                                    const __nv_bfloat16* z, TextGdnWeights weights,
                                    const TextGdnState* initial, const TextGdnState* tail,
                                    __nv_bfloat16* observable, __nv_bfloat16* output,
                                    cudaStream_t stream);

cudaError_t decode_verify_gdn_batch_local(GdnVerificationBatch batch, TextGdnWeights weights,
                                          cudaStream_t stream);

cudaError_t decode_verify_gdn_compact_local(CompactGdnVerificationBatch batch, int lanes,
                                            TextGdnWeights weights, CompactGdnFp8Rows output,
                                            cudaStream_t stream);

}  // namespace decoder

/// Packed multi-lane form of text_gdn_verify_tail_local for the TP2 eager
/// verify advance: one call per GDN layer replays every lane, replacing one
/// call per lane. Lanes carry per-lane pointers (replay rows, states,
/// observable and output are per-lane); all lanes replay the same
/// token_count rows. Additive TP2 path: the per-lane entries stay
/// authoritative for TP1 and for rounds the packed form declines.
constexpr int gdn_verify_tail_max_lanes = 16;

struct GdnVerifyTailLanes {
    int lane_count;
    decoder::GdnVerificationLane lanes[gdn_verify_tail_max_lanes];
};

cudaError_t text_gdn_verify_tail_local_lanes(int token_count, const GdnVerifyTailLanes& jobs,
                                             TextGdnWeights weights, cudaStream_t stream);

}  // namespace sglang
