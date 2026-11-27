#pragma once

#include "sglang/decoder_cuda.cuh"

namespace sglang {

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
    static_assert(value_heads % 4 == 0 && value_heads % 8 == 0);
};

std::size_t text_gdn_local_chunk_scratch_bytes(int rows);

using GdnFull = GdnShape<text_gdn_qk_heads, text_gdn_value_heads>;
using GdnLocal = GdnShape<text_gdn_qk_heads / 2, text_gdn_value_heads / 2>;
static_assert(GdnLocal::heads_per_qk == GdnFull::heads_per_qk);

std::size_t text_gdn_local_history_bytes();
std::size_t text_gdn_local_recurrent_bytes();
std::size_t text_gdn_local_workspace_bytes(int rows);

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

}

constexpr int gdn_verify_tail_max_lanes = 16;

struct GdnVerifyTailLanes {
    int lane_count;
    decoder::GdnVerificationLane lanes[gdn_verify_tail_max_lanes];
};

cudaError_t text_gdn_verify_tail_local_lanes(int token_count, const GdnVerifyTailLanes& jobs,
                                             TextGdnWeights weights, cudaStream_t stream);

}
