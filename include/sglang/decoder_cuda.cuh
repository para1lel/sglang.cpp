#pragma once

#include "sglang/multimodal_prompt.h"
#include "sglang/text_attention.h"
#include "sglang/text_ops.h"

#include <array>
#include <optional>

namespace sglang::decoder {

struct Fp8Weight;

#ifdef __CUDACC__
__device__ inline int position_value(int position) { return position; }
__device__ inline int position_value(const int* position) { return *position; }
#endif

cudaError_t decode_embedding(const __nv_bfloat16* embedding, int vocab_size, int width,
                             const TextLogitCandidate* previous, __nv_bfloat16* output,
                             cudaStream_t stream);
cudaError_t set_decode_position(int* position, int value, cudaStream_t stream,
                                int* rotary = nullptr, int rotary_value = 0);
cudaError_t advance_decode_position(int* position, int* rotary, cudaStream_t stream);
cudaError_t decode_bf16_gdn_gates(const __nv_bfloat16* weight_b,
                                  const __nv_bfloat16* weight_a, const __nv_bfloat16* input,
                                  int rows, int width, __nv_bfloat16* output_b,
                                  __nv_bfloat16* output_a, cudaStream_t stream);
struct VerifyGdnGateRow {
    const __nv_bfloat16* input;
    __nv_bfloat16* output_b;
    __nv_bfloat16* output_a;
};

std::optional<cudaError_t> try_verify_bf16_gdn_gates(
    const __nv_bfloat16* weight_b, const __nv_bfloat16* weight_a,
    const VerifyGdnGateRow* rows, int count, cudaStream_t stream);
std::optional<cudaError_t> try_verify_bf16_gdn_gates_compact(
    const __nv_bfloat16* weight_b, const __nv_bfloat16* weight_a,
    const VerifyGdnGateRow* rows, int count, cudaStream_t stream);
cudaError_t decode_bf16_lm_head(const __nv_bfloat16* weight, const __nv_bfloat16* input,
                               int rows, int width, __nv_bfloat16* output, cudaStream_t stream);
cudaError_t quantize_draft_head_int8(const __nv_bfloat16* input, int rows,
                                    std::uint8_t* values, float* maxima, cudaStream_t stream);
cudaError_t decode_draft_head_int8(const std::uint8_t* weight, const float* weight_maxima,
                                  const __nv_bfloat16* input, std::uint8_t* input_values,
                                  float* input_maximum,
                                  __nv_bfloat16* output, cudaStream_t stream);
struct DraftInt8Projection {
    const __nv_bfloat16* input;
    std::uint8_t* values;
    float* maximum;
    __nv_bfloat16* output;
};

inline constexpr int draft_wave_rows = 4;
struct DraftArgmaxRow {
    const __nv_bfloat16* logits;
    TextLogitCandidate* scratch;
    TextLogitCandidate* winner;
};
struct DraftArgmaxRows {
    DraftArgmaxRow rows[draft_wave_rows];
};

cudaError_t draft_argmax_wave(DraftArgmaxRows rows, int count, int width, bool finite,
                              cudaStream_t stream);
cudaError_t decode_draft_head_int8_batch(const std::uint8_t* weight, const float* weight_maxima,
                                        const DraftInt8Projection* rows, int count,
                                        cudaStream_t stream);
using DecodeHeadProjections = std::array<TextBf16Projection, 4>;
cudaError_t decode_bf16_lm_head_batch(const __nv_bfloat16* weight, int rows, int width,
                                     const DecodeHeadProjections& projections,
                                     cudaStream_t stream);
cudaError_t decode_quantize_fp8(const __nv_bfloat16* input, int width, TextFp8Activation output,
                                cudaStream_t stream);

struct VerifyQuantizeRow {
    const __nv_bfloat16* input;
    const __nv_bfloat16* up;
    TextFp8Activation output;
};

struct VerifyRmsQuantizeRow {
    const __nv_bfloat16* branch;
    __nv_bfloat16* residual;
    __nv_bfloat16* normalized;
    TextFp8Activation output;
};

inline constexpr int fused_verify_rows = 5;

struct VerifyQuantizeRows {
    VerifyQuantizeRow entries[fused_verify_rows];
};

struct WideVerifyQuantizeRows {
    static constexpr int capacity = 160;
    VerifyQuantizeRow entries[capacity];
};

struct VerifyRmsQuantizeRows {
    VerifyRmsQuantizeRow entries[fused_verify_rows];
};

struct WideVerifyRmsQuantizeRows {
    static constexpr int capacity = WideVerifyQuantizeRows::capacity;
    VerifyRmsQuantizeRow entries[capacity];
};

cudaError_t verify_quantize_fp8(VerifyQuantizeRows rows, int width, cudaStream_t stream);
cudaError_t verify_quantize_fp8_wide(WideVerifyQuantizeRows rows, int count, int width,
                                     cudaStream_t stream);
cudaError_t verify_rms_quantize(VerifyRmsQuantizeRows rows, const __nv_bfloat16* weight,
                                cudaStream_t stream);
cudaError_t verify_rms_quantize_wide(WideVerifyRmsQuantizeRows rows, int count,
                                     const __nv_bfloat16* weight, cudaStream_t stream);

std::optional<cudaError_t> try_prefill_bf16_gdn_gates(
    const __nv_bfloat16* weight_b, const __nv_bfloat16* weight_a, const __nv_bfloat16* input,
    int rows, __nv_bfloat16* output_b, __nv_bfloat16* output_a, cudaStream_t stream);

std::optional<cudaError_t> try_prefill_bf16_gdn_gates_sm100(
    const __nv_bfloat16* weight_b, const __nv_bfloat16* weight_a, const __nv_bfloat16* input,
    int rows, __nv_bfloat16* output_b, __nv_bfloat16* output_a, void* scratch,
    std::size_t scratch_bytes, cudaStream_t stream);

cudaError_t tensor_core_prefill_gdn(int rows, const __nv_bfloat16* qkv, const __nv_bfloat16* z,
                                    const __nv_bfloat16* b, const __nv_bfloat16* a,
                                    TextGdnWeights weights, TextGdnState state,
                                    __nv_bfloat16* workspace, float2* gates,
                                    __nv_bfloat16* output, cudaStream_t stream,
                                    void* chunk_scratch = nullptr, std::size_t chunk_bytes = 0);

cudaError_t tensor_core_prefill_gdn(int rows, const __nv_bfloat16* qkv, const __nv_bfloat16* z,
                                    const __nv_bfloat16* b, const __nv_bfloat16* a,
                                    TextGdnWeights weights, TextGdnState state,
                                    __nv_bfloat16* workspace, float2* gates,
                                    TextFp8Activation output, cudaStream_t stream,
                                    void* chunk_scratch = nullptr, std::size_t chunk_bytes = 0);

inline constexpr int max_gdn_prefill_segments = 4;

struct GdnPrefillSegment {
    int rows;
    const __nv_bfloat16* qkv;
    const __nv_bfloat16* z;
    const __nv_bfloat16* b;
    const __nv_bfloat16* a;
    TextGdnState state;
    __nv_bfloat16* workspace;
    float2* gates;
    TextFp8Activation output;
    void* chunk_scratch;
    std::size_t chunk_bytes;
};

std::optional<cudaError_t> tensor_core_prefill_gdn_group(
    const GdnPrefillSegment* segments, int count, TextGdnWeights weights, cudaStream_t stream);

cudaError_t tensor_core_prefill_gqa(Mxfp8KvCache cache, Mxfp8PageTable pages, int position,
                                    int rows, const __nv_bfloat16* queries,
                                    __nv_bfloat16* output, cudaStream_t stream,
                                    const int* device_position = nullptr);

std::optional<cudaError_t> try_prefill_gqa_postgate_quantize(
    int rows, const __nv_bfloat16* q_gate, const __nv_bfloat16* attention,
    TextFp8Activation activation, cudaStream_t stream);

enum class RmsQuantizeLaunch {
    ordinary,
    packed_gate_up,
    prefetch_decode_weights,
    full_group_prefill,
};

cudaError_t tensor_core_prefill_rms_quantize(
    const __nv_bfloat16* branch, __nv_bfloat16* residual, const __nv_bfloat16* weight,
    int width, int rows, __nv_bfloat16* normalized, TextFp8Activation activation,
    cudaStream_t stream, RmsQuantizeLaunch launch = RmsQuantizeLaunch::ordinary);
cudaError_t tensor_core_decode_rms_project(
    const __nv_bfloat16* branch, __nv_bfloat16* residual, const __nv_bfloat16* weight,
    int width, int rows, __nv_bfloat16* normalized, TextFp8Activation activation,
    const TextFp8DecodeTarget* targets, int count, cudaStream_t stream);
cudaError_t tensor_core_prefill_swiglu_quantize(
    const __nv_bfloat16* gate, const __nv_bfloat16* up, int width, int rows,
    TextFp8Activation activation, cudaStream_t stream);

cudaError_t decode_gqa_preprocess(const int* position, const __nv_bfloat16* q_gate,
                                  const __nv_bfloat16* keys, const __nv_bfloat16* q_norm,
                                  const __nv_bfloat16* k_norm, __nv_bfloat16* rope_queries,
                                  __nv_bfloat16* rope_keys, cudaStream_t stream);
cudaError_t mrope_gqa_preprocess(const MropePosition* positions, int rows,
                                const __nv_bfloat16* q_gate, const __nv_bfloat16* keys,
                                const __nv_bfloat16* q_norm, const __nv_bfloat16* k_norm,
                                __nv_bfloat16* rope_queries, __nv_bfloat16* rope_keys,
                                cudaStream_t stream);
cudaError_t prefill_gqa_queries(int position, const MropePosition* positions, int rows,
                                const __nv_bfloat16* q_gate, const __nv_bfloat16* norm,
                                __nv_bfloat16* queries, cudaStream_t stream);
cudaError_t mrope_gqa_preprocess_keys(const MropePosition* positions, int rows,
                                     const __nv_bfloat16* keys, const __nv_bfloat16* k_norm,
                                     __nv_bfloat16* rope_keys, cudaStream_t stream);
cudaError_t decode_kv_append(Mxfp8KvCache cache, Mxfp8PageTable pages, const int* position,
                             const __nv_bfloat16* keys, const __nv_bfloat16* values,
                             cudaStream_t stream);
cudaError_t decode_gqa_attention(Mxfp8KvCache cache, Mxfp8PageTable pages, const int* position,
                                 const __nv_bfloat16* queries, __nv_bfloat16* output,
                                 cudaStream_t stream);

struct ExactGqaWorkspace {
    float* scores;
    int capacity;
};

struct DraftAttentionRow {
    Mxfp8KvCache cache;
    Mxfp8PageTable pages;
    int position;
    const __nv_bfloat16* queries;
    __nv_bfloat16* output;
    ExactGqaWorkspace workspace;
};

cudaError_t mtp_gqa_attention_batch(const DraftAttentionRow* rows, int count, bool b200,
                                    cudaStream_t stream);

cudaError_t mtp_seed_project_kv(const Fp8Weight& key, const Fp8Weight& value,
                               TextFp8ActivationView input, int rows,
                               __nv_bfloat16* keys, __nv_bfloat16* values,
                               cudaStream_t stream);

cudaError_t exact_decode_gqa_attention(
    Mxfp8KvCache cache, Mxfp8PageTable pages, int position, const __nv_bfloat16* queries,
    __nv_bfloat16* output, ExactGqaWorkspace workspace, cudaStream_t stream,
    std::size_t workspace_bytes = 0, bool prefill = false);
cudaError_t exact_decode_gqa_attention(
    Mxfp8KvCache cache, Mxfp8PageTable pages, const int* position, const __nv_bfloat16* queries,
    __nv_bfloat16* output, ExactGqaWorkspace workspace, cudaStream_t stream,
    std::size_t workspace_bytes = 0);
cudaError_t decode_kv_append(const Mxfp8KvCache* cache, const Mxfp8PageTable* pages,
                             const int* position, const __nv_bfloat16* keys,
                             const __nv_bfloat16* values, cudaStream_t stream);
cudaError_t exact_decode_gqa_attention(
    const Mxfp8KvCache* cache, const Mxfp8PageTable* pages, const int* position,
    const __nv_bfloat16* queries, __nv_bfloat16* output, ExactGqaWorkspace workspace,
    cudaStream_t stream);
cudaError_t exact_decode_gqa_attention(
    const Mxfp8KvCache* cache, const Mxfp8PageTable* pages, const int* position,
    const __nv_bfloat16* queries, __nv_bfloat16* output, const ExactGqaWorkspace* workspace,
    cudaStream_t stream, std::size_t workspace_bytes = 0);
cudaError_t vector_decode_gqa_attention(
    Mxfp8KvCache cache, Mxfp8PageTable pages, int position, const __nv_bfloat16* queries,
    __nv_bfloat16* output, ExactGqaWorkspace workspace, cudaStream_t stream);
cudaError_t vector_decode_gqa_attention(
    const Mxfp8KvCache* cache, const Mxfp8PageTable* pages, const int* position,
    const __nv_bfloat16* queries, __nv_bfloat16* output, const ExactGqaWorkspace* workspace,
    cudaStream_t stream);

inline constexpr int paged_decode_lanes = 4;
inline constexpr int paged_decode_partitions = 16;
inline constexpr int paged_decode_min_visible = 8192;
inline constexpr int paged_decode_max_visible = 16384;
inline constexpr std::size_t paged_decode_scratch_bytes =
    paged_decode_partitions * mxfp8_query_heads * (mxfp8_head_dim + 4) * sizeof(float);

#ifdef __CUDACC__
__host__ __device__ inline constexpr bool paged_decode_position(int position) {
    return position >= paged_decode_min_visible - 1 && position < paged_decode_max_visible;
}
#else
inline constexpr bool paged_decode_position(int position) {
    return position >= paged_decode_min_visible - 1 && position < paged_decode_max_visible;
}
#endif

struct PagedDecodeBuffers {
    const __nv_bfloat16* queries;
    __nv_bfloat16* output;
    void* scratch;
};

struct PagedDecodeLane {
    Mxfp8KvCache cache;
    Mxfp8PageTable pages;
    int position;
    PagedDecodeBuffers buffers;
};

struct BoundPagedDecodeLane {
    const Mxfp8KvCache* cache;
    const Mxfp8PageTable* pages;
    const int* position;
    const ExactGqaWorkspace* scores;
    PagedDecodeBuffers buffers;
};

inline constexpr int target_split_first_position = 2047;
inline constexpr int target_split_end_position = 4096;
inline constexpr int target_split_eager_max_rows = 4 * text_gdn_max_verify_rows;

#ifdef __CUDACC__
__host__ __device__ inline constexpr bool target_split_position(
    int position, int end = target_split_end_position) {
    return position >= target_split_first_position && position < end;
}
#else
inline constexpr bool target_split_position(int position, int end = target_split_end_position) {
    return position >= target_split_first_position && position < end;
}
#endif

cudaError_t target_split_gqa(PagedDecodeLane lane, cudaStream_t stream,
                            int end = target_split_end_position);
cudaError_t target_split_gqa(BoundPagedDecodeLane lane, cudaStream_t stream,
                            int end = target_split_end_position);
cudaError_t target_split_gqa(const BoundPagedDecodeLane* lanes, int count, int end,
                            cudaStream_t stream);
cudaError_t target_split_gqa(const PagedDecodeLane* lanes, int count, int end,
                            cudaStream_t stream);

cudaError_t paged_decode_gqa(const PagedDecodeLane* lanes, int count, cudaStream_t stream);
cudaError_t paged_decode_gqa(const BoundPagedDecodeLane* lanes, int count, cudaStream_t stream);

cudaError_t decode_gdn(const __nv_bfloat16* qkv, const __nv_bfloat16* z,
                       const __nv_bfloat16* b, const __nv_bfloat16* a,
                       TextGdnWeights weights, const TextGdnState* state,
                       __nv_bfloat16* workspace, __nv_bfloat16* output, cudaStream_t stream);

cudaError_t decode_verify_gdn(int rows, TextGdnReplayRows inputs, const __nv_bfloat16* z,
                              TextGdnWeights weights, const TextGdnState* initial,
                              const TextGdnState* tail, __nv_bfloat16* observable,
                              __nv_bfloat16* output, cudaStream_t stream);

struct GdnVerificationLane {
    TextGdnReplayRows inputs;
    const __nv_bfloat16* z;
    TextGdnState initial;
    TextGdnState tail;
    __nv_bfloat16* observable;
    __nv_bfloat16* output;
};

struct GdnVerificationBatch {
    static constexpr int lanes = 4;
    static constexpr int rows = text_gdn_max_verify_rows;
    GdnVerificationLane entries[lanes];
};

cudaError_t decode_verify_gdn_batch(GdnVerificationBatch batch, TextGdnWeights weights,
                                    cudaStream_t stream);

struct CompactGdnVerificationBatch {
    static constexpr int lanes = 32;
    static constexpr int rows = GdnVerificationBatch::rows;
    GdnVerificationLane entries[lanes];
};

__host__ __device__ constexpr int compact_gdn_row(int lane, int row) {
    constexpr int packet_lanes = GdnVerificationBatch::lanes;
    constexpr int packet_rows = packet_lanes * GdnVerificationBatch::rows;
    return (lane / packet_lanes) * packet_rows + row * packet_lanes + lane % packet_lanes;
}

struct CompactGdnFp8Rows {
    std::uint8_t* values;
    float* scales[CompactGdnVerificationBatch::lanes * CompactGdnVerificationBatch::rows];
};

cudaError_t decode_verify_gdn_compact(CompactGdnVerificationBatch batch, int lanes,
                                      TextGdnWeights weights, CompactGdnFp8Rows output,
                                      cudaStream_t stream);

struct DecodeFp8Projection {
    const std::uint8_t* values;
    const __nv_bfloat16* scales;
    __nv_bfloat16* output;
    int rows;
};

// Read-only activation and weight spans may overlap. Output spans must be disjoint from every
// input span and from each other, including when they are views of one allocation. Invalid spans
// return cudaErrorInvalidValue before launch. Gate/up has the fused exception documented below.
cudaError_t decode_fp8_project(TextFp8ActivationView input, DecodeFp8Projection projection,
                               int width, cudaStream_t stream);
cudaError_t decode_fp8_qkv(TextFp8ActivationView input, DecodeFp8Projection query,
                           DecodeFp8Projection key, DecodeFp8Projection value, int width,
                           cudaStream_t stream);
// The fused 5120-by-17408 path requires 16-byte-aligned FP8 values. It writes only gate.output and
// accepts identical gate and up output spans. Fallback uses up.output as scratch and requires
// disjoint spans. Both paths return SwiGLU in gate.output.
cudaError_t decode_fp8_gate_up(TextFp8ActivationView input, DecodeFp8Projection gate,
                               DecodeFp8Projection up, int width, cudaStream_t stream);

}
