#pragma once

#include "sglang/multimodal_prompt.h"
#include "sglang/text_attention.h"
#include "sglang/text_ops.h"

namespace sglang::decoder {

#ifdef __CUDACC__
__device__ inline int position_value(int position) { return position; }
__device__ inline int position_value(const int* position) { return *position; }
#endif

cudaError_t decode_embedding(const __nv_bfloat16* embedding, int vocab_size, int width,
                             const TextLogitCandidate* previous, __nv_bfloat16* output,
                             cudaStream_t stream);
cudaError_t set_decode_position(int* position, int value, cudaStream_t stream);
cudaError_t advance_decode_position(int* position, cudaStream_t stream);
cudaError_t decode_bf16_gdn_gates(const __nv_bfloat16* weight_b,
                                  const __nv_bfloat16* weight_a, const __nv_bfloat16* input,
                                  int rows, int width, __nv_bfloat16* output_b,
                                  __nv_bfloat16* output_a, cudaStream_t stream);
cudaError_t decode_bf16_lm_head(const __nv_bfloat16* weight, const __nv_bfloat16* input,
                               int rows, int width, __nv_bfloat16* output, cudaStream_t stream);
cudaError_t decode_quantize_fp8(const __nv_bfloat16* input, int width, TextFp8Activation output,
                                cudaStream_t stream);

cudaError_t tensor_core_prefill_gdn(int rows, const __nv_bfloat16* qkv, const __nv_bfloat16* z,
                                    const __nv_bfloat16* b, const __nv_bfloat16* a,
                                    TextGdnWeights weights, TextGdnState state,
                                    __nv_bfloat16* workspace, float2* gates,
                                    __nv_bfloat16* output, cudaStream_t stream,
                                    void* chunk_scratch = nullptr, std::size_t chunk_bytes = 0);

cudaError_t tensor_core_prefill_gqa(Mxfp8KvCache cache, Mxfp8PageTable pages, int position,
                                    int rows, const __nv_bfloat16* queries,
                                    __nv_bfloat16* output, cudaStream_t stream);

cudaError_t tensor_core_prefill_rms_quantize(
    const __nv_bfloat16* branch, __nv_bfloat16* residual, const __nv_bfloat16* weight,
    int width, int rows, __nv_bfloat16* normalized, TextFp8Activation activation,
    cudaStream_t stream);
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

cudaError_t exact_decode_gqa_attention(
    Mxfp8KvCache cache, Mxfp8PageTable pages, int position, const __nv_bfloat16* queries,
    __nv_bfloat16* output, ExactGqaWorkspace workspace, cudaStream_t stream,
    std::size_t workspace_bytes = 0);
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
