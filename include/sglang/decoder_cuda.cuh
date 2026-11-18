#pragma once

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

cudaError_t decode_gqa_preprocess(const int* position, const __nv_bfloat16* q_gate,
                                  const __nv_bfloat16* keys, const __nv_bfloat16* q_norm,
                                  const __nv_bfloat16* k_norm, __nv_bfloat16* rope_queries,
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

cudaError_t exact_decode_gqa_attention(
    Mxfp8KvCache cache, Mxfp8PageTable pages, int position, const __nv_bfloat16* queries,
    __nv_bfloat16* output, ExactGqaWorkspace workspace, cudaStream_t stream);
cudaError_t exact_decode_gqa_attention(
    Mxfp8KvCache cache, Mxfp8PageTable pages, const int* position, const __nv_bfloat16* queries,
    __nv_bfloat16* output, ExactGqaWorkspace workspace, cudaStream_t stream);

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
