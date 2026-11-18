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

}
