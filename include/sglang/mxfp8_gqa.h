#pragma once

#include "sglang/runtime/cache/mxfp8_cache_geometry.h"
#include "sglang/text/gqa/geometry.cuh"

#include <cstddef>
#include <cstdint>
#include <optional>

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

namespace sglang {

struct MropePosition;
struct TextFp8Activation;

struct GqaGeometry {
    int query_heads;
    int kv_heads;
};

inline constexpr GqaGeometry gqa_full_geometry{mxfp8_query_heads, mxfp8_kv_heads};
inline constexpr GqaGeometry gqa_rank_geometry{mxfp8_query_heads / 2, mxfp8_kv_heads / 2};

constexpr bool gqa_geometry_valid(GqaGeometry geometry) {
    return geometry.query_heads == mxfp8_queries_per_kv_head * geometry.kv_heads &&
           (geometry.kv_heads == mxfp8_kv_heads || geometry.kv_heads == mxfp8_kv_heads / 2);
}

struct Mxfp8KvCache {
    std::uint8_t* key_payload;
    std::uint8_t* value_payload;
    std::uint8_t* key_scales;
    std::uint8_t* value_scales;
    int page_count;
    int kv_heads = mxfp8_kv_heads;
};

struct Mxfp8PageTable {
    const std::int32_t* device_pages;
    int page_count;
};

constexpr bool gqa_geometry_matches(GqaGeometry geometry, const Mxfp8KvCache& cache) {
    return gqa_geometry_valid(geometry) && geometry.kv_heads == cache.kv_heads;
}

inline int gqa_head_pairs(const Mxfp8KvCache& cache) {
    return gqa_query_heads(cache.kv_heads) / 2;
}

cudaError_t mxfp8_kv_append(Mxfp8KvCache cache, Mxfp8PageTable pages, int first_position,
                           int token_count, const __nv_bfloat16* keys,
                           const __nv_bfloat16* values, cudaStream_t stream,
                           GqaGeometry geometry = gqa_full_geometry);

cudaError_t mxfp8_gqa_attention(Mxfp8KvCache cache, Mxfp8PageTable pages, int valid_tokens,
                               int first_query_position, int query_count,
                               const __nv_bfloat16* queries, __nv_bfloat16* output,
                               cudaStream_t stream, GqaGeometry geometry = gqa_full_geometry);

namespace decoder {

struct ExactGqaWorkspace;
struct PagedDecodeLane;
struct BoundPagedDecodeLane;

cudaError_t decode_gqa_preprocess(const int* position, const __nv_bfloat16* q_gate,
                                  const __nv_bfloat16* keys, const __nv_bfloat16* q_norm,
                                  const __nv_bfloat16* k_norm, __nv_bfloat16* rope_queries,
                                  __nv_bfloat16* rope_keys, cudaStream_t stream,
                                  GqaGeometry geometry);

cudaError_t mrope_gqa_preprocess(const MropePosition* positions, int rows,
                                 const __nv_bfloat16* q_gate, const __nv_bfloat16* keys,
                                 const __nv_bfloat16* q_norm, const __nv_bfloat16* k_norm,
                                 __nv_bfloat16* rope_queries, __nv_bfloat16* rope_keys,
                                 cudaStream_t stream, GqaGeometry geometry);

cudaError_t prefill_gqa_queries(int position, const MropePosition* positions, int rows,
                                const __nv_bfloat16* q_gate, const __nv_bfloat16* norm,
                                __nv_bfloat16* queries, cudaStream_t stream,
                                GqaGeometry geometry);

cudaError_t mrope_gqa_preprocess_keys(const MropePosition* positions, int rows,
                                      const __nv_bfloat16* keys, const __nv_bfloat16* k_norm,
                                      __nv_bfloat16* rope_keys, cudaStream_t stream,
                                      GqaGeometry geometry);

std::optional<cudaError_t> try_prefill_gqa_postgate_quantize(int rows,
                                                             const __nv_bfloat16* q_gate,
                                                             const __nv_bfloat16* attention,
                                                             TextFp8Activation activation,
                                                             cudaStream_t stream,
                                                             GqaGeometry geometry);

bool short_verify_supported(const PagedDecodeLane* lanes, int count);

cudaError_t short_verify_gqa(const PagedDecodeLane* lanes, int count, cudaStream_t stream);

}

cudaError_t text_gqa_preprocess(int first_position, int token_count,
                                const __nv_bfloat16* q_gate, const __nv_bfloat16* keys,
                                const __nv_bfloat16* q_norm, const __nv_bfloat16* k_norm,
                                __nv_bfloat16* rope_queries, __nv_bfloat16* rope_keys,
                                cudaStream_t stream, GqaGeometry geometry);

cudaError_t text_gqa_preprocess_keys(int first_position, int token_count,
                                     const __nv_bfloat16* keys, const __nv_bfloat16* k_norm,
                                     __nv_bfloat16* rope_keys, cudaStream_t stream,
                                     GqaGeometry geometry);

cudaError_t text_gqa_postgate(int token_count, const __nv_bfloat16* q_gate,
                              const __nv_bfloat16* attention, __nv_bfloat16* output,
                              cudaStream_t stream, GqaGeometry geometry);

namespace decoder {

cudaError_t tensor_core_multirow_prefill_gqa(
    Mxfp8KvCache cache, Mxfp8PageTable pages, int position, int rows,
    const __nv_bfloat16* queries, __nv_bfloat16* output, cudaStream_t stream,
    GqaGeometry geometry = gqa_full_geometry);

cudaError_t mtp_gqa_attention(Mxfp8KvCache cache, Mxfp8PageTable pages, int position,
                              const __nv_bfloat16* queries, __nv_bfloat16* output,
                              ExactGqaWorkspace workspace, cudaStream_t stream,
                              GqaGeometry geometry = gqa_full_geometry);

}

}