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

/// Rank-local GQA attention geometry for one target forward chain. TP1 (and
/// the rank0 MTP draft chain under TP2) uses the full 24Q/4KV checkpoint
/// geometry; each TP2 target rank uses half of it.
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

/// Separate byte arrays for K and V. Payload is [page, token, kv_heads, channel].
/// Scales are [page, kv_heads, token % 32, token / 32, channel / 32].
struct Mxfp8KvCache {
    std::uint8_t* key_payload;
    std::uint8_t* value_payload;
    std::uint8_t* key_scales;
    std::uint8_t* value_scales;
    int page_count;
    /// KV head count this allocation was built for. Defaults to the full
    /// geometry so TP1 aggregate initializations stay valid; rank-local TP2
    /// caches must set it to gqa_rank_geometry.kv_heads. Attention entry
    /// points reject a cache whose kv_heads disagrees with the requested
    /// geometry.
    int kv_heads = mxfp8_kv_heads;
};

/// Device array from logical page to physical page. Used entries must be in
/// [0, cache.page_count). Entries for the same request must be distinct.
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

/// All arrays reside on the stream's device. The caller owns capacity, page
/// mapping, valid prefix length, and ordering and lifetime through completion.
/// Read and write buffers must not overlap. BF16 inputs are finite. Attention
/// requires inputs for which dequantization and FP32 accumulation stay finite.
/// Returns launch status. The caller observes completion on the supplied stream.
/// A zero-length operation returns success without accessing device arrays.
/// Geometry overloads (trailing GqaGeometry) run the rank-local layout; the
/// default is the full checkpoint geometry, so TP1 callers are unchanged.

/// Writes [first_position, first_position + token_count). Each input is
/// contiguous [token_count, kv_heads, 256]. Apply RoPE to K before calling.
cudaError_t mxfp8_kv_append(Mxfp8KvCache cache, Mxfp8PageTable pages, int first_position,
                           int token_count, const __nv_bfloat16* keys,
                           const __nv_bfloat16* values, cudaStream_t stream,
                           GqaGeometry geometry = gqa_full_geometry);

/// Queries and output are contiguous [query_count, query_heads, 256]. Apply
/// RoPE to Q. Query row r attends through first_query_position + r, inclusive.
/// Cache positions [0, valid_tokens) must be initialized before the read.
cudaError_t mxfp8_gqa_attention(Mxfp8KvCache cache, Mxfp8PageTable pages, int valid_tokens,
                               int first_query_position, int query_count,
                               const __nv_bfloat16* queries, __nv_bfloat16* output,
                               cudaStream_t stream, GqaGeometry geometry = gqa_full_geometry);

namespace decoder {

struct ExactGqaWorkspace;
struct PagedDecodeLane;
struct BoundPagedDecodeLane;

/// Rank-local preprocess/postgate entries. These overload the full-geometry
/// declarations in decoder_cuda.cuh and text_attention.h; both headers include
/// this one, so the overloads are visible wherever the originals are. TP1
/// callers keep using the originals (full geometry). Every entry taking an
/// Mxfp8KvCache infers its geometry from cache.kv_heads instead.

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

/// Eager batched short-context verify for lanes whose positions sit
/// strictly below the split range: one exact-scores pass and one replay
/// pass over the whole batch, bit-identical to the per-lane vector read.
/// Callers must only dispatch when the baseline read would be the vector
/// read. Every lane's scratch buffer backs its score workspace, so the
/// combined score bytes must fit paged_decode_scratch_bytes — checked by
/// short_verify_supported, which is side-effect free so a caller can fall
/// back before any preprocess or cache append.
bool short_verify_supported(const PagedDecodeLane* lanes, int count);

cudaError_t short_verify_gqa(const PagedDecodeLane* lanes, int count, cudaStream_t stream);

}  // namespace decoder

/// Rank-local text-side preprocess/postgate overloads of the entries declared
/// in text_attention.h. Row strides follow the geometry: q_gate rows are
/// [2 * query_heads, 256], queries/attention rows [query_heads, 256], keys
/// [kv_heads, 256].
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

/// Tensor-core precision policy. Eligible tiles share decoded KV across adjacent rows.
/// Shared and single-row paths replay nonfinite FP32 scores by head and
/// nonfinite FP32 quotients by channel in FP64, using finite FP32 dequantization.
cudaError_t tensor_core_multirow_prefill_gqa(
    Mxfp8KvCache cache, Mxfp8PageTable pages, int position, int rows,
    const __nv_bfloat16* queries, __nv_bfloat16* output, cudaStream_t stream,
    GqaGeometry geometry = gqa_full_geometry);

/// MTP draft attention. The draft chain always runs the full checkpoint
/// geometry, even when the target chain is rank-local under TP2.
cudaError_t mtp_gqa_attention(Mxfp8KvCache cache, Mxfp8PageTable pages, int position,
                              const __nv_bfloat16* queries, __nv_bfloat16* output,
                              ExactGqaWorkspace workspace, cudaStream_t stream,
                              GqaGeometry geometry = gqa_full_geometry);

}

}