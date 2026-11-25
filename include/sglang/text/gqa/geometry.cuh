#pragma once

// Rank-local GQA geometry helpers, valid on host and device. Included by
// sglang/mxfp8_gqa.h, so every translation unit that sees Mxfp8KvCache also
// sees these; device code may include this header directly.

#include "sglang/runtime/cache/mxfp8_cache_geometry.h"

#include <cstddef>

#include <cuda_runtime.h>

namespace sglang {

/// Queries per KV head. Invariant under rank-local tensor parallelism: the
/// checkpoint groups 6 adjacent query heads per KV head, and each TP2 rank
/// keeps its 12 query heads over its 2 KV heads with the same grouping.
inline constexpr int mxfp8_queries_per_kv_head = mxfp8_query_heads / mxfp8_kv_heads;

/// Query-head count for a cache with the given KV-head count.
__host__ __device__ constexpr int gqa_query_heads(int kv_heads) {
    return mxfp8_queries_per_kv_head * kv_heads;
}

/// Payload bytes per page for a cache with the given KV-head count.
__host__ __device__ constexpr std::size_t gqa_payload_bytes_per_page(int kv_heads) {
    return static_cast<std::size_t>(kv_heads) * mxfp8_page_size * mxfp8_head_dim;
}

/// Scale bytes per page for a cache with the given KV-head count.
__host__ __device__ constexpr std::size_t gqa_scale_bytes_per_page(int kv_heads) {
    return static_cast<std::size_t>(kv_heads) * mxfp8_page_size * mxfp8_groups;
}

}  // namespace sglang
