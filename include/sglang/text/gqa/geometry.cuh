#pragma once

#include "sglang/runtime/cache/mxfp8_cache_geometry.h"

#include <cstddef>

#include <cuda_runtime.h>

namespace sglang {

inline constexpr int mxfp8_queries_per_kv_head = mxfp8_query_heads / mxfp8_kv_heads;

__host__ __device__ constexpr int gqa_query_heads(int kv_heads) {
    return mxfp8_queries_per_kv_head * kv_heads;
}

__host__ __device__ constexpr std::size_t gqa_payload_bytes_per_page(int kv_heads) {
    return static_cast<std::size_t>(kv_heads) * mxfp8_page_size * mxfp8_head_dim;
}

__host__ __device__ constexpr std::size_t gqa_scale_bytes_per_page(int kv_heads) {
    return static_cast<std::size_t>(kv_heads) * mxfp8_page_size * mxfp8_groups;
}

}
