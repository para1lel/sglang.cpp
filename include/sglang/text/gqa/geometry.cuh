#pragma once

#include "sglang/runtime/cache/mxfp8_cache_geometry.h"

#include <cstddef>

#if defined(__CUDACC__) && defined(SGLANG_GQA_DEVICE_ONLY)
#define SGLANG_GQA_HOST_DEVICE __device__
#elif defined(__CUDACC__)
#define SGLANG_GQA_HOST_DEVICE __host__ __device__
#else
#define SGLANG_GQA_HOST_DEVICE
#endif

namespace sglang {

inline constexpr int mxfp8_queries_per_kv_head = mxfp8_query_heads / mxfp8_kv_heads;

SGLANG_GQA_HOST_DEVICE int gqa_query_heads(int kv_heads);
SGLANG_GQA_HOST_DEVICE std::size_t gqa_payload_bytes_per_page(int kv_heads);
SGLANG_GQA_HOST_DEVICE std::size_t gqa_scale_bytes_per_page(int kv_heads);

}
