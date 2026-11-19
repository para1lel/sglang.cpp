#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

namespace sglang {

inline constexpr int mxfp8_query_heads = 24;
inline constexpr int mxfp8_kv_heads = 4;
inline constexpr int mxfp8_head_dim = 256;
inline constexpr int mxfp8_group_size = 32;
inline constexpr int mxfp8_groups = mxfp8_head_dim / mxfp8_group_size;
inline constexpr int mxfp8_page_size = 128;
inline constexpr std::size_t mxfp8_payload_bytes_per_page =
    mxfp8_page_size * mxfp8_kv_heads * mxfp8_head_dim;
inline constexpr std::size_t mxfp8_scale_bytes_per_page =
    mxfp8_page_size * mxfp8_kv_heads * mxfp8_groups;

/// Separate byte arrays for K and V. Payload is [page, token, head, channel].
/// Scales are [page, head, token % 32, token / 32, channel / 32].
struct Mxfp8KvCache {
    std::uint8_t* key_payload;
    std::uint8_t* value_payload;
    std::uint8_t* key_scales;
    std::uint8_t* value_scales;
    int page_count;
};

/// Device array from logical page to physical page. Used entries must be in
/// [0, cache.page_count). Entries for the same request must be distinct.
struct Mxfp8PageTable {
    const std::int32_t* device_pages;
    int page_count;
};

/// All arrays reside on the stream's device. The caller owns capacity, page
/// mapping, valid prefix length, and ordering and lifetime through completion.
/// Read and write buffers must not overlap. BF16 inputs are finite. Attention
/// requires inputs for which dequantization and FP32 accumulation stay finite.
/// Returns launch status. The caller observes completion on the supplied stream.
/// A zero-length operation returns success without accessing device arrays.

/// Writes [first_position, first_position + token_count). Each input is
/// contiguous [token_count, 4, 256]. Apply RoPE to K before calling.
cudaError_t mxfp8_kv_append(Mxfp8KvCache cache, Mxfp8PageTable pages, int first_position,
                           int token_count, const __nv_bfloat16* keys,
                           const __nv_bfloat16* values, cudaStream_t stream);

/// Queries and output are contiguous [query_count, 24, 256]. Apply RoPE to Q.
/// Query row r attends through first_query_position + r, inclusive.
/// Cache positions [0, valid_tokens) must be initialized before the read.
cudaError_t mxfp8_gqa_attention(Mxfp8KvCache cache, Mxfp8PageTable pages, int valid_tokens,
                               int first_query_position, int query_count,
                               const __nv_bfloat16* queries, __nv_bfloat16* output,
                               cudaStream_t stream);

namespace decoder {

/// Tensor-core precision policy. Adjacent rows share decoded KV in eligible chunks.
/// Both paired and single-row launches replay nonfinite FP32 scores by head and
/// nonfinite FP32 quotients by channel in FP64, using finite FP32 dequantization.
cudaError_t tensor_core_multirow_prefill_gqa(
    Mxfp8KvCache cache, Mxfp8PageTable pages, int position, int rows,
    const __nv_bfloat16* queries, __nv_bfloat16* output, cudaStream_t stream);

}

}
