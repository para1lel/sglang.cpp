#pragma once

#include <cstddef>

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

}
