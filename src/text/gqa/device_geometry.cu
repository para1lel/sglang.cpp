#include "sglang/text/gqa/device_geometry.cuh"
#include "sglang/decoder_cuda.cuh"

#if defined(__CUDA_ARCH__) || !defined(SGLANG_GQA_DEVICE_ONLY)

namespace sglang {

SGLANG_GQA_HOST_DEVICE __inline_hint__ int gqa_query_heads(int kv_heads) {
    return mxfp8_queries_per_kv_head * kv_heads;
}

SGLANG_GQA_HOST_DEVICE __inline_hint__ std::size_t gqa_payload_bytes_per_page(int kv_heads) {
    return static_cast<std::size_t>(kv_heads) * mxfp8_page_size * mxfp8_head_dim;
}

SGLANG_GQA_HOST_DEVICE __inline_hint__ std::size_t gqa_scale_bytes_per_page(int kv_heads) {
    return static_cast<std::size_t>(kv_heads) * mxfp8_page_size * mxfp8_groups;
}

namespace decoder {

SGLANG_GQA_HOST_DEVICE __inline_hint__ bool target_split_position(int position, int end) {
    return position >= target_split_first_position && position < end;
}

SGLANG_GQA_HOST_DEVICE __inline_hint__ bool paged_decode_position(int position) {
    return position >= paged_decode_min_visible - 1 && position < paged_decode_max_visible;
}

}

namespace gqa_device {

__device__ __inline_hint__ std::size_t row_offset(std::size_t token, int heads, int head) {
    return (token * heads + head) * mxfp8_head_dim;
}

__device__ __inline_hint__ std::size_t payload_offset(int kv_heads, int physical_page,
                                                      int token, int head) {
    return static_cast<std::size_t>(physical_page) * gqa_payload_bytes_per_page(kv_heads)
         + row_offset(token, kv_heads, head);
}

__device__ __inline_hint__ std::size_t scale_offset(int kv_heads, int physical_page,
                                                    int token, int head, int group) {
    constexpr int token_groups = mxfp8_page_size / mxfp8_group_size;
    const auto page_head = static_cast<std::size_t>(physical_page) * kv_heads + head;
    const auto token_lane = page_head * mxfp8_group_size + token % mxfp8_group_size;
    return (token_lane * token_groups + token / mxfp8_group_size) * mxfp8_groups + group;
}

__device__ __inline_hint__ int position_value(int position) { return position; }
__device__ __inline_hint__ int position_value(const int* position) { return *position; }

}

}
#endif
