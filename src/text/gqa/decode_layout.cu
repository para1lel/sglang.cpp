#include "sglang/text/gqa/gqa_decode_long.cuh"

#if defined(__CUDA_ARCH__) || !defined(SGLANG_GQA_DEVICE_ONLY)

namespace sglang::gqa_decode {

template <int Stride>
SGLANG_GQA_HOST_DEVICE __inline_hint__ Partial* partials(void* scratch, int head) {
    return static_cast<Partial*>(scratch) + head * Stride;
}

SGLANG_GQA_HOST_DEVICE __inline_hint__ int split_count(int position) {
    const int chunks = position / split_chunk_tokens + 1;
    const int splits = (chunks + min_split_chunks - 1) / min_split_chunks;
    return splits < max_splits ? splits : max_splits;
}

SGLANG_GQA_HOST_DEVICE __inline_hint__ bool readable(const decoder::PagedDecodeLane& lane) {
    const int last_page = lane.position / mxfp8_page_size;
    return lane.position >= 0 && lane.position < text_attention_max_tokens &&
           last_page < lane.cache.page_count && last_page < lane.pages.page_count &&
           lane.pages.device_pages && lane.cache.key_payload && lane.cache.value_payload &&
           lane.cache.key_scales && lane.cache.value_scales;
}

SGLANG_GQA_HOST_DEVICE __inline_hint__ bool tensor_aligned(const decoder::PagedDecodeLane& lane) {
    const auto addresses = reinterpret_cast<std::uintptr_t>(lane.buffers.queries) |
        reinterpret_cast<std::uintptr_t>(lane.cache.key_payload) |
        reinterpret_cast<std::uintptr_t>(lane.cache.value_payload) |
        reinterpret_cast<std::uintptr_t>(lane.cache.key_scales) |
        reinterpret_cast<std::uintptr_t>(lane.cache.value_scales);
    return addresses % 16 == 0;
}

SGLANG_GQA_HOST_DEVICE __inline_hint__ decoder::PagedDecodeLane LongRequest::lane(int row) const {
    return {cache, pages, first + row, rows[row]};
}

__device__ __inline_hint__ decoder::PagedDecodeLane resolve(TargetLane lane) {
    const auto& bound = lane.bound;
    return bound.position ? decoder::PagedDecodeLane{*bound.cache, *bound.pages, *bound.position,
                                                      bound.buffers} : lane.eager;
}

__device__ __inline_hint__ decoder::PagedDecodeLane resolve(const TargetBatch& batch) {
    return resolve(TargetLane{{}, batch.lanes[blockIdx.z], batch.end});
}

__device__ __inline_hint__ decoder::PagedDecodeLane resolve(const EagerTargetBatch& batch) {
    return batch.lanes[blockIdx.z];
}

#ifdef SGLANG_GQA_DEVICE_ONLY
__device__ __inline_hint__ decoder::PagedDecodeLane resolve(const LongBatch& batch) {
    const auto& request = batch.requests[blockIdx.z / long_rows];
    const int row = blockIdx.z % long_rows;
    return row < request.count ? request.lane(row) : decoder::PagedDecodeLane{{}, {}, -1, {}};
}

__device__ __inline_hint__ decoder::PagedDecodeLane resolve(const BoundLongBatch& batch) {
    const auto& request = batch.requests[blockIdx.z / long_rows]->request;
    const auto row = static_cast<int>(blockIdx.z % long_rows);
    if (row < request.count)
        return request.lane(row);
    decoder::PagedDecodeLane inactive{};
    inactive.position = -1;
    return inactive;
}
    #endif

template SGLANG_GQA_HOST_DEVICE Partial* partials<max_splits>(void*, int);
template SGLANG_GQA_HOST_DEVICE Partial* partials<long_rank_wide_splits>(void*, int);
template SGLANG_GQA_HOST_DEVICE Partial* partials<long_single_rank_splits>(void*, int);
}
#endif
