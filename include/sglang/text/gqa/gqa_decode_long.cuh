#pragma once

#include "sglang/runtime/backend/cuda_backend_policy.h"
#include "sglang/text/gqa_decode_split.cuh"

#include <cuda.h>

namespace sglang::gqa_decode {

inline constexpr int long_rows = text_gdn_max_verify_rows;
inline constexpr int long_requests = decoder::max_wide_mtp_lanes;
inline constexpr int long_min_position = 32768;
inline constexpr int long_single_rank_splits = 32;
inline constexpr int long_rank_wide_requests = 2 * decoder::wide_mtp_lanes;
inline constexpr int long_rank_wide_splits = 9;
inline constexpr int long_rank_wide_end = long_min_position + 4096;
static_assert(sizeof(Partial) * long_rank_wide_splits * gqa_rank_geometry.query_heads <=
              decoder::paged_decode_scratch_bytes);
static_assert(sizeof(Partial) * long_single_rank_splits * gqa_rank_geometry.query_heads <=
              decoder::paged_decode_scratch_bytes);

struct LongRequest {
    Mxfp8KvCache cache;
    Mxfp8PageTable pages;
    decoder::PagedDecodeBuffers rows[long_rows];
    int first;
    int count = long_rows;

    SGLANG_GQA_HOST_DEVICE decoder::PagedDecodeLane lane(int row) const;
};

struct LongBatch {
    static constexpr int end = text_attention_max_tokens;
    LongRequest requests[long_requests];
};

struct alignas(128) LongTensorMaps {
    CUtensorMap payload[2];
    CUtensorMap scale[2];
};

struct LongGraphBinding {
    LongRequest request;
    LongTensorMaps maps{};
};

struct BoundLongBatch {
    static constexpr int end = LongBatch::end;
    const LongGraphBinding* requests[long_requests];
};

struct LongDecodeShape {
    int splits;
    bool remap_rows;

    bool operator==(const LongDecodeShape& other) const;
};

cudaError_t long_decode_status(cudaError_t status);
std::optional<cudaError_t> launch_long_decode(LongBatch& batch, int requests, cudaStream_t stream);
LongDecodeShape long_decode_shape(const LongBatch& batch, int requests);
std::optional<cudaError_t> prepare_long_decode_binding(LongGraphBinding& binding);
std::optional<cudaError_t> launch_long_decode(BoundLongBatch& batch, int requests, int kv_heads,
                                              LongDecodeShape shape, cudaStream_t stream);
cudaError_t combine_long_decode(const LongBatch& batch, int requests, int splits,
                                cudaStream_t stream);
cudaError_t combine_long_decode(const BoundLongBatch& batch, int requests, int kv_heads, int splits,
                                cudaStream_t stream);

#ifdef __CUDACC__
__device__ decoder::PagedDecodeLane resolve(const LongBatch& batch);
__device__ decoder::PagedDecodeLane resolve(const BoundLongBatch& batch);
#endif

}

namespace sglang::decoder {
cudaError_t target_split_gqa_graph(const BoundPagedDecodeLane* lanes, int count,
                                   gqa_decode::SplitVariant variant, int kv_heads,
                                   cudaStream_t stream);
}
