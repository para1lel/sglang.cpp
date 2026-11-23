#pragma once

#include "sglang/runtime/backend/cuda_backend_policy.h"
#include "sglang/text/gqa_decode_split.cuh"

namespace sglang::gqa_decode {

inline constexpr int long_rows = text_gdn_max_verify_rows;
inline constexpr int long_requests = decoder::max_wide_mtp_lanes;
inline constexpr int long_min_position = 32768;

struct LongRequest {
    Mxfp8KvCache cache;
    Mxfp8PageTable pages;
    decoder::PagedDecodeBuffers rows[long_rows];
    int first;

    __host__ __device__ decoder::PagedDecodeLane lane(int row) const {
        return {cache, pages, first + row, rows[row]};
    }
};

struct LongBatch {
    static constexpr int end = text_attention_max_tokens;
    LongRequest requests[long_requests];
};

cudaError_t long_decode_status(cudaError_t status);
std::optional<cudaError_t> launch_long_decode(const LongBatch& batch, int requests,
                                             cudaStream_t stream);
cudaError_t combine_long_decode(const LongBatch& batch, int requests, cudaStream_t stream);

#ifdef __CUDACC__
__device__ inline decoder::PagedDecodeLane resolve(const LongBatch& batch) {
    return batch.requests[blockIdx.z / long_rows].lane(blockIdx.z % long_rows);
}
#endif

}
