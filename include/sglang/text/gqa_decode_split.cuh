#pragma once

#include "sglang/decoder_cuda.cuh"

#include "sglang/text/gqa/geometry.cuh"
#include "sglang/mxfp8_gqa.h"

#include <cstdint>
#include <optional>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace sglang::gqa_decode {

inline constexpr int max_splits = 16;
inline constexpr int split_chunk_tokens = 64;
inline constexpr int min_split_chunks = 4;

static_assert(mxfp8_page_size % split_chunk_tokens == 0);

struct alignas(16) Partial {
    float value[mxfp8_head_dim];
    float maximum;
    float mass;
};

static_assert(sizeof(Partial) * max_splits * mxfp8_query_heads <=
              decoder::paged_decode_scratch_bytes);

template <int Stride = max_splits>
SGLANG_GQA_HOST_DEVICE Partial* partials(void* scratch, int head);

SGLANG_GQA_HOST_DEVICE int split_count(int position);
SGLANG_GQA_HOST_DEVICE bool readable(const decoder::PagedDecodeLane& lane);
SGLANG_GQA_HOST_DEVICE bool tensor_aligned(const decoder::PagedDecodeLane& lane);

bool supports_eager_wide_split(int position);
bool valid_buffers(const decoder::PagedDecodeBuffers& buffers);

struct TargetLane {
    decoder::PagedDecodeLane eager;
    decoder::BoundPagedDecodeLane bound;
    int end = decoder::target_split_end_position;
};

struct TargetBatch {
    decoder::BoundPagedDecodeLane lanes[text_gdn_max_verify_rows];
    int end;
};

struct EagerTargetBatch {
    decoder::PagedDecodeLane lanes[decoder::target_split_eager_max_rows];
    int end;
    int position_bound = text_attention_max_tokens;
};

cudaError_t launch(TargetLane lane, cudaStream_t stream);
cudaError_t launch(const TargetBatch& batch, int count, cudaStream_t stream);
cudaError_t launch(const EagerTargetBatch& batch, int count, cudaStream_t stream);
cudaError_t launch_multistage(const EagerTargetBatch& batch, int count, cudaStream_t stream);
std::optional<cudaError_t> try_launch_reuse(const EagerTargetBatch& batch, int count,
                                           cudaStream_t stream);

#ifdef __CUDACC__
__device__ decoder::PagedDecodeLane resolve(TargetLane lane);
__device__ decoder::PagedDecodeLane resolve(const TargetBatch& batch);
__device__ decoder::PagedDecodeLane resolve(const EagerTargetBatch& batch);
#endif

}
