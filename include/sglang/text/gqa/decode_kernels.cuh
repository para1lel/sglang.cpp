#pragma once

#include "sglang/text/gqa/device_geometry.cuh"
#include "sglang/text/gqa/gqa_decode_long.cuh"

#include <type_traits>

namespace sglang::gqa_device {

inline constexpr int score_warps = 8;
inline constexpr int replay_tile = 32;
inline constexpr int replay_channels = 2 * mxfp8_group_size;
inline constexpr int replay_threads = mxfp8_head_dim;
static_assert(mxfp8_page_size % replay_tile == 0 && replay_tile == mxfp8_group_size);
static_assert(replay_threads % replay_channels == 0 && mxfp8_head_dim % replay_channels == 0);

inline constexpr int vector_replay_threads = 4 * mxfp8_group_size;
inline constexpr int vector_replay_channels = mxfp8_group_size;
inline constexpr int queries_per_kv_head = mxfp8_queries_per_kv_head;
inline constexpr unsigned int warp_mask = 0xffffffffu;
inline constexpr float softmax_scale = 1.0f / 16.0f;

template <typename... Parameters>
cudaError_t launch_gqa_kernel(void (*kernel)(Parameters...), dim3 grid, dim3 block,
                              cudaStream_t stream,
                              std::enable_if_t<std::is_trivially_copyable_v<Parameters>,
                                               Parameters>... arguments);

__device__ float decode(std::uint8_t payload, std::uint8_t exponent);

__device__ float warp_sum(float value);

__device__ float decode_e8m0(std::uint8_t payload, std::uint8_t exponent);

__device__ float warp_max(float value);

__device__ void load_query(int query_heads, const __nv_bfloat16* queries, int head, int lane,
                           float (&query)[mxfp8_groups]);

__device__ float attention_score(Mxfp8KvCache cache, int physical_page, int token, int kv_head,
                                 int lane, const float (&query)[mxfp8_groups]);

__device__ void exact_scores(Mxfp8KvCache cache, const std::int32_t* pages, int last,
                              const __nv_bfloat16* queries,
                              decoder::ExactGqaWorkspace workspace);

__device__ bool replay_values_aligned(Mxfp8KvCache cache);

__device__ void load_replay_values(
    Mxfp8KvCache cache, int page, int token, int kv_head, int group, int count,
    bool aligned_values, float (&values)[replay_tile][vector_replay_channels]);

template <bool Unit>
__device__ float replay_vector_channel(
                                        const float (&values)[replay_tile][vector_replay_channels],
                                        const float2* weights, int count, int lane,
                                        float weighted);

__device__ __noinline__ void vector_replay_body(Mxfp8KvCache cache, const std::int32_t* pages,
                                                int last, decoder::ExactGqaWorkspace workspace,
                                                __nv_bfloat16* output, int head, int group);

template <bool RecomputeScores>
__device__ __noinline__ void ordered_replay(
    Mxfp8KvCache cache, const std::int32_t* pages, int last, const float* scores,
    const __nv_bfloat16* queries, __nv_bfloat16* output);

__device__ __noinline__ void wide_decode_head(decoder::PagedDecodeLane input, int head);

cudaError_t launch_target_fallback(decoder::BoundPagedDecodeLane lane, int end,
                                   cudaStream_t stream);

cudaError_t launch_draft_range_scores(Mxfp8KvCache cache, const std::int32_t* pages, int count,
                                      const __nv_bfloat16* queries,
                                      decoder::ExactGqaWorkspace workspace, cudaStream_t stream);

cudaError_t launch_draft_vector_attention(Mxfp8KvCache cache, const std::int32_t* pages,
                                          int position, const __nv_bfloat16* queries,
                                          __nv_bfloat16* output,
                                          decoder::ExactGqaWorkspace workspace,
                                          cudaStream_t stream);

cudaError_t combine_target_batch(const gqa_decode::EagerTargetBatch& batch, int count,
                                 cudaStream_t stream);

}
