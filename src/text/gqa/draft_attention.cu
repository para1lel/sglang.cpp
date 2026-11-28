#include "sglang/text/gqa/decode_kernels.cuh"
#include "sglang/runtime/execution/tp_verify_graph.h"
#include <algorithm>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <math_constants.h>

namespace sglang {

using namespace gqa_device;
using decoder::cache_covers;

namespace {

constexpr int draft_score_chunk = 4096;
constexpr int draft_replay_groups = mxfp8_head_dim / vector_replay_channels;
constexpr int draft_carry_stride = vector_replay_channels + 2;
constexpr int draft_carry_per_head = draft_replay_groups * draft_carry_stride;
static_assert(draft_score_chunk % mxfp8_page_size == 0);

__device__ bool draft_replay_weights(
    float score, int count, int lane, float& maximum, float2* weights) {
    const bool unit = isfinite(maximum) && __all_sync(warp_mask, score <= maximum);
    float prefix = fmaxf(maximum, score);
    if (unit) {
        if (lane < count) {
            weights[lane] = {1.0f, expf(score - prefix)};
        }
        return true;
    }
    #pragma unroll
    for (int offset = 1; offset < replay_tile; offset *= 2) {
        const float earlier = __shfl_up_sync(warp_mask, prefix, offset);
        if (lane >= offset) {
            prefix = fmaxf(earlier, prefix);
        }
    }
    const float previous = __shfl_up_sync(warp_mask, prefix, 1);
    if (lane < count) {
        weights[lane] = {expf((lane == 0 ? maximum : previous) - prefix), expf(score - prefix)};
    }
    maximum = __shfl_sync(warp_mask, prefix, replay_tile - 1);
    return false;
}

__device__ void draft_range_replay(
    Mxfp8KvCache cache, const std::int32_t* pages, int count, const float* scores,
    float* state, bool initial, bool last_range, __nv_bfloat16* output) {
    __shared__ alignas(16) float values[replay_tile][vector_replay_channels];
    __shared__ float2 weights[replay_tile];
    __shared__ float denominator;
    const int head = blockIdx.x;
    const int group = blockIdx.y;
    const int kv_head = head / queries_per_kv_head;
    const int thread = threadIdx.x;
    auto* carry = state + (head * draft_replay_groups + group) * draft_carry_stride;
    scores += head * draft_score_chunk;
    float maximum = initial ? -CUDART_INF_F : carry[vector_replay_channels];
    float mass = initial ? 0.0f : carry[vector_replay_channels + 1];
    float weighted = thread < vector_replay_channels && !initial ? carry[thread] : 0.0f;
    const bool aligned_values = replay_values_aligned(cache);
    for (int first = 0; first < count; first += replay_tile) {
        const int tile = min(replay_tile, count - first);
        const int page = pages[first / mxfp8_page_size];
        load_replay_values(cache, page, first % mxfp8_page_size, kv_head, group, tile,
                           aligned_values, values);
        bool unit = false;
        if (thread < vector_replay_channels) {
            const float score = thread < tile ? scores[first + thread] : -CUDART_INF_F;
            unit = draft_replay_weights(score, tile, thread, maximum, weights);
        }
        __syncthreads();
        if (thread < vector_replay_channels) {
            weighted = unit ? replay_vector_channel<true>(values, weights, tile, thread, weighted)
                            : replay_vector_channel<false>(values, weights, tile, thread, weighted);
        }
        if (thread == vector_replay_channels) {
            #pragma unroll 8
            for (int index = 0; index < tile; ++index) {
                const auto factor = weights[index];
                mass = fmaf(factor.x, mass, factor.y);
            }
        }
        __syncthreads();
    }
    if (thread == vector_replay_channels) {
        denominator = mass;
        if (!last_range) {
            carry[vector_replay_channels + 1] = mass;
        }
    }
    __syncthreads();
    if (thread < vector_replay_channels) {
        if (last_range) {
            output[row_offset(0, gqa_query_heads(cache.kv_heads), head) +
                   group * vector_replay_channels + thread] =
                __float2bfloat16_rn(weighted / denominator);
        } else {
            carry[thread] = weighted;
            if (thread == 0) {
                carry[vector_replay_channels] = maximum;
            }
        }
    }
}

__global__ void draft_range_replay_kernel(
    Mxfp8KvCache cache, const std::int32_t* pages, int count, const float* scores,
    float* state, bool initial, bool last_range, __nv_bfloat16* output) {
    draft_range_replay(cache, pages, count, scores, state, initial, last_range, output);
}

struct DraftAttentionBatch {
    decoder::DraftAttentionRow rows[decoder::draft_wave_rows];
};

__global__ void draft_range_scores_batch(
    const __grid_constant__ DraftAttentionBatch batch, int first) {
    const auto& row = batch.rows[blockIdx.z];
    const int count = min(draft_score_chunk, row.position + 1 - first);
    exact_scores(row.cache, row.pages.device_pages + first / mxfp8_page_size, count - 1,
                 row.queries, {row.workspace.scores, draft_score_chunk});
}

__global__ void draft_range_replay_batch(
    const __grid_constant__ DraftAttentionBatch batch, int first) {
    const auto& row = batch.rows[blockIdx.z];
    const int remaining = row.position + 1 - first;
    const int count = min(draft_score_chunk, remaining);
    float* scores = row.workspace.scores;
    draft_range_replay(row.cache, row.pages.device_pages + first / mxfp8_page_size, count,
                       scores, scores + gqa_query_heads(row.cache.kv_heads) * draft_score_chunk,
                       first == 0, count == remaining, row.output);
}

cudaError_t launch_long_draft_batch(
    const decoder::DraftAttentionRow* rows, int size, int visible, cudaStream_t stream) {
    for (int first = 0; first < visible; first += draft_score_chunk) {
        DraftAttentionBatch batch{};
        int active = 0;
        for (int index = 0; index < size; ++index) {
            if (rows[index].position >= first) {
                batch.rows[active++] = rows[index];
            }
        }
        const int count = std::min(draft_score_chunk, visible - first);
        const int query_heads = size ? gqa_query_heads(rows[0].cache.kv_heads) : 0;
        const int blocks = std::min(32, (count + score_warps - 1) / score_warps);
        void* arguments[] = {&batch, &first};
        auto status = cudaLaunchKernel(
            draft_range_scores_batch, dim3(blocks, query_heads, active),
            score_warps * mxfp8_group_size, arguments, 0, stream);
        if (status != cudaSuccess) {
            return status;
        }
        status = cudaLaunchKernel(
            draft_range_replay_batch, dim3(query_heads, draft_replay_groups, active),
            vector_replay_threads, arguments, 0, stream);
        if (status != cudaSuccess) {
            return status;
        }
    }
    return cudaSuccess;
}

cudaError_t launch_chunked_ordered_attention(Mxfp8KvCache cache, const std::int32_t* pages,
                                             int visible, const __nv_bfloat16* queries,
                                             __nv_bfloat16* output,
                                             decoder::ExactGqaWorkspace workspace,
                                             cudaStream_t stream) {
    auto* carry = workspace.scores + gqa_query_heads(cache.kv_heads) * draft_score_chunk;
    workspace.capacity = draft_score_chunk;
    for (int first = 0; first < visible; first += draft_score_chunk) {
        const auto* range_pages = pages + first / mxfp8_page_size;
        const int count = std::min(draft_score_chunk, visible - first);
        auto status = launch_draft_range_scores(cache, range_pages, count, queries, workspace,
                                                stream);
        if (status != cudaSuccess) {
            return status;
        }
        status = launch_gqa_kernel(
            draft_range_replay_kernel, dim3(gqa_query_heads(cache.kv_heads), draft_replay_groups),
            vector_replay_threads, stream, cache, range_pages, count, workspace.scores, carry,
            first == 0, count == visible - first, output);
        if (status != cudaSuccess) {
            return status;
        }
    }
    return cudaSuccess;
}

__global__ void recompute_replay_kernel(
    Mxfp8KvCache cache, const std::int32_t* pages, int position, const __nv_bfloat16* queries,
    __nv_bfloat16* output) {
    ordered_replay<true>(cache, pages, position, nullptr, queries, output);
}

using decoder::DraftAttentionRow;
using decoder::PagedDecodeLane;

bool valid_draft_attention(const DraftAttentionRow& row, std::int64_t visible) {
    return row.queries && row.output && cache_covers(row.cache, row.pages, visible) &&
           gqa_geometry_matches(gqa_full_geometry, row.cache);
}

bool draft_split_workspace(const DraftAttentionRow& row, std::int64_t visible) {
    static_assert(sizeof(float) * mxfp8_query_heads * (draft_score_chunk + draft_carry_per_head)
                  >= sizeof(gqa_decode::Partial) * gqa_decode::max_splits * mxfp8_query_heads);
    return row.workspace.scores && visible > decoder::target_split_first_position &&
           visible <= text_attention_max_tokens &&
           row.workspace.capacity >= draft_score_chunk + draft_carry_per_head;
}

std::optional<PagedDecodeLane> bind_online_draft(const DraftAttentionRow& row) {
    const PagedDecodeLane lane{row.cache, row.pages, row.position,
                              {row.queries, row.output, row.workspace.scores}};
    if (!gqa_decode::valid_buffers(lane.buffers) || !gqa_decode::tensor_aligned(lane)) {
        return std::nullopt;
    }
    return lane;
}

std::optional<cudaError_t> try_grouped_draft_attention(const DraftAttentionRow* rows,
                                                       cudaStream_t stream) {
    constexpr int count = decoder::draft_attention_batch_rows;
    static_assert(count <= decoder::target_split_eager_max_rows);
    static_assert(count % decoder::draft_wave_rows == 0);
    gqa_decode::EagerTargetBatch batch{{}, text_attention_max_tokens};
    for (int index = 0; index < count; ++index) {
        const auto& row = rows[index];
        const auto visible = static_cast<std::int64_t>(row.position) + 1;
        if (row.position < gqa_decode::long_min_position || !valid_draft_attention(row, visible) ||
            !draft_split_workspace(row, visible)) return std::nullopt;
        const auto lane = bind_online_draft(row);
        if (!lane) return std::nullopt;
        batch.lanes[index] = *lane;
    }
    cudaStreamCaptureStatus capture;
    const auto status = cudaStreamIsCapturing(stream, &capture);
    if (status != cudaSuccess) return status;
    if (capture != cudaStreamCaptureStatusNone) return std::nullopt;
    const auto split = gqa_decode::launch_multistage(batch, count, stream);
    return split == cudaSuccess ? combine_target_batch(batch, count, stream) : split;
}

}

std::optional<cudaError_t>
decoder::try_chunked_ordered_gqa(Mxfp8KvCache cache, Mxfp8PageTable pages, int position,
                                 const __nv_bfloat16* queries, __nv_bfloat16* output,
                                 ExactGqaWorkspace workspace, cudaStream_t stream) {
    const auto visible = static_cast<std::int64_t>(position) + 1;
    if (visible <= workspace.capacity || visible > text_attention_max_tokens ||
        workspace.scores == nullptr ||
        workspace.capacity < draft_score_chunk + draft_carry_per_head) {
        return std::nullopt;
    }
    if (queries == nullptr || output == nullptr || !cache_covers(cache, pages, visible)) {
        return cudaErrorInvalidValue;
    }
    return launch_chunked_ordered_attention(cache, pages.device_pages, static_cast<int>(visible),
                                            queries, output, workspace, stream);
}

cudaError_t decoder::mtp_gqa_attention(Mxfp8KvCache cache, Mxfp8PageTable pages, int position,
                                      const __nv_bfloat16* queries, __nv_bfloat16* output,
                                      ExactGqaWorkspace workspace, cudaStream_t stream,
                                      GqaGeometry geometry) {
    constexpr int min_tiled_tokens = 8;
    constexpr int max_tiled_tokens = 2 * mxfp8_page_size;
    const auto visible = static_cast<std::int64_t>(position) + 1;
    if (!queries || !output || !cache_covers(cache, pages, visible)
        || !gqa_geometry_matches(geometry, cache)) {
        return cudaErrorInvalidValue;
    }
    if (visible > max_tiled_tokens && workspace.scores && visible <= workspace.capacity) {
        workspace.capacity = static_cast<int>(visible);
        return launch_draft_vector_attention(
            cache, pages.device_pages, position, queries, output, workspace, stream);
    }
    if (const auto status =
            try_chunked_ordered_gqa(cache, pages, position, queries, output, workspace, stream)) {
        return *status;
    }
    if (visible < min_tiled_tokens || visible > max_tiled_tokens) {
        return mxfp8_gqa_attention(cache, pages, static_cast<int>(visible), position, 1,
                                  queries, output, stream, geometry);
    }
    return launch_gqa_kernel(recompute_replay_kernel,
        dim3(gqa_query_heads(cache.kv_heads), mxfp8_head_dim / replay_channels),
        replay_threads, stream, cache, pages.device_pages, position, queries, output);
}

cudaError_t decoder::mtp_gqa_attention_batch(const DraftAttentionRow* rows, int count, bool b200,
                                             cudaStream_t stream) {
    if (rows && count == draft_attention_batch_rows) {
        if (b200) {
            if (const auto status = try_grouped_draft_attention(rows, stream)) return *status;
        }
        for (int first = 0; first < count; first += draft_wave_rows) {
            const auto status = mtp_gqa_attention_batch(rows + first, draft_wave_rows, b200,
                                                        stream);
            if (status != cudaSuccess) return status;
        }
        return cudaSuccess;
    }
    if (!rows || count <= 0 || count > draft_wave_rows) {
        return cudaErrorInvalidValue;
    }
    DraftAttentionBatch long_rows{};
    int long_count = 0;
    int longest = 0;
    PagedDecodeLane online_lanes[draft_wave_rows];
    int online_count = 0;
    for (int index = 0; index < count; ++index) {
        const auto& row = rows[index];
        const auto visible = static_cast<std::int64_t>(row.position) + 1;
        if (!valid_draft_attention(row, visible)) {
            return cudaErrorInvalidValue;
        }
        const bool long_row = visible > row.workspace.capacity;
        if (draft_split_workspace(row, visible)) {
            if (b200) {
                if (const auto lane = bind_online_draft(row)) {
                    online_lanes[online_count++] = *lane;
                    continue;
                }
            }
            if (long_row) {
                long_rows.rows[long_count++] = row;
                longest = std::max(longest, static_cast<int>(visible));
                continue;
            }
        }
        const auto status = mtp_gqa_attention(row.cache, row.pages, row.position, row.queries,
                                              row.output, row.workspace, stream);
        if (status != cudaSuccess) {
            return status;
        }
    }
    if (online_count > 0) {
        const auto status = target_split_gqa(online_lanes, online_count, text_attention_max_tokens,
                                             stream);
        if (status != cudaSuccess) {
            return status;
        }
    }
    return launch_long_draft_batch(long_rows.rows, long_count, longest, stream);
}

namespace {

namespace gqa_verify {
__device__ decoder::PagedDecodeLane verify_lane(const gqa_decode::EagerTargetBatch& input) {
    return input.lanes[blockIdx.z];
}

__device__ decoder::ExactGqaWorkspace verify_scores(const gqa_decode::EagerTargetBatch& input,
                                                    int lane) {
    return {static_cast<float*>(input.lanes[lane].buffers.scratch), input.position_bound};
}
}

__global__ void verify_scores_kernel(const __grid_constant__ gqa_decode::EagerTargetBatch input) {
    const auto lane = gqa_verify::verify_lane(input);
    const int head = static_cast<int>(blockIdx.y);
    if (decoder::target_split_position(lane.position, input.end) || !gqa_decode::readable(lane) ||
        head >= gqa_query_heads(lane.cache.kv_heads) ||
        gqa_query_heads(lane.cache.kv_heads) > static_cast<int>(gridDim.y) ||
        lane.position > input.position_bound) {
        return;
    }
    exact_scores(lane.cache, lane.pages.device_pages, lane.position, lane.buffers.queries,
                 gqa_verify::verify_scores(input, static_cast<int>(blockIdx.z)));
}

__global__ void verify_replay_kernel(const __grid_constant__ gqa_decode::EagerTargetBatch input) {
    const auto lane = gqa_verify::verify_lane(input);
    const int head = static_cast<int>(blockIdx.x);
    if (decoder::target_split_position(lane.position, input.end) || !gqa_decode::readable(lane) ||
        head >= gqa_query_heads(lane.cache.kv_heads) ||
        gqa_query_heads(lane.cache.kv_heads) > static_cast<int>(gridDim.x) ||
        lane.position > input.position_bound) {
        return;
    }
    vector_replay_body(lane.cache, lane.pages.device_pages, lane.position,
                       gqa_verify::verify_scores(input, static_cast<int>(blockIdx.z)),
                       lane.buffers.output, head, static_cast<int>(blockIdx.y));
}

__global__ void live_scores(const decoder::tp_verify_graph::GqaBinding* binding) {
    const auto& lane = binding->reads[blockIdx.z];
    constexpr int capacity = decoder::tp_verify_graph::score_capacity;
    exact_scores(lane.cache, lane.pages.device_pages, lane.position, lane.buffers.queries,
                 {static_cast<float*>(lane.buffers.scratch), capacity});
}

__global__ void live_replay(const decoder::tp_verify_graph::GqaBinding* binding) {
    const auto& lane = binding->reads[blockIdx.z];
    constexpr int capacity = decoder::tp_verify_graph::score_capacity;
    vector_replay_body(lane.cache, lane.pages.device_pages, lane.position,
                       {static_cast<float*>(lane.buffers.scratch), capacity},
                       lane.buffers.output, blockIdx.x, blockIdx.y);
}

}

bool decoder::short_verify_supported(const PagedDecodeLane* lanes, int count) {
    if (!lanes || count < 1 || count > target_split_eager_max_rows) {
        return false;
    }
    int position_bound = 0;
    int kv_heads = 0;
    for (int index = 0; index < count; ++index) {
        const auto& lane = lanes[index];
        if (lane.position >= target_split_first_position || !gqa_decode::readable(lane) ||
            !gqa_decode::valid_buffers(lane.buffers)) {
            return false;
        }
        position_bound = std::max<int>(position_bound, lane.position + 1);
        kv_heads = std::max<int>(kv_heads, lane.cache.kv_heads);
    }
    const auto score_bytes = static_cast<std::size_t>(position_bound)
        * gqa_query_heads(kv_heads) * sizeof(float);
    return score_bytes <= paged_decode_scratch_bytes;
}

cudaError_t decoder::short_verify_gqa(const PagedDecodeLane* lanes, int count,
                                      cudaStream_t stream) {
    if (!short_verify_supported(lanes, count)) {
        return cudaErrorInvalidValue;
    }
    int position_bound = 0;
    int kv_heads = 0;
    for (int index = 0; index < count; ++index) {
        position_bound = std::max<int>(position_bound, lanes[index].position + 1);
        kv_heads = std::max<int>(kv_heads, lanes[index].cache.kv_heads);
    }
    const GqaGeometry geometry{gqa_query_heads(kv_heads), kv_heads};
    gqa_decode::EagerTargetBatch batch{};
    for (int index = 0; index < count; ++index) {
        batch.lanes[index] = lanes[index];
    }
    batch.end = text_attention_max_tokens;
    batch.position_bound = position_bound;
    constexpr int score_threads = score_warps * mxfp8_group_size;
    void* args[] = {&batch};
    const int score_blocks = (position_bound + score_warps) / score_warps;
    const int blocks = 32 < score_blocks ? 32 : score_blocks;
    auto status = cudaLaunchKernel(verify_scores_kernel,
        dim3(blocks, geometry.query_heads, count), score_threads, args, 0, stream);
    if (status == cudaSuccess) {
        status = cudaLaunchKernel(verify_replay_kernel,
            dim3(geometry.query_heads, mxfp8_head_dim / vector_replay_channels, count),
            vector_replay_threads, args, 0, stream);
    }
    return status;
}

cudaError_t decoder::tp_verify_graph::read(const GqaBinding* live, int query_heads,
                                          cudaStream_t stream) {
    static_assert(score_capacity % score_warps == 0);
    constexpr int score_threads = score_warps * mxfp8_group_size;
    auto status = launch_gqa_kernel(live_scores,
        dim3(score_capacity / score_warps, query_heads, rows), dim3(score_threads), stream, live);
    if (status == cudaSuccess) {
        status = launch_gqa_kernel(live_replay,
            dim3(query_heads, mxfp8_head_dim / vector_replay_channels, rows),
            dim3(vector_replay_threads), stream, live);
    }
    return status;
}

}
