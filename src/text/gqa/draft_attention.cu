#include "sglang/text/gqa/decode_kernels.cuh"
#include "sglang/runtime/execution/tp_verify_graph.h"
#include <algorithm>
#include <array>
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
    const GqaGeometry geometry{gqa_query_heads(row.cache.kv_heads), row.cache.kv_heads};
    return row.queries && row.output && cache_covers(row.cache, row.pages, visible) &&
           gqa_geometry_valid(geometry);
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
        if (index != 0 && row.cache.kv_heads != rows[0].cache.kv_heads)
            return std::nullopt;
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
        if (row.cache.kv_heads != rows[0].cache.kv_heads)
            return cudaErrorInvalidValue;
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
        const GqaGeometry geometry{gqa_query_heads(row.cache.kv_heads), row.cache.kv_heads};
        const auto status = mtp_gqa_attention(row.cache, row.pages, row.position, row.queries,
                                              row.output, row.workspace, stream, geometry);
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

struct LiveVerifyBatch {
    static constexpr int end = text_attention_max_tokens;
    static constexpr int position_bound = decoder::tp_verify_graph::short_capacity;
    const decoder::tp_verify_graph::GqaBinding* binding;
    int indices[decoder::tp_verify_graph::max_rows];
};

__device__ const decoder::PagedDecodeLane& verify_lane(const gqa_decode::EagerTargetBatch& batch) {
    return batch.lanes[blockIdx.z];
}

__device__ const decoder::PagedDecodeLane& verify_lane(const LiveVerifyBatch& batch) {
    return batch.binding->reads[batch.indices[blockIdx.z]];
}

template <typename Batch>
__device__ bool verify_head(const Batch& input, const decoder::PagedDecodeLane& lane, int head,
                            int heads) {
    return !decoder::target_split_position(lane.position, input.end) &&
           gqa_decode::readable(lane) && head < gqa_query_heads(lane.cache.kv_heads) &&
           gqa_query_heads(lane.cache.kv_heads) <= heads && lane.position < input.position_bound;
}

template <typename Batch>
__global__ void verify_scores_kernel(const __grid_constant__ Batch input) {
    const auto& lane = verify_lane(input);
    if (verify_head(input, lane, blockIdx.y, gridDim.y)) {
        exact_scores(lane.cache, lane.pages.device_pages, lane.position, lane.buffers.queries,
                     {static_cast<float*>(lane.buffers.scratch), input.position_bound});
    }
}

template <typename Batch>
__global__ void verify_replay_kernel(const __grid_constant__ Batch input) {
    const auto& lane = verify_lane(input);
    if (verify_head(input, lane, blockIdx.x, gridDim.x)) {
        vector_replay_body(lane.cache, lane.pages.device_pages, lane.position,
                           {static_cast<float*>(lane.buffers.scratch), input.position_bound},
                           lane.buffers.output, blockIdx.x, blockIdx.y);
    }
}

template <typename Batch>
cudaError_t launch_verify(Batch& batch, int count, int heads, cudaStream_t stream) {
    const int blocks = std::min(32, (batch.position_bound + score_warps) / score_warps);
    void* args[] = {&batch};
    const auto status = cudaLaunchKernel(verify_scores_kernel<Batch>, dim3(blocks, heads, count),
                                         score_warps * mxfp8_group_size, args, 0, stream);
    if (status != cudaSuccess)
        return status;
    return cudaLaunchKernel(verify_replay_kernel<Batch>,
                            dim3(heads, mxfp8_head_dim / vector_replay_channels, count),
                            vector_replay_threads, args, 0, stream);
}

using VerifyRequestIds = std::array<const void*, decoder::target_split_eager_max_rows>;

struct VerifyRows {
    int rows[text_gdn_max_verify_rows]{};
    int count = 0;
};

struct VerifyGroups {
    VerifyRows groups[decoder::target_split_eager_max_rows]{};
    int count = 0;

    void add(int row, const void* request, VerifyRequestIds& identities) {
        int group = 0;
        while (group < count &&
               (identities[group] != request || groups[group].count == text_gdn_max_verify_rows))
            ++group;
        if (group == count)
            identities[count++] = request;
        auto& selected = groups[group];
        selected.rows[selected.count++] = row;
    }
};

template <typename Input> struct SharedVerifyBatch {
    Input input;
    VerifyGroups selection;
};

__device__ const decoder::PagedDecodeLane& verify_row(const gqa_decode::EagerTargetBatch& input,
                                                      int index) {
    return input.lanes[index];
}

struct BoundVerifyBatch {
    static constexpr int end = text_attention_max_tokens;
    static constexpr int position_bound = decoder::target_split_first_position;
    const decoder::PagedDecodeLane* reads;
};

__device__ const decoder::PagedDecodeLane& verify_row(const BoundVerifyBatch& input, int index) {
    return input.reads[index];
}

template <typename Input>
__global__ void shared_verify_scores(const __grid_constant__ SharedVerifyBatch<Input> batch) {
    const auto& group = batch.selection.groups[blockIdx.z];
    const auto& input = batch.input;
    const int thread = static_cast<int>(threadIdx.x);
    const int warp = thread / mxfp8_group_size;
    const int lane = thread % mxfp8_group_size;
    const int head = static_cast<int>(blockIdx.y);
    const int kv_head = head / queries_per_kv_head;
    const int first = (static_cast<int>(blockIdx.x) * score_warps) + warp;
    const int stride = static_cast<int>(gridDim.x) * score_warps;
    const auto newest = verify_row(input, group.rows[group.count - 1]);
    if (!verify_head(input, newest, head, gridDim.y) || first > newest.position)
        return;
    float queries[text_gdn_max_verify_rows][mxfp8_groups];
    float* scores[text_gdn_max_verify_rows];
    int positions[text_gdn_max_verify_rows];
    for (int row = 0; row < group.count; ++row) {
        const auto view = verify_row(input, group.rows[row]);
        load_query(gqa_query_heads(view.cache.kv_heads), view.buffers.queries, head, lane,
                   queries[row]);
        scores[row] = static_cast<float*>(view.buffers.scratch) + head * input.position_bound;
        positions[row] = view.position;
    }
    for (int index = first; index <= newest.position; index += stride) {
        const int page = newest.pages.device_pages[index / mxfp8_page_size];
        const int token = index % mxfp8_page_size;
        const auto payload = payload_offset(newest.cache.kv_heads, page, token, kv_head) + lane;
        const auto scale = scale_offset(newest.cache.kv_heads, page, token, kv_head, 0);
        float key[mxfp8_groups];
#pragma unroll
        for (int channel = 0; channel < mxfp8_groups; ++channel) {
            const auto key_index = payload + (channel * mxfp8_group_size);
            key[channel] = decode_e8m0(newest.cache.key_payload[key_index],
                                       newest.cache.key_scales[scale + channel]);
        }
        for (int row = 0; row < group.count; ++row) {
            if (index > positions[row])
                continue;
            float dot = 0.0F;
#pragma unroll
            for (int channel = 0; channel < mxfp8_groups; ++channel) {
                const float product = warp_sum(queries[row][channel] * key[channel]);
                if (lane == 0)
                    dot += product;
            }
            if (lane == 0)
                scores[row][index] = dot * softmax_scale;
        }
    }
}

__device__ __forceinline__ bool short_tile_weights(const float* scores, int count, int lane,
                                                   float2* weights, float& maximum) {
    const float score = lane < count ? scores[lane] : -CUDART_INF_F;
    const bool unit = isfinite(maximum) && (__all_sync(warp_mask, int{score <= maximum}) != 0);
    float prefix = fmaxf(maximum, score);
    if (unit) {
        if (lane < count)
            weights[lane] = {1.0F, expf(score - prefix)};
        return true;
    }
#pragma unroll
    for (int offset = 1; offset < replay_tile; offset *= 2) {
        const float earlier = __shfl_up_sync(warp_mask, prefix, offset);
        if (lane >= offset)
            prefix = fmaxf(earlier, prefix);
    }
    const float previous = __shfl_up_sync(warp_mask, prefix, 1);
    if (lane < count) {
        weights[lane] = {
            expf((lane == 0 ? maximum : previous) - prefix),
            expf(score - prefix),
        };
    }
    maximum = __shfl_sync(warp_mask, prefix, replay_tile - 1);
    return false;
}

__device__ __forceinline__ float short_tile_mass(float mass, const float2* weights, int size) {
#pragma unroll 8
    for (int index = 0; index < size; ++index) {
        const auto factor = weights[index];
        mass = fmaf(factor.x, mass, factor.y);
    }
    return mass;
}

template <typename Input>
__global__ void shared_verify_values(const __grid_constant__ SharedVerifyBatch<Input> batch) {
    __shared__ alignas(16) float values[replay_tile][vector_replay_channels];
    __shared__ float2 weights[text_gdn_max_verify_rows][replay_tile];
    __shared__ float denominators[text_gdn_max_verify_rows];
    const auto& group = batch.selection.groups[blockIdx.z];
    const auto& input = batch.input;
    const int thread = static_cast<int>(threadIdx.x);
    const int lane = thread % mxfp8_group_size;
    const int row = thread / mxfp8_group_size;
    const int head = static_cast<int>(blockIdx.x);
    const int channel = static_cast<int>(blockIdx.y);
    const auto newest = verify_row(input, group.rows[group.count - 1]);
    if (!verify_head(input, newest, head, gridDim.x))
        return;
    const auto view = row < group.count ? verify_row(input, group.rows[row]) : newest;
    const auto* scores = static_cast<float*>(view.buffers.scratch) + head * input.position_bound;
    const bool aligned = replay_values_aligned(newest.cache);
    float weighted = 0.0F;
    float maximum = -CUDART_INF_F;
    float mass = 0.0F;
    for (int first = 0; first <= newest.position; first += replay_tile) {
        const int last_count = min(replay_tile, newest.position - first + 1);
        const int count = min(replay_tile, view.position - first + 1);
        if (thread < vector_replay_threads) {
            load_replay_values(newest.cache, newest.pages.device_pages[first / mxfp8_page_size],
                               first % mxfp8_page_size, head / queries_per_kv_head, channel,
                               last_count, aligned, values);
        }
        const bool active = row < group.count && count > 0;
        const bool unit =
            active && short_tile_weights(scores + first, count, lane, weights[row], maximum);
        __syncthreads();
        if (active) {
            weighted =
                unit ? replay_vector_channel<true>(values, weights[row], count, lane, weighted)
                     : replay_vector_channel<false>(values, weights[row], count, lane, weighted);
        }
        if (row == text_gdn_max_verify_rows && lane < group.count) {
            const auto denominator_view = verify_row(input, group.rows[lane]);
            const int size = min(replay_tile, denominator_view.position - first + 1);
            mass = short_tile_mass(mass, weights[lane], size);
            denominators[lane] = mass;
        }
        __syncthreads();
    }
    if (row < group.count) {
        view.buffers.output[row_offset(0, gqa_query_heads(newest.cache.kv_heads), head) +
                            channel * vector_replay_channels + lane] =
            __float2bfloat16_rn(weighted / denominators[row]);
    }
}

template <typename Input>
cudaError_t launch_shared_verify(SharedVerifyBatch<Input>& batch, int heads, cudaStream_t stream) {
    const int blocks = std::min(32, (batch.input.position_bound + score_warps) / score_warps);
    const int groups = batch.selection.count;
    void* args[] = {&batch};
    const auto status = cudaLaunchKernel(shared_verify_scores<Input>, dim3(blocks, heads, groups),
                                         score_warps * mxfp8_group_size, args, 0, stream);
    if (status != cudaSuccess)
        return status;
    return cudaLaunchKernel(shared_verify_values<Input>,
                            dim3(heads, mxfp8_head_dim / vector_replay_channels, groups),
                            (text_gdn_max_verify_rows + 1) * mxfp8_group_size, args, 0, stream);
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
    gqa_decode::EagerTargetBatch batch{};
    for (int index = 0; index < count; ++index) {
        batch.lanes[index] = lanes[index];
    }
    batch.end = text_attention_max_tokens;
    batch.position_bound = position_bound;
    return launch_verify(batch, count, gqa_query_heads(kv_heads), stream);
}

cudaError_t decoder::short_verify_gqa_shared(const PagedDecodeLane* lanes,
                                             const void* const* requests, int count,
                                             cudaStream_t stream) {
    if (!short_verify_supported(lanes, count))
        return cudaErrorInvalidValue;
    SharedVerifyBatch<gqa_decode::EagerTargetBatch> batch{};
    batch.input.end = text_attention_max_tokens;
    batch.input.position_bound = 0;
    VerifyRequestIds identities{};
    for (int index = 0; index < count; ++index) {
        batch.input.lanes[index] = lanes[index];
        batch.input.position_bound =
            std::max(batch.input.position_bound, lanes[index].position + 1);
        batch.selection.add(index, requests[index], identities);
    }
    return launch_shared_verify(batch, gqa_query_heads(lanes[0].cache.kv_heads), stream);
}

cudaError_t decoder::short_verify_gqa_shared(const PagedDecodeLane* device_lanes,
                                             const int* indices, const void* const* requests,
                                             int count, GqaGeometry geometry, cudaStream_t stream) {
    SharedVerifyBatch<BoundVerifyBatch> batch{{device_lanes}, {}};
    VerifyRequestIds identities{};
    for (int index = 0; index < count; ++index) {
        batch.selection.add(indices[index], requests[index], identities);
    }
    return launch_shared_verify(batch, geometry.query_heads, stream);
}

cudaError_t decoder::tp_verify_graph::read(const GqaBinding* live,
                                           const std::array<int, max_rows>& indices, int rows,
                                           int query_heads, cudaStream_t stream) {
    static_assert(static_cast<std::size_t>(short_capacity) * gqa_rank_geometry.query_heads *
                      sizeof(float) <=
                  paged_decode_scratch_bytes);
    LiveVerifyBatch batch{live, {}};
    std::copy_n(indices.begin(), rows, batch.indices);
    return launch_verify(batch, rows, query_heads, stream);
}
}
