#pragma once

// Eager batched short-context verification: the exact-scores/replay kernel
// pair over EagerTargetBatch value lanes. Mid-translation-unit fragment like
// gqa_target_split.cuh — include it where that header is included, after the
// mxfp8_gqa.cu helpers (exact_scores, vector_replay_body, score_warps,
// launch_gqa_kernel) are visible.

namespace {

namespace gqa_verify {
__device__ decoder::PagedDecodeLane verify_lane(const gqa_decode::EagerTargetBatch& input) {
    return input.lanes[blockIdx.z];
}

__device__ decoder::ExactGqaWorkspace verify_scores(const gqa_decode::EagerTargetBatch& input,
                                                    int lane) {
    return {static_cast<float*>(input.lanes[lane].buffers.scratch), input.position_bound};
}
}  // namespace gqa_verify

/// Exact scores for the lanes of one batch: each block serves score_warps
/// tokens of one query head. Lanes outside the batch contract (position in
/// the split range, unreadable, or past the position bound) are skipped.
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

/// Value replay for the same lanes; consumes the scores written by
/// verify_scores_kernel under the identical skip conditions.
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

}  // namespace

/// Pure eligibility check for short_verify_gqa so callers can fall back
/// before running any side effects (preprocess or cache appends). Eligible
/// lanes sit strictly below the split range and their combined score
/// workspace fits each lane's scratch buffer.
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
        position_bound = max(position_bound, lane.position + 1);
        kv_heads = max(kv_heads, lane.cache.kv_heads);
    }
    const auto score_bytes = static_cast<std::size_t>(position_bound)
        * gqa_query_heads(kv_heads) * sizeof(float);
    return score_bytes <= paged_decode_scratch_bytes;
}

/// Eager batched short-context verify: the scores/replay kernel pair over
/// value lanes whose positions sit strictly below the split range. Each
/// lane's scratch buffer backs its score workspace; short_verify_supported
/// (including the room requirement) gates the entry before any side effect.
cudaError_t decoder::short_verify_gqa(const PagedDecodeLane* lanes, int count,
                                      cudaStream_t stream) {
    if (!short_verify_supported(lanes, count)) {
        return cudaErrorInvalidValue;
    }
    int position_bound = 0;
    int kv_heads = 0;
    for (int index = 0; index < count; ++index) {
        position_bound = max(position_bound, lanes[index].position + 1);
        kv_heads = max(kv_heads, lanes[index].cache.kv_heads);
    }
    const GqaGeometry geometry{gqa_query_heads(kv_heads), kv_heads};
    gqa_decode::EagerTargetBatch batch{};
    for (int index = 0; index < count; ++index) {
        batch.lanes[index] = lanes[index];
    }
    batch.end = text_attention_max_tokens;
    batch.position_bound = position_bound;
    constexpr int score_threads = score_warps * mxfp8_group_size;
    auto status = launch_gqa_kernel(
        verify_scores_kernel,
        dim3(min(32, (position_bound + score_warps) / score_warps), geometry.query_heads, count),
        dim3(score_threads), stream, batch);
    if (status == cudaSuccess) {
        status = launch_gqa_kernel(
            verify_replay_kernel,
            dim3(geometry.query_heads, mxfp8_head_dim / vector_replay_channels, count),
            dim3(vector_replay_threads), stream, batch);
    }
    return status;
}
