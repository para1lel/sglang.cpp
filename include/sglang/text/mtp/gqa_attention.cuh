#pragma once

namespace {

#include "sglang/text/mtp/gqa_long.cuh"

__global__ void recompute_replay_kernel(
    Mxfp8KvCache cache, const std::int32_t* pages, int position, const __nv_bfloat16* queries,
    __nv_bfloat16* output) {
    ordered_replay<true>(cache, pages, position, nullptr, queries, output);
}

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
        return launch_exact_attention<false, false, true>(
            cache, pages.device_pages, position, queries, output, workspace, stream);
    }
    if (visible > workspace.capacity && visible <= text_attention_max_tokens && workspace.scores &&
        workspace.capacity >= draft_score_chunk + draft_carry_per_head) {
        return launch_long_draft_attention(
            cache, pages.device_pages, static_cast<int>(visible), queries, output, workspace,
            stream);
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
        if (!row.queries || !row.output || !cache_covers(row.cache, row.pages, visible)) {
            return cudaErrorInvalidValue;
        }
        if (row.workspace.scores && visible > row.workspace.capacity &&
            visible <= text_attention_max_tokens &&
            row.workspace.capacity >= draft_score_chunk + draft_carry_per_head && count > 1) {
            static_assert(sizeof(float) * mxfp8_query_heads *
                          (draft_score_chunk + draft_carry_per_head) >=
                          sizeof(gqa_decode::Partial) * gqa_decode::max_splits * mxfp8_query_heads);
            const PagedDecodeLane lane{row.cache, row.pages, row.position,
                {row.queries, row.output, row.workspace.scores}};
            if (b200 && gqa_decode::valid_buffers(lane.buffers) &&
                gqa_decode::tensor_aligned(lane)) {
                online_lanes[online_count++] = lane;
                continue;
            }
            long_rows.rows[long_count++] = row;
            longest = std::max(longest, static_cast<int>(visible));
        } else {
            const auto status = mtp_gqa_attention(row.cache, row.pages, row.position, row.queries,
                                                  row.output, row.workspace, stream);
            if (status != cudaSuccess) {
                return status;
            }
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
