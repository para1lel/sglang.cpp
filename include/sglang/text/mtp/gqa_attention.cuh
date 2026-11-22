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
                                      ExactGqaWorkspace workspace, cudaStream_t stream) {
    constexpr int min_tiled_tokens = 8;
    constexpr int max_tiled_tokens = 2 * mxfp8_page_size;
    const auto visible = static_cast<std::int64_t>(position) + 1;
    if (!queries || !output || !cache_covers(cache, pages, visible)) {
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
                                  queries, output, stream);
    }
    return launch_gqa_kernel(recompute_replay_kernel,
        dim3(mxfp8_query_heads, mxfp8_head_dim / replay_channels),
        replay_threads, stream, cache, pages.device_pages, position, queries, output);
}

cudaError_t decoder::mtp_gqa_attention_batch(const DraftAttentionRow* rows, int count,
                                             cudaStream_t stream) {
    if (!rows || count <= 0 || count > draft_wave_rows) {
        return cudaErrorInvalidValue;
    }
    DraftAttentionBatch long_rows{};
    int long_count = 0;
    int longest = 0;
    for (int index = 0; index < count; ++index) {
        const auto& row = rows[index];
        const auto visible = static_cast<std::int64_t>(row.position) + 1;
        if (!row.queries || !row.output || !cache_covers(row.cache, row.pages, visible)) {
            return cudaErrorInvalidValue;
        }
        if (row.workspace.scores && visible > row.workspace.capacity &&
            visible <= text_attention_max_tokens &&
            row.workspace.capacity >= draft_score_chunk + draft_carry_per_head && count > 1) {
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
    return launch_long_draft_batch(long_rows.rows, long_count, longest, stream);
}
