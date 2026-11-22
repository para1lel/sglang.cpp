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
    void* arguments[] = {&cache, &pages.device_pages, &position, &queries, &output};
    return cudaLaunchKernel(recompute_replay_kernel,
                            dim3(mxfp8_query_heads, mxfp8_head_dim / replay_channels),
                            replay_threads, arguments, 0, stream);
}
