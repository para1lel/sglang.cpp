template <bool Narrow = false, bool Prefill = false, bool Vector = false,
          bool SkipTargetSplit = false, typename Position, typename Cache, typename Pages,
          typename Scores>
cudaError_t launch_exact_attention(
    Cache cache, Pages pages, Position position, const __nv_bfloat16* queries,
    __nv_bfloat16* output, Scores workspace, cudaStream_t stream, int split_end = 0) {
    const int capacity = score_grid_capacity(workspace);
    if (!queries || !output || capacity <= 0 || capacity > text_attention_max_tokens) {
        return cudaErrorInvalidValue;
    }
    const int query_heads = [&] {
        if constexpr (std::is_pointer_v<Cache>) {
            return mxfp8_query_heads;
        } else {
            return gqa_query_heads(cache.kv_heads);
        }
    }();
    const int blocks = std::min(32, (capacity + score_warps - 1) / score_warps);
    const auto status = launch_gqa_kernel(
        exact_scores_kernel<Narrow, SkipTargetSplit, Position, Cache, Pages, Scores>,
        dim3(blocks, query_heads), score_warps * mxfp8_group_size, stream,
        cache, pages, position, queries, workspace, split_end);
    if (status != cudaSuccess) {
        return status;
    }
    if constexpr (Prefill) {
        return launch_gqa_kernel(prefill_replay_kernel,
            dim3(query_heads, mxfp8_head_dim / replay_channels),
            prefill_replay_threads, stream, cache, pages, position, workspace, output);
    } else if constexpr (Vector) {
        return launch_gqa_kernel(
            vector_replay_kernel<SkipTargetSplit, Position, Cache, Pages, Scores>,
            dim3(query_heads, mxfp8_head_dim / vector_replay_channels),
            vector_replay_threads, stream, cache, pages, position, workspace, output, split_end);
    } else {
        return launch_gqa_kernel(
            exact_replay_kernel<Narrow, Position, Cache, Pages, Scores>,
            dim3(query_heads, mxfp8_head_dim / replay_channels),
            replay_threads, stream, cache, pages, position, workspace, queries, output);
    }
}
