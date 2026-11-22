template <bool Narrow = false, bool Prefill = false, bool Vector = false, typename Position,
          typename Cache, typename Pages, typename Scores>
cudaError_t launch_exact_attention(
    Cache cache, Pages pages, Position position, const __nv_bfloat16* queries,
    __nv_bfloat16* output, Scores workspace, cudaStream_t stream) {
    const int capacity = score_grid_capacity(workspace);
    if (!queries || !output || capacity <= 0 || capacity > text_attention_max_tokens) {
        return cudaErrorInvalidValue;
    }
    constexpr int max_score_blocks = 32;
    const int blocks = std::min(max_score_blocks, (capacity + score_warps - 1) / score_warps);
    void* score_arguments[] = {&cache, &pages, &position, &queries, &workspace};
    const auto status = cudaLaunchKernel(
        exact_scores_kernel<Narrow, Position, Cache, Pages, Scores>,
        dim3(blocks, mxfp8_query_heads), score_warps * mxfp8_group_size, score_arguments,
        0, stream);
    if (status != cudaSuccess) {
        return status;
    }
    if constexpr (Prefill) {
        void* replay_arguments[] = {&cache, &pages, &position, &workspace, &output};
        return cudaLaunchKernel(prefill_replay_kernel,
            dim3(mxfp8_query_heads, mxfp8_head_dim / replay_channels),
            prefill_replay_threads, replay_arguments, 0, stream);
    }
    if constexpr (Vector) {
        void* replay_arguments[] = {&cache, &pages, &position, &workspace, &output};
        return cudaLaunchKernel(vector_replay_kernel<Position, Cache, Pages, Scores>,
            dim3(mxfp8_query_heads, mxfp8_head_dim / vector_replay_channels),
            vector_replay_threads, replay_arguments, 0, stream);
    }
    void* replay_arguments[] = {&cache, &pages, &position, &workspace, &queries, &output};
    return cudaLaunchKernel(exact_replay_kernel<Narrow, Position, Cache, Pages, Scores>,
                            dim3(mxfp8_query_heads, mxfp8_head_dim / replay_channels),
                            replay_threads,
                            replay_arguments, 0, stream);
}
