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
            output[row_offset(0, mxfp8_query_heads, head) +
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
                       scores, scores + mxfp8_query_heads * draft_score_chunk, first == 0,
                       count == remaining, row.output);
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
        const int blocks = std::min(32, (count + score_warps - 1) / score_warps);
        void* arguments[] = {&batch, &first};
        auto status = cudaLaunchKernel(
            draft_range_scores_batch, dim3(blocks, mxfp8_query_heads, active),
            score_warps * mxfp8_group_size, arguments, 0, stream);
        if (status != cudaSuccess) {
            return status;
        }
        status = cudaLaunchKernel(
            draft_range_replay_batch, dim3(mxfp8_query_heads, draft_replay_groups, active),
            vector_replay_threads, arguments, 0, stream);
        if (status != cudaSuccess) {
            return status;
        }
    }
    return cudaSuccess;
}

cudaError_t launch_long_draft_attention(
    Mxfp8KvCache cache, const std::int32_t* pages, int visible, const __nv_bfloat16* queries,
    __nv_bfloat16* output, decoder::ExactGqaWorkspace workspace, cudaStream_t stream) {
    auto* carry = workspace.scores + mxfp8_query_heads * draft_score_chunk;
    workspace.capacity = draft_score_chunk;
    for (int first = 0; first < visible; first += draft_score_chunk) {
        const auto* range_pages = pages + first / mxfp8_page_size;
        const int count = std::min(draft_score_chunk, visible - first);
        const int blocks = std::min(32, (count + score_warps - 1) / score_warps);
        const auto scores = exact_scores_kernel<
            false, false, int, Mxfp8KvCache, const std::int32_t*, decoder::ExactGqaWorkspace>;
        auto status = launch_gqa_kernel(scores, dim3(blocks, mxfp8_query_heads),
            score_warps * mxfp8_group_size, stream, cache, range_pages, count - 1,
            queries, workspace, 0);
        if (status != cudaSuccess) {
            return status;
        }
        status = launch_gqa_kernel(
            draft_range_replay_kernel, dim3(mxfp8_query_heads, draft_replay_groups),
            vector_replay_threads, stream, cache, range_pages, count, workspace.scores, carry,
            first == 0, count == visible - first, output);
        if (status != cudaSuccess) {
            return status;
        }
    }
    return cudaSuccess;
}
