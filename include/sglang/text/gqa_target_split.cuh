#pragma once

namespace {

template <typename Binding, int FixedSplits = 0>
__global__ void target_split_combine(const __grid_constant__ Binding binding) {
    const auto lane = gqa_decode::resolve(binding);
    if (!decoder::target_split_position(lane.position, binding.end) ||
        !gqa_decode::readable(lane)) {
        return;
    }
    const int head = blockIdx.x;
    // Bound descriptors launch the combine with the maximum head grid; the
    // lane's own cache decides which blocks carry work.
    if (head >= gqa_query_heads(lane.cache.kv_heads)) {
        return;
    }
    if (!gqa_decode::tensor_aligned(lane) || !gqa_decode::combine_head<FixedSplits>(lane, head)) {
        wide_decode_head(lane, head);
    }
}

/// Rank-local combine width: a TP2 lane owns `gqa_query_heads(cache.kv_heads)`
/// query heads instead of the full 24, so the combine grid must shrink to
/// match the lane's own cache or blocks would write past its output rows.
/// Bound descriptors keep their cache on the device; the host cannot read its
/// head count, so they use the maximum grid and rely on the kernel guard.
int target_split_heads(const decoder::PagedDecodeLane& lane) {
    return gqa_query_heads(lane.cache.kv_heads);
}

int target_split_heads(const decoder::BoundPagedDecodeLane&) {
    return mxfp8_query_heads;
}

int target_split_heads(const gqa_decode::TargetLane& lane) {
    return lane.bound.position ? target_split_heads(lane.bound)
                               : target_split_heads(lane.eager);
}

cudaError_t launch_target_split(gqa_decode::TargetLane lane, cudaStream_t stream) {
    const auto status = gqa_decode::launch(lane, stream);
    if (status != cudaSuccess) {
        return status;
    }
    target_split_combine<<<target_split_heads(lane), mxfp8_head_dim, 0, stream>>>(lane);
    return cudaGetLastError();
}

template <typename Batch>
cudaError_t launch_target_batch(const Batch& batch, int count, cudaStream_t stream) {
    const auto status = gqa_decode::launch(batch, count, stream);
    if (status != cudaSuccess) {
        return status;
    }
    target_split_combine<<<dim3(target_split_heads(batch.lanes[0]), 1, count),
                          mxfp8_head_dim, 0, stream>>>(
        batch);
    return cudaGetLastError();
}

bool valid_target_lane(const decoder::PagedDecodeLane& lane, int end) {
    return decoder::target_split_position(lane.position, end) && gqa_decode::readable(lane) &&
           gqa_decode::valid_buffers(lane.buffers);
}

bool valid_target_binding(const decoder::BoundPagedDecodeLane& lane) {
    return lane.cache && lane.pages && lane.position && lane.scores &&
           gqa_decode::valid_buffers(lane.buffers);
}

int eager_target_batch_width(const decoder::PagedDecodeLane* lanes, int count, int end) {
    constexpr int reference_width = text_gdn_max_verify_rows;
    constexpr int max_average_chunks_per_split = 10;
    if (end != text_attention_max_tokens || count < 2 * reference_width) {
        return reference_width;
    }
    for (int index = 0; index < count; ++index) {
        const auto& lane = lanes[index];
        const int chunks = lane.position / gqa_decode::split_chunk_tokens + 1;
        if (!gqa_decode::tensor_aligned(lane) ||
            chunks > max_average_chunks_per_split * gqa_decode::split_count(lane.position)) {
            return reference_width;
        }
    }
    return count;
}

cudaError_t launch_target_fallback(decoder::BoundPagedDecodeLane lane, int end,
                                   cudaStream_t stream) {
    return launch_exact_attention<false, false, true, true>(
        lane.cache, lane.pages, lane.position, lane.buffers.queries, lane.buffers.output,
        lane.scores, stream, end);
}

}

cudaError_t decoder::target_split_gqa(PagedDecodeLane lane, cudaStream_t stream, int end) {
    if (!valid_target_lane(lane, end)) {
        return cudaErrorInvalidValue;
    }
    return launch_target_split({lane, {}, end}, stream);
}

cudaError_t decoder::target_split_gqa(BoundPagedDecodeLane lane, cudaStream_t stream, int end) {
    if (!valid_target_binding(lane)) {
        return cudaErrorInvalidValue;
    }
    const auto status = launch_target_split({{}, lane, end}, stream);
    return status == cudaSuccess ? launch_target_fallback(lane, end, stream) : status;
}

cudaError_t decoder::target_split_gqa(const BoundPagedDecodeLane* lanes, int count, int end,
                                     cudaStream_t stream) {
    if (!lanes || count < 1 || count > text_gdn_max_verify_rows) {
        return cudaErrorInvalidValue;
    }
    gqa_decode::TargetBatch batch{{}, end};
    for (int index = 0; index < count; ++index) {
        if (!valid_target_binding(lanes[index])) {
            return cudaErrorInvalidValue;
        }
        batch.lanes[index] = lanes[index];
    }
    auto status = launch_target_batch(batch, count, stream);
    for (int index = 0; index < count && status == cudaSuccess; ++index) {
        status = launch_target_fallback(lanes[index], end, stream);
    }
    return status;
}

cudaError_t decoder::target_split_gqa(const PagedDecodeLane* lanes, int count, int end,
                                     cudaStream_t stream) {
    if (!lanes || count < 1 || count > target_split_eager_max_rows ||
        (count > text_gdn_max_verify_rows && end != text_attention_max_tokens)) {
        return cudaErrorInvalidValue;
    }
    for (int index = 0; index < count; ++index) {
        if (!valid_target_lane(lanes[index], end)) {
            return cudaErrorInvalidValue;
        }
    }
    if (count == target_split_eager_max_rows) {
        gqa_decode::EagerTargetBatch batch{{}, end};
        std::copy_n(lanes, count, batch.lanes);
        if (const auto status = gqa_decode::try_launch_reuse(batch, count, stream)) {
            if (*status != cudaSuccess) {
                return *status;
            }
            target_split_combine<<<
                dim3(target_split_heads(batch.lanes[0]), 1, count), mxfp8_head_dim, 0, stream>>>(
                batch);
            return cudaGetLastError();
        }
    }

    const int width = eager_target_batch_width(lanes, count, end);
    for (int offset = 0; offset < count; offset += width) {
        const int size = std::min(width, count - offset);
        gqa_decode::EagerTargetBatch batch{{}, end};
        std::copy_n(lanes + offset, size, batch.lanes);
        const auto status = launch_target_batch(batch, size, stream);
        if (status != cudaSuccess) {
            return status;
        }
    }
    return cudaSuccess;
}

cudaError_t gqa_decode::combine_long_decode(const LongBatch& batch, int requests,
                                            cudaStream_t stream) {
    // The long verifier starts beyond the position that saturates split_count.
    static_assert(long_min_position >= (max_splits - 1) * min_split_chunks * split_chunk_tokens);
    void* arguments[] = {const_cast<LongBatch*>(&batch)};
    return cudaLaunchKernel(target_split_combine<LongBatch, max_splits>,
                            dim3(gqa_query_heads(batch.requests[0].cache.kv_heads), 1,
                                 requests * long_rows),
                            mxfp8_head_dim, arguments, 0, stream);
}

namespace {

template <int Capacity>
__global__ void gqa_verify_append_kernel(
    const __grid_constant__ decoder::GqaVerifyRows<decoder::GqaVerifyAppendRow, Capacity> batch) {
    const auto& row = batch.rows[blockIdx.x];
    append_head(row.cache, row.pages.device_pages, row.position, 0, row.keys, row.values);
}

}

template <int Capacity>
cudaError_t decoder::gqa_verify_append(const GqaVerifyRows<GqaVerifyAppendRow, Capacity>& batch,
                                       int count, cudaStream_t stream) {
    if (count <= 0 || count > Capacity) {
        return cudaErrorInvalidValue;
    }
    for (int index = 0; index < count; ++index) {
        const auto& row = batch.rows[index];
        if (row.position < 0 || !row.keys || !row.values ||
            row.cache.kv_heads != batch.rows[0].cache.kv_heads ||
            !cache_covers(row.cache, row.pages, static_cast<std::int64_t>(row.position) + 1)) {
            return cudaErrorInvalidValue;
        }
    }
    void* arguments[] = {const_cast<GqaVerifyRows<GqaVerifyAppendRow, Capacity>*>(&batch)};
    return cudaLaunchKernel(gqa_verify_append_kernel<Capacity>,
                            dim3(count, batch.rows[0].cache.kv_heads),
                            mxfp8_head_dim, arguments, 0, stream);
}

template cudaError_t decoder::gqa_verify_append(
    const GqaVerifyRows<GqaVerifyAppendRow>&, int, cudaStream_t);
template cudaError_t decoder::gqa_verify_append(
    const GqaVerifyRows<GqaVerifyAppendRow, max_wide_mtp_rows>&, int, cudaStream_t);
