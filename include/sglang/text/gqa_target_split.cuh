#pragma once

namespace {

template <typename Binding>
__global__ void target_split_combine(const __grid_constant__ Binding binding) {
    const auto lane = gqa_decode::resolve(binding);
    if (!decoder::target_split_position(lane.position, binding.end) ||
        !gqa_decode::readable(lane)) {
        return;
    }
    const int head = blockIdx.x;
    if (!gqa_decode::tensor_aligned(lane) || !gqa_decode::combine_head(lane, head)) {
        wide_decode_head(lane, head);
    }
}

cudaError_t launch_target_split(gqa_decode::TargetLane lane, cudaStream_t stream) {
    const auto status = gqa_decode::launch(lane, stream);
    if (status != cudaSuccess) {
        return status;
    }
    target_split_combine<<<mxfp8_query_heads, mxfp8_head_dim, 0, stream>>>(lane);
    return cudaGetLastError();
}

template <typename Batch>
cudaError_t launch_target_batch(const Batch& batch, int count, cudaStream_t stream) {
    const auto status = gqa_decode::launch(batch, count, stream);
    if (status != cudaSuccess) {
        return status;
    }
    target_split_combine<<<dim3(mxfp8_query_heads, 1, count), mxfp8_head_dim, 0, stream>>>(
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
            target_split_combine<<<dim3(mxfp8_query_heads, 1, count), mxfp8_head_dim, 0, stream>>>(
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
