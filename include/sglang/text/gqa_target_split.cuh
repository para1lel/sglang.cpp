#pragma once

namespace {

__global__ void target_split_combine(const __grid_constant__ gqa_decode::TargetLane binding) {
    const auto lane = gqa_decode::resolve(binding);
    if (!decoder::target_split_position(lane.position) || !gqa_decode::readable(lane)) {
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

}

cudaError_t decoder::target_split_gqa(PagedDecodeLane lane, cudaStream_t stream) {
    if (!target_split_position(lane.position) || !gqa_decode::readable(lane) ||
        !gqa_decode::valid_buffers(lane.buffers)) {
        return cudaErrorInvalidValue;
    }
    return launch_target_split({lane, {}}, stream);
}

cudaError_t decoder::target_split_gqa(BoundPagedDecodeLane lane, cudaStream_t stream) {
    if (!lane.cache || !lane.pages || !lane.position || !lane.scores ||
        !gqa_decode::valid_buffers(lane.buffers)) {
        return cudaErrorInvalidValue;
    }
    const auto status = launch_target_split({{}, lane}, stream);
    if (status != cudaSuccess) {
        return status;
    }
    return launch_exact_attention<false, false, true, true>(
        lane.cache, lane.pages, lane.position, lane.buffers.queries, lane.buffers.output,
        lane.scores, stream);
}
