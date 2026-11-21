#pragma once

struct GateRows {
    const __nv_bfloat16* rows[gqa_decode::max_lanes];
};

__global__ void __launch_bounds__(mxfp8_head_dim) combine_replay(
    const __grid_constant__ gqa_decode::Batch batch, const __grid_constant__ GateRows gates) {
    const auto input = resolve_lane(batch.lanes[blockIdx.y]);
    if (!gqa_decode::readable(input)) {
        return;
    }
    const int head = static_cast<int>(blockIdx.x);
    if (!gqa_decode::tensor_aligned(input) || !gqa_decode::combine_head(input, head)) {
        wide_decode_head(input, head);
    }
    if (const auto* gate = gates.rows[blockIdx.y]) {
        const int channel = static_cast<int>(threadIdx.x);
        auto& output = input.buffers.output[head * mxfp8_head_dim + channel];
        output = __float2bfloat16(gqa_decode::gate_attention(
            gate[(head * 2 + 1) * mxfp8_head_dim + channel], output));
    }
}

cudaError_t launch_split_decode(const decoder::BoundPagedDecodeLane* lanes, int count,
                                cudaStream_t stream, const __nv_bfloat16* const* gates) {
    gqa_decode::Batch batch{};
    auto status = gqa_decode::make_batch(lanes, count, batch);
    if (status == cudaSuccess) {
        status = gqa_decode::launch(batch, count, stream);
    }
    if (status != cudaSuccess) {
        return status;
    }
    GateRows rows{};
    if (gates) {
        std::copy_n(gates, count, rows.rows);
    }
    combine_replay<<<dim3(mxfp8_query_heads, count), mxfp8_head_dim, 0, stream>>>(batch, rows);
    return cudaGetLastError();
}
