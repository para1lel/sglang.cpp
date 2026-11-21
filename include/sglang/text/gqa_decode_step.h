#pragma once

#include "sglang/decoder_cuda.cuh"

namespace sglang::decoder {

inline constexpr int gqa_decode_step_max_lanes = 64;

// One decode token of one lane. rope_queries (equal to read.buffers.queries) and rope_keys
// receive the rotated Q/K; read.buffers.output receives the gated attention.
struct GqaDecodeStepLane {
    BoundPagedDecodeLane read;
    __nv_bfloat16* rope_queries;
    const __nv_bfloat16* q_gate;
    const __nv_bfloat16* keys;
    const __nv_bfloat16* values;
    __nv_bfloat16* rope_keys;
};

// Normalizes and rotates Q/K, appends MXFP8 K/V, runs split-KV attention and applies the
// output gate for 1..gqa_decode_step_max_lanes lanes with a fixed number of launches.
// Positions are read on the device, so the call is graph-capturable.
// A negative position marks padding and leaves its buffers untouched. Each lane needs
// 16-byte aligned scratch of paged_decode_scratch_bytes. Active lanes must provide
// valid physical page IDs and backing payloads and scales through the current position.
// Device bindings, buffers and norm weights must remain valid through stream completion
// and each graph replay. A launch failure can leave earlier stages enqueued.
cudaError_t decode_gqa_step(const GqaDecodeStepLane* lanes, int count,
                            const __nv_bfloat16* q_norm, const __nv_bfloat16* k_norm,
                            cudaStream_t stream);

}
