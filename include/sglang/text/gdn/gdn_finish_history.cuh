#pragma once

// Shared private GDN helper (moved from src/text_attention_gdn.cu so the
// batched replay twin in src/text/text_gdn_batched_replay.cu uses the identical
// definition instead of a copy). Include gdn_local.h before this header.
// Device-only template, so the definition is safe across translation units.

#include "sglang/text/gdn/gdn_prefill_kernels.cuh"

namespace sglang {

/// Writes one feature's history taps: the last text_gdn_history_taps rows of
/// the committed prefix, sampled through raw_sample (the pre-infix rows fall
/// back to the pre-replay history).
template <typename S>
__device__ void finish_history(int token_count, const __nv_bfloat16* qkv,
                               const __nv_bfloat16* history, __nv_bfloat16* final_history,
                               int feature) {
    #pragma unroll
    for (int tap = 0; tap < text_gdn_history_taps; ++tap) {
        final_history[feature * text_gdn_history_taps + tap] =
            raw_sample<S>(qkv, history, token_count - text_gdn_history_taps + tap, feature);
    }
}

}  // namespace sglang
