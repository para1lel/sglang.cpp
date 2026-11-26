#pragma once

#include "sglang/text/gdn/gdn_prefill_kernels.cuh"

namespace sglang {

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

}
