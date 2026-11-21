#pragma once

#include "sglang/mxfp8_gqa.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <optional>

namespace sglang {

inline constexpr int sm100_prefill_max_visible = 32768;
bool sm100_prefill_enabled();

// Causal chunked-prefill GQA attention on SM100 tensor memory. Returns nullopt when the
// device or the request is ineligible so the caller can fall back to the portable kernels.
std::optional<cudaError_t> launch_prefill_attn_sm100(
    Mxfp8KvCache cache, Mxfp8PageTable pages, int position, int rows,
    const __nv_bfloat16* queries, __nv_bfloat16* output, cudaStream_t stream);

namespace decoder {

inline cudaError_t tensor_core_model_prefill_gqa(
    Mxfp8KvCache cache, Mxfp8PageTable pages, int position, int rows,
    const __nv_bfloat16* queries, __nv_bfloat16* output, cudaStream_t stream) {
    if (const auto status = launch_prefill_attn_sm100(cache, pages, position, rows,
                                                      queries, output, stream)) {
        return *status;
    }
    return tensor_core_multirow_prefill_gqa(cache, pages, position, rows, queries, output, stream);
}

}

}
