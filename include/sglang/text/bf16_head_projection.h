#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <optional>

namespace sglang {

struct Bf16HeadRow {
    const __nv_bfloat16* input;
    __nv_bfloat16* output;
};

std::optional<cudaError_t> project_bf16_head(const __nv_bfloat16* weight, int out_width,
                                             int in_width, const Bf16HeadRow* rows, int count,
                                             __nv_bfloat16* packed, __nv_bfloat16* packed_logits,
                                             cudaStream_t stream);

}
