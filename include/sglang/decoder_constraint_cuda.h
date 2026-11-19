#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

namespace sglang::decoder {

cudaError_t gather_constrained_logits(const __nv_bfloat16* source, const int* ids,
                                     __nv_bfloat16* logits, int count, cudaStream_t stream);
cudaError_t remap_constrained_token(const __nv_bfloat16* logits, const int* ids,
                                   int count, int* token, cudaStream_t stream);

}
