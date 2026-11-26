#pragma once

#include "sglang/text_ops.h"

namespace sglang::decoder::draft_stream {

cudaError_t prepare();
cudaError_t project_fc(const __nv_bfloat16* weight, const TextBf16Projection* rows, int count,
                       cudaStream_t stream);
cudaError_t project_kv(const std::uint8_t* key, const __nv_bfloat16* key_scales,
                       const std::uint8_t* value, const __nv_bfloat16* value_scales,
                       const TextFp8Projection* rows, int count, cudaStream_t stream);

}
