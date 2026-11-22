#pragma once

#include <optional>

#include "sglang/text_ops.h"

namespace sglang {

inline constexpr int text_target_fp8_max_rows = 24;

std::optional<cudaError_t> try_target_fp8_tc(const TextFp8DecodeTarget* targets,
                                           int target_count, int in_width,
                                           const TextFp8Projection* rows, int count,
                                           __nv_bfloat16* scratch, cudaStream_t stream);

}
