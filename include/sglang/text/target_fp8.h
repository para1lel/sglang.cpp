#pragma once

#include <optional>

#include "sglang/text_ops.h"

namespace sglang {

inline constexpr int text_target_fp8_max_rows = 24;
inline constexpr int text_target_fp8_m40_rows = 40;
inline constexpr int text_target_fp8_m80_rows = 80;
inline constexpr int text_target_fp8_m160_rows = 160;

constexpr bool target_fp8_wide_rows(int count) {
    return count == text_target_fp8_m40_rows || count == text_target_fp8_m80_rows ||
           count == text_target_fp8_m160_rows;
}

std::optional<cudaError_t> try_target_fp8_tc(const TextFp8DecodeTarget* targets,
                                           int target_count, int in_width,
                                           const TextFp8Projection* rows, int count,
                                           __nv_bfloat16* scratch, cudaStream_t stream);

}
