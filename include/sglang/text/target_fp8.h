#pragma once

#include <optional>

#include "sglang/text_ops.h"

namespace sglang {

inline constexpr int text_target_fp8_max_rows = 24;
inline constexpr int text_target_fp8_m10_rows = 10;
inline constexpr int text_target_fp8_m40_rows = 40;
inline constexpr int text_target_fp8_m80_rows = 80;
inline constexpr int text_target_fp8_m160_rows = 160;

bool target_fp8_wide_rows(int count);

std::optional<cudaError_t> try_target_fp8_tc(const TextFp8DecodeTarget* targets,
                                           int target_count, int in_width,
                                           const TextFp8Projection* rows, int count,
                                           __nv_bfloat16* scratch, cudaStream_t stream);

std::optional<cudaError_t> try_target_fp8_gate_up(
    const TextFp8DecodeTarget (&targets)[2], int in_width,
    const TextFp8Activation (&inputs)[text_target_fp8_m10_rows],
    const TextFp8Activation (&quantized)[text_target_fp8_m10_rows], cudaStream_t stream);

}
