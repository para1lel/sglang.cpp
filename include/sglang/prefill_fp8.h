#pragma once

#include "sglang/text_ops.h"

#include <optional>

namespace sglang {

inline constexpr int serial_prefill_max_rows = 2048;

/// Uses the contiguous FP8 layouts and finite-input contract of text_fp8_gemv_rows.
/// Empty means the backend declined before submission and SIMT fallback is safe.
/// A status must be handled normally, with storage retained through stream completion.
std::optional<cudaError_t> try_text_fp8_gemm_sm100(
    const std::uint8_t* weight, const __nv_bfloat16* weight_scales, TextFp8ActivationView input,
    int rows, int out_width, int in_width, __nv_bfloat16* output, cudaStream_t stream);

std::optional<cudaError_t> try_text_fp8_gemm_sm100(
    const std::uint8_t* weight, const __nv_bfloat16* weight_scales, TextFp8ActivationView input,
    int rows, int out_width, int in_width, __nv_bfloat16* output, cudaStream_t stream,
    bool allow_large_rows);

}
