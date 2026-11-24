#pragma once

#include "sglang/text_ops.h"

#include <initializer_list>
#include <optional>

namespace sglang {

inline constexpr int serial_prefill_max_rows = 2048;
inline constexpr int grouped_prefill_max_rows = 4096;

/// Uses the contiguous FP8 layouts and finite-input contract of text_fp8_gemv_rows.
/// Empty means the backend declined before submission and SIMT fallback is safe.
/// A status must be handled normally, with storage retained through stream completion.
std::optional<cudaError_t> try_text_fp8_gemm_sm100(
    const std::uint8_t* weight, const __nv_bfloat16* weight_scales, TextFp8ActivationView input,
    int rows, int out_width, int in_width, __nv_bfloat16* output, cudaStream_t stream);

std::optional<cudaError_t> try_text_fp8_gemm_sm100(
    const std::uint8_t* weight, const __nv_bfloat16* weight_scales, TextFp8ActivationView input,
    int rows, int out_width, int in_width, __nv_bfloat16* output, cudaStream_t stream,
    bool allow_large_rows, bool grouped_projection = false);

/// Reuses distinct dead MLP gate/up buffers for scale conversion.
/// Empty indicates an unsupported geometry or device before any submission.
std::optional<cudaError_t> try_prefill_scaled_sm100(
    TextFp8ActivationView input, const std::uint8_t* weight, const __nv_bfloat16* weight_scales,
    int rows, int out_width, int in_width, __nv_bfloat16* output,
    __nv_bfloat16* gate_scratch, __nv_bfloat16* up_scratch, cudaStream_t stream);

struct PrefillFp8Projection {
    const std::uint8_t* weight;
    const __nv_bfloat16* scales;
    __nv_bfloat16* output;
    int width;
};

/// Submits two or three same-input projections with contiguous, block-aligned weight and scales.
/// Each projection writes a separate array under the same lifetime contract as FP8 GEMM.
std::optional<cudaError_t> try_prefill_fp8_packed_sm100(
    TextFp8ActivationView input, int rows, int in_width,
    std::initializer_list<PrefillFp8Projection> projections, cudaStream_t stream,
    bool grouped_projection = false);

}
