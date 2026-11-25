#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <optional>

namespace sglang {

// BF16 head projection with a single full-weight read across all rows.
//
// Replaces the <=5-row text_bf16_gemm groups used by the verification score
// path for the canonical rank-0 BF16 lm_head: at 10 rows the grouped path
// reads the 2.5 GB weight twice (1.62 ms measured); this wrapper packs the
// per-row inputs, routes them through the capture-safe tcgen05 decode
// operator (try_text_bf16_decode_sm100, fp32 TMEM accumulation, bf16 store),
// and scatters the packed logits back to the per-row outputs (~0.40 ms
// end-to-end, flat from 5 to 20 rows; argmax and logprob parity verified
// against the grouped path on real checkpoint weights).
//
// Contract:
struct Bf16HeadRow {
    const __nv_bfloat16* input;
    __nv_bfloat16* output;
};

//   count          in [1, 32]; 0 or negative rows and over-wide batches are
//                  "no work / unsupported" and return nullopt without
//                  touching any buffer, so callers fall back to the grouped
//                  text_bf16_gemm path.
//   widths         in_width and out_width in [1, 2^28]; values outside that
//                  range return cudaErrorInvalidValue (they cannot be
//                  indexed safely by the 32-bit block math).
//   pointers       weight, packed, packed_logits, rows, and every
//                  rows[r].input / rows[r].output must be non-null
//                  (cudaErrorInvalidValue otherwise). rows[r].input/output
//                  have no alignment requirement beyond bf16 element size;
//                  the pack/scatter kernels access them element-wise.
//   alignment      weight, packed and packed_logits must be 16-byte aligned
//                  and in_width must be a multiple of 64 elements (the
//                  tcgen05 operator's TMA chunk); otherwise nullopt is
//                  returned before any kernel launch.
//   scratch        packed needs count * in_width bf16 values and
//                  packed_logits needs count * out_width bf16 values.
//                  Aliasing: packed, packed_logits, weight and every
//                  rows[].input / rows[].output range must be pairwise
//                  disjoint. The scatter reads packed_logits row r while
//                  other blocks write rows[r'].output, so any cross-row or
//                  partially overlapping aliasing of outputs with
//                  packed_logits is a data race - including the case where
//                  destinations are permuted copies of identical ranges.
//                  The pack has the same hazard between packed and the
//                  input rows. Identical same-row ranges degenerate to
//                  element-wise in-place copies, but the wrapper does not
//                  rely on that: treat all aliasing as forbidden.
//   errors         kernel launch failures (pack, projection, scatter) are
//                  returned immediately as the optional error value; the
//                  optional is empty only for layouts the tcgen05 operator
//                  declines, where no output row has been written.
//   capture safety plain stream launches only; no allocations, host syncs,
//                  or cross-stream work. Safe inside CUDA graph capture.
std::optional<cudaError_t> project_bf16_head(const __nv_bfloat16* weight, int out_width,
                                             int in_width, const Bf16HeadRow* rows, int count,
                                             __nv_bfloat16* packed, __nv_bfloat16* packed_logits,
                                             cudaStream_t stream);

}  // namespace sglang
