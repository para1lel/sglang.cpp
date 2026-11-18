#pragma once

#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

namespace sglang {

inline constexpr int text_fp8_group_size = 128;
inline constexpr int text_fp8_max_batch = 5;

constexpr int text_fp8_group_count(int width) {
    return width / text_fp8_group_size + (width % text_fp8_group_size != 0);
}

struct TextFp8Activation {
    std::uint8_t* values;
    float* scales;
};

struct TextFp8ActivationView {
    const std::uint8_t* values;
    const float* scales;
};

struct TextFp8Projection {
    TextFp8ActivationView input;
    __nv_bfloat16* output;
};

struct TextBf16Projection {
    const __nv_bfloat16* input;
    __nv_bfloat16* output;
};

struct TextLogitCandidate {
    float score;
    int token_id;
};

constexpr int text_argmax_scratch_count(int vocab_size) {
    constexpr int tile = 1024;
    return vocab_size / tile + (vocab_size % tile != 0);
}

/// All pointers refer to contiguous device arrays. The caller owns capacity,
/// ordering and lifetime through completion. Dimensions must be positive.
/// Launches are asynchronous and return launch status. Use finite model inputs
/// and FP32-representable intermediate arithmetic. Buffers are disjoint except
/// for the exact in-place aliases specified by each operation.

/// Copies token_id from a BF16 [vocab_size, width] embedding table.
cudaError_t text_embedding(const __nv_bfloat16* weight, int vocab_size, int width, int token_id,
                           __nv_bfloat16* output, cudaStream_t stream);

inline constexpr int text_embedding_max_rows = 128;
cudaError_t text_embedding_rows(const __nv_bfloat16* weight, int vocab_size, int width,
                                const int* host_token_ids, int rows, __nv_bfloat16* output,
                                cudaStream_t stream);

/// BF16 [out_width, in_width] matrix times a BF16 vector, with FP32
/// accumulation and BF16 output. Also applies to the LM head.
cudaError_t text_bf16_gemv(const __nv_bfloat16* weight, const __nv_bfloat16* input,
                           int out_width, int in_width, __nv_bfloat16* output,
                           cudaStream_t stream);

cudaError_t text_bf16_gemv_rows(const __nv_bfloat16* weight, const __nv_bfloat16* input,
                                int out_width, int in_width, int rows, __nv_bfloat16* output,
                                cudaStream_t stream);
cudaError_t text_bf16_gemm(const __nv_bfloat16* weight, int out_width, int in_width,
                           const TextBf16Projection* projections, int count, cudaStream_t stream);

/// Writes width E4M3 bytes and text_fp8_group_count(width) FP32 scales.
/// Each scale is max(amax, 1e-10) / 448. Quantization rounds to nearest even
/// and saturates to the finite E4M3 range.
cudaError_t text_quantize_fp8(const __nv_bfloat16* input, int width, TextFp8Activation output,
                             cudaStream_t stream);

cudaError_t text_quantize_fp8_rows(const __nv_bfloat16* input, int width, int rows,
                                  TextFp8Activation output, cudaStream_t stream);

/// E4M3 [out_width, in_width] weights and BF16 multiply scales laid out as
/// [text_fp8_group_count(out_width), text_fp8_group_count(in_width)].
/// Each input group dot product is accumulated and scaled in FP32. Quantized
/// input can be reused across projections with the same input.
cudaError_t text_fp8_gemv(const std::uint8_t* weight, const __nv_bfloat16* weight_scales,
                          TextFp8ActivationView input, int out_width, int in_width,
                          __nv_bfloat16* output, cudaStream_t stream);

/// Apply shared FP8 weights to independent vectors using GEMV arithmetic.
/// Projections is a host array. Its output spans must be disjoint.
cudaError_t text_fp8_gemm(const std::uint8_t* weight, const __nv_bfloat16* weight_scales,
                          int out_width, int in_width, const TextFp8Projection* projections,
                          int count, cudaStream_t stream);

cudaError_t text_fp8_gemv_rows(const std::uint8_t* weight, const __nv_bfloat16* weight_scales,
                               TextFp8ActivationView input, int out_width, int in_width, int rows,
                               __nv_bfloat16* output, cudaStream_t stream);

/// Computes RMSNorm with eps 1e-6 and FP32 (1 + weight). A null residual
/// selects standalone normalization. A nonnull residual_output receives the
/// BF16-rounded sum, while normalization uses the FP32 sum. Output may alias
/// branch and residual_output may alias residual, each at the same base.
cudaError_t text_rms_norm(const __nv_bfloat16* branch, const __nv_bfloat16* residual,
                         const __nv_bfloat16* weight, int width, __nv_bfloat16* output,
                         __nv_bfloat16* residual_output, cudaStream_t stream);

cudaError_t text_rms_norm_rows(const __nv_bfloat16* branch, const __nv_bfloat16* residual,
                              const __nv_bfloat16* weight, int width, int rows,
                              __nv_bfloat16* output, __nv_bfloat16* residual_output,
                              cudaStream_t stream);

cudaError_t text_rms_norm_pair_rows(const __nv_bfloat16* first, const __nv_bfloat16* second,
                                   const __nv_bfloat16* first_weight,
                                   const __nv_bfloat16* second_weight, int width, int rows,
                                   __nv_bfloat16* output, cudaStream_t stream);

/// Stores BF16(BF16(SiLU(gate)) * up). Output may alias gate or up exactly.
cudaError_t text_swiglu(const __nv_bfloat16* gate, const __nv_bfloat16* up, int width,
                        __nv_bfloat16* output, cudaStream_t stream);

/// Selects the largest BF16 logit, breaking equal scores by lowest token ID.
/// NaNs are ignored. All NaNs produce {-infinity, -1}. Allocate
/// text_argmax_scratch_count(vocab_size) candidates and one output candidate.
/// The caller can also copy the BF16 logits to host for top scores.
cudaError_t text_argmax(const __nv_bfloat16* logits, int vocab_size, TextLogitCandidate* scratch,
                        TextLogitCandidate* output, cudaStream_t stream);

}
