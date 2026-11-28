#pragma once

#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

namespace sglang {

inline constexpr int text_fp8_group_size = 128;
inline constexpr int text_fp8_max_batch = 5;
inline constexpr int text_argmax_tile = 1024;
inline constexpr int text_decode_max_targets = 3;

int text_fp8_group_count(int width);

struct TextFp8Activation {
    std::uint8_t* values;
    float* scales;
};

struct TextFp8ActivationView {
    const std::uint8_t* values;
    const float* scales;
};

struct TextFp8QuantizeRow {
    const __nv_bfloat16* input;
    TextFp8Activation output;
};

struct TextFp8Projection {
    TextFp8ActivationView input;
    __nv_bfloat16* output;
};

struct TextFp8DecodeTarget {
    const std::uint8_t* weight;
    const __nv_bfloat16* weight_scales;
    int out_width;
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

int text_argmax_scratch_count(int vocab_size);

cudaError_t text_embedding(const __nv_bfloat16* weight, int vocab_size, int width, int token_id,
                           __nv_bfloat16* output, cudaStream_t stream);

inline constexpr int text_embedding_max_rows = 128;
cudaError_t text_embedding_rows(const __nv_bfloat16* weight, int vocab_size, int width,
                                const int* host_token_ids, int rows, __nv_bfloat16* output,
                                cudaStream_t stream);

cudaError_t text_bf16_gemv(const __nv_bfloat16* weight, const __nv_bfloat16* input,
                           int out_width, int in_width, __nv_bfloat16* output,
                           cudaStream_t stream);

cudaError_t text_bf16_gemv_rows(const __nv_bfloat16* weight, const __nv_bfloat16* input,
                                int out_width, int in_width, int rows, __nv_bfloat16* output,
                                cudaStream_t stream);
cudaError_t text_bf16_seed_fc(const __nv_bfloat16* weight, const __nv_bfloat16* input,
                             int out_width, int in_width, int rows, __nv_bfloat16* output,
                             cudaStream_t stream);
cudaError_t text_bf16_gemm(const __nv_bfloat16* weight, int out_width, int in_width,
                           const TextBf16Projection* projections, int count, cudaStream_t stream);

cudaError_t text_quantize_fp8(const __nv_bfloat16* input, int width, TextFp8Activation output,
                             cudaStream_t stream);

cudaError_t text_quantize_fp8(const TextFp8QuantizeRow* rows, int count, int width,
                              cudaStream_t stream);

cudaError_t text_quantize_fp8_rows(const __nv_bfloat16* input, int width, int rows,
                                  TextFp8Activation output, cudaStream_t stream);

cudaError_t text_fp8_gemv(const std::uint8_t* weight, const __nv_bfloat16* weight_scales,
                          TextFp8ActivationView input, int out_width, int in_width,
                          __nv_bfloat16* output, cudaStream_t stream);

cudaError_t text_fp8_gemm(const std::uint8_t* weight, const __nv_bfloat16* weight_scales,
                          int out_width, int in_width, const TextFp8Projection* projections,
                          int count, cudaStream_t stream);

cudaError_t text_fp8_gemv_rows(const std::uint8_t* weight, const __nv_bfloat16* weight_scales,
                               TextFp8ActivationView input, int out_width, int in_width, int rows,
                               __nv_bfloat16* output, cudaStream_t stream);

cudaError_t text_fp8_decode_rows(const std::uint8_t* weight, const __nv_bfloat16* weight_scales,
                                 TextFp8ActivationView input, int out_width, int in_width,
                                 int rows, __nv_bfloat16* output, cudaStream_t stream);

cudaError_t text_fp8_decode_rows_fused(const TextFp8DecodeTarget* targets, int count,
                                       TextFp8ActivationView input, int in_width, int rows,
                                       cudaStream_t stream);

cudaError_t text_bf16_decode_rows(const __nv_bfloat16* weight, const __nv_bfloat16* input,
                                  int out_width, int in_width, int rows, __nv_bfloat16* output,
                                  cudaStream_t stream);

cudaError_t text_bf16_decode_gates(const __nv_bfloat16* weight_b,
                                   const __nv_bfloat16* weight_a,
                                   const __nv_bfloat16* input, int rows,
                                   __nv_bfloat16* output_b, __nv_bfloat16* output_a,
                                   cudaStream_t stream);

cudaError_t text_rms_norm(const __nv_bfloat16* branch, const __nv_bfloat16* residual,
                         const __nv_bfloat16* weight, int width, __nv_bfloat16* output,
                         __nv_bfloat16* residual_output, cudaStream_t stream);

struct TextModelStateRow {
    __nv_bfloat16* hidden = nullptr;
    __nv_bfloat16* residual = nullptr;
};

cudaError_t text_final_rms_bound(const __nv_bfloat16* branch, const __nv_bfloat16* residual,
                                const __nv_bfloat16* weight, __nv_bfloat16* output,
                                const TextModelStateRow* device_state, cudaStream_t stream);

cudaError_t text_rms_norm_rows(const __nv_bfloat16* branch, const __nv_bfloat16* residual,
                              const __nv_bfloat16* weight, int width, int rows,
                              __nv_bfloat16* output, __nv_bfloat16* residual_output,
                              cudaStream_t stream);

cudaError_t text_rms_norm_pair_rows(const __nv_bfloat16* first, const __nv_bfloat16* second,
                                   const __nv_bfloat16* first_weight,
                                   const __nv_bfloat16* second_weight, int width, int rows,
                                   __nv_bfloat16* output, cudaStream_t stream);

cudaError_t text_swiglu(const __nv_bfloat16* gate, const __nv_bfloat16* up, int width,
                        __nv_bfloat16* output, cudaStream_t stream);

cudaError_t text_argmax(const __nv_bfloat16* logits, int vocab_size, TextLogitCandidate* scratch,
                        TextLogitCandidate* output, cudaStream_t stream);
cudaError_t text_argmax(const float* logits, int vocab_size, TextLogitCandidate* scratch,
                        TextLogitCandidate* output, cudaStream_t stream);
cudaError_t text_argmax_finite(const __nv_bfloat16* logits, int vocab_size,
                               TextLogitCandidate* scratch, TextLogitCandidate* output,
                               cudaStream_t stream);

}
