#pragma once

#include "sglang/decoder.h"

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

namespace sglang::decoder {

struct SamplingParameters {
    double temperature;
    double top_p;
    int top_k;
    std::uint64_t seed;
    double min_p = 0.0;
};

struct OutputPenalties {
    float* frequency_totals = nullptr;
    std::uint8_t* seen = nullptr;
    float frequency = 0.0f;
    float presence = 0.0f;
    double repetition = 1.0;
};

struct LogitMasks {
    const std::uint32_t* allowed = nullptr;
    const std::uint32_t* denied = nullptr;
};

struct GeneratedLogprobReadback;

struct RepetitionSelection {
    const SamplingParameters* sampling = nullptr;
    int position = 0;
    int* token = nullptr;
    int top = -1;
    GeneratedLogprobReadback* logprobs = nullptr;
    DecoderLogit* debug = nullptr;
    int debug_count = 0;
};

SamplingParameters request_sampling(const DecoderOptions& options);
std::size_t sampling_workspace_bytes();
cudaError_t sample_logits(const __nv_bfloat16* scores, int vocabulary, int* token,
                          SamplingParameters parameters, int position, void* workspace,
                          cudaStream_t stream);
std::size_t biased_sampling_workspace_bytes(int vocabulary);
cudaError_t sample_logits(const float* scores, int vocabulary, int* token,
                          SamplingParameters parameters, int position, void* workspace,
                          std::size_t workspace_bytes, cudaStream_t stream);
cudaError_t adjust_logits(const __nv_bfloat16* source, float* scores, int vocabulary,
                          const DecoderLogitBias* bias, int count, LogitMasks masks,
                          OutputPenalties penalties, int previous_token, cudaStream_t stream);
std::size_t repetition_workspace_bytes(int vocabulary, bool ranked);
cudaError_t select_repetition_logits(
    const __nv_bfloat16* source, int vocabulary, const DecoderLogitBias* bias, int bias_count,
    LogitMasks masks, OutputPenalties penalties, int previous_token,
    RepetitionSelection selection, void* workspace, std::size_t bytes, cudaStream_t stream);
cudaError_t check_allowed_logit(const float* scores, int* token, cudaStream_t stream);

}
