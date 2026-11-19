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
};

SamplingParameters request_sampling(const DecoderOptions& options);
std::size_t sampling_workspace_bytes();
cudaError_t sample_logits(const __nv_bfloat16* scores, int vocabulary, int* token,
                          SamplingParameters parameters, int position, void* workspace,
                          cudaStream_t stream);

}
