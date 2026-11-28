#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

namespace sglang::decoder {

struct VerificationCopyRows {
    static constexpr int capacity = 160;
    __nv_bfloat16* destination[capacity];
};

struct VerificationEmbeddingTokens {
    int id[VerificationCopyRows::capacity];
};

cudaError_t gather_verification_embeddings(const __nv_bfloat16* weight, int vocabulary,
                                           const VerificationEmbeddingTokens& tokens,
                                           const VerificationCopyRows& output, int count,
                                           cudaStream_t stream);
cudaError_t gather_verification_embeddings(const __nv_bfloat16* weight, const int* device_tokens,
                                           const VerificationCopyRows& output, int count,
                                           cudaStream_t stream);

cudaError_t scatter_verification_hidden(const __nv_bfloat16* source,
                                        const VerificationCopyRows& output, int count,
                                        cudaStream_t stream);

}
