#include "sglang/runtime/execution/verification_row_copy.h"

#include "sglang/runtime/model/qwen_text_model.h"

#include <cstddef>

namespace sglang::decoder {
namespace {

constexpr int threads = 128;
constexpr int row_bytes = qwen_text_model.hidden_size * sizeof(__nv_bfloat16);
constexpr int vectors_per_row = row_bytes / sizeof(uint4);
static_assert(row_bytes % (threads * sizeof(uint4)) == 0);

__global__ void gather_embeddings(const uint4* weight,
                                  const __grid_constant__ VerificationEmbeddingTokens tokens,
                                  const __grid_constant__ VerificationCopyRows output) {
    const int row = blockIdx.x;
    const int column = blockIdx.y * blockDim.x + threadIdx.x;
    const auto token = static_cast<std::size_t>(tokens.id[row]);
    auto* destination = reinterpret_cast<uint4*>(output.destination[row]);
    destination[column] = weight[token * vectors_per_row + column];
}

__global__ void scatter_hidden(const uint4* source,
                               const __grid_constant__ VerificationCopyRows output) {
    const int row = blockIdx.x;
    const int column = blockIdx.y * blockDim.x + threadIdx.x;
    auto* destination = reinterpret_cast<uint4*>(output.destination[row]);
    destination[column] = source[row * vectors_per_row + column];
}

}

cudaError_t gather_verification_embeddings(const __nv_bfloat16* weight, int vocabulary,
                                           const VerificationEmbeddingTokens& tokens,
                                           const VerificationCopyRows& output, int count,
                                           cudaStream_t stream) {
    if (!weight || count < 1 || count > VerificationCopyRows::capacity) {
        return cudaErrorInvalidValue;
    }
    for (int row = 0; row < count; ++row) {
        if (tokens.id[row] < 0 || tokens.id[row] >= vocabulary || !output.destination[row]) {
            return cudaErrorInvalidValue;
        }
    }
    gather_embeddings<<<dim3(count, vectors_per_row / threads), threads, 0, stream>>>(
        reinterpret_cast<const uint4*>(weight), tokens, output);
    return cudaGetLastError();
}

cudaError_t scatter_verification_hidden(const __nv_bfloat16* source,
                                        const VerificationCopyRows& output, int count,
                                        cudaStream_t stream) {
    if (!source || count < 1 || count > VerificationCopyRows::capacity) {
        return cudaErrorInvalidValue;
    }
    scatter_hidden<<<dim3(count, vectors_per_row / threads), threads, 0, stream>>>(
        reinterpret_cast<const uint4*>(source), output);
    return cudaGetLastError();
}

}
