#include "sglang/decoder_constraint_cuda.h"

#include <cmath>

namespace sglang::decoder {
namespace {

constexpr int block_size = 128;

__global__ void gather_logits(const __nv_bfloat16* source, const int* ids,
                              __nv_bfloat16* logits, int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        logits[index] = source[ids[index]];
    }
}

__global__ void mask_logits(const __nv_bfloat16* source, const std::uint32_t* mask,
                            __nv_bfloat16* logits, int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        const bool allowed = (mask[index / 32] >> (index % 32)) & 1U;
        logits[index] = allowed ? source[index] : __ushort_as_bfloat16(0xff80);
    }
}

__global__ void remap_token(const __nv_bfloat16* logits, const int* ids, int count, int* token) {
    const int index = *token;
    *token = index >= 0 && index < count && isfinite(__bfloat162float(logits[index]))
        ? (ids ? ids[index] : index) : -1;
}

}

cudaError_t mask_constrained_logits(const __nv_bfloat16* source, const std::uint32_t* mask,
                                   __nv_bfloat16* logits, int count, cudaStream_t stream) {
    mask_logits<<<(count + block_size - 1) / block_size, block_size, 0, stream>>>(
        source, mask, logits, count);
    return cudaGetLastError();
}

cudaError_t gather_constrained_logits(const __nv_bfloat16* source, const int* ids,
                                     __nv_bfloat16* logits, int count, cudaStream_t stream) {
    gather_logits<<<(count + block_size - 1) / block_size, block_size, 0, stream>>>(
        source, ids, logits, count);
    return cudaGetLastError();
}

cudaError_t remap_constrained_token(const __nv_bfloat16* logits, const int* ids,
                                   int count, int* token, cudaStream_t stream) {
    remap_token<<<1, 1, 0, stream>>>(logits, ids, count, token);
    return cudaGetLastError();
}

}
