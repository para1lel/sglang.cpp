#include "sglang/decoder_internal.h"

#include <cupti.h>
#include <dlfcn.h>

#include <cstdio>
#include <cstdlib>

namespace {

constexpr std::size_t packed_hidden_bytes = sizeof(sglang::decoder::Bf16) *
    sglang::decoder::hidden_size * sglang::decoder_batch_max_lanes;
void* previous = nullptr;
bool prepared = false;
bool injected = false;

}

extern "C" cudaError_t cudaMemset(void* buffer, int value, std::size_t bytes) {
    static const auto clear = reinterpret_cast<decltype(&cudaMemset)>(
        dlsym(RTLD_NEXT, "cudaMemset"));
    const auto status = clear(buffer, value, bytes);
    if (!injected && status == cudaSuccess && value == 0 && bytes == packed_hidden_bytes) {
        prepared = prepared || (previous && previous != buffer);
        previous = buffer;
    }
    return status;
}

extern "C" cudaError_t cudaDeviceSynchronize() {
    static const auto synchronize = reinterpret_cast<decltype(&cudaDeviceSynchronize)>(
        dlsym(RTLD_NEXT, "cudaDeviceSynchronize"));
    const auto status = synchronize();
    if (!prepared || injected || status != cudaSuccess) {
        return status;
    }
    injected = true;
    std::uint64_t timestamp = 0;
    if (cuptiGetTimestamp(&timestamp) != CUPTI_SUCCESS || timestamp == 0) {
        std::abort();
    }
    std::fprintf(stderr, "inject: packed setup illegal address at %llu\n",
                 static_cast<unsigned long long>(timestamp));
    return cudaErrorIllegalAddress;
}
