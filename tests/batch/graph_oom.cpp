#include "sglang/decoder_batch.h"
#include "sglang/decoder_internal.h"

#include <dlfcn.h>

#include <cstdio>
#include <cstdlib>
#include <string_view>

namespace {

constexpr std::size_t binding_bytes =
    sizeof(sglang::decoder::BatchDecodeBinding) * sglang::decoder_batch_max_lanes;
constexpr std::size_t first_workspace_bytes =
    sizeof(sglang::decoder::Bf16) * sglang::decoder::hidden_size;
constexpr std::size_t long_score_bytes = 5248 * sglang::mxfp8_query_heads * sizeof(float);
void* bindings = nullptr;
bool injected = false;

bool enabled(std::string_view name) {
    const auto* mode = std::getenv("BATCH_GRAPH_FAILURE");
    return mode && mode == name;
}

}

extern "C" cudaError_t cudaMalloc(void** address, std::size_t bytes) {
    static const auto real =
        reinterpret_cast<decltype(&cudaMalloc)>(dlsym(RTLD_NEXT, "cudaMalloc"));
    if (!injected && enabled("score-oom") && bytes == long_score_bytes) {
        injected = true;
        std::fprintf(stderr, "inject: score-oom\n");
        return cudaErrorMemoryAllocation;
    }
    if (!injected && enabled("scratch-oom") && bindings && bytes == first_workspace_bytes) {
        injected = true;
        std::fprintf(stderr, "inject: scratch-oom after allocating graph bindings\n");
        return cudaErrorMemoryAllocation;
    }
    if (!injected) {
        bindings = nullptr;
    }
    const auto status = real(address, bytes);
    if (!injected && status == cudaSuccess && bytes == binding_bytes && enabled("scratch-oom")) {
        bindings = *address;
    }
    return status;
}

extern "C" cudaError_t cudaFree(void* address) {
    static const auto real = reinterpret_cast<decltype(&cudaFree)>(dlsym(RTLD_NEXT, "cudaFree"));
    const auto status = real(address);
    if (address && address == bindings && status == cudaSuccess) {
        if (injected) {
            std::fprintf(stderr, "released partial graph scratch bindings\n");
        }
        bindings = nullptr;
    }
    return status;
}

extern "C" cudaError_t cudaGraphInstantiate(cudaGraphExec_t* executable, cudaGraph_t graph,
                                            unsigned long long flags) {
    static const auto real = reinterpret_cast<decltype(&cudaGraphInstantiate)>(
        dlsym(RTLD_NEXT, "cudaGraphInstantiate"));
    if (!injected && enabled("instantiate-oom")) {
        injected = true;
        std::fprintf(stderr, "inject: instantiate-oom\n");
        return cudaErrorMemoryAllocation;
    }
    return real(executable, graph, flags);
}
