#include "sglang/decoder_batch.h"
#include "sglang/decoder_internal.h"

#include <dlfcn.h>

#include <cstdio>
#include <cstdlib>
#include <string_view>

namespace {

constexpr std::size_t binding_bytes =
    sizeof(sglang::decoder::BatchDecodeBinding) * sglang::decoder_batch_max_graph_lanes;
constexpr std::size_t first_workspace_bytes =
    sizeof(sglang::decoder::Bf16) * sglang::decoder::hidden_size;
constexpr std::size_t long_score_bytes = 5248 * sglang::mxfp8_query_heads * sizeof(float);
constexpr std::size_t packed_hidden_bytes =
    first_workspace_bytes * sglang::decoder_batch_max_lanes;
void* bindings = nullptr;
void* packed_hidden = nullptr;
bool injected = false;

bool enabled(std::string_view name) {
    const auto* mode = std::getenv("BATCH_GRAPH_FAILURE");
    return mode && mode == name;
}

}

extern "C" cudaError_t cudaMalloc(void** address, std::size_t bytes) {
    static const auto real =
        reinterpret_cast<decltype(&cudaMalloc)>(dlsym(RTLD_NEXT, "cudaMalloc"));
    if (enabled("packed-boundary")) {
        std::fprintf(stderr, "cudaMalloc %zu\n", bytes);
    }
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
    const bool packed = !injected && enabled("packed-oom") && bytes == packed_hidden_bytes;
    if (packed && packed_hidden) {
        injected = true;
        std::fprintf(stderr, "inject: packed-oom after allocating packed hidden rows\n");
        return cudaErrorMemoryAllocation;
    }
    if (!injected) {
        bindings = nullptr;
    }
    const auto status = real(address, bytes);
    if (!injected && status == cudaSuccess && bytes == binding_bytes && enabled("scratch-oom")) {
        bindings = *address;
    }
    if (packed && status == cudaSuccess) {
        packed_hidden = *address;
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
    if (address && address == packed_hidden && status == cudaSuccess) {
        if (injected) {
            std::fprintf(stderr, "released partial packed hidden rows\n");
        }
        packed_hidden = nullptr;
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
