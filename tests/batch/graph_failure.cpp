#include "sglang/decoder_verify.h"

#include <dlfcn.h>

#include <cstdio>
#include <cstdlib>
#include <string_view>

namespace {

bool failed = false;
bool pending = false;
bool fail_drain = false;
int launches = 0;

bool enabled(std::string_view boundary) {
    const auto* mode = std::getenv("BATCH_GRAPH_FAILURE");
    return mode && mode == boundary;
}

bool inject(std::string_view boundary) {
    if (!failed && enabled(boundary)) {
        failed = true;
        std::fprintf(stderr, "inject: %.*s\n", static_cast<int>(boundary.size()), boundary.data());
        return true;
    }
    return false;
}

}

extern "C" cudaError_t cudaMallocHost(void** address, std::size_t bytes) {
    static const auto real =
        reinterpret_cast<decltype(&cudaMallocHost)>(dlsym(RTLD_NEXT, "cudaMallocHost"));
    if (bytes == sizeof(sglang::decoder::VerificationGraphBinding) && inject("prepare")) {
        return cudaErrorMemoryAllocation;
    }
    return real(address, bytes);
}

extern "C" cudaError_t cudaMemcpyAsync(void* destination, const void* source, std::size_t bytes,
                                       cudaMemcpyKind kind, cudaStream_t stream) {
    static const auto real =
        reinterpret_cast<decltype(&cudaMemcpyAsync)>(dlsym(RTLD_NEXT, "cudaMemcpyAsync"));
    if (launches == 1 && kind == cudaMemcpyHostToDevice &&
        bytes == sizeof(sglang::decoder::VerificationGraphBinding) && inject("upload")) {
        return cudaErrorInvalidValue;
    }
    return real(destination, source, bytes, kind, stream);
}

extern "C" cudaError_t cudaStreamBeginCapture(cudaStream_t stream, cudaStreamCaptureMode mode) {
    static const auto real = reinterpret_cast<decltype(&cudaStreamBeginCapture)>(
        dlsym(RTLD_NEXT, "cudaStreamBeginCapture"));
    if (inject("begin")) {
        return cudaErrorStreamCaptureUnsupported;
    }
    return real(stream, mode);
}

extern "C" cudaError_t cudaGraphInstantiate(cudaGraphExec_t* executable,
                                            cudaGraph_t graph, unsigned long long flags) {
    static const auto real = reinterpret_cast<decltype(&cudaGraphInstantiate)>(
        dlsym(RTLD_NEXT, "cudaGraphInstantiate"));
    if (inject("instantiate")) {
        return cudaErrorNotSupported;
    }
    return real(executable, graph, flags);
}

extern "C" cudaError_t cudaGraphLaunch(cudaGraphExec_t executable, cudaStream_t stream) {
    static const auto real =
        reinterpret_cast<decltype(&cudaGraphLaunch)>(dlsym(RTLD_NEXT, "cudaGraphLaunch"));
    if (++launches == 2 && inject("launch")) {
        return cudaErrorInvalidResourceHandle;
    }
    const auto* mode = std::getenv("BATCH_GRAPH_FAILURE");
    if (launches == 2 && mode && std::string_view(mode) == "exec-cleanup") {
        return cudaErrorInvalidResourceHandle;
    }
    const auto status = real(executable, stream);
    pending = status == cudaSuccess;
    if (pending && launches == 2 && inject("submitted")) {
        return cudaErrorInvalidResourceHandle;
    }
    return status;
}

extern "C" cudaError_t cudaGraphDestroy(cudaGraph_t graph) {
    static const auto real =
        reinterpret_cast<decltype(&cudaGraphDestroy)>(dlsym(RTLD_NEXT, "cudaGraphDestroy"));
    if (inject("cleanup")) {
        return cudaErrorInvalidResourceHandle;
    }
    return real(graph);
}

extern "C" cudaError_t cudaGraphExecDestroy(cudaGraphExec_t executable) {
    static const auto real = reinterpret_cast<decltype(&cudaGraphExecDestroy)>(
        dlsym(RTLD_NEXT, "cudaGraphExecDestroy"));
    if (inject("exec-cleanup")) {
        return cudaErrorInvalidResourceHandle;
    }
    return real(executable);
}

extern "C" cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
    static const auto real = reinterpret_cast<decltype(&cudaStreamSynchronize)>(
        dlsym(RTLD_NEXT, "cudaStreamSynchronize"));
    if (fail_drain) {
        fail_drain = false;
        std::fputs("inject: terminal drain failure\n", stderr);
        return cudaErrorIllegalAddress;
    }
    if (pending && inject("hard-drain")) {
        pending = false;
        fail_drain = true;
        return cudaErrorInvalidValue;
    }
    if (pending && inject("hard")) {
        pending = false;
        return cudaErrorIllegalAddress;
    }
    return real(stream);
}
