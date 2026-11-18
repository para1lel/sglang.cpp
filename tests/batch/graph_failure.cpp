#include <cuda_runtime_api.h>
#include <dlfcn.h>

#include <cstdio>
#include <cstdlib>
#include <string_view>

namespace {

bool failed = false;
bool pending = false;
int launches = 0;

bool inject(std::string_view boundary) {
    const auto* mode = std::getenv("BATCH_GRAPH_FAILURE");
    if (!failed && mode && mode == boundary) {
        failed = true;
        std::fprintf(stderr, "inject: %.*s\n", static_cast<int>(boundary.size()), boundary.data());
        return true;
    }
    return false;
}

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
    const auto status = real(executable, stream);
    pending = status == cudaSuccess;
    return status;
}

extern "C" cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
    static const auto real = reinterpret_cast<decltype(&cudaStreamSynchronize)>(
        dlsym(RTLD_NEXT, "cudaStreamSynchronize"));
    if (pending && inject("hard")) {
        pending = false;
        return cudaErrorIllegalAddress;
    }
    return real(stream);
}
