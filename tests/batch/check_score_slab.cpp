#include "sglang/decoder_internal.h"
#include "sglang/runtime/backend/device_arena.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>

namespace {

struct Runtime {
    int allocations = 0;
    int frees = 0;
    void* backing = nullptr;
    int captures = 0;
    int launches = 0;
} runtime;

void require(bool condition, const char* description) {
    if (!condition) {
        throw std::runtime_error(description);
    }
}

void check_driver_statuses() {
    using sglang::decoder::CudaError;
    struct Case {
        CUresult driver;
        cudaError_t expected;
        bool terminal;
    };
    constexpr std::array cases{
        Case{CUDA_SUCCESS, cudaSuccess, false},
        Case{CUDA_ERROR_INVALID_VALUE, cudaErrorInvalidValue, false},
        Case{CUDA_ERROR_OUT_OF_MEMORY, cudaErrorMemoryAllocation, false},
        Case{CUDA_ERROR_NOT_INITIALIZED, cudaErrorInitializationError, false},
        Case{CUDA_ERROR_DEINITIALIZED, cudaErrorCudartUnloading, true},
        Case{CUDA_ERROR_INVALID_CONTEXT, cudaErrorIncompatibleDriverContext, false},
        Case{CUDA_ERROR_ECC_UNCORRECTABLE, cudaErrorECCUncorrectable, true},
        Case{CUDA_ERROR_INVALID_HANDLE, cudaErrorInvalidResourceHandle, false},
        Case{CUDA_ERROR_ILLEGAL_ADDRESS, cudaErrorIllegalAddress, true},
        Case{CUDA_ERROR_LAUNCH_TIMEOUT, cudaErrorLaunchTimeout, true},
        Case{CUDA_ERROR_CONTEXT_IS_DESTROYED, cudaErrorContextIsDestroyed, true},
        Case{CUDA_ERROR_ASSERT, cudaErrorAssert, true},
        Case{CUDA_ERROR_HARDWARE_STACK_ERROR, cudaErrorHardwareStackError, true},
        Case{CUDA_ERROR_ILLEGAL_INSTRUCTION, cudaErrorIllegalInstruction, true},
        Case{CUDA_ERROR_MISALIGNED_ADDRESS, cudaErrorMisalignedAddress, true},
        Case{CUDA_ERROR_INVALID_ADDRESS_SPACE, cudaErrorInvalidAddressSpace, true},
        Case{CUDA_ERROR_INVALID_PC, cudaErrorInvalidPc, true},
        Case{CUDA_ERROR_LAUNCH_FAILED, cudaErrorLaunchFailure, true},
        Case{CUDA_ERROR_NOT_SUPPORTED, cudaErrorNotSupported, false},
        Case{CUDA_ERROR_UNKNOWN, cudaErrorUnknown, false},
        Case{CUDA_ERROR_NOT_FOUND, cudaErrorUnknown, false},
    };
    for (const auto& test : cases) {
        const auto status = CudaError::from_driver(test.driver);
        require(status == test.expected, "Wrong CUDA status for Driver result");
        require(CudaError::terminal(status) == test.terminal, "Wrong CUDA terminal classification");
    }
}

template <typename Handle>
Handle handle(std::uintptr_t value) {
    return reinterpret_cast<Handle>(value);
}

}

extern "C" {

cudaError_t cudaGetDevice(int* device) { *device = 0; return cudaSuccess; }
cudaError_t cudaSetDevice(int) { return cudaSuccess; }
cudaError_t cudaGetLastError() {
    return cudaSuccess;
}
const char* cudaGetErrorString(cudaError_t) { return "injected test status"; }

cudaError_t cudaMemGetInfo(std::size_t* free, std::size_t* total) {
    *free = *total = std::size_t{3} << 30;
    return cudaSuccess;
}

cudaError_t cudaMalloc(void** address, std::size_t bytes) {
    require(runtime.allocations == 0 && bytes != 0, "Invalid physical arena allocation");
    ++runtime.allocations;
    runtime.backing = *address = handle<void*>(0x100000);
    return cudaSuccess;
}

cudaError_t cudaFree(void* address) {
    require(runtime.frees == 0 && address == runtime.backing, "Invalid physical release");
    ++runtime.frees;
    return cudaSuccess;
}

cudaError_t cudaFreeHost(void*) {
    throw std::runtime_error("Unexpected pinned host release");
}

cudaError_t cudaFreeAsync(void*, cudaStream_t) {
    throw std::runtime_error("Unexpected asynchronous release");
}

cudaError_t cudaStreamCreateWithFlags(cudaStream_t* stream, unsigned int) {
    *stream = handle<cudaStream_t>(1);
    return cudaSuccess;
}
cudaError_t cudaStreamBeginCapture(cudaStream_t, cudaStreamCaptureMode) {
    ++runtime.captures;
    return cudaSuccess;
}
cudaError_t cudaStreamEndCapture(cudaStream_t, cudaGraph_t* graph) {
    *graph = handle<cudaGraph_t>(2);
    return cudaSuccess;
}
cudaError_t cudaGraphInstantiate(cudaGraphExec_t* executable, cudaGraph_t, unsigned long long) {
    *executable = handle<cudaGraphExec_t>(3);
    return cudaSuccess;
}
cudaError_t cudaGraphLaunch(cudaGraphExec_t executable, cudaStream_t) {
    require(executable == handle<cudaGraphExec_t>(3), "Unexpected executable");
    ++runtime.launches;
    return cudaSuccess;
}
cudaError_t cudaStreamIsCapturing(cudaStream_t, cudaStreamCaptureStatus* value) {
    *value = cudaStreamCaptureStatusNone;
    return cudaSuccess;
}
cudaError_t cudaGraphDestroy(cudaGraph_t) { return cudaSuccess; }
cudaError_t cudaGraphExecDestroy(cudaGraphExec_t) { return cudaSuccess; }
cudaError_t cudaStreamSynchronize(cudaStream_t) { return cudaSuccess; }
cudaError_t cudaStreamDestroy(cudaStream_t) { return cudaSuccess; }

}

static void capacity_bounds(sglang::decoder::DeviceArena& arena) {
    using namespace sglang::decoder;
    Request request(0);
    const auto base = request.memory.bytes();
    const auto available = arena.available_bytes();
    const auto* scratch = reinterpret_cast<float*>(request.work.logits);
    for (const int length : {1, 5119, 5120, 5248, 8192, 32768, 262144}) {
        const auto scores = request.work.gqa_scores(length);
        require(scores.scores == scratch && scores.capacity == std::min(length, 5120),
                "Exact attention lost its bounded logits backing");
    }
    require(request.memory.bytes() == base && arena.available_bytes() == available,
            "Binding exact scores changed request storage");
}

static void binding_rejection(sglang::decoder::DeviceArena& arena, int failed_lane) {
    using namespace sglang::decoder;
    constexpr int width = 4;
    Allocations execution(0);
    execution.open_stream();
    std::array<Request, width> lanes{Request(0), Request(0), Request(0), Request(0)};
    std::array<Workspace*, width> bindings{};
    for (int slot = 0; slot < width; ++slot)
        bindings[slot] = &lanes[slot].work;
    const auto base = lanes.front().memory.bytes();
    const auto available = arena.available_bytes();
    auto* const reserved = arena.allocate(available);
    const auto launches = runtime.launches;
    const auto captures = runtime.captures;
    int visited = 0;
    auto bind = [&] {
        visited = 0;
        for (const auto* workspace : bindings) {
            ++visited;
            if (!workspace)
                return false;
            require(workspace->gqa_scores(5120).scores ==
                        reinterpret_cast<float*>(workspace->logits),
                    "Wrong short score backing");
        }
        return true;
    };
    const auto nop = [] {};
    bindings[failed_lane] = nullptr;
    for (int attempt = 0; attempt < 2; ++attempt) {
        require(!execution.replay_batch_graph(width, nop, nop, bind) && visited == failed_lane + 1,
                "Missing graph binding advanced beyond the rejected lane");
    }
    require(runtime.launches == launches && arena.available_bytes() == 0 &&
                cudaGetLastError() == cudaSuccess,
            "Rejected binding submitted, allocated storage, or left a CUDA error");
    for (const auto& lane : lanes) {
        require(lane.memory.bytes() == base, "Rejected binding changed request ownership");
    }
    arena.release(reserved);
    lanes[failed_lane].memory.close();
    Request replacement(0);
    bindings[failed_lane] = &replacement.work;
    require(replacement.memory.bytes() == base && arena.available_bytes() == available,
            "Replacing the rejected request changed its storage budget");
    require(execution.replay_batch_graph(width, nop, nop, bind) && visited == width,
            "Replacing the rejected lane did not replay");
    require(runtime.captures == captures + 1 && runtime.launches == launches + 1,
            "Rejected binding discarded its reusable graph");
    std::cout << "Score allocation rejected at packed lane " << failed_lane << '\n';
}

int main() {
    using namespace sglang::decoder;
    check_driver_statuses();
    {
        const auto arena = DeviceArena::start(0, default_gpu_memory_utilization);
        arena->commit();
        const auto available = arena->available_bytes();
        capacity_bounds(*arena);
        require(arena->available_bytes() == available, "Capacity test leaked leases");
        for (int slot = 0; slot < 4; ++slot) {
            binding_rejection(*arena, slot);
            require(arena->available_bytes() == available, "Rejected binding leaked leases");
        }
        require(runtime.allocations == 1 && runtime.frees == 0, "Serving changed physical storage");
    }
    require(runtime.frees == 1, "Arena backing did not release on shutdown");
    std::cout << "Score bounds, bounded ownership and executable reuse passed\n";
}
