#include "sglang/decoder_internal.h"

#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace {

struct Runtime {
    std::size_t requested = 0;
    std::size_t attempts = 0;
    std::uintptr_t next_address = 0x10000;
    std::unordered_set<const void*> live;
    std::size_t rejected_attempt = 0;
    cudaError_t last = cudaSuccess;
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
cudaError_t cudaGetLastError() { return std::exchange(runtime.last, cudaSuccess); }
const char* cudaGetErrorString(cudaError_t) { return "injected test status"; }

cudaError_t cudaMalloc(void** address, std::size_t bytes) {
    runtime.requested = bytes;
    ++runtime.attempts;
    if (runtime.attempts == runtime.rejected_attempt) {
        runtime.last = cudaErrorMemoryAllocation;
        return runtime.last;
    }
    *address = handle<void*>(runtime.next_address);
    runtime.next_address += 0x10000;
    require(runtime.live.insert(*address).second, "Duplicate allocation");
    return cudaSuccess;
}

cudaError_t cudaFree(void* address) {
    require(runtime.live.erase(address) == 1, "Invalid or duplicate release");
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

void capacity_bounds() {
    using sglang::decoder::Request;
    Request request(0);
    const auto base = request.memory.bytes();
    const auto count = runtime.attempts;
    require(request.work.gqa_scores(262144).capacity == 5120 &&
            request.work.gqa_scores(5119).capacity == 5119 &&
            runtime.attempts == count, "Inline workspace changed");
    struct Case {
        int pages;
        int positions;
        std::size_t bytes;
    };
    for (const auto test : {Case{41, 5248, 503808}, {64, 8192, 786432},
                            {72, 9216, 884736}, {73, 9344, 897024},
                            {2048, 262144, 25165824}}) {
        Request lane(0);
        lane.pages.page_count = test.pages;
        const auto before = runtime.attempts;
        const auto scores = lane.batch_graph_scores();
        require(scores.scores && scores.capacity == test.positions &&
                runtime.requested == test.bytes && lane.memory.bytes() == base + test.bytes,
                "Wrong page-rounded score capacity");
        require(lane.batch_graph_scores().scores == scores.scores &&
                runtime.attempts == before + 1, "Rebinding allocated another score slab");
        lane.memory.close();
        require(runtime.live.find(scores.scores) == runtime.live.end(),
                "Released slab remains live");
    }
}

void binding_rejection(int failed_lane) {
    using namespace sglang::decoder;
    constexpr int width = 4;
    constexpr std::size_t slab_bytes = 503808;
    Allocations execution(0);
    execution.open_stream();
    std::array<Request, width> lanes{Request(0), Request(0), Request(0), Request(0)};
    const auto base = lanes.front().memory.bytes();
    for (auto& lane : lanes) {
        lane.pages.page_count = 41;
    }
    auto bind = [&] {
        for (auto& lane : lanes) {
            if (lane.batch_graph_scores().capacity <= 5120) {
                return false;
            }
        }
        return true;
    };
    const auto attempts = runtime.attempts;
    const auto launches = runtime.launches;
    const auto captures = runtime.captures;
    runtime.rejected_attempt = attempts + failed_lane + 1;
    const auto nop = [] {};
    const auto ready = [] { return true; };
    require(!execution.replay_batch_graph(width, nop, nop, bind), "OOM submitted a graph");
    require(!execution.replay_batch_graph(width, nop, nop, bind), "Remembered OOM submitted");
    require(runtime.launches == launches && runtime.attempts == runtime.rejected_attempt &&
            cudaGetLastError() == cudaSuccess, "OOM relaunched, retried or left an error");
    for (int slot = 0; slot < width; ++slot) {
        const auto bytes = slot < failed_lane ? base + slab_bytes : base;
        require(lanes[slot].memory.bytes() == bytes, "Unexpected partial binding ownership");
    }
    lanes[failed_lane].memory.close();
    require(execution.replay_batch_graph(width, nop, nop, ready), "Short binding lost executable");
    Request replacement(0);
    replacement.pages.page_count = 73;
    require(replacement.batch_graph_scores().capacity == 9344 &&
            replacement.memory.bytes() == base + 897024, "Replacement could not grow");
    require(runtime.captures == captures + 1 && runtime.launches == launches + 1,
            "Rejected binding discarded or disabled its executable");
    std::cout << "Score allocation rejected at packed lane " << failed_lane << '\n';
}

int main() {
    check_driver_statuses();
    capacity_bounds();
    require(runtime.live.empty(), "Capacity test leaked allocations");
    for (int slot = 0; slot < 4; ++slot) {
        binding_rejection(slot);
        require(runtime.live.empty(), "Rejected binding leaked allocations");
    }
    std::cout << "Score bounds, bounded ownership and executable reuse passed\n";
}
