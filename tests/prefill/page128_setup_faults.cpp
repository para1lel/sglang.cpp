#include "sglang/text/prefill_attn_sm100.h"

#include <cuda.h>
#include <cudaTypedefs.h>

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <utility>

namespace {

struct Faults {
    cudaError_t pending = cudaSuccess;
    cudaError_t lookup = cudaSuccess;
    bool missing_encoder = false;
    CUresult encode = CUDA_SUCCESS;
    int encode_call = 0;
    cudaError_t attribute = cudaSuccess;
    cudaError_t post_rejection_error = cudaSuccess;
    cudaError_t producer = cudaSuccess;
    cudaError_t release = cudaSuccess;
};

struct Calls {
    int lookup = 0;
    int encode = 0;
    int bulk_attribute = 0;
    int allocations = 0;
    int initializations = 0;
    unsigned producer_threads = 0;
    int launches = 0;
    int releases = 0;
};

Faults faults;
Calls calls;
int portable_calls = 0;
constexpr int position = 8192;
constexpr int rows = 1024;
constexpr int page_count = (position + rows) / sglang::mxfp8_page_size;
alignas(32) std::uint8_t payload[32]{};
alignas(16) std::uint8_t scales[16]{};
alignas(16) __nv_bfloat16 queries[8]{};
alignas(16) __nv_bfloat16 output[8]{};
int pages[page_count]{};
alignas(256) std::uint8_t scratch[128 * 1024 * 1024]{};

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

CUresult encode_map(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*,
                    const cuuint64_t*, const cuuint64_t*, const cuuint32_t*,
                    const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                    CUtensorMapL2promotion, CUtensorMapFloatOOBfill) {
    return ++calls.encode == faults.encode_call ? faults.encode : CUDA_SUCCESS;
}

cudaError_t runtime_rejection(cudaError_t status) {
    if (status != cudaSuccess) {
        faults.pending = faults.post_rejection_error == cudaSuccess ?
            status : faults.post_rejection_error;
    }
    return status;
}

void run(const char* label, Faults injected, cudaError_t expected, unsigned threads,
         int lookup, int encodes, int bulk_attribute, bool malformed = false,
         const char* mode = "sm100-tma") {
    std::cout << label << std::endl;
    faults = injected;
    calls = {};
    portable_calls = 0;
    require(mode ? setenv("SGLANG_PREFILL_ATTN", mode, 1) == 0 :
                   unsetenv("SGLANG_PREFILL_ATTN") == 0, "Could not set prefill mode");
    const sglang::Mxfp8KvCache cache{
        payload, payload, malformed ? nullptr : scales, scales, page_count};
    const sglang::Mxfp8PageTable table{pages, page_count};
    const auto status = sglang::decoder::tensor_core_model_prefill_gqa(
        cache, table, position, rows, queries, output, nullptr);
    if (status != expected) {
        std::cerr << "Expected " << cudaGetErrorName(expected) << ", got "
                  << cudaGetErrorName(status) << '\n';
        throw std::runtime_error("Unexpected model prefill status");
    }
    require(portable_calls == (mode == nullptr ? 1 : 0), "Unexpected portable dispatch");
    require(calls.lookup == lookup && calls.encode == encodes &&
            calls.bulk_attribute == bulk_attribute, "Unexpected TMA setup calls");
    require(calls.producer_threads == threads, "Unexpected producer");
    const int submissions = threads == 0 ? 0 : 1;
    require(calls.allocations == submissions && calls.initializations == submissions &&
            calls.releases == submissions, "Unexpected scratch or stream work");
    const int launches = injected.producer == cudaSuccess ? 2 : 1;
    require(calls.launches == submissions * launches, "Unexpected retry or finish submission");
    if (status == cudaSuccess) {
        require(faults.pending == cudaSuccess, "Setup rejection left a stale CUDA error");
    }
}

void check_encoder() {
    for (const auto status : {CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_NOT_SUPPORTED}) {
        Faults injected;
        injected.encode = status;
        injected.encode_call = 3;
        run("Recoverable descriptor rejection after partial encoding",
            injected, cudaSuccess, 544, 1, 3, 0);
    }
    const std::pair<CUresult, cudaError_t> failures[] = {
        {CUDA_ERROR_INVALID_CONTEXT, cudaErrorIncompatibleDriverContext},
        {CUDA_ERROR_CONTEXT_IS_DESTROYED, cudaErrorContextIsDestroyed},
        {CUDA_ERROR_NOT_INITIALIZED, cudaErrorInitializationError},
        {CUDA_ERROR_DEINITIALIZED, cudaErrorCudartUnloading},
        {CUDA_ERROR_OUT_OF_MEMORY, cudaErrorMemoryAllocation},
        {CUDA_ERROR_ILLEGAL_ADDRESS, cudaErrorIllegalAddress},
        {CUDA_ERROR_LAUNCH_FAILED, cudaErrorLaunchFailure},
        {CUDA_ERROR_UNKNOWN, cudaErrorUnknown},
    };
    for (const auto& failure : failures) {
        Faults injected;
        injected.encode = failure.first;
        injected.encode_call = 2;
        run("Unexpected encoder failure stops before stream work",
            injected, failure.second, 0, 1, 2, 0);
    }
}

void check_runtime() {
    Faults injected;
    injected.missing_encoder = true;
    run("Missing encoder selects split", injected, cudaSuccess, 544, 1, 0, 0);
    injected = {};
    injected.lookup = cudaErrorNotSupported;
    run("Unsupported encoder lookup selects split", injected, cudaSuccess, 544, 1, 0, 0);
    injected.post_rejection_error = cudaErrorIllegalAddress;
    run("Different lookup last error propagates", injected, cudaErrorIllegalAddress, 0, 1, 0, 0);
    injected = {};
    injected.lookup = cudaErrorInitializationError;
    run("Unexpected encoder lookup error propagates",
        injected, cudaErrorInitializationError, 0, 1, 0, 0);
    for (const auto status : {cudaErrorInvalidValue, cudaErrorNotSupported,
                              cudaErrorInvalidDeviceFunction, cudaErrorNoKernelImageForDevice}) {
        injected = {};
        injected.attribute = status;
        run("Recoverable TMA attribute rejection selects split",
            injected, cudaSuccess, 544, 1, 4, 1);
    }
    injected = {};
    injected.attribute = cudaErrorInvalidValue;
    injected.post_rejection_error = cudaErrorLaunchFailure;
    run("Different attribute last error propagates", injected, cudaErrorLaunchFailure, 0, 1, 4, 1);
    injected = {};
    injected.attribute = cudaErrorMemoryAllocation;
    run("Unexpected TMA attribute error propagates",
        injected, cudaErrorMemoryAllocation, 0, 1, 4, 1);
    injected = {};
    injected.pending = cudaErrorInvalidValue;
    run("Pre-existing runtime error survives recoverable status classification",
        injected, cudaErrorInvalidValue, 0, 0, 0, 0);
    require(faults.pending == cudaErrorInvalidValue, "Pre-existing CUDA error was cleared");
}

}

namespace sglang::decoder {

cudaError_t tensor_core_multirow_prefill_gqa(
    Mxfp8KvCache, Mxfp8PageTable, int, int, const __nv_bfloat16*, __nv_bfloat16*, cudaStream_t) {
    ++portable_calls;
    return cudaSuccess;
}

}

extern "C" cudaError_t __wrap_cudaGetDevice(int* device) {
    *device = 0;
    return cudaSuccess;
}

extern "C" cudaError_t __wrap_cudaDeviceGetAttribute(int* value, cudaDeviceAttr attribute, int) {
    switch (attribute) {
        case cudaDevAttrComputeCapabilityMajor: *value = 10; break;
        case cudaDevAttrComputeCapabilityMinor: *value = 0; break;
        case cudaDevAttrMaxSharedMemoryPerBlockOptin: *value = 232448; break;
        case cudaDevAttrMemoryPoolsSupported: *value = 1; break;
        case cudaDevAttrMultiProcessorCount: *value = 148; break;
        default: return cudaErrorInvalidValue;
    }
    return cudaSuccess;
}

extern "C" cudaError_t __wrap_cudaGetDriverEntryPointByVersion(
    const char*, void** entry, unsigned int, unsigned long long, cudaDriverEntryPointQueryResult*) {
    ++calls.lookup;
    *entry = faults.missing_encoder ? nullptr :
        reinterpret_cast<void*>(static_cast<PFN_cuTensorMapEncodeTiled_v12000>(encode_map));
    return runtime_rejection(faults.lookup);
}

extern "C" cudaError_t __wrap_cudaFuncSetAttribute(const void*, cudaFuncAttribute, int value) {
    if (value == 230400) {
        ++calls.bulk_attribute;
        return runtime_rejection(faults.attribute);
    }
    return cudaSuccess;
}

extern "C" cudaError_t __wrap_cudaPeekAtLastError() {
    return faults.pending;
}

extern "C" cudaError_t __wrap_cudaGetLastError() {
    return std::exchange(faults.pending, cudaSuccess);
}

extern "C" cudaError_t __wrap_cudaMallocAsync(void** pointer, std::size_t bytes, cudaStream_t) {
    ++calls.allocations;
    require(bytes <= sizeof(scratch), "Test scratch is too small");
    *pointer = scratch;
    return cudaSuccess;
}

extern "C" cudaError_t __wrap_cudaMemsetAsync(void*, int, std::size_t, cudaStream_t) {
    ++calls.initializations;
    return cudaSuccess;
}

extern "C" cudaError_t __wrap_cudaLaunchKernel(
    const void*, dim3, dim3 block, void**, std::size_t shared, cudaStream_t) {
    ++calls.launches;
    if (shared != 0) {
        calls.producer_threads = block.x;
        return faults.producer;
    }
    return cudaSuccess;
}

extern "C" cudaError_t __wrap_cudaFreeAsync(void* pointer, cudaStream_t) {
    ++calls.releases;
    require(pointer == scratch, "Unexpected scratch release");
    return faults.release;
}

int main() try {
    run("Default mode reaches the portable dispatcher",
        {}, cudaSuccess, 0, 0, 0, 0, false, nullptr);
    run("Original opt-in leaves TMA setup alone", {}, cudaSuccess, 544, 0, 0, 0, false, "sm100");
    run("Aligned opt-in reaches TMA setup and producer", {}, cudaSuccess, 576, 1, 4, 1);
    run("Malformed cache fails before TMA setup", {}, cudaErrorInvalidValue, 0, 0, 0, 0, true);
    check_encoder();
    check_runtime();
    Faults injected;
    injected.producer = cudaErrorLaunchFailure;
    injected.release = cudaErrorInvalidValue;
    run("First stream error wins after TMA submission without split retry",
        injected, cudaErrorLaunchFailure, 576, 1, 4, 1);
    std::cout << "Page-128 setup fault controls passed\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
