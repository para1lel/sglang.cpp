#include "sglang/decoder_internal.h"
#include "sglang/mxfp8_gqa.h"

#include <cuda_runtime_api.h>
#include <unistd.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <string_view>

namespace {

constexpr std::size_t pages = 1024 / sglang::mxfp8_page_size;
constexpr std::size_t payload_bytes =
    sglang::decoder::full_attention_layers * pages * sglang::mxfp8_payload_bytes_per_page;
std::atomic<unsigned int> payloads{0};
std::atomic<void*> partial_payload{nullptr};
std::atomic_bool context_fault{false};
std::atomic_bool reported_context_fault{false};

}

extern "C" cudaError_t __real_cudaMalloc(void** address, std::size_t bytes);
extern "C" cudaError_t __real_cudaFree(void* address);
extern "C" cudaError_t __real_cudaStreamSynchronize(cudaStream_t stream);

extern "C" cudaError_t __wrap_cudaMalloc(void** address, std::size_t bytes) {
    const auto* marker = std::getenv("BATCH_KV_FAULT_MARKER");
    if (bytes == payload_bytes && marker && access(marker, F_OK) == 0) {
        const auto match = ++payloads;
        if (match == 1) {
            const auto result = __real_cudaMalloc(address, bytes);
            if (result == cudaSuccess) {
                partial_payload.store(*address);
                std::fprintf(stderr, "test: first KV payload allocated bytes=%zu\n", bytes);
            }
            return result;
        }
        if (match == 2 && partial_payload.load()) {
            *address = nullptr;
            const auto* mode = std::getenv("BATCH_KV_FAULT_MODE");
            const std::string_view fault = mode ? mode : "";
            const auto status = fault == "kv-context"
                ? cudaErrorIllegalAddress : cudaErrorMemoryAllocation;
            context_fault.store(fault == "kv-context" || fault == "kv-drain");
            std::fprintf(stderr, "test: second KV payload fault bytes=%zu status=%d\n",
                         bytes, static_cast<int>(status));
            return status;
        }
    }
    return __real_cudaMalloc(address, bytes);
}

extern "C" cudaError_t __wrap_cudaFree(void* address) {
    const auto status = __real_cudaFree(address);
    auto* expected = address;
    if (address && status == cudaSuccess &&
        partial_payload.compare_exchange_strong(expected, nullptr)) {
        std::fprintf(stderr, "test: partial KV payload freed\n");
    }
    return status;
}

extern "C" cudaError_t __wrap_cudaStreamSynchronize(cudaStream_t stream) {
    const auto status = __real_cudaStreamSynchronize(stream);
    if (!context_fault.load() || status != cudaSuccess) {
        return status;
    }
    if (!reported_context_fault.exchange(true)) {
        std::fprintf(stderr, "test: terminal context fault persisted at synchronization\n");
    }
    return cudaErrorIllegalAddress;
}
