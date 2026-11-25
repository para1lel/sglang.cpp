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
constexpr std::size_t table_bytes = pages * sizeof(std::int32_t);
constexpr std::size_t kv_bytes = 2 * sglang::decoder::full_attention_layers * pages *
    (sglang::mxfp8_payload_bytes_per_page + sglang::mxfp8_scale_bytes_per_page);
std::atomic<void*> page_table{nullptr};
thread_local cudaStream_t table_release_stream = nullptr;
std::atomic_bool injected{false};
std::atomic_bool context_fault{false};
std::atomic_bool reported_context_fault{false};

}

extern "C" cudaError_t __real_cudaMalloc(void** address, std::size_t bytes);
extern "C" cudaError_t __real_cudaFree(void* address);
extern "C" cudaError_t __real_cudaFreeAsync(void* address, cudaStream_t stream);
extern "C" cudaError_t __real_cudaStreamSynchronize(cudaStream_t stream);

extern "C" cudaError_t __wrap_cudaMalloc(void** address, std::size_t bytes) {
    const auto* marker = std::getenv("BATCH_KV_FAULT_MARKER");
    if ((bytes == table_bytes || bytes == kv_bytes) && marker && access(marker, F_OK) == 0) {
        if (bytes == table_bytes) {
            const auto result = __real_cudaMalloc(address, bytes);
            if (result == cudaSuccess) {
                page_table.store(*address);
                std::fprintf(stderr, "test: KV page table allocated bytes=%zu\n", bytes);
            }
            return result;
        }
        if (page_table.load()) {
            *address = nullptr;
            const auto* mode = std::getenv("BATCH_KV_FAULT_MODE");
            const std::string_view fault = mode ? mode : "";
            const auto status = fault == "kv-context"
                ? cudaErrorIllegalAddress : cudaErrorMemoryAllocation;
            context_fault.store(fault == "kv-context" || fault == "kv-drain");
            if (!injected.exchange(true)) {
                std::fprintf(stderr, "test: KV allocation fault bytes=%zu status=%d\n",
                             bytes, static_cast<int>(status));
            }
            return status;
        }
    }
    return __real_cudaMalloc(address, bytes);
}

extern "C" cudaError_t __wrap_cudaFree(void* address) {
    const auto status = __real_cudaFree(address);
    auto* expected = address;
    if (address && status == cudaSuccess &&
        page_table.compare_exchange_strong(expected, nullptr)) {
        std::fprintf(stderr, "test: partial KV page table freed\n");
    }
    return status;
}

extern "C" cudaError_t __wrap_cudaFreeAsync(void* address, cudaStream_t stream) {
    const auto status = __real_cudaFreeAsync(address, stream);
    auto* expected = address;
    if (address && status == cudaSuccess && page_table.compare_exchange_strong(expected, nullptr)) {
        table_release_stream = stream;
    }
    return status;
}

extern "C" cudaError_t __wrap_cudaStreamSynchronize(cudaStream_t stream) {
    const auto status = __real_cudaStreamSynchronize(stream);
    if (status == cudaSuccess && table_release_stream && table_release_stream == stream) {
        table_release_stream = nullptr;
        std::fprintf(stderr, "test: partial KV page table freed\n");
    }
    if (!context_fault.load() || status != cudaSuccess) {
        return status;
    }
    if (!reported_context_fault.exchange(true)) {
        std::fprintf(stderr, "test: terminal context fault persisted at synchronization\n");
    }
    return cudaErrorIllegalAddress;
}
