#include "sglang/decoder_internal.h"
#include "sglang/runtime/backend/device_arena.h"

#include <cuda_runtime_api.h>
#include <unistd.h>

#include <atomic>
#include <iostream>
#include <cstdlib>
#include <string_view>

namespace {

using sglang::decoder::Allocations;
using sglang::decoder::CudaError;
using sglang::decoder::DeviceArena;

constexpr std::size_t pages = 9;
constexpr std::size_t table_bytes = pages * sizeof(std::int32_t);
constexpr std::size_t history_elements =
    sglang::decoder::gdn_layers * sglang::decoder::gdn_history_elements;
std::atomic<void*> page_table{nullptr};
std::atomic_bool injected{false};
std::atomic_bool context_fault{false};
std::atomic_bool reported_context_fault{false};

}

extern "C" void*
real_arena_lease(DeviceArena* arena,
                 std::size_t bytes) asm("__real__ZN6sglang7decoder11DeviceArena8allocateEm");
extern "C" void*
wrap_arena_lease(DeviceArena* arena,
                 std::size_t bytes) asm("__wrap__ZN6sglang7decoder11DeviceArena8allocateEm");
extern "C" __nv_bfloat16* real_request_bf16_lease(Allocations* owner, std::size_t count) asm(
    "__real__ZN6sglang7decoder11Allocations8allocateI13__nv_bfloat16EEPT_m");
extern "C" __nv_bfloat16* wrap_request_bf16_lease(Allocations* owner, std::size_t count) asm(
    "__wrap__ZN6sglang7decoder11Allocations8allocateI13__nv_bfloat16EEPT_m");
extern "C" void real_arena_release(DeviceArena* arena, void* address) noexcept
    asm("__real__ZN6sglang7decoder11DeviceArena7releaseEPv");
extern "C" void wrap_arena_release(DeviceArena* arena, void* address) noexcept
    asm("__wrap__ZN6sglang7decoder11DeviceArena7releaseEPv");
extern "C" cudaError_t __real_cudaStreamSynchronize(cudaStream_t stream);

void* wrap_arena_lease(DeviceArena* arena, std::size_t bytes) {
    auto* address = real_arena_lease(arena, bytes);
    const auto* marker = std::getenv("BATCH_KV_FAULT_MARKER");
    if (bytes == table_bytes && marker != nullptr && access(marker, F_OK) == 0) {
        page_table.store(address);
        std::cerr << "test: KV page table leased bytes=" << bytes << '\n';
    }
    return address;
}

__nv_bfloat16* wrap_request_bf16_lease(Allocations* owner, std::size_t count) {
    const auto* marker = std::getenv("BATCH_KV_FAULT_MARKER");
    if (count == history_elements && page_table.load() != nullptr && marker != nullptr &&
        access(marker, F_OK) == 0) {
        const auto* mode = std::getenv("BATCH_KV_FAULT_MODE");
        const std::string_view fault = mode != nullptr ? mode : "";
        const auto status =
            fault == "kv-context" ? cudaErrorIllegalAddress : cudaErrorMemoryAllocation;
        context_fault.store(fault == "kv-context" || fault == "kv-drain");
        if (!injected.exchange(true)) {
            std::cerr << "test: GDN history lease fault bytes=" << count * sizeof(__nv_bfloat16)
                      << " status=" << static_cast<int>(status) << '\n';
        }
        throw CudaError(status, "Test GDN history arena lease");
    }
    return real_request_bf16_lease(owner, count);
}

void wrap_arena_release(DeviceArena* arena, void* address) noexcept {
    auto* expected = address;
    const bool tracked =
        address != nullptr && page_table.compare_exchange_strong(expected, nullptr);
    real_arena_release(arena, address);
    if (tracked)
        std::cerr << "test: partial KV page table returned to arena\n";
}

extern "C" cudaError_t __wrap_cudaStreamSynchronize(cudaStream_t stream) {
    const auto status = __real_cudaStreamSynchronize(stream);
    if (!context_fault.load() || status != cudaSuccess) {
        return status;
    }
    if (!reported_context_fault.exchange(true)) {
        std::cerr << "test: terminal context fault persisted at synchronization\n";
    }
    return cudaErrorIllegalAddress;
}
