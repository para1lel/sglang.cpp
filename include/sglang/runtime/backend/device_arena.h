#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <memory>

namespace sglang::decoder {

inline constexpr double default_gpu_memory_utilization = 0.88;

struct ArenaAddress {
    void* base;
    std::size_t bytes;
    std::size_t offset;
};

class DeviceArena {
  public:
    static std::shared_ptr<DeviceArena> start(int device, double utilization);
    static std::shared_ptr<DeviceArena> current(int device);
    static bool active(int device);

    ~DeviceArena();
    DeviceArena(const DeviceArena&) = delete;
    DeviceArena& operator=(const DeviceArena&) = delete;
    DeviceArena(DeviceArena&&) = delete;
    DeviceArena& operator=(DeviceArena&&) = delete;

    void charge_startup_bytes(std::size_t bytes);
    void refund_startup_bytes(std::size_t bytes) noexcept;
    std::size_t startup_remaining_bytes() const;
    bool committed() const;
    void commit(std::size_t cache_bytes = 0);
    void enable_peer_access(int reader);
    bool allows_peer_access(int reader) const;
    void* allocate(std::size_t bytes);
    void* allocate_cache(std::size_t bytes);
    void release(void* pointer) noexcept;
    cudaError_t release_after(void* pointer, cudaStream_t stream) noexcept;
    void quarantine() noexcept;
    void reclaim();

    std::size_t capacity() const;
    std::size_t budget_bytes() const;
    std::size_t available_bytes();
    std::size_t cache_capacity() const;
    std::size_t cache_available_bytes();
    ArenaAddress export_address(const void* pointer, std::size_t bytes = 1) const;

  private:
    struct State;
    explicit DeviceArena(std::unique_ptr<State> state);
    std::unique_ptr<State> state_;
};

}
