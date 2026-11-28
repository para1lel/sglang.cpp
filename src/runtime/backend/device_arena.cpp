#include "sglang/runtime/backend/device_arena.h"

#include "sglang/runtime/backend/cuda_resources.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iterator>
#include <limits>
#include <map>
#include <mutex>
#include <utility>
#include <vector>

namespace sglang::decoder {
namespace {

constexpr std::size_t block_alignment = 256;
constexpr std::size_t backing_alignment = std::size_t{2} * 1024 * 1024;
constexpr std::size_t startup_backing = std::size_t{1024} * 1024 * 1024;
constexpr std::size_t driver_headroom = std::size_t{2} * 1024 * 1024 * 1024;

std::size_t align_up(std::size_t bytes, std::size_t alignment) {
    if (bytes > std::numeric_limits<std::size_t>::max() - alignment + 1) {
        throw std::overflow_error("CUDA arena allocation size overflow");
    }
    return (bytes + alignment - 1) / alignment * alignment;
}

std::size_t align_down(std::size_t bytes, std::size_t alignment) {
    return bytes / alignment * alignment;
}

void report(cudaError_t status, const char* operation) noexcept {
    if (status != cudaSuccess) {
        static_cast<void>(std::fputs(operation, stderr));
        static_cast<void>(std::fputs(": ", stderr));
        static_cast<void>(std::fputs(cudaGetErrorString(status), stderr));
        static_cast<void>(std::fputc('\n', stderr));
    }
}

class ArenaDevice {
  public:
    explicit ArenaDevice(int device) noexcept {
        status = cudaGetDevice(&previous_);
        if (status == cudaSuccess && previous_ != device) {
            status = cudaSetDevice(device);
            switched_ = status == cudaSuccess;
        }
    }

    ArenaDevice(const ArenaDevice&) = delete;
    ArenaDevice& operator=(const ArenaDevice&) = delete;
    ArenaDevice(ArenaDevice&&) = delete;
    ArenaDevice& operator=(ArenaDevice&&) = delete;

    ~ArenaDevice() {
        if (switched_) {
            report(cudaSetDevice(previous_), "Restore device after CUDA arena operation");
        }
    }

    cudaError_t status;

  private:
    int previous_ = 0;
    bool switched_ = false;
};

struct Regions {
    void* base = nullptr;
    std::size_t bytes = 0;
    std::size_t available = 0;
    std::map<std::size_t, std::size_t> free;
    std::map<std::size_t, std::size_t> occupied;

    void open(std::size_t size) {
        if (size == 0)
            return;
        free.emplace(0, size);
        check(cudaMalloc(&base, size), "Create CUDA arena backing");
        bytes = available = size;
    }

    void close() const noexcept {
        if (base != nullptr)
            report(cudaFree(base), "Destroy CUDA arena backing");
    }

    bool contains(const void* pointer) const {
        const auto address = reinterpret_cast<std::uintptr_t>(pointer);
        const auto first = reinterpret_cast<std::uintptr_t>(base);
        return address >= first && address - first < bytes;
    }

    std::size_t offset(const void* pointer) const {
        return reinterpret_cast<std::uintptr_t>(pointer) - reinterpret_cast<std::uintptr_t>(base);
    }

    void* take(std::size_t count) {
        const auto block = std::find_if(
            free.begin(), free.end(), [count](const auto& entry) { return count <= entry.second; });
        if (block == free.end())
            return nullptr;
        const auto [start, size] = *block;
        occupied.emplace(start, count);
        auto remainder = free.extract(block);
        if (size > count) {
            remainder.key() += count;
            remainder.mapped() -= count;
            free.insert(std::move(remainder));
        }
        available -= count;
        return static_cast<std::byte*>(base) + start;
    }

    void give(void* pointer) noexcept {
        auto block = occupied.extract(offset(pointer));
        if (block.empty())
            return;
        available += block.mapped();
        auto next = free.lower_bound(block.key());
        if (next != free.end() && block.key() + block.mapped() == next->first) {
            block.mapped() += next->second;
            next = free.erase(next);
        }
        if (next != free.begin()) {
            const auto previous = std::prev(next);
            if (previous->first + previous->second == block.key()) {
                previous->second += block.mapped();
                return;
            }
        }
        free.insert(next, std::move(block));
    }
};

struct Registry {
    std::mutex mutex;
    std::map<int, std::weak_ptr<DeviceArena>> devices;
};

Registry& registry() {
    static Registry entries;
    return entries;
}

}

struct DeviceArena::State {
    struct Deferred {
        void* pointer;
        cudaEvent_t event;
    };

    struct Config {
        int device;
        double utilization;
    };

    explicit State(Config config) : device(config.device), utilization(config.utilization) {
        if (!std::isfinite(utilization) || utilization <= 0.0 || utilization > 1.0) {
            throw std::invalid_argument("GPU memory utilization must be in (0, 1]");
        }
        const ArenaDevice selected(device);
        check(selected.status, "Select device for CUDA arena startup");
        std::size_t free = 0;
        std::size_t total = 0;
        check(cudaMemGetInfo(&free, &total), "Measure CUDA arena budget");
        budget = std::min(static_cast<std::size_t>(static_cast<double>(total) * utilization),
                          free - std::min(free, driver_headroom));
    }

    State(const State&) = delete;
    State& operator=(const State&) = delete;
    State(State&&) = delete;
    State& operator=(State&&) = delete;

    ~State() {
        const ArenaDevice selected(device);
        if (selected.status != cudaSuccess) {
            report(selected.status, "Select device to destroy CUDA arena");
            return;
        }
        if (!deferred.empty() || unsafe_release) {
            const auto status = cudaDeviceSynchronize();
            if (status != cudaSuccess) {
                report(status, "Drain CUDA arena before teardown");
                return;
            }
        }
        for (const auto& entry : deferred) {
            report(cudaEventDestroy(entry.event), "Destroy arena event");
        }
        cache.close();
        general.close();
        for (const auto& region : startup)
            region.close();
    }

    std::size_t remaining() const {
        std::size_t free = 0;
        std::size_t total = 0;
        check(cudaMemGetInfo(&free, &total), "Measure remaining CUDA arena budget");
        return std::min(budget - charged, free - std::min(free, driver_headroom));
    }

    void charge(std::size_t bytes) {
        if (committed)
            throw std::logic_error("CUDA arena has already started");
        if (bytes > remaining()) {
            throw CudaError(cudaErrorMemoryAllocation, "Charge CUDA arena startup allocation");
        }
        charged += bytes;
    }

    void* allocate_startup(std::size_t bytes) {
        if (auto* pointer = take_startup(bytes))
            return pointer;
        const auto minimum = align_up(bytes, backing_alignment);
        const auto limit = align_down(remaining(), backing_alignment);
        if (minimum > limit) {
            throw CudaError(cudaErrorMemoryAllocation, "Lease startup CUDA model storage");
        }
        const auto backing = std::min(limit, std::max(minimum, startup_backing));
        startup.emplace_back();
        try {
            startup.back().open(backing);
        } catch (...) {
            startup.pop_back();
            throw;
        }
        charged += backing;
        return startup.back().take(bytes);
    }

    void* take_startup(std::size_t count) {
        for (auto& region : startup) {
            if (auto* pointer = region.take(count))
                return pointer;
        }
        return nullptr;
    }

    void give(void* pointer) noexcept {
        if (general.contains(pointer)) {
            general.give(pointer);
        } else if (cache.contains(pointer)) {
            cache.give(pointer);
        } else {
            for (auto& region : startup) {
                if (region.contains(pointer)) {
                    region.give(pointer);
                    return;
                }
            }
        }
    }

    void reclaim() {
        if (deferred.empty())
            return;
        const ArenaDevice selected(device);
        check(selected.status, "Select device to reclaim CUDA arena leases");
        for (std::size_t i = 0; i < deferred.size();) {
            const auto entry = deferred[i];
            const auto status = cudaEventQuery(entry.event);
            if (status == cudaErrorNotReady) {
                ++i;
                continue;
            }
            check(status, "Query CUDA arena lease completion");
            check(cudaEventDestroy(entry.event), "Destroy completed CUDA arena event");
            give(entry.pointer);
            deferred[i] = deferred.back();
            deferred.pop_back();
        }
    }

    void* allocate(Regions& region, std::size_t bytes) {
        reclaim();
        const auto count = align_up(bytes, block_alignment);
        if (auto* pointer = region.take(count))
            return pointer;
        if (&region == &general) {
            if (auto* pointer = take_startup(count))
                return pointer;
        }
        throw CudaError(cudaErrorMemoryAllocation, "Lease CUDA arena storage");
    }

    int device;
    const double utilization;
    mutable std::mutex mutex;
    std::size_t budget = 0;
    std::size_t charged = 0;
    bool committed = false;
    bool unsafe_release = false;
    std::vector<int> peer_readers;
    Regions general;
    Regions cache;
    std::vector<Regions> startup;
    std::vector<Deferred> deferred;
};

std::shared_ptr<DeviceArena> DeviceArena::start(int device, double utilization) {
    auto& entries = registry();
    const std::scoped_lock lock(entries.mutex);
    auto& slot = entries.devices[device];
    if (auto arena = slot.lock()) {
        if (arena->state_->utilization != utilization) {
            throw std::invalid_argument(
                "CUDA arena already has a different GPU memory utilization");
        }
        if (!arena->committed())
            throw std::logic_error("CUDA arena startup is still in progress");
        return arena;
    }
    auto arena = std::shared_ptr<DeviceArena>(
        new DeviceArena(std::make_unique<State>(State::Config{device, utilization})));
    slot = arena;
    return arena;
}

std::shared_ptr<DeviceArena> DeviceArena::current(int device) {
    auto& entries = registry();
    const std::scoped_lock lock(entries.mutex);
    const auto slot = entries.devices.find(device);
    if (slot != entries.devices.end()) {
        if (auto arena = slot->second.lock())
            return arena;
    }
    throw std::logic_error("CUDA device has no startup arena");
}

bool DeviceArena::active(int device) {
    auto& entries = registry();
    const std::scoped_lock lock(entries.mutex);
    const auto slot = entries.devices.find(device);
    return slot != entries.devices.end() && !slot->second.expired();
}

DeviceArena::DeviceArena(std::unique_ptr<State> state) : state_(std::move(state)) {
}

DeviceArena::~DeviceArena() = default;

void DeviceArena::charge_startup_bytes(std::size_t bytes) {
    const std::scoped_lock lock(state_->mutex);
    const ArenaDevice selected(state_->device);
    check(selected.status, "Select device for CUDA arena startup charge");
    state_->charge(bytes);
}

void DeviceArena::refund_startup_bytes(std::size_t bytes) noexcept {
    const std::scoped_lock lock(state_->mutex);
    state_->charged -= std::min(state_->charged, bytes);
}

std::size_t DeviceArena::startup_remaining_bytes() const {
    const std::scoped_lock lock(state_->mutex);
    if (state_->committed)
        throw std::logic_error("CUDA arena has already started");
    const ArenaDevice selected(state_->device);
    check(selected.status, "Select device to measure CUDA arena budget");
    return state_->remaining();
}

bool DeviceArena::committed() const {
    const std::scoped_lock lock(state_->mutex);
    return state_->committed;
}

void DeviceArena::commit(std::size_t cache_bytes) {
    const std::scoped_lock lock(state_->mutex);
    if (state_->committed)
        throw std::logic_error("CUDA arena has already started");
    const ArenaDevice selected(state_->device);
    check(selected.status, "Select device to reserve CUDA arena backing");
    const auto amount = align_down(state_->remaining(), backing_alignment);
    const auto cache_amount = align_down(std::min(cache_bytes, amount), backing_alignment);
    try {
        state_->general.open(amount - cache_amount);
        state_->cache.open(cache_amount);
    } catch (...) {
        state_->general.close();
        state_->general = {};
        state_->cache = {};
        throw;
    }
    state_->committed = true;
}

void DeviceArena::enable_peer_access(int reader) {
    const std::scoped_lock lock(state_->mutex);
    const auto& readers = state_->peer_readers;
    if (std::find(readers.begin(), readers.end(), reader) != readers.end())
        return;
    if (reader == state_->device)
        return;
    int access = 0;
    check(cudaDeviceCanAccessPeer(&access, reader, state_->device), "Check CUDA arena peer access");
    if (access == 0)
        return;
    const ArenaDevice selected(reader);
    check(selected.status, "Select reader for CUDA arena peer access");
    const auto status = cudaDeviceEnablePeerAccess(state_->device, 0);
    if (status == cudaErrorPeerAccessAlreadyEnabled || status == cudaErrorTooManyPeers ||
        status == cudaErrorPeerAccessUnsupported || status == cudaErrorMemoryAllocation) {
        const auto recorded = cudaGetLastError();
        if (recorded != status)
            check(recorded, "Check state after CUDA arena peer setup");
        if (status != cudaErrorPeerAccessAlreadyEnabled)
            return;
    } else {
        check(status, "Enable CUDA arena peer access");
    }
    state_->peer_readers.push_back(reader);
}

bool DeviceArena::allows_peer_access(int reader) const {
    const std::scoped_lock lock(state_->mutex);
    const auto& readers = state_->peer_readers;
    return std::find(readers.begin(), readers.end(), reader) != readers.end();
}

std::size_t DeviceArena::budget_bytes() const {
    return state_->budget;
}

void* DeviceArena::allocate(std::size_t bytes) {
    if (bytes == 0)
        throw CudaError(cudaErrorInvalidValue, "Lease empty CUDA arena storage");
    const std::scoped_lock lock(state_->mutex);
    if (state_->committed)
        return state_->allocate(state_->general, bytes);
    const ArenaDevice selected(state_->device);
    check(selected.status, "Select device to allocate startup CUDA storage");
    return state_->allocate_startup(align_up(bytes, block_alignment));
}

void* DeviceArena::allocate_cache(std::size_t bytes) {
    if (bytes == 0)
        throw CudaError(cudaErrorInvalidValue, "Lease empty CUDA cache storage");
    const std::scoped_lock lock(state_->mutex);
    if (!state_->committed)
        throw std::logic_error("CUDA arena cache backing is not initialized");
    return state_->allocate(state_->cache, bytes);
}

void DeviceArena::release(void* pointer) noexcept {
    if (pointer == nullptr)
        return;
    const std::scoped_lock lock(state_->mutex);
    state_->give(pointer);
}

cudaError_t DeviceArena::release_after(void* pointer, cudaStream_t stream) noexcept {
    if (pointer == nullptr)
        return cudaSuccess;
    const std::scoped_lock lock(state_->mutex);
    const ArenaDevice selected(state_->device);
    if (selected.status != cudaSuccess) {
        state_->unsafe_release = true;
        return selected.status;
    }
    cudaEvent_t event = nullptr;
    auto status = cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
    if (status == cudaSuccess)
        status = cudaEventRecord(event, stream);
    if (status == cudaSuccess) {
        try {
            state_->deferred.push_back({pointer, event});
            return cudaSuccess;
        } catch (...) {
            status = cudaErrorMemoryAllocation;
        }
    }
    if (event != nullptr)
        report(cudaEventDestroy(event), "Destroy unqueued CUDA arena event");
    if (cudaStreamSynchronize(stream) == cudaSuccess)
        state_->give(pointer);
    else
        state_->unsafe_release = true;
    return status;
}

void DeviceArena::quarantine() noexcept {
    const std::scoped_lock lock(state_->mutex);
    state_->unsafe_release = true;
}

void DeviceArena::reclaim() {
    const std::scoped_lock lock(state_->mutex);
    state_->reclaim();
}

std::size_t DeviceArena::capacity() const {
    const std::scoped_lock lock(state_->mutex);
    return state_->charged + state_->general.bytes + state_->cache.bytes;
}

std::size_t DeviceArena::available_bytes() {
    const std::scoped_lock lock(state_->mutex);
    state_->reclaim();
    auto available = state_->general.available;
    for (const auto& region : state_->startup)
        available += region.available;
    return available;
}

std::size_t DeviceArena::cache_capacity() const {
    const std::scoped_lock lock(state_->mutex);
    return state_->cache.bytes;
}

std::size_t DeviceArena::cache_available_bytes() {
    const std::scoped_lock lock(state_->mutex);
    state_->reclaim();
    return state_->cache.available;
}

ArenaAddress DeviceArena::export_address(const void* pointer, std::size_t bytes) const {
    const std::scoped_lock lock(state_->mutex);
    if (bytes == 0 || !state_->cache.contains(pointer)) {
        throw std::invalid_argument("CUDA IPC address is outside the cache backing");
    }
    const auto offset = state_->cache.offset(pointer);
    auto block = state_->cache.occupied.upper_bound(offset);
    if (block == state_->cache.occupied.begin()) {
        throw std::invalid_argument("CUDA IPC address has no cache lease");
    }
    --block;
    const auto inside = offset - block->first;
    if (inside >= block->second || bytes > block->second - inside) {
        throw std::invalid_argument("CUDA IPC span exceeds its cache lease");
    }
    return {state_->cache.base, state_->cache.bytes, offset};
}

}
