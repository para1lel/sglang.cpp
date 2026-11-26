#include "sglang/decoder_prefill.h"

#include "sglang/prefill_fp8.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <mutex>
#include <utility>

namespace sglang::decoder {

struct PrefillPool {
    PrefillPool(int ordinal, bool large, bool transient)
        : device(ordinal), large(large), transient(transient) {}

    ~PrefillPool() {
        if (handle) {
            cudaMemPoolDestroy(handle);
        }
    }

    const int device;
    const bool large;
    const bool transient;
    cudaMemPool_t handle = nullptr;
    void* pending = nullptr;
    bool retains_backing = false;
    PrefillPool* next_retained = nullptr;
    std::unique_ptr<PrefillPool> next;
};

}

namespace sglang {
namespace {

using namespace decoder;

void check_prefill_pool_result(cudaError_t status, bool can_decline, const char* operation) {
    if (status == cudaSuccess) return;
    const auto recorded = cudaGetLastError();
    if (recorded != status && !CudaError::terminal(status)) check(recorded, operation);
    if (!can_decline) check(status, operation);
}

class PrefillPools {
public:
    void configure_retention(PrefillPool& pool) {
        {
            const std::lock_guard<std::mutex> lock(mutex_);
            for (auto* owner = retained_; owner; owner = owner->next_retained) {
                if (owner->device == pool.device && owner->large == pool.large) {
                    return;
                }
            }
            pool.next_retained = retained_;
            retained_ = &pool;
        }

        std::uint64_t threshold = (pool.large ? 1024 : 32) * 1024 * 1024;
        const auto status =
            cudaMemPoolSetAttribute(pool.handle, cudaMemPoolAttrReleaseThreshold, &threshold);
        if (status == cudaSuccess) {
            pool.retains_backing = true;
            return;
        }

        {
            const std::lock_guard<std::mutex> lock(mutex_);
            auto** link = &retained_;
            while (*link != &pool) {
                link = &(*link)->next_retained;
            }
            *link = pool.next_retained;
            pool.next_retained = nullptr;
        }
        check_prefill_pool_result(status, status == cudaErrorNotSupported,
                                  "Configure prefill pool retention");
    }

    std::unique_ptr<PrefillPool> take_idle(int device, bool large, bool transient) {
        const std::lock_guard<std::mutex> lock(mutex_);
        if (!transient) {
            if (auto pool = take(idle_, device, large, false, true)) return pool;
        }
        return take(idle_, device, large, transient);
    }

    std::unique_ptr<PrefillPool> take_pending(int device) {
        if (!has_pending_.load(std::memory_order_relaxed)) {
            return {};
        }
        const std::lock_guard<std::mutex> lock(mutex_);
        std::unique_ptr<PrefillPool> pending;
        while (auto pool = take(pending_, device)) {
            pool->next = std::move(pending);
            pending = std::move(pool);
        }
        has_pending_.store(pending_ != nullptr, std::memory_order_relaxed);
        return pending;
    }

    void put(std::unique_ptr<PrefillPool> pool) {
        const std::lock_guard<std::mutex> lock(mutex_);
        auto& list = pool->pending ? pending_ : idle_;
        pool->next = std::move(list);
        list = std::move(pool);
        if (pending_) {
            has_pending_.store(true, std::memory_order_relaxed);
        }
    }

private:
    static std::unique_ptr<PrefillPool> take(std::unique_ptr<PrefillPool>& list, int device,
                                             std::optional<bool> large = std::nullopt,
                                             std::optional<bool> transient = std::nullopt,
                                             bool must_retain_backing = false) {
        auto* link = &list;
        while (*link && ((*link)->device != device ||
                        (large && (*link)->large != *large) ||
                        (transient && (*link)->transient != *transient) ||
                        (must_retain_backing && !(*link)->retains_backing))) {
            link = &(*link)->next;
        }
        auto pool = std::move(*link);
        if (pool) {
            *link = std::move(pool->next);
        }
        return pool;
    }

    std::mutex mutex_;
    PrefillPool* retained_ = nullptr;
    std::atomic<bool> has_pending_{false};
    std::unique_ptr<PrefillPool> idle_;
    std::unique_ptr<PrefillPool> pending_;
};

PrefillPools& prefill_pools() {
    static PrefillPools pools;
    return pools;
}

void recover_prefill_pool(PrefillPool& pool) noexcept {
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess || device != pool.device ||
        cudaDeviceSynchronize() != cudaSuccess) {
        return;
    }
    std::uint64_t used = 0;
    if (cudaMemPoolGetAttribute(pool.handle, cudaMemPoolAttrUsedMemCurrent, &used) != cudaSuccess) {
        return;
    }
    if (used == 0 ||
        (cudaFreeAsync(pool.pending, cudaStreamPerThread) == cudaSuccess &&
         cudaStreamSynchronize(cudaStreamPerThread) == cudaSuccess)) {
        pool.pending = nullptr;
    }
}

bool prefill_allocation_unavailable(cudaError_t status) {
    return status == cudaErrorMemoryAllocation || status == cudaErrorNotSupported;
}

}

void decoder::reclaim_prefill_workspaces(int device) {
    auto& pools = prefill_pools();
    auto pending = pools.take_pending(device);
    while (pending) {
        auto pool = std::move(pending);
        pending = std::move(pool->next);
        recover_prefill_pool(*pool);
        pools.put(std::move(pool));
    }
}

PrefillWorkspace::PrefillWorkspace(cudaStream_t stream, int capacity)
    : PrefillWorkspace(stream, capacity, Retention::existing) {}

PrefillWorkspace::PrefillWorkspace(cudaStream_t stream, int capacity, Retention retention)
    : PrefillWorkspace(stream, capacity, retention, 0) {}

PrefillWorkspace::PrefillWorkspace(cudaStream_t stream, int capacity, int staged_visible)
    : PrefillWorkspace(stream, capacity, Retention::existing, staged_visible) {}

PrefillWorkspace::PrefillWorkspace(cudaStream_t stream, int capacity, Retention retention,
                                     int staged_visible)
    : stream_(stream), capacity_(capacity) {
    if (!stream || capacity < 1 || capacity > grouped_prefill_max_rows) {
        throw std::invalid_argument("Invalid decoder prefill workspace");
    }
    if (capacity == 1) {
        return;
    }
    check(cudaGetLastError(), "Check CUDA state before allocating prefill workspace");
    int device = 0;
    check(cudaGetDevice(&device), "Get CUDA device for prefill workspace");
    const bool transient = retention == Retention::transient;
    if (!transient) reclaim_prefill_workspaces(device);
    const bool large = capacity > prefill_chunk_rows;
    auto& pools = prefill_pools();
    auto pool = pools.take_idle(device, large, transient);
    if (!pool) {
        pool = std::make_unique<PrefillPool>(device, large, transient);
        cudaMemPoolProps props{};
        props.allocType = cudaMemAllocationTypePinned;
        props.location.type = cudaMemLocationTypeDevice;
        props.location.id = device;
        const auto created = cudaMemPoolCreate(&pool->handle, &props);
        check_prefill_pool_result(created, prefill_allocation_unavailable(created),
                                  "Create prefill workspace pool");
        if (created != cudaSuccess) return;
        int disabled = 0;
        const auto configured = cudaMemPoolSetAttribute(
            pool->handle, cudaMemPoolReuseAllowInternalDependencies, &disabled);
        check_prefill_pool_result(configured, false, "Configure prefill workspace pool");
        if (!transient) pools.configure_retention(*pool);
    }
    const auto recycle = [&pools](PrefillPool* idle) noexcept {
        pools.put(std::unique_ptr<PrefillPool>(idle));
    };
    std::unique_ptr<PrefillPool, decltype(recycle)> reservation(pool.release(), recycle);
    const auto base = prefill_workspace_bytes(capacity);
    constexpr std::size_t alignment = 128;
    const auto offset = (base + alignment - 1) / alignment * alignment;
    const auto visible = std::clamp(staged_visible, 0, staged_serial_max_tokens);
    const auto pages = (visible + mxfp8_page_size - 1) / mxfp8_page_size;
    const auto page_heads = std::size_t(pages) * gqa_rank_geometry.kv_heads;
    const auto plane = page_heads * mxfp8_page_size * mxfp8_head_dim * sizeof(__nv_bfloat16);
    const auto values = plane + page_heads * sizeof(int);
    auto bytes = plane ? offset + plane + values : base;
    void* data = nullptr;
    check(cudaGetLastError(), "Check CUDA state before allocating prefill workspace");
    auto status = cudaMallocFromPoolAsync(&data, bytes, reservation->handle, stream);
    if (plane && prefill_allocation_unavailable(status)) {
        check_prefill_pool_result(status, true, "Reserve long prefill staging");
        bytes = base;
        status = cudaMallocFromPoolAsync(&data, bytes, reservation->handle, stream);
    }
    if (status != cudaSuccess) {
        check_prefill_pool_result(status, prefill_allocation_unavailable(status),
                                  "Allocate prefill workspace");
        return;
    }
    pool_.reset(reservation.release());
    data_ = data;
    bytes_ = bytes;
    layout_bytes_ = base;
    if (bytes != base) {
        auto* keys = static_cast<std::byte*>(data) + offset;
        staging_ = {keys, plane, keys + plane, values};
    }
}

PrefillWorkspace::~PrefillWorkspace() {
    if (release_status_ == cudaSuccess) {
        const auto status = release();
        if (status != cudaSuccess) {
            std::fprintf(stderr, "Release prefill workspace: %s\n", cudaGetErrorString(status));
        }
    }
}

cudaError_t PrefillWorkspace::release() noexcept {
    if (data_ && release_status_ == cudaSuccess) {
        release_status_ = cudaFreeAsync(data_, stream_);
        if (release_status_ != cudaSuccess) {
            pool_->pending = data_;
            if (!pool_->transient) recover_prefill_pool(*pool_);
        }
        prefill_pools().put(std::move(pool_));
        data_ = nullptr;
        staging_ = {};
    }
    stream_ = nullptr;
    return release_status_;
}

void PrefillWorkspace::close() {
    check(release(), "Release prefill workspace");
}

}
