#include "sglang/decoder_batch.h"
#include "sglang/decoder_internal.h"
#include "sglang/runtime/backend/cuda_resources.h"
#include "sglang/runtime/backend/device_arena.h"
#include "sglang/runtime/cache/lmcache_group.h"
#include "sglang/runtime/cache/prefix_residency.h"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <future>
#include <iostream>
#include <stdexcept>

using namespace sglang;
using namespace sglang::decoder;

namespace {

struct Faults {
    cudaError_t record = cudaSuccess;
    cudaError_t query = cudaSuccess;
    int physical_releases = 0;
};

Faults& release_faults() {
    thread_local Faults faults;
    return faults;
}

struct BackupGate {
    std::atomic_bool armed{false};
    std::promise<void> entered;
    std::promise<void> resume;
    std::future<void> resumed = resume.get_future();
};

BackupGate& backup_gate() {
    static BackupGate gate;
    return gate;
}

void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

constexpr std::size_t bytes = 1 << 20;

struct Detached {
    Detached(int device, bool resident)
        : arena(DeviceArena::current(device)), retired(device, resident, true),
          original(device, &retired), available(arena->available_bytes()),
          pointer(original.allocate<std::byte>(bytes)), first(original.retain_block(pointer)),
          last(original.retain_block(pointer)) {
        require(retired.unavailable_bytes() == 0, "Charged a live block twice");
        original.recycle_after_completion();
        require(retired.unavailable_bytes() == bytes && retired.pending_bytes() == 0 &&
                    arena->available_bytes() == available - bytes,
                "Missing detached charge or released a retained block");
        first.reset();
        retired.clear();
        require(retired.unavailable_bytes() == bytes, "Drain dropped the remaining lease");
    }

    void finish() {
        last.reset();
        retired.clear();
        require(retired.unavailable_bytes() == 0 && retired.pending_bytes() == 0 &&
                    retired.failed_release_bytes() == 0 && arena->available_bytes() == available,
                "Final alias did not return exactly one arena lease");
        retired.check_releases();
    }

    std::shared_ptr<DeviceArena> arena;
    RequestBlockCache retired;
    Allocations original;
    std::size_t available;
    std::byte* pointer;
    std::shared_ptr<const void> first;
    std::shared_ptr<const void> last;
};

void check_detached_releases(int devices) {
    for (int rank = 0; rank < devices; ++rank) {
        for (const bool resident : {false, true}) {
            const int releases = release_faults().physical_releases;
            Detached held(rank, resident);
            auto* competing = held.arena->allocate(bytes);
            require(competing != held.pointer, "Reused a block with an outstanding alias");
            held.arena->release(competing);
            held.finish();
            require(release_faults().physical_releases == releases,
                    "Detached lease called physical CUDA free");
        }
    }
    if (devices == 2) {
        const int releases = release_faults().physical_releases;
        Detached rank0(0, true);
        Detached rank1(1, true);
        rank1.finish();
        require(rank0.retired.unavailable_bytes() == bytes &&
                    rank0.arena->available_bytes() == rank0.available - bytes,
                "Rank 1 completion acknowledged rank 0 storage");
        rank0.finish();
        require(release_faults().physical_releases == releases,
                "Rank teardown called physical CUDA free");
    }
}

void check_fenced_release(int device) {
    const DeviceScope selected(device);
    const auto arena = DeviceArena::current(device);
    Allocations stream_owner(device);
    stream_owner.open_stream();
    auto* const stream = stream_owner.stream();
    const auto available = arena->available_bytes();
    auto& calls = release_faults();
    const int releases = calls.physical_releases;

    auto* pointer = arena->allocate(bytes);
    check(cudaMemsetAsync(pointer, 27, bytes, stream), "Queue work on leased allocation");
    check(arena->release_after(pointer, stream), "Fence arena release");
    calls.query = cudaErrorNotReady;
    require(arena->available_bytes() == available - bytes, "Premature fenced lease reuse");
    auto* competing = arena->allocate(bytes);
    require(competing != pointer, "Reused incomplete lease address");
    arena->release(competing);

    calls.query = cudaErrorInvalidResourceHandle;
    auto actual = cudaSuccess;
    try {
        arena->reclaim();
    } catch (const CudaError& error) {
        actual = error.status;
    }
    require(actual == calls.query, "Event query failure lost its CUDA status");
    calls.query = cudaErrorNotReady;
    require(arena->available_bytes() == available - bytes, "Query failure released the lease");
    check(cudaStreamSynchronize(stream), "Complete work before retrying arena query");
    calls.query = cudaSuccess;
    require(arena->available_bytes() == available, "Completed fence left storage unavailable");

    pointer = arena->allocate(bytes);
    check(cudaMemsetAsync(pointer, 42, bytes, stream), "Queue work for record failure");
    calls.record = cudaErrorInvalidValue;
    require(arena->release_after(pointer, stream) == calls.record,
            "Event record failure lost its CUDA status");
    require(cudaStreamQuery(stream) == cudaSuccess && arena->available_bytes() == available,
            "Failed record did not synchronously confirm lease completion");
    calls.record = cudaSuccess;
    require(calls.physical_releases == releases, "Fenced release called physical CUDA free");
}

void check_pending_before_rank_store() {
    using namespace std::chrono_literals;
    const DeviceScope selected(0);
    PrefixResidency residency({0}, true, false, 1, 1, 1);
    const auto io =
        std::make_shared<LmCacheGroup>(LmCacheConfig{"/dev/null/sglang.sock", "sglang", 100}, 1);
    const PrefixRankKeys keys{
        std::make_shared<PrefixCacheKeys>(std::vector<int>(mxfp8_page_size, 198),
                                          "pre-transport-pending"),
    };
    require(!residency.backup_pending(), "Empty residency was marked pending");
    auto& gate = backup_gate();
    const auto entered = gate.entered.get_future();
    gate.armed = true;
    const ResidentPrefixRead cold;
    auto pages = residency.reserve_pages(cold, mxfp8_page_size, false);
    residency.publish_pages(keys, io, {pages.data(), 0, mxfp8_page_size}, false);
    pages.clear();
    residency.notify_released();

    const auto reached = entered.wait_for(5s);
    const bool pending = residency.backup_pending();
    const auto rank_pending = io->store_metrics().pending;
    gate.resume.set_value();
    residency.reset();
    DecoderPrefixCacheStats stats;
    residency.stats(stats);
    require(reached == std::future_status::ready, "Backup did not reach IPC export");
    require(pending && rank_pending == 0, "Backup was missing from pending before transport");
    require(!residency.backup_pending() && stats.backups_started == 1 && stats.backups_failed == 1,
            "Backup stayed pending after failure and reset");
}
}

extern "C" cudaError_t real_event_record(cudaEvent_t event,
                                         cudaStream_t stream) asm("__real_cudaEventRecord");
extern "C" cudaError_t wrap_event_record(cudaEvent_t event,
                                         cudaStream_t stream) asm("__wrap_cudaEventRecord");
extern "C" cudaError_t wrap_event_record(cudaEvent_t event, cudaStream_t stream) {
    const auto status = release_faults().record;
    return status == cudaSuccess ? real_event_record(event, stream) : status;
}

extern "C" cudaError_t real_event_query(cudaEvent_t event) asm("__real_cudaEventQuery");
extern "C" cudaError_t wrap_event_query(cudaEvent_t event) asm("__wrap_cudaEventQuery");
extern "C" cudaError_t wrap_event_query(cudaEvent_t event) {
    const auto status = release_faults().query;
    return status == cudaSuccess ? real_event_query(event) : status;
}

extern "C" cudaError_t real_free(void* pointer) asm("__real_cudaFree");
extern "C" cudaError_t wrap_free(void* pointer) asm("__wrap_cudaFree");
extern "C" cudaError_t wrap_free(void* pointer) {
    if (pointer != nullptr)
        ++release_faults().physical_releases;
    return real_free(pointer);
}

extern "C" cudaError_t real_ipc_export(cudaIpcMemHandle_t* handle,
                                       void* allocation) asm("__real_cudaIpcGetMemHandle");
extern "C" cudaError_t wrap_ipc_export(cudaIpcMemHandle_t* handle,
                                       void* allocation) asm("__wrap_cudaIpcGetMemHandle");
extern "C" cudaError_t wrap_ipc_export(cudaIpcMemHandle_t* handle, void* allocation) {
    auto& gate = backup_gate();
    if (!gate.armed.exchange(false))
        return real_ipc_export(handle, allocation);
    gate.entered.set_value();
    (void)gate.resumed.wait_for(std::chrono::seconds(10));
    return cudaErrorInvalidValue;
}

int main(int argc, char* const* argv) try {
    require(argc <= 2, "Usage: lmcache-release [TP2 checkpoint]");
    int devices = 0;
    check(cudaGetDeviceCount(&devices), "Count leased test devices");
    require(devices == argc,
            "Exactly one GPU is required for direct checks, or two with a checkpoint");
    check(cudaSetDevice(0), "Select leased device");
    if (argc == 1) {
        const auto arena = DeviceArena::start(0, default_gpu_memory_utilization);
        arena->commit(std::size_t{1} << 30);
        check_detached_releases(devices);
        check_fenced_release(0);
        check_pending_before_rank_store();
    } else {
        const Decoder decoder(argv[1], 0, false, 2);
        check_detached_releases(devices);
        for (int rank = 0; rank < devices; ++rank)
            check_fenced_release(rank);
        check_pending_before_rank_store();
    }
    std::cout << "PASS\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "Release probe: " << error.what() << '\n';
    return 1;
}
