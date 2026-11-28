#include "sglang/decoder_batch.h"
#include "sglang/decoder_internal.h"
#include "sglang/runtime/backend/cuda_resources.h"
#include "sglang/runtime/cache/prefix_payload.h"

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

using namespace sglang;
using namespace sglang::decoder;

namespace {

void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

class CompletionGate {
  public:
    void hold() {
        std::unique_lock lock(mutex_);
        entered_ = true;
        changed_.notify_all();
        passed_ = changed_.wait_for(lock, std::chrono::seconds(15), [this] { return open_; });
    }

    void wait_until_held() {
        std::unique_lock lock(mutex_);
        require(changed_.wait_for(lock, std::chrono::seconds(10), [this] { return entered_; }),
                "Retired CUDA stream did not reach its completion check");
    }

    void open() {
        const std::scoped_lock lock(mutex_);
        open_ = true;
        changed_.notify_all();
    }

    bool passed() {
        const std::scoped_lock lock(mutex_);
        return passed_;
    }

  private:
    std::mutex mutex_;
    std::condition_variable changed_;
    bool entered_ = false;
    bool open_ = false;
    bool passed_ = false;
};

struct ReleaseObservation {
    ReleaseObservation() = default;
    ReleaseObservation(const ReleaseObservation&) = delete;
    ReleaseObservation& operator=(const ReleaseObservation&) = delete;
    ReleaseObservation(ReleaseObservation&&) = delete;
    ReleaseObservation& operator=(ReleaseObservation&&) = delete;
    ~ReleaseObservation() {
        active().store(nullptr);
    }

    static std::atomic<ReleaseObservation*>& active() {
        static std::atomic<ReleaseObservation*> observation{nullptr};
        return observation;
    }

    std::atomic<void*> memory{nullptr};
    int device = -1;
    cudaError_t enqueue_error = cudaSuccess;
    cudaError_t completion_error = cudaSuccess;
    CompletionGate* gate = nullptr;
    std::atomic<int> enqueues{0};
    std::atomic<int> completions{0};
    std::atomic<int> fallbacks{0};
};

ReleaseObservation* observe_release(void* memory) {
    auto* watched = ReleaseObservation::active().load();
    if (watched == nullptr || memory == nullptr)
        return nullptr;
    auto* target = watched->memory.load();
    if (target == nullptr && watched->device >= 0) {
        int device = -1;
        if (cudaGetDevice(&device) == cudaSuccess && device == watched->device &&
            watched->memory.compare_exchange_strong(target, memory)) {
            target = memory;
        }
    }
    return target == memory ? watched : nullptr;
}

struct QueuedRelease {
    ReleaseObservation* observation = nullptr;
    cudaStream_t stream = nullptr;
};

QueuedRelease& queued_release() {
    static thread_local QueuedRelease release;
    return release;
}

}

extern "C" cudaError_t real_cuda_free(void*) asm("__real_cudaFree");
extern "C" cudaError_t wrap_cuda_free(void* memory) asm("__wrap_cudaFree");
extern "C" cudaError_t wrap_cuda_free(void* memory) {
    if (auto* watched = ReleaseObservation::active().load();
        watched != nullptr && memory != nullptr && memory == watched->memory.load()) {
        ++watched->fallbacks;
    }
    return real_cuda_free(memory);
}

extern "C" cudaError_t real_cuda_free_async(void*, cudaStream_t) asm("__real_cudaFreeAsync");
extern "C" cudaError_t wrap_cuda_free_async(void* memory,
                                            cudaStream_t stream) asm("__wrap_cudaFreeAsync");
extern "C" cudaError_t wrap_cuda_free_async(void* memory, cudaStream_t stream) {
    auto* watched = observe_release(memory);
    if (watched != nullptr) {
        ++watched->enqueues;
        if (watched->enqueue_error != cudaSuccess)
            return watched->enqueue_error;
    }
    const auto status = real_cuda_free_async(memory, stream);
    if (watched != nullptr && status == cudaSuccess)
        queued_release() = {watched, stream};
    return status;
}

extern "C" cudaError_t real_cuda_stream_sync(cudaStream_t) asm("__real_cudaStreamSynchronize");
extern "C" cudaError_t wrap_stream_sync(cudaStream_t stream) asm("__wrap_cudaStreamSynchronize");
extern "C" cudaError_t wrap_stream_sync(cudaStream_t stream) {
    auto& queued = queued_release();
    auto* watched = stream == queued.stream ? std::exchange(queued, {}).observation : nullptr;
    if (watched != nullptr && watched->gate != nullptr)
        watched->gate->hold();
    const auto status = real_cuda_stream_sync(stream);
    if (watched != nullptr) {
        ++watched->completions;
        if (status == cudaSuccess)
            return watched->completion_error;
    }
    return status;
}

namespace {

void check_detached_accounting(int device, bool resident) {
    constexpr std::size_t bytes = 1 << 20;
    const DeviceScope scope(device);
    ReleaseObservation release;
    CompletionGate gate;
    RequestBlockCache retired(device, resident, true);
    Allocations original(device, &retired);
    auto* memory = original.allocate<std::byte>(bytes);
    auto first = original.retain_block(memory);
    auto last = original.retain_block(memory);
    require(retired.unavailable_bytes() == 0, "Live source was also charged as detached");
    original.recycle_after_completion();
    require(retired.unavailable_bytes() == bytes && retired.pending_bytes() == 0,
            "One physical block did not produce one detached charge");
    first.reset();
    require(retired.unavailable_bytes() == bytes, "One alias dropped the remaining lease");
    release.memory = memory;
    release.gate = &gate;
    ReleaseObservation::active().store(&release);
    last.reset();
    try {
        gate.wait_until_held();
        require(retired.unavailable_bytes() == bytes && retired.pending_bytes() == bytes,
                "Moving a detached block to pending changed its single charge");
        require(release.enqueues == 1 && release.completions == 0,
                "Pending charge did not cover the outstanding stream completion");
    } catch (...) {
        gate.open();
        throw;
    }
    gate.open();
    retired.clear();
    require(gate.passed(), "The retired CUDA stream timed out before explicit release");
    require(retired.unavailable_bytes() == 0 && retired.pending_bytes() == 0 &&
                retired.failed_release_bytes() == 0,
            "Freed device block stayed charged");
    retired.check_releases();
    require(release.enqueues == 1 && release.completions == 1 && release.fallbacks == 0,
            "A detached block did not complete exactly one asynchronous release");
}

template <typename Operation>
void expect_release_error(cudaError_t expected, const Operation& operation) {
    cudaError_t actual = cudaSuccess;
    try {
        operation();
    } catch (const CudaError& error) {
        actual = error.status;
    }
    require(actual == expected, "Release failure did not preserve its original CUDA status");
}

void check_failed_free(int device, bool resident, bool deferred, cudaError_t enqueue_error,
                       cudaError_t completion_error = cudaSuccess) {
    constexpr std::size_t bytes = 1 << 20;
    const DeviceScope scope(device);
    ReleaseObservation release;
    RequestBlockCache retired(device, resident, deferred);
    Allocations original(device, &retired);
    auto* memory = original.allocate<std::byte>(bytes);
    auto lease = original.retain_block(memory);
    original.recycle_after_completion();
    require(retired.unavailable_bytes() == bytes, "Detached source was not charged");
    release.memory = memory;
    release.enqueue_error = enqueue_error;
    release.completion_error = completion_error;
    ReleaseObservation::active().store(&release);
    lease.reset();

    const bool queued = enqueue_error == cudaSuccess;
    const bool fallback = enqueue_error == cudaErrorNotSupported;
    const auto async_error = queued ? completion_error : enqueue_error;
    const auto expected = fallback ? cudaSuccess : async_error;
    const auto failed_bytes = expected == cudaSuccess ? 0 : bytes;
    for (int repeat = 0; repeat < 2; ++repeat) {
        expect_release_error(expected, [&] { retired.clear(); });
        expect_release_error(expected, [&] { retired.check_releases(); });
        require(retired.unavailable_bytes() == failed_bytes &&
                    retired.failed_release_bytes() == failed_bytes && retired.pending_bytes() == 0,
                "A detached failure lost or duplicated the unrecovered device charge");
    }
    require(release.enqueues == 1 && release.completions == static_cast<int>(queued) &&
                release.fallbacks == static_cast<int>(fallback),
            "A detached block was freed twice or took a fallback for an ambiguous failure");
    if (!queued && !fallback)
        check(real_cuda_free(memory), "Release unsubmitted test allocation");
}

void check_serving_release_failure(const Decoder& decoder, int failing_rank) {
    ReleaseObservation release;
    DecoderBatchOptions batch_options;
    batch_options.max_lanes = 1;
    batch_options.resident_memory_limit = std::numeric_limits<std::size_t>::max();
    DecoderBatch batch(decoder, batch_options);
    std::array<PrefixDeviceTransfer, 2> terminal;
    DecoderOptions options;
    options.max_new_tokens = 2;
    options.ignore_eos = true;
    options.store_prefix = [&](PrefixDeviceTransfer&& view, const int*, int,
                               const std::shared_ptr<void>&,
                               const std::shared_ptr<std::atomic_uint64_t>&) {
        if (view.prefill_checkpoint)
            return;
        int rank = -1;
        check(cudaGetDevice(&rank), "Identify terminal export rank");
        require(rank >= 0 && rank < 2 && !terminal[rank].store_owner,
                "A serving lane published duplicate terminal owners");
        terminal[rank] = std::move(view);
    };
    const std::vector<int> prompt(257, 42);
    require(batch.admit(0, prompt, options) == 0, "Fault fixture unexpectedly restored a prefix");
    std::array<bool, decoder_batch_max_lanes> selected{};
    selected[0] = true;
    for (int step = 0; selected[0] && step < 4; ++step) {
        const auto result = batch.advance_chunked(selected)[0];
        require(!result.sampling_failed && !result.constraint_failed && !result.prefix_pending,
                "Fault fixture did not produce an ordinary terminal STORE");
        if (result.finish_reason) {
            require(*result.finish_reason == DecoderFinishReason::length,
                    "Fault fixture terminated early");
            selected[0] = false;
        }
    }
    require(!selected[0] && terminal[0].store_owner && terminal[1].store_owner,
            "Missing retained terminal allocations from either serving rank");

    batch.release_retired_allocations();
    constexpr auto failure = cudaErrorInvalidResourceHandle;
    release.device = failing_rank;
    release.completion_error = failure;
    ReleaseObservation::active().store(&release);
    terminal[failing_rank] = {};
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    auto stats = batch.prefix_cache_stats();
    while (stats.failed_release_bytes == 0 && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        stats = batch.prefix_cache_stats();
    }
    require(stats.failed_release_bytes > 0,
            "Failed rank acknowledged unconfirmed terminal release bytes");
    expect_release_error(failure, [&] { batch.admit(0, prompt, options); });
    require(!batch.available(), "A rank release failure left the serving batch available");
    terminal[1 - failing_rank] = {};
    expect_release_error(failure, [&] { batch.release_retired_allocations(); });
    stats = batch.prefix_cache_stats();
    require(stats.failed_release_bytes > 0 && stats.pending_release_bytes == 0,
            "Idle drain discarded the failed charge or left a rank pending");
    require(release.enqueues == 1 && release.completions == 1 && release.fallbacks == 0,
            "The failed serving release was retried during admission or idle drain");
    std::cout << "rank=" << failing_rank << " serving admission disabled, "
              << stats.failed_release_bytes << " bytes still charged after drain\n";
}

void check_detached_releases(int peer) {
    check_detached_accounting(0, true);
    check_detached_accounting(peer, false);
    for (const bool resident : {false, true}) {
        const int device = resident ? 0 : peer;
        check_failed_free(device, resident, true, cudaErrorNotSupported);
        check_failed_free(device, resident, true, cudaErrorInvalidValue);
        check_failed_free(device, resident, true, cudaSuccess, cudaErrorInvalidResourceHandle);
    }
    check_failed_free(0, false, false, cudaErrorNotSupported);
    check_failed_free(0, false, false, cudaErrorInvalidValue);
    std::cout
        << "Both cache roles kept unconfirmed releases charged and latched their first error\n";
}

}

int main(int argc, char* const* argv) try {
    require(argc <= 2, "Usage: lmcache-release [TP2 checkpoint]");
    int devices = 0;
    check(cudaGetDeviceCount(&devices), "Count leased test devices");
    require(devices == argc,
            "Exactly one GPU is required for direct checks, or two with a checkpoint");
    check(cudaSetDevice(0), "Select leased device");
    check_detached_releases(devices - 1);
    if (argc == 2) {
        const Decoder decoder(argv[1], 0, false, 2);
        check_serving_release_failure(decoder, 0);
        check_serving_release_failure(decoder, 1);
    }
    std::cout << "PASS\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "Release probe: " << error.what() << '\n';
    return 1;
}
