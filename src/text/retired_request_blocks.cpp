#include "sglang/decoder_internal.h"

#include <condition_variable>
#include <cstdio>
#include <mutex>
#include <thread>

namespace sglang::decoder {

struct RetiredBlockReclaimer::State {
    State(int device, cudaStream_t stream) : device(device), stream(stream) {
        queued.reserve(block_limit);
        active.reserve(block_limit);
        worker = std::thread([this] { process(); });
    }

    void process() {
        const auto selected = cudaSetDevice(device);
        std::unique_lock lock(mutex);
        fail(selected, "Select CUDA device for request reclamation");
        for (;;) {
            changed.wait(lock, [&] { return closing || !queued.empty(); });
            if (queued.empty()) return;
            queued.swap(active);
            const bool usable = failure == cudaSuccess;
            lock.unlock();
            std::size_t released = 0;
            std::size_t quarantined = 0;
            cudaError_t enqueue_error = cudaSuccess;
            for (const auto block : active) {
                if (!usable || enqueue_error != cudaSuccess) {
                    quarantined += block.bytes;
                    continue;
                }
                const auto status = cudaFreeAsync(block.pointer, stream);
                if (status == cudaSuccess) {
                    released += block.bytes;
                } else {
                    quarantined += block.bytes;
                    enqueue_error = status;
                }
            }
            const auto sync = usable ? cudaStreamSynchronize(stream) : cudaSuccess;
            lock.lock();
            if (sync != cudaSuccess) {
                quarantined += released;
                released = 0;
            }
            fail(enqueue_error, "Queue retired CUDA block");
            fail(sync, "Drain retired CUDA blocks");
            pending_bytes -= released + quarantined;
            uncertain_bytes += quarantined;
            pending_blocks -= active.size();
            active.clear();
            changed.notify_all();
        }
    }

    static void log(cudaError_t status, const char* operation) {
        if (status != cudaSuccess)
            std::fprintf(stderr, "sglang decoder: %s: %s\n", operation,
                         cudaGetErrorString(status));
    }

    void fail(cudaError_t status, const char* operation) {
        log(status, operation);
        if (status != cudaSuccess && failure == cudaSuccess) failure = status;
        if (CudaError::terminal(status)) failure = status;
    }

    int device;
    cudaStream_t stream;
    mutable std::mutex mutex;
    std::condition_variable changed;
    std::vector<DeviceBlock> queued;
    std::vector<DeviceBlock> active;
    std::size_t pending_bytes = 0;
    std::size_t pending_blocks = 0;
    std::size_t uncertain_bytes = 0;
    cudaError_t failure = cudaSuccess;
    bool closing = false;
    std::thread worker;
};

std::unique_ptr<RetiredBlockReclaimer> RetiredBlockReclaimer::create(int device) {
    int supported = 0;
    check(cudaDeviceGetAttribute(&supported, cudaDevAttrMemoryPoolsSupported, device),
          "Query stream-ordered request reclamation");
    if (!supported) return nullptr;
    DeviceScope scope(device);
    cudaStream_t stream = nullptr;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
          "Create retired request stream");
    try {
        return std::unique_ptr<RetiredBlockReclaimer>(new RetiredBlockReclaimer(device, stream));
    } catch (...) {
        State::log(cudaStreamDestroy(stream), "Destroy unused retired request stream");
        throw;
    }
}

RetiredBlockReclaimer::RetiredBlockReclaimer(int device, cudaStream_t stream)
    : state_(std::make_unique<State>(device, stream)) {}

RetiredBlockReclaimer::~RetiredBlockReclaimer() {
    {
        const std::lock_guard lock(state_->mutex);
        state_->closing = true;
    }
    state_->changed.notify_one();
    state_->worker.join();
    try {
        DeviceScope device(state_->device);
        State::log(cudaStreamDestroy(state_->stream), "Destroy retired request stream");
    } catch (const CudaError& error) {
        State::log(error.status, "Select device to destroy retired request stream");
    }
}

bool RetiredBlockReclaimer::submit(DeviceBlock block) {
    const std::lock_guard lock(state_->mutex);
    if (CudaError::terminal(state_->failure))
        check(state_->failure, "Retire decoder allocation");
    if (state_->failure != cudaSuccess || state_->pending_blocks == block_limit ||
        block.bytes > byte_limit - state_->pending_bytes)
        return false;
    state_->queued.push_back(block);
    state_->pending_bytes += block.bytes;
    ++state_->pending_blocks;
    state_->changed.notify_one();
    return true;
}

std::size_t RetiredBlockReclaimer::unavailable_bytes() const {
    const std::lock_guard lock(state_->mutex);
    if (CudaError::terminal(state_->failure))
        check(state_->failure, "Check retired decoder allocations");
    return state_->pending_bytes + state_->uncertain_bytes;
}

std::size_t RetiredBlockReclaimer::pending_bytes() const {
    const std::lock_guard lock(state_->mutex);
    return state_->pending_bytes;
}

void RetiredBlockReclaimer::drain() {
    std::unique_lock lock(state_->mutex);
    state_->changed.wait(lock, [&] { return state_->pending_blocks == 0; });
    if (CudaError::terminal(state_->failure))
        check(state_->failure, "Drain retired decoder allocations");
}

std::size_t reserve_request_memory(RequestBlockCache& retired, DecoderOptions& request,
                                   int prompt, std::size_t capacity, std::size_t charged,
                                   std::size_t extra_bytes) {
    const auto headroom = [&] {
        return capacity - std::min(capacity, charged + retired.unavailable_bytes());
    };
    const auto after_reserve = [](std::size_t free) {
        return free - std::min(free, decoder_shared_reserve);
    };
    auto free = available_device_bytes();
    try {
        request.max_new_tokens = limit_resident_output(
            prompt, request, capacity, std::min(headroom(), after_reserve(free + retired.bytes())),
            extra_bytes);
    } catch (const DecoderCapacityError& error) {
        if (!error.retryable || !retired.pending_bytes()) throw;
        retired.clear();
        free = available_device_bytes();
        request.max_new_tokens = limit_resident_output(
            prompt, request, capacity, std::min(headroom(), after_reserve(free)), extra_bytes);
    }
    request.min_new_tokens = std::min(request.min_new_tokens, request.max_new_tokens);
    request.incremental_kv = false;
    const auto reserved = extra_bytes + request_resident_bytes(
        RequestProgress::capacity(prompt, request),
        reserved_verification_rows(request.max_new_tokens, request.mtp_steps),
        request.mtp_steps != 0);
    if (reserved > after_reserve(free) || retired.bytes() > headroom() - reserved) {
        retired.clear();
        if (reserved > headroom() || reserved > after_reserve(available_device_bytes())) {
            throw DecoderCapacityError("Insufficient free resident decoder memory. Retry later.",
                                       true);
        }
    }
    return reserved;
}

}
