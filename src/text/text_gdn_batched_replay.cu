#include "sglang/text/gdn/gdn_replay_batch.h"
#include <cstring>

#include <cuda_runtime.h>
#include <new>
#include <stdexcept>

namespace sglang {

TextGdnReplayBatch::TextGdnReplayBatch() noexcept = default;

int TextGdnReplayBatch::capacity() const { return capacity_; }

int TextGdnReplayBatch::size() const { return static_cast<int>(entries_.size()); }

int TextGdnReplayBatch::available() const { return capacity_ - size(); }

void TextGdnReplayBatch::reset() { entries_.clear(); }

bool TextGdnReplayBatch::add(const TextGdnReplayEntry& entry) {
    if (available() <= 0) return false;
    entries_.push_back(entry);
    return true;
}

TextGdnReplayBatch::~TextGdnReplayBatch() {
    if (copy_pending_ && copy_done_) {
        (void)cudaEventSynchronize(copy_done_);
        copy_pending_ = false;
    }
    if (pinned_) (void)cudaFreeHost(pinned_);
    if (device_) (void)cudaFree(device_);
    if (copy_done_) (void)cudaEventDestroy(copy_done_);
}

bool TextGdnReplayBatch::reserve(int capacity) {
    if (capacity <= capacity_) return true;
    try {
        entries_.reserve(capacity);
    } catch (const std::bad_alloc&) {
        return false;
    } catch (const std::length_error&) {
        return false;
    }
    TextGdnReplayEntry* pinned = nullptr;
    TextGdnReplayEntry* device = nullptr;
    cudaEvent_t copy_done = nullptr;
    const std::size_t bytes =
        static_cast<std::size_t>(capacity) * sizeof(TextGdnReplayEntry);
    if (cudaMallocHost(&pinned, bytes) != cudaSuccess ||
        cudaMalloc(&device, bytes) != cudaSuccess ||
        cudaEventCreateWithFlags(&copy_done, cudaEventDisableTiming) != cudaSuccess) {
        if (pinned) (void)cudaFreeHost(pinned);
        if (device) (void)cudaFree(device);
        if (copy_done) (void)cudaEventDestroy(copy_done);
        return false;
    }
    if (copy_pending_) {
        (void)cudaEventSynchronize(copy_done_);
        copy_pending_ = false;
    }
    if (pinned_) (void)cudaFreeHost(pinned_);
    if (device_) (void)cudaFree(device_);
    if (copy_done_) (void)cudaEventDestroy(copy_done_);
    pinned_ = pinned;
    device_ = device;
    copy_done_ = copy_done;
    capacity_ = capacity;
    entries_.clear();
    return true;
}

cudaError_t TextGdnReplayBatch::launch(bool rank_local, cudaStream_t stream) {
    const int count = size();
    if (count == 0) return cudaSuccess;
    if (copy_pending_) {
        const auto waited = cudaEventSynchronize(copy_done_);
        if (waited != cudaSuccess) return waited;
        copy_pending_ = false;
    }
    const std::size_t bytes =
        static_cast<std::size_t>(count) * sizeof(TextGdnReplayEntry);
    std::memcpy(pinned_, entries_.data(), bytes);
    auto status = cudaMemcpyAsync(device_, pinned_, bytes, cudaMemcpyHostToDevice, stream);
    if (status != cudaSuccess) return status;
    status = cudaEventRecord(copy_done_, stream);
    if (status != cudaSuccess) {
        (void)cudaStreamSynchronize(stream);
        return status;
    }
    copy_pending_ = true;
    entries_.clear();
    return gdn_replay::launch(device_, count, rank_local, stream);
}

}
