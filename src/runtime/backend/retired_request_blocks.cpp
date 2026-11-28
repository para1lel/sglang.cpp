#include "sglang/runtime/backend/cuda_resources.h"
#include "sglang/runtime/backend/device_arena.h"

#include <condition_variable>
#include <mutex>
#include <utility>
#include <thread>
#include <vector>

namespace sglang::decoder {

struct RetiredBlockReclaimer::State {
    explicit State(std::shared_ptr<DeviceArena> owner) : arena(std::move(owner)) {
        queued.reserve(block_limit);
        active.reserve(block_limit);
        worker = std::thread([this] { process(); });
    }

    void process() {
        std::unique_lock lock(mutex);
        for (;;) {
            changed.wait(lock, [&] { return closing || !queued.empty(); });
            if (queued.empty()) return;
            queued.swap(active);
            lock.unlock();
            std::size_t completed = 0;
            for (const auto& block : active) {
                arena->release(block.pointer);
                completed += block.bytes;
            }
            lock.lock();
            pending_bytes -= completed;
            pending_blocks -= active.size();
            active.clear();
            changed.notify_all();
        }
    }

    std::shared_ptr<DeviceArena> arena;
    mutable std::mutex mutex;
    std::condition_variable changed;
    std::vector<DeviceBlock> queued;
    std::vector<DeviceBlock> active;
    std::size_t pending_bytes = 0;
    std::size_t pending_blocks = 0;
    bool closing = false;
    std::thread worker;
};

std::unique_ptr<RetiredBlockReclaimer> RetiredBlockReclaimer::create(int device) {
    return std::unique_ptr<RetiredBlockReclaimer>(
        new RetiredBlockReclaimer(DeviceArena::current(device)));
}

RetiredBlockReclaimer::RetiredBlockReclaimer(std::shared_ptr<DeviceArena> arena)
    : state_(std::make_unique<State>(std::move(arena))) {
}

RetiredBlockReclaimer::~RetiredBlockReclaimer() {
    {
        const std::lock_guard lock(state_->mutex);
        state_->closing = true;
    }
    state_->changed.notify_one();
    state_->worker.join();
}

bool RetiredBlockReclaimer::submit(const DeviceBlock& block) {
    const std::lock_guard lock(state_->mutex);
    if (state_->pending_blocks == block_limit || state_->pending_bytes >= byte_limit ||
        block.bytes > byte_limit - state_->pending_bytes)
        return false;
    state_->queued.push_back(block);
    state_->pending_bytes += block.bytes;
    ++state_->pending_blocks;
    state_->changed.notify_one();
    return true;
}

std::size_t RetiredBlockReclaimer::pending_bytes() const {
    const std::lock_guard lock(state_->mutex);
    return state_->pending_bytes;
}

void RetiredBlockReclaimer::drain() {
    std::unique_lock lock(state_->mutex);
    state_->changed.wait(lock, [&] { return state_->pending_blocks == 0; });
}

}
