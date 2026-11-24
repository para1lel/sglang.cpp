#include "sglang/runtime/server/single_device_batch_executor.h"

#include <future>
#include <limits>
#include <optional>
#include <utility>

namespace sglang::server {
namespace {

class SingleDeviceBatchSession final : public BatchSession {
public:
    SingleDeviceBatchSession(const Decoder& decoder, const DecoderBatchOptions& options)
        : batch_(decoder, options) {}

    int admit(int slot, std::vector<int>&& prompt, const DecoderOptions& options) override {
        return batch_.admit(slot, std::move(prompt), options);
    }

    int admit_multimodal(int slot, std::vector<int>&& prompt, const DecoderOptions& options,
                         const std::vector<DecoderMedia>& media) override {
        return batch_.admit_multimodal(slot, std::move(prompt), options, media);
    }

    void cancel(int slot) override { batch_.cancel(slot); }

    Steps advance_chunked(const Selection& selected, DecoderBatchStepStats* stats) override {
        return batch_.advance_chunked(selected, stats);
    }

    bool available() const override { return batch_.available(); }
    DecoderPrefixCacheStats prefix_cache_stats() const override {
        return batch_.prefix_cache_stats();
    }
    void reset_prefix_cache() override { batch_.reset_prefix_cache(); }

    std::size_t request_device_allocation_bytes() const override {
        return batch_.request_device_allocation_bytes();
    }

    void release_retired_allocations() override { batch_.release_retired_allocations(); }

private:
    DecoderBatch batch_;
};

}

SingleDeviceBatchExecutor::SingleDeviceBatchExecutor(const Decoder& decoder, Options options,
                                                     Ready ready, Process process)
    : decoder_(decoder), options_(options), ready_(std::move(ready)),
      process_(std::move(process)) {}

SingleDeviceBatchExecutor::~SingleDeviceBatchExecutor() { stop(); }

void SingleDeviceBatchExecutor::start() {
    std::promise<void> started;
    auto ready = started.get_future();
    worker_ = std::thread(&SingleDeviceBatchExecutor::run, this, std::move(started));
    try {
        ready.get();
    } catch (...) {
        stop();
        throw;
    }
}

void SingleDeviceBatchExecutor::stop() {
    if (worker_.joinable()) {
        worker_.join();
    }
}

BatchTopology SingleDeviceBatchExecutor::topology() const noexcept {
    return single_device_batch_topology;
}

void SingleDeviceBatchExecutor::run(std::promise<void> started) {
    std::optional<SingleDeviceBatchSession> session;
    try {
        session.emplace(decoder_, DecoderBatchOptions{
            options_.cuda_graphs, options_.max_active, options_.prefix_cache_entries,
            options_.max_num_batched_tokens, std::numeric_limits<std::size_t>::max()});
        if (ready_) {
            ready_(*session);
        }
    } catch (...) {
        started.set_exception(std::current_exception());
        return;
    }
    started.set_value();
    if (process_) {
        process_(*session);
    }
}

}
