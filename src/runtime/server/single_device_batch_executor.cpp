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
    DecoderBatchPrefixUpdate poll_prefix(int slot) override { return batch_.poll_prefix(slot); }

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

SingleDeviceBatchExecutor::SingleDeviceBatchExecutor(const Decoder& decoder, Options options)
    : BatchExecutor(options), decoder_(decoder) {}

SingleDeviceBatchExecutor::~SingleDeviceBatchExecutor() { join(); }

void SingleDeviceBatchExecutor::start(Ready ready, Process process) {
    std::promise<void> started;
    auto result = started.get_future();
    worker_ = std::thread(&SingleDeviceBatchExecutor::run, this, std::move(ready),
                          std::move(process), std::move(started));
    try {
        result.get();
    } catch (...) {
        join();
        throw;
    }
}

void SingleDeviceBatchExecutor::join() noexcept {
    if (worker_.joinable()) {
        worker_.join();
    }
}

BatchTopology SingleDeviceBatchExecutor::topology() const noexcept {
    return single_device_batch_topology;
}

void SingleDeviceBatchExecutor::run(Ready ready, Process process, std::promise<void> started) {
    std::optional<SingleDeviceBatchSession> session;
    try {
        const auto& config = options();
        session.emplace(decoder_, DecoderBatchOptions{
            config.cuda_graphs, config.max_active, config.prefix_cache_entries,
            config.max_num_batched_tokens, std::numeric_limits<std::size_t>::max()});
        ready(*session);
    } catch (...) {
        started.set_exception(std::current_exception());
        return;
    }
    started.set_value();
    process(*session);
}

}
