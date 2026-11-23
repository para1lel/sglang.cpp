#include "sglang/runtime/server/single_device_batch_executor.h"

#include <future>
#include <limits>
#include <memory>
#include <utility>

namespace sglang::server {

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

void SingleDeviceBatchExecutor::run(std::promise<void> started) {
    std::unique_ptr<DecoderBatch> batch;
    try {
        batch = std::make_unique<DecoderBatch>(decoder_,
            DecoderBatchOptions{options_.cuda_graphs, options_.max_active,
                                options_.prefix_cache_entries,
                                options_.max_num_batched_tokens,
                                std::numeric_limits<std::size_t>::max()});
        if (ready_) {
            ready_(*batch);
        }
    } catch (...) {
        started.set_exception(std::current_exception());
        return;
    }
    started.set_value();
    if (process_) {
        process_(*batch);
    }
}

}
