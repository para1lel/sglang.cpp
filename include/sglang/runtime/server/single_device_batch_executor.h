#pragma once

#include "sglang/runtime/server/batch_executor.h"

#include <future>
#include <thread>

namespace sglang::server {

class SingleDeviceBatchExecutor final : public BatchExecutor {
public:
    SingleDeviceBatchExecutor(const Decoder& decoder, Options options);
    ~SingleDeviceBatchExecutor() override;

    BatchTopology topology() const noexcept override;
    void start(Ready ready, Process process) override;
    void join() noexcept override;

private:
    void run(Ready ready, Process process, std::promise<void> started);

    const Decoder& decoder_;
    std::thread worker_;
};

}
