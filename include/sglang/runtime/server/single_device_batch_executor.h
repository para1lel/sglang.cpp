#pragma once

#include "sglang/decoder.h"
#include "sglang/runtime/server/batch_session.h"

#include <functional>
#include <future>
#include <thread>

namespace sglang::server {

class SingleDeviceBatchExecutor {
public:
    struct Options {
        bool cuda_graphs;
        int max_active;
        int prefix_cache_entries;
        int max_num_batched_tokens;
    };

    using Ready = std::function<void(BatchSession&)>;
    using Process = std::function<void(BatchSession&)>;

    SingleDeviceBatchExecutor(const Decoder& decoder, Options options, Ready ready,
                              Process process);
    ~SingleDeviceBatchExecutor();

    void start();
    void stop();
    BatchTopology topology() const noexcept;

private:
    void run(std::promise<void> started);

    const Decoder& decoder_;
    Options options_;
    Ready ready_;
    Process process_;
    std::thread worker_;
};

}
