#pragma once

#include "sglang/decoder.h"
#include "sglang/decoder_batch.h"

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

    using Ready = std::function<void(DecoderBatch&)>;
    using Process = std::function<void(DecoderBatch&)>;

    SingleDeviceBatchExecutor(const Decoder& decoder, Options options, Ready ready,
                              Process process);
    ~SingleDeviceBatchExecutor();

    void start();
    void stop();

private:
    void run(std::promise<void> started);

    const Decoder& decoder_;
    Options options_;
    Ready ready_;
    Process process_;
    std::thread worker_;
};

}
