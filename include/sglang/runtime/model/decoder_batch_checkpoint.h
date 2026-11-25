#pragma once

#include "sglang/decoder_prefix_cache.h"

#include <chrono>
#include <future>
#include <optional>
#include <utility>

namespace sglang::decoder {

struct PendingBatchPrefix {
    explicit PendingBatchPrefix(PrefixDeviceTransfer&& view) : command(std::move(view)) {}

    ~PendingBatchPrefix() {
        // Keep the descriptor and every borrowed CUDA allocation live through daemon quiescence.
        if (completion.valid()) completion.wait();
    }

    bool ready() const {
        return completion.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
    }

    PrefixDeviceTransfer command;
    bool discard = false;
    std::shared_future<bool> completion;
};

struct DeferredPrefixStore {
    const Workspace* row = nullptr;
    std::shared_ptr<void> reservation = nullptr;
    bool terminal = false;
};

}
