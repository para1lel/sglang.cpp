#pragma once

#include "sglang/decoder_prefix_cache.h"

#include <chrono>
#include <future>
#include <optional>
#include <utility>

namespace sglang::decoder {

struct PendingBatchPrefix {
    PendingBatchPrefix(PrefixDeviceTransfer&& command, int end, bool restore, bool terminal)
        : command(std::move(command)), end(end), restore(restore), terminal(terminal) {}

    ~PendingBatchPrefix() {
        // Keep the descriptor and every borrowed CUDA allocation live through daemon quiescence.
        if (completion.valid()) completion.wait();
    }

    bool ready() const {
        return completion.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
    }

    PrefixDeviceTransfer command;
    const int end;
    const bool restore;
    const bool terminal;
    bool discard = false;
    std::shared_future<bool> completion;
};

}
