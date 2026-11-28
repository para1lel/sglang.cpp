#pragma once

#include "sglang/decoder_prefix_cache.h"

#include <chrono>
#include <future>
#include <optional>
#include <utility>

namespace sglang::decoder {

struct PendingBatchPrefix {
    explicit PendingBatchPrefix(PrefixDeviceTransfer&& view);

    ~PendingBatchPrefix();

    bool ready() const;

    PrefixDeviceTransfer command;
    bool discard = false;
    std::shared_future<bool> completion;
};

struct DeferredPrefixStore {
    const Workspace* row = nullptr;
    std::shared_ptr<void> reservation = nullptr;
};

}
