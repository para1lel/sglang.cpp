#pragma once

#include "sglang/runtime/server/batch_session.h"

#include <functional>

namespace sglang::server {

class BatchExecutor {
public:
    struct Options {
        bool cuda_graphs;
        int max_active;
        int prefix_cache_entries;
        int max_num_batched_tokens;
    };

    using Ready = std::function<void(BatchSession&)>;
    using Process = std::function<void(BatchSession&)>;

    virtual ~BatchExecutor() = default;

    const Options& options() const noexcept { return options_; }
    virtual BatchTopology topology() const noexcept = 0;

    /// Start once with valid callbacks, both invoked on the session's worker.
    /// Return after Ready succeeds.
    /// A startup exception reaches the caller after worker and session teardown.
    virtual void start(Ready ready, Process process) = 0;
    /// Wait for worker and session teardown after the service signals closure.
    /// Safe to call before startup and after a previous join.
    virtual void join() noexcept = 0;

protected:
    explicit BatchExecutor(Options options) : options_(options) {}

private:
    const Options options_;
};

}
