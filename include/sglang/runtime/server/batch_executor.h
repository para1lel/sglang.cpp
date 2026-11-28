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
        bool tp_verification_graphs = false;
        bool resident_prefix_cache = false;
        int prefix_gqa_pages = 0;
        int prefix_gdn_slots = 0;
    };

    using Ready = std::function<void(BatchSession&)>;
    using Process = std::function<void(BatchSession&)>;

    virtual ~BatchExecutor();

    const Options& options() const noexcept;
    virtual BatchTopology topology() const noexcept = 0;

    virtual void start(Ready ready, Process process) = 0;
    virtual void join() noexcept = 0;

protected:
    explicit BatchExecutor(Options options);

private:
    const Options options_;
};

}
