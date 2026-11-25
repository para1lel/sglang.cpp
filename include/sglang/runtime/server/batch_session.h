#pragma once

#include "sglang/decoder_batch.h"

namespace sglang::server {

struct BatchTopology {
    int dp_size;
    int tp_size;
    int pp_size;

    constexpr int device_count() const noexcept { return dp_size * tp_size * pp_size; }
};

inline constexpr BatchTopology single_device_batch_topology{1, 1, 1};

/// Call these operations on the worker thread that owns the session.
class BatchSession {
public:
    using Selection = std::array<bool, decoder_batch_max_lanes>;
    using Steps = std::array<DecoderBatchStep, decoder_batch_max_lanes>;

    /// Return the number of prompt positions restored from the prefix cache.
    virtual int admit(int slot, std::vector<int>&& prompt, const DecoderOptions& options) = 0;
    virtual int admit_multimodal(int slot, std::vector<int>&& prompt,
                                 const DecoderOptions& options,
                                 const std::vector<DecoderMedia>& media) = 0;
    virtual void cancel(int slot) = 0;
    virtual DecoderBatchPrefixUpdate poll_prefix(int) { return {}; }
    virtual Steps advance_chunked(const Selection& selected, DecoderBatchStepStats* stats) = 0;

    /// Report whether the session can continue after an admission or step exception.
    virtual bool available() const = 0;
    virtual DecoderPrefixCacheStats prefix_cache_stats() const = 0;
    virtual void reset_prefix_cache() = 0;
    virtual std::size_t request_device_allocation_bytes() const = 0;
    virtual void release_retired_allocations() = 0;

protected:
    ~BatchSession() = default;
};

}
