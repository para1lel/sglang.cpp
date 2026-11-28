#pragma once

#include "sglang/decoder_batch.h"

namespace sglang::server {

struct BatchTopology {
    int dp_size;
    int tp_size;
    int pp_size;

    int device_count() const noexcept;
};

inline constexpr BatchTopology single_device_batch_topology{1, 1, 1};

class BatchSession {
public:
    using Selection = std::array<bool, decoder_batch_max_lanes>;
    using Steps = std::array<DecoderBatchStep, decoder_batch_max_lanes>;

    virtual int admit(int slot, std::vector<int>&& prompt, const DecoderOptions& options) = 0;
    virtual int admit_multimodal(int slot, std::vector<int>&& prompt,
                                 const DecoderOptions& options,
                                 const std::vector<DecoderMedia>& media) = 0;
    virtual void cancel(int slot) = 0;
    virtual DecoderBatchPrefixUpdate poll_prefix(int);
    virtual Steps advance_chunked(const Selection& selected, DecoderBatchStepStats* stats) = 0;

    virtual bool available() const = 0;
    virtual DecoderPrefixCacheStats prefix_cache_stats() const = 0;
    virtual bool prefix_backup_pending() const = 0;
    virtual void reset_prefix_cache() = 0;
    virtual std::size_t request_device_allocation_bytes() const = 0;
    virtual void release_retired_allocations() = 0;
    virtual bool prefix_restore_waiting(int slot) const;

  protected:
    ~BatchSession();
};

}
