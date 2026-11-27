#pragma once

#include "sglang/decoder_batch.h"
#include "sglang/server/batch_service.h"

#include <array>

namespace sglang::server {

class BatchTrace {
public:
    using Steps = std::array<DecoderBatchStep, decoder_batch_max_lanes>;

    BatchTrace(bool trace_cache, const DecoderPrefixCacheStats& cache);
    void failed(std::size_t bytes);
    void step(int width, const Steps& steps, const DecoderBatchStepStats& stats, std::size_t bytes,
              int max_active, int max_num_batched_tokens, int mtp_steps);
    void cache(const DecoderPrefixCacheStats& current, const Steps& steps);

private:
    bool trace_cache_;
    DecoderPrefixCacheStats last_cache_;
    std::size_t last_bytes_ = 0;
};

BatchProfileStep batch_profile_step(std::uint64_t elapsed, int width,
                                    const BatchTrace::Steps& steps,
                                    const DecoderBatchStepStats& stats);

}
