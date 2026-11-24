#pragma once

#include "sglang/decoder_batch.h"
#include "sglang/server/batch_service.h"

#include <algorithm>
#include <array>
#include <iostream>
#include <numeric>
#include <string>

namespace sglang::server {

inline void trace_verification(const std::array<DecoderBatchStep, decoder_batch_max_lanes>& steps,
                               const DecoderBatchStepStats& stats) {
    std::cerr << ("batch verify lanes=" + std::to_string(stats.verification_lanes) +
                  " rows=" + std::to_string(stats.verification_rows) +
                  " peak_request_bytes=" + std::to_string(stats.peak_request_bytes) + '\n');
    for (int slot = 0; slot < decoder_batch_max_lanes; ++slot) {
        const auto& step = steps[slot];
        if (step.mtp_verified_rows || step.mtp_boundary) {
            std::cerr << ("batch mtp slot=" + std::to_string(slot) +
                          " drafted=" + std::to_string(step.mtp_drafted_tokens) +
                          " accepted=" + std::to_string(step.mtp_accepted_tokens) +
                          " verified=" + std::to_string(step.mtp_verified_rows) +
                          " replayed=" + std::to_string(step.mtp_replayed_rows) +
                          " boundary=" + std::to_string(step.mtp_boundary) + '\n');
        }
    }
}

inline void trace_batch_details(const std::array<DecoderBatchStep, decoder_batch_max_lanes>& steps,
                                const DecoderBatchStepStats& stats, int max_active,
                                int max_num_batched_tokens, int mtp_steps) {
    std::string line = "batch prefill_rows=";
    for (int slot = 0; slot < std::max(max_active, decoder_batch_max_mtp_lanes); ++slot) {
        if (slot != 0) {
            line += ',';
        }
        line += std::to_string(stats.prefill_rows[slot]);
    }
    std::cerr << (line + " workspace_bytes=" + std::to_string(stats.prefill_workspace_bytes) +
                  '\n');
    std::cerr << ("batch target_rows=" + std::to_string(stats.target_rows) +
                  " step_budget=" + std::to_string(max_num_batched_tokens) + '\n');
    if (stats.layerwise_prefill_groups) {
        std::cerr << ("batch layerwise_prefill_groups=" +
                      std::to_string(stats.layerwise_prefill_groups) +
                      " rows=" + std::to_string(stats.layerwise_prefill_rows) + '\n');
    }
    if (stats.graph_slots) {
        std::cerr << ("batch graph_slots=" + std::to_string(stats.graph_slots) + '\n');
    }
    if (stats.sampling_workspace_bytes) {
        std::cerr << ("batch sampling_workspace_bytes=" +
                      std::to_string(stats.sampling_workspace_bytes) + '\n');
    }
    if (stats.constraint_workspace_bytes) {
        std::cerr << ("batch constraint_workspace_bytes=" +
                      std::to_string(stats.constraint_workspace_bytes) + '\n');
    }
    if (mtp_steps != 0) {
        trace_verification(steps, stats);
    } else {
        std::cerr << ("batch peak_request_bytes=" + std::to_string(stats.peak_request_bytes) +
                      '\n');
    }
}

inline void trace_batch_step(int width, int decode, std::size_t request_bytes) {
    std::cerr << ("batch width=" + std::to_string(width) + " decode=" + std::to_string(decode) +
                  " request_bytes=" + std::to_string(request_bytes) + '\n');
}

inline void trace_prefix_cache(const DecoderPrefixCacheStats& stats) {
    std::cerr << ("prefix cache entries=" + std::to_string(stats.entries) +
                  " pages=" + std::to_string(stats.pages) +
                  " device_bytes=" + std::to_string(stats.device_bytes) +
                  " hits=" + std::to_string(stats.hits) +
                  " cached_tokens=" + std::to_string(stats.cached_tokens) +
                  " prefill_tokens=" + std::to_string(stats.prefill_tokens) +
                  " stores=" + std::to_string(stats.stores) +
                  " evictions=" + std::to_string(stats.evictions) + '\n');
}

class BatchTrace {
public:
    using Steps = std::array<DecoderBatchStep, decoder_batch_max_lanes>;

    BatchTrace(bool trace_cache, const DecoderPrefixCacheStats& cache)
        : trace_cache_(trace_cache), last_cache_(cache) {
        if (trace_cache_) trace_prefix_cache(cache);
    }

    void failed(std::size_t bytes) {
        trace_batch_step(0, 0, bytes);
        last_bytes_ = 0;
    }

    void step(int width, const Steps& steps, const DecoderBatchStepStats& stats, std::size_t bytes,
              int max_active, int max_num_batched_tokens, int mtp_steps) {
        if (!width && bytes == last_bytes_) return;
        const int decode = std::count_if(steps.begin(), steps.end(),
                                         [](const auto& step) { return step.decode; });
        trace_batch_step(width, decode, bytes);
        if (width) trace_batch_details(steps, stats, max_active, max_num_batched_tokens, mtp_steps);
        last_bytes_ = bytes;
    }

    void cache(const DecoderPrefixCacheStats& current, const Steps& steps) {
        if (!trace_cache_) return;
        const bool finished = std::any_of(steps.begin(), steps.end(), [](const auto& step) {
            return step.finish_reason.has_value();
        });
        if (finished || current.hits != last_cache_.hits || current.stores != last_cache_.stores) {
            trace_prefix_cache(current);
            last_cache_ = current;
        }
    }

private:
    bool trace_cache_;
    DecoderPrefixCacheStats last_cache_;
    std::size_t last_bytes_ = 0;
};

inline BatchProfileStep batch_profile_step(std::uint64_t elapsed, int width,
                                           const BatchTrace::Steps& steps,
                                           const DecoderBatchStepStats& stats) {
    const int decode = std::count_if(steps.begin(), steps.end(),
                                     [](const auto& step) { return step.decode; });
    const int rows = std::accumulate(stats.prefill_rows.begin(), stats.prefill_rows.end(), 0);
    return {elapsed, width, rows, decode, stats.graph_slots};
}

}
