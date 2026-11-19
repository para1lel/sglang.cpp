#pragma once

#include "sglang/decoder_batch.h"

namespace sglang::server {

struct BatchMetrics {
    DecoderPrefixCacheStats cache;
    int running = 0;
    int cache_capacity = 0;
    std::uint64_t prompt_tokens = 0;
    std::uint64_t generation_tokens = 0;
    std::uint64_t prefill_requests = 0;
};

struct BatchProfileStep {
    std::uint64_t elapsed_ns = 0;
    int width = 0;
    int prefill_rows = 0;
    int decode_lanes = 0;
    std::uint32_t graph_slots = 0;
};

struct BatchProfile {
    std::string name;
    std::vector<BatchProfileStep> steps;
};

class BatchService {
public:
    BatchService(const Decoder& decoder, bool trace, bool cuda_graphs = false, int mtp_steps = 0,
                 int max_active = decoder_batch_max_mtp_lanes, int prefix_cache_entries = 0);
    ~BatchService();

    DecoderResult generate(int slot, const std::vector<int>& prompt, const DecoderOptions& options,
                           const DecoderCallbacks& callbacks);
    void claim_connection(int slot);
    void release_connection(int slot);
    BatchMetrics metrics() const;
    bool reset_prefix_cache();
    bool start_profile(std::string name, std::size_t iterations);
    std::optional<BatchProfile> stop_profile();

private:
    struct State;
    std::unique_ptr<State> state_;
};

void trace_batch_step(int width, int decode, std::size_t request_bytes);

}
