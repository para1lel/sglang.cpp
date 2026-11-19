#pragma once

#include "sglang/decoder_batch.h"

namespace sglang::server {

class BatchService {
public:
    BatchService(const Decoder& decoder, bool trace, bool cuda_graphs = false, int mtp_steps = 0,
                 int max_active = decoder_batch_max_mtp_lanes, int prefix_cache_entries = 0);
    ~BatchService();

    DecoderResult generate(int slot, const std::vector<int>& prompt, const DecoderOptions& options,
                           const DecoderCallbacks& callbacks);

private:
    struct State;
    std::unique_ptr<State> state_;
};

void trace_batch_step(int width, int decode, std::size_t request_bytes);

}
