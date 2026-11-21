#pragma once

#include "sglang/decoder_batch.h"

#include <array>
#include <cstddef>

namespace sglang::server {

struct OutputMailbox {
    std::array<DecoderBatchStep, 2> steps{};
    std::size_t head = 0;
    std::size_t size = 0;

    bool empty() const { return size == 0; }
    bool full() const { return size == steps.size(); }
    const DecoderBatchStep& front() const { return steps[head]; }

    void push(const DecoderBatchStep& step) {
        steps[(head + size) % steps.size()] = step;
        ++size;
    }

    void pop() {
        head = (head + 1) % steps.size();
        --size;
    }
};

inline bool emit_step(const DecoderBatchStep& step, const DecoderCallbacks& callbacks,
                      DecoderResult& result) {
    result.mtp_rounds += step.mtp_drafted_tokens != 0;
    result.mtp_drafted_tokens += step.mtp_drafted_tokens;
    result.output_token_ids.push_back(*step.token);
    result.mtp_accepted_tokens += step.mtp_accepted_tokens > 0;
    if (callbacks.on_token && !callbacks.on_token(*step.token)) {
        return false;
    }
    for (int index = 0; index < step.extra_token_count; ++index) {
        if (callbacks.is_cancelled && callbacks.is_cancelled()) {
            return false;
        }
        result.output_token_ids.push_back(step.extra_tokens[index]);
        result.mtp_accepted_tokens += index + 1 < step.mtp_accepted_tokens;
        if (callbacks.on_token && !callbacks.on_token(step.extra_tokens[index])) {
            return false;
        }
    }
    return true;
}

}
