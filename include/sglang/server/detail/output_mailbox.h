#pragma once

#include "sglang/decoder_batch.h"

#include <array>
#include <cstddef>

namespace sglang::server {

struct OutputMailbox {
    std::array<DecoderBatchStep, 2> steps{};
    std::size_t head = 0;
    std::size_t size = 0;

    bool empty() const;
    bool full() const;
    const DecoderBatchStep& front() const;
    void push(const DecoderBatchStep& step);
    void pop();
};

bool emit_step(const DecoderBatchStep& step, const DecoderCallbacks& callbacks,
               DecoderResult& result);

}
