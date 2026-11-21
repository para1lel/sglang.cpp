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

}
