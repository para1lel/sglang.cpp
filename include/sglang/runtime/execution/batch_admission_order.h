#pragma once

#include "sglang/decoder_batch.h"

#include <algorithm>
#include <array>
#include <numeric>

namespace sglang {

class BatchAdmissionOrder {
public:
    explicit BatchAdmissionOrder(int capacity)
        : capacity_(std::clamp(capacity, 0, decoder_batch_max_lanes)) {
        std::iota(slots_.begin(), slots_.begin() + capacity_, 0);
    }

    void admitted(int slot) {
        const auto end = slots_.begin() + capacity_;
        const auto previous = std::find(slots_.begin(), end, slot);
        std::rotate(previous, previous + 1, end);
    }

    auto begin() const { return slots_.begin(); }
    auto end() const { return slots_.begin() + capacity_; }

private:
    std::array<int, decoder_batch_max_lanes> slots_{};
    int capacity_;
};

}
