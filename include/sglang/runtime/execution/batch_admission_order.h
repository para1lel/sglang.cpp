#pragma once

#include "sglang/decoder_batch.h"

#include <array>

namespace sglang {

class BatchAdmissionOrder {
public:
    explicit BatchAdmissionOrder(int capacity);

    void admitted(int slot);

    const int* begin() const;
    const int* end() const;

private:
    std::array<int, decoder_batch_max_lanes> slots_{};
    int capacity_;
};

}
