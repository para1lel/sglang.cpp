#include "sglang/runtime/execution/batch_admission_order.h"
#include <algorithm>
#include <numeric>

namespace sglang {

BatchAdmissionOrder::BatchAdmissionOrder(int capacity)
    : capacity_(std::clamp(capacity, 0, decoder_batch_max_lanes)) {
    std::iota(slots_.begin(), slots_.begin() + capacity_, 0);
}

void BatchAdmissionOrder::admitted(int slot) {
    const auto end = slots_.begin() + capacity_;
    const auto previous = std::find(slots_.begin(), end, slot);
    std::rotate(previous, previous + 1, end);
}

const int* BatchAdmissionOrder::begin() const { return slots_.begin(); }

const int* BatchAdmissionOrder::end() const { return slots_.begin() + capacity_; }

}
