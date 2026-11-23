#include "sglang/runtime/cache/kv_residency_policy.h"

#include "sglang/decoder_batch.h"

#include <algorithm>

namespace sglang::decoder {

KvResidencyPolicy::KvResidencyPolicy(const DecoderBatchOptions& options)
    : KvResidencyPolicy(Options{options.resident_memory_limit}) {}

KvResidencyPolicy::KvResidencyPolicy(const Options& options)
    : resident_(options.resident_memory_limit.has_value()),
      requested_capacity_(options.resident_memory_limit),
      capacity_(std::nullopt) {}

bool KvResidencyPolicy::resident() const { return resident_; }

std::size_t KvResidencyPolicy::capacity() const { return capacity_.value_or(0); }

void KvResidencyPolicy::measure_available(std::size_t free_bytes) {
    if (!resident_) return;
    capacity_ = std::min(*requested_capacity_,
                         free_bytes - std::min(free_bytes, decoder_shared_reserve));
}

std::pair<int, int> KvResidencyPolicy::request_storage(int prompt, int capacity,
                                                       bool incremental) const {
    if (incremental) return {prompt, capacity};
    return {capacity, 0};
}

}
