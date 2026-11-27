#pragma once

#include "sglang/text_attention.h"

#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>

namespace sglang {

struct TextGdnReplayEntry {
    TextGdnReplayRows rows;
    TextGdnWeights weights;
    TextGdnState initial;
    TextGdnState result;
    int count;
};

bool text_gdn_replay_batchable(int count, const TextGdnReplayEntry& entry);
class TextGdnReplayBatch {
public:
    TextGdnReplayBatch() = default;
    ~TextGdnReplayBatch();
    TextGdnReplayBatch(const TextGdnReplayBatch&) = delete;
    TextGdnReplayBatch& operator=(const TextGdnReplayBatch&) = delete;

    bool reserve(int capacity);
    int capacity() const { return capacity_; }
    int size() const { return static_cast<int>(entries_.size()); }
    int available() const { return capacity_ - size(); }
    void reset() { entries_.clear(); }
    bool add(const TextGdnReplayEntry& entry) {
        if (available() <= 0) return false;
        entries_.push_back(entry);
        return true;
    }

    cudaError_t launch(bool rank_local, cudaStream_t stream);

private:
    std::vector<TextGdnReplayEntry> entries_;
    TextGdnReplayEntry* pinned_ = nullptr;
    TextGdnReplayEntry* device_ = nullptr;
    int capacity_ = 0;
    cudaEvent_t copy_done_ = nullptr;
    bool copy_pending_ = false;
};

template <typename Source>
bool try_fill_gdn_replay_batch(TextGdnReplayBatch* batch, int count, int layers,
                               Source&& source) {
    if (batch == nullptr || batch->available() < layers) return false;
    for (int layer = 0; layer < layers; ++layer) {
        if (!text_gdn_replay_batchable(count, source(layer))) return false;
    }
    for (int layer = 0; layer < layers; ++layer) {
        if (!batch->add(source(layer))) {
            throw std::logic_error("Collect batched GDN prefix replay");
        }
    }
    return true;
}
}
