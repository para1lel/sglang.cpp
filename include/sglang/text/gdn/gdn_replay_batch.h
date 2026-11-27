#pragma once

#include "sglang/text_attention.h"

#include <cuda_runtime.h>
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

namespace gdn_replay {
cudaError_t launch(const TextGdnReplayEntry* table, int count, bool rank_local,
                   cudaStream_t stream);
}

class TextGdnReplayBatch {
public:
    TextGdnReplayBatch() noexcept;
    ~TextGdnReplayBatch();
    TextGdnReplayBatch(const TextGdnReplayBatch&) = delete;
    TextGdnReplayBatch& operator=(const TextGdnReplayBatch&) = delete;

    bool reserve(int capacity);
    int capacity() const;
    int size() const;
    int available() const;
    void reset();
    bool add(const TextGdnReplayEntry& entry);

    cudaError_t launch(bool rank_local, cudaStream_t stream);

private:
    std::vector<TextGdnReplayEntry> entries_;
    TextGdnReplayEntry* pinned_ = nullptr;
    TextGdnReplayEntry* device_ = nullptr;
    int capacity_ = 0;
    cudaEvent_t copy_done_ = nullptr;
    bool copy_pending_ = false;
};

}
