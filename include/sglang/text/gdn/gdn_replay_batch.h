#pragma once

#include "sglang/text_attention.h"

#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>

namespace sglang {

/// One per-layer GDN prefix replay captured for batched dispatch. The fields
/// mirror the text_gdn_replay_prefix[_local] argument list; count is the
/// accepted prefix length to replay.
struct TextGdnReplayEntry {
    TextGdnReplayRows rows;
    TextGdnWeights weights;
    TextGdnState initial;
    TextGdnState result;
    int count;
};

/// Eligibility for the batched GDN prefix replay: mirrors valid_replay
/// (src/text_attention_gdn.cu), which gates the per-layer launches. The
/// batched kernels dereference exactly the buffers checked here.
bool text_gdn_replay_batchable(int count, const TextGdnReplayEntry& entry);
/// Caller-owned batch of GDN prefix replays: collects per-layer entries across
/// the active lanes of one commit group and dispatches them as a single
/// address-table launch per rank (one recurrent kernel plus one history kernel
/// instead of two launches per layer). Entries preserve the per-layer kernel
/// order and write disjoint state buffers, so the replayed states are identical
/// to per-layer text_gdn_replay_prefix[_local] dispatch. All methods fall back
/// gracefully: a reserve() failure or exhausted capacity makes the caller
/// revert to the per-layer launches.
class TextGdnReplayBatch {
public:
    TextGdnReplayBatch() = default;
    ~TextGdnReplayBatch();
    TextGdnReplayBatch(const TextGdnReplayBatch&) = delete;
    TextGdnReplayBatch& operator=(const TextGdnReplayBatch&) = delete;

    /// Allocate pinned staging and device tables and pre-size the host
    /// staging vector, so add() never allocates on the commit path; false on
    /// any failure (the caller then keeps the per-layer dispatch). Re-reserve
    /// with a larger capacity frees the previous tables.
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

    /// Upload the collected entries and launch the batched replay kernels on
    /// the given stream. rank_local selects the GdnLocal strides (true) or the
    /// full-width GdnFull strides (false), matching the verifier's
    /// rank_local_gdn flag. The batch is reset on success.
    ///
    /// Lifetime: the pinned staging table is owned by this object and is
    /// never overwritten while a previous asynchronous upload is still in
    /// flight. Each launch records a completion event after its host-to-device
    /// copy; the next launch, a re-reserve, and destruction all wait on that
    /// event first, so back-to-back commits without an intervening fence
    /// cannot race the host-side table write.
    cudaError_t launch(bool rank_local, cudaStream_t stream);

private:
    std::vector<TextGdnReplayEntry> entries_;
    TextGdnReplayEntry* pinned_ = nullptr;
    TextGdnReplayEntry* device_ = nullptr;
    int capacity_ = 0;
    cudaEvent_t copy_done_ = nullptr;
    bool copy_pending_ = false;
};

/// Preflight every entry before any table write, then fill the table: the
/// batch covers every layer of the caller's verifier or none of them, so a
/// mid-table decline can never strand a partial batch next to the per-layer
/// fallback. source(layer) returns that layer's TextGdnReplayEntry; a false
/// return means the caller must replay per layer instead.
template <typename Source>
bool try_fill_gdn_replay_batch(TextGdnReplayBatch* batch, int count, int layers,
                               Source&& source) {
    if (batch == nullptr || batch->available() < layers) return false;
    for (int layer = 0; layer < layers; ++layer) {
        if (!text_gdn_replay_batchable(count, source(layer))) return false;
    }
    for (int layer = 0; layer < layers; ++layer) {
        // Cannot fail after the preflight above; a false return would mean a
        // partially filled table, so fail loudly instead of replaying a subset.
        if (!batch->add(source(layer))) {
            throw std::logic_error("Collect batched GDN prefix replay");
        }
    }
    return true;
}
}
