#pragma once

#include "sglang/decoder_internal.h"

namespace sglang::decoder {

struct PrefillProjections;

struct MtpPrefillBuffers {
    Bf16* embedding;
    Bf16* concatenated;
    Bf16* layer_input;
    Bf16* keys;
    Bf16* values;
    Bf16* rope_keys;
};

class MtpDraftWeights {
public:
    MtpDraftWeights(const std::filesystem::path& checkpoint, int device,
                    std::shared_ptr<const Allocations> target_weights, const Bf16* embedding,
                    Bf16Weight lm_head, bool int8_head);
    ~MtpDraftWeights();

    std::size_t device_allocation_bytes() const;

private:
    struct State;
    std::unique_ptr<const State> state_;
    friend class MtpDraftRequest;
};

/// Retain this request in the submitted stream's ownership until completion.
/// Serialize use on one stream and select the weights' CUDA device before enqueueing.
class MtpDraftRequest {
public:
    MtpDraftRequest(std::shared_ptr<const MtpDraftWeights> weights, int capacity,
                    RequestBlockCache* cache = nullptr);
    ~MtpDraftRequest();

    /// Append at valid_prefix() using the shifted token and the corresponding
    /// normalized hidden row. An unscored enqueue only appends KV and returns nullptr.
    /// A scored enqueue returns a device row usable as the next enqueue's input.
    const Bf16* enqueue(int position, int shifted_token, const Bf16* hidden, bool score,
                        cudaStream_t stream, const ConstraintWorkspace* scratch = nullptr,
                        const ConstraintCandidates* allowed = nullptr);

    void seed_prefill(int position, const int* shifted_tokens, const Bf16* hidden,
                      const PrefillProjections& projections, MtpPrefillBuffers buffers);

    /// Read the most recent scored enqueue after its stream has completed.
    int read_token() const;
    bool retry_invalid_int8(cudaStream_t stream);

    int valid_prefix() const;
    /// Shorten logical KV coverage. Subsequent enqueues overwrite the suffix.
    void rewind(int valid_tokens);
    std::size_t device_allocation_bytes() const;
    void recycle_after_completion();

private:
    friend class PrefixCache;
    void snapshot_prefix(Mxfp8KvCache snapshot, int rows, cudaStream_t stream) const;
    void restore_prefix(Mxfp8KvCache snapshot, int rows, cudaStream_t stream);

    struct State;
    std::unique_ptr<State> state_;
};

}
