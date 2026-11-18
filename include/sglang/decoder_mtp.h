#pragma once

#include "sglang/decoder_internal.h"

namespace sglang::decoder {

class MtpDraftWeights {
public:
    MtpDraftWeights(const std::filesystem::path& checkpoint, int device,
                    std::shared_ptr<const Allocations> target_weights, const Bf16* embedding,
                    Bf16Weight lm_head);
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
    MtpDraftRequest(std::shared_ptr<const MtpDraftWeights> weights, int capacity);
    ~MtpDraftRequest();

    /// Append at valid_prefix() using the shifted token and the corresponding
    /// normalized hidden row. The returned device row survives until the next
    /// enqueue writes it. The row can also be passed as that enqueue's input.
    const Bf16* enqueue(int position, int shifted_token, const Bf16* hidden, bool score,
                        cudaStream_t stream);

    /// Read the most recent scored enqueue after its stream has completed.
    int read_token() const;

    int valid_prefix() const;
    /// Shorten logical KV coverage. Subsequent enqueues overwrite the suffix.
    void rewind(int valid_tokens);
    std::size_t device_allocation_bytes() const;

private:
    struct State;
    std::unique_ptr<State> state_;
};

}
