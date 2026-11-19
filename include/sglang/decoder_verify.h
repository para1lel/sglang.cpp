#pragma once

#include "sglang/decoder_internal.h"

namespace sglang::decoder {

inline constexpr int target_verify_max_rows = decoder_max_mtp_steps + 1;
static_assert(target_verify_max_rows <= text_fp8_max_batch);
static_assert(target_verify_max_rows <= text_gdn_max_verify_rows);

struct VerificationRow {
    Request* request;
    Workspace* work;
    int position;
    TargetVerification* verification;
    bool first;
    static constexpr auto mode = TextGdnMode::decode;
};

struct VerificationSequence {
    TargetVerification* verification;
    int position;
    const int* tokens;
    int count;
};

class TargetVerification {
public:
    TargetVerification(Request& request, int capacity, int prefix, int rows, bool logits,
                       cudaStream_t stream = nullptr);

    void begin(int position, int count, cudaStream_t stream);
    VerificationRow row(int index);
    void bind_gdn(int layer);
    void advance_gdn(int layer, TextGdnWeights weights);
    void read();
    static void read_all(TargetVerification* const* verifiers, int count, cudaStream_t stream);
    void commit(int count);

    int winner(int index) const;
    const Bf16* logits(int index) const;

private:
    void enqueue_results();

    struct ReplayLayer {
        TextGdnReplayRows inputs;
        TextGdnWeights weights;
    };

    Request& request_;
    cudaStream_t stream_;
    int capacity_;
    int valid_prefix_;
    int pending_rows_ = 0;
    bool ready_ = false;
    GdnBank tail_;
    std::array<ReplayLayer, gdn_layers> replay_;
    std::vector<Workspace> work_;
    Bf16* observable_;
    TextLogitCandidate* device_winners_;
    TextLogitCandidate* host_winners_;
    Bf16* host_logits_;
};

}
