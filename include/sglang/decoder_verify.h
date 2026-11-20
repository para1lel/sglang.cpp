#pragma once

#include "sglang/decoder_internal.h"

namespace sglang::decoder {

struct SamplingParameters;
struct VerificationGraphBinding;
struct VerificationWorkspace;

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
    const VerificationGraphBinding* binding = nullptr;
    VerificationWorkspace* graph_workspace = nullptr;
};

struct VerificationSequence {
    TargetVerification* verification;
    int position;
    const int* tokens;
    int count;
};

inline constexpr int k1_verification_rows = 2;

struct VerificationWorkspace {
    VerificationWorkspace(Allocations& memory, int rows);
    void bind_gdn(int layer, int rows);

    struct ReplayLayer {
        TextGdnReplayRows inputs;
        TextGdnWeights weights;
    };

    std::array<ReplayLayer, gdn_layers> replay;
    std::vector<Workspace> work;
    Bf16* observable;
    TextLogitCandidate* winners;
};

struct VerificationGraphBinding {
    std::array<TextLogitCandidate, k1_verification_rows> inputs;
    std::array<int, k1_verification_rows> positions;
    std::array<ExactGqaWorkspace, k1_verification_rows> scores;
    Mxfp8PageTable pages;
    std::array<Mxfp8KvCache, full_attention_layers> caches;
    GdnBank initial;
    GdnBank tail;
};

struct VerificationGraphScratch {
    explicit VerificationGraphScratch(int device);
    void upload(const VerificationSequence& sequence, cudaStream_t stream);

    Allocations memory;
    VerificationWorkspace workspace;
    VerificationGraphBinding* device_binding;
    std::shared_ptr<VerificationGraphBinding> host_binding;
};

struct VerificationGraph {
    explicit VerificationGraph(Allocations& owner) : execution(owner) {}

    Allocations& execution;
    std::unique_ptr<VerificationGraphScratch> scratch;
    bool warmed = false;
};

class TargetVerification {
public:
    TargetVerification(Request& request, int capacity, int prefix, int rows, bool logits,
                       cudaStream_t stream = nullptr);

    void begin(int position, int count, cudaStream_t stream);
    VerificationRow row(int index);
    bool graph_eligible(const VerificationSequence& sequence) const;
    void bind_graph(VerificationGraphBinding& binding) const;
    void use_graph_workspace(VerificationWorkspace& workspace);
    void bind_gdn(int layer);
    void advance_gdn(int layer, TextGdnWeights weights);
    void sample(const SamplingParameters& parameters, int first_output, void* scratch);
    void read();
    static void read_all(TargetVerification* const* verifiers, int count, cudaStream_t stream);
    void commit(int count);

    int winner(int index, bool sampled = false) const;
    const Bf16* logits(int index) const;

private:
    void enqueue_results();
    VerificationWorkspace& workspace() { return *active_workspace_; }

    Request& request_;
    cudaStream_t stream_;
    int capacity_;
    int valid_prefix_;
    int pending_rows_ = 0;
    bool ready_ = false;
    GdnBank tail_;
    std::optional<VerificationWorkspace> eager_workspace_;
    VerificationWorkspace* active_workspace_ = nullptr;
    TextLogitCandidate* host_winners_;
    Bf16* host_logits_;
};

}
