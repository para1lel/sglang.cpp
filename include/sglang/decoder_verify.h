#pragma once

#include "sglang/decoder_internal.h"

namespace sglang::decoder {

struct DraftDepthScratch;
struct MtpDraftInput;
struct SamplingParameters;
struct VerificationGraphBinding;
struct VerificationWorkspace;
struct GdnVerificationLane;

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
    bool constrained = false;
    int rotary_position = -1;
};

struct VerificationSequence {
    TargetVerification* verification;
    int position;
    const int* tokens;
    int count;
    bool constrained = false;
    int rotary_delta = 0;
};

inline constexpr int k1_verification_rows = 2;
static_assert(wide_mtp_lanes == 2 * decoder_batch_max_mtp_lanes);
inline constexpr int verification_packet_rows =
    decoder_batch_max_mtp_lanes * target_verify_max_rows;
inline constexpr int wide_mtp_rows = 2 * verification_packet_rows;
inline constexpr int max_wide_mtp_rows = max_wide_mtp_lanes * target_verify_max_rows;

void project_verification_bf16(Bf16Weight weight, const VerificationRow* rows, int count,
                               Bf16* Workspace::*output, cudaStream_t stream);

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
    // Packet-capacity staging for the single-read BF16 head projection:
    // packed inputs followed by packed logits, carved from one allocation
    // owned by this workspace's arena (freed with it, never resized).
    Bf16* bf16_head_packed = nullptr;
    Bf16* bf16_head_packed_logits = nullptr;
};

struct VerificationGraphBinding {
    std::array<TextLogitCandidate, k1_verification_rows> inputs;
    std::array<int, k1_verification_rows> positions;
    std::array<int, k1_verification_rows> rotary_positions;
    std::array<ExactGqaWorkspace, k1_verification_rows> scores;
    Mxfp8PageTable pages;
    std::array<Mxfp8KvCache, full_attention_layers> caches;
    GdnBank initial;
    GdnBank tail;
    std::array<TextModelStateRow, k1_verification_rows> final_states;
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

// Holds candidate uploads for draft and target scores on one speculative branch.
class MtpConstraintRound {
public:
    MtpConstraintRound(Request& request, const TokenConstraint& constraint, int rows);

    void begin(const ConstraintCursor& committed, int remaining);
    bool advance(int row, int token);
    const ConstraintCandidates& candidates(int row) const { return rows_[row]; }
    int rows() const { return prepared_rows_; }

private:
    void prepare(int row, int remaining);

    const TokenConstraint* constraint_;
    ConstraintCursor cursor_;
    std::byte* staging_;
    std::size_t stride_;
    int remaining_ = 0;
    int prepared_rows_ = 0;
    std::array<ConstraintCandidates, target_verify_max_rows> rows_{};
};

struct MtpRound {
    MtpDraftRequest* draft;
    TargetVerification* target;
    Request* request;
    MtpConstraintRound* constraint;
    const SamplingParameters* sampling;
    std::array<int, target_verify_max_rows> tokens;
    int position;
    int first_output;
    int remaining;
    int proposal_limit;
    const Bf16* draft_hidden = nullptr;
    int drafted = 0;
    int consumed = 0;

    const MediaPrefill* media = nullptr;

    int rotary_delta() const;
    MtpDraftInput draft_input(int row_position, int token, const Bf16* hidden) const;
    using Emit = std::function<bool(int, const Bf16*, bool, int)>;
    using Cancel = std::function<bool()>;

    int rows() const;
    static bool propose_all(MtpRound* rounds, int count, const ConstraintWorkspace& scratch,
                            cudaStream_t stream, const Cancel& cancel);
    static bool propose_packets(MtpRound* rounds, int count, const ConstraintWorkspace& scratch,
                                cudaStream_t stream, const Cancel& cancel);
    static void propose_depth(MtpRound* rounds, int count, const DraftDepthScratch& scratch,
                              cudaStream_t stream);
    int text_stop_after(const DecoderOptions& options, const RequestProgress& progress) const;
    bool commit(const Emit& emit, cudaStream_t stream, const Cancel& cancel,
                MtpDraftInput* repair, const Workspace** committed_row = nullptr);
};

class TargetVerification {
public:
    TargetVerification(Request& request, int capacity, int prefix, int rows, bool logits,
                       cudaStream_t stream = nullptr,
                       std::shared_ptr<TextLogitCandidate> winners = nullptr,
                       int generated_top_logprobs = -1, bool rank_local_gdn = false);

    void begin(int position, int count, cudaStream_t stream);
    VerificationRow row(int index);
    const Workspace* checkpoint_row(int index);
    bool graph_eligible(const VerificationSequence& sequence) const;
    void bind_graph(VerificationGraphBinding& binding) const;
    void use_graph_workspace(VerificationWorkspace& workspace);
    void bind_gdn(int layer);
    GdnVerificationLane gdn_lane(int layer, TextGdnWeights weights);
    void advance_gdn(int layer, TextGdnWeights weights);
    void sample(const SamplingParameters& parameters, int first_output, void* scratch);
    void constrain(const MtpConstraintRound& constraint, const ConstraintWorkspace& scratch,
                   const SamplingParameters* sampling, int first_output, void* sampling_scratch);
    void enqueue_generated_logprobs(void* scratch);
    void read();
    static void read_all(TargetVerification* const* verifiers, int count, cudaStream_t stream);
    void commit(int count);

    int winner(int index, bool sampled = false, bool constrained = false) const;
    bool needs_logits() const { return host_logits_ != nullptr; }
    const Bf16* logits(int index) const;
    bool needs_generated_logprobs() const { return host_generated_logprobs_ != nullptr; }
    std::shared_ptr<const DecoderTokenLogprobs> generated_logprobs(int index) const;

    Bf16* bf16_head_staging() const {
        return active_workspace_ ? active_workspace_->bf16_head_packed : nullptr;
    }
    Bf16* bf16_head_logits_staging() const {
        return active_workspace_ ? active_workspace_->bf16_head_packed_logits : nullptr;
    }
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
    GeneratedLogprobReadback* host_generated_logprobs_ = nullptr;
    int generated_top_logprobs_;
    bool rank_local_gdn_ = false;
};

std::shared_ptr<TextLogitCandidate> prepare_verification_winners(int slots);

}
