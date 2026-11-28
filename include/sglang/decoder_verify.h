#pragma once

#include "sglang/decoder_internal.h"
#include "sglang/text/gqa/gqa_decode_long.cuh"

namespace sglang::decoder {

struct DraftDepthScratch;
struct MtpDraftInput;
struct SamplingParameters;
struct VerificationGraphBinding;
struct VerificationWorkspace;

inline constexpr int target_verify_max_rows = decoder_max_mtp_steps + 1;
static_assert(target_verify_max_rows <= text_fp8_max_batch);
static_assert(target_verify_max_rows <= text_gdn_max_verify_rows);

enum class VerificationAttention { direct, split, long_context };

struct VerificationRow {
    Request* request;
    Workspace* work;
    int position;
    TargetVerification* verification;
    bool first;
    static constexpr auto mode = TextGdnMode::decode;
    const VerificationGraphBinding* binding = nullptr;
    VerificationAttention graph_attention = VerificationAttention::direct;
    bool constrained = false;
    int rotary_position = -1;
    int binding_lane = 0;
    int binding_row = 0;
    int binding_index = 0;
    gqa_decode::SplitVariant graph_split = gqa_decode::SplitVariant::multi_stage;
    gqa_decode::LongDecodeShape graph_long_shape{};
};

struct VerificationSequence {
    TargetVerification* verification;
    int position;
    const int* tokens;
    int count;
    bool constrained = false;
    int rotary_delta = 0;
    bool exact_head = false;
};

static_assert(wide_mtp_lanes == 2 * decoder_batch_max_mtp_lanes);
inline constexpr int verification_packet_rows =
    decoder_batch_max_mtp_lanes * target_verify_max_rows;
inline constexpr int wide_mtp_rows = 2 * verification_packet_rows;
inline constexpr int max_wide_mtp_rows = max_wide_mtp_lanes * target_verify_max_rows;

void project_verification_bf16(Bf16Weight weight, const VerificationRow* rows, int count,
                               Bf16* Workspace::*output, cudaStream_t stream);

struct VerificationWorkspace {
    VerificationWorkspace(Allocations& memory, int rows, bool rank_local = false,
                          const VerificationWorkspace* shared_head = nullptr);

    TextGdnReplayRows gdn{};
    std::vector<Workspace> work;
    Bf16* observable;
    TextLogitCandidate* winners;
    Bf16* bf16_head_packed = nullptr;
    Bf16* bf16_head_packed_logits = nullptr;
};

struct VerificationGraphBinding {
    int tokens[max_wide_mtp_rows];
    ExactGqaWorkspace scores[max_wide_mtp_rows];
    int rotary[max_wide_mtp_rows];
    gdn_replay_ssm::Lane gdn[gdn_layers][max_wide_mtp_lanes];
    PagedDecodeLane reads[full_attention_layers][max_wide_mtp_rows];
    gqa_decode::LongGraphBinding history[max_wide_mtp_lanes][full_attention_layers];
};

template <>
VerificationGraphBinding* Allocations::allocate<VerificationGraphBinding>(std::size_t count);

struct VerificationGraphLaneShape {
    int rows = 0;
    int split_begin = 0;
    int long_begin = 0;
    bool operator==(const VerificationGraphLaneShape& other) const;
};

struct VerificationGraphShape {
    std::array<VerificationGraphLaneShape, max_wide_mtp_lanes> lanes;
    static constexpr int packets = max_wide_mtp_rows / verification_packet_rows;
    std::array<int, packets> packet_rows{};
    std::array<gqa_decode::SplitVariant, packets> split{};
    std::array<gqa_decode::LongDecodeShape, packets> long_decode{};
    bool packed = false;
    bool operator==(const VerificationGraphShape& other) const;
};

struct VerificationGraphScratch {
    explicit VerificationGraphScratch(int device);
    void prepare(const DecoderModel& model, int lanes, bool packed);
    bool upload(const VerificationSequence* sequences, int count, const VerificationRow* rows,
                int total, const VerificationGraphShape& shape, cudaStream_t stream);

    Allocations memory;
    std::array<std::unique_ptr<Allocations>, max_wide_mtp_lanes> workspace_allocations;
    int prepared_lanes = 0;
    std::array<std::optional<VerificationWorkspace>, max_wide_mtp_lanes> workspaces;
    std::uint8_t* packed_activation = nullptr;
    VerificationGraphBinding* device_binding;
    std::shared_ptr<VerificationGraphBinding> host_binding;
};

struct VerificationGraph {
    explicit VerificationGraph(Allocations& owner);
    bool replay(const DecoderModel& model, const VerificationSequence* sequences, int count,
                VerificationRow* rows, int total, bool packed, cudaStream_t stream);

    Allocations& execution;
    struct Slot {
        VerificationGraphShape shape{};
        std::uint64_t used = 0;
    };
    std::array<Slot, Allocations::verification_graph_slots> slots{};
    std::uint64_t clock = 0;
    std::unique_ptr<VerificationGraphScratch> scratch;
};

class MtpConstraintRound {
public:
    MtpConstraintRound(Request& request, const TokenConstraint& constraint, int rows);

    void begin(const ConstraintCursor& committed, int remaining);
    bool advance(int row, int token);
    const ConstraintCandidates& candidates(int row) const;
    int rows() const;

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
    const DecoderOptions* minimum_options = nullptr;
    const AdjustedLogitsWorkspace* adjustment = nullptr;

    int rotary_delta() const;
    MtpDraftInput draft_input(int row_position, int token, const Bf16* hidden) const;
    using Emit = std::function<bool(int, const Bf16*, bool, int)>;
    using Cancel = std::function<bool()>;

    int rows() const;
    bool masks_stops(int row) const;
    int winner(int row) const;
    static bool propose_all(MtpRound* rounds, int count, const ConstraintWorkspace& scratch,
                            cudaStream_t stream, const Cancel& cancel);
    static bool propose_packets(MtpRound* rounds, int count, const ConstraintWorkspace& scratch,
                                cudaStream_t stream, const Cancel& cancel);
    static void propose_depth(MtpRound* rounds, int count, const DraftDepthScratch& scratch,
                              cudaStream_t stream);
    int text_stop_after(const DecoderOptions& options, const RequestProgress& progress) const;
    bool commit(const Emit& emit, cudaStream_t stream, const Cancel& cancel, MtpDraftInput* repair,
                const Workspace** committed_row = nullptr);
};

class TargetVerification {
public:
    TargetVerification(Request& request, int capacity, int prefix, int rows, bool logits,
                       cudaStream_t stream = nullptr,
                       std::shared_ptr<TextLogitCandidate> winners = nullptr,
                       int generated_top_logprobs = -1, bool rank_local_gdn = false);

    void begin(int position, int count, cudaStream_t stream);
    void cancel_unlaunched();
    bool can_begin(int position, int count, cudaStream_t stream) const;
    VerificationRow row(int index);
    const Workspace* checkpoint_row(int index);
    bool graph_eligible(const VerificationSequence& sequence, bool split_attention) const;
    void use_graph_workspace(VerificationWorkspace& workspace);
    gdn_replay_ssm::Lane gdn_lane(int layer);
    void constrain(const MtpConstraintRound& constraint, const ConstraintWorkspace& scratch,
                   const SamplingParameters* sampling, int first_output, void* sampling_scratch);
    void enqueue_generated_logprobs(int row, void* scratch, const float* adjusted);
    void read();
    static void read_all(TargetVerification* const* verifiers, int count, cudaStream_t stream);
    void commit(int count);

    int winner(int index, bool sampled = false, bool constrained = false) const;
    bool needs_logits() const;
    const Bf16* logits(int index) const;
    bool needs_generated_logprobs() const;
    std::shared_ptr<const DecoderTokenLogprobs> generated_logprobs(int index) const;
    int pending_rows() const;
    bool rank_local_gdn() const;
    cudaStream_t stream() const;

    Bf16* bf16_head_staging() const;
    Bf16* bf16_head_logits_staging() const;
private:
    void enqueue_results();
    VerificationWorkspace& workspace();

    Request& request_;
    cudaStream_t stream_;
    int capacity_;
    int valid_prefix_;
    int pending_rows_ = 0;
    bool ready_ = false;
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
