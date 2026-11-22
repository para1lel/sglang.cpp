#pragma once

#include "sglang/decoder.h"
#include "sglang/decoder_batch.h"
#include "sglang/decoder_constraint.h"
#include "sglang/decoder_cuda.cuh"
#include "sglang/decoder_sampling.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace sglang {
class Checkpoint;
struct DecoderBatchOptions;
struct DecoderBatchStep;
}

namespace sglang::decoder {

class MtpDraftWeights;
class MtpDraftRequest;
class PrefillWorkspace;
struct PrefillGrant;
struct PromptLogprobCapture;
struct MediaPrefill;
struct MultimodalInput;
class TargetVerification;
struct VerificationSequence;
struct VerificationRow;
struct VerificationGraph;
struct MtpRound;
struct GeneratedLogprobReadback;

using Bf16 = __nv_bfloat16;

class SerialOutput {
public:
    SerialOutput(const DecoderCallbacks& callbacks, int top_count, DecoderResult& result);
    const std::function<bool()>& cancel_probe() const { return callbacks_.is_cancelled; }
    bool cancelled() const { return cancel_probe() && cancel_probe()(); }
    bool emit(int token, const Bf16* logits, const float* adjusted_logits,
              const std::vector<DecoderLogit>* repetition_logits);

private:
    const DecoderCallbacks& callbacks_;
    const int top_count_;
    DecoderResult& result_;
    std::vector<DecoderLogit> ranking_;
};

inline constexpr int hidden_size = 5120;
inline constexpr int intermediate_size = 17408;
inline constexpr int vocab_size = 248320;
inline constexpr int layer_count = 64;
inline constexpr int full_attention_interval = 4;
inline constexpr int full_attention_layers = layer_count / full_attention_interval;
inline constexpr int gdn_layers = layer_count - full_attention_layers;
inline constexpr int prefill_chunk_rows = text_embedding_max_rows;
inline constexpr std::size_t gdn_history_elements = text_gdn_qkv_width * text_gdn_history_taps;
inline constexpr std::size_t gdn_state_elements =
    text_gdn_value_heads * text_gdn_head_dim * text_gdn_head_dim;

constexpr bool is_full_attention(int layer_index) {
    return (layer_index + 1) % full_attention_interval == 0;
}

void check(cudaError_t status, const char* operation);
struct CudaError : std::runtime_error {
    CudaError(cudaError_t status, const char* operation);
    static bool terminal(cudaError_t error) {
        switch (error) {
            case cudaErrorCudartUnloading:
            case cudaErrorECCUncorrectable:
            case cudaErrorIllegalAddress:
            case cudaErrorLaunchTimeout:
            case cudaErrorContextIsDestroyed:
            case cudaErrorAssert:
            case cudaErrorHardwareStackError:
            case cudaErrorIllegalInstruction:
            case cudaErrorMisalignedAddress:
            case cudaErrorInvalidAddressSpace:
            case cudaErrorInvalidPc:
            case cudaErrorLaunchFailure:
                return true;
            default:
                return false;
        }
    }
    cudaError_t status;
};

std::size_t available_device_bytes();
inline constexpr std::size_t decoder_shared_reserve = std::size_t{2} * 1024 * 1024 * 1024;
void validate_batch_options(const DecoderBatchOptions& options);
int reserved_verification_rows(int output, int mtp_steps);
std::size_t request_resident_bytes(int capacity, int verification_rows, bool draft);
int limit_resident_output(int prompt, const DecoderOptions& options, std::size_t total_bytes,
                          std::size_t free_bytes, std::size_t extra_bytes);
void validate_request(const std::vector<int>& prompt, const DecoderOptions& options,
                      bool allow_mtp = false);

class DeviceScope {
public:
    explicit DeviceScope(int device);
    ~DeviceScope();
    DeviceScope(const DeviceScope&) = delete;
    DeviceScope& operator=(const DeviceScope&) = delete;

    void restore();

private:
    int previous_ = 0;
    bool changed_ = false;
};

struct TransferBuffers {
    std::unique_ptr<Checkpoint> checkpoint;
    std::vector<std::int32_t> logical_pages;
    int winner = -1;
    std::vector<Bf16> logits;
    std::vector<float> adjusted_logits;
    std::vector<DecoderLogit> repetition_logits;
    std::shared_ptr<void> verification_results;
    std::shared_ptr<void> mtp_constraint_inputs;
};

struct DeviceBlock {
    void* pointer;
    std::size_t bytes;
};

class RequestBlockCache;

class Allocations {
public:
    explicit Allocations(int device, RequestBlockCache* cache = nullptr);
    ~Allocations();
    Allocations(const Allocations&) = delete;
    Allocations& operator=(const Allocations&) = delete;

    template <typename Element>
    Element* allocate(std::size_t count) {
        if (count > std::numeric_limits<std::size_t>::max() / sizeof(Element)) {
            throw std::overflow_error("CUDA allocation size overflow");
        }
        return static_cast<Element*>(allocate_bytes(count * sizeof(Element)));
    }

    std::size_t bytes() const;
    TransferBuffers& transfers();
    TextLogitCandidate* prepare_packed_winners();
    GeneratedLogprobReadback* prepare_generated_logprobs();
    ExactGqaWorkspace prepare_singleton_scores();
    cudaStream_t stream() const;
    cudaGraphExec_t graph() const;
    void capture_graph(const std::function<void()>& enqueue);
    void discard_graph();
    void free_block(void* pointer, std::size_t bytes);
    bool replay_batch_graph(int width, const std::function<void()>& prepare,
                            const std::function<void()>& capture,
                            const std::function<bool()>& bind, bool target_verify = false);
    bool replay_packed_graph(int bucket, int rows, const std::function<void()>& prepare,
                             const std::function<void()>& capture,
                             const std::function<bool()>& bind);
    bool verification_graph_failed() const;
    void reclaim_deferred();
    void open_stream(std::shared_ptr<const void> owners = nullptr);
    void close_stream();
    void close();
    void recycle_after_completion();

private:
    friend class RequestBlockCache;
    struct State;
    void* allocate_bytes(std::size_t bytes);
    bool replay_graph_slot(int index, int width, const std::function<void()>& prepare,
                           const std::function<void()>& capture,
                           const std::function<bool()>& bind, bool target_verify);

    std::unique_ptr<State> state_;
};

class RequestBlockCache {
public:
    explicit RequestBlockCache(int device);
    ~RequestBlockCache();
    RequestBlockCache(const RequestBlockCache&) = delete;
    RequestBlockCache& operator=(const RequestBlockCache&) = delete;

    std::size_t bytes() const;
    std::size_t unavailable_bytes() const;
    void clear();

private:
    friend class Allocations;
    bool accepts(int device) const;
    void* take(std::size_t bytes);
    bool retain(DeviceBlock block);
    void quarantine(Allocations::State& state);

    static constexpr std::size_t max_bytes = 768 * 1024 * 1024;
    static constexpr std::size_t max_blocks = 384;
    std::unique_ptr<Allocations::State> state_;
    std::size_t quarantined_bytes_ = 0;
    bool reusable_ = true;
};

std::size_t reserve_request_memory(RequestBlockCache& retired, DecoderOptions& request,
                                   int prompt, std::size_t capacity, std::size_t charged,
                                   std::size_t extra_bytes);

struct Fp8Weight {
    const std::uint8_t* values;
    const Bf16* scales;
    int rows;
    int columns;
};

struct Bf16Weight {
    const Bf16* values;
    int rows;
    int columns;
};

struct GdnLayer {
    Fp8Weight qkv;
    Fp8Weight z;
    Bf16Weight b;
    Bf16Weight a;
    TextGdnWeights recurrent;
    Fp8Weight output;
};

struct GqaLayer {
    Fp8Weight q_gate;
    Fp8Weight key;
    Fp8Weight value;
    const Bf16* query_norm;
    const Bf16* key_norm;
    Fp8Weight output;
};

struct Layer {
    const Bf16* input_norm;
    const Bf16* attention_norm;
    Fp8Weight mlp_gate;
    Fp8Weight mlp_up;
    Fp8Weight mlp_down;
    GdnLayer gdn;
    GqaLayer gqa;
};

struct LayerResidual {
    const Bf16* branch;
    Bf16* residual;
};

template <typename Normalize, typename Gqa, typename Gdn, typename Mlp>
void execute_layers_with_attention_input(const std::array<Layer, layer_count>& layers,
                                         Normalize normalize, Gqa gqa, Gdn gdn, Mlp mlp) {
    int gqa_index = 0;
    int gdn_index = 0;
    for (int index = 0; index < layer_count; ++index) {
        const auto& layer = layers[index];
        const auto input = [first = index == 0](const auto& work) {
            return first ? LayerResidual{work.residual, nullptr}
                         : LayerResidual{work.hidden, work.residual};
        };
        if (is_full_attention(index)) {
            gqa(layer, gqa_index++, input);
        } else {
            gdn(layer, gdn_index++, input);
        }
        normalize(layer.attention_norm, "Normalize post-attention residual", [](const auto& work) {
            return LayerResidual{work.projected_attention, work.residual};
        });
        mlp(layer);
    }
}

template <typename Normalize, typename Gqa, typename Gdn, typename Mlp>
void execute_layers(const std::array<Layer, layer_count>& layers, Normalize normalize,
                    Gqa gqa, Gdn gdn, Mlp mlp) {
    execute_layers_with_attention_input(
        layers, normalize,
        [&](const Layer& layer, int bank, const auto& input) {
            normalize(layer.input_norm, "Normalize layer input", input);
            gqa(layer.gqa, bank);
        },
        [&](const Layer& layer, int bank, const auto& input) {
            normalize(layer.input_norm, "Normalize layer input", input);
            gdn(layer.gdn, bank);
        }, mlp);
}

struct Workspace {
    Workspace() = default;
    explicit Workspace(Allocations& allocations, Bf16* verification_attention = nullptr,
                       TextLogitCandidate* verification_winner = nullptr);

    ExactGqaWorkspace gqa_scores(int capacity) const;

    Bf16* hidden = nullptr;
    Bf16* normalized = nullptr;
    Bf16* residual = nullptr;
    Bf16* projected_attention = nullptr;
    Bf16* attention = nullptr;
    Bf16* mlp_gate = nullptr;
    Bf16* mlp_up = nullptr;
    Bf16* q_gate = nullptr;
    Bf16* keys = nullptr;
    Bf16* values = nullptr;
    Bf16* rope_queries = nullptr;
    Bf16* rope_keys = nullptr;
    Bf16* gdn_qkv = nullptr;
    Bf16* gdn_z = nullptr;
    Bf16* gdn_b = nullptr;
    Bf16* gdn_a = nullptr;
    Bf16* gdn_scratch = nullptr;
    Bf16* logits = nullptr;
    TextFp8Activation activation{};
    TextLogitCandidate* argmax_scratch = nullptr;
    TextLogitCandidate* argmax = nullptr;
};

using GdnBank = std::array<TextGdnState, gdn_layers>;
void allocate_gdn(Allocations& memory, GdnBank& states);

struct ConstraintWorkspace {
    int* ids = nullptr;
    Bf16* logits = nullptr;
    std::uint32_t* mask = nullptr;
    Bf16* masked_logits = nullptr;

    void allocate(Allocations& memory, bool wide);
    std::size_t bytes() const;
};

struct ConstraintCandidates {
    int count;
    const int* ids;
    const std::uint32_t* mask;
};

struct LogitSelection {
    const Bf16* logits;
    TextLogitCandidate* argmax_scratch;
    TextLogitCandidate* argmax;
};

void select_constrained(LogitSelection work, const ConstraintWorkspace& scratch,
                        ConstraintCandidates allowed, const SamplingParameters* sampling,
                        int position, void* sampling_workspace, cudaStream_t stream);

void select_constrained(Workspace& work, const ConstraintWorkspace& scratch,
                        const ConstraintCursor& cursor, const SamplingParameters* sampling,
                        int position, void* sampling_workspace, cudaStream_t stream);

struct Request {
    explicit Request(int device, bool reusable = false, RequestBlockCache* cache = nullptr);
    Request(int device, int capacity, std::shared_ptr<const void> owners, bool reusable = false,
            int kv_limit = 0);

    void initialize(int capacity, cudaStream_t stream, int kv_limit = 0);
    void grow_kv(int positions, int written, cudaStream_t stream);
    void reset(cudaStream_t stream);
    void reset();
    void prepare_minimum_stops(const DecoderOptions& options, cudaStream_t stream);
    void prepare_logit_adjustments(const DecoderOptions& options, cudaStream_t stream);
    ExactGqaWorkspace batch_graph_scores();

    std::vector<std::uint32_t> minimum_stop_upload;
    Allocations memory;
    Workspace work;
    int* decode_position;
    Mxfp8PageTable pages;
    std::array<Mxfp8KvCache, full_attention_layers> caches;
    int kv_limit = 0;
    GdnBank states;
    std::optional<ExactGqaWorkspace> long_graph_scores;
    std::byte* sampling_workspace = nullptr;
    ConstraintWorkspace constraint_workspace;
    const DecoderLogitBias* logit_bias = nullptr;
    const std::uint32_t* minimum_stop_mask = nullptr;
    OutputPenalties output_penalties;
};

std::size_t logit_adjustment_bytes(const DecoderOptions& options);

struct AdjustedLogitsWorkspace {
    float* logits = nullptr;
    std::uint32_t* mask = nullptr;
    std::byte* sampling = nullptr;
    std::size_t sampling_bytes = 0;
    std::byte* repetition = nullptr;
    std::size_t repetition_bytes = 0;
    bool repetition_ranked = false;

    void prepare(Allocations& memory, Request& request, const DecoderOptions& options,
                 cudaStream_t stream);
    void select(const Request& request, Workspace& work, const DecoderOptions& options,
                const ConstraintCursor* allowed, const SamplingParameters* parameters,
                int position, int previous_token, cudaStream_t stream,
                GeneratedLogprobReadback* logprobs = nullptr) const;
    void apply(const Request& request, const Workspace& work, const DecoderOptions& options,
               const std::uint32_t* allowed, int generated, int previous_token,
               cudaStream_t stream) const;
    void select_repetition(const Request& request, const Workspace& work,
                           const DecoderOptions& options, const std::uint32_t* allowed,
                           int previous_token, RepetitionSelection selection,
                           cudaStream_t stream) const;
};

struct RequestProgress {
    // Target prefix length, excluding unconsumed verification rows.
    int position = 0;
    int generated = 0;
    int last_token = -1;
    ConstraintCursor constraint{};

    static int capacity(int prompt, const DecoderOptions& options) {
        return prompt + std::max(options.max_new_tokens,
                                 int(options.score_prompt || options.label_token_ids)) - 1;
    }
    int remaining_prompt(int prompt) const { return position < prompt ? prompt - position : 0; }
    int remaining_output(const DecoderOptions& options) const {
        return options.max_new_tokens - generated;
    }
    void advance(int rows) { position += rows; }
    void select(Request& request, Workspace& work, const DecoderOptions& options,
                const ConstraintWorkspace& scratch, const SamplingParameters* sampling,
                void* sampling_workspace, cudaStream_t stream,
                const AdjustedLogitsWorkspace* adjusted = nullptr);
    std::optional<DecoderFinishReason> commit(const DecoderOptions& options, int token,
                                              const Bf16* logits = nullptr,
                                              SerialOutput* output = nullptr,
                                              const float* adjusted_logits = nullptr,
                                              const std::vector<DecoderLogit>* ranked = nullptr);
};

int validate_winner(int winner, bool sampled, bool constrained);
int read_token(Request& request, bool sampled, bool constrained = false);

struct ForwardLane {
    Request* request;
    int token_id;
    int position;
    TextGdnMode mode;
    bool narrow_gqa = false;
    const Bf16* feature = nullptr;
    const MropePosition* mrope = nullptr;
    int rotary_position = -1;
    bool long_gqa = false;
    bool packed = false;
    ExactGqaWorkspace singleton_scores{};
};

// Packed decode runs one padded row per lane. Buckets bound graph executables; eager packed steps
// pad to the same bucket so both routes enqueue identical work.
inline constexpr int packed_decode_min_lanes = 2;
inline constexpr std::array<int, 12> packed_decode_buckets{1, 2, 4, 8, 12, 16, 24, 32,
                                                           40, 48, 56, 64};
inline constexpr int packed_decode_max_rows = decoder_batch_max_lanes;

inline constexpr int singleton_score_min_visible = 32768;
inline constexpr int singleton_score_max_visible = 36864;
static_assert(singleton_score_max_visible % mxfp8_page_size == 0 &&
              singleton_score_max_visible <= text_attention_max_tokens);

struct BatchDecodeBinding {
    TextLogitCandidate input;
    int position;
    ExactGqaWorkspace scores;
    Mxfp8PageTable pages;
    std::array<Mxfp8KvCache, full_attention_layers> caches;
    std::array<TextGdnState, gdn_layers> states;
};

struct BatchGraphLane {
    Workspace* work;
    const BatchDecodeBinding* binding;
};

struct BatchGraphScratch {
    explicit BatchGraphScratch(int device);

    Allocations memory;
    std::array<std::optional<Workspace>, decoder_batch_max_graph_lanes> work;
    BatchDecodeBinding* device_bindings;
    std::array<BatchDecodeBinding, decoder_batch_max_graph_lanes> bindings{};
};

}

namespace sglang {

struct Decoder::Model {
    Model(const std::filesystem::path& checkpoint, int cuda_device, Fp8PrefillPolicy fp8_prefill);

    Fp8PrefillPolicy fp8_prefill() const { return fp8_prefill_; }
    bool is_b200() const { return b200_; }
    void forward(decoder::Request& request, int token_id, int position, TextGdnMode mode) const;
    void forward(const decoder::ForwardLane* lanes, int count, cudaStream_t stream) const;
    bool verify_mtp(decoder::MtpRound* rounds, int count,
                    const decoder::ConstraintWorkspace& scratch, void* sampling_scratch,
                    cudaStream_t stream, decoder::VerificationGraph* graph = nullptr) const;
    bool forward_verify(const decoder::VerificationSequence* sequences, int count,
                        cudaStream_t stream, decoder::VerificationGraph* graph = nullptr) const;
    void enqueue_verification(const decoder::VerificationRow* rows, int count,
                              cudaStream_t stream) const;
    void prefill_chunk(decoder::Request& request, const int* tokens, int rows, int position,
                       decoder::PrefillWorkspace& workspace,
                       decoder::MtpDraftRequest* draft = nullptr, int prompt_remaining = 0,
                       const decoder::MediaPrefill* media = nullptr,
                       const decoder::PromptLogprobCapture* scores = nullptr,
                       bool exact = false) const;
    void prefill_grants(const decoder::PrefillGrant* grants, int count, cudaStream_t stream,
                        bool tensor_cache, bool layerwise, bool mixed,
                        DecoderBatchStepStats& stats, bool label_scores = false) const;
    void decode_graph(decoder::Request& request) const;
    void capture_batch_decode(const decoder::BatchGraphLane* lanes, int count,
                              cudaStream_t stream) const;
    using DecodeHeadRows = std::array<decoder::Workspace*, 4>;
    void compute_decode_logits(const DecodeHeadRows& rows, cudaStream_t stream) const;
    bool compute_eager_decode_logits(const decoder::ForwardLane* lanes, int count,
                                   cudaStream_t stream) const;
    void seed_draft(decoder::Workspace& work, decoder::MtpDraftRequest& draft, int position,
                    int shifted_token, cudaStream_t stream) const;
    void normalize_final(decoder::Workspace& work, cudaStream_t stream) const;
    void compute_prefill_logits(decoder::Workspace& work, cudaStream_t stream) const;
    void enqueue_label_logits(decoder::Request& request, cudaStream_t stream) const;
    void compute_logits(decoder::Request& request, bool decode = false) const;
    void compute_logits(decoder::Workspace& work, cudaStream_t stream, bool decode = false) const;
    bool prefill_serial(decoder::Request& request, decoder::RequestProgress& progress,
                        const std::vector<int>& prompt, const decoder::SerialOutput& output,
                        DecoderResult& result, const decoder::MediaPrefill* media,
                        int step_budget) const;
    void advance_serial(decoder::Request& request, decoder::RequestProgress& progress,
                        int input, bool media, int rotary_delta) const;
    void generate_serial(const std::vector<int>& prompt, const DecoderOptions& options,
                         const DecoderCallbacks& callbacks, DecoderResult& result,
                         std::shared_ptr<const decoder::MultimodalInput> media = nullptr,
                         const std::vector<int>* teacher = nullptr) const;
    DecoderFinishReason generate_mtp(const std::vector<int>& prompt,
                                     const DecoderOptions& options,
                                     const DecoderCallbacks& callbacks,
                                     DecoderResult& result) const;

    template <typename Lane>
    void forward_layers(const Lane* lanes, int count, cudaStream_t stream) const;

    int device;
    std::string accelerator_name;
    std::shared_ptr<decoder::Allocations> memory;
    std::array<decoder::Layer, decoder::layer_count> layers{};
    const decoder::Bf16* embedding = nullptr;
    const decoder::Bf16* final_norm = nullptr;
    decoder::Bf16Weight lm_head{};
    std::shared_ptr<const decoder::MtpDraftWeights> mtp_weights;
    mutable std::atomic_flag request_in_use = ATOMIC_FLAG_INIT;
    mutable std::unique_ptr<decoder::Request> decode_request;

private:
    Model(const std::filesystem::path& checkpoint, int cuda_device);

    Fp8PrefillPolicy fp8_prefill_ = Fp8PrefillPolicy::exact;
    bool b200_ = false;
};

// Names the private decoder model for the decode-step runner in src/text/.
struct DecoderModelAccess {
    using Model = Decoder::Model;
};

}

namespace sglang::decoder {

class PackedDecode;

using DecoderModel = DecoderModelAccess::Model;

struct DecodeGraphs {
    DecodeGraphs();
    ~DecodeGraphs();
    DecodeGraphs(DecodeGraphs&&) noexcept;
    DecodeGraphs& operator=(DecodeGraphs&&) noexcept;

    std::unique_ptr<BatchGraphScratch> narrow;
    std::array<bool, decoder_batch_max_graph_lanes> narrow_warmed{};
    std::unique_ptr<PackedDecode> packed;
    bool packed_unavailable = false;
};

struct DecodeRun {
    std::array<Workspace*, decoder_batch_max_lanes> ready{};
    std::array<const TextLogitCandidate*, decoder_batch_max_lanes> winners{};
    std::uint64_t graphed = 0;
};

struct DecodeRunOptions {
    bool cuda_graphs;
    bool graph_step;
    bool narrow_allowed;
    bool packed;
    std::uint64_t greedy_winners;
    std::uint64_t eager_score_rows;
};

// Enqueues one decode step for lanes. ready[index] names the workspace that already holds logits
// and the greedy winner for lanes[index]; a null entry leaves scoring to the caller.
DecodeRun run_decode_lanes(const DecoderModel& model, Allocations& execution,
                           DecodeGraphs& graphs, ForwardLane* lanes, int count,
                           DecodeRunOptions options);

}
