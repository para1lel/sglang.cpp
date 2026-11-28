#pragma once

#include "sglang/decoder.h"
#include "sglang/decoder_batch.h"
#include "sglang/decoder_constraint.h"
#include "sglang/decoder_cuda.cuh"
#include "sglang/decoder_sampling.h"
#include "sglang/runtime/backend/cuda_backend_policy.h"
#include "sglang/runtime/backend/cuda_resources.h"
#include "sglang/runtime/cache/kv_residency_policy.h"
#include "sglang/runtime/execution/decoder_execution_policy.h"
#include "sglang/runtime/model/qwen_request.h"
#include "sglang/runtime/model/qwen_weight_shard.h"

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
class TensorParallel;
struct TcVocabScratch;
struct TcVocabWeights;
struct TpPartials;
struct VerificationSequence;
struct VerificationRow;
struct VerificationGraph;
struct MtpRound;
class PeerDraftRepair;
namespace tp_verify_graph { class Graph; }
struct GeneratedLogprobReadback;

class SerialOutput {
public:
    SerialOutput(const DecoderCallbacks& callbacks, int top_count, DecoderResult& result);
    const std::function<bool()>& cancel_probe() const;
    bool cancelled() const;
    bool emit(int token, const Bf16* logits, const float* adjusted_logits,
              const std::vector<DecoderLogit>* repetition_logits);

private:
    const DecoderCallbacks& callbacks_;
    const int top_count_;
    DecoderResult& result_;
    std::vector<DecoderLogit> ranking_;
};

inline constexpr int prefill_chunk_rows = text_embedding_max_rows;

void validate_batch_options(const DecoderBatchOptions& options);
int reserved_verification_rows(int output, int mtp_steps);
std::size_t request_resident_bytes(int capacity, int verification_rows, bool draft);
int limit_resident_output(int prompt, const DecoderOptions& options, std::size_t total_bytes,
                          std::size_t free_bytes, std::size_t extra_bytes);
void validate_request(const std::vector<int>& prompt, const DecoderOptions& options,
                      bool allow_mtp = false);

std::size_t reserve_request_memory(RequestBlockCache& retired, DecoderOptions& request,
                                   int prompt, std::size_t capacity, std::size_t charged,
                                   std::size_t extra_bytes);

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
    
    int position = 0;
    int generated = 0;
    int last_token = -1;
    ConstraintCursor constraint{};

    static int capacity(int prompt, const DecoderOptions& options);
    int remaining_prompt(int prompt) const;
    int remaining_output(const DecoderOptions& options) const;
    void advance(int rows);
    void select(Request& request, Workspace& work, const DecoderOptions& options,
                const ConstraintWorkspace& scratch, const SamplingParameters* sampling,
                void* sampling_workspace, cudaStream_t stream,
                const AdjustedLogitsWorkspace* adjusted = nullptr);
    static std::optional<DecoderFinishReason> finish_token(const DecoderOptions& options,
                                                           ConstraintCursor& constraint,
                                                           int token, int remaining);
    std::optional<DecoderFinishReason> commit(const DecoderOptions& options, int token,
                                              const Bf16* logits = nullptr,
                                              SerialOutput* output = nullptr,
                                              const float* adjusted_logits = nullptr,
                                              const std::vector<DecoderLogit>* ranked = nullptr);
    std::optional<DecoderFinishReason> commit_single(
        const DecoderOptions& options, int token, const Bf16* logits, SerialOutput* output,
        const float* adjusted_logits = nullptr, const std::vector<DecoderLogit>* ranked = nullptr);
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
    ExactGqaWorkspace singleton_scores{};
};

struct BatchDecodeBinding {
    TextLogitCandidate input;
    int position;
    int rotary_position;
    ExactGqaWorkspace scores;
    Mxfp8PageTable pages;
    std::array<Mxfp8KvCache, full_attention_layers> caches;
    GdnReplayBank gdn_replay;
    TextModelStateRow final_state;
};

template <>
BatchDecodeBinding* Allocations::allocate<BatchDecodeBinding>(std::size_t count);

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







struct TpPartials {
    Bf16* attention = nullptr;  
    Bf16* hidden = nullptr;     
};
constexpr int tp_max_forward_lanes = 160;  
}

namespace sglang {

struct Decoder::Model {
    Model(const std::filesystem::path& checkpoint, int cuda_device,
          const decoder::WeightSharding& sharding = decoder::WeightSharding{});

    bool is_b200() const;
    int target_split_end() const;
    void forward(const decoder::ForwardLane* lanes, int count, cudaStream_t stream) const;
    void forward_tp(const Model& peer, const decoder::ForwardLane* primary_lanes,
                    const decoder::ForwardLane* peer_lanes, int count, cudaStream_t stream,
                    cudaStream_t peer_stream, decoder::TensorParallel& group,
                    const decoder::TpPartials& primary_partials,
                    const decoder::TpPartials& peer_partials) const;
    bool verify_mtp(decoder::MtpRound* rounds, int count,
                    const decoder::ConstraintWorkspace& scratch, void* sampling_scratch,
                    cudaStream_t stream, decoder::VerificationGraph* graph,
                    const decoder::TcVocabScratch* vocabulary,
                    std::uint8_t* packed_activation = nullptr,
                    void* logprob_workspace = nullptr) const;
    
    
    
    
    
    
    
    
    
    bool verify_mtp_tp(const Model& peer, decoder::MtpRound* rounds,
                       const decoder::MtpRound* peer_rounds, int count,
                       const decoder::ConstraintWorkspace& scratch, void* sampling_scratch,
                       cudaStream_t stream, cudaStream_t peer_stream,
                       decoder::TensorParallel& group,
                       const decoder::TpPartials& primary_partials,
                       const decoder::TpPartials& peer_partials,
                       const decoder::TcVocabScratch* vocabulary,
                       std::uint8_t* primary_packed, std::uint8_t* peer_packed,
                       void* logprob_workspace,
                       bool capture, decoder::PeerDraftRepair* peer_repair) const;
    bool forward_verify(const decoder::VerificationSequence* sequences, int count,
                        cudaStream_t stream, decoder::VerificationGraph* graph,
                        const decoder::TcVocabScratch* vocabulary,
                        std::uint8_t* packed_activation = nullptr) const;
    void enqueue_verification(const decoder::VerificationRow* rows, int count,
                              cudaStream_t stream,
                              const decoder::TcVocabScratch* vocabulary = nullptr,
                              std::uint8_t* packed_activation = nullptr) const;
    void forward_verification_layers(const decoder::VerificationRow* rows, int count,
                                     cudaStream_t stream, std::uint8_t* packed_activation) const;
    void prefill_chunk(decoder::Request& request, const int* tokens, int rows, int position,
                       decoder::PrefillWorkspace& workspace,
                       decoder::MtpDraftRequest* draft = nullptr, int prompt_remaining = 0,
                       const decoder::MediaPrefill* media = nullptr,
                       const decoder::PromptLogprobCapture* scores = nullptr,
                       int logical_capacity = 0,
                       const decoder::PrefillGrant* checkpoint_grant = nullptr) const;
    void prefill_grants(const decoder::PrefillGrant* grants, int count, cudaStream_t stream,
                        DecoderBatchStepStats& stats) const;
    
    
    
    
    
    void prefill_grants_tp(const Model& peer, const decoder::PrefillGrant* grants,
                           const decoder::PrefillGrant* peer_grants, int count,
                           cudaStream_t stream, cudaStream_t peer_stream,
                           decoder::TensorParallel& group,
                           DecoderBatchStepStats& stats) const;
    void decode_graph(decoder::Request& request) const;
    void capture_batch_decode(const decoder::BatchGraphLane* lanes, int count,
                              cudaStream_t stream) const;
    using DecodeHeadRows = std::array<decoder::Workspace*, 4>;
    using DecodeStateRows = std::array<const TextModelStateRow*, 4>;
    void compute_decode_logits(const DecodeHeadRows& rows, cudaStream_t stream,
                               const DecodeStateRows* states = nullptr) const;
    bool compute_eager_decode_logits(const decoder::ForwardLane* lanes, int count,
                                   cudaStream_t stream) const;
    void seed_draft(decoder::Workspace& work, decoder::MtpDraftRequest& draft, int position,
                    int shifted_token, cudaStream_t stream,
                    const decoder::MediaPrefill* media = nullptr) const;
    void normalize_final(decoder::Workspace& work, cudaStream_t stream,
                         const TextModelStateRow* state = nullptr) const;
    void compute_prefill_logits(decoder::Workspace& work, cudaStream_t stream) const;
    void enqueue_label_logits(decoder::Request& request, cudaStream_t stream) const;
    void compute_logits(decoder::Request& request, bool decode = false) const;
    void compute_logits(decoder::Workspace& work, cudaStream_t stream, bool decode = false,
                        const TextModelStateRow* state = nullptr) const;
    bool prefill_serial(decoder::Request& request, decoder::RequestProgress& progress,
                        const std::vector<int>& prompt, const decoder::SerialOutput& output,
                        DecoderResult& result, const decoder::MediaPrefill* media,
                        int step_budget) const;
    void advance_serial(decoder::Request& request, decoder::RequestProgress& progress,
                        int input, int rotary_delta) const;
    bool serial_graph(const DecoderOptions& options, int prompt_count, int limit) const;
    void generate_serial(const std::vector<int>& prompt, const DecoderOptions& options,
                         const DecoderCallbacks& callbacks, DecoderResult& result,
                         std::shared_ptr<const decoder::MultimodalInput> media = nullptr,
                         const std::vector<int>* teacher = nullptr) const;
    DecoderFinishReason generate_mtp(const std::vector<int>& prompt,
                                     const DecoderOptions& options,
                                     const DecoderCallbacks& callbacks,
                                     DecoderResult& result) const;

    template <typename Lane>
    void forward_layers(const Lane* lanes, int count, cudaStream_t stream,
                        std::uint8_t* packed_activation = nullptr) const;

    
    
    
    template <typename Lane>
    bool forward_layers_tp(const Model& peer, const Lane* primary_lanes,
                           const Lane* peer_lanes, int count, cudaStream_t stream,
                           cudaStream_t peer_stream, decoder::TensorParallel& group,
                           const decoder::TpPartials& primary_partials,
                           const decoder::TpPartials& peer_partials,
                           std::uint8_t* primary_packed = nullptr,
                           std::uint8_t* peer_packed = nullptr,
                           bool singleton_verify = false) const;
    decoder::tp_verify_graph::Graph* forward_verification_tp(
        const Model& peer, const decoder::VerificationRow* rows,
        const decoder::VerificationRow* peer_rows, int count, int total,
        cudaStream_t stream, cudaStream_t peer_stream, decoder::TensorParallel& group,
        const decoder::TpPartials& primary_partials, const decoder::TpPartials& peer_partials,
        std::uint8_t* primary_packed, std::uint8_t* peer_packed, bool greedy, bool capture,
        bool& eager_replicated) const;

    int device;
    std::string accelerator_name;
    decoder::CudaBackendPolicy backend{};
    decoder::DecoderExecutionPolicy execution{};
    std::shared_ptr<decoder::Allocations> memory;
    std::array<decoder::Layer, decoder::layer_count> layers{};
    const decoder::Bf16* embedding = nullptr;
    const decoder::Bf16* final_norm = nullptr;
    decoder::Bf16Weight lm_head{};
    std::shared_ptr<const decoder::MtpDraftWeights> mtp_weights;
    std::shared_ptr<const decoder::TcVocabWeights> tc_vocab;
    
    const TpRuntime* tp = nullptr;
    mutable std::atomic_flag request_in_use = ATOMIC_FLAG_INIT;
    mutable std::unique_ptr<decoder::Request> decode_request;
};






struct Decoder::TpRuntime {
    int device = 0;
    int peer_device = 0;
    cudaStream_t stream = nullptr;
    cudaStream_t peer_stream = nullptr;
    std::unique_ptr<Decoder::Model> peer;
    std::unique_ptr<decoder::TensorParallel> group;
    decoder::TpPartials partials[2]{};
    
    
    
    
    
    std::uint8_t* packed[2] = {};
    mutable std::shared_ptr<decoder::tp_verify_graph::Graph> verification_capture;
    mutable cudaStream_t verification_capture_stream = nullptr;
    mutable bool verification_capture_disabled = false;
    ~TpRuntime();
};


struct DecoderModelAccess {
    using Model = Decoder::Model;
};

}

namespace sglang::decoder {

using DecoderModel = DecoderModelAccess::Model;

struct DecodeGraphs {
    std::unique_ptr<BatchGraphScratch> narrow;
    std::array<bool, decoder_batch_max_graph_lanes> narrow_warmed{};
};

struct DecodeRun {
    std::array<Workspace*, decoder_batch_max_lanes> ready{};
    std::uint64_t graphed = 0;
};

struct DecodeRunOptions {
    bool cuda_graphs;
    bool graph_step;
    bool narrow_allowed;
};



DecodeRun run_decode_lanes(const DecoderModel& model, Allocations& execution,
                           DecodeGraphs& graphs, ForwardLane* lanes, int count,
                           DecodeRunOptions options);

}
