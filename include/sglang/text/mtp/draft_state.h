#pragma once

#include "sglang/decoder_cuda.cuh"
#include "sglang/decoder_mtp.h"
#include "sglang/runtime/backend/cuda_backend_policy.h"

namespace sglang::decoder {

bool use_streaming_projection(bool supported, cudaStream_t stream);

struct MtpDraftWeights::State {
    State(const std::filesystem::path& directory, int device,
          std::shared_ptr<const Allocations> target_weights, const Bf16* embedding,
          Bf16Weight lm_head, DraftOperatorCapabilities capabilities);

    int device;
    const DraftOperatorCapabilities capabilities;
    bool streaming_projection = false;
    std::shared_ptr<const Allocations> target_weights;
    Allocations memory;
    const Bf16* embedding;
    Bf16Weight lm_head;
    const std::uint8_t* quantized_head = nullptr;
    const float* quantized_head_maxima = nullptr;
    std::shared_ptr<const TcVocabWeights> shared_vocab;
    const Bf16* embedding_norm;
    const Bf16* hidden_norm;
    Bf16Weight fc;
    const Bf16* input_norm;
    const Bf16* attention_norm;
    GqaLayer gqa;
    Fp8Weight mlp_gate;
    Fp8Weight mlp_up;
    Fp8Weight mlp_down;
    const Bf16* final_norm;
};

struct DraftWorkspace {
    static constexpr int score_storage_count = vocab_size * sizeof(Bf16) / sizeof(float);
    static_assert(score_storage_count * sizeof(float) == vocab_size * sizeof(Bf16));

    explicit DraftWorkspace(Allocations& memory);

    ExactGqaWorkspace attention_scores() const;

    Bf16* embedding;
    Bf16* concatenated;
    Bf16* residual;
    Bf16* normalized;
    Bf16* branch;
    Bf16* q_gate;
    Bf16* keys;
    Bf16* values;
    Bf16* rope_queries;
    Bf16* rope_keys;
    Bf16* attention;
    Bf16* gate;
    Bf16* up;
    float* score_storage;
    Bf16* logits;
    TextFp8Activation activation;
    TextLogitCandidate* argmax_scratch;
    TextLogitCandidate* argmax;
};

struct MtpDraftRequest::State {
    State(std::shared_ptr<const MtpDraftWeights> weights, int capacity, RequestBlockCache* cache,
          PrefixStorage storage);
    void validate(int position, int token, const Bf16* hidden) const;
    void embed(int shifted_token, const Bf16* hidden, cudaStream_t stream,
               const Bf16* feature = nullptr);
    void prepare_attention(cudaStream_t stream);
    void append(int position, bool score, const MropePosition* mrope, int rotary_position,
                cudaStream_t stream);
    DraftAttentionRow attention_row(int position);
    void gate_attention(cudaStream_t stream);
    void prepare_mlp(cudaStream_t stream);
    void activate_mlp(cudaStream_t stream);
    void normalize_output(cudaStream_t stream);
    void score(cudaStream_t stream, const ConstraintWorkspace* scratch,
               const ConstraintCandidates* allowed);
    void select_score(cudaStream_t stream, const ConstraintWorkspace* scratch,
                      const ConstraintCandidates* allowed);
    bool prepare_mapped_score(cudaStream_t stream);
    int score_token();
    static void score_wave(State* const* batch, int count, cudaStream_t stream);
    static void select_wave(State* const* batch, int count, cudaStream_t stream);

    std::shared_ptr<const MtpDraftWeights> weights;
    PrefixPages resident_pages;
    Allocations memory;
    DraftWorkspace work;
    int capacity;
    Mxfp8KvCache cache;
    Mxfp8PageTable pages;
    int prefix = 0;
    bool scored = false;
    bool constrained_score = false;
    bool int8_score = false;
    bool retried_score = false;
    bool mapped_score = false;
    TextLogitCandidate* device_score = nullptr;
    TextLogitCandidate retry_winner{};
    bool failed = false;
};

}
