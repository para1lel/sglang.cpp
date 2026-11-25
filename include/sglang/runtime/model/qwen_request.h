#pragma once

#include "sglang/checkpoint.h"
#include "sglang/decoder.h"
#include "sglang/decoder_constraint.h"
#include "sglang/decoder_cuda.cuh"
#include "sglang/decoder_sampling.h"
#include "sglang/runtime/backend/cuda_resources.h"
#include "sglang/runtime/model/qwen_layers.h"

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace sglang::decoder {

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
    int cached_gqa_tokens = 0;
    bool prefix_checkpoint_outputs = false;
    GdnBank states;
    std::optional<ExactGqaWorkspace> long_graph_scores;
    std::byte* sampling_workspace = nullptr;
    ConstraintWorkspace constraint_workspace;
    const DecoderLogitBias* logit_bias = nullptr;
    const std::uint32_t* minimum_stop_mask = nullptr;
    OutputPenalties output_penalties;
};

inline constexpr int singleton_score_min_visible = 32768;
inline constexpr int singleton_score_max_visible = 36864;
static_assert(singleton_score_max_visible % mxfp8_page_size == 0 &&
              singleton_score_max_visible <= text_attention_max_tokens);

}
