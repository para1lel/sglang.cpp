#pragma once

#include "sglang/decoder.h"
#include "sglang/runtime/model/qwen_text_model.h"

#include <array>
#include <cstdint>

namespace sglang {

namespace decoder { class SerialOutput; }

inline constexpr int decoder_batch_max_lanes = 32;
inline constexpr int decoder_default_max_running_requests = 8;
inline constexpr int decoder_batch_max_step_tokens = 32768;
inline constexpr int decoder_batch_max_graph_lanes = 4;
inline constexpr int decoder_batch_max_mtp_lanes = 4;
inline constexpr int decoder_batch_max_mtp_steps = decoder_max_mtp_steps;
inline constexpr int decoder_batch_prefix_max_tokens =
    decoder::QwenTextModel::max_position_embeddings;

struct DecoderBatchPrefixUpdate {
    bool pending = false;
    std::optional<int> restored;
};

struct DecoderBatchStep {
    std::optional<int> token;
    std::optional<DecoderFinishReason> finish_reason;
    bool decode = false;
    std::optional<int> extra_token = std::nullopt;
    int mtp_drafted_tokens = 0;
    int mtp_accepted_tokens = 0;
    int mtp_replayed_rows = 0;
    bool mtp_boundary = false;
    std::array<int, decoder_batch_max_mtp_steps> extra_tokens{};
    int extra_token_count = 0;
    bool sampling_failed = false;
    bool constraint_failed = false;
    int mtp_verified_rows = 0;
    std::shared_ptr<const DecoderTokenLogprobs> generated_logprobs = nullptr;
    std::array<std::shared_ptr<const DecoderTokenLogprobs>, decoder_batch_max_mtp_steps>
        extra_generated_logprobs{};
    std::shared_ptr<const DecoderPromptLogprobs> prompt_logprobs = nullptr;
    std::shared_ptr<const std::vector<double>> label_logprobs = nullptr;
    bool prefix_pending = false;
};

struct DecoderBatchStepStats {
    std::array<int, decoder_batch_max_lanes> prefill_rows{};
    std::size_t prefill_workspace_bytes = 0;
    std::uint32_t graph_slots = 0;
    std::size_t peak_request_bytes = 0;
    int verification_lanes = 0;
    int verification_rows = 0;
    std::size_t sampling_workspace_bytes = 0;
    std::size_t constraint_workspace_bytes = 0;
    int target_rows = 0;
    std::uint32_t verification_graph_slots = 0;
    int layerwise_prefill_groups = 0;
    int layerwise_prefill_rows = 0;
};

struct DecoderBatchOptions {
    bool cuda_graphs = false;
    int max_lanes = decoder_default_max_running_requests;
    int prefix_cache_entries = 0;
    int max_num_batched_tokens = decoder_default_max_num_batched_tokens;
    std::optional<std::size_t> resident_memory_limit = std::nullopt;
    bool tp_verification_graphs = false;
};

struct DecoderPrefixCacheStats {
    std::size_t device_bytes = 0;
    int entries = 0;
    int pages = 0;
    std::uint64_t hits = 0;
    std::uint64_t cached_tokens = 0;
    std::uint64_t prefill_tokens = 0;
    std::uint64_t stores = 0;
    std::uint64_t evictions = 0;
    std::size_t draft_device_bytes = 0;
    std::uint64_t draft_cached_rows = 0;
    std::uint64_t draft_prefill_rows = 0;
    std::uint64_t draft_misses = 0;
    std::uint64_t gqa_restored_tokens = 0;
    std::uint64_t gqa_read_only_tokens = 0;
    std::uint64_t gqa_computed_tokens = 0;
    std::uint64_t gqa_matched_tokens = 0;
    std::uint64_t gdn_matched_tokens = 0;
    std::uint64_t prefill_checkpoints_planned = 0;
    std::uint64_t prefill_checkpoints_resident = 0;
    std::uint64_t prefill_checkpoints_published = 0;
    std::uint64_t prefill_checkpoints_declined = 0;
};

class DecoderBatch {
public:
    explicit DecoderBatch(const Decoder& decoder);
    DecoderBatch(const Decoder& decoder, DecoderBatchOptions options);
    ~DecoderBatch();

    int admit(int slot, std::vector<int> prompt, const DecoderOptions& options);
    int admit_multimodal(int slot, std::vector<int> prompt, const DecoderOptions& options,
                         const std::vector<DecoderMedia>& media);
    void cancel(int slot);
    DecoderBatchPrefixUpdate poll_prefix(int slot);

    std::array<DecoderBatchStep, decoder_batch_max_lanes>
    advance(const std::array<bool, decoder_batch_max_lanes>& selected);

    std::array<DecoderBatchStep, decoder_batch_max_lanes>
    advance_chunked(const std::array<bool, decoder_batch_max_lanes>& selected,
                    DecoderBatchStepStats* stats = nullptr);

    std::size_t request_device_allocation_bytes() const;
    void release_retired_allocations();
    DecoderPrefixCacheStats prefix_cache_stats() const;
    void reset_prefix_cache();
    bool available() const;

private:
    friend class Decoder;
    void generate_mtp(const std::vector<int>& prompt, const DecoderOptions& options,
                      decoder::SerialOutput& output, DecoderResult& result,
                      const std::vector<DecoderMedia>* media = nullptr);
    int admit_impl(int slot, std::vector<int> prompt, const DecoderOptions& options,
                   const std::vector<DecoderMedia>* media);
    struct TpDriver;
    struct State;
    std::unique_ptr<State> state_;
};

}
