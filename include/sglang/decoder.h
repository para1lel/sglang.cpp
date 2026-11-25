#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <future>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace sglang {

struct DecoderMedia;
class MediaDecoderCheck;

namespace decoder {
struct CudaBackendPolicy;
struct PrefixCacheRead;
struct PrefixDeviceTransfer;
}

inline constexpr int decoder_max_mtp_steps = 4;
inline constexpr int decoder_max_generated_top_logprobs = 20;
inline constexpr std::size_t decoder_prefix_max_salt_bytes = 256;
inline constexpr int decoder_default_max_num_batched_tokens = 32768;

class TokenConstraint;

struct DecoderLogitBias {
    int token_id;
    float bias;
};

struct DecoderOptions {
    int max_new_tokens = 1;
    std::optional<int> stop_token_id;
    int top_logits = 0;
    int mtp_steps = 0;
    std::unordered_set<int> additional_stop_token_ids;
    bool eager = false;
    double temperature = 0.0;
    double top_p = 1.0;
    std::int64_t top_k = -1;
    std::optional<std::int64_t> seed = std::nullopt;
    std::shared_ptr<const TokenConstraint> constraint = nullptr;
    std::string prefix_cache_salt;
    double min_p = 0.0;
    int max_num_batched_tokens = decoder_default_max_num_batched_tokens;
    bool incremental_kv = false;
    int generated_top_logprobs = -1;
    std::shared_ptr<const std::vector<DecoderLogitBias>> logit_bias;
    double frequency_penalty = 0.0;
    double presence_penalty = 0.0;
    double repetition_penalty = 1.0;
    int min_new_tokens = 0;
    bool ignore_eos = false;
    bool score_prompt = false;
    int prompt_logprob_start = 0;
    std::shared_ptr<const std::vector<int>> label_token_ids = nullptr;
    std::function<int(const int*, int, bool)> text_stop = {};
    std::function<bool()> claim_completion = {};
    std::shared_ptr<const decoder::PrefixCacheRead> prefix_cache_read = nullptr;
    std::function<std::shared_future<bool>(const decoder::PrefixDeviceTransfer&)> restore_prefix;
    std::function<bool(std::shared_ptr<void>&)> reserve_prefix_store;
    // Increment the shared counter once per acknowledged STORE, even after lane retirement.
    std::function<void(decoder::PrefixDeviceTransfer&&, const int*, int,
                       std::shared_ptr<void>, std::shared_ptr<std::atomic_uint64_t>)> store_prefix;
    bool publish_prefix_checkpoints = true;

    bool samples() const { return temperature >= 1e-6 && top_k != 1; }

    bool has_output_penalties() const {
        return frequency_penalty != 0 || presence_penalty != 0 || repetition_penalty != 1;
    }

    bool adjusts_logits(int generated) const {
        return has_output_penalties() || (logit_bias && !logit_bias->empty()) ||
            generated < min_new_tokens;
    }

    bool requires_target_only() const { return adjusts_logits(0); }

    bool stops_at(int token) const {
        return !ignore_eos &&
            (stop_token_id == token || additional_stop_token_ids.count(token) != 0);
    }
};

class DecoderSamplingError : public std::runtime_error {
public:
    DecoderSamplingError()
        : std::runtime_error("Sampled logits must contain a finite score and no NaN or +inf") {}
};

class DecoderConstraintError : public std::runtime_error {
public:
    DecoderConstraintError()
        : std::runtime_error("Constrained selection did not produce a valid token") {}
};

class DecoderCapacityError : public std::runtime_error {
public:
    DecoderCapacityError(const char* message, bool retryable)
        : std::runtime_error(message), retryable(retryable) {}

    const bool retryable;
};

enum class DecoderFinishReason {
    length,
    stop,
    cancelled,
    constraint,
    aborted,
};

struct DecoderTokenLogprobs;

struct DecoderCallbacks {
    std::function<bool(int)> on_token;
    std::function<bool()> is_cancelled;
    std::function<bool()> on_admitted = {};
    std::function<bool()> on_stream_poll = {};
    bool buffer_decode_output = false;
    std::function<bool(std::size_t)> on_prompt_admitted = {};
    std::function<bool(int, const std::shared_ptr<const DecoderTokenLogprobs>&)>
        on_scored_token = {};
};

struct DecoderLogit {
    int token_id;
    float logit;
};

struct DecoderLogprob {
    int token_id;
    float value;
};

struct DecoderTokenLogprobs {
    DecoderLogprob selected;
    std::vector<DecoderLogprob> alternatives;
};

using DecoderPromptLogprobs = std::vector<std::shared_ptr<const DecoderTokenLogprobs>>;

struct DecoderResult {
    std::vector<int> output_token_ids;
    DecoderFinishReason finish_reason = DecoderFinishReason::length;
    std::vector<std::vector<DecoderLogit>> step_top_logits;
    std::size_t peak_device_allocation_bytes = 0;
    std::size_t mtp_drafted_tokens = 0;
    std::size_t mtp_accepted_tokens = 0;
    std::size_t mtp_rounds = 0;
    std::size_t cached_prompt_tokens = 0;
    std::optional<std::size_t> matched_text_stop = std::nullopt;
    std::vector<std::shared_ptr<const DecoderTokenLogprobs>> generated_logprobs{};
    std::shared_ptr<const DecoderPromptLogprobs> prompt_logprobs = nullptr;
    std::shared_ptr<const std::vector<double>> label_logprobs = nullptr;
};

class Decoder {
public:
    explicit Decoder(const std::filesystem::path& checkpoint, int cuda_device = 0);
    Decoder(const std::filesystem::path& checkpoint, int cuda_device, bool enable_mtp);
    ~Decoder();

    Decoder(const Decoder&) = delete;
    Decoder& operator=(const Decoder&) = delete;

    DecoderResult generate(const std::vector<int>& prompt_token_ids,
                           const DecoderOptions& options = DecoderOptions{},
                           const DecoderCallbacks& callbacks = DecoderCallbacks{}) const;
    /// Supply feature views in the same order as wrapped media pads in the prompt.
    DecoderResult generate_multimodal(const std::vector<int>& template_token_ids,
                                      const std::vector<DecoderMedia>& media,
                                      const DecoderOptions& options = DecoderOptions{},
                                      const DecoderCallbacks& callbacks = DecoderCallbacks{}) const;
    std::size_t model_device_allocation_bytes() const;
    std::string_view accelerator_name() const;
    const decoder::CudaBackendPolicy& backend_policy() const;
    bool uses_b200_operators() const;
    bool has_int8_draft_head() const;
    bool has_certified_vocab() const;
    int paired_prefill_threshold() const;

private:
    friend class DecoderBatch;
    friend class MediaDecoderCheck;
    friend struct DecoderModelAccess;
    DecoderResult generate_multimodal_impl(const std::vector<int>& template_token_ids,
                                           const std::vector<DecoderMedia>& media,
                                           const DecoderOptions& options,
                                           const DecoderCallbacks& callbacks,
                                           const std::vector<int>* teacher) const;
    struct Model;
    std::unique_ptr<Model> model_;
};

namespace decoder {

inline constexpr int wmma_seed_min_prompt_tokens = 4096;
inline constexpr int wmma_seed_min_rows = 64;
inline constexpr int wmma_seed_max_rows = 128;

}

}
