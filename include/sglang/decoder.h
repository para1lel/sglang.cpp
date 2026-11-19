#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace sglang {

inline constexpr int decoder_max_mtp_steps = 4;

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

    bool samples() const { return temperature >= 1e-6 && top_k != 1; }

    bool stops_at(int token) const {
        return stop_token_id == token || additional_stop_token_ids.count(token) != 0;
    }
};

class DecoderSamplingError : public std::runtime_error {
public:
    DecoderSamplingError()
        : std::runtime_error("Sampled logits must contain a finite score and no NaN or +inf") {}
};

enum class DecoderFinishReason {
    length,
    stop,
    cancelled,
};

struct DecoderCallbacks {
    std::function<bool(int)> on_token;
    std::function<bool()> is_cancelled;
};

struct DecoderLogit {
    int token_id;
    float logit;
};

struct DecoderResult {
    std::vector<int> output_token_ids;
    DecoderFinishReason finish_reason = DecoderFinishReason::length;
    std::vector<std::vector<DecoderLogit>> step_top_logits;
    std::size_t peak_device_allocation_bytes = 0;
    std::size_t mtp_drafted_tokens = 0;
    std::size_t mtp_accepted_tokens = 0;
    std::size_t mtp_rounds = 0;
};

enum class Fp8PrefillPolicy {
    exact,
    tensor_core,
};

class Decoder {
public:
    explicit Decoder(const std::filesystem::path& checkpoint, int cuda_device = 0);
    Decoder(const std::filesystem::path& checkpoint, int cuda_device, bool enable_mtp);
    Decoder(const std::filesystem::path& checkpoint, int cuda_device, bool enable_mtp,
            Fp8PrefillPolicy fp8_prefill);
    ~Decoder();

    Decoder(const Decoder&) = delete;
    Decoder& operator=(const Decoder&) = delete;

    DecoderResult generate(const std::vector<int>& prompt_token_ids,
                           const DecoderOptions& options = DecoderOptions{},
                           const DecoderCallbacks& callbacks = DecoderCallbacks{}) const;
    std::size_t model_device_allocation_bytes() const;

private:
    friend class DecoderBatch;
    struct Model;
    std::unique_ptr<Model> model_;
};

}
