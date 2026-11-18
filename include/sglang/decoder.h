#pragma once

#include <cstddef>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <vector>

namespace sglang {

inline constexpr int decoder_max_mtp_steps = 4;

struct DecoderOptions {
    int max_new_tokens = 1;
    std::optional<int> stop_token_id;
    int top_logits = 0;
    int mtp_steps = 0;
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
