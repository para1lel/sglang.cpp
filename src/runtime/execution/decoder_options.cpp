#include "sglang/decoder.h"

namespace sglang {

bool DecoderOptions::samples() const {
    return temperature >= 1e-6 && top_k != 1;
}

bool DecoderOptions::has_output_penalties() const {
    return frequency_penalty != 0 || presence_penalty != 0 || repetition_penalty != 1;
}

bool DecoderOptions::adjusts_logits(int generated) const {
    return has_output_penalties() || (logit_bias && !logit_bias->empty()) ||
        generated < min_new_tokens;
}

bool DecoderOptions::requires_target_only() const {
    return adjusts_logits(0);
}

bool DecoderOptions::stops_at(int token) const {
    return !ignore_eos &&
        (stop_token_id == token || additional_stop_token_ids.count(token) != 0);
}

DecoderSamplingError::DecoderSamplingError()
    : std::runtime_error("Sampled logits must contain a finite score and no NaN or +inf") {}

DecoderConstraintError::DecoderConstraintError()
    : std::runtime_error("Constrained selection did not produce a valid token") {}

DecoderCapacityError::DecoderCapacityError(const char* message, bool retryable)
    : std::runtime_error(message), retryable(retryable) {}

}
