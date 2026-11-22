#pragma once

#include "sglang/server/batch_service.h"
#include "sglang/tokenizer/tokenizer.h"

#include <string_view>

namespace sglang::server {

class Socket;

namespace score_detail {

inline constexpr std::size_t max_score_items = 1024;
inline constexpr std::size_t max_score_prompt_tokens = 4 * checkpoint_context;
inline constexpr std::size_t max_score_text_bytes = 8 * 1024 * 1024;
inline constexpr std::size_t max_score_response_values = checkpoint_context;

inline bool consume(std::size_t& remaining, std::size_t count, std::size_t repetitions = 1) {
    if (repetitions && count > remaining / repetitions) return false;
    remaining -= count * repetitions;
    return true;
}

std::vector<double> probabilities(const std::vector<double>& logprobs, double temperature,
                                  bool normalize);

}

struct ScoreRequest {
    std::vector<BatchPrompt> prompts;
    std::string model;
    double temperature;
    bool apply_softmax;
    bool return_token_logprobs;
};

class ScoreValidationError : public std::invalid_argument {
public:
    using std::invalid_argument::invalid_argument;
};

ScoreRequest parse_score_request(std::string_view body, const Tokenizer& tokenizer);
std::string score_response(const ScoreRequest& request, const std::vector<DecoderResult>& results);
void serve_score(const Socket& connection, BatchService& service, int slot,
                 const Tokenizer& tokenizer, std::string_view body);

}
