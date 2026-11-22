#pragma once

#include "sglang/server/batch_service.h"

#include <string_view>

namespace sglang {
class Tokenizer;
}

namespace sglang::server {

class Socket;

namespace score_detail {

inline bool consume(std::size_t& remaining, std::size_t count, std::size_t repetitions = 1) {
    if (repetitions && count > remaining / repetitions) return false;
    remaining -= count * repetitions;
    return true;
}

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
