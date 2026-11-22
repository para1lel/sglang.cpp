#pragma once

#include "sglang/json.h"

#include <chrono>

namespace sglang::server {

enum class OllamaCompletion { generate, chat };

struct OllamaRequest {
    OllamaCompletion completion = OllamaCompletion::generate;
    std::string prompt;
    json::Array messages;
    json::OrderedObject options;
    bool stream = true;
};

struct OllamaGenerateUsage {
    std::chrono::nanoseconds duration;
    std::size_t prompt_tokens;
    std::size_t output_tokens;
};

OllamaRequest parse_ollama_request(std::string_view body, OllamaCompletion completion);
json::Object ollama_sampling(const json::OrderedObject& options);
std::string ollama_frame(OllamaCompletion completion, std::string_view text, bool done);
std::string ollama_result(OllamaCompletion completion, std::string_view text,
                          std::optional<OllamaGenerateUsage> usage = std::nullopt);

}
