#pragma once

#include "sglang/json.h"

#include <chrono>

namespace sglang::server {

struct OllamaGenerateRequest {
    std::string prompt;
    json::OrderedObject options;
    bool stream = true;
};

struct OllamaGenerateUsage {
    std::chrono::nanoseconds duration;
    std::size_t prompt_tokens;
    std::size_t output_tokens;
};

OllamaGenerateRequest parse_ollama_generate(std::string_view body);
json::Object ollama_sampling(const json::OrderedObject& options);
std::string ollama_generate_frame(std::string_view text, bool done);
std::string ollama_generate_result(std::string_view text,
                                  std::optional<OllamaGenerateUsage> usage = std::nullopt);

}
