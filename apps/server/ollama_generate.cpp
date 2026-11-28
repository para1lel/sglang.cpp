#include "sglang/server/ollama_generate.h"

#include "sglang/server/tokenization_validation.h"
#include "sglang/tokenizer/tokenizer.h"
#include "sglang/utf8.h"

#include <algorithm>
#include <cmath>
#include <ctime>
#include <iomanip>
#include <sstream>

namespace sglang::server {
namespace {

using request_validation::invalid;
using request_validation::optional_bool;

std::string_view optional_text(const json::OrderedObject& object, std::string_view key) {
    if (const auto* value = supplied(object, key)) {
        if (const auto text = value->string()) return *text;
        invalid(key, "Input should be a valid string");
    }
    return {};
}

bool context_integer(const json::Value& value) {
    if (value.boolean() || value.signed_integer() || value.unsigned_integer() ||
        value.wide_integer()) return true;
    if (const auto text = value.string())
        return detail::scan_integer_text(*text).has_value();
    const auto number = value.number();
    return number && std::isfinite(*number) && std::trunc(*number) == *number &&
        std::abs(*number) < 9223372036854775808.0;
}

void validate_images(const json::OrderedObject& object) {
    if (const auto* value = supplied(object, "images")) {
        const auto* list = value->array();
        if (!list || !std::all_of(list->begin(), list->end(), [](const auto& item) {
                return item.string().has_value();
            })) {
            invalid("images", "Input should be a list of strings");
        }
    }
}

void validate_common(const json::OrderedObject& object) {
    if (const auto* value = supplied(object, "format");
        value && !value->ordered_object() && value->string() != "json") {
        invalid("format", "Expected 'json' or a dictionary");
    }
    if (const auto* value = supplied(object, "keep_alive");
        value && !value->string() && !value->number() && !value->boolean() &&
        !value->wide_integer()) {
        invalid("keep_alive", "Expected a number or a string");
    }
}

void validate_generate(const json::OrderedObject& object) {
    optional_text(object, "suffix");
    optional_text(object, "template");
    optional_bool(json::get(object, "raw"), "raw");
    optional_bool(supplied(object, "think"), "think");
    if (const auto* value = supplied(object, "context")) {
        const auto* list = value->array();
        if (!list || !std::all_of(list->begin(), list->end(), context_integer)) {
            invalid("context", "Input should be a list of integers");
        }
    }
    validate_images(object);
}

void require_text(const json::OrderedObject& object, std::string_view key) {
    const auto* value = json::get(object, key);
    if (!value) invalid(key, "Field required");
    if (!value->string()) invalid(key, "Input should be a valid string");
}

json::Array chat_messages(const json::OrderedObject& object) {
    const auto* value = json::get(object, "messages");
    if (!value) invalid("messages", "Field required");
    const auto* messages = value->array();
    if (!messages) invalid("messages", "Input should be a valid list");
    json::Array output;
    output.reserve(messages->size());
    for (const auto& item : *messages) {
        const auto* message = item.ordered_object();
        if (!message) invalid("messages", "Input should be a valid dictionary");
        require_text(*message, "role");
        require_text(*message, "content");
        validate_images(*message);
        output.emplace_back(json::OrderedObject{
            {"role", *json::get(*message, "role")}, {"content", *json::get(*message, "content")},
        });
    }
    return output;
}

void validate_chat(const json::OrderedObject& object) {
    const auto* think = supplied(object, "think");
    if (think && think->string() != "low" && think->string() != "medium" &&
        think->string() != "high") {
        optional_bool(think, "think");
    }
}

std::string plain_prompt(const json::OrderedObject& object) {
    require_text(object, "prompt");
    validate_generate(object);
    std::string prompt;
    if (const auto system = optional_text(object, "system"); !system.empty()) {
        prompt = system;
        prompt += "\n\n";
    }
    prompt += *json::get(object, "prompt")->string();
    const auto characters = read_utf8(prompt);
    if (std::all_of(characters.begin(), characters.end(), python_whitespace)) {
        prompt.clear();
    }
    return prompt;
}

std::string timestamp() {
    const auto now = std::chrono::system_clock::now();
    const auto seconds = std::chrono::floor<std::chrono::seconds>(now);
    const auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(now - seconds);
    const auto unix_time = std::chrono::system_clock::to_time_t(seconds);
    std::tm utc{};
    gmtime_r(&unix_time, &utc);
    std::ostringstream formatted;
    formatted << std::put_time(&utc, "%Y-%m-%dT%H:%M:%S") << '.'
              << std::setfill('0') << std::setw(3) << millis.count() << 'Z';
    return formatted.str();
}

}

OllamaRequest parse_ollama_request(std::string_view body, OllamaCompletion completion) {
    auto parsed = json::parse(body, json::Syntax::python_request);
    auto* object = parsed.ordered_object();
    if (!object) invalid("body", "Input should be a valid dictionary");
    require_text(*object, "model");
    OllamaRequest request;
    request.completion = completion;
    if (completion == OllamaCompletion::chat) {
        request.messages = chat_messages(*object);
        validate_chat(*object);
    } else {
        request.prompt = plain_prompt(*object);
    }
    validate_common(*object);
    request.stream = optional_bool(json::get(*object, "stream"), "stream", true);
    for (auto& [key, value] : *object) {
        if (key == "options" && !value.is_null()) {
            auto* options = value.ordered_object();
            if (!options) invalid(key, "Input should be a valid dictionary");
            request.options = std::move(*options);
            break;
        }
    }
    return request;
}

json::Object ollama_sampling(const json::OrderedObject& options) {
    if (json::get(options, "seed")) {
        invalid("seed", "Pinned sampling parameters reject the seed keyword");
    }
    json::Object sampling{{"max_new_tokens", 2048}};
    for (const auto* key : {"temperature", "top_p", "top_k", "num_predict", "stop",
                            "presence_penalty", "frequency_penalty"}) {
        if (const auto* value = json::get(options, key)) {
            const std::string_view name = key;
            const auto mapped = name == "num_predict" ? "max_new_tokens" : key;
            const auto boolean = name == "stop" ? std::nullopt : value->boolean();
            sampling.insert_or_assign(mapped, boolean ? json::Value(int(*boolean)) : *value);
        }
    }
    if (const auto* temperature = json::get(sampling, "temperature")) {
        if (const auto number = temperature->number(); number && 0 <= *number && *number < 1e-6) {
            sampling.insert_or_assign("temperature", 1);
            sampling.insert_or_assign("top_k", 1);
        }
    }
    return sampling;
}

std::string ollama_frame(OllamaCompletion completion, std::string_view text, bool done) {
    std::string output = R"({"model":)";
    json::append_quoted(output, model_name);
    output += R"(,"created_at":)";
    json::append_quoted(output, timestamp());
    const bool chat = completion == OllamaCompletion::chat;
    output += chat ? R"(,"message":{"role":"assistant","content":)" : R"(,"response":)";
    json::append_quoted(output, text);
    if (chat) output += R"(,"images":null})";
    output += done ? R"(,"done":true,"done_reason":"stop"})"
                   : R"(,"done":false,"done_reason":null})";
    return output;
}

std::string ollama_result(OllamaCompletion completion, std::string_view text,
                          std::optional<OllamaGenerateUsage> usage) {
    auto output = ollama_frame(completion, text, true);
    output.pop_back();
    if (completion == OllamaCompletion::generate) output += R"(,"context":null)";
    output += R"(,"total_duration":)";
    output += usage ? std::to_string(usage->duration.count()) : "null";
    output += R"(,"load_duration":null,"prompt_eval_count":)";
    output += usage ? std::to_string(usage->prompt_tokens) : "null";
    output += R"(,"prompt_eval_duration":null,"eval_count":)";
    output += usage ? std::to_string(usage->output_tokens) : "null";
    output += R"(,"eval_duration":null})";
    return output;
}

}
