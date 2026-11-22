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

[[noreturn]] void invalid(std::string_view field, std::string_view description) {
    throw std::invalid_argument(std::string(field) + ": " + std::string(description));
}

const json::Value* supplied(const json::OrderedObject& object, std::string_view key) {
    const auto* value = json::get(object, key);
    return value && !value->is_null() ? value : nullptr;
}

std::string_view optional_text(const json::OrderedObject& object, std::string_view key) {
    if (const auto* value = supplied(object, key)) {
        if (const auto text = value->string()) return *text;
        invalid(key, "Input should be a valid string");
    }
    return {};
}

bool read_flag(const json::Value* value, std::string_view key, bool fallback) {
    if (!value) return fallback;
    if (const auto flag = chat_boolean(*value)) return *flag;
    invalid(key, "Input should be a valid boolean");
}

bool unicode_whitespace(char32_t character) {
    return (character >= 9 && character <= 13) || character == 0x20 ||
        character == 0x85 || character == 0xa0 || character == 0x1680 ||
        (character >= 0x2000 && character <= 0x200a) || character == 0x2028 ||
        character == 0x2029 || character == 0x202f || character == 0x205f ||
        character == 0x3000;
}

bool python_whitespace(char32_t character) {
    return (character >= 0x1c && character <= 0x1f) || unicode_whitespace(character);
}

bool decimal_digit(char32_t character) {
    return '0' <= character && character <= '9';
}

bool context_integer_text(std::string_view text) {
    const auto characters = read_utf8(text);
    std::u32string_view number = characters;
    while (!number.empty() && unicode_whitespace(number.front())) number.remove_prefix(1);
    while (!number.empty() && unicode_whitespace(number.back())) number.remove_suffix(1);
    if (!number.empty() && (number.front() == '+' || number.front() == '-')) {
        number.remove_prefix(1);
    }
    const auto point = number.find('.');
    const auto integer = number.substr(0, point);
    if (integer.empty() || !decimal_digit(integer.front()) || !decimal_digit(integer.back())) {
        return false;
    }
    bool separator = false;
    for (const auto character : integer) {
        if (!decimal_digit(character) && (character != '_' || separator)) return false;
        separator = character == '_';
    }
    if (point == number.npos) return true;
    const auto fraction = number.substr(point + 1);
    return !fraction.empty() && fraction.find_first_not_of('0') == fraction.npos;
}

bool context_integer(const json::Value& value) {
    if (value.boolean() || value.signed_integer() || value.unsigned_integer() ||
        value.wide_integer()) return true;
    if (const auto text = value.string()) return context_integer_text(*text);
    const auto number = value.number();
    return number && std::isfinite(*number) && std::trunc(*number) == *number &&
        std::abs(*number) < 9223372036854775808.0;
}

void validate_unused(const json::OrderedObject& object) {
    optional_text(object, "suffix");
    optional_text(object, "template");
    read_flag(json::get(object, "raw"), "raw", false);
    read_flag(supplied(object, "think"), "think", false);
    if (const auto* value = supplied(object, "context")) {
        const auto* list = value->array();
        if (!list || !std::all_of(list->begin(), list->end(), context_integer)) {
            invalid("context", "Input should be a list of integers");
        }
    }
    if (const auto* value = supplied(object, "images")) {
        const auto* list = value->array();
        if (!list || !std::all_of(list->begin(), list->end(), [](const auto& item) {
                return item.string().has_value();
            })) {
            invalid("images", "Input should be a list of strings");
        }
    }
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

OllamaGenerateRequest parse_ollama_generate(std::string_view body) {
    auto parsed = json::parse(body, json::Syntax::python_request);
    auto* object = parsed.ordered_object();
    if (!object) invalid("body", "Input should be a valid dictionary");
    for (const auto* field : {"model", "prompt"}) {
        const auto* value = json::get(*object, field);
        if (!value) invalid(field, "Field required");
        if (!value->string()) invalid(field, "Input should be a valid string");
    }
    validate_unused(*object);
    OllamaGenerateRequest request;
    request.stream = read_flag(json::get(*object, "stream"), "stream", true);
    if (const auto system = optional_text(*object, "system"); !system.empty()) {
        request.prompt = system;
        request.prompt += "\n\n";
    }
    request.prompt += *json::get(*object, "prompt")->string();
    const auto characters = read_utf8(request.prompt);
    if (std::all_of(characters.begin(), characters.end(), python_whitespace)) {
        request.prompt.clear();
    }
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

std::string ollama_generate_frame(std::string_view text, bool done) {
    std::string output = R"({"model":)";
    json::append_quoted(output, model_name);
    output += R"(,"created_at":)";
    json::append_quoted(output, timestamp());
    output += R"(,"response":)";
    json::append_quoted(output, text);
    output += done ? R"(,"done":true,"done_reason":"stop"})"
                   : R"(,"done":false,"done_reason":null})";
    return output;
}

std::string ollama_generate_result(std::string_view text,
                                  std::optional<OllamaGenerateUsage> usage) {
    auto output = ollama_generate_frame(text, true);
    output.pop_back();
    output += R"(,"context":null,"total_duration":)";
    output += usage ? std::to_string(usage->duration.count()) : "null";
    output += R"(,"load_duration":null,"prompt_eval_count":)";
    output += usage ? std::to_string(usage->prompt_tokens) : "null";
    output += R"(,"prompt_eval_duration":null,"eval_count":)";
    output += usage ? std::to_string(usage->output_tokens) : "null";
    output += R"(,"eval_duration":null})";
    return output;
}

}
