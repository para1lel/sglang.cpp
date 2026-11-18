#include "text_completion.h"

#include "sglang/json.h"
#include "sglang/server/http.h"

namespace sglang::server {
namespace {

[[noreturn]] void bad_request(const char* message) {
    throw HttpError("400 Bad Request", message);
}

bool read_bool(const json::Object& object, std::string_view key) {
    const auto* value = json::get(object, key);
    if (!value) {
        return false;
    }
    const auto result = value->boolean();
    if (!result) {
        throw std::invalid_argument("Expected boolean " + std::string(key));
    }
    return *result;
}

int text_budget(const json::Object& object) {
    const auto* value = json::get(object, "max_tokens");
    if (!value) {
        return 16;
    }
    const auto budget = value->unsigned_integer();
    if (!budget) {
        bad_request("Expected a nonnegative integer max_tokens");
    }
    if (*budget >= checkpoint_context) {
        bad_request("Prompt and output budget must fit the 262144-token context");
    }
    return static_cast<int>(*budget);
}

void set_stops(const json::Object& object, DecoderOptions& options) {
    if (!read_bool(object, "ignore_eos")) {
        options.additional_stop_token_ids = {end_of_text, end_of_message};
    }
    if (const auto* value = json::get(object, "stop_token_ids")) {
        const auto* stops = value->array();
        if (!stops || stops->size() > model_vocab_size) {
            bad_request("Expected an array of stop token IDs");
        }
        for (const auto& value : *stops) {
            const auto id = value.unsigned_integer();
            if (!id || *id >= model_vocab_size) {
                bad_request("Stop token ID falls outside the vocabulary");
            }
            options.additional_stop_token_ids.insert(static_cast<int>(*id));
        }
    }
}

TextRequest parse_text(const json::Object& object, const CompletionTokenizer& tokenizer) {
    for (const auto& entry : object) {
        const auto& key = entry.first;
        if (key != "model" && key != "prompt" && key != "max_tokens" && key != "stream" &&
            key != "stream_options" && key != "ignore_eos" && key != "stop_token_ids" &&
            key != "return_token_ids") {
            throw std::invalid_argument("Unsupported text completion option: " + key);
        }
    }
    if (json::get_string(object, "model") != model_name) {
        bad_request("Expected model Qwen3.8-27B-FP8 for a string prompt");
    }
    const auto prompt = json::get_string(object, "prompt");
    if (!prompt || prompt->empty() || prompt->size() > 1024 * 1024) {
        bad_request("Expected a nonempty prompt string of up to 1048576 bytes");
    }
    TextRequest request;
    request.stream = read_bool(object, "stream");
    request.return_token_ids = read_bool(object, "return_token_ids");
    request.options.max_new_tokens = text_budget(object);
    if (const auto* value = json::get(object, "stream_options")) {
        const auto* options = value->object();
        if (!request.stream || !options || options->size() != 1 ||
            !json::get_boolean(*options, "include_usage").has_value()) {
            bad_request("Expected stream_options.include_usage for a stream");
        }
        request.include_usage = read_bool(*options, "include_usage");
    }
    set_stops(object, request.options);
    try {
        const auto ids = tokenizer.get().encode(*prompt,
            checkpoint_context - request.options.max_new_tokens);
        request.prompt.assign(ids.begin(), ids.end());
    } catch (const std::length_error&) {
        bad_request("Prompt and output budget must fit the 262144-token context");
    }
    return request;
}

}

CompletionTokenizer::CompletionTokenizer(const std::filesystem::path& checkpoint)
    : path_(checkpoint / "tokenizer.json") {}

const Tokenizer& CompletionTokenizer::get() const {
    std::call_once(loaded_, [this] { tokenizer_ = std::make_unique<const Tokenizer>(path_); });
    return *tokenizer_;
}

TextRequest parse_text_request(std::string_view body, const CompletionTokenizer& tokenizer) {
    json::Value parsed;
    try {
        parsed = json::parse(body);
    } catch (const json::ParseError&) {
        bad_request("Invalid JSON request body");
    }
    const auto* object = parsed.object();
    if (!object) {
        bad_request("Expected a JSON object");
    }
    return parse_text(*object, tokenizer);
}

}
