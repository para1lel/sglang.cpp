#include "text_completion.h"

#include "chat_template.h"
#include "sglang/decoder_constraint.h"
#include "sglang/json.h"
#include "sglang/server/http.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <limits>

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

void set_sampling(const json::Object& object, DecoderOptions& options) {
    if (const auto* value = json::get(object, "temperature")) {
        auto temperature = value->number();
        if (const auto* wide = value->wide_integer()) {
            double parsed = 0;
            const auto* end = wide->digits.data() + wide->digits.size();
            const auto conversion = std::from_chars(wide->digits.data(), end, parsed);
            if (conversion.ec == std::errc{} && conversion.ptr == end) {
                temperature = parsed;
            }
        }
        if (!temperature || !std::isfinite(*temperature) || *temperature < 0) {
            bad_request("temperature must be finite and nonnegative");
        }
        options.temperature = *temperature;
    }
    if (const auto* value = json::get(object, "top_p")) {
        const auto top_p = value->number();
        if (!top_p || !(*top_p > 0 && *top_p <= 1)) {
            bad_request("top_p must be in (0, 1]");
        }
        options.top_p = *top_p;
    }
    if (const auto* value = json::get(object, "top_k")) {
        const auto positive = value->unsigned_integer();
        const auto* wide = value->wide_integer();
        if (value->signed_integer() == -1) {
            options.top_k = -1;
        } else if (positive && *positive > 0) {
            options.top_k = std::min<std::uint64_t>(*positive, model_vocab_size);
        } else if (wide && wide->digits.front() != '-') {
            options.top_k = model_vocab_size;
        } else {
            bad_request("top_k must be -1 or a positive integer");
        }
    }
    if (const auto* value = json::get(object, "seed")) {
        const auto positive = value->unsigned_integer();
        options.seed = value->signed_integer();
        if (positive && *positive <= std::numeric_limits<std::int64_t>::max()) {
            options.seed = static_cast<std::int64_t>(*positive);
        }
        if (!options.seed) {
            bad_request("seed must be a signed 64-bit integer");
        }
    }
}

void set_constraint(const json::Object& object, const CompletionTokenizer& tokenizer,
                    DecoderOptions& options) {
    const auto* value = json::get(object, "ebnf");
    if (!value) {
        return;
    }
    if (!value->string()) {
        bad_request("ebnf must be a string");
    }
    if (read_bool(object, "ignore_eos") || json::get(object, "stop_token_ids")) {
        bad_request("ebnf requires model EOS stops without stop_token_ids overrides");
    }
    options.constraint = TokenConstraint::compile(tokenizer.get(), *value->string());
    if (options.max_new_tokens < options.constraint->minimum_tokens()) {
        bad_request("max_tokens is too small to complete an ebnf choice");
    }
}

TextRequest text_options(const json::Object& object, const CompletionTokenizer& tokenizer) {
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
    set_sampling(object, request.options);
    set_constraint(object, tokenizer, request.options);
    set_cache_salt(object, request.options);
    return request;
}

void validate_prompt(std::string_view prompt) {
    if (prompt.empty() || prompt.size() > 1024 * 1024) {
        bad_request("Expected a nonempty prompt string of up to 1048576 bytes");
    }
}

void encode_prompt(TextRequest& request, std::string_view prompt,
                   const CompletionTokenizer& tokenizer) {
    try {
        const auto ids = tokenizer.get().encode(prompt,
            checkpoint_context - request.options.max_new_tokens);
        request.prompt.assign(ids.begin(), ids.end());
    } catch (const std::length_error&) {
        bad_request("Prompt and output budget must fit the 262144-token context");
    }
}

bool is_text_option(std::string_view key) {
    return key == "model" || key == "max_tokens" || key == "stream" || key == "stream_options" ||
        key == "ignore_eos" || key == "stop_token_ids" || key == "return_token_ids" ||
        key == "temperature" || key == "top_p" || key == "top_k" || key == "seed" ||
        key == "ebnf" || key == "cache_salt";
}

TextRequest parse_text(const json::Object& object, const CompletionTokenizer& tokenizer) {
    for (const auto& [key, value] : object) {
        if (key != "prompt" && !is_text_option(key)) {
            throw std::invalid_argument("Unsupported text completion option: " + key);
        }
    }
    if (json::get_string(object, "model") != model_name) {
        bad_request("Expected model Qwen3.8-27B-FP8 for a string prompt");
    }
    const auto prompt = json::get_string(object, "prompt").value_or("");
    validate_prompt(prompt);
    auto request = text_options(object, tokenizer);
    encode_prompt(request, prompt, tokenizer);
    return request;
}

json::Object chat_text_options(const json::OrderedObject& chat) {
    json::Object options;
    for (const auto& [key, value] : chat) {
        if (key == "stream_options" && value.ordered_object()) {
            const auto& fields = *value.ordered_object();
            options.emplace(key, json::Object(fields.begin(), fields.end()));
        } else if (is_text_option(key)) {
            options.emplace(key, value);
        } else if (key == "max_completion_tokens") {
            if (json::get(chat, "max_tokens")) {
                bad_request("Use one of max_tokens and max_completion_tokens");
            }
            options.emplace("max_tokens", value);
        } else if (key == "n") {
            if (value.unsigned_integer() != 1) {
                bad_request("Only n=1 is supported");
            }
        } else if (key != "messages" && key != "tools" && key != "tool_choice" &&
                   key != "add_generation_prompt" && key != "add_vision_id" &&
                   key != "enable_thinking" && key != "preserve_thinking" &&
                   key != "reasoning_effort") {
            throw std::invalid_argument("Unsupported chat completion option: " + key);
        }
    }
    if (json::get_string(options, "model") != model_name) {
        bad_request("Expected model Qwen3.8-27B-FP8 for a chat request");
    }
    return options;
}

json::Value parse_body(std::string_view body, json::Syntax syntax) {
    try {
        return json::parse(body, syntax);
    } catch (const json::ParseError&) {
        bad_request("Invalid JSON request body");
    }
}

}

CompletionTokenizer::CompletionTokenizer(const std::filesystem::path& checkpoint)
    : checkpoint_(checkpoint) {}

CompletionTokenizer::~CompletionTokenizer() = default;

const Tokenizer& CompletionTokenizer::get() const {
    std::call_once(loaded_, [this] {
        tokenizer_ = std::make_unique<const Tokenizer>(checkpoint_ / "tokenizer.json");
    });
    return *tokenizer_;
}

const ChatTemplate& CompletionTokenizer::chat_template() const {
    std::call_once(chat_loaded_, [this] {
        chat_template_ = std::make_unique<const ChatTemplate>(checkpoint_);
    });
    return *chat_template_;
}

TextRequest parse_text_request(const json::Object& body, const CompletionTokenizer& tokenizer) {
    return parse_text(body, tokenizer);
}

void set_cache_salt(const json::Object& object, DecoderOptions& options) {
    if (json::get(object, "cache_salt")) {
        const auto salt = json::get_string(object, "cache_salt");
        if (!salt || salt->size() > decoder_prefix_max_salt_bytes) {
            bad_request("Expected a cache_salt string of up to 256 bytes");
        }
        options.prefix_cache_salt = *salt;
    }
}

TextRequest parse_chat_request(std::string_view body, const CompletionTokenizer& tokenizer) {
    const auto parsed = parse_body(body, json::Syntax::ordered);
    const auto* chat = parsed.ordered_object();
    if (!chat) {
        bad_request("Expected a JSON object for a chat request");
    }
    auto request = text_options(chat_text_options(*chat), tokenizer);
    request.chat = true;
    const auto prompt = tokenizer.chat_template().render(*chat, ChatInput::http);
    validate_prompt(prompt);
    encode_prompt(request, prompt, tokenizer);
    return request;
}

}
