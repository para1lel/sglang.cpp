#include "sglang/server/tokenization.h"

#include "sglang/json.h"
#include "sglang/server/completions.h"
#include "sglang/server/http.h"
#include "sglang/tokenizer/tokenizer.h"
#include "sglang/utf8.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <iostream>

namespace sglang::server {
namespace {

constexpr std::string_view bad_request = "400 Bad Request";

bool boolean_option(const json::Object& body, std::string_view field) {
    const auto* value = json::get(body, field);
    if (!value) {
        return true;
    }
    if (const auto boolean = value->boolean()) {
        return *boolean;
    }
    if (const auto number = value->number(); number && (*number == 0 || *number == 1)) {
        return *number == 1;
    }
    if (const auto text = value->string(); text && text->size() <= 5) {
        std::string lower(*text);
        for (auto& character : lower) {
            if (character >= 'A' && character <= 'Z') {
                character += 'a' - 'A';
            }
        }
        if (lower == "1" || lower == "true" || lower == "t" || lower == "on" ||
            lower == "yes" || lower == "y") {
            return true;
        }
        if (lower == "0" || lower == "false" || lower == "f" || lower == "off" ||
            lower == "no" || lower == "n") {
            return false;
        }
    }
    throw HttpError(bad_request, (std::string(field) + " must be a boolean").c_str());
}

void append_tokens(std::string& output, const std::vector<TokenId>& tokens) {
    output += '[';
    for (const auto& token : tokens) {
        if (&token != &tokens.front()) {
            output += ',';
        }
        output += std::to_string(token);
    }
    output += ']';
}

bool truthy(const json::Value& value) {
    if (const auto boolean = value.boolean()) {
        return *boolean;
    }
    if (const auto number = value.number()) {
        return *number != 0;
    }
    if (const auto text = value.string()) {
        return !text->empty();
    }
    if (const auto* array = value.array()) {
        return !array->empty();
    }
    if (const auto* object = value.object()) {
        return !object->empty();
    }
    return !value.is_null();
}

void tokenize(const Socket& connection, const CompletionTokenizer& loader,
              const json::Object& body) {
    const auto* prompt = json::get(body, "prompt");
    const auto* messages = json::get(body, "messages");
    if (messages && !messages->is_null()) {
        throw HttpError(bad_request, "Tokenization supports raw prompt text");
    }
    const auto text = prompt ? prompt->string() : std::nullopt;
    const auto* texts = prompt ? prompt->array() : nullptr;
    if (!text && (!texts || !std::all_of(texts->begin(), texts->end(), [](const auto& item) {
        return item.string().has_value();
    }))) {
        throw HttpError(bad_request, "Expected prompt to be a string or an array of strings");
    }
    boolean_option(body, "add_special_tokens");
    if (const auto* stream = json::get(body, "stream"); stream && truthy(*stream)) {
        write_tokenization_error(connection, "501 Not Implemented", "NotImplementedError",
                                 "OpenAIServingTokenize does not support streaming requests");
        return;
    }
    const auto& tokenizer = loader.get();
    std::string output = R"({"tokens":)";
    std::string count;
    if (texts) {
        output += '[';
        count += '[';
        for (const auto& item : *texts) {
            if (&item != &texts->front()) {
                output += ',';
                count += ',';
            }
            const auto tokens = tokenizer.encode(*item.string());
            append_tokens(output, tokens);
            count += std::to_string(tokens.size());
        }
        output += ']';
        count += ']';
    } else {
        const auto tokens = tokenizer.encode(*text);
        append_tokens(output, tokens);
        count = std::to_string(tokens.size());
    }
    output += R"(,"count":)" + count;
    output += R"(,"max_model_len":)" + std::to_string(checkpoint_context) + '}';
    write_json(connection, "200 OK", output);
}

bool integer_space(char32_t scalar) {
    return (scalar >= U'\t' && scalar <= U'\r') || scalar == U' ' || scalar == 0x85 ||
           scalar == 0xa0 || scalar == 0x1680 || (scalar >= 0x2000 && scalar <= 0x200a) ||
           scalar == 0x2028 || scalar == 0x2029 || scalar == 0x202f || scalar == 0x205f ||
           scalar == 0x3000;
}

bool digit(char32_t scalar) {
    return scalar >= U'0' && scalar <= U'9';
}

[[noreturn]] void invalid_ids() {
    throw HttpError(bad_request, "Expected tokens to be integers in a flat or nested array");
}

std::string integer_text(std::string_view text) {
    const auto scalars = read_utf8(text);
    std::u32string_view input(scalars);
    while (!input.empty() && integer_space(input.front())) {
        input.remove_prefix(1);
    }
    while (!input.empty() && integer_space(input.back())) {
        input.remove_suffix(1);
    }
    const bool negative = !input.empty() && input.front() == U'-';
    if (!input.empty() && (negative || input.front() == U'+')) {
        input.remove_prefix(1);
    }
    if (input.empty() || !digit(input.front())) {
        invalid_ids();
    }
    const auto point = input.find(U'.');
    const auto whole = input.substr(0, point);
    if (point != std::u32string_view::npos) {
        const auto fraction = input.substr(point + 1);
        if (fraction.empty() || fraction.find_first_not_of(U'0') != std::u32string_view::npos) {
            invalid_ids();
        }
    }
    std::string digits;
    for (std::size_t index = 0; index < whole.size(); ++index) {
        const auto scalar = whole[index];
        if (digit(scalar)) {
            if (scalar != U'0' || !digits.empty()) {
                digits += static_cast<char>(scalar);
            }
        } else if (scalar != U'_' || !digit(whole[index - 1]) ||
                   index + 1 == whole.size() || !digit(whole[index + 1])) {
            invalid_ids();
        }
    }
    if (digits.size() > 4300) {
        invalid_ids();
    }
    if (digits.empty()) {
        return "0";
    }
    if (negative) {
        digits.insert(digits.begin(), '-');
    }
    return digits;
}

TokenId checked_id(const json::Value& value, std::optional<std::string>& invalid) {
    if (const auto integer = value.unsigned_integer(); integer && *integer < model_vocab_size) {
        return static_cast<TokenId>(*integer);
    }
    if (const auto boolean = value.boolean()) {
        return *boolean;
    }
    std::string digits;
    if (const auto integer = value.signed_integer()) {
        digits = std::to_string(*integer);
    } else if (const auto integer = value.unsigned_integer()) {
        digits = std::to_string(*integer);
    } else if (const auto* wide = value.wide_integer()) {
        digits = wide->digits;
    } else if (const auto text = value.string()) {
        digits = integer_text(*text);
    } else if (const auto number = value.number(); number && std::trunc(*number) == *number &&
               *number > -0x1p63 && *number < 0x1p63) {
        const auto integer = static_cast<std::int64_t>(*number);
        if (integer >= 0 && integer < model_vocab_size) {
            return static_cast<TokenId>(integer);
        }
        digits = std::to_string(integer);
    } else {
        invalid_ids();
    }
    TokenId id = 0;
    const auto* end = digits.data() + digits.size();
    const auto parsed = std::from_chars(digits.data(), end, id);
    if (parsed.ec == std::errc{} && parsed.ptr == end && id < model_vocab_size) {
        return id;
    }
    if (!invalid) {
        invalid = std::move(digits);
    }
    return 0;
}

std::vector<TokenId> read_ids(const json::Array& values, std::optional<std::string>& invalid) {
    std::vector<TokenId> ids;
    ids.reserve(values.size());
    for (const auto& value : values) {
        ids.push_back(checked_id(value, invalid));
    }
    return ids;
}

void append_text(std::string& output, const Tokenizer& tokenizer,
                 const std::vector<TokenId>& ids, bool skip_special) {
    Utf8Stream utf8;
    std::string text;
    for (const auto id : ids) {
        text += utf8.push(tokenizer.piece(id, skip_special));
    }
    text += utf8.finish();
    json::append_quoted(output, text);
}

void detokenize(const Socket& connection, const CompletionTokenizer& loader,
                const json::Object& body) {
    const auto* tokens = json::get(body, "tokens");
    const auto* values = tokens ? tokens->array() : nullptr;
    if (!values) {
        invalid_ids();
    }
    const bool skip_special = boolean_option(body, "skip_special_tokens");
    if (values->empty()) {
        write_json(connection, "200 OK", R"({"text":""})");
        return;
    }
    const bool nested = values->front().array() != nullptr;
    std::optional<std::string> invalid;
    std::vector<std::vector<TokenId>> items;
    if (nested) {
        items.reserve(values->size());
        for (const auto& value : *values) {
            const auto* item = value.array();
            if (!item) {
                invalid_ids();
            }
            items.push_back(read_ids(*item, invalid));
        }
    } else {
        items.push_back(read_ids(*values, invalid));
    }
    if (invalid) {
        write_tokenization_error(connection, bad_request, "BadRequestError",
            "Invalid input: token id " + *invalid + " is out of vocabulary; valid range is [0, " +
            std::to_string(model_vocab_size) + ").");
        return;
    }
    const auto& tokenizer = loader.get();
    std::string output = R"({"text":)";
    if (nested) {
        output += '[';
    }
    for (const auto& ids : items) {
        if (&ids != &items.front()) {
            output += ',';
        }
        append_text(output, tokenizer, ids, skip_special);
    }
    if (nested) {
        output += ']';
    }
    output += '}';
    write_json(connection, "200 OK", output);
}

}

void serve_tokenization(const Socket& connection, TokenizationRoute route,
                        const CompletionTokenizer& tokenizer, std::string_view body) {
    try {
        const auto value = json::parse(body, json::Syntax::completion);
        const auto* object = value.object();
        if (!object) {
            throw HttpError(bad_request, "Expected a JSON request object");
        }
        if (const auto* model = json::get(*object, "model"); model && !model->string()) {
            throw HttpError(bad_request, "model must be a string");
        }
        if (route == TokenizationRoute::tokenize) {
            tokenize(connection, tokenizer, *object);
        } else {
            detokenize(connection, tokenizer, *object);
        }
    } catch (const HttpError& error) {
        write_tokenization_error(connection, error.status, "Bad Request", error.what());
    } catch (const json::ParseError&) {
        write_tokenization_error(connection, bad_request, "Bad Request",
                                 "Invalid JSON request body");
    } catch (const std::exception& error) {
        std::cerr << "Tokenization request failed: " << error.what() << '\n';
        write_tokenization_error(connection, "500 Internal Server Error", "InternalServerError",
                                 "Checkpoint tokenization failed");
    }
}

}
