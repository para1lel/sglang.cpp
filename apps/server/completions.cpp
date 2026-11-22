#include "sglang/server/completions.h"

#include "sglang/json.h"
#include "sglang/server/batch_service.h"
#include "sglang/server/http.h"

#include "sglang/server/text_completion.h"
#include "sglang/server/vision_completion.h"

#include <cstdint>
#include <iostream>
#include <limits>
#include <utility>

namespace sglang::server {
namespace {

struct CompletionRequest {
    std::vector<int> prompt;
    DecoderOptions options;
};

std::optional<std::uint64_t> unsigned_integer(const json::Object& object, std::string_view key) {
    const auto* value = json::get(object, key);
    return value ? value->unsigned_integer() : std::nullopt;
}

CompletionRequest parse_benchmark(const json::Object* object) {
    const auto* stops = object ? json::get(*object, "stop_token_ids") : nullptr;
    const bool has_salt = object && json::get(*object, "cache_salt");
    const auto* temperature = object ? json::get(*object, "temperature") : nullptr;
    if (!object || object->size() != 11u + (stops != nullptr) + has_salt ||
        json::get_string(*object, "model") != "benchmark" ||
        !temperature || temperature->number() != 0.0 || unsigned_integer(*object, "seed") != 0 ||
        json::get_boolean(*object, "ignore_eos") != true ||
        json::get_boolean(*object, "stream") != true ||
        json::get_boolean(*object, "return_token_ids") != true ||
        unsigned_integer(*object, "stream_interval") != 1 ||
        json::get_boolean(*object, "add_special_tokens") != false) {
        throw HttpError("400 Bad Request", "Expected the frozen benchmark streaming options");
    }
    const auto* stream_field = json::get(*object, "stream_options");
    const auto* stream_options = stream_field ? stream_field->object() : nullptr;
    if (!stream_options || stream_options->size() != 1 ||
        json::get_boolean(*stream_options, "include_usage") != true) {
        throw HttpError("400 Bad Request", "Expected stream_options.include_usage=true");
    }
    const auto output_budget = unsigned_integer(*object, "max_tokens");
    if (!output_budget || *output_budget > std::numeric_limits<int>::max()) {
        throw HttpError("400 Bad Request", "Expected a nonnegative integer max_tokens");
    }
    const auto* prompt_field = json::get(*object, "prompt");
    const auto* prompt = prompt_field ? prompt_field->array() : nullptr;
    if (!prompt || prompt->empty()) {
        throw HttpError("400 Bad Request", "Expected a nonempty prompt token ID array");
    }
    CompletionRequest request;
    request.options.max_new_tokens = static_cast<int>(*output_budget);
    set_cache_salt(*object, request.options);
    if (stops) {
        const auto* tokens = stops->array();
        const auto id = tokens && tokens->size() == 1
            ? tokens->front().unsigned_integer() : std::nullopt;
        if (!id || *id > std::numeric_limits<int>::max()) {
            throw HttpError("400 Bad Request", "Expected a single nonnegative integer stop ID");
        }
        request.options.stop_token_id = static_cast<int>(*id);
    }
    request.prompt.reserve(prompt->size());
    for (const auto& value : *prompt) {
        const auto id = value.unsigned_integer();
        if (!id || *id > std::numeric_limits<int>::max()) {
            throw HttpError("400 Bad Request", "Expected nonnegative integer prompt token IDs");
        }
        request.prompt.push_back(static_cast<int>(*id));
    }
    return request;
}

std::string choice_event(std::optional<int> token, const std::vector<int>* prompt,
                         const char* finish_reason) {
    std::string output = R"({"choices":[{"index":0,"token_ids":)";
    append_ids(output, token ? &*token : nullptr, token ? 1 : 0);
    if (prompt) {
        output += R"(,"prompt_token_ids":)";
        append_ids(output, prompt->data(), prompt->size());
    }
    output += R"(,"finish_reason":)";
    if (finish_reason) {
        json::append_quoted(output, finish_reason);
    } else {
        output += "null";
    }
    return output + "}]}";
}

std::string usage_event(std::size_t prompt_tokens, std::size_t completion_tokens,
                        int cached_tokens) {
    std::string output = R"({"choices":[],"usage":{"prompt_tokens":)";
    output += std::to_string(prompt_tokens);
    output += R"(,"completion_tokens":)" + std::to_string(completion_tokens);
    output += R"(,"prompt_tokens_details":{"cached_tokens":)" +
              std::to_string(cached_tokens) + "}}}";
    return output;
}

std::string error_body(const char* message) {
    std::string output = R"({"error":{"message":)";
    json::append_quoted(output, message);
    return output + "}}";
}

void finish_stream(EventStream& stream, const CompletionRequest& request,
                   const DecoderResult& result) {
    const auto output_count = result.output_token_ids.size();
    const auto budget = static_cast<std::size_t>(request.options.max_new_tokens);
    const auto stop_id = request.options.stop_token_id;
    const bool stop_matched = output_count && result.output_token_ids.back() == stop_id;
    const bool stopped = result.finish_reason == DecoderFinishReason::stop;
    if (stopped ? (!stop_matched || output_count > budget)
                : (result.finish_reason != DecoderFinishReason::length ||
                   stop_matched || output_count != budget)) {
        throw std::logic_error("Decoder returned an inconsistent completion");
    }
    const auto* prompt = output_count == 0 ? &request.prompt : nullptr;
    if (stream.start() &&
        stream.write(choice_event(std::nullopt, prompt, stopped ? "stop" : "length")) &&
        stream.write(usage_event(request.prompt.size(), output_count,
                                 result.cached_prompt_tokens)) &&
        stream.write("[DONE]")) {
        stream.end();
    }
}

void stream_completion(const Socket& connection, const CompletionRequest& request,
                       const TextGenerator& generate) {
    EventStream stream(connection);
    bool first_token = true;
    const DecoderCallbacks callbacks{
        [&](int token) {
            const auto* prompt = std::exchange(first_token, false) ? &request.prompt : nullptr;
            return stream.start() && stream.write(choice_event(token, prompt, nullptr));
        },
        [&] { return disconnected(connection); },
        [&] { return stream.start(); },
        [&] { return stream.poll(); },
        true,
    };

    try {
        const auto result = generate(request.prompt, request.options, callbacks);
        if (result.finish_reason != DecoderFinishReason::cancelled) {
            finish_stream(stream, request, result);
        }
    } catch (const std::exception& error) {
        if (!stream.started()) {
            throw;
        }
        std::cerr << "Completion failed: " << error.what() << '\n';
        if (stream.write(error_body("Native decoder failed"))) {
            stream.end();
        }
    }
}

const json::Object* native_text_body(const json::Value& parsed) {
    const auto* object = parsed.object();
    const auto model = object ? json::get_string(*object, "model") : std::nullopt;
    const auto* prompt = object ? json::get(*object, "prompt") : nullptr;
    return object && model != "benchmark" &&
        (model == model_name || (prompt && (prompt->string() || prompt->array()))) ?
        object : nullptr;
}

json::Value parse_completion_body(std::string_view body, bool allow_text) {
    try {
        auto parsed = json::parse(body);
        if (allow_text && parsed.object()) {
            const auto* top_k = json::get(*parsed.object(), "top_k");
            const bool wide_top_k = top_k && top_k->number() && !top_k->unsigned_integer() &&
                !top_k->signed_integer();
            if ((wide_top_k || json::get(*parsed.object(), "response_format") ||
                 json::get(*parsed.object(), "logit_bias")) &&
                native_text_body(parsed)) {
                return json::parse(body, json::Syntax::completion);
            }
        }
        return parsed;
    } catch (const json::NumberRangeError&) {
        try {
            auto parsed = json::parse(body, json::Syntax::benchmark);
            if (parsed.object() && json::get_string(*parsed.object(), "model") == "benchmark") {
                return parsed;
            }
        } catch (const json::NumberRangeError&) {
            if (!allow_text) {
                throw;
            }
        }
        if (allow_text) {
            auto parsed = json::parse(body, json::Syntax::completion);
            if (native_text_body(parsed)) {
                return parsed;
            }
        }
        throw;
    }
}

void dispatch_completion(const Socket& connection, BatchService& service, int slot,
                         const CompletionTokenizer* tokenizer, std::string_view body,
                         bool chat = false, VisionCompletion* vision = nullptr,
                         bool anthropic = false) {
    service.rethrow_failure();
    const TextGenerator generate = [&](const auto& prompt, const auto& options,
                                       const auto& callbacks) {
        return service.generate(slot, prompt, options, callbacks);
    };
    if (chat && tokenizer) {
        MediaGenerator generate_media;
        if (vision) {
            generate_media = [&](const auto& prompt, const auto& options, const auto& callbacks) {
                return vision->generate(service, slot, prompt, options, callbacks);
            };
        }
        complete_chat_request(connection, body, *tokenizer, generate, generate_media,
                              service, slot, anthropic);
        return;
    }
    json::Value parsed;
    try {
        parsed = parse_completion_body(body, tokenizer != nullptr);
    } catch (const json::ParseError&) {
        throw HttpError("400 Bad Request", "Invalid JSON request body");
    }
    if (const auto* text = tokenizer ? native_text_body(parsed) : nullptr) {
        auto completion = parse_text_request(*text, *tokenizer);
        if (!completion.batch.empty()) {
            complete_text_batch(connection, std::move(completion), *tokenizer, service, slot);
        } else if (completion.request.options.generated_top_logprobs >= 0) {
            complete_scored_text(connection, completion.request, *tokenizer, service, slot);
        } else {
            complete_text(connection, completion.request, *tokenizer, generate);
        }
        return;
    }
    stream_completion(connection, parse_benchmark(parsed.object()), generate);
}

}

void write_error(const Socket& connection, std::string_view status, const char* message) {
    write_json(connection, status, error_body(message));
}

void complete(const Socket& connection, BatchService& service, int slot, std::string_view body) {
    dispatch_completion(connection, service, slot, nullptr, body);
}

void complete(const Socket& connection, BatchService& service, int slot,
              const CompletionTokenizer& tokenizer, std::string_view body) {
    dispatch_completion(connection, service, slot, &tokenizer, body);
}

void complete_chat(const Socket& connection, BatchService& service, int slot,
                   const CompletionTokenizer& tokenizer, std::string_view body,
                   VisionCompletion* vision) {
    dispatch_completion(connection, service, slot, &tokenizer, body, true, vision);
}

void complete_messages(const Socket& connection, BatchService& service, int slot,
                       const CompletionTokenizer& tokenizer, std::string_view body,
                       VisionCompletion* vision) {
    constexpr auto bad_request = "400 Bad Request";
    constexpr auto invalid = "invalid_request_error";
    try {
        if (body.empty()) throw std::invalid_argument("Field required");
        dispatch_completion(connection, service, slot, &tokenizer, body, true, vision, true);
    } catch (const HttpError& error) {
        write_anthropic_error(connection, error.status, invalid, error.what());
    } catch (const json::ParseError&) {
        write_anthropic_error(connection, bad_request, invalid, "Invalid JSON request body");
    } catch (const std::invalid_argument& error) {
        write_anthropic_error(connection, bad_request, invalid, error.what());
    } catch (const std::exception& error) {
        std::cerr << "Messages request failed: " << error.what() << '\n';
        write_anthropic_error(connection, "500 Internal Server Error", "api_error",
                              "Internal server error");
    }
}

}
