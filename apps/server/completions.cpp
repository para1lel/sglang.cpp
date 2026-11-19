#include "sglang/server/completions.h"

#include "sglang/json.h"
#include "sglang/server/batch_service.h"
#include "sglang/server/http.h"

#include "sglang/server/text_completion.h"
#include "sglang/server/vision_completion.h"

#include <cstdint>
#include <iostream>
#include <limits>

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

bool begin_stream(const Socket& connection, bool& streaming) {
    if (!streaming) {
        streaming = start_events(connection);
    }
    return streaming;
}

void finish_stream(const Socket& connection, const CompletionRequest& request,
                   const DecoderResult& result, bool& streaming) {
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
    const auto* prompt = streaming ? nullptr : &request.prompt;
    if (begin_stream(connection, streaming) &&
        write_event(connection, choice_event(std::nullopt, prompt, stopped ? "stop" : "length")) &&
        write_event(connection, usage_event(request.prompt.size(), output_count,
                                            result.cached_prompt_tokens)) &&
        write_event(connection, "[DONE]")) {
        end_events(connection);
    }
}

void stream_completion(const Socket& connection, const CompletionRequest& request,
                       const TextGenerator& generate,
                       bool trace_serial = false) {
    bool streaming = false;
    bool first_token = true;
    const DecoderCallbacks callbacks{
        [&](int token) {
            if (trace_serial) {
                trace_batch_step(1, first_token ? 0 : 1, 0);
                first_token = false;
            }
            const auto* prompt = streaming ? nullptr : &request.prompt;
            return begin_stream(connection, streaming) &&
                   write_event(connection, choice_event(token, prompt, nullptr));
        },
        [&] { return disconnected(connection); },
    };

    try {
        const auto result = generate(request.prompt, request.options, callbacks);
        if (result.finish_reason != DecoderFinishReason::cancelled) {
            finish_stream(connection, request, result, streaming);
        } else if (trace_serial) {
            trace_batch_step(0, 0, 0);
        }
    } catch (const std::exception& error) {
        if (!streaming) {
            throw;
        }
        std::cerr << "Completion failed: " << error.what() << '\n';
        if (write_event(connection, error_body("Native decoder failed"))) {
            end_events(connection);
        }
    }
}

const json::Object* native_text_body(const json::Value& parsed) {
    const auto* object = parsed.object();
    const auto model = object ? json::get_string(*object, "model") : std::nullopt;
    return object && model != "benchmark" &&
        (json::get_string(*object, "prompt") || model == model_name) ? object : nullptr;
}

json::Value parse_completion_body(std::string_view body, bool allow_text) {
    try {
        auto parsed = json::parse(body);
        if (allow_text && parsed.object()) {
            const auto* top_k = json::get(*parsed.object(), "top_k");
            const bool wide_top_k = top_k && top_k->number() && !top_k->unsigned_integer() &&
                !top_k->signed_integer();
            if ((wide_top_k || json::get(*parsed.object(), "response_format")) &&
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

void dispatch_completion(const Socket& connection, std::string_view body,
                         const CompletionTokenizer* tokenizer, const TextGenerator& generate,
                         bool trace, bool chat,
                         const MediaGenerator& generate_media = MediaGenerator{}) {
    if (chat && tokenizer) {
        complete_chat_request(connection, body, *tokenizer, generate, generate_media, trace);
        return;
    }
    json::Value parsed;
    try {
        parsed = parse_completion_body(body, tokenizer != nullptr);
    } catch (const json::ParseError&) {
        throw HttpError("400 Bad Request", "Invalid JSON request body");
    }
    if (const auto* text = tokenizer ? native_text_body(parsed) : nullptr) {
        complete_text(connection, parse_text_request(*text, *tokenizer), *tokenizer,
                      generate, trace);
        return;
    }
    stream_completion(connection, parse_benchmark(parsed.object()), generate, trace);
}

void complete_serial(const Socket& connection, const Decoder& decoder,
                     const CompletionTokenizer* tokenizer, std::string_view body,
                     bool trace, int mtp_steps, bool eager, bool chat = false,
                     VisionCompletion* vision = nullptr) {
    MediaGenerator generate_media;
    if (vision) {
        generate_media = [&](const auto& prompt, auto options, const auto& callbacks) {
            options.eager = eager;
            return vision->generate(decoder, prompt, options, callbacks);
        };
    }
    dispatch_completion(connection, body, tokenizer,
                        [&](const auto& prompt, auto options, const auto& callbacks) {
        if (!options.requires_target_only()) {
            options.mtp_steps = mtp_steps;
        }
        options.eager = eager;
        return decoder.generate(prompt, options, callbacks);
    }, trace, chat, generate_media);
}

void complete_batch(const Socket& connection, BatchService& service, int slot,
                    const CompletionTokenizer* tokenizer, std::string_view body,
                    bool chat = false) {
    dispatch_completion(connection, body, tokenizer,
                        [&](const auto& prompt, const auto& options, const auto& callbacks) {
        return service.generate(slot, prompt, options, callbacks);
    }, false, chat);
}

}

void write_error(const Socket& connection, std::string_view status, const char* message) {
    write_json(connection, status, error_body(message));
}

void complete(const Socket& connection, const Decoder& decoder, std::string_view body,
              bool trace, int mtp_steps) {
    complete_serial(connection, decoder, nullptr, body, trace, mtp_steps, false);
}

void complete(const Socket& connection, BatchService& service, int slot, std::string_view body) {
    complete_batch(connection, service, slot, nullptr, body);
}

void complete(const Socket& connection, const Decoder& decoder,
              const CompletionTokenizer& tokenizer, std::string_view body,
              bool trace, int mtp_steps, bool eager) {
    complete_serial(connection, decoder, &tokenizer, body, trace, mtp_steps, eager);
}

void complete(const Socket& connection, BatchService& service, int slot,
              const CompletionTokenizer& tokenizer, std::string_view body) {
    complete_batch(connection, service, slot, &tokenizer, body);
}

void complete_chat(const Socket& connection, const Decoder& decoder,
                   const CompletionTokenizer& tokenizer, std::string_view body,
                   bool trace, int mtp_steps, bool eager, VisionCompletion* vision) {
    complete_serial(connection, decoder, &tokenizer, body, trace, mtp_steps, eager, true, vision);
}

void complete_chat(const Socket& connection, BatchService& service, int slot,
                   const CompletionTokenizer& tokenizer, std::string_view body) {
    complete_batch(connection, service, slot, &tokenizer, body, true);
}

}
