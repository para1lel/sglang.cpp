#include "sglang/server/text_completion.h"

#include "sglang/json.h"
#include "sglang/server/batch_service.h"
#include "sglang/server/http.h"
#include "sglang/utf8.h"

#include <atomic>
#include <ctime>
#include <iostream>

namespace sglang::server {

void append_ids(std::string& output, const int* ids, std::size_t count) {
    output += '[';
    for (std::size_t index = 0; index < count; ++index) {
        if (index) {
            output += ',';
        }
        output += std::to_string(ids[index]);
    }
    output += ']';
}

namespace {

void append_usage(std::string& output, std::size_t prompt, std::size_t completion,
                  int cached_tokens) {
    output += R"(,"usage":{"prompt_tokens":)" + std::to_string(prompt);
    output += R"(,"completion_tokens":)" + std::to_string(completion);
    output += R"(,"total_tokens":)" + std::to_string(prompt + completion);
    output += R"(,"prompt_tokens_details":{"cached_tokens":)" +
              std::to_string(cached_tokens) + "}}";
}

std::string response_prefix(const TextRequest& request) {
    if (!request.chat) {
        return "{";
    }
    static std::atomic<std::uint64_t> next_id{1};
    std::string output = R"({"id":"chatcmpl-)";
    output += std::to_string(next_id.fetch_add(1, std::memory_order_relaxed));
    output += R"(","object":)";
    output += request.stream ? R"("chat.completion.chunk")" : R"("chat.completion")";
    output += R"(,"created":)" + std::to_string(std::time(nullptr));
    output += R"(,"model":)";
    json::append_quoted(output, model_name);
    output += ',';
    return output;
}

const char* finish_reason(const TextRequest& request, const DecoderResult& result) {
    const auto count = result.output_token_ids.size();
    const auto budget = static_cast<std::size_t>(request.options.max_new_tokens);
    if (result.finish_reason == DecoderFinishReason::constraint &&
        request.options.constraint && count && count <= budget) {
        return "stop";
    }
    const bool stop_matched = count && request.options.stops_at(result.output_token_ids.back());
    if (result.finish_reason == DecoderFinishReason::stop && stop_matched && count <= budget) {
        return "stop";
    }
    if (result.finish_reason == DecoderFinishReason::length && !stop_matched && count == budget) {
        return "length";
    }
    throw std::logic_error("Decoder returned an inconsistent completion");
}

class TextResponse {
public:
    TextResponse(const Socket& connection, const Tokenizer& tokenizer, const TextRequest& request)
        : connection_(connection), tokenizer_(tokenizer), request_(request),
          prefix_(response_prefix(request)) {}

    bool token(int id);
    void finish(const DecoderResult& result);
    bool started() const { return streaming_; }

private:
    void append_media_prompt_ids(std::string& output) const;
    void append_text(std::string& output, std::string_view text, bool finished) const;
    std::string choice(std::string_view text, const int* ids, std::size_t count,
                       const char* reason = nullptr) const;
    bool event(std::string_view data);
    void finish_stream(const char* reason, std::size_t count, std::string_view tail,
                       int cached_tokens);

    const Socket& connection_;
    const Tokenizer& tokenizer_;
    const TextRequest& request_;
    const std::string prefix_;
    Utf8Stream utf8_;
    std::string text_;
    bool streaming_ = false;
};

void TextResponse::append_media_prompt_ids(std::string& output) const {
    if (request_.media && request_.return_token_ids) {
        output += R"(,"prompt_token_ids":)";
        append_ids(output, request_.prompt.data(), request_.prompt.size());
    }
}

void TextResponse::append_text(std::string& output, std::string_view text, bool finished) const {
    if (!request_.chat) {
        output += R"(,"text":)";
        json::append_quoted(output, text);
    } else if (!request_.stream) {
        output += R"(,"message":{"role":"assistant","content":)";
        json::append_quoted(output, text);
        output += '}';
    } else {
        output += R"(,"delta":{)";
        if (!finished) {
            output += R"("content":)";
            json::append_quoted(output, text);
        }
        output += '}';
    }
}

std::string TextResponse::choice(std::string_view text, const int* ids, std::size_t count,
                                 const char* reason) const {
    auto output = prefix_ + R"("choices":[{"index":0)";
    append_text(output, text, reason != nullptr);
    if (request_.return_token_ids) {
        output += R"(,"token_ids":)";
        append_ids(output, ids, count);
        if (!streaming_ && !request_.media) {
            output += R"(,"prompt_token_ids":)";
            append_ids(output, request_.prompt.data(), request_.prompt.size());
        }
    }
    output += R"(,"finish_reason":)";
    if (reason) {
        json::append_quoted(output, reason);
    } else {
        output += "null";
    }
    output += "}]";
    if (!request_.stream) {
        append_media_prompt_ids(output);
    }
    return output;
}

bool TextResponse::event(std::string_view data) {
    if (!streaming_) {
        streaming_ = start_events(connection_);
        if (streaming_ && request_.chat) {
            auto role = prefix_ + R"("choices":[{"index":0,"delta":{"role":"assistant"},)"
                                  R"("finish_reason":null}])";
            append_media_prompt_ids(role);
            if (!write_event(connection_, role + '}')) {
                return false;
            }
        }
    }
    return streaming_ && write_event(connection_, data);
}

bool TextResponse::token(int id) {
    std::string delta;
    if (!request_.options.stops_at(id)) {
        delta = utf8_.push(tokenizer_.piece(id, true));
    }
    if (!request_.stream) {
        text_ += delta;
        return !disconnected(connection_);
    }
    if (!delta.empty() || request_.return_token_ids) {
        return event(choice(delta, &id, 1) + '}');
    }
    return !disconnected(connection_);
}

void TextResponse::finish_stream(const char* reason, std::size_t count, std::string_view tail,
                                 int cached_tokens) {
    if (!tail.empty() && !event(choice(tail, nullptr, 0) + '}')) {
        return;
    }
    if (!event(choice({}, nullptr, 0, reason) + '}')) {
        return;
    }
    if (request_.include_usage) {
        auto usage = prefix_ + R"("choices":[])";
        append_usage(usage, request_.prompt.size(), count, cached_tokens);
        if (!event(usage + '}')) {
            return;
        }
    }
    if (event("[DONE]")) {
        end_events(connection_);
    }
}

void TextResponse::finish(const DecoderResult& result) {
    const auto reason = finish_reason(request_, result);
    const auto count = result.output_token_ids.size();
    const auto tail = utf8_.finish();
    if (request_.stream) {
        finish_stream(reason, count, tail, result.cached_prompt_tokens);
        return;
    }
    text_ += tail;
    auto output = choice(text_, result.output_token_ids.data(), count, reason);
    append_usage(output, request_.prompt.size(), count, result.cached_prompt_tokens);
    write_json(connection_, "200 OK", output + '}');
}

}

void complete_text(const Socket& connection, const TextRequest& request,
                   const CompletionTokenizer& tokenizer, const TextGenerator& generate,
                   bool trace_serial) {
    if (disconnected(connection)) {
        return;
    }
    TextResponse response(connection, tokenizer.get(), request);
    bool first_token = true;
    const DecoderCallbacks callbacks{
        [&](int token) {
            if (trace_serial) {
                trace_batch_step(1, first_token ? 0 : 1, 0);
                first_token = false;
            }
            return response.token(token);
        },
        [&] { return disconnected(connection); },
    };
    try {
        const auto result = generate(request.prompt, request.options, callbacks);
        if (result.finish_reason == DecoderFinishReason::cancelled) {
            if (trace_serial) {
                trace_batch_step(0, 0, 0);
            }
        } else if (!disconnected(connection)) {
            response.finish(result);
        }
    } catch (const std::exception& error) {
        if (!response.started()) {
            throw;
        }
        std::cerr << "Completion failed: " << error.what() << '\n';
        if (write_event(connection, R"({"error":{"message":"Native decoder failed"}})")) {
            end_events(connection);
        }
    }
}

void complete_chat_request(const Socket& connection, std::string_view body,
                           const CompletionTokenizer& tokenizer, const TextGenerator& generate,
                           const MediaGenerator& generate_media, bool trace_serial) {
    if (disconnected(connection)) {
        return;
    }
    const auto request = parse_chat_request(body, tokenizer, bool(generate_media));
    if (!request.media) {
        complete_text(connection, request, tokenizer, generate, trace_serial);
        return;
    }
    complete_text(connection, request, tokenizer,
                  [&](const auto&, const auto& options, const auto& callbacks) {
                      return generate_media(*request.media, options, callbacks);
                  }, trace_serial);
}

}
