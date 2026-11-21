#include "sglang/server/text_completion.h"

#include "sglang/json.h"
#include "sglang/server/http.h"
#include "sglang/utf8.h"

#include <algorithm>
#include <atomic>
#include <ctime>
#include <deque>
#include <iostream>
#include <utility>

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
                  std::size_t cached_tokens) {
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

std::string choice_chunk(const std::string& prefix, const std::string& choice) {
    return prefix + R"("choices":[)" + choice + "]}";
}

void completion_failure(EventStream& stream, const std::exception& error) {
    if (!stream.started()) {
        throw;
    }
    std::cerr << "Completion failed: " << error.what() << '\n';
    if (stream.write(R"({"error":{"message":"Native decoder failed"}})")) {
        stream.end();
    }
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
    const bool reached_limit = count == budget ||
        (request.options.incremental_kv && count > 0 && count < budget);
    if (result.finish_reason == DecoderFinishReason::length && !stop_matched && reached_limit) {
        return "length";
    }
    throw std::logic_error("Decoder returned an inconsistent completion");
}

struct ToolCall {
    std::string id;
    std::string name;
    std::string arguments;
};

void append_call(std::string& output, std::size_t index, std::string_view id,
                 std::string_view name, std::string_view arguments, bool delta) {
    output += '{';
    if (delta) {
        output += R"("index":)" + std::to_string(index) + ',';
    }
    if (!id.empty()) {
        output += R"("type":"function","id":)";
        json::append_quoted(output, id);
        output += ',';
    }
    output += R"("function":{)";
    if (!name.empty()) {
        output += R"("name":)";
        json::append_quoted(output, name);
        output += ',';
    }
    output += R"("arguments":)";
    json::append_quoted(output, arguments);
    output += "}}";
}

class TextResponse {
public:
    TextResponse(const Socket& connection, EventStream& stream, const Tokenizer& tokenizer,
                 const TextRequest& request, const std::vector<int>& prompt,
                 std::size_t index = 0)
        : connection_(connection), stream_(stream), tokenizer_(tokenizer), request_(request),
          prompt_(prompt), index_(index),
          prefix_(response_prefix(request)) {
        if (request.chat_output.enabled()) {
            parser_.emplace(request.chat_output);
        }
    }

    bool token(int id);
    bool start() {
        return !request_.stream || stream_.start();
    }
    bool poll() { return stream_.poll(); }
    void finish(const DecoderResult& result);
    std::optional<std::string> finish_choice(const DecoderResult& result, std::size_t count);
    bool stopped() const { return text_stopped_ && !write_failed_; }

private:
    std::string filter_stop(std::string_view delta, bool final = false);
    void append_media_prompt_ids(std::string& output) const;
    void append_message(std::string& output, std::string_view text) const;
    void append_delta(std::string& output, const ChatOutputEvent* delta,
                      std::string_view call_id) const;
    std::string choice(std::string_view text, const int* ids, std::size_t count,
                       const char* reason = nullptr, const ChatOutputEvent* delta = nullptr,
                       std::string_view call_id = std::string_view()) const;
    bool event(std::string_view data);
    bool accept(const ChatOutputEvent& delta, const int* id);
    bool project(std::string_view text, const int* id = nullptr);
    void finish_stream(const std::string& choice, std::size_t count, std::size_t cached_tokens);

    const Socket& connection_;
    EventStream& stream_;
    const Tokenizer& tokenizer_;
    const TextRequest& request_;
    const std::vector<int>& prompt_;
    std::size_t index_;
    const std::string prefix_;
    Utf8Stream utf8_;
    std::optional<ChatOutputParser> parser_;
    std::string text_;
    std::string reasoning_;
    std::vector<ToolCall> tool_calls_;
    std::string pending_stop_;
    bool text_stopped_ = false;
    bool sent_event_ = false;
    bool write_failed_ = false;
};

void TextResponse::append_media_prompt_ids(std::string& output) const {
    if (request_.media && request_.return_token_ids) {
        output += R"(,"prompt_token_ids":)";
        append_ids(output, prompt_.data(), prompt_.size());
    }
}

void TextResponse::append_message(std::string& output, std::string_view text) const {
    output += R"(,"message":{"role":"assistant","content":)";
    if (text.empty() && !tool_calls_.empty()) {
        output += "null";
    } else {
        json::append_quoted(output, text);
    }
    if (!reasoning_.empty()) {
        output += R"(,"reasoning_content":)";
        json::append_quoted(output, reasoning_);
    }
    if (!tool_calls_.empty()) {
        output += R"(,"tool_calls":[)";
        for (const auto& call : tool_calls_) {
            if (&call != &tool_calls_.front()) {
                output += ',';
            }
            append_call(output, 0, call.id, call.name, call.arguments, false);
        }
        output += ']';
    }
    output += '}';
}

void TextResponse::append_delta(std::string& output, const ChatOutputEvent* delta,
                                std::string_view call_id) const {
    output += R"(,"delta":{)";
    if (delta) {
        switch (delta->kind) {
        case ChatOutputKind::content:
        case ChatOutputKind::reasoning:
            output += delta->kind == ChatOutputKind::content ? R"("content":)" :
                                                               R"("reasoning_content":)";
            json::append_quoted(output, delta->text);
            break;
        case ChatOutputKind::tool_call:
            output += R"("tool_calls":[)";
            append_call(output, delta->index, call_id, call_id.empty() ? std::string_view{} :
                        delta->name, call_id.empty() ? delta->text : std::string_view{}, true);
            output += ']';
            break;
        }
    }
    output += '}';
}

std::string TextResponse::choice(std::string_view text, const int* ids, std::size_t count,
                                 const char* reason, const ChatOutputEvent* delta,
                                 std::string_view call_id) const {
    auto output = R"({"index":)" + std::to_string(index_);
    if (!request_.chat) {
        output += R"(,"text":)";
        json::append_quoted(output, text);
    } else if (!request_.stream) {
        append_message(output, text);
    } else {
        const ChatOutputEvent raw{ChatOutputKind::content, text, 0, {}};
        append_delta(output, reason || parser_ ? delta : &raw, call_id);
    }
    if (request_.return_token_ids) {
        output += R"(,"token_ids":)";
        append_ids(output, ids, count);
        if (!sent_event_ && !request_.media) {
            output += R"(,"prompt_token_ids":)";
            append_ids(output, prompt_.data(), prompt_.size());
        }
    }
    output += R"(,"finish_reason":)";
    if (reason) {
        json::append_quoted(output, reason);
    } else {
        output += "null";
    }
    output += '}';
    return output;
}

bool TextResponse::event(std::string_view data) {
    if (!start()) {
        write_failed_ = true;
        return false;
    }
    if (!std::exchange(sent_event_, true) && request_.chat) {
        auto role = prefix_ + R"("choices":[{"index":0,"delta":{"role":"assistant"},)"
                              R"("finish_reason":null}])";
        append_media_prompt_ids(role);
        if (!stream_.write(role + '}')) {
            write_failed_ = true;
            return false;
        }
    }
    write_failed_ = !stream_.write(data);
    return !write_failed_;
}

std::string TextResponse::filter_stop(std::string_view delta, bool final) {
    if (request_.stop_text.empty()) {
        return std::string(delta);
    }
    if (text_stopped_) {
        return {};
    }
    pending_stop_ += delta;
    std::size_t stop_position = std::string::npos;
    std::size_t held = 0;
    for (const auto& stop : request_.stop_text) {
        stop_position = std::min(stop_position, pending_stop_.find(stop));
        if (!final && stop_position == std::string::npos) {
            const std::string_view pending = pending_stop_;
            for (auto size = std::min(stop.size() - 1, pending.size()); size > held; --size) {
                if (pending.substr(pending.size() - size) ==
                    std::string_view(stop).substr(0, size)) {
                    held = size;
                    break;
                }
            }
        }
    }
    text_stopped_ = stop_position != std::string::npos;
    const auto ready = text_stopped_ ? stop_position : pending_stop_.size() - held;
    auto output = pending_stop_.substr(0, ready);
    pending_stop_.erase(0, ready);
    return output;
}

bool TextResponse::accept(const ChatOutputEvent& delta, const int* id) {
    std::string call_id;
    if (delta.kind == ChatOutputKind::tool_call) {
        static std::atomic<std::uint64_t> next_call_id{1};
        call_id = "call_" + std::to_string(next_call_id.fetch_add(1, std::memory_order_relaxed));
    }
    if (request_.stream) {
        return event(choice_chunk(prefix_, choice({}, id, id ? 1 : 0, nullptr, &delta, call_id))) &&
            (call_id.empty() || event(choice_chunk(prefix_, choice({}, nullptr, 0, nullptr,
                                                                   &delta))));
    }
    switch (delta.kind) {
    case ChatOutputKind::content:
        text_ += delta.text;
        break;
    case ChatOutputKind::reasoning:
        reasoning_ += delta.text;
        break;
    case ChatOutputKind::tool_call:
        tool_calls_.push_back({std::move(call_id), std::string(delta.name),
                               std::string(delta.text)});
        break;
    }
    return true;
}

bool TextResponse::project(std::string_view text, const int* id) {
    if (parser_) {
        if (!parser_->push(text, [&](const ChatOutputEvent& delta) {
                const bool sent = accept(delta, id);
                id = nullptr;
                return sent;
            })) {
            return false;
        }
        if (request_.stream && request_.return_token_ids && id) {
            return event(choice_chunk(prefix_, choice({}, id, 1)));
        }
    } else if (!request_.stream) {
        text_ += text;
    } else if (!text.empty() || (id && request_.return_token_ids)) {
        return event(choice_chunk(prefix_, choice(text, id, id ? 1 : 0)));
    }
    return !disconnected(connection_);
}

bool TextResponse::token(int id) {
    std::string delta;
    if (!request_.options.stops_at(id)) {
        delta = utf8_.push(tokenizer_.piece(id, true));
    }
    if (!request_.stop_text.empty()) {
        delta = filter_stop(delta);
    }
    return project(delta, &id) && !text_stopped_;
}

void TextResponse::finish_stream(const std::string& choice, std::size_t count,
                                 std::size_t cached_tokens) {
    if (!event(choice_chunk(prefix_, choice))) {
        return;
    }
    if (request_.include_usage) {
        auto usage = prefix_ + R"("choices":[])";
        append_usage(usage, prompt_.size(), count, cached_tokens);
        if (!event(usage + '}')) {
            return;
        }
    }
    if (event("[DONE]")) {
        stream_.end();
    }
}

std::optional<std::string> TextResponse::finish_choice(const DecoderResult& result,
                                                       std::size_t count) {
    const auto tail = filter_stop(utf8_.finish(), true);
    auto reason = text_stopped_ ? "stop" : finish_reason(request_, result);
    if (!project(tail)) {
        return std::nullopt;
    }
    if (parser_) {
        const bool stopped = std::string_view(reason) == "stop";
        if (!parser_->finish(stopped ? ChatEnd::stop : ChatEnd::length,
                            [&](const auto& delta) { return accept(delta, nullptr); })) {
            return std::nullopt;
        }
        if (stopped && parser_->calls()) {
            reason = "tool_calls";
        }
    }
    if (request_.stream) {
        return choice({}, nullptr, 0, reason);
    }
    return choice(text_, result.output_token_ids.data(), count, reason);
}

void TextResponse::finish(const DecoderResult& result) {
    const auto count = result.output_token_ids.size();
    const auto final = finish_choice(result, count);
    if (!final) {
        return;
    }
    if (request_.stream) {
        finish_stream(*final, count, result.cached_prompt_tokens);
        return;
    }
    auto output = prefix_ + R"("choices":[)" + *final + ']';
    append_media_prompt_ids(output);
    append_usage(output, prompt_.size(), count, result.cached_prompt_tokens);
    write_json(connection_, "200 OK", output + '}');
}

struct BatchTextChoice {
    BatchTextChoice(const Socket& connection, EventStream& stream, const Tokenizer& tokenizer,
                    const TextRequest& request, BatchPrompt& prompt, std::size_t index)
        : response(connection, stream, tokenizer, request, prompt.prompt, index), stop_id("") {
        if (!request.stop_text.empty()) {
            prompt.request_id = &stop_id;
        }
    }

    bool consume(const Socket& connection, const DecoderResult& result, bool terminal) {
        while (!response.stopped() && count < result.output_token_ids.size()) {
            if (response.token(result.output_token_ids[count++])) {
                continue;
            }
            if (!response.stopped() || disconnected(connection)) {
                return false;
            }
            if (!terminal) {
                stop_id.abort(true);
            }
        }
        return true;
    }

    TextResponse response;
    BatchRequestId stop_id;
    std::size_t count = 0;
    std::string final;
};

}

void complete_text(const Socket& connection, const TextRequest& request,
                   const CompletionTokenizer& tokenizer, const TextGenerator& generate) {
    if (disconnected(connection)) {
        return;
    }
    EventStream stream(connection);
    TextResponse response(connection, stream, tokenizer.get(), request, request.prompt);
    const DecoderCallbacks callbacks{
        [&](int token) { return response.token(token); },
        [&] { return disconnected(connection); },
        [&] { return response.start(); },
        [&] { return response.poll(); },
    };
    try {
        const auto result = generate(request.prompt, request.options, callbacks);
        const bool complete = result.finish_reason != DecoderFinishReason::cancelled;
        if ((complete || response.stopped()) && !disconnected(connection)) {
            response.finish(result);
        }
    } catch (const std::exception& error) {
        completion_failure(stream, error);
    }
}

void complete_text_batch(const Socket& connection, TextCompletionRequest completion,
                         const CompletionTokenizer& tokenizer, BatchService& service, int slot) {
    if (disconnected(connection)) {
        return;
    }
    const auto& request = completion.request;
    EventStream stream(connection);
    const auto& tokens = tokenizer.get();
    std::deque<BatchTextChoice> choices;
    std::size_t prompt_tokens = 0;
    for (auto& prompt : completion.batch) {
        choices.emplace_back(connection, stream, tokens, request, prompt, choices.size());
        prompt_tokens += prompt.prompt.size();
    }
    std::size_t completion_tokens = 0;
    std::size_t cached_tokens = 0;
    const BatchCallbacks callbacks{
        [&] { return disconnected(connection); },
        [&] { return !request.stream || stream.start(); },
        [&] { return !request.stream || stream.poll(); },
        [&](std::size_t index, const DecoderResult& result, bool terminal) {
            auto& choice = choices[index];
            if (!choice.consume(connection, result, terminal)) {
                return false;
            }
            if (!terminal) {
                return true;
            }
            auto final = choice.response.finish_choice(result, choice.count);
            if (!final) {
                return false;
            }
            completion_tokens += choice.count;
            cached_tokens += result.cached_prompt_tokens;
            if (request.stream) {
                return stream.write(choice_chunk("{", *final));
            }
            choice.final = std::move(*final);
            return true;
        },
    };
    try {
        if (!service.generate_batch(slot, completion.batch, callbacks) ||
            disconnected(connection)) {
            return;
        }
        std::string output = R"({"choices":[)";
        if (request.stream) {
            output += ']';
            if (request.include_usage) {
                append_usage(output, prompt_tokens, completion_tokens, cached_tokens);
                if (!stream.write(output + '}')) {
                    return;
                }
            }
            if (stream.write("[DONE]")) {
                stream.end();
            }
            return;
        }
        for (const auto& choice : choices) {
            if (&choice != &choices.front()) {
                output += ',';
            }
            output += choice.final;
        }
        output += ']';
        append_usage(output, prompt_tokens, completion_tokens, cached_tokens);
        write_json(connection, "200 OK", output + '}');
    } catch (const std::exception& error) {
        completion_failure(stream, error);
    }
}

void complete_chat_request(const Socket& connection, std::string_view body,
                           const CompletionTokenizer& tokenizer, const TextGenerator& generate,
                           const MediaGenerator& generate_media) {
    if (disconnected(connection)) {
        return;
    }
    const auto request = parse_chat_request(body, tokenizer, bool(generate_media));
    if (!request.media) {
        complete_text(connection, request, tokenizer, generate);
        return;
    }
    complete_text(connection, request, tokenizer,
                  [&](const auto&, const auto& options, const auto& callbacks) {
                      return generate_media(*request.media, options, callbacks);
                  });
}

}
