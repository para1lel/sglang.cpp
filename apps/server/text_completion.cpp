#include "sglang/server/text_completion.h"
#include "sglang/server/text_stop.h"

#include "sglang/json.h"
#include "sglang/server/anthropic_count.h"
#include "sglang/server/anthropic_messages.h"
#include "sglang/server/http.h"
#include "sglang/utf8.h"

#include <algorithm>
#include <atomic>
#include <charconv>
#include <ctime>
#include <deque>
#include <iostream>
#include <utility>

namespace sglang::server {

const Tokenizer& CompletionTokenizer::get() const {
    std::call_once(loaded_, [this] {
        tokenizer_ = std::make_unique<const Tokenizer>(checkpoint_ / "tokenizer.json");
    });
    return *tokenizer_;
}

std::string completion_echo(const json::Value& input, const std::vector<int>& ids,
                            const Tokenizer& tokenizer) {
    if (const auto text = input.string()) return std::string(*text);
    Utf8Stream utf8;
    std::string text;
    for (const int id : ids) text += utf8.push(tokenizer.piece(id, true));
    text += utf8.finish();
    return text;
}

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

using ScoreRow = std::shared_ptr<const DecoderTokenLogprobs>;

struct CompletionScores {
    const std::vector<int>& prompt;
    const DecoderPromptLogprobs* input;
    const ScoreRow* output;
    std::size_t count;

    std::size_t input_count() const { return input ? prompt.size() : 0; }
    const DecoderTokenLogprobs* row(std::size_t index) const {
        if (index >= input_count()) return output[index - input_count()].get();
        return index ? (*input)[index - 1].get() : nullptr;
    }
};

void append_score(std::string& output, float value) {
    char number[32];
    const auto formatted = std::to_chars(number, number + sizeof(number), value);
    output.append(number, formatted.ptr);
}

std::string completion_score_text(const Tokenizer& tokenizer, int id) {
    const auto bytes = tokenizer.piece(id, false);
    if (valid_utf8(bytes)) {
        return std::string(bytes);
    }
    std::string text;
    for (const unsigned char byte : bytes) {
        append_utf8(text, byte);
    }
    return text;
}

void append_completion_scores(std::string& output, const Tokenizer& tokenizer,
                              const CompletionScores& scores, int top) {
    const auto count = scores.input_count() + scores.count;
    output += R"({"tokens":[)";
    for (std::size_t index = 0; index < count; ++index) {
        if (index) output += ',';
        const auto* row = scores.row(index);
        const int id = row ? row->selected.token_id : scores.prompt.front();
        json::append_quoted(output, completion_score_text(tokenizer, id));
    }
    output += R"(],"token_logprobs":[)";
    for (std::size_t index = 0; index < count; ++index) {
        if (index) output += ',';
        if (const auto* row = scores.row(index)) {
            append_score(output, row->selected.value);
        } else {
            output += "null";
        }
    }
    output += R"(],"top_logprobs":[)";
    if (top > 0) {
        for (std::size_t index = 0; index < count; ++index) {
            if (index) output += ',';
            const auto* row = scores.row(index);
            if (!row) {
                output += "null";
                continue;
            }
            output += '{';
            std::vector<std::pair<std::string, float>> alternatives;
            for (const auto& score : row->alternatives) {
                auto text = completion_score_text(tokenizer, score.token_id);
                const auto found = std::find_if(alternatives.begin(), alternatives.end(),
                    [&](const auto& alternative) { return alternative.first == text; });
                if (found != alternatives.end()) {
                    found->second = score.value;
                } else {
                    alternatives.emplace_back(std::move(text), score.value);
                }
            }
            for (const auto& [token, score] : alternatives) {
                if (&token != &alternatives.front().first) output += ',';
                json::append_quoted(output, token);
                output += ':';
                append_score(output, score);
            }
            output += '}';
        }
    }
    output += R"(],"text_offset":[)";
    for (std::size_t index = 0; index < count; ++index) {
        if (index) output += ',';
        output += "-1";
    }
    output += "]}";
}

void append_chat_score(std::string& output, const Tokenizer& tokenizer,
                       const DecoderLogprob& score) {
    const auto bytes = tokenizer.piece(score.token_id, false);
    Utf8Stream utf8;
    auto text = utf8.push(bytes);
    text += utf8.finish();
    output += R"({"token":)";
    json::append_quoted(output, text);
    output += R"(,"bytes":[)";
    for (const auto& byte : bytes) {
        if (&byte != bytes.data()) output += ',';
        output += std::to_string(static_cast<unsigned char>(byte));
    }
    output += R"(],"logprob":)";
    append_score(output, score.value);
}

void append_chat_scores(std::string& output, const Tokenizer& tokenizer,
                        const ScoreRow* rows, std::size_t count) {
    output += R"({"content":[)";
    for (std::size_t index = 0; index < count; ++index) {
        if (index) output += ',';
        const auto& row = rows[index];
        append_chat_score(output, tokenizer, row->selected);
        output += R"(,"top_logprobs":[)";
        for (const auto& score : row->alternatives) {
            if (&score != row->alternatives.data()) output += ',';
            append_chat_score(output, tokenizer, score);
            output += '}';
        }
        output += "]}";
    }
    output += "]}";
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
                 std::size_t index = 0, const std::string* echo = nullptr)
        : connection_(connection), stream_(stream), tokenizer_(tokenizer), request_(request),
          prompt_(prompt), index_(index),
          echo_(echo ? echo : request.echo ? &*request.echo : nullptr),
          prefix_(response_prefix(request)) {
        if (request.anthropic) {
            anthropic_.emplace(connection, stream, request.stream, prompt.size());
        }
        if (request.chat_output.enabled()) {
            parser_.emplace(request.chat_output);
        }
        if (!request.stop_text.empty()) {
            stop_patterns_.emplace(request.stop_text);
            stop_buffer_.emplace(*stop_patterns_);
        }
    }

    bool token(int id, const ScoreRow* score = nullptr);
    bool start() {
        return !request_.stream || stream_.start();
    }
    bool admit_messages(std::size_t cached) { anthropic_->admit(cached); return true; }
    void fail_messages(std::string_view type, const char* message) {
        anthropic_->fail(type, message);
    }
    bool poll() { return stream_.poll(); }
    void finish(const DecoderResult& result, std::size_t count);
    std::optional<std::string> finish_choice(const DecoderResult& result, std::size_t count);
    bool stopped() const { return text_stopped_ && !write_failed_; }
    void set_prompt_scores(const DecoderResult& result);

private:
    std::string filter_stop(std::string_view delta, bool final = false);
    void append_media_prompt_ids(std::string& output) const;
    void append_message(std::string& output, std::string_view text) const;
    void append_delta(std::string& output, const ChatOutputEvent* delta,
                      std::string_view call_id) const;
    std::string choice(std::string_view text, const int* ids, std::size_t count,
                       const char* reason = nullptr, const ChatOutputEvent* delta = nullptr,
                       std::string_view call_id = std::string_view(),
                       const ScoreRow* scores = nullptr) const;
    bool event(std::string_view data);
    bool accept(const ChatOutputEvent& delta, const int* id);
    bool project(std::string_view text, const int* id = nullptr, const ScoreRow* score = nullptr);
    const char* finish_output(const DecoderResult& result);
    void finish_stream(const std::string& choice, std::size_t count, std::size_t cached_tokens);

    const Socket& connection_;
    EventStream& stream_;
    const Tokenizer& tokenizer_;
    const TextRequest& request_;
    const std::vector<int>& prompt_;
    std::size_t index_;
    const std::string* echo_;
    std::shared_ptr<const DecoderPromptLogprobs> prompt_scores_;
    const std::string prefix_;
    Utf8Stream utf8_;
    std::optional<ChatOutputParser> parser_;
    std::optional<AnthropicMessages> anthropic_;
    std::string text_;
    std::string reasoning_;
    std::vector<ToolCall> tool_calls_;
    std::optional<TextStops> stop_patterns_;
    std::optional<TextStopBuffer> stop_buffer_;
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
                                 std::string_view call_id, const ScoreRow* scores) const {
    auto output = R"({"index":)" + std::to_string(index_);
    const bool first = !sent_event_;
    if (!request_.chat) {
        output += R"(,"text":)";
        if (first && echo_) {
            json::append_quoted(output, *echo_ + std::string(text));
        } else {
            json::append_quoted(output, text);
        }
    } else if (!request_.stream) {
        append_message(output, text);
    } else {
        const ChatOutputEvent raw{ChatOutputKind::content, text, 0, {}};
        append_delta(output, reason || parser_ ? delta : &raw, call_id);
    }
    const int top = request_.options.generated_top_logprobs;
    const auto* input = first ? prompt_scores_.get() : nullptr;
    if (top >= 0) {
        output += R"(,"logprobs":)";
        if (request_.stream && !scores && !input) {
            output += "null";
        } else if (request_.chat) {
            append_chat_scores(output, tokenizer_, scores, count);
        } else {
            append_completion_scores(output, tokenizer_, {prompt_, input, scores, count}, top);
        }
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

void TextResponse::set_prompt_scores(const DecoderResult& result) {
    if (!request_.options.score_prompt || prompt_scores_) return;
    const auto& scores = result.prompt_logprobs;
    if (!scores) {
        throw std::logic_error("Missing prompt logprobs");
    }
    if (scores->size() + 1 != prompt_.size()) {
        throw std::logic_error("Prompt logprob count differs from prompt token count: " +
                               std::to_string(scores->size()));
    }
    for (std::size_t index = 0; index < scores->size(); ++index) {
        const auto& row = (*scores)[index];
        if (!row || row->selected.token_id != prompt_[index + 1]) {
            throw std::logic_error("Prompt logprob ID differs from prompt token ID");
        }
    }
    prompt_scores_ = scores;
}

bool TextResponse::event(std::string_view data) {
    if (!start()) {
        write_failed_ = true;
        return false;
    }
    if (!std::exchange(sent_event_, true) && request_.chat) {
        auto role = prefix_ + R"("choices":[{"index":0,"delta":{"role":"assistant"})";
        if (request_.options.generated_top_logprobs >= 0) {
            role += R"(,"logprobs":null)";
        }
        role += R"(,"finish_reason":null}])";
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
    if (!stop_buffer_) {
        return std::string(delta);
    }
    if (text_stopped_) {
        return {};
    }
    stop_buffer_->push(delta);
    const auto match = stop_buffer_->match();
    text_stopped_ = match.has_value();
    return stop_buffer_->take(final, match);
}

bool TextResponse::accept(const ChatOutputEvent& delta, const int* id) {
    if (anthropic_) {
        write_failed_ = !anthropic_->accept(delta);
        return !write_failed_;
    }
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

bool TextResponse::project(std::string_view text, const int* id, const ScoreRow* score) {
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
    } else if (anthropic_) {
        return accept({ChatOutputKind::content, text, 0, {}}, id) && !disconnected(connection_);
    } else if (!request_.stream) {
        text_ += text;
    } else if (!text.empty() || (id && request_.return_token_ids) || score) {
        return event(choice_chunk(prefix_, choice(text, id, id ? 1 : 0, nullptr, nullptr, {},
                                                  score)));
    }
    return !disconnected(connection_);
}

bool TextResponse::token(int id, const ScoreRow* score) {
    std::string delta;
    if (!request_.options.stops_at(id)) {
        delta = utf8_.push(tokenizer_.piece(id, true));
    }
    if (!request_.stop_text.empty()) {
        delta = filter_stop(delta);
    }
    return project(delta, &id, score) && !text_stopped_;
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

const char* TextResponse::finish_output(const DecoderResult& result) {
    const auto tail = filter_stop(utf8_.finish(), true);
    auto reason = text_stopped_ ? "stop" : finish_reason(request_, result);
    if (!project(tail)) {
        return nullptr;
    }
    if (parser_) {
        const bool stopped = std::string_view(reason) == "stop";
        if (!parser_->finish(stopped ? ChatEnd::stop : ChatEnd::length,
                            [&](const auto& delta) { return accept(delta, nullptr); })) {
            return nullptr;
        }
        if (stopped && parser_->calls()) {
            reason = "tool_calls";
        }
    }
    return reason;
}

std::optional<std::string> TextResponse::finish_choice(const DecoderResult& result,
                                                       std::size_t count) {
    const auto* reason = finish_output(result);
    if (!reason) return std::nullopt;
    if (request_.stream) {
        return choice({}, nullptr, 0, reason);
    }
    return choice(text_, result.output_token_ids.data(), count, reason, nullptr, {},
                  result.generated_logprobs.data());
}

void TextResponse::finish(const DecoderResult& result, std::size_t count) {
    if (anthropic_) {
        if (const auto* reason = finish_output(result)) {
            anthropic_->finish(reason, count, result.cached_prompt_tokens);
        }
        return;
    }
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
                    const TextRequest& request, BatchPrompt& prompt, std::size_t index,
                    const std::string* echo = nullptr)
        : response(connection, stream, tokenizer, request, prompt.prompt, index, echo), stop_id(""),
          scored(request.options.generated_top_logprobs >= 0) {
        if (!request.stop_text.empty()) {
            prompt.request_id = &stop_id;
        }
    }

    bool consume(const Socket& connection, const DecoderResult& result, bool terminal) {
        response.set_prompt_scores(result);
        const auto& scores = result.generated_logprobs;
        if (scored && scores.size() != result.output_token_ids.size()) {
            throw std::logic_error("Generated logprob count differs from output token count");
        }
        while (!response.stopped() && count < result.output_token_ids.size()) {
            const auto* score = scored ? &scores[count] : nullptr;
            const int id = result.output_token_ids[count++];
            if (score && (!*score || (*score)->selected.token_id != id)) {
                throw std::logic_error("Generated logprob ID differs from output token ID");
            }
            if (response.token(id, score)) {
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
    const bool scored;
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
    DecoderCallbacks callbacks{
        [&](int token) { return response.token(token); },
        [&] { return disconnected(connection); },
        [&] { return response.start(); },
        [&] { return response.poll(); },
    };
    if (request.anthropic && request.stream) {
        callbacks.on_admitted = {};
        callbacks.on_prompt_admitted = [&](std::size_t cached) {
            return response.admit_messages(cached);
        };
    }
    try {
        const auto result = generate(request.prompt, request.options, callbacks);
        const bool complete = result.finish_reason != DecoderFinishReason::cancelled;
        if ((complete || response.stopped()) && !disconnected(connection)) {
            response.finish(result, result.output_token_ids.size());
        }
    } catch (const std::exception& error) {
        if (request.anthropic && request.stream) {
            std::cerr << "Messages generation failed: " << error.what() << '\n';
            if (dynamic_cast<const std::invalid_argument*>(&error)) {
                response.fail_messages("invalid_request_error", error.what());
            } else {
                response.fail_messages("api_error", "Internal server error");
            }
            return;
        }
        completion_failure(stream, error);
    }
}

void complete_scored_text(const Socket& connection, const TextRequest& request,
                          const CompletionTokenizer& tokenizer, BatchService& service, int slot) {
    if (disconnected(connection)) {
        return;
    }
    if (!service.supports_generated_logprobs()) {
        throw std::invalid_argument("Generated logprobs require server MTP disabled");
    }
    EventStream stream(connection);
    std::vector<BatchPrompt> prompts{{request.prompt, request.options}};
    BatchTextChoice choice(connection, stream, tokenizer.get(), request, prompts.front(), 0);
    const BatchCallbacks callbacks{
        [&] { return disconnected(connection); },
        [&] { return choice.response.start(); },
        [&] { return !request.stream || choice.response.poll(); },
        [&](std::size_t, const DecoderResult& result, bool terminal) {
            return choice.consume(connection, result, terminal);
        },
    };
    try {
        const auto results = service.generate_batch(slot, prompts, callbacks);
        if (results && !disconnected(connection)) {
            choice.response.finish(results->front(), choice.count);
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
    if (request.options.generated_top_logprobs >= 0 && !service.supports_generated_logprobs()) {
        throw std::invalid_argument("Generated logprobs require server MTP disabled");
    }
    EventStream stream(connection);
    const auto& tokens = tokenizer.get();
    std::deque<BatchTextChoice> choices;
    std::size_t prompt_tokens = 0;
    for (auto& prompt : completion.batch) {
        const auto index = choices.size();
        const auto* echo = completion.batch_echoes.empty() ? nullptr :
            &completion.batch_echoes[index];
        choices.emplace_back(connection, stream, tokens, request, prompt, index, echo);
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

TextRequest parse_chat_request(std::string_view body, const CompletionTokenizer& tokenizer,
                               bool media_enabled) {
    json::Value parsed;
    try {
        parsed = json::parse(body, json::Syntax::ordered);
    } catch (const json::ParseError&) {
        throw HttpError("400 Bad Request", "Invalid JSON request body");
    }
    if (!parsed.ordered_object()) {
        throw HttpError("400 Bad Request", "Expected a JSON object for a chat request");
    }
    return parse_chat_request(*parsed.ordered_object(), tokenizer, media_enabled);
}

void complete_chat_request(const Socket& connection, std::string_view body,
                           const CompletionTokenizer& tokenizer, const TextGenerator& generate,
                           const MediaGenerator& generate_media, BatchService& service, int slot,
                           bool anthropic) {
    if (disconnected(connection)) {
        return;
    }
    auto request = anthropic ? parse_chat_request(
        anthropic_messages_chat(json::parse(body, json::Syntax::python_request),
                               tokenizer.chat_parsers()), tokenizer, bool(generate_media)) :
        parse_chat_request(body, tokenizer, bool(generate_media));
    request.anthropic = anthropic;
    if (anthropic) request.chat_output.complete_truncated_calls = false;
    if (request.options.generated_top_logprobs >= 0) {
        complete_scored_text(connection, request, tokenizer, service, slot);
        return;
    }
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
