#include "sglang/json.h"
#include "sglang/server/anthropic_count.h"
#include "sglang/server/anthropic_messages.h"
#include "sglang/server/chat_template.h"
#include "sglang/server/http.h"
#include "sglang/server/responses.h"
#include "sglang/server/text_completion.h"
#include "sglang/server/text_stop.h"
#include "sglang/server/vision_completion.h"
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
        tokenizer_ = std::make_unique<const Tokenizer>(checkpoint_ / profile_.tokenizer_file);
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

void append_usage(std::string& output, CompletionUsage usage, bool include_total = true) {
    output += R"(,"usage":{"prompt_tokens":)" + std::to_string(usage.prompt);
    output += R"(,"completion_tokens":)" + std::to_string(usage.output);
    if (include_total) {
        output += R"(,"total_tokens":)" + std::to_string(usage.prompt + usage.output);
    }
    output += R"(,"prompt_tokens_details":{"cached_tokens":)" + std::to_string(usage.cached) + "}}";
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
            const auto alternatives =
                distinct_score_alternatives(tokenizer, *row, completion_score_text);
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
    const auto text = repair_utf8(bytes);
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

std::string response_prefix(const TextRequest& request, std::string_view served_name,
                            std::time_t created) {
    if (!request.chat && request.choices == 1) {
        return "{";
    }
    static std::atomic<std::uint64_t> next_id{1};
    std::string output = request.chat ? R"({"id":"chatcmpl-)" : R"({"id":"cmpl-)";
    output += std::to_string(next_id.fetch_add(1, std::memory_order_relaxed));
    output += R"(","object":)";
    output += !request.chat ? R"("text_completion")" :
        request.stream ? R"("chat.completion.chunk")" : R"("chat.completion")";
    output += R"(,"created":)" + std::to_string(created);
    output += R"(,"model":)";
    json::append_quoted(output, served_name);
    output += ',';
    return output;
}

}

void finish_completion_stream(EventStream& stream, std::string_view prefix,
                              std::optional<CompletionUsage> usage, bool include_total) {
    if (usage) {
        std::string output(prefix);
        output += R"("choices":[])";
        append_usage(output, *usage, include_total);
        if (!stream.write(output + '}'))
            return;
    }
    if (stream.write("[DONE]"))
        stream.end();
}

void append_score(std::string& output, float value) {
    char number[32];
    const auto formatted = std::to_chars(number, number + sizeof(number), value);
    output.append(number, formatted.ptr);
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

void check_generated_scores(const DecoderResult& result) {
    if (result.generated_logprobs.size() != result.output_token_ids.size()) {
        throw std::logic_error("Generated logprob count differs from output token count");
    }
}

const ScoreRow& generated_score(const DecoderResult& result, std::size_t index) {
    const auto& row = result.generated_logprobs[index];
    if (!row || row->selected.token_id != result.output_token_ids[index]) {
        throw std::logic_error("Generated logprob ID differs from output token ID");
    }
    return row;
}

std::vector<std::pair<std::string, float>>
distinct_score_alternatives(const Tokenizer& tokenizer, const DecoderTokenLogprobs& row,
                            std::string (*display)(const Tokenizer&, int)) {
    std::vector<std::pair<std::string, float>> alternatives;
    alternatives.reserve(row.alternatives.size());
    for (const auto& score : row.alternatives) {
        auto text = display(tokenizer, score.token_id);
        const auto found =
            std::find_if(alternatives.begin(), alternatives.end(),
                         [&](const auto& alternative) { return alternative.first == text; });
        if (found == alternatives.end())
            alternatives.emplace_back(std::move(text), score.value);
        else
            found->second = score.value;
    }
    return alternatives;
}

bool write_completion_event(EventStream& stream, std::string_view type, json::OrderedObject fields,
                            int* sequence) {
    fields.emplace_back("type", std::string(type));
    if (sequence != nullptr)
        fields.emplace_back("sequence_number", (*sequence)++);
    return stream.write(chat_template_json(fields), type);
}

const char* completion_reason(const DecoderOptions& options, const DecoderResult& result,
                              bool matched_text_stop, bool constraint_budget_is_length) {
    const auto count = result.output_token_ids.size();
    const auto budget = static_cast<std::size_t>(options.max_new_tokens);
    const bool stop_matched = count != 0 && options.stops_at(result.output_token_ids.back());
    if (result.finish_reason == DecoderFinishReason::constraint && options.constraint &&
        count != 0 && count <= budget) {
        return constraint_budget_is_length && count == budget && !stop_matched ? "length" : "stop";
    }
    if (result.finish_reason == DecoderFinishReason::stop &&
        (stop_matched || matched_text_stop) && count <= budget) {
        return "stop";
    }
    const bool reached_limit =
        count == budget || (options.incremental_kv && count != 0 && count < budget);
    if (result.finish_reason == DecoderFinishReason::length && !stop_matched && reached_limit) {
        return "length";
    }
    throw std::logic_error(constraint_budget_is_length
                               ? "Decoder returned an inconsistent /generate result"
                               : "Decoder returned an inconsistent completion");
}

namespace {

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
  TextResponse(const Socket& connection, EventStream& stream, const CompletionTokenizer& loader,
               const TextRequest& request, const std::vector<int>& prompt, std::size_t index = 0,
               const std::string* echo = nullptr, std::time_t created = std::time(nullptr))
      : connection_(connection), stream_(stream), tokenizer_(loader.get()), request_(request),
        prompt_(prompt), index_(index), echo_(echo),
        prefix_(request.anthropic || request.responses != nullptr
                    ? std::string{}
                    : response_prefix(request, loader.serving_profile().served_name, created)) {
      if (echo_ == nullptr && request.echo) {
          echo_ = &*request.echo;
      }
      if (request.anthropic) {
          anthropic_.emplace(connection, stream, request.stream, prompt.size());
      }
      if (request.responses != nullptr) {
          responses_.emplace(connection, stream, *request.responses, request);
      }
      if (request.chat_output.enabled()) {
          loader.serving_profile().start_chat_output(parser_, request.chat_output);
      }
      if (!request.stop_text.empty()) {
          stop_patterns_ = std::make_shared<const TextStops>(request.stop_text);
          stop_buffer_.emplace(*stop_patterns_);
      }
  }

    bool token(int id, const ScoreRow* score = nullptr);
    bool start() {
        if (responses_) return responses_->start();
        return !request_.stream || stream_.start();
    }
    bool admit_messages(std::size_t cached) { anthropic_->admit(cached); return true; }
    void fail_messages(std::string_view type, const char* message) {
        anthropic_->fail(type, message);
    }
    void fail_responses() { responses_->fail(); }
    bool queue_responses() { return responses_->queue(); }
    bool cancelled() const;
    bool poll() { return stream_.poll(); }
    void finish(const DecoderResult& result, std::size_t count);
    std::optional<std::string> finish_choice(const DecoderResult& result, std::size_t count);
    const std::string& prefix() const { return prefix_; }
    bool write_choice(const std::string& choice);
    std::function<int(const int*, int, bool)> text_stop() const {
        auto stop = std::make_shared<TokenTextStop>(stop_patterns_);
        return [stop = std::move(stop), tokenizer = &tokenizer_, options = &request_.options]
               (const int* ids, int count, bool terminal) {
            for (int index = 0; index < count; ++index) {
                const int id = ids[index];
                const auto piece = options->stops_at(id) ? std::string_view{} :
                                                           tokenizer->piece(id, true);
                stop->push(piece, terminal && index + 1 == count);
                if (stop->match(false)) return index + 1;
            }
            return 0;
        };
    }
    void set_prompt_scores(const DecoderResult& result);

private:
  std::string filter_stop(std::string delta, bool final);
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
  std::optional<ResponsesOutput> responses_;
  std::string text_;
  std::string reasoning_;
  std::vector<ToolCall> tool_calls_;
  std::shared_ptr<const TextStops> stop_patterns_;
  std::optional<TextStopBuffer> stop_buffer_;
  bool text_stopped_ = false;
  bool sent_event_ = false;
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
    if (!start())
        return false;
    if (!std::exchange(sent_event_, true) && request_.chat) {
        auto role = prefix_ + R"("choices":[{"index":)" + std::to_string(index_) +
            R"(,"delta":{"role":"assistant"})";
        if (request_.options.generated_top_logprobs >= 0) {
            role += R"(,"logprobs":null)";
        }
        role += R"(,"finish_reason":null}])";
        append_media_prompt_ids(role);
        if (!stream_.write(role + '}'))
            return false;
    }
    return stream_.write(data);
}

bool TextResponse::write_choice(const std::string& choice) {
    auto output = prefix_ + R"("choices":[)" + choice + ']';
    if (!request_.chat && request_.choices > 1) {
        output += R"(,"usage":null)";
    }
    return event(output + '}');
}

std::string TextResponse::filter_stop(std::string delta, bool final) {
    if (!stop_buffer_)
        return delta;
    if (text_stopped_) {
        return {};
    }
    stop_buffer_->push(delta);
    const auto match = stop_buffer_->match();
    text_stopped_ = match.has_value();
    return stop_buffer_->take(final, match);
}

bool TextResponse::accept(const ChatOutputEvent& delta, const int* id) {
    if (anthropic_)
        return anthropic_->accept(delta);
    if (responses_)
        return responses_->accept(delta);
    std::string call_id;
    if (delta.kind == ChatOutputKind::tool_call) {
        static std::atomic<std::uint64_t> next_call_id{1};
        call_id = "call_" + std::to_string(next_call_id.fetch_add(1, std::memory_order_relaxed));
    }
    if (request_.stream) {
        return write_choice(choice({}, id, id ? 1 : 0, nullptr, &delta, call_id)) &&
            (call_id.empty() || write_choice(choice({}, nullptr, 0, nullptr, &delta)));
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
            return write_choice(choice({}, id, 1));
        }
    } else if (anthropic_ || responses_) {
        if (!accept({ChatOutputKind::content, text, 0, {}}, id)) {
            return false;
        }
    } else if (!request_.stream) {
        text_ += text;
    } else if (!text.empty() || (id && request_.return_token_ids) || score) {
        return write_choice(choice(text, id, id ? 1 : 0, nullptr, nullptr, {}, score));
    }
    return !cancelled();
}

bool TextResponse::cancelled() const {
    return responses_ && request_.responses->background ? stopping() || responses_->cancelled()
                                                        : disconnected(connection_);
}

bool TextResponse::token(int id, const ScoreRow* score) {
    std::string delta;
    if (responses_) responses_->token(tokenizer_.piece(id, false));
    if (!request_.options.stops_at(id)) {
        delta = utf8_.push(tokenizer_.piece(id, true));
    }
    if (stop_buffer_)
        delta = filter_stop(std::move(delta), false);
    return project(delta, &id, score);
}

const char* TextResponse::finish_output(const DecoderResult& result) {
    const auto tail = filter_stop(utf8_.finish(), true);
    const auto* reason = completion_reason(request_.options, result, text_stopped_);
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
    if (anthropic_ || responses_) {
        if (const auto* reason = finish_output(result)) {
            if (anthropic_) {
                anthropic_->finish(reason, count, result.cached_prompt_tokens);
            } else if (responses_) {
                responses_->finish(reason, count, result, tokenizer_);
            }
        }
        return;
    }
    const auto final = finish_choice(result, count);
    if (!final) {
        return;
    }
    if (request_.stream) {
        if (write_choice(*final)) {
            const CompletionUsage usage{prompt_.size(), count, result.cached_prompt_tokens};
            const auto trailer = request_.include_usage ? std::optional{usage} : std::nullopt;
            finish_completion_stream(stream_, prefix_, trailer);
        }
        return;
    }
    auto output = prefix_ + R"("choices":[)" + *final + ']';
    append_media_prompt_ids(output);
    append_usage(output, {prompt_.size(), count, result.cached_prompt_tokens});
    write_json(connection_, "200 OK", output + '}');
}

struct BatchTextChoice {
    BatchTextChoice(const Socket& connection, EventStream& stream,
                    const CompletionTokenizer& loader, const TextRequest& request,
                    BatchPrompt& prompt, std::size_t index,
                    const std::string* echo = nullptr, std::time_t created = std::time(nullptr))
        : response(connection, stream, loader, request, prompt.prompt, index, echo, created),
          scored(request.options.generated_top_logprobs >= 0) {
        if (!request.stop_text.empty()) {
            prompt.options.text_stop = response.text_stop();
        }
    }

    bool consume(const DecoderResult& result) {
        response.set_prompt_scores(result);
        if (scored)
            check_generated_scores(result);
        while (count < result.output_token_ids.size()) {
            const auto index = count++;
            const auto* score = scored ? &generated_score(result, index) : nullptr;
            if (!response.token(result.output_token_ids[index], score))
                return false;
        }
        return true;
    }

    TextResponse response;
    const bool scored;
    std::size_t count = 0;
    std::string final;
};

void prepare_choice_prompts(TextCompletionRequest& completion) {
    const auto& request = completion.request;
    if (request.choices == 1) {
        return;
    }
    if (request.options.incremental_kv) {
        throw HttpError("400 Bad Request", "Multiple chat choices require an explicit budget");
    }
    const auto tokens = request.prompt.size() + request.options.max_new_tokens;
    if (tokens > checkpoint_context / request.choices) {
        throw HttpError("400 Bad Request", "Choice budgets must total at most 262144 tokens");
    }
    completion.batch.reserve(request.choices);
    for (std::size_t index = 0; index < request.choices; ++index) {
        auto options = request.options;
        if (options.samples() && options.seed && index) {
            constexpr std::int64_t stride = 0x1d1b54a32d192ed0LL;
            *options.seed ^= stride * static_cast<std::int64_t>(index);
        }
        completion.batch.push_back({request.prompt, std::move(options)});
    }
}

bool emit_batch_choice(BatchTextChoice& choice, const TextRequest& request,
                       const DecoderResult& result, std::size_t index, CompletionUsage& usage,
                       std::size_t& last_choice) {
    auto final = choice.response.finish_choice(result, choice.count);
    if (!final)
        return false;
    usage.output += choice.count;
    if (index == 0 || request.choices == 1)
        usage.cached += result.cached_prompt_tokens;
    last_choice = index;
    if (request.stream)
        return choice.response.write_choice(*final);
    choice.final = std::move(*final);
    return true;
}

void write_batch_choices(const Socket& connection, const std::deque<BatchTextChoice>& choices,
                         const CompletionUsage& usage) {
    auto output = choices.front().response.prefix() + R"("choices":[)";
    for (const auto& choice : choices) {
        if (&choice != &choices.front())
            output += ',';
        output += choice.final;
    }
    output += ']';
    append_usage(output, usage);
    write_json(connection, "200 OK", output + '}');
}

}

void complete_text(const Socket& connection, const TextRequest& request,
                   const CompletionTokenizer& tokenizer, BatchService& service, int slot,
                   VisionCompletion* vision) {
    if (disconnected(connection)) {
        return;
    }
    EventStream stream(connection);
    TextResponse response(connection, stream, tokenizer, request, request.prompt);
    const bool background = request.responses && request.responses->background;
    if (background && !response.queue_responses()) return;
    const auto cancelled = [&] { return response.cancelled(); };
    DecoderCallbacks callbacks{
        [&](int token) { return response.token(token); },
        cancelled,
        [&] { return response.start(); },
        [&] { return background ? !cancelled() : response.poll(); },
    };
    if (request.options.generated_top_logprobs >= 0) {
        callbacks.on_scored_token = [&](int token, const ScoreRow& score) {
            return response.token(token, &score);
        };
    }
    if (request.anthropic && request.stream) {
        callbacks.on_admitted = {};
        callbacks.on_prompt_admitted = [&](std::size_t cached) {
            return response.admit_messages(cached);
        };
    }
    try {
        std::optional<DecoderOptions> options;
        if (!request.stop_text.empty()) {
            options = request.options;
            options->text_stop = response.text_stop();
        }
        const auto& effective = options ? *options : request.options;
        const auto result =
            request.media ? vision->generate(service, slot, *request.media, effective, callbacks)
                          : service.generate(slot, request.prompt, effective, callbacks);
        if (result.finish_reason != DecoderFinishReason::cancelled && !cancelled()) {
            response.finish(result, result.output_token_ids.size());
        }
    } catch (const std::exception& error) {
        if (request.responses) {
            if (!background && !stream.started() &&
                dynamic_cast<const std::invalid_argument*>(&error)) {
                responses_error(connection, 400, error.what());
            } else {
                std::cerr << "Responses generation failed: " << error.what() << '\n';
                response.fail_responses();
            }
            return;
        }
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

void complete_openai(const Socket& connection, TextCompletionRequest completion,
                     const CompletionTokenizer& tokenizer, BatchService& service, int slot,
                     VisionCompletion* vision) {
    const auto& request = completion.request;
    const bool single = completion.batch.empty() && request.choices == 1;
    if (single && (request.chat || request.options.generated_top_logprobs < 0)) {
        complete_text(connection, request, tokenizer, service, slot, vision);
        return;
    }
    if (disconnected(connection))
        return;
    if (single) {
        completion.batch.push_back({request.prompt, request.options});
    } else {
        prepare_choice_prompts(completion);
    }
    EventStream stream(connection);
    std::deque<BatchTextChoice> choices;
    CompletionUsage usage{};
    const auto created = std::time(nullptr);
    for (auto& prompt : completion.batch) {
        const auto index = choices.size();
        const auto* echo = completion.batch_echoes.empty() ? nullptr :
            &completion.batch_echoes[index];
        choices.emplace_back(connection, stream, tokenizer, request, prompt, index, echo, created);
        if (index == 0 || request.choices == 1)
            usage.prompt += prompt.prompt.size();
    }
    std::size_t last_choice = 0;
    const BatchCallbacks callbacks{
        [&] { return disconnected(connection); },
        [&] { return !request.stream || stream.start(); },
        [&] { return !request.stream || stream.poll(); },
        [&](std::size_t index, const DecoderResult& result, bool terminal) {
            auto& choice = choices[index];
            return choice.consume(result) &&
                   (!terminal || single ||
                    emit_batch_choice(choice, request, result, index, usage, last_choice));
        },
    };
    try {
        const auto results = service.generate_batch(slot, completion.batch, callbacks);
        if (!results || disconnected(connection))
            return;
        if (single) {
            choices.front().response.finish(results->front(), choices.front().count);
            return;
        }
        if (request.stream) {
            finish_completion_stream(stream, choices[last_choice].response.prefix(),
                                     request.include_usage ? std::optional{usage} : std::nullopt);
            return;
        }
        write_batch_choices(connection, choices, usage);
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

void complete_chat_request(const Socket& connection, BatchService& service, int slot,
                           const CompletionTokenizer& tokenizer, std::string_view body,
                           VisionCompletion* vision, bool anthropic) {
    service.rethrow_failure();
    if (disconnected(connection)) {
        return;
    }
    auto request =
        anthropic ? parse_chat_request(
                        anthropic_messages_chat(json::parse(body, json::Syntax::python_request),
                                                tokenizer.chat_parsers()),
                        tokenizer, vision != nullptr)
                  : parse_chat_request(body, tokenizer, vision != nullptr);
    request.anthropic = anthropic;
    if (anthropic) request.chat_output.complete_truncated_calls = false;
    if (request.choices == 1) {
        complete_text(connection, request, tokenizer, service, slot, vision);
    } else {
        complete_openai(connection, {std::move(request), {}, {}}, tokenizer, service, slot, vision);
    }
}
}
