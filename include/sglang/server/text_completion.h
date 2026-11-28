#pragma once

#include "sglang/decoder.h"
#include "sglang/json.h"
#include "sglang/server/batch_service.h"
#include "sglang/server/chat_media.h"
#include "sglang/server/chat_output.h"
#include "sglang/server/completions.h"

namespace sglang::server {

struct ResponsesRequest;
class EventStream;

struct TextRequest {
    std::vector<int> prompt;
    DecoderOptions options;
    std::size_t choices = 1;
    bool stream = false;
    bool include_usage = false;
    bool return_token_ids = false;
    bool chat = false;
    bool anthropic = false;
    const ResponsesRequest* responses = nullptr;
    std::vector<std::string> stop_text;
    std::optional<MediaPrompt> media;
    ChatOutputOptions chat_output;
    std::optional<std::string> echo;
};

struct TextCompletionRequest {
    TextRequest request;
    std::vector<BatchPrompt> batch;
    std::vector<std::string> batch_echoes;
};

struct CompletionUsage {
    std::size_t prompt;
    std::size_t output;
    std::size_t cached;
};

TextCompletionRequest parse_text_request(const json::Object& body,
                                         const CompletionTokenizer& tokenizer);
TextRequest parse_chat_request(const json::OrderedObject& chat,
                               const CompletionTokenizer& tokenizer,
                               bool media_enabled = false);
TextRequest parse_chat_request(std::string_view body, const CompletionTokenizer& tokenizer,
                               bool media_enabled = false);
void set_cache_salt(const json::Object& object, DecoderOptions& options);
std::string completion_echo(const json::Value& input, const std::vector<int>& ids,
                            const Tokenizer& tokenizer);
void append_ids(std::string& output, const int* ids, std::size_t count);
std::optional<int> logprob_width(const json::Value* value);
void append_score(std::string& output, float value);
void check_generated_scores(const DecoderResult& result);
const std::shared_ptr<const DecoderTokenLogprobs>& generated_score(const DecoderResult& result,
                                                                   std::size_t index);
std::vector<std::pair<std::string, float>>
distinct_score_alternatives(const Tokenizer& tokenizer, const DecoderTokenLogprobs& row,
                            std::string (*display)(const Tokenizer&, int));
bool write_completion_event(EventStream& stream, std::string_view type, json::OrderedObject fields,
                            int* sequence = nullptr);
const char* completion_reason(const DecoderOptions& options, const DecoderResult& result,
                              bool matched_text_stop, bool constraint_budget_is_length = false);
void completion_failure(EventStream& stream, const std::exception& error);
void finish_completion_stream(EventStream& stream, std::string_view prefix = "{",
                              std::optional<CompletionUsage> usage = std::nullopt,
                              bool include_total = true);
void complete_text(const Socket& connection, const TextRequest& request,
                   const CompletionTokenizer& tokenizer, BatchService& service, int slot,
                   VisionCompletion* vision = nullptr);
void complete_openai(const Socket& connection, TextCompletionRequest completion,
                     const CompletionTokenizer& tokenizer, BatchService& service, int slot,
                     VisionCompletion* vision = nullptr);
void complete_chat_request(const Socket& connection, BatchService& service, int slot,
                           const CompletionTokenizer& tokenizer, std::string_view body,
                           VisionCompletion* vision, bool anthropic = false);
}
