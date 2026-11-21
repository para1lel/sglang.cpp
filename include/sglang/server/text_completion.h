#pragma once

#include "sglang/decoder.h"
#include "sglang/json.h"
#include "sglang/server/batch_service.h"
#include "sglang/server/chat_media.h"
#include "sglang/server/chat_output.h"
#include "sglang/server/completions.h"

namespace sglang::server {

struct TextRequest {
    std::vector<int> prompt;
    DecoderOptions options;
    bool stream = false;
    bool include_usage = false;
    bool return_token_ids = false;
    bool chat = false;
    std::vector<std::string> stop_text;
    std::optional<MediaPrompt> media;
    ChatOutputOptions chat_output;
};

struct TextCompletionRequest {
    TextRequest request;
    std::vector<BatchPrompt> batch;
};

using TextGenerator = std::function<DecoderResult(const std::vector<int>&, const DecoderOptions&,
                                                   const DecoderCallbacks&)>;
using MediaGenerator = std::function<DecoderResult(const MediaPrompt&, const DecoderOptions&,
                                                    const DecoderCallbacks&)>;

TextCompletionRequest parse_text_request(const json::Object& body,
                                         const CompletionTokenizer& tokenizer);
TextRequest parse_chat_request(std::string_view body, const CompletionTokenizer& tokenizer,
                               bool media_enabled = false);
void set_cache_salt(const json::Object& object, DecoderOptions& options);
void append_ids(std::string& output, const int* ids, std::size_t count);
void complete_text(const Socket& connection, const TextRequest& request,
                   const CompletionTokenizer& tokenizer, const TextGenerator& generate);
void complete_scored_text(const Socket& connection, const TextRequest& request,
                          const CompletionTokenizer& tokenizer, BatchService& service, int slot);
void complete_text_batch(const Socket& connection, TextCompletionRequest completion,
                         const CompletionTokenizer& tokenizer, BatchService& service, int slot);
void complete_chat_request(const Socket& connection, std::string_view body,
                           const CompletionTokenizer& tokenizer, const TextGenerator& generate,
                           const MediaGenerator& generate_media, BatchService& service, int slot);

}
