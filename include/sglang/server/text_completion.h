#pragma once

#include "sglang/decoder.h"
#include "sglang/json.h"
#include "sglang/server/chat_media.h"
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
};

using TextGenerator = std::function<DecoderResult(const std::vector<int>&, const DecoderOptions&,
                                                   const DecoderCallbacks&)>;
using MediaGenerator = std::function<DecoderResult(const MediaPrompt&, const DecoderOptions&,
                                                    const DecoderCallbacks&)>;

TextRequest parse_text_request(const json::Object& body, const CompletionTokenizer& tokenizer);
TextRequest parse_chat_request(std::string_view body, const CompletionTokenizer& tokenizer,
                               bool media_enabled = false);
void set_cache_salt(const json::Object& object, DecoderOptions& options);
void append_ids(std::string& output, const int* ids, std::size_t count);
void complete_text(const Socket& connection, const TextRequest& request,
                   const CompletionTokenizer& tokenizer, const TextGenerator& generate,
                   bool trace_serial);
void complete_chat_request(const Socket& connection, std::string_view body,
                           const CompletionTokenizer& tokenizer, const TextGenerator& generate,
                           const MediaGenerator& generate_media, bool trace_serial);

}
