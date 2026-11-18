#pragma once

#include "sglang/decoder.h"
#include "sglang/server/completions.h"

namespace sglang::server {

struct TextRequest {
    std::vector<int> prompt;
    DecoderOptions options;
    bool stream = false;
    bool include_usage = false;
    bool return_token_ids = false;
};

using TextGenerator = std::function<DecoderResult(const std::vector<int>&, const DecoderOptions&,
                                                   const DecoderCallbacks&)>;

TextRequest parse_text_request(std::string_view body, const CompletionTokenizer& tokenizer);
void complete_text(const Socket& connection, std::string_view body,
                   const CompletionTokenizer& tokenizer, const TextGenerator& generate,
                   bool trace_serial = false);

}
