#pragma once

#include "sglang/decoder.h"

#include <string_view>

namespace sglang::server {

class BatchService;
class CompletionTokenizer;
class Socket;

struct GenerateRequest {
    std::vector<int> prompt;
    DecoderOptions options;
    bool stream = false;
};

GenerateRequest parse_generate_request(std::string_view body,
                                       const CompletionTokenizer& tokenizer);
void complete_generate(const Socket& connection, BatchService& service, int slot,
                       const CompletionTokenizer& tokenizer, std::string_view body);

}
