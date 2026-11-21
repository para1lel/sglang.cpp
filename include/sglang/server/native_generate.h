#pragma once

#include "sglang/server/batch_service.h"

#include <string_view>

namespace sglang::server {

class CompletionTokenizer;
class Socket;

struct GenerateRequests {
    std::vector<BatchPrompt> items;
    bool stream = false;
    bool batch = false;
};

GenerateRequests parse_generate_requests(std::string_view body,
                                         const CompletionTokenizer& tokenizer);
void complete_generate(const Socket& connection, BatchService& service, int slot,
                       const CompletionTokenizer& tokenizer, std::string_view body);

}
