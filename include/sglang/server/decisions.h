#pragma once

#include "sglang/server/completions.h"
#include "sglang/server/score.h"

namespace sglang::server {

void serve_decisions(const Socket& connection, BatchService& service, int slot,
                     const CompletionTokenizer& tokenizer, std::string_view body);
void serve_systemone(const Socket& connection, BatchService& service, int slot,
                     const CompletionTokenizer& tokenizer, std::string_view body,
                     std::string_view model);

}
