#pragma once

#include <string_view>

namespace sglang::server {

class CompletionTokenizer;
class Socket;
enum class TokenizationRoute;

void serve_tokenization(const Socket& connection, TokenizationRoute route,
                        const CompletionTokenizer& tokenizer, std::string_view body);

}
