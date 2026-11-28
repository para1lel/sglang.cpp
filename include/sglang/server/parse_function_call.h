#pragma once

#include <string_view>

namespace sglang::server {

class Socket;

void parse_function_call(const Socket& connection, std::string_view body);
void separate_reasoning(const Socket& connection, std::string_view body);
}
