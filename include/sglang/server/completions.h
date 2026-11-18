#pragma once

#include <string_view>

namespace sglang {

class Decoder;

namespace server {

class BatchService;
class Socket;

void complete(const Socket& connection, const Decoder& decoder, std::string_view body,
              bool trace = false);
void complete(const Socket& connection, BatchService& service, int slot, std::string_view body);
void write_error(const Socket& connection, std::string_view status, const char* message);

}
}
