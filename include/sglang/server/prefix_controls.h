#pragma once

#include <functional>
#include <string_view>

namespace sglang::server {

class BatchService;
class Socket;

using BatchCompletion = std::function<void(const Socket&, int, std::string_view, bool)>;

void serve_prefix_connections(const Socket& listener, BatchService& service, int max_active,
                              const BatchCompletion& complete);

}
