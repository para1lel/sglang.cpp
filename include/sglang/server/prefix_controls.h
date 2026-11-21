#pragma once

#include <functional>
#include <string_view>

namespace sglang::server {

class BatchService;
class Socket;

using BatchCompletion = std::function<void(const Socket&, int, std::string_view, bool)>;
using BatchGenerate = std::function<void(const Socket&, int, std::string_view)>;

void serve_connections(const Socket& listener, BatchService& service, int max_active,
                       const BatchCompletion& complete, std::string_view health,
                       bool prefix_controls, const BatchGenerate& generate = BatchGenerate{});

}
