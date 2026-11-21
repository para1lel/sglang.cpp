#pragma once

#include <functional>
#include <optional>
#include <string_view>

namespace sglang::server {

class BatchService;
class Socket;
enum class TokenizationRoute;

using BatchCompletion = std::function<void(const Socket&, int, std::string_view, bool)>;
struct QueuedGenerate {
    std::function<void(const Socket&, std::optional<int>)> complete;
    std::function<bool()> aborted;
};
using BatchGenerate = std::function<QueuedGenerate(std::string_view)>;
using NativeAbort = std::function<void(const Socket&, std::string_view)>;
using Tokenization = std::function<void(const Socket&, TokenizationRoute, std::string_view)>;

void serve_connections(const Socket& listener, BatchService& service, int max_active,
                       const BatchCompletion& complete, std::string_view health,
                       bool prefix_controls, const BatchGenerate& generate = BatchGenerate{},
                       const Tokenization& tokenize = Tokenization{},
                       std::string_view model_info = std::string_view{},
                       const NativeAbort& abort = NativeAbort{});

}
