#pragma once

#include <functional>
#include <optional>
#include <string>
#include <string_view>

namespace sglang::server {

class BatchService;
class NativeGenerate;
class OllamaDiscovery;
class ResponseStore;
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
using ServerInfo = std::function<std::string()>;
using BatchScore = std::function<void(const Socket&, int, std::string_view)>;
using BatchMessages = BatchScore;
using BatchResponses = BatchScore;
using BatchDecisions = std::function<void(const Socket&, int, std::string_view, bool)>;

void serve_connections(const Socket& listener, BatchService& service, int max_active,
                       const BatchCompletion& complete, std::string_view health,
                       bool prefix_controls, const BatchGenerate& generate = BatchGenerate{},
                       const Tokenization& tokenize = Tokenization{},
                       std::string_view model_info = std::string_view{},
                       const NativeAbort& abort = NativeAbort{},
                       const ServerInfo& server_info = ServerInfo{},
                       const BatchScore& score = BatchScore{},
                       const OllamaDiscovery* discovery = nullptr,
                       const BatchMessages& messages = BatchMessages{},
                       const BatchResponses& responses = BatchResponses{},
                       ResponseStore* response_store = nullptr,
                       const BatchDecisions& decisions = BatchDecisions{},
                       std::string_view accelerator = std::string_view{},
                       NativeGenerate* session_control = nullptr);

}
