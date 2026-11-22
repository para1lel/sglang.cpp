#pragma once

#include <optional>
#include <regex>
#include <string>
#include <string_view>

namespace sglang::server {

class Socket;
class NativeGenerate;
struct HttpRequest;
struct QueuedGenerate;

class OllamaDiscovery {
public:
    enum class Route { unmatched, responded, generate };

    explicit OllamaDiscovery(NativeGenerate& native_generate);
    bool overrides_count(const HttpRequest& request) const;
    Route respond_exact(const Socket& connection, const HttpRequest& request,
                        bool later_full, std::string_view earlier_allow) const;
    bool respond_unmatched(const Socket& connection, const HttpRequest& request) const;
    QueuedGenerate prepare_generate(const Socket& connection, const HttpRequest& request) const;

private:
    struct Path {
        explicit Path(std::string_view path);
        bool matches(std::string_view path) const;

        std::string literal;
        std::optional<std::regex> pattern;
    };

    bool matches(std::string_view path) const;
    static bool matches(const Path& route, const HttpRequest& request);
    NativeGenerate& native_generate_;
    bool ollama_root_;
    Path root_;
    Path generate_;
    Path tags_;
    Path show_;
};

}
