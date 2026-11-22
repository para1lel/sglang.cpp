#pragma once

#include <optional>
#include <regex>
#include <string>
#include <string_view>

namespace sglang::server {

class Socket;
struct HttpRequest;

class OllamaDiscovery {
public:
    OllamaDiscovery();
    bool matches_show(const HttpRequest& request) const;
    bool respond_exact(const Socket& connection, const HttpRequest& request,
                       bool later_full, std::string_view earlier_allow) const;
    bool respond_unmatched(const Socket& connection, const HttpRequest& request) const;

private:
    struct Path {
        explicit Path(std::string_view path);
        bool matches(std::string_view path) const;

        std::string literal;
        std::optional<std::regex> pattern;
    };

    bool matches(std::string_view path) const;
    bool ollama_root_;
    Path root_;
    Path tags_;
    Path show_;
};

}
