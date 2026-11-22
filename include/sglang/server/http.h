#pragma once

#include <chrono>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

namespace sglang::server {

class Socket {
public:
    explicit Socket(int descriptor) : descriptor_(descriptor) {}
    ~Socket();
    Socket(const Socket&) = delete;
    Socket& operator=(const Socket&) = delete;
    Socket(Socket&& other) noexcept;

    int get() const { return descriptor_; }

private:
    int descriptor_;
};

struct HttpRequest {
    std::string method;
    std::string path;
    std::string body;
    std::string host;
    std::string query;
    std::optional<bool> body_is_json = std::nullopt;
    bool body_is_application_json = false;

    std::string_view route_path() const;
};

std::string_view matchable_route_path(std::string_view path);
inline constexpr std::string_view separate_reasoning_route = "/separate_reasoning";
inline constexpr std::string_view parse_function_call_route = "/parse_function_call";
inline constexpr std::string_view anthropic_count_route = "/v1/messages/count_tokens";
inline constexpr std::string_view anthropic_messages_route = "/v1/messages";
inline constexpr std::string_view responses_route = "/v1/responses";
inline constexpr std::string_view score_route = "/v1/score";
inline constexpr std::string_view decisions_route = "/v1/decisions";

enum class GenerateRoute { none, allowed, method_not_allowed };
enum class TokenizationRoute { none, tokenize, detokenize, anthropic_count };

struct HttpError : std::runtime_error {
    HttpError(std::string_view status, const char* message)
        : std::runtime_error(message), status(status) {}

    std::string_view status;
};

void install_shutdown_handlers();
void stop_serving();
bool stopping();
Socket listen_loopback(int port);
bool disconnected(const Socket& connection);
std::string decode_url(std::string_view text, bool form = false);
GenerateRoute generate_route(const HttpRequest& request);
TokenizationRoute tokenization_route(const HttpRequest& request);
using AnthropicCountOverride = std::function<bool(const HttpRequest&)>;
void read_connections(const Socket& listener, int capacity,
                      const std::function<void(Socket&, HttpRequest)>& handle,
                      const AnthropicCountOverride& overrides_count);
bool write_json(const Socket& connection, std::string_view status, std::string_view json,
                bool head_only = false);
void write_raw_response(const Socket& connection, std::string_view status, std::string_view body);
void write_method_not_allowed(const Socket& connection, std::string_view methods, bool head_only);
void write_redirect(const Socket& connection, const HttpRequest& request, std::string_view path);
void write_get_only(const Socket& connection, bool head_only);
void write_tokenization_error(const Socket& connection, std::string_view status,
                              std::string_view type, std::string_view message);
void write_anthropic_error(const Socket& connection, std::string_view status,
                           std::string_view type, std::string_view message);
std::string anthropic_error_body(std::string_view type, std::string_view message);
void write_internal_server_error(const Socket& connection);
void write_readiness(const Socket& connection, bool ready);
void write_generation_health(const Socket& connection, bool healthy);
bool write_metrics(const Socket& connection, std::string_view body);

class EventStream {
public:
    enum class Format { sse, ndjson };

    explicit EventStream(const Socket& connection, Format format = Format::sse)
        : connection_(connection), format_(format) {}

    bool started() const { return last_write_.has_value(); }
    bool start();
    bool poll();
    bool write(std::string_view data, std::string_view event = std::string_view{});
    bool end();

private:
    bool send(std::string_view bytes);

    const Socket& connection_;
    const Format format_;
    std::optional<std::chrono::steady_clock::time_point> last_write_;
};

}
