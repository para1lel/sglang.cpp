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

    std::string_view route_path() const;
};

std::string_view matchable_route_path(std::string_view path);
inline constexpr std::string_view separate_reasoning_route = "/separate_reasoning";

enum class GenerateRoute { none, allowed, method_not_allowed };
enum class TokenizationRoute { none, tokenize, detokenize };

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
void read_connections(const Socket& listener, int capacity,
                      const std::function<void(Socket&, HttpRequest)>& handle);
bool write_json(const Socket& connection, std::string_view status, std::string_view json);
void write_raw_response(const Socket& connection, std::string_view status, std::string_view body);
void write_method_not_allowed(const Socket& connection, std::string_view methods, bool head_only);
void write_redirect(const Socket& connection, const HttpRequest& request, std::string_view path);
void write_get_only(const Socket& connection, bool head_only);
void write_tokenization_error(const Socket& connection, std::string_view status,
                              std::string_view type, std::string_view message);
void write_internal_server_error(const Socket& connection);
void write_readiness(const Socket& connection, bool ready);
void write_generation_health(const Socket& connection, bool healthy);
bool write_metrics(const Socket& connection, std::string_view body);

class EventStream {
public:
    explicit EventStream(const Socket& connection) : connection_(connection) {}

    bool started() const { return last_write_.has_value(); }
    bool start();
    bool poll();
    bool write(std::string_view data);
    bool end();

private:
    bool send(std::string_view bytes);

    const Socket& connection_;
    std::optional<std::chrono::steady_clock::time_point> last_write_;
};

}
