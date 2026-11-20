#pragma once

#include <functional>
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
};

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
void read_connections(const Socket& listener, int capacity,
                      const std::function<void(Socket&, HttpRequest)>& handle);
bool write_json(const Socket& connection, std::string_view status, std::string_view json);
void write_readiness(const Socket& connection, bool ready);
bool write_metrics(const Socket& connection, std::string_view body);
bool start_events(const Socket& connection);
bool write_event(const Socket& connection, std::string_view data);
bool end_events(const Socket& connection);

}
