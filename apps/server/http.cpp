#include "sglang/server/http.h"

#include "sglang/json.h"
#include "sglang/server/completions.h"

#include <algorithm>
#include <arpa/inet.h>
#include <atomic>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <list>
#include <netinet/tcp.h>
#include <optional>
#include <poll.h>
#include <sys/socket.h>
#include <system_error>
#include <unistd.h>
#include <utility>
#include <vector>

namespace sglang::server {
namespace {

using Clock = std::chrono::steady_clock;

constexpr std::size_t max_header_bytes = 8192;
constexpr std::size_t max_body_bytes = 2 * 1024 * 1024;
constexpr auto io_timeout = std::chrono::seconds(5);
constexpr auto shutdown_poll = std::chrono::milliseconds(250);
constexpr auto reader_idle_grace = std::chrono::milliseconds(250);
constexpr short terminal_events = POLLHUP | POLLERR | POLLNVAL;

std::atomic<bool> stop_requested{false};
static_assert(std::atomic<bool>::is_always_lock_free);

void request_shutdown(int) {
    stop_serving();
}

[[noreturn]] void socket_error(const char* operation) {
    throw std::system_error(errno, std::generic_category(), operation);
}

bool wait_ready(const Socket& socket, short events, Clock::time_point deadline) {
    while (!stopping()) {
        const auto remaining = deadline - Clock::now();
        const auto time_left = std::chrono::ceil<std::chrono::milliseconds>(remaining);
        if (time_left.count() <= 0) {
            break;
        }
        pollfd ready{socket.get(), events, 0};
        const auto interval = std::min(time_left, shutdown_poll);
        const int count = poll(&ready, 1, static_cast<int>(interval.count()));
        if (count > 0) {
            return (ready.revents & events) && !(ready.revents & terminal_events);
        }
        if (count < 0 && errno != EINTR) {
            break;
        }
    }
    return false;
}

bool send_all(const Socket& socket, std::string_view bytes) {
    const auto deadline = Clock::now() + io_timeout;
    while (!bytes.empty() && !stopping()) {
        const auto count = send(socket.get(), bytes.data(), bytes.size(), MSG_NOSIGNAL);
        if (count > 0) {
            bytes.remove_prefix(static_cast<std::size_t>(count));
        } else if (count == 0) {
            return false;
        } else if (errno != EINTR && (errno != EAGAIN || !wait_ready(socket, POLLOUT, deadline))) {
            return false;
        }
    }
    return bytes.empty();
}

std::string_view take_line(std::string_view& headers) {
    const auto end = headers.find("\r\n");
    if (end == std::string_view::npos) {
        throw HttpError("400 Bad Request", "Expected CRLF in HTTP headers");
    }
    const auto line = headers.substr(0, end);
    headers.remove_prefix(end + 2);
    return line;
}

std::string_view trim(std::string_view text) {
    const auto begin = text.find_first_not_of(" \t");
    return begin == std::string_view::npos
        ? std::string_view{} : text.substr(begin, text.find_last_not_of(" \t") - begin + 1);
}

bool ascii_alphanumeric(unsigned char character) {
    return (character >= 'a' && character <= 'z') || (character >= 'A' && character <= 'Z') ||
           (character >= '0' && character <= '9');
}

bool ascii_case_equal(std::string_view left, std::string_view right) {
    const auto lower = [](char character) {
        return character >= 'A' && character <= 'Z' ? character + ('a' - 'A') : character;
    };
    return std::equal(left.begin(), left.end(), right.begin(), right.end(),
                      [&](char first, char second) { return lower(first) == lower(second); });
}

bool token_character(unsigned char character) {
    return ascii_alphanumeric(character) ||
           std::string_view("!#$%&'*+-.^_`|~").find(character) != std::string_view::npos;
}

HttpRequest parse_request_line(std::string_view line) {
    const auto first = line.find(' ');
    const auto last = line.rfind(' ');
    const auto method = line.substr(0, first);
    if (first == last || method.empty() || line.substr(last + 1) != "HTTP/1.1" ||
        !std::all_of(method.begin(), method.end(), token_character)) {
        throw HttpError("400 Bad Request", "Expected an HTTP/1.1 request line");
    }
    const auto path = line.substr(first + 1, last - first - 1);
    if (path.empty() || path.front() != '/' ||
        std::any_of(path.begin(), path.end(), [](unsigned char character) {
            return character <= 32 || character >= 127;
        })) {
        throw HttpError("400 Bad Request", "Expected an HTTP origin-form request target");
    }
    return {std::string(method), std::string(path), {}};
}

std::size_t content_length(std::string_view headers, const HttpRequest& request) {
    std::optional<std::size_t> length;
    std::optional<std::string_view> content_type;
    bool host = false;
    while (!headers.empty()) {
        const auto line = take_line(headers);
        const auto colon = line.find(':');
        const auto name = line.substr(0, colon);
        if (colon == std::string_view::npos || name.empty() ||
            !std::all_of(name.begin(), name.end(), token_character)) {
            throw HttpError("400 Bad Request", "Invalid HTTP header");
        }
        const auto value = trim(line.substr(colon + 1));
        if (std::any_of(value.begin(), value.end(), [](unsigned char character) {
            return (character < 32 && character != '\t') || character == 127;
        })) {
            throw HttpError("400 Bad Request", "Invalid HTTP header value");
        }
        if (ascii_case_equal(name, "Host")) {
            if (host || value.empty()) {
                throw HttpError("400 Bad Request", "Expected one nonempty Host header");
            }
            host = true;
        } else if (ascii_case_equal(name, "Content-Length")) {
            if (length || value.empty()) {
                throw HttpError("400 Bad Request", "Invalid Content-Length");
            }
            std::size_t size = 0;
            const auto parsed = std::from_chars(value.data(), value.data() + value.size(), size);
            if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size()) {
                throw HttpError("400 Bad Request", "Invalid Content-Length");
            }
            length = size;
        } else if (ascii_case_equal(name, "Content-Type")) {
            if (content_type) {
                throw HttpError("400 Bad Request", "Expected one Content-Type header");
            }
            content_type = value;
        } else if (ascii_case_equal(name, "Transfer-Encoding") ||
                   ascii_case_equal(name, "Content-Encoding") || ascii_case_equal(name, "Expect")) {
            throw HttpError("400 Bad Request", "Unsupported HTTP request encoding or expectation");
        }
    }
    if (!host) {
        throw HttpError("400 Bad Request", "Missing Host header");
    }
    if (length.value_or(0) > max_body_bytes) {
        throw HttpError("413 Content Too Large", "Request body exceeds 2 MiB");
    }
    if (request.method == "POST" &&
        (request.path == "/v1/completions" || request.path == "/generate")) {
        if (!length) {
            throw HttpError("411 Length Required", "Expected Content-Length");
        }
        if (!content_type || !ascii_case_equal(*content_type, "application/json")) {
            throw HttpError("415 Unsupported Media Type", "Expected application/json");
        }
    }
    if (tokenization_route(request) != TokenizationRoute::none &&
        (!content_type ||
         !ascii_case_equal(content_type->substr(0, content_type->find(';')), "application/json"))) {
        throw HttpError("400 Bad Request",
                        "Unsupported Media Type: Only 'application/json' is allowed");
    }
    return length.value_or(0);
}

struct IncomingRequest {
    explicit IncomingRequest(Socket connection) : socket(std::move(connection)) {}

    Socket socket;
    Clock::time_point last_read = Clock::now();
    const Clock::time_point deadline = last_read + io_timeout;
    std::optional<HttpRequest> request;

    Clock::time_point replace_after() const { return last_read + reader_idle_grace; }

    bool read_chunk();
    bool complete() const { return request && request->body.size() == length; }

private:
    std::string headers;
    std::size_t length = 0;
};

bool IncomingRequest::read_chunk() {
    char buffer[16 * 1024];
    auto& bytes = request ? request->body : headers;
    const auto limit = request ? length : max_header_bytes;
    const auto count = recv(socket.get(), buffer,
                            std::min(sizeof(buffer), limit - bytes.size()), 0);
    if (count <= 0) {
        return count < 0 && (errno == EAGAIN || errno == EINTR);
    }
    last_read = Clock::now();
    bytes.append(buffer, static_cast<std::size_t>(count));
    if (!request) {
        const auto end = headers.find("\r\n\r\n");
        if (end == std::string::npos) {
            if (headers.size() == max_header_bytes) {
                throw HttpError("431 Request Header Fields Too Large", "HTTP headers exceed 8 KiB");
            }
            return true;
        }
        std::string_view fields(headers.data(), end + 2);
        request.emplace(parse_request_line(take_line(fields)));
        length = content_length(fields, *request);
        request->body = headers.substr(end + 4, length);
        headers.clear();
    }
    return true;
}

bool read_pending(IncomingRequest& incoming, short events,
                  const std::function<void(Socket&, HttpRequest)>& handle) {
    if (stopping() || incoming.deadline <= Clock::now() || (events & terminal_events)) {
        return false;
    }
    if (!(events & POLLIN)) {
        return true;
    }
    try {
        if (!incoming.read_chunk()) {
            return false;
        }
        if (!incoming.complete()) {
            return true;
        }
        handle(incoming.socket, std::move(*incoming.request));
    } catch (const HttpError& error) {
        if (incoming.request && tokenization_route(*incoming.request) != TokenizationRoute::none) {
            write_tokenization_error(incoming.socket, error.status, error.status.substr(4),
                                     error.what());
        } else {
            write_error(incoming.socket, error.status, error.what());
        }
    }
    return false;
}

}

Socket::~Socket() {
    if (descriptor_ >= 0) {
        close(descriptor_);
    }
}

Socket::Socket(Socket&& other) noexcept : descriptor_(std::exchange(other.descriptor_, -1)) {}

void install_shutdown_handlers() {
    struct sigaction action {};
    action.sa_handler = request_shutdown;
    if (sigemptyset(&action.sa_mask) < 0 || sigaction(SIGINT, &action, nullptr) < 0 ||
        sigaction(SIGTERM, &action, nullptr) < 0) {
        socket_error("Install shutdown handlers");
    }
}

void stop_serving() { stop_requested.store(true, std::memory_order_relaxed); }

bool stopping() { return stop_requested.load(std::memory_order_relaxed); }

TokenizationRoute tokenization_route(const HttpRequest& request) {
    if (request.method == "POST") {
        if (request.path == "/tokenize" || request.path == "/v1/tokenize") {
            return TokenizationRoute::tokenize;
        }
        if (request.path == "/detokenize" || request.path == "/v1/detokenize") {
            return TokenizationRoute::detokenize;
        }
    }
    return TokenizationRoute::none;
}

Socket listen_loopback(int port) {
    Socket listener(socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC | SOCK_NONBLOCK, 0));
    const int enable = 1;
    if (listener.get() < 0 ||
        setsockopt(listener.get(), SOL_SOCKET, SO_REUSEADDR, &enable, sizeof(enable)) < 0) {
        socket_error("Create loopback listener");
    }
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(static_cast<std::uint16_t>(port));
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (bind(listener.get(), reinterpret_cast<sockaddr*>(&address), sizeof(address)) < 0 ||
        listen(listener.get(), 16) < 0) {
        socket_error("Listen on 127.0.0.1");
    }
    return listener;
}

static Socket accept_connection(const Socket& listener) {
    Socket client(accept4(listener.get(), nullptr, nullptr, SOCK_CLOEXEC | SOCK_NONBLOCK));
    if (client.get() < 0) {
        if (errno == EINTR || errno == EAGAIN || errno == ECONNABORTED) {
            return client;
        }
        socket_error("Accept connection");
    }
    const int enable = 1;
    if (setsockopt(client.get(), IPPROTO_TCP, TCP_NODELAY, &enable, sizeof(enable)) < 0) {
        return Socket(-1);
    }
    return client;
}

bool disconnected(const Socket& connection) {
    if (stopping()) {
        return true;
    }
    pollfd ready{connection.get(), 0, 0};
    const int count = poll(&ready, 1, 0);
    return (count < 0 && errno != EINTR) || (ready.revents & terminal_events);
}

static void accept_next(const Socket& listener, std::list<IncomingRequest>& incoming,
                        std::size_t limit) {
    auto replace = incoming.end();
    if (incoming.size() == limit) {
        const auto now = Clock::now();
        replace = std::find_if(incoming.begin(), incoming.end(), [&](const auto& request) {
            if (request.replace_after() > now) {
                return false;
            }
            pollfd ready{request.socket.get(), POLLIN, 0};
            return poll(&ready, 1, 0) == 0;
        });
        if (replace == incoming.end()) {
            return;
        }
    }
    auto connection = accept_connection(listener);
    if (connection.get() >= 0) {
        if (replace != incoming.end()) {
            incoming.erase(replace);
        }
        incoming.emplace_back(std::move(connection));
    }
}

void read_connections(const Socket& listener, int capacity,
                      const std::function<void(Socket&, HttpRequest)>& handle) {
    const auto limit = static_cast<std::size_t>(capacity);
    std::list<IncomingRequest> incoming;
    std::vector<pollfd> sockets;
    sockets.reserve(limit + 1);
    while (!stopping()) {
        const bool full = incoming.size() == limit;
        const auto now = Clock::now();
        short accept_events = full ? 0 : POLLIN;
        auto deadline = now + shutdown_poll;
        sockets.clear();
        sockets.push_back({listener.get(), 0, 0});
        for (const auto& request : incoming) {
            sockets.push_back({request.socket.get(), POLLIN, 0});
            deadline = std::min(deadline, request.deadline);
            if (full) {
                const auto replace = request.replace_after();
                if (replace <= now) {
                    accept_events = POLLIN;
                } else {
                    deadline = std::min(deadline, replace);
                }
            }
        }
        sockets.front().events = accept_events;
        const auto remaining =
            std::chrono::ceil<std::chrono::milliseconds>(deadline - Clock::now());
        const int timeout = std::max(0, static_cast<int>(remaining.count()));
        if (poll(sockets.data(), sockets.size(), timeout) < 0) {
            if (errno == EINTR) {
                continue;
            }
            socket_error("Poll HTTP connections");
        }
        if (sockets.front().revents & terminal_events) {
            throw std::runtime_error("HTTP listener closed unexpectedly");
        }
        std::size_t index = 1;
        for (auto it = incoming.begin(); it != incoming.end();) {
            if (read_pending(*it, sockets[index++].revents, handle)) {
                ++it;
            } else {
                it = incoming.erase(it);
            }
        }
        if (!stopping() && (sockets.front().revents & POLLIN)) {
            // A readable socket may still have bytes left after its bounded read this turn.
            accept_next(listener, incoming, limit);
        }
    }
}

static bool write_response(const Socket& connection, std::string_view status,
                           std::string_view content_type, std::string_view body,
                           std::string_view header = std::string_view{}, bool head_only = false) {
    std::string response = "HTTP/1.1 ";
    response += status;
    response += "\r\nConnection: close\r\nContent-Type: ";
    response += content_type;
    response += header;
    response += "\r\nContent-Length: ";
    response += std::to_string(body.size());
    response += "\r\n\r\n";
    if (!head_only) {
        response += body;
    }
    return send_all(connection, response);
}

bool write_json(const Socket& connection, std::string_view status, std::string_view json) {
    return write_response(connection, status, "application/json", json);
}

void write_get_only(const Socket& connection, bool head_only) {
    write_response(connection, "405 Method Not Allowed", "application/json",
                   R"({"detail":"Method Not Allowed"})", "\r\nAllow: GET", head_only);
}

void write_tokenization_error(const Socket& connection, std::string_view status,
                              std::string_view type, std::string_view message) {
    std::string body = R"({"object":"error","message":)";
    json::append_quoted(body, message);
    body += R"(,"type":)";
    json::append_quoted(body, type);
    body += R"(,"param":null,"code":)";
    body += status.substr(0, 3);
    body += '}';
    write_json(connection, status, body);
}

void write_readiness(const Socket& connection, bool ready) {
    if (ready) {
        write_json(connection, "200 OK", R"({"ready":true})");
    } else {
        write_json(connection, "503 Service Unavailable",
                   R"({"ready":false,"reason":"worker_failed"})");
    }
}

bool write_metrics(const Socket& connection, std::string_view body) {
    return write_response(connection, "200 OK", "text/plain; version=0.0.4", body);
}

bool EventStream::send(std::string_view bytes) {
    if (!send_all(connection_, bytes)) {
        return false;
    }
    last_write_ = std::chrono::steady_clock::now();
    return true;
}

bool EventStream::start() {
    return started() ||
        send("HTTP/1.1 200 OK\r\nConnection: close\r\nContent-Type: text/event-stream\r\n"
             "Cache-Control: no-cache\r\nTransfer-Encoding: chunked\r\n\r\n");
}

bool EventStream::poll() {
    return !last_write_ ||
        std::chrono::steady_clock::now() - *last_write_ < std::chrono::seconds(15) ||
        send("3\r\n:\n\n\r\n");
}

bool EventStream::write(std::string_view data) {
    char size[2 * sizeof(std::size_t)];
    const auto encoded = std::to_chars(size, size + sizeof(size), data.size() + 8, 16);
    std::string chunk(size, encoded.ptr);
    chunk += "\r\ndata: ";
    chunk += data;
    chunk += "\n\n\r\n";
    return send(chunk);
}

bool EventStream::end() {
    return send("0\r\n\r\n");
}

}
