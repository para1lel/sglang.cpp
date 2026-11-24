#include "sglang/runtime/cache/lmcache_client.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <poll.h>
#include <stdexcept>
#include <string_view>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <utility>

namespace sglang::decoder {
namespace {

using Clock = std::chrono::steady_clock;
constexpr std::string_view magic = "SLC1";
constexpr std::string_view daemon_version = "0.5.8";
constexpr std::size_t max_record = std::size_t{1} << 30;
constexpr std::size_t max_metadata = std::size_t{64} << 20;
constexpr std::size_t max_keys = std::size_t{1} << 20;
constexpr std::size_t max_error = 1024;
constexpr std::size_t error_status = 2;
constexpr std::size_t miss_status = 1;
constexpr std::size_t header_size = 16;
enum class Operation { ready = 1, lookup, get, put, reset };

[[noreturn]] void transport_error(std::string detail) {
    throw std::runtime_error("LMCache: " + std::move(detail));
}

[[noreturn]] void socket_error(std::string_view operation) {
    const int error = errno;
    transport_error(std::string(operation) + ": " + std::strerror(error));
}

void encode_number(char* output, std::size_t width, std::uint64_t value) {
    while (width != 0) {
        output[--width] = static_cast<char>(value & 0xff);
        value >>= 8;
    }
}

std::uint64_t decode_number(std::string_view input) {
    std::uint64_t value = 0;
    for (const unsigned char byte : input) value = (value << 8) | byte;
    return value;
}

void append_number(std::string& output, std::size_t width, std::uint64_t value) {
    const std::size_t offset = output.size();
    output.resize(offset + width);
    encode_number(output.data() + offset, width, value);
}

void append_key(std::string& output, const LmCacheKey& key) {
    const int family = static_cast<int>(key.family);
    if (family < 1 || family > 3 || key.identity.empty() || key.identity.size() > 65535) {
        throw std::invalid_argument("Invalid LMCache record key");
    }
    if (output.size() + 3 + key.identity.size() > max_metadata) {
        throw std::invalid_argument("LMCache lookup metadata is too large");
    }
    append_number(output, 1, family);
    append_number(output, 2, key.identity.size());
    output += key.identity;
}

void validate_record_size(std::size_t size) {
    if (size == 0 || size > max_record) {
        throw std::invalid_argument("Invalid LMCache record size");
    }
}

class Connection {
public:
    explicit Connection(const LmCacheConfig& config)
        : deadline_(Clock::now() + std::chrono::milliseconds(config.timeout_ms)) {
        sockaddr_un address{};
        address.sun_family = AF_UNIX;
        std::memcpy(address.sun_path, config.endpoint.c_str(), config.endpoint.size() + 1);
        fd_ = socket(AF_UNIX, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
        if (fd_ < 0) socket_error("socket");
        try {
            connect_to(address);
            struct ucred credentials {};
            socklen_t size = sizeof(credentials);
            if (getsockopt(fd_, SOL_SOCKET, SO_PEERCRED, &credentials, &size) != 0) {
                socket_error("peer credentials");
            }
            if (size != sizeof(credentials) || credentials.uid != geteuid()) {
                transport_error("unexpected daemon user");
            }
        } catch (...) {
            close(fd_);
            throw;
        }
    }

    ~Connection() { close(fd_); }
    Connection(const Connection&) = delete;
    Connection& operator=(const Connection&) = delete;

    void request(Operation op, std::string_view metadata, const std::byte* data = nullptr,
                 std::size_t bytes = 0) const {
        std::array<char, header_size> header{};
        std::copy(magic.begin(), magic.end(), header.begin());
        encode_number(header.data() + 4, 4, static_cast<unsigned>(op));
        encode_number(header.data() + 8, 8, metadata.size() + bytes);
        send_all(header.data(), header.size());
        send_all(metadata.data(), metadata.size());
        send_all(data, bytes);
    }

    bool response(std::size_t expected, bool allow_miss = false) const {
        std::array<char, header_size> header{};
        read_all(header.data(), header.size());
        const std::string_view frame(header.data(), header.size());
        if (frame.substr(0, 4) != magic) transport_error("invalid daemon frame");
        const auto status = decode_number(frame.substr(4, 4));
        const auto size = decode_number(frame.substr(8, 8));
        if (status == error_status && size <= max_error) {
            std::string detail(size, '\0');
            read_all(detail.data(), detail.size());
            transport_error(std::move(detail));
        }
        if (allow_miss && status == miss_status && size == 0) return false;
        if (status != 0 || size != expected) transport_error("invalid daemon response");
        return true;
    }

    void read_all(void* data, std::size_t bytes) const {
        auto* next = static_cast<char*>(data);
        while (bytes != 0) {
            check_deadline();
            const ssize_t count = recv(fd_, next, bytes, 0);
            if (count == 0) transport_error("daemon closed connection");
            if (count < 0) {
                if (errno == EINTR) continue;
                if (errno != EAGAIN && errno != EWOULDBLOCK) socket_error("receive");
                wait_for(POLLIN);
                continue;
            }
            next += count;
            bytes -= static_cast<std::size_t>(count);
        }
    }

private:
    void connect_to(const sockaddr_un& address) const {
        for (;;) {
            check_deadline();
            if (connect(fd_, reinterpret_cast<const sockaddr*>(&address), sizeof(address)) == 0) {
                return;
            }
            if (errno == EINTR) continue;
            if (errno != EINPROGRESS && errno != EALREADY) socket_error("connect");
            wait_for(POLLOUT);
            int error = 0;
            socklen_t size = sizeof(error);
            if (getsockopt(fd_, SOL_SOCKET, SO_ERROR, &error, &size) != 0) {
                socket_error("connect status");
            }
            if (error == 0) return;
            errno = error;
            socket_error("connect");
        }
    }

    void send_all(const void* data, std::size_t bytes) const {
        auto* next = static_cast<const char*>(data);
        while (bytes != 0) {
            check_deadline();
            const ssize_t count = send(fd_, next, bytes, MSG_NOSIGNAL);
            if (count == 0) transport_error("daemon stopped receiving");
            if (count < 0) {
                if (errno == EINTR) continue;
                if (errno != EAGAIN && errno != EWOULDBLOCK) socket_error("send");
                wait_for(POLLOUT);
                continue;
            }
            next += count;
            bytes -= static_cast<std::size_t>(count);
        }
    }

    void check_deadline() const {
        if (Clock::now() >= deadline_) transport_error("daemon request timed out");
    }

    void wait_for(short events) const {
        pollfd wake{fd_, events, 0};
        for (;;) {
            check_deadline();
            const auto left =
                std::chrono::ceil<std::chrono::milliseconds>(deadline_ - Clock::now());
            const int status = poll(&wake, 1, static_cast<int>(left.count()));
            if (status > 0) return;
            if (status < 0 && errno != EINTR) socket_error("poll");
        }
    }

    Clock::time_point deadline_;
    int fd_;
};

std::string request_prefix(const LmCacheConfig& config) {
    std::string metadata;
    append_number(metadata, 2, config.name_space.size());
    metadata += config.name_space;
    return metadata;
}

}  // namespace

std::string default_lmcache_endpoint() {
    const char* runtime_dir = std::getenv("XDG_RUNTIME_DIR");
    if (runtime_dir != nullptr && runtime_dir[0] == '/') {
        std::string endpoint(runtime_dir);
        if (endpoint.back() != '/') endpoint += '/';
        return endpoint + "sglang-lmcache.sock";
    }
    return "/tmp/sglang-lmcache-" + std::to_string(geteuid()) + ".sock";
}

LmCacheClient::LmCacheClient(LmCacheConfig config) : config_(std::move(config)) {
    constexpr std::string_view namespace_characters =
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-";
    if (config_.endpoint.empty() || config_.endpoint.front() != '/' ||
        config_.endpoint.find('\0') != std::string::npos ||
        config_.endpoint.size() >= sizeof(sockaddr_un::sun_path) || config_.timeout_ms <= 0 ||
        config_.name_space.empty() || config_.name_space.size() > 128 ||
        config_.name_space.find_first_not_of(namespace_characters) != std::string::npos) {
        throw std::invalid_argument("Invalid LMCache endpoint, namespace, or timeout");
    }
}

bool LmCacheClient::ready() const {
    const Connection connection(config_);
    connection.request(Operation::ready, request_prefix(config_));
    connection.response(daemon_version.size());
    std::string version(daemon_version.size(), '\0');
    connection.read_all(version.data(), version.size());
    if (version != daemon_version) transport_error("unsupported daemon version");
    return true;
}

std::vector<bool> LmCacheClient::lookup(const std::vector<LmCacheKey>& keys) const {
    if (keys.empty()) return {};
    if (keys.size() > max_keys) throw std::invalid_argument("Too many LMCache lookup keys");
    std::string metadata = request_prefix(config_);
    append_number(metadata, 4, keys.size());
    for (const auto& key : keys) append_key(metadata, key);
    const Connection connection(config_);
    connection.request(Operation::lookup, metadata);
    connection.response(keys.size());
    std::vector<unsigned char> raw(keys.size());
    connection.read_all(raw.data(), raw.size());
    std::vector<bool> found;
    found.reserve(keys.size());
    for (const auto byte : raw) {
        if (byte > 1) transport_error("invalid lookup result");
        found.push_back(byte != 0);
    }
    return found;
}

std::unique_ptr<std::byte[]> LmCacheClient::get(
    const LmCacheKey& key, std::size_t expected_bytes) const {
    validate_record_size(expected_bytes);
    auto metadata = request_prefix(config_);
    append_key(metadata, key);
    append_number(metadata, 8, expected_bytes);
    const Connection connection(config_);
    connection.request(Operation::get, metadata);
    if (!connection.response(expected_bytes, true)) return {};
    std::unique_ptr<std::byte[]> data(new std::byte[expected_bytes]);
    connection.read_all(data.get(), expected_bytes);
    return data;
}

bool LmCacheClient::put(const LmCacheKey& key, const std::byte* data, std::size_t size) const {
    validate_record_size(size);
    if (data == nullptr) throw std::invalid_argument("Missing LMCache record data");
    auto metadata = request_prefix(config_);
    append_key(metadata, key);
    const Connection connection(config_);
    connection.request(Operation::put, metadata, data, size);
    return connection.response(0, true);
}

void LmCacheClient::reset() const {
    const Connection connection(config_);
    connection.request(Operation::reset, request_prefix(config_));
    connection.response(0);
}

}  // namespace sglang::decoder
