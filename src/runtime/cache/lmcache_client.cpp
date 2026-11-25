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
#include <sys/syscall.h>
#include <unistd.h>
#include <utility>

namespace sglang::decoder {
namespace {

using Clock = std::chrono::steady_clock;
constexpr std::string_view magic = "SLC2";
constexpr std::string_view daemon_version = "0.5.8";
constexpr std::size_t max_record = std::size_t{1} << 30;
constexpr std::size_t max_metadata = std::size_t{64} << 20;
constexpr std::size_t max_keys = std::size_t{1} << 20;
constexpr std::size_t max_error = 1024;
constexpr std::uint64_t error_status = 2;
constexpr std::uint64_t active_status = 4;
constexpr std::size_t receipt_size = 28;
constexpr std::size_t header_size = 16;
#if defined(SYS_pidfd_open)
constexpr long pidfd_open_syscall = SYS_pidfd_open;
#elif defined(__x86_64__)
constexpr long pidfd_open_syscall = 434;  // Linux x86_64 UAPI, absent in old sysroots.
#else
#error "Linux pidfd_open syscall definition is required"
#endif
enum class Operation { ready = 1, lookup, retrieve, store, reset, erase, lookup_prefix, lease };

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

    void request(Operation op, std::string_view metadata) const {
        std::array<char, header_size> header{};
        std::copy(magic.begin(), magic.end(), header.begin());
        encode_number(header.data() + 4, 4, static_cast<unsigned>(op));
        encode_number(header.data() + 8, 8, metadata.size());
        send_all(header.data(), header.size());
        send_all(metadata.data(), metadata.size());
    }

    std::pair<std::uint64_t, std::uint64_t> response_header() const {
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
        return {status, size};
    }

    void response(std::size_t expected) const {
        const auto [status, size] = response_header();
        if (status != 0 || size != expected) transport_error("invalid daemon response");
    }

    LmCacheStatus transfer_response() const {
        const auto [status, size] = response_header();
        if (status == 1 && size == 0) return LmCacheStatus::miss;
        if (status == 3 && size == 0) return LmCacheStatus::declined;
        if (status != 0 || size != receipt_size) transport_error("invalid GPU transfer response");
        return LmCacheStatus::complete;
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

    void arm_transfer(int process_fd) {
        const char armed = 1;
        send_all(&armed, 1);
        deadline_ = Clock::time_point::max();
        process_fd_ = process_fd;
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
        std::array<pollfd, 2> wake{{{fd_, events, 0}, {process_fd_, POLLIN, 0}}};
        for (;;) {
            check_deadline();
            const int timeout = deadline_ == Clock::time_point::max() ? -1 :
                static_cast<int>(std::chrono::ceil<std::chrono::milliseconds>(
                    deadline_ - Clock::now()).count());
            const int status = poll(wake.data(), wake.size(), timeout);
            if (status > 0) {
                if (wake[0].revents) return;
                if (wake[1].revents & (POLLIN | POLLHUP)) transport_error("daemon process exited");
            }
            if (status < 0 && errno != EINTR) socket_error("poll");
        }
    }

    Clock::time_point deadline_;
    int fd_;
    int process_fd_ = -1;
};

std::string request_prefix(const LmCacheConfig& config) {
    std::string metadata;
    append_number(metadata, 2, config.name_space.size());
    metadata += config.name_space;
    return metadata;
}

template <std::size_t N>
void append_bytes(std::string& output, const std::array<std::uint8_t, N>& bytes) {
    output.append(reinterpret_cast<const char*>(bytes.data()), bytes.size());
}

void append_region(std::string& output, const LmCacheGpuRegion& region) {
    if (output.size() > max_metadata - 56) {
        throw std::invalid_argument("LMCache GPU metadata is too large");
    }
    append_number(output, 4, region.allocation);
    append_number(output, 8, region.device_offset);
    append_number(output, 8, region.device_pitch);
    append_number(output, 8, region.record_offset);
    append_number(output, 8, region.record_pitch);
    append_number(output, 8, region.width);
    append_number(output, 4, region.rows);
}

std::string encode_transfer(const LmCacheConfig& config, const LmCacheGpuTransfer& transfer) {
    if (transfer.allocations.empty() || transfer.allocations.size() > 64 ||
        transfer.records.empty() || transfer.records.size() > max_keys) {
        throw std::invalid_argument("Invalid LMCache GPU transfer size");
    }
    auto output = request_prefix(config);
    append_bytes(output, transfer.device);
    append_bytes(output, transfer.ready_event);
    append_number(output, 4, transfer.allocations.size());
    for (const auto& allocation : transfer.allocations) {
        if (allocation.bytes == 0) throw std::invalid_argument("Empty GPU allocation");
        append_bytes(output, allocation.handle);
        append_number(output, 8, allocation.bytes);
    }
    append_number(output, 4, transfer.records.size());
    for (const auto& record : transfer.records) {
        validate_record_size(record.bytes);
        if (record.regions.empty() || record.regions.size() > max_keys) {
            throw std::invalid_argument("Invalid LMCache GPU region count");
        }
        append_key(output, record.key);
        append_number(output, 8, record.bytes);
        append_number(output, 4, record.tokens);
        append_number(output, 4, record.regions.size());
        for (const auto& region : record.regions) append_region(output, region);
    }
    return output;
}

class GpuLease {
public:
    explicit GpuLease(const Connection& connection) {
        connection.read_all(admission_.data(), admission_.size());
        const auto pid = decode_number(std::string_view(admission_.data() + 32, 4));
        if (pid == 0 || pid > 0x7fffffff) transport_error("invalid daemon process ID");
        pidfd_ = static_cast<int>(syscall(pidfd_open_syscall, static_cast<pid_t>(pid), 0));
        if (pidfd_ < 0) socket_error("open daemon pidfd");
    }
    ~GpuLease() { close(pidfd_); }
    GpuLease(const GpuLease&) = delete;
    GpuLease& operator=(const GpuLease&) = delete;
    int process_fd() const { return pidfd_; }

    void wait_for_release(const LmCacheConfig& config) const {
        auto metadata = request_prefix(config);
        metadata.append(admission_.data(), 32);
        for (;;) {
            pollfd process{pidfd_, POLLIN, 0};
            if (poll(&process, 1, 100) > 0 && (process.revents & (POLLIN | POLLHUP))) return;
            try {
                const Connection status(config);
                status.request(Operation::lease, metadata);
                const auto [state, bytes] = status.response_header();
                if (state == 0 && bytes == 0) return;
            } catch (const std::runtime_error&) {
            }
        }
    }

private:
    std::array<char, 36> admission_{};
    int pidfd_ = -1;
};

LmCacheStatus transfer_records(const LmCacheConfig& config, Operation operation,
                               const LmCacheGpuTransfer& transfer, LmCacheReceipt* receipt) {
    if (receipt) *receipt = {};
    constexpr std::size_t admission_bytes = 36;
    const auto metadata = encode_transfer(config, transfer);
    Connection connection(config);
    connection.request(operation, metadata);
    const auto [admission, size] = connection.response_header();
    if (admission == 3 && size == 0) return LmCacheStatus::declined;
    if (admission != active_status || size != admission_bytes) {
        transport_error("GPU transfer admission rejected");
    }
    const GpuLease lease(connection);
    connection.arm_transfer(lease.process_fd());
    try {
        const auto status = connection.transfer_response();
        if (status != LmCacheStatus::complete) return status;
        std::array<char, receipt_size> raw{};
        connection.read_all(raw.data(), raw.size());
        if (receipt) {
            const std::string_view frame(raw.data(), raw.size());
            for (std::size_t family = 0; family < receipt->bytes.size(); ++family) {
                receipt->bytes[family] = decode_number(frame.substr(8 * family, 8));
            }
            receipt->gqa_written_tokens =
                static_cast<std::uint32_t>(decode_number(frame.substr(24)));
        }
        return status;
    } catch (...) {
        lease.wait_for_release(config);
        throw;
    }
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

LmCachePrefix LmCacheClient::lookup_prefix(std::string_view digests, int max_restore,
                                           bool draft) const {
    constexpr std::size_t digest_bytes = 64;
    const auto count = digests.size() / digest_bytes;
    auto metadata = request_prefix(config_);
    if (digests.size() % digest_bytes || count > max_keys ||
        digests.size() > max_metadata - metadata.size() - 9 || max_restore < 0 ||
        static_cast<std::size_t>(max_restore) > count) {
        throw std::invalid_argument("Invalid LMCache prefix lookup metadata");
    }
    append_number(metadata, 4, count);
    append_number(metadata, 4, max_restore);
    append_number(metadata, 1, draft);
    metadata += digests;
    const Connection connection(config_);
    connection.request(Operation::lookup_prefix, metadata);
    connection.response(16);
    std::array<char, 16> raw{};
    connection.read_all(raw.data(), raw.size());
    const std::string_view frame(raw.data(), raw.size());
    const auto gqa = decode_number(frame.substr(0, 4));
    const auto gdn = decode_number(frame.substr(4, 4));
    const auto draft_rows = decode_number(frame.substr(8, 4));
    const auto matched_gdn = decode_number(frame.substr(12));
    if (gqa > count || matched_gdn > count || gdn > matched_gdn ||
        gdn > std::min(gqa, static_cast<std::uint64_t>(max_restore)) ||
        draft_rows != (draft && gdn ? gdn - 1 : 0)) {
        transport_error("invalid prefix coverage result");
    }
    return {static_cast<int>(gqa), static_cast<int>(gdn), static_cast<int>(draft_rows),
            static_cast<int>(matched_gdn)};
}

LmCacheStatus LmCacheClient::store(const LmCacheGpuTransfer& transfer,
                                    LmCacheReceipt* receipt) const {
    return transfer_records(config_, Operation::store, transfer, receipt);
}

LmCacheStatus LmCacheClient::retrieve(const LmCacheGpuTransfer& transfer,
                                       LmCacheReceipt* receipt) const {
    return transfer_records(config_, Operation::retrieve, transfer, receipt);
}

void LmCacheClient::reset() const {
    const Connection connection(config_);
    connection.request(Operation::reset, request_prefix(config_));
    connection.response(0);
}

}  // namespace sglang::decoder
