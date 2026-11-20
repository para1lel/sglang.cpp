#include "sglang/server/batch_service.h"
#include "sglang/server/http.h"
#include "sglang/server/prefix_controls.h"

#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <iostream>
#include <iterator>
#include <memory>
#include <mutex>
#include <netinet/in.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <thread>
#include <utility>

namespace {

using namespace std::chrono_literals;
using namespace sglang::server;
using Clock = std::chrono::steady_clock;
constexpr auto timeout = 3s;
constexpr int capacity = 2;

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

struct Gate {
    bool entered = false;
    bool released = false;
    int slot = -1;
    std::optional<bool> sent;
};

class Gates {
public:
    void enter(Gate& gate, int slot = -1) {
        std::unique_lock lock(mutex_);
        require(!gate.entered, "A cancelled or retired request entered its handler again");
        gate.entered = true;
        gate.slot = slot;
        changed_.notify_all();
        changed_.wait(lock, [&] { return gate.released || stopped_; });
    }

    int wait_entered(const Gate& gate) {
        std::unique_lock lock(mutex_);
        require(changed_.wait_for(lock, timeout, [&] { return gate.entered; }),
                "Expected completion or reset barrier was not reached");
        return gate.slot;
    }

    void release(Gate& gate) {
        const std::lock_guard lock(mutex_);
        gate.released = true;
        changed_.notify_all();
    }

    void record(Gate& gate, bool sent) {
        const std::lock_guard lock(mutex_);
        gate.sent = sent;
        changed_.notify_all();
    }

    bool finished(const Gate& gate) {
        const std::lock_guard lock(mutex_);
        return gate.sent.has_value();
    }

    void expect_write(const Gate& gate, bool sent) {
        std::unique_lock lock(mutex_);
        require(changed_.wait_for(lock, timeout, [&] { return gate.sent.has_value(); }),
                "Final response write did not return");
        require(*gate.sent == sent, "Unexpected final response write result");
    }

    void release_all() {
        const std::lock_guard lock(mutex_);
        stopped_ = true;
        changed_.notify_all();
    }

private:
    std::mutex mutex_;
    std::condition_variable changed_;
    bool stopped_ = false;
};

struct BufferedBurst {
    explicit BufferedBurst(int requests) : remaining(requests) {}

    Gate accept;
    int remaining;
    bool started = false;
    std::atomic<int> preloaded{0};
};

Gates gates;
Gate reset_ack;
std::atomic<bool> delay_reset{false};
std::atomic<bool> worker_failed{false};
std::atomic<bool> fail_accept{false};
std::atomic<BufferedBurst*> buffered_burst{nullptr};
BufferedBurst idle_burst(2);
BufferedBurst saturated_burst(4);
constexpr int large_body_bytes = 48 * 1024;

}

namespace sglang {

struct Decoder::Model {};
Decoder::Decoder(const std::filesystem::path&, int) {}
Decoder::~Decoder() = default;

namespace server {

struct BatchService::State { int cache_capacity; };

BatchService::BatchService(const Decoder&, bool, bool, int, int, int cache, int)
    : state_(std::make_unique<State>(State{cache})) {}
BatchService::~BatchService() = default;

bool BatchService::ready() const { return !worker_failed; }

void BatchService::rethrow_failure() const {
    if (!ready()) {
        throw std::runtime_error("Injected decoder worker failure");
    }
}

BatchMetrics BatchService::metrics() const {
    BatchMetrics metrics;
    metrics.cache_capacity = state_->cache_capacity;
    return metrics;
}

bool BatchService::reset_prefix_cache() {
    rethrow_failure();
    if (delay_reset.exchange(false)) {
        gates.enter(reset_ack);
    }
    return true;
}

bool BatchService::start_profile(std::string, std::size_t) { return false; }
std::optional<BatchProfile> BatchService::stop_profile() { return std::nullopt; }

}
}

namespace {

void wait_readable(const Socket& socket, std::chrono::milliseconds duration) {
    pollfd ready{socket.get(), POLLIN, 0};
    require(poll(&ready, 1, static_cast<int>(duration.count())) == 1,
            "Timed out waiting for HTTP response or peer closure");
}

std::string read_closed(const Socket& socket, std::chrono::milliseconds limit = timeout) {
    std::string response;
    const auto deadline = Clock::now() + limit;
    for (;;) {
        const auto remaining =
            std::chrono::ceil<std::chrono::milliseconds>(deadline - Clock::now());
        require(remaining.count() > 0, "HTTP response did not close");
        wait_readable(socket, remaining);
        char buffer[4096];
        const auto count = recv(socket.get(), buffer, sizeof(buffer), 0);
        if (count == 0 || (count < 0 && errno == ECONNRESET)) {
            return response;
        }
        require(count > 0, "Cannot read HTTP response");
        response.append(buffer, static_cast<std::size_t>(count));
    }
}

void require_status(const std::string& response, std::string_view status) {
    const auto prefix = "HTTP/1.1 " + std::string(status) + ' ';
    if (response.compare(0, prefix.size(), prefix) != 0) {
        throw std::runtime_error("Expected " + std::string(status) + ", received: " + response);
    }
}

void write_request(const Socket& socket, std::string_view bytes) {
    while (!bytes.empty()) {
        const auto sent = send(socket.get(), bytes.data(), bytes.size(), MSG_NOSIGNAL);
        require(sent > 0, "Cannot send HTTP request");
        bytes.remove_prefix(static_cast<std::size_t>(sent));
    }
}

std::string request(std::string_view method, std::string_view path,
                    std::string_view body = std::string_view{}) {
    return std::string(method) + ' ' + std::string(path) +
        " HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: " +
        std::to_string(body.size()) + "\r\n\r\n" + std::string(body);
}

enum RequestId {
    terminal, reset_waiter, nondraining, buffered_idle_first, buffered_idle_second,
    first, second, survivor, contender_text, contender_chat, replacement, overflow,
    buffered_first, buffered_second, buffered_overflow_first, buffered_overflow_second,
    last_first, last_second, abandoned, request_count
};

bool is_chat(RequestId id) { return id == survivor || id == contender_chat; }

bool buffered(RequestId id) {
    return id == buffered_idle_first || id == buffered_idle_second ||
           (id >= buffered_first && id <= buffered_overflow_second);
}

std::string completion_body(RequestId id) {
    const auto token = std::to_string(id + 101);
    if (is_chat(id)) {
        return R"({"model":"Qwen3.8-27B-FP8","max_tokens":1,"messages":[{"role":"user","content":")"
            + token + R"("}]})";
    }
    auto body = R"({"model":"benchmark","max_tokens":1,"prompt":[)" + token + "]}";
    if (buffered(id)) {
        body.resize(large_body_bytes, ' ');
    }
    return body;
}

std::string response_body(RequestId id) {
    return "{\"request\":" + std::to_string(id + 101) + '}';
}

class Server {
public:
    Server(BatchService& service, int cache) : cache_(cache), worker_([&, this] {
        try {
            serve_connections(listener_, service, capacity,
                [&](const Socket& socket, int slot, std::string_view body, bool chat) {
                    complete(socket, slot, body, chat);
                }, R"({"max_active":2})", cache_ != 0);
        } catch (...) {
            failure_ = std::current_exception();
        }
        returned_ = true;
    }) {}

    ~Server() {
        if (worker_.joinable()) {
            stop_serving();
            gates.release_all();
            worker_.join();
        }
    }

    Socket submit(std::string_view message, bool reset_on_close = false,
                  bool small_window = false) {
        Socket client(socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0));
        require(client.get() >= 0, "Cannot create client socket");
        if (reset_on_close) {
            const linger rst{1, 0};
            require(setsockopt(client.get(), SOL_SOCKET, SO_LINGER, &rst, sizeof(rst)) == 0,
                    "Cannot enable reset on close");
        }
        if (small_window) {
            const int bytes = 4096;
            require(setsockopt(client.get(), SOL_SOCKET, SO_RCVBUF, &bytes, sizeof(bytes)) == 0,
                    "Cannot set a small client window");
        }
        sockaddr_in address{};
        socklen_t length = sizeof(address);
        require(getsockname(listener_.get(), reinterpret_cast<sockaddr*>(&address), &length) == 0 &&
                connect(client.get(), reinterpret_cast<const sockaddr*>(&address), length) == 0,
                "Cannot connect to test listener");
        write_request(client, message);
        return client;
    }

    Socket completion(RequestId id, bool rst = false) {
        return submit(request("POST", is_chat(id) ? "/v1/chat/completions" : "/v1/completions",
                              completion_body(id)), rst, id == nondraining);
    }

    std::string control(std::string_view path, std::string_view status = "200") {
        const auto response = read_closed(submit(request("GET", path)));
        require_status(response, status);
        return response;
    }

    Socket reset_request() { return submit(request("POST", "/reset_prefix_cache", "{}")); }

    void wait_queue(int running, int waiting) {
        if (cache_ == 0) {
            control("/health");
            return;
        }
        const auto expected_running = "vllm:num_requests_running{engine=\"0\"} " +
            std::to_string(running) + '\n';
        const auto expected_waiting = "vllm:num_requests_waiting{engine=\"0\"} " +
            std::to_string(waiting) + '\n';
        const auto deadline = Clock::now() + timeout;
        do {
            const auto metrics = control("/metrics");
            if (metrics.find(expected_running) != std::string::npos &&
                metrics.find(expected_waiting) != std::string::npos) {
                return;
            }
            std::this_thread::yield();
        } while (Clock::now() < deadline);
        throw std::runtime_error("Queue metrics did not reach the expected state");
    }

    void check_controls(std::string_view reset_status) {
        require(control("/health").find(R"({"max_active":2})") != std::string::npos,
                "Health lost the server capacity");
        require(control("/ready").find(R"({"ready":true})") != std::string::npos,
                "A busy worker was reported unready");
        control("/metrics", cache_ == 0 ? "404" : "200");
        require_status(read_closed(reset_request()), cache_ == 0 ? "404" : reset_status);
    }

    Gate& gate(RequestId id) { return completed_[id]; }
    bool returned() const { return returned_; }

    void join() {
        worker_.join();
        if (failure_) {
            std::rethrow_exception(failure_);
        }
    }

private:
    void complete(const Socket& socket, int slot, std::string_view body, bool chat) {
        require(slot >= 0 && slot < capacity, "Completion received a foreign slot");
        for (int index = 0; index < request_count; ++index) {
            const auto id = static_cast<RequestId>(index);
            if (is_chat(id) != chat || body != completion_body(id)) {
                continue;
            }
            // The fake model has finished. Only the final response remains.
            gates.enter(gate(id), slot);
            if (id == nondraining) {
                const int bytes = 4096;
                require(setsockopt(socket.get(), SOL_SOCKET, SO_SNDBUF, &bytes, sizeof(bytes)) == 0,
                        "Cannot set a small server send buffer");
                const auto large = "{\"text\":\"" + std::string(512 * 1024, 'x') + "\"}";
                gates.record(gate(id), write_json(socket, "200 OK", large));
            } else {
                gates.record(gate(id), write_json(socket, "200 OK", response_body(id)));
            }
            return;
        }
        throw std::logic_error("Completion received the wrong request or route");
    }

    const int cache_;
    const Socket listener_ = listen_loopback(0);
    std::array<Gate, request_count> completed_{};
    std::exception_ptr failure_;
    std::atomic<bool> returned_{false};
    std::thread worker_;
};

void require_completion(const Socket& client, RequestId id) {
    const auto response = read_closed(client);
    require_status(response, "200");
    const auto body = "\r\n\r\n" + response_body(id);
    require(response.size() >= body.size() &&
            response.compare(response.size() - body.size(), body.size(), body) == 0,
            "Completion response reached the wrong client");
}

constexpr std::string_view incomplete_header = "GET /ready HTTP/1.1\r\nHost: localhost\r\n";
constexpr std::string_view incomplete_body = "POST /v1/completions HTTP/1.1\r\nHost: localhost\r\n"
    "Content-Type: application/json\r\nContent-Length: 32\r\n\r\n{";

void check_readers(Server& server) {
    const auto oldest = server.submit(incomplete_header);
    const auto middle = server.submit(incomplete_body);
    server.control("/health");
    server.control("/ready");
    const auto recent = server.submit(incomplete_header);
    const auto newest = server.submit(incomplete_body);
    require(read_closed(oldest).empty(), "Oldest incomplete request survived ingress overflow");
    write_request(recent, "\r\n");
    require_status(read_closed(recent), "200");
    server.check_controls("200");
}

void check_read_deadline(Server& server) {
    const auto start = Clock::now();
    const auto header = server.submit(incomplete_header);
    const auto body = server.submit(incomplete_body);
    server.control("/health");
    pollfd ready[] = {{header.get(), POLLIN, 0}, {body.get(), POLLIN, 0}};
    require(poll(ready, 2, 3000) == 0, "Incomplete request closed before its deadline");
    write_request(header, "X-Still-Uploading: 1\r\n");
    write_request(body, "\"key\":");
    server.control("/ready");
    const auto deadline = start + 7s;
    for (const auto* client : {&header, &body}) {
        const auto remaining =
            std::chrono::ceil<std::chrono::milliseconds>(deadline - Clock::now());
        require(remaining.count() > 0 && read_closed(*client, remaining).empty(),
                "Progress extended the original incomplete request deadline");
    }
}

void check_buffered_readers(Server& server) {
    buffered_burst = &idle_burst;
    const auto first_client = server.completion(buffered_idle_first);
    gates.wait_entered(idle_burst.accept);
    const auto second_client = server.completion(buffered_idle_second);
    const auto first_stall = server.submit(incomplete_header);
    const auto second_stall = server.submit(incomplete_body);
    gates.release(idle_burst.accept);
    server.control("/health");
    require(idle_burst.preloaded == 2, "Complete requests were not already buffered on accept");

    int first_slot;
    try {
        first_slot = gates.wait_entered(server.gate(buffered_idle_first));
    } catch (const std::runtime_error&) {
        char byte;
        const auto count = recv(first_client.get(), &byte, 1, MSG_PEEK | MSG_DONTWAIT);
        require(count != 0 && !(count < 0 && errno == ECONNRESET),
                "A fully buffered completion closed before reaching its handler");
        throw;
    }
    require(gates.wait_entered(server.gate(buffered_idle_second)) != first_slot,
            "Two buffered completions did not claim separate idle slots");
    gates.release(server.gate(buffered_idle_first));
    gates.release(server.gate(buffered_idle_second));
    require_completion(first_client, buffered_idle_first);
    require_completion(second_client, buffered_idle_second);
    server.wait_queue(0, 0);
}

void check_write_lease(Server& server, bool cached) {
    const auto body = completion_body(terminal);
    const auto message = request("POST", "/v1/completions", body);
    const auto client = server.submit(std::string_view(message).substr(0, message.size() - 2));
    server.check_controls("200");
    write_request(client, std::string_view(body).substr(body.size() - 2));
    gates.wait_entered(server.gate(terminal));
    server.wait_queue(1, 0);
    server.check_controls("409");
    gates.release(server.gate(terminal));
    require_completion(client, terminal);
    server.wait_queue(0, 0);
    if (cached) {
        delay_reset = true;
        const auto reset = server.reset_request();
        gates.wait_entered(reset_ack);
        const auto waiter = server.completion(reset_waiter);
        server.wait_queue(0, 1);
        server.control("/health");
        server.control("/ready");
        require_status(read_closed(server.reset_request()), "409");
        gates.release(reset_ack);
        require_status(read_closed(reset), "200");
        gates.wait_entered(server.gate(reset_waiter));
        gates.release(server.gate(reset_waiter));
        require_completion(waiter, reset_waiter);
        server.wait_queue(0, 0);
        require_status(read_closed(server.reset_request()), "200");
    }
    {
        const auto stalled = server.completion(nondraining, true);
        gates.wait_entered(server.gate(nondraining));
        gates.release(server.gate(nondraining));
        wait_readable(stalled, timeout);
        server.check_controls("409");
        require(!gates.finished(server.gate(nondraining)),
                "Nondraining socket did not block its write");
    }
    gates.expect_write(server.gate(nondraining), false);
    server.wait_queue(0, 0);
    server.check_controls("200");
}

void check_queue(Server& server) {
    const auto active_first = server.completion(first);
    const auto first_slot = gates.wait_entered(server.gate(first));
    const auto active_second = server.completion(second);
    require(gates.wait_entered(server.gate(second)) != first_slot, "Active requests shared a slot");
    const auto first_queued = server.completion(survivor);
    server.wait_queue(2, 1);
    {
        const auto plain = server.completion(contender_text, true);
        const auto chat = server.completion(contender_chat, true);
        pollfd ready[] = {{plain.get(), POLLIN, 0}, {chat.get(), POLLIN, 0}};
        require(poll(ready, 2, std::chrono::milliseconds(timeout).count()) == 1,
                "Expected one rejection for the last pending position");
        require_status(read_closed(ready[0].revents ? plain : chat), "503");
        server.check_controls("409");
    }
    // Admission must discard the reset peer without a metrics call or a retiring worker.
    const auto last_queued = server.completion(replacement);
    const auto rejected = server.completion(overflow);
    require_status(read_closed(rejected), "503");
    server.wait_queue(2, 2);

    gates.release(server.gate(first));
    require_completion(active_first, first);
    require(gates.wait_entered(server.gate(survivor)) == first_slot,
            "FIFO survivor lost the freed slot");
    gates.release(server.gate(survivor));
    require_completion(first_queued, survivor);
    require(gates.wait_entered(server.gate(replacement)) == first_slot,
            "Replacement lost the freed slot");

    buffered_burst = &saturated_burst;
    const auto large_first = server.completion(buffered_first);
    gates.wait_entered(saturated_burst.accept);
    const auto large_second = server.completion(buffered_second);
    const auto large_extra_first = server.completion(buffered_overflow_first);
    const auto large_extra_second = server.completion(buffered_overflow_second);
    gates.release(saturated_burst.accept);
    require_status(read_closed(large_extra_first), "503");
    require_status(read_closed(large_extra_second), "503");
    require(saturated_burst.preloaded == 4,
            "Complete requests were not already buffered on accept");
    server.wait_queue(2, 2);

    gates.release(server.gate(replacement));
    require_completion(last_queued, replacement);
    require(gates.wait_entered(server.gate(buffered_first)) == first_slot,
            "First buffered completion lost its FIFO position");
    gates.release(server.gate(buffered_first));
    require_completion(large_first, buffered_first);
    require(gates.wait_entered(server.gate(buffered_second)) == first_slot,
            "Second buffered completion lost its FIFO position");
    gates.release(server.gate(buffered_second));
    require_completion(large_second, buffered_second);
    gates.release(server.gate(second));
    require_completion(active_second, second);
    server.wait_queue(0, 0);
}

void check_shutdown(Server& server, bool reader_failure) {
    const auto active_first = server.completion(last_first);
    gates.wait_entered(server.gate(last_first));
    const auto active_second = server.completion(last_second);
    gates.wait_entered(server.gate(last_second));
    const auto queued = server.completion(abandoned);
    server.wait_queue(2, 1);
    const auto partial = server.submit(incomplete_body);
    server.control("/health");
    if (reader_failure) {
        fail_accept = true;
    } else {
        stop_serving();
    }
    // Wake the listener while the active callbacks are still held.
    const auto wake = server.submit({});
    require(read_closed(queued).empty(), "Shutdown delivered a queued response");
    require(read_closed(partial).empty(), "Shutdown did not close the incomplete request");
    require(!server.returned(), "Serving returned while completion gates were held");
    gates.release(server.gate(last_first));
    gates.release(server.gate(last_second));
    require(read_closed(active_first).empty() && read_closed(active_second).empty(),
            "Active completions did not close during shutdown");
    bool accept_error = false;
    try {
        server.join();
    } catch (const std::system_error& error) {
        if (!reader_failure || error.code().value() != EMFILE) {
            throw;
        }
        accept_error = true;
    }
    require(accept_error == reader_failure, "Reader failure did not reach the caller");
}

std::ptrdiff_t fd_count() {
    return std::distance(std::filesystem::directory_iterator("/proc/self/fd"),
                         std::filesystem::directory_iterator{});
}

}

extern "C" int __real_accept4(int listener, sockaddr* address, socklen_t* length, int flags);
extern "C" int __wrap_accept4(int listener, sockaddr* address, socklen_t* length, int flags) {
    if (fail_accept.exchange(false)) {
        errno = EMFILE;
        return -1;
    }
    auto* burst = buffered_burst.load();
    if (burst && !burst->started) {
        burst->started = true;
        gates.enter(burst->accept);
    }
    const int fd = __real_accept4(listener, address, length, flags);
    if (fd >= 0 && burst) {
        static const auto request_size = static_cast<int>(
            request("POST", "/v1/completions", completion_body(buffered_idle_first)).size());
        int available = 0;
        if (ioctl(fd, FIONREAD, &available) == 0 && available == request_size) {
            ++burst->preloaded;
        }
        if (--burst->remaining == 0) {
            buffered_burst = nullptr;
        }
    }
    return fd;
}

int main(int argc, char** argv) {
    const char* phase = "configuration";
    try {
        require(argc == 2,
                "Expected --cache-off, --cache-on, --reader-failure or --worker-failure");
        const std::string_view mode = argv[1];
        const bool reader_failure = mode == "--reader-failure";
        const bool broken_worker = mode == "--worker-failure";
        const bool cached = mode == "--cache-on" || reader_failure;
        require(cached || broken_worker || mode == "--cache-off", "Unknown transport test mode");
        const auto descriptors = fd_count();
        const sglang::Decoder decoder(std::filesystem::path{});
        BatchService service(decoder, false, false, 0, capacity, cached ? 2 : 0);
        for (const int invalid : {0, sglang::decoder_batch_max_lanes + 1}) {
            bool rejected = false;
            try {
                serve_connections(Socket(-1), service, invalid, {}, "{}", cached);
            } catch (const std::invalid_argument&) {
                rejected = true;
            }
            require(rejected, "Invalid listener capacity was accepted");
        }
        {
            Server server(service, cached ? 2 : 0);
            if (broken_worker) {
                phase = "worker failure";
                worker_failed = true;
                server.control("/health");
                require(server.control("/ready", "503").find("worker_failed") != std::string::npos,
                        "Unavailable worker omitted the readiness reason");
                require_status(read_closed(server.completion(terminal)), "500");
                stop_serving();
                server.join();
            } else {
                phase = "incomplete requests";
                check_readers(server);
                if (mode == "--cache-on") {
                    phase = "incomplete request deadline";
                    check_read_deadline(server);
                }
                phase = "buffered completions with idle slots";
                check_buffered_readers(server);
                phase = "response lease";
                check_write_lease(server, cached);
                phase = "completion queue";
                check_queue(server);
                phase = "shutdown";
                check_shutdown(server, reader_failure);
            }
        }
        phase = "descriptor count";
        require(fd_count() == descriptors, "Transport leaked file descriptors");
        std::cout << "HTTP reader queue passed " << mode << '\n';
    } catch (const std::exception& error) {
        std::cerr << "HTTP reader queue failed during " << phase << ": " << error.what() << '\n';
        return 1;
    }
}
