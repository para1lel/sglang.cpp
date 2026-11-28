#include "sglang/server/batch_service.h"
#include "sglang/server/completions.h"
#include "sglang/server/http.h"
#include "sglang/server/native_generate.h"
#include "sglang/server/prefix_controls.h"
#include "sglang/server/tokenization.h"

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
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <thread>
#include <utility>
#include <vector>

namespace {

using namespace std::chrono_literals;
using namespace sglang::server;
using Clock = std::chrono::steady_clock;
constexpr auto timeout = 3s;
std::ptrdiff_t fd_count();
constexpr int capacity = 2;

class ReaderQueueExecutor final : public BatchExecutor {
public:
    ReaderQueueExecutor()
        : BatchExecutor({false, capacity, 0, sglang::decoder_default_max_num_batched_tokens}) {}
    BatchTopology topology() const noexcept override { return single_device_batch_topology; }
    void start(Ready, Process) override { throw std::logic_error("Unexpected decoder startup"); }
    void join() noexcept override {}
};

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

    void wait_released(const Gate& gate) {
        std::unique_lock lock(mutex_);
        require(changed_.wait_for(lock, timeout, [&] { return gate.released; }),
                "Shutdown did not stop restores before joining completion workers");
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
Gate restore_stop;
std::atomic<bool> delay_reset{false};
std::atomic<int> cache_resets{0};
std::atomic<bool> worker_failed{false};
std::atomic<bool> fail_accept{false};
std::atomic<BufferedBurst*> buffered_burst{nullptr};
BufferedBurst idle_burst(2);
BufferedBurst saturated_burst(4);
constexpr int large_body_bytes = 48 * 1024;

}

namespace sglang::server {

struct BatchService::State {
    std::atomic<bool> abort_notified{false};
};

BatchService::BatchService(std::unique_ptr<BatchExecutor>, bool, int)
    : state_(std::make_unique<State>()) {}
BatchService::~BatchService() = default;

std::optional<std::vector<DecoderResult>> BatchService::generate_batch(
    int, const std::vector<BatchPrompt>&, const BatchCallbacks&) {
    throw std::logic_error("Unexpected decoder batch generation in reader queue fixture");
}

DecoderResult BatchService::generate(int slot, const std::vector<int>& prompt,
                                     const DecoderOptions& options,
                                     [[maybe_unused]] const DecoderCallbacks& callbacks,
                                     std::vector<DecoderMedia> media,
                                     [[maybe_unused]] std::size_t expanded_tokens,
                                     [[maybe_unused]] BatchRequestId* request_id) {
    const auto aborted = state_->abort_notified.exchange(false);
    media.clear();
    throw std::logic_error("Unexpected decoder generation in slot " + std::to_string(slot) +
                           " for " + std::to_string(prompt.size()) + " prompt tokens and " +
                           std::to_string(options.max_new_tokens) + " output tokens" +
                           (aborted ? " after an abort notification" : ""));
}

void BatchService::notify_abort() {
    state_->abort_notified = true;
}

bool BatchService::ready() const { return !worker_failed; }
void BatchService::stop_cache_restores() noexcept { gates.release(restore_stop); }

bool BatchService::wait_for_generation(Clock::time_point, Clock::time_point,
                                       const std::function<bool()>& cancelled) {
    return ready() && !cancelled();
}

void BatchService::rethrow_failure() const {
    if (!ready()) {
        throw std::runtime_error("Injected decoder worker failure");
    }
}

BatchMetrics BatchService::metrics() const { return {}; }

bool BatchService::reset_prefix_cache() {
    rethrow_failure();
    ++cache_resets;
    if (delay_reset.exchange(false)) {
        gates.enter(reset_ack);
    }
    return true;
}

bool BatchService::start_profile(std::string, std::size_t) { return false; }
std::optional<BatchProfile> BatchService::stop_profile() { return std::nullopt; }

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
    terminal,
    reset_waiter,
    nondraining,
    buffered_idle_first,
    buffered_idle_second,
    first,
    second,
    survivor,
    contender_text,
    contender_chat,
    replacement,
    overflow,
    buffered_first,
    buffered_second,
    buffered_overflow_first,
    buffered_overflow_second,
    auxiliary_text,
    auxiliary_chat,
    prepare_text,
    prepare_peer,
    prepare_waiter,
    last_first,
    last_second,
    abandoned,
    request_count,
};

bool is_chat(RequestId id) {
    return id == survivor || id == contender_chat || id == auxiliary_chat || id == prepare_waiter;
}

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
    Server(BatchService& service, int cache)
        : cache_(cache), native_(service, tokenizer_), worker_([&, this] { serve(service); }) {
    }

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

    unsigned listen_backlog() const {
        tcp_info info{};
        socklen_t size = sizeof(info);
        require(getsockopt(listener_.get(), IPPROTO_TCP, TCP_INFO, &info, &size) == 0,
                "Cannot read the listener backlog limit");
        return info.tcpi_sacked;
    }

    static constexpr std::string_view gated_tokenization = R"({"tokens": []})";
    static constexpr std::string_view shutdown_tokenization = R"({ "tokens":[] })";

    Socket generate(bool gated, bool rst = false) {
        return submit(request("POST", "/generate", gated ? generate_body_ : plain_generate), rst);
    }

    Socket abort_waiting() {
        return submit(request("POST", "/abort_request", R"({"rid":"waiting"})"));
    }

    Gate& preparing() {
        return preparing_;
    }
    Gate& cpu_handler() {
        return cpu_handler_;
    }
    Gate& cpu_shutdown() {
        return cpu_shutdown_;
    }

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
    void serve(BatchService& service) {
        try {
            serve_connections(
                listener_, service, capacity,
                [&](const Socket& socket, int slot, std::string_view body, bool chat) {
                    complete(socket, slot, body, chat);
                },
                R"({"max_active":2})", cache_ != 0,
                [&](std::string_view body) {
                    auto prepared = native_.prepare(body);
                    if (body == generate_body_) {
                        gates.enter(preparing_);
                    }
                    auto cancelled = prepared.aborted;
                    prepared.complete = [cancelled = std::move(cancelled)](
                                            const Socket& socket, std::optional<int> slot) {
                        require(!slot && cancelled(), "Waiting abort used a completion slot");
                        write_json(socket, "200 OK", R"({"aborted":true})");
                    };
                    return prepared;
                },
                [&](const Socket& socket, TokenizationRoute route, std::string_view body) {
                    if (body == gated_tokenization) {
                        gates.enter(cpu_handler_);
                    } else if (body == shutdown_tokenization) {
                        gates.enter(cpu_shutdown_);
                    }
                    serve_tokenization(socket, route, tokenizer_, body);
                },
                {},
                [&](const Socket& socket, std::string_view body) { native_.abort(socket, body); });
        } catch (...) {
            failure_ = std::current_exception();
        }
        returned_ = true;
    }

    void complete(const Socket& socket, int slot, std::string_view body, bool chat) {
        require(slot >= 0 && slot < capacity, "Completion received a foreign slot");
        for (int index = 0; index < request_count; ++index) {
            const auto id = static_cast<RequestId>(index);
            if (is_chat(id) != chat || body != completion_body(id)) {
                continue;
            }
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
    static constexpr sglang::model::ServingProfile profile_{};
    static constexpr std::string_view plain_generate = R"({"input_ids":[101],"rid":"waiting"})";
    const std::string generate_body_ =
        std::string(plain_generate) + std::string(large_body_bytes - plain_generate.size(), ' ');
    static constexpr ChatParsers parsers_{};
    const CompletionTokenizer tokenizer_{"", SamplingDefaults::model, parsers_, profile_};
    NativeGenerate native_;
    Gate preparing_;
    Gate cpu_handler_;
    Gate cpu_shutdown_;
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

void require_flush(const Socket& client, std::string_view status, std::string_view body) {
    const auto response = read_closed(client);
    require_status(response, status);
    const auto headers = response.find("\r\n\r\n");
    require(headers != std::string::npos && response.substr(headers + 4) == body,
            "Incorrect flush result");
}

void check_flush_queue(Server& server) {
    const auto active = server.completion(terminal);
    gates.wait_entered(server.gate(terminal));
    const auto initial_resets = cache_resets.load();
    const auto started = Clock::now();
    const auto first = server.submit(request("POST", "/flush_cache?timeout=.45"));
    server.control("/health");
    const auto second = server.submit(request("GET", "/flush_cache?timeout=.25"));
    require_status(read_closed(server.reset_request()), "409");
    pollfd pending[] = {{first.get(), POLLIN, 0}, {second.get(), POLLIN, 0}};
    constexpr std::string_view expired = "Timed out waiting for idle state.";
    constexpr int timeout_ms = std::chrono::milliseconds(timeout).count();
    require(poll(pending, 2, timeout_ms) > 0 &&
                (static_cast<unsigned>(pending[0].revents) & unsigned{POLLIN}) != 0,
            "A flush bypassed the busy FIFO head or the head did not expire");
    require_flush(first, "400", expired);
    require(Clock::now() - started >= 450ms, "First flush expired early");
    const auto third = server.submit(request("GET", "/flush_cache"));
    pollfd following_second[] = {{second.get(), POLLIN, 0}, {third.get(), POLLIN, 0}};
    require(poll(following_second, 2, timeout_ms) > 0 &&
                (static_cast<unsigned>(following_second[0].revents) & unsigned{POLLIN}) != 0,
            "The zero-timeout flush bypassed the second waiter or the waiter did not expire");
    require_flush(second, "400", expired);
    require(Clock::now() - started >= 700ms, "Second timeout overlapped the first budget");
    require_flush(third, "400", "Flush cache failed.\n");
    require(cache_resets == initial_resets, "A busy flush reached the cache");
    {
        const auto forever = server.submit(request("GET", "/flush_cache?timeout=inf"), true);
        server.control("/health");
        const auto cancelled = server.submit(request("GET", "/flush_cache?timeout=inf"), true);
        server.control("/health");
        const auto immediate = server.submit(request("GET", "/flush_cache"));
        const auto overflow = server.submit(request("GET", "/flush_cache"));
        pollfd deferred[] = {{immediate.get(), POLLIN, 0}, {overflow.get(), POLLIN, 0}};
        require(poll(deferred, 2, timeout_ms) > 0, "Flush queue did not reject overload");
        require_status(read_closed(deferred[0].revents != 0 ? immediate : overflow), "503");
    }
    const auto after_disconnect = server.submit(request("GET", "/flush_cache"));
    require_flush(after_disconnect, "400", "Flush cache failed.\n");
    require(cache_resets == initial_resets, "A cancelled flush reached the cache");
    const auto waiting = server.submit(request("GET", "/flush_cache?timeout=60"));
    server.control("/health");
    const auto following = server.submit(request("POST", "/flush_cache"));
    server.control("/ready");
    pollfd deferred[] = {{waiting.get(), POLLIN, 0}, {following.get(), POLLIN, 0}};
    require(poll(deferred, 2, 50) == 0, "A deferred flush completed before idle");
    gates.release(server.gate(terminal));
    require_completion(active, terminal);
    constexpr std::string_view flushed = "Cache flushed.\nPlease check backend logs for more "
        "details. (When there are running or waiting requests, the operation will not be "
        "performed.)\n";
    require_flush(waiting, "200", flushed);
    require_flush(following, "200", flushed);
    require(cache_resets == initial_resets + 2, "Deferred flushes did not each reset the cache");
    require_status(read_closed(server.reset_request()), "200");
    require(cache_resets == initial_resets + 3, "Legacy reset did not reach the cache");
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

void check_read_limits(Server& server) {
    constexpr std::size_t header_limit = 8192;
    constexpr std::size_t body_limit = std::size_t{2} * 1024 * 1024;
    require(server.listen_backlog() == 16, "Unexpected TCP listener backlog limit");
    auto header = std::string(incomplete_header) + "X-Pad: ";
    header.resize(header_limit - 4, 'a');
    require_status(read_closed(server.submit(header + "\r\n\r\n")), "200");
    require_status(read_closed(server.submit(header + "a\r\n\r\n")), "431");
    const auto too_large = std::string(incomplete_header) +
                           "Content-Length: " + std::to_string(body_limit + 1) + "\r\n\r\n";
    require_status(read_closed(server.submit(too_large)), "413");
    const auto unknown = server.submit(request("POST", "/chat", std::string(body_limit, ' ')));
    require_status(read_closed(unknown), "404");

    const auto initial_descriptors = fd_count();
    std::vector<Socket> clients;
    clients.reserve(10);
    for (int i = 0; i < 10; ++i) {
        clients.push_back(server.submit(incomplete_body, true));
    }
    server.control("/health");
    require(fd_count() <=
                initial_descriptors + static_cast<std::ptrdiff_t>(clients.size()) + capacity + 1,
            "The reader exceeded its accepted descriptor bound");
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

void check_auxiliary_queue(Server& server) {
    const auto tokenize = server.submit(request("POST", "/detokenize", Server::gated_tokenization));
    gates.wait_entered(server.cpu_handler());
    const auto text = server.completion(auxiliary_text);
    const int text_slot = gates.wait_entered(server.gate(auxiliary_text));
    const auto chat = server.completion(auxiliary_chat);
    require(gates.wait_entered(server.gate(auxiliary_chat)) != text_slot,
            "A CPU tokenizer occupied a completion slot");
    std::optional<Socket> parser;
    {
        const auto abandoned = server.submit(request("POST", "/parse_function_call", "{}"), true);
        parser.emplace(
            server.submit(request("POST", "/separate_reasoning",
                                  R"({"reasoning_parser":"qwen3","text":"<think>r</think>x"})")));
        require(shutdown(parser->get(), SHUT_WR) == 0, "Cannot half-close the parser request");
        server.wait_queue(2, 0);
        pollfd waiting{parser->get(), POLLIN, 0};
        require(poll(&waiting, 1, 0) == 0, "A CPU parser bypassed the auxiliary FIFO");
        const auto full =
            read_closed(server.submit(request("POST", "/v1/messages/count_tokens", "{}")));
        require_status(full, "503");
        require(full.find(R"("type":"api_error")") != std::string::npos,
                "Auxiliary overload returned the wrong error envelope");
    }
    const auto tool_parser =
        server.submit(request("POST", "/parse_function_call",
                              R"({"tool_call_parser":"qwen3_coder","text":"unchanged"})"));
    server.control("/health");
    gates.release(server.cpu_handler());
    require_flush(tokenize, "200", R"({"text":""})");
    require_flush(*parser, "200", R"({"reasoning_text":"r","text":"x"})");
    require_status(read_closed(tool_parser), "200");
    gates.release(server.gate(auxiliary_text));
    gates.release(server.gate(auxiliary_chat));
    require_completion(text, auxiliary_text);
    require_completion(chat, auxiliary_chat);
    server.wait_queue(0, 0);

    std::optional<Socket> following;
    {
        const auto body = R"({"reasoning_parser":"qwen3","return_blocks":true,"text":")" +
                          std::string((2 * 1024 * 1024) - 128, 'x') + R"("})";
        const auto blocked =
            server.submit(request("POST", "/separate_reasoning", body), true, true);
        wait_readable(blocked, timeout);
        server.control("/health");
        following.emplace(server.submit(request("POST", "/detokenize", R"({"tokens":[]})")));
        server.control("/ready");
        pollfd pending{following->get(), POLLIN, 0};
        require(poll(&pending, 1, 0) == 0,
                "The large parser response did not stall its CPU worker");
    }
    require_flush(*following, "200", R"({"text":""})");
}

void check_generate_preparation(Server& server) {
    const auto text = server.completion(prepare_text);
    const int text_slot = gates.wait_entered(server.gate(prepare_text));
    const auto peer = server.completion(prepare_peer);
    gates.wait_entered(server.gate(prepare_peer));
    const auto waiter = server.completion(prepare_waiter);
    server.wait_queue(2, 1);
    {
        const auto cancelled = server.generate(true, true);
        gates.wait_entered(server.preparing());
        gates.release(server.gate(prepare_text));
        require_completion(text, prepare_text);
        require(gates.wait_entered(server.gate(prepare_waiter)) == text_slot,
                "A ready chat request could not take its freed slot during native preparation");
    }
    gates.release(server.preparing());
    server.wait_queue(2, 0);
    const auto waiting = server.generate(false);
    require_flush(server.abort_waiting(), "200", "");
    require_flush(waiting, "200", R"({"aborted":true})");
    server.wait_queue(2, 0);
    gates.release(server.gate(prepare_waiter));
    gates.release(server.gate(prepare_peer));
    require_completion(waiter, prepare_waiter);
    require_completion(peer, prepare_peer);
    server.wait_queue(0, 0);
}

void check_shutdown(Server& server, bool reader_failure) {
    const auto active_first = server.completion(last_first);
    gates.wait_entered(server.gate(last_first));
    const auto active_second = server.completion(last_second);
    gates.wait_entered(server.gate(last_second));
    const auto queued = server.completion(abandoned);
    server.wait_queue(2, 1);
    const auto active_cpu =
        server.submit(request("POST", "/detokenize", Server::shutdown_tokenization));
    gates.wait_entered(server.cpu_shutdown());
    const auto queued_cpu = server.submit(request("POST", "/separate_reasoning", "{}"));
    const auto partial = server.submit(incomplete_body);
    server.control("/health");
    if (reader_failure) {
        fail_accept = true;
    } else {
        stop_serving();
    }
    const auto wake = server.submit({});
    require(read_closed(queued).empty(), "Shutdown delivered a queued response");
    require(read_closed(queued_cpu).empty(), "Shutdown delivered a queued CPU response");
    require(read_closed(partial).empty(), "Shutdown did not close the incomplete request");
    gates.wait_released(restore_stop);
    require(!server.returned(), "Serving returned while completion gates were held");
    gates.release(server.gate(last_first));
    gates.release(server.gate(last_second));
    gates.release(server.cpu_shutdown());
    require(read_closed(active_cpu).empty(), "Active CPU request did not close during shutdown");
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
        require(argc == 2, "Expected a reader queue test mode");
        const std::string_view mode = argv[1];
        const bool reader_failure = mode == "--reader-failure";
        const bool broken_worker = mode == "--worker-failure";
        const bool flush = mode == "--flush";
        const bool cached = mode == "--cache-on" || reader_failure || flush;
        require(cached || broken_worker || mode == "--cache-off", "Unknown transport test mode");
        const auto descriptors = fd_count();
        BatchService service(std::make_unique<ReaderQueueExecutor>(), false);
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
            if (flush) {
                phase = "ordered cache flush";
                server.control("/server_info/", "404");
                server.control("/get_server_info/", "404");
                check_flush_queue(server);
                stop_serving();
                server.join();
            } else if (broken_worker) {
                phase = "worker failure";
                worker_failed = true;
                server.control("/health");
                require(server.control("/ready", "503").find("worker_failed") != std::string::npos,
                        "Unavailable worker omitted the readiness reason");
                require_status(read_closed(server.completion(terminal)), "500");
                require_flush(server.submit(request("POST", "/detokenize", R"({"tokens":[]})")),
                              "200", R"({"text":""})");
                stop_serving();
                server.join();
            } else {
                phase = "incomplete requests";
                check_readers(server);
                if (mode == "--cache-on") {
                    phase = "incomplete request deadline";
                    check_read_deadline(server);
                    phase = "request and descriptor limits";
                    check_read_limits(server);
                    phase = "independent auxiliary queue";
                    check_auxiliary_queue(server);
                    phase = "native generation preparation and abort";
                    check_generate_preparation(server);
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
