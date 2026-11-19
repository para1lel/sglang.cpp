#include "sglang/server/batch_service.h"
#include "sglang/server/http.h"
#include "sglang/server/prefix_controls.h"

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <exception>
#include <filesystem>
#include <iostream>
#include <memory>
#include <mutex>
#include <poll.h>
#include <sys/socket.h>
#include <thread>
#include <utility>

namespace sglang {

struct Decoder::Model {};
Decoder::Decoder(const std::filesystem::path&, int) {}
Decoder::~Decoder() = default;

namespace server {

struct BatchService::State {
    std::mutex mutex;
    std::array<bool, decoder_batch_max_lanes> claimed{};
};

BatchService::BatchService(const Decoder&, bool, bool, int, int, int)
    : state_(std::make_unique<State>()) {}
BatchService::~BatchService() = default;

void BatchService::claim_connection(int slot) {
    const std::lock_guard lock(state_->mutex);
    if (std::exchange(state_->claimed.at(slot), true)) {
        throw std::logic_error("Slot already claimed");
    }
}

void BatchService::release_connection(int slot) {
    const std::lock_guard lock(state_->mutex);
    if (!std::exchange(state_->claimed.at(slot), false)) {
        throw std::logic_error("Slot was not claimed");
    }
}

BatchMetrics BatchService::metrics() const {
    const std::lock_guard lock(state_->mutex);
    BatchMetrics metrics;
    for (bool claimed : state_->claimed) {
        metrics.running += claimed;
    }
    return metrics;
}

bool BatchService::reset_prefix_cache() { return metrics().running == 0; }
bool BatchService::start_profile(std::string, std::size_t) { return false; }
std::optional<BatchProfile> BatchService::stop_profile() { return std::nullopt; }

}
}

namespace {

using namespace std::chrono_literals;
using namespace sglang::server;
using Clock = std::chrono::steady_clock;
constexpr auto timeout = 3s;
constexpr int capacity = 2;
constexpr auto reader_failure = "Injected reader accept failure";

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

struct Gate {
    std::size_t entered = 0;
    bool released = false;
};

thread_local Gate* reader_gate = nullptr;

class Transport {
public:
    Socket submit(std::string_view method, std::string_view path, std::string_view body = "",
                  Gate* gate = nullptr, bool extra_reader = false) {
        int descriptors[2];
        require(socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, descriptors) == 0,
                "Cannot create socket pair");
        Socket client(descriptors[0]);
        Socket server(descriptors[1]);
        std::string request = std::string(method) + " " + std::string(path) +
            " HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: " +
            std::to_string(body.size()) + "\r\n\r\n" + std::string(body);
        require(send(client.get(), request.data(), request.size(), MSG_NOSIGNAL) ==
                    static_cast<ssize_t>(request.size()), "Cannot submit complete request");
        {
            const std::lock_guard lock(mutex_);
            incoming_.push_back({std::move(server), gate, extra_reader});
        }
        changed_.notify_all();
        return client;
    }

    Socket accept() {
        std::unique_lock lock(mutex_);
        const auto available = [&] {
            return !incoming_.empty() &&
                (!incoming_.front().extra_reader || caller_ != std::this_thread::get_id());
        };
        const auto should_fail = [&] {
            return fail_accept_ && caller_ != std::this_thread::get_id();
        };
        changed_.wait_for(lock, 20ms, [&] { return stopping() || should_fail() || available(); });
        if (stopping()) {
            return Socket(-1);
        }
        if (should_fail()) {
            fail_accept_ = false;
            throw std::runtime_error(reader_failure);
        }
        if (!available()) {
            return Socket(-1);
        }
        auto ticket = std::move(incoming_.front());
        incoming_.pop_front();
        reader_gate = ticket.gate;
        return std::move(ticket.socket);
    }

    void identify_caller() {
        const std::lock_guard lock(mutex_);
        caller_ = std::this_thread::get_id();
    }

    void fail_next_accept() {
        {
            const std::lock_guard lock(mutex_);
            fail_accept_ = true;
        }
        changed_.notify_all();
    }

    void hold(Gate& gate) {
        std::unique_lock lock(mutex_);
        hold(lock, gate);
    }

    void wait_entered(Gate& gate, std::size_t count = 1,
                      const char* message = "Expected reader or completion to enter its gate") {
        std::unique_lock lock(mutex_);
        require(changed_.wait_for(lock, timeout, [&] { return gate.entered == count; }), message);
    }

    void release(Gate& gate) {
        {
            const std::lock_guard lock(mutex_);
            gate.released = true;
        }
        changed_.notify_all();
    }

    void stop() {
        stop_serving();
        changed_.notify_all();
    }

    void release_all() {
        {
            const std::lock_guard lock(mutex_);
            release_all_ = true;
        }
        changed_.notify_all();
    }

private:
    struct Ticket {
        Socket socket;
        Gate* gate;
        bool extra_reader;
    };

    void hold(std::unique_lock<std::mutex>& lock, Gate& gate) {
        ++gate.entered;
        changed_.notify_all();
        changed_.wait(lock, [&] { return gate.released || release_all_; });
    }

    std::mutex mutex_;
    std::condition_variable changed_;
    std::deque<Ticket> incoming_;
    std::thread::id caller_;
    bool fail_accept_ = false;
    bool release_all_ = false;
};

Transport transport;

std::string read_closed(const Socket& socket) {
    std::string response;
    const auto deadline = Clock::now() + timeout;
    for (;;) {
        const auto remaining =
            std::chrono::ceil<std::chrono::milliseconds>(deadline - Clock::now());
        pollfd ready{socket.get(), POLLIN, 0};
        require(remaining.count() > 0 && poll(&ready, 1, static_cast<int>(remaining.count())) == 1,
                "Timed out waiting for HTTP response or peer closure");
        char buffer[4096];
        const auto count = recv(socket.get(), buffer, sizeof(buffer), 0);
        require(count >= 0, "Cannot read HTTP response");
        if (count == 0) {
            return response;
        }
        response.append(buffer, static_cast<std::size_t>(count));
    }
}

void require_status(const std::string& response, std::string_view status) {
    const auto prefix = "HTTP/1.1 " + std::string(status) + " ";
    if (response.compare(0, prefix.size(), prefix) != 0) {
        throw std::runtime_error("Unexpected HTTP response: " + response);
    }
}

std::string request_body(int id, bool chat) {
    if (chat) {
        return R"({"model":"Qwen3.8-27B-FP8","max_tokens":1,"messages":[{"role":"user","content":")"
            + std::to_string(id) + R"("}]})";
    }
    return R"({"model":"benchmark","max_tokens":1,"prompt":[)" + std::to_string(id) + "]}";
}

Socket completion(int id, Gate* gate = nullptr, bool chat = false) {
    return transport.submit("POST", chat ? "/v1/chat/completions" : "/v1/completions",
                            request_body(id, chat), gate);
}

void require_completion(const Socket& client, int id) {
    const auto response = read_closed(client);
    require_status(response, "200");
    const auto body = "\r\n\r\n{\"request\":" + std::to_string(id) + "}";
    require(response.size() >= body.size() &&
            response.compare(response.size() - body.size(), body.size(), body) == 0,
            "Completion response reached the wrong peer");
}

void wait_for_queue(int running, int pending) {
    const auto deadline = Clock::now() + timeout;
    const auto expected_running = "vllm:num_requests_running{engine=\"0\"} " +
        std::to_string(running) + '\n';
    const auto expected_pending = "vllm:num_requests_waiting{engine=\"0\"} " +
        std::to_string(pending) + '\n';
    do {
        const auto metrics = read_closed(transport.submit("GET", "/metrics"));
        require_status(metrics, "200");
        if (metrics.find(expected_running) != std::string::npos &&
            metrics.find(expected_pending) != std::string::npos) {
            return;
        }
    } while (Clock::now() < deadline);
    throw std::runtime_error("Queue metrics did not reach the expected state");
}

struct Requests {
    enum { first_active, second_active, first_queued, plain, chat, last_active, pending, count };
    std::array<Gate, count> done;
    Gate contenders;
    Gate retiring_reader;

    static int token(int id) { return id + 101; }

    void complete(const Socket& socket, int slot, std::string_view body, bool is_chat) {
        require(slot >= 0 && slot < capacity, "Completion received a foreign slot");
        for (int id = 0; id < count; ++id) {
            if ((id == chat || id == first_queued) == is_chat &&
                body == request_body(token(id), is_chat)) {
                transport.hold(done[id]);
                write_json(socket, "200 OK", "{\"request\":" + std::to_string(token(id)) + "}");
                return;
            }
        }
        throw std::logic_error("Completion received the wrong request or route");
    }
};

class Server {
public:
    Server(BatchService& service, Requests& requests) : worker_([&, this] {
        transport.identify_caller();
        try {
            serve_prefix_connections(Socket(-1), service, capacity,
                [&](const Socket& socket, int slot, std::string_view body, bool chat) {
                    requests.complete(socket, slot, body, chat);
                });
        } catch (...) {
            failure_ = std::current_exception();
        }
        returned_ = true;
    }) {}

    ~Server() {
        if (worker_.joinable()) {
            transport.stop();
            transport.release_all();
            worker_.join();
        }
    }

    bool returned() const { return returned_.load(); }

    void join() {
        worker_.join();
        if (failure_) {
            std::rethrow_exception(failure_);
        }
    }

private:
    std::exception_ptr failure_;
    std::atomic<bool> returned_{false};
    std::thread worker_;
};

int rejected_contender(const Socket& plain, const Socket& chat) {
    pollfd ready[] = {{plain.get(), POLLIN, 0}, {chat.get(), POLLIN, 0}};
    require(poll(ready, 2, std::chrono::milliseconds(timeout).count()) == 1,
            "Expected exactly one rejection with one pending position left");
    const bool plain_rejected = ready[0].revents != 0;
    require_status(read_closed(plain_rejected ? plain : chat), "503");
    return plain_rejected ? Requests::plain : Requests::chat;
}

void check_queue(Requests& requests, const Socket& first_active) {
    const auto first_queued = completion(Requests::token(Requests::first_queued), nullptr, true);
    wait_for_queue(2, 1);
    const auto plain = completion(Requests::token(Requests::plain), &requests.contenders);
    const auto chat = completion(Requests::token(Requests::chat), &requests.contenders, true);
    transport.wait_entered(requests.contenders, 2, "Both contenders must reach the reader barrier");
    require_status(read_closed(transport.submit("GET", "/health")), "200");
    require_status(read_closed(transport.submit("POST", "/reset_prefix_cache", "{}")), "409");
    transport.release(requests.contenders);
    const auto rejected = rejected_contender(plain, chat);
    const auto accepted = rejected == Requests::plain ? Requests::chat : Requests::plain;
    wait_for_queue(2, 2);

    transport.release(requests.done[Requests::first_active]);
    require_completion(first_active, Requests::token(Requests::first_active));
    transport.wait_entered(requests.done[Requests::first_queued]);
    transport.release(requests.done[Requests::first_queued]);
    require_completion(first_queued, Requests::token(Requests::first_queued));
    transport.wait_entered(requests.done[accepted]);
    transport.release(requests.done[accepted]);
    require_completion(accepted == Requests::plain ? plain : chat, Requests::token(accepted));
    wait_for_queue(1, 0);
}

void check_shutdown(Requests& requests, Server& server, const Socket& second_active,
                    bool inject_failure) {
    const auto last_active = completion(Requests::token(Requests::last_active));
    transport.wait_entered(requests.done[Requests::last_active]);
    const auto pending = completion(Requests::token(Requests::pending));
    wait_for_queue(2, 1);
    const auto reader = transport.submit("GET", "/health", {}, &requests.retiring_reader, true);
    transport.wait_entered(requests.retiring_reader);
    if (inject_failure) {
        transport.fail_next_accept();
    } else {
        transport.stop();
    }
    require(read_closed(pending).empty(), "Shutdown dispatched a queued request");
    require(!server.returned(), "Serving returned while completion and reader gates were held");
    transport.release(requests.done[Requests::second_active]);
    require(read_closed(second_active).empty(), "Completion did not close during shutdown");
    transport.release(requests.done[Requests::last_active]);
    require(read_closed(last_active).empty(), "Completion did not close during shutdown");
    require(!server.returned(), "Serving returned while a reader gate was held");
    transport.release(requests.retiring_reader);
    std::string failure;
    try {
        server.join();
    } catch (const std::runtime_error& error) {
        failure = error.what();
    }
    require(failure == (inject_failure ? reader_failure : ""), "Unexpected serve failure");
    require(read_closed(reader).empty(), "Stopped reader replied to an unprocessed request");
}

}

extern "C" {

Socket __wrap__ZN6sglang6server17accept_connectionERKNS0_6SocketE(const Socket&) {
    return transport.accept();
}

std::optional<HttpRequest> __real__ZN6sglang6server12read_requestERKNS0_6SocketE(const Socket&);

std::optional<HttpRequest>
__wrap__ZN6sglang6server12read_requestERKNS0_6SocketE(const Socket& socket) {
    auto request = __real__ZN6sglang6server12read_requestERKNS0_6SocketE(socket);
    if (auto* gate = std::exchange(reader_gate, nullptr); request && gate) {
        transport.hold(*gate);
    }
    return request;
}

}

int main(int argc, char** argv) {
    try {
        const bool inject_failure = argc == 2 && std::string_view(argv[1]) == "--reader-failure";
        require(argc == 1 || inject_failure, "Expected optional --reader-failure");
        const sglang::Decoder decoder(std::filesystem::path{});
        BatchService service(decoder, false);
        Requests requests;
        Server server(service, requests);
        const auto first_active = completion(Requests::token(Requests::first_active));
        const auto second_active = completion(Requests::token(Requests::second_active));
        transport.wait_entered(requests.done[Requests::first_active]);
        transport.wait_entered(requests.done[Requests::second_active]);
        check_queue(requests, first_active);
        check_shutdown(requests, server, second_active, inject_failure);
        require(service.metrics().running == 0, "Slot remained claimed after shutdown");
        for (const int invalid : {0, sglang::decoder_batch_max_lanes + 1}) {
            bool rejected = false;
            try {
                serve_prefix_connections(Socket(-1), service, invalid, {});
            } catch (const std::invalid_argument&) {
                rejected = true;
            }
            require(rejected, "Invalid listener capacity was accepted");
        }
        std::cout << "prefix reader queue passed\n";
    } catch (const std::exception& error) {
        std::cerr << "prefix reader queue failed: " << error.what() << '\n';
        return 1;
    }
}
