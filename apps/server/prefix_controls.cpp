#include "sglang/server/prefix_controls.h"

#include "sglang/json.h"
#include "sglang/mxfp8_gqa.h"
#include "sglang/server/batch_service.h"
#include "sglang/server/completions.h"
#include "sglang/server/http.h"

#include <condition_variable>
#include <exception>
#include <iostream>
#include <list>
#include <mutex>
#include <thread>

namespace sglang::server {
namespace {

struct Completion {
    Socket connection;
    HttpRequest request;
};

bool chat_route(const HttpRequest& request) { return request.path == "/v1/chat/completions"; }

bool completion_route(const HttpRequest& request) {
    return request.method == "POST" &&
           (request.path == "/v1/completions" || chat_route(request));
}

template <typename Handle>
void handle_errors(const Socket& connection, const Handle& handle) {
    try {
        handle();
    } catch (const HttpError& error) {
        write_error(connection, error.status, error.what());
    } catch (const DecoderCapacityError& error) {
        const char* status = error.retryable ? "503 Service Unavailable" : "413 Content Too Large";
        write_error(connection, status, error.what());
    } catch (const std::invalid_argument& error) {
        write_error(connection, "400 Bad Request", error.what());
    } catch (const std::exception& error) {
        std::cerr << "Request failed: " << error.what() << '\n';
        write_error(connection, "500 Internal Server Error", "Native request failed");
    }
}

std::string format_metrics(const BatchMetrics& metrics, std::size_t waiting) {
    std::string text = "# Native sglang.cpp metrics with frozen W3-prefix compatibility names\n";
    const auto emit = [&text](std::string_view name, auto value) {
        text += name;
        text += "{engine=\"0\"} ";
        text += std::to_string(value);
        text += '\n';
    };
    const int capacity = metrics.cache_capacity * (decoder_batch_prefix_max_tokens /
                                                   mxfp8_page_size);
    emit("vllm:num_requests_running", metrics.running);
    emit("vllm:num_requests_waiting", waiting);
    emit("vllm:kv_cache_usage_perc", capacity ? double(metrics.cache.pages) / capacity : 0.0);
    emit("vllm:prompt_tokens_total", metrics.prompt_tokens);
    emit("vllm:generation_tokens_total", metrics.generation_tokens);
    emit("vllm:external_prefix_cache_hits_total", 0);
    emit("vllm:prefix_cache_hits_total", metrics.cache.cached_tokens);
    emit("vllm:request_prefill_kv_computed_tokens_count", metrics.prefill_requests);
    emit("vllm:request_prefill_kv_computed_tokens_sum", metrics.cache.prefill_tokens);
    emit("sglang_cpp:prefix_cache_entries", metrics.cache.entries);
    emit("sglang_cpp:prefix_cache_pages", metrics.cache.pages);
    emit("sglang_cpp:prefix_cache_device_bytes", metrics.cache.device_bytes);
    emit("sglang_cpp:prefix_cache_hits_total", metrics.cache.hits);
    emit("sglang_cpp:prefix_cache_stores_total", metrics.cache.stores);
    emit("sglang_cpp:prefix_cache_evictions_total", metrics.cache.evictions);
    return text;
}

json::Value control_body(const HttpRequest& request) {
    try {
        return json::parse(request.body);
    } catch (const json::ParseError&) {
        throw HttpError("400 Bad Request", "Expected a JSON control object");
    }
}

void require_empty_body(const HttpRequest& request) {
    const auto body = control_body(request);
    if (!body.object() || !body.object()->empty()) {
        throw HttpError("400 Bad Request", "Expected an empty JSON control object");
    }
}

void start_profile(const Socket& connection, BatchService& service, const HttpRequest& request) {
    const auto body = control_body(request);
    const auto* object = body.object();
    const auto* limit = object ? json::get(*object, "max_iterations") : nullptr;
    const auto* options = object ? json::get(*object, "profiler_kwargs") : nullptr;
    const auto iterations = limit ? limit->unsigned_integer() : std::nullopt;
    const auto name = object ? json::get_string(*object, "profile_prefix") : std::nullopt;
    const auto* kwargs = options ? options->object() : nullptr;
    if (!object || object->size() != 3 || !iterations || *iterations < 1 || *iterations > 64 ||
        !name || name->empty() || name->size() > 64 ||
        !kwargs || kwargs->size() != 1 || json::get_boolean(*kwargs, "with_stack") != false) {
        throw HttpError("400 Bad Request", "Expected a bounded native profile without stacks");
    }
    if (!service.start_profile(std::string(*name), *iterations)) {
        throw HttpError("409 Conflict", "A native profile is already active");
    }
    write_json(connection, "200 OK", R"({"success":true,"profiler":"native-batch-steps"})");
}

void stop_profile(const Socket& connection, BatchService& service, const HttpRequest& request) {
    require_empty_body(request);
    const auto profile = service.stop_profile();
    if (!profile) {
        throw HttpError("409 Conflict", "No native profile is active");
    }
    std::string report = R"({"success":true,"profiler":"native-batch-steps","profile_prefix":)";
    json::append_quoted(report, profile->name);
    report += R"(,"steps":[)";
    for (const auto& step : profile->steps) {
        if (&step != &profile->steps.front()) {
            report += ',';
        }
        report += R"({"elapsed_ns":)" + std::to_string(step.elapsed_ns);
        report += R"(,"width":)" + std::to_string(step.width);
        report += R"(,"prefill_rows":)" + std::to_string(step.prefill_rows);
        report += R"(,"decode_lanes":)" + std::to_string(step.decode_lanes);
        report += R"(,"graph_slots":)" + std::to_string(step.graph_slots) + '}';
    }
    write_json(connection, "200 OK", report + "]}");
}

class ConnectionQueue {
public:
    ConnectionQueue(BatchService& service, int capacity, const BatchCompletion& complete,
                    std::string_view health)
        : service_(service), capacity_(capacity), complete_(complete), health_(health) {}

    void read_connections(const Socket& listener);
    void run(int slot);
    void stop();

private:
    void route(Socket& connection, HttpRequest request);
    std::optional<Completion> take(int slot);
    void serve_control(const Socket& connection, const HttpRequest& request);
    void discard_disconnected();

    BatchService& service_;
    const int capacity_;
    const BatchCompletion& complete_;
    const std::string_view health_;
    std::mutex mutex_;
    std::condition_variable changed_;
    std::list<Completion> pending_;
    bool closing_ = false;
};

void ConnectionQueue::discard_disconnected() {
    pending_.remove_if([](const auto& job) { return disconnected(job.connection); });
}

void ConnectionQueue::serve_control(const Socket& connection, const HttpRequest& request) {
    if (request.method == "GET" && request.body.empty()) {
        if (request.path == "/health") {
            write_json(connection, "200 OK", health_);
            return;
        }
        if (request.path == "/ready") {
            write_readiness(connection, service_.ready());
            return;
        }
        if (request.path == "/metrics") {
            std::string body;
            {
                const std::lock_guard lock(mutex_);
                discard_disconnected();
                body = format_metrics(service_.metrics(), pending_.size());
            }
            write_metrics(connection, body);
            return;
        }
    }
    if (request.method == "POST") {
        if (request.path == "/reset_prefix_cache") {
            require_empty_body(request);
            {
                const std::lock_guard lock(mutex_);
                discard_disconnected();
                if (!pending_.empty() || !service_.reset_prefix_cache()) {
                    throw HttpError("409 Conflict",
                                    "Cannot reset prefix cache with active requests");
                }
            }
            write_json(connection, "200 OK", R"({"success":true})");
            return;
        }
        if (request.path == "/start_profile") {
            start_profile(connection, service_, request);
            return;
        }
        if (request.path == "/stop_profile") {
            stop_profile(connection, service_, request);
            return;
        }
    }
    throw HttpError("404 Not Found", "Unknown native server route");
}

void ConnectionQueue::route(Socket& connection, HttpRequest request) {
    if (!completion_route(request)) {
        serve_control(connection, request);
        return;
    }
    service_.rethrow_failure();
    {
        const std::lock_guard lock(mutex_);
        if (closing_) {
            return;
        }
        discard_disconnected();
        if (pending_.size() >= static_cast<std::size_t>(capacity_)) {
            throw HttpError("503 Service Unavailable", "Native completion queue is full");
        }
        pending_.push_back({std::move(connection), std::move(request)});
    }
    changed_.notify_one();
}

void ConnectionQueue::read_connections(const Socket& listener) {
    while (!stopping()) {
        auto connection = accept_connection(listener);
        if (connection.get() >= 0) {
            handle_errors(connection, [&] {
                if (auto request = read_request(connection); request && !stopping()) {
                    route(connection, std::move(*request));
                }
            });
        }
    }
}

std::optional<Completion> ConnectionQueue::take(int slot) {
    std::unique_lock lock(mutex_);
    changed_.wait(lock, [&] { return closing_ || !pending_.empty(); });
    while (!closing_ && !pending_.empty()) {
        auto job = std::move(pending_.front());
        pending_.pop_front();
        if (!disconnected(job.connection)) {
            service_.claim_connection(slot);
            return job;
        }
    }
    return std::nullopt;
}

void ConnectionQueue::run(int slot) {
    for (;;) {
        auto job = take(slot);
        if (!job) {
            const std::lock_guard lock(mutex_);
            if (closing_) {
                return;
            }
            continue;
        }
        handle_errors(job->connection, [&] {
            complete_(job->connection, slot, job->request.body, chat_route(job->request));
        });
        service_.release_connection(slot);
    }
}

void ConnectionQueue::stop() {
    {
        const std::lock_guard lock(mutex_);
        closing_ = true;
        pending_.clear();
    }
    changed_.notify_all();
}

}

void serve_prefix_connections(const Socket& listener, BatchService& service, int max_active,
                              const BatchCompletion& complete, std::string_view health) {
    if (max_active < 1 || max_active > decoder_batch_max_lanes) {
        throw std::invalid_argument("Invalid prefix listener capacity");
    }
    ConnectionQueue queue(service, max_active, complete, health);
    std::vector<std::thread> workers;
    workers.reserve(2 * max_active);
    std::mutex failure_mutex;
    std::exception_ptr failure;
    const auto stop_on_error = [&] {
        const std::lock_guard lock(failure_mutex);
        if (!failure) {
            failure = std::current_exception();
        }
        stop_serving();
    };
    try {
        for (int slot = 0; slot < max_active; ++slot) {
            workers.emplace_back([&, slot] {
                try {
                    queue.run(slot);
                } catch (...) {
                    stop_on_error();
                }
            });
        }
        for (int reader = 0; reader < max_active; ++reader) {
            workers.emplace_back([&] {
                try {
                    queue.read_connections(listener);
                } catch (...) {
                    stop_on_error();
                }
            });
        }
        queue.read_connections(listener);
    } catch (...) {
        stop_on_error();
    }
    queue.stop();
    for (auto& worker : workers) {
        worker.join();
    }
    if (failure) {
        std::rethrow_exception(failure);
    }
}

}
