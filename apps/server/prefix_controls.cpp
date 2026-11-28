#include "sglang/server/prefix_controls.h"
#include "sglang/server/responses.h"

#include "sglang/json.h"
#include "sglang/server/batch_service.h"
#include "sglang/server/completions.h"
#include "sglang/server/flush_cache.h"
#include "sglang/server/http.h"
#include "sglang/server/loads.h"
#include "sglang/server/native_generate.h"
#include "sglang/server/ollama_discovery.h"
#include "sglang/server/parse_function_call.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <ctime>
#include <exception>
#include <iostream>
#include <list>
#include <mutex>
#include <thread>
#include <utility>

namespace sglang::server {
namespace {

using Clock = std::chrono::steady_clock;
constexpr auto generation_health_timeout = std::chrono::seconds(20);
constexpr auto flush_probe_retry = std::chrono::milliseconds(10);
constexpr auto flush_disconnect_poll = std::chrono::milliseconds(250);
constexpr std::string_view model_prefix = "/v1/models/";

struct Completion {
    Socket connection;
    std::optional<HttpRequest> request;
    QueuedGenerate generate = QueuedGenerate{};
};

struct AuxiliaryRequest {
    Socket connection;
    HttpRequest request;
};

struct HealthCheck {
    Socket connection;
    Clock::time_point started;
};

struct FlushCache {
    Socket connection;
    double timeout;
};

bool chat_route(const HttpRequest& request) {
    return request.route_path() == "/v1/chat/completions";
}

bool completion_route(const HttpRequest& request) {
    return request.method == "POST" &&
           (request.route_path() == "/v1/completions" || chat_route(request));
}

bool model_retrieval_route(std::string_view path) {
    return path.substr(0, model_prefix.size()) == model_prefix &&
           path.find('\n') == std::string_view::npos;
}

template <typename Handle>
void handle_errors(const Socket& connection, const Handle& handle) {
    try {
        handle();
    } catch (const HttpError& error) {
        write_error(connection, error.status, error.what());
    } catch (const CacheResetUnavailable& error) {
        write_error(connection, "503 Service Unavailable", error.what());
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
                    std::string_view health, bool prefix_controls, const BatchGenerate& generate,
                    const Tokenization& tokenize, std::string_view model_info,
                    const NativeAbort& abort, const ServerInfo& server_info,
                    const BatchScore& score, const OllamaDiscovery* discovery,
                    const BatchMessages& messages, const BatchResponses& responses,
                    ResponseStore* response_store, const BatchDecisions& decisions,
                    std::string_view accelerator, NativeGenerate* session_control)
        : service_(service), capacity_(capacity), complete_(complete), generate_(generate),
          tokenize_(tokenize), abort_(abort), health_(health), model_info_(model_info),
          server_info_(server_info), score_(score), discovery_(discovery), messages_(messages),
          responses_(responses), response_store_(response_store),
          decisions_(decisions), accelerator_(accelerator), session_control_(session_control),
          prefix_controls_(prefix_controls) {}

    void read(const Socket& listener);
    void run(int slot);
    void run_auxiliary();
    void run_aborted();
    void run_health();
    void run_flushes();
    void stop();

private:
    struct Lease {
        ConnectionQueue& queue;
        bool active = false;
        ~Lease();
    };

    struct Preparation {
        explicit Preparation(ConnectionQueue& owner) : queue(&owner) {
        }
        Preparation(const Preparation&) = delete;
        Preparation& operator=(const Preparation&) = delete;
        Preparation(Preparation&&) = delete;
        Preparation& operator=(Preparation&&) = delete;
        ConnectionQueue* queue;
        bool active = false;
        ~Preparation();
    };

    struct CacheReset {
        ConnectionQueue& queue;
        ~CacheReset();
    };

    void route(Socket& connection, HttpRequest request);
    std::optional<Completion> take(Lease& lease);
    bool server_info_route(std::string_view path) const {
        return server_info_ && (path == "/server_info" || path == "/get_server_info");
    }
    struct NativeRoute {
        std::string_view allow;
        bool full = false;
        bool after_discovery = false;
    };

    NativeRoute native_route(std::string_view path, std::string_view method) const;
    void enqueue(Socket& connection, HttpRequest request, bool generate, bool tokenization_errors,
                 QueuedGenerate prepared);
    void enqueue_auxiliary(Socket& connection, HttpRequest request);
    template <typename Work, typename... Args>
    bool enqueue_control(std::list<Work>& pending, std::condition_variable& ready,
                         Socket& connection, Args&&... args);
    template <typename Work>
    std::optional<Work> take_control(std::list<Work>& pending, std::condition_variable& ready);
    void serve_control(Socket& connection, const HttpRequest& request);
    void queue_health(Socket& connection, const HttpRequest& request);
    void queue_reset(Socket& connection, const HttpRequest& request);
    bool idle_for_cache_reset();
    void reset_cache(const Socket& connection);
    void queue_flush(Socket& connection, double timeout);
    FlushCacheResult flush_cache(const FlushCache& work);
    void discard_disconnected();

    BatchService& service_;
    const int capacity_;
    const BatchCompletion& complete_;
    const BatchGenerate& generate_;
    const Tokenization& tokenize_;
    const NativeAbort& abort_;
    const std::string_view health_;
    const std::string_view model_info_;
    const ServerInfo& server_info_;
    const BatchScore& score_;
    const OllamaDiscovery* discovery_;
    const BatchMessages& messages_;
    const BatchResponses& responses_;
    ResponseStore* response_store_;
    const BatchDecisions& decisions_;
    const std::string_view accelerator_;
    NativeGenerate* session_control_;
    const bool prefix_controls_;
    std::mutex mutex_;
    std::condition_variable changed_;
    std::condition_variable auxiliary_ready_;
    std::condition_variable abort_ready_;
    std::condition_variable health_ready_;
    std::condition_variable cache_ready_;
    std::list<Completion> pending_;
    std::list<AuxiliaryRequest> auxiliary_;
    std::list<HealthCheck> health_checks_;
    std::list<FlushCache> flushes_;
    std::optional<Socket> reset_;
    int running_ = 0;
    bool preparing_ = false;
    bool resetting_ = false;
    bool closing_ = false;
};

ConnectionQueue::Lease::~Lease() {
    if (active) {
        const std::lock_guard lock(queue.mutex_);
        if (--queue.running_ == 0) {
            queue.cache_ready_.notify_one();
        }
    }
}

ConnectionQueue::Preparation::~Preparation() {
    if (active) {
        {
            const std::scoped_lock lock(queue->mutex_);
            queue->preparing_ = false;
        }
        queue->cache_ready_.notify_one();
    }
}

void ConnectionQueue::discard_disconnected() {
    const bool had_pending = !pending_.empty();
    pending_.remove_if([](const auto& job) { return disconnected(job.connection); });
    if (had_pending && pending_.empty()) {
        cache_ready_.notify_one();
    }
}

bool ConnectionQueue::idle_for_cache_reset() {
    discard_disconnected();
    return pending_.empty() && running_ == 0 && !preparing_ && !resetting_;
}

ConnectionQueue::CacheReset::~CacheReset() {
    {
        const std::lock_guard lock(queue.mutex_);
        queue.resetting_ = false;
    }
    queue.changed_.notify_all();
    queue.cache_ready_.notify_one();
}

void ConnectionQueue::queue_reset(Socket& connection, const HttpRequest& request) {
    require_empty_body(request);
    {
        const std::lock_guard lock(mutex_);
        if (!idle_for_cache_reset()) {
            throw HttpError("409 Conflict", "Cannot reset prefix cache with active requests");
        }
        reset_.emplace(std::move(connection));
        resetting_ = true;
    }
    changed_.notify_one();
}

void ConnectionQueue::reset_cache(const Socket& connection) {
    const CacheReset reset{*this};
    handle_errors(connection, [&] {
        if (!disconnected(connection)) {
            if (!service_.reset_prefix_cache()) {
                throw HttpError("409 Conflict", "Cannot reset prefix cache with active requests");
            }
            write_json(connection, "200 OK", R"({"success":true})");
        }
    });
}

template <typename Work, typename... Args>
bool ConnectionQueue::enqueue_control(std::list<Work>& pending, std::condition_variable& ready,
                                      Socket& connection, Args&&... args) {
    const std::scoped_lock lock(mutex_);
    if (closing_ || disconnected(connection)) {
        return false;
    }
    pending.remove_if([](const auto& work) { return disconnected(work.connection); });
    if (pending.size() >= static_cast<std::size_t>(capacity_)) {
        return false;
    }
    pending.push_back({std::move(connection), std::forward<Args>(args)...});
    ready.notify_one();
    return true;
}

template <typename Work>
std::optional<Work> ConnectionQueue::take_control(std::list<Work>& pending,
                                                  std::condition_variable& ready) {
    std::unique_lock lock(mutex_);
    ready.wait(lock, [&] { return closing_ || !pending.empty(); });
    if (pending.empty()) {
        return std::nullopt;
    }
    std::optional<Work> work(std::in_place, std::move(pending.front()));
    pending.pop_front();
    return work;
}

void ConnectionQueue::queue_flush(Socket& connection, double timeout) {
    if (!enqueue_control(flushes_, cache_ready_, connection, timeout)) {
        throw HttpError("503 Service Unavailable", "Native cache flush queue is full");
    }
}

FlushCacheResult ConnectionQueue::flush_cache(const FlushCache& work) {
    using Seconds = std::chrono::duration<double>;
    const auto started = Clock::now();
    std::unique_lock lock(mutex_);
    while (!closing_ && !disconnected(work.connection)) {
        const bool idle = idle_for_cache_reset();
        if (idle) {
            resetting_ = true;
            lock.unlock();
            {
                const CacheReset reset{*this};
                if (service_.reset_prefix_cache()) {
                    return FlushCacheResult::flushed;
                }
            }
            lock.lock();
        }
        if (work.timeout == 0.0) {
            return FlushCacheResult::busy;
        }
        const auto remaining = Seconds(work.timeout) - Seconds(Clock::now() - started);
        if (remaining <= Seconds::zero()) {
            return FlushCacheResult::timed_out;
        }
        const auto wait_limit = idle ? flush_probe_retry : flush_disconnect_poll;
        cache_ready_.wait_for(lock, std::min(remaining, Seconds(wait_limit)));
    }
    return FlushCacheResult::busy;
}

void ConnectionQueue::run_flushes() {
    while (auto work = take_control(flushes_, cache_ready_)) {
        handle_errors(work->connection, [&] {
            const auto result = flush_cache(*work);
            if (!disconnected(work->connection)) {
                write_flush_cache(work->connection, result);
            }
        });
    }
}

std::string model_card() {
    std::string body = R"({"id":)";
    json::append_quoted(body, model_name);
    body += R"(,"object":"model","created":)" + std::to_string(std::time(nullptr));
    body += R"(,"owned_by":"sglang","root":)";
    json::append_quoted(body, model_name);
    body += R"(,"parent":null,"max_model_len":)" + std::to_string(checkpoint_context) + '}';
    return body;
}

void write_models(const Socket& connection) {
    write_json(connection, "200 OK", R"({"object":"list","data":[)" + model_card() + "]}");
}

void retrieve_model(const Socket& connection, const HttpRequest& request,
                    std::string_view model) {
    if (request.method != "GET") {
        write_get_only(connection, request.method == "HEAD");
        return;
    }
    if (model == model_name) {
        write_json(connection, "200 OK", model_card());
        return;
    }
    std::string body = R"({"error":{"message":)";
    json::append_quoted(body, "The model '" + std::string(model) + "' does not exist");
    body += R"(,"type":"invalid_request_error","param":"model","code":"model_not_found"}})";
    write_json(connection, "404 Not Found", body);
}

void ConnectionQueue::serve_control(Socket& connection, const HttpRequest& request) {
    const auto flush = read_flush_cache(connection, request);
    if (flush.matched) {
        if (flush.timeout) {
            queue_flush(connection, *flush.timeout);
        }
        return;
    }
    const auto path = request.route_path();
    if (path == "/health_generate") {
        queue_health(connection, request);
        return;
    }
    if (abort_ && request.method == "POST" && path == "/abort_request") {
        {
            const std::lock_guard lock(mutex_);
            discard_disconnected();
        }
        abort_(connection, request.body);
        const std::lock_guard lock(mutex_);
        abort_ready_.notify_one();
        return;
    }
    if (!model_info_.empty() &&
        (path == "/model_info" || path == "/get_model_info")) {
        if (request.method != "GET") {
            write_get_only(connection, request.method == "HEAD");
            return;
        }
        if (!request.body.empty()) {
            throw HttpError("404 Not Found", "Expected an empty model information request");
        }
        write_json(connection, "200 OK", model_info_);
        return;
    }
    if (model_retrieval_route(path)) {
        retrieve_model(connection, request, path.substr(model_prefix.size()));
        return;
    }
    if (request.method == "GET" && request.body.empty()) {
        if (path == "/health") {
            write_json(connection, "200 OK", health_);
            return;
        }
        if (path == "/ready") {
            write_readiness(connection, service_.ready());
            return;
        }
        if (path == "/v1/models") {
            write_models(connection);
            return;
        }
        if (prefix_controls_ && path == "/metrics") {
            int running = 0;
            std::size_t pending = 0;
            {
                const std::lock_guard lock(mutex_);
                discard_disconnected();
                running = running_;
                pending = pending_.size();
            }
            write_metrics(connection, format_metrics(service_.metrics(), running, pending));
            return;
        }
    }
    if (prefix_controls_ && request.method == "POST") {
        if (path == "/reset_prefix_cache") {
            queue_reset(connection, request);
            return;
        }
        if (path == "/start_profile") {
            start_profile(connection, service_, request);
            return;
        }
        if (path == "/stop_profile") {
            stop_profile(connection, service_, request);
            return;
        }
    }
    if (prefix_controls_) {
        throw HttpError("404 Not Found", "Unknown native server route");
    }
    std::string routes = "Expected GET /health, /ready, /health_generate, or /v1/models";
    if (!model_info_.empty()) {
        routes += " or /model_info or /get_model_info";
    }
    routes += ", or POST /v1/completions or /v1/chat/completions";
    if (generate_) {
        routes += " or /generate";
    }
    if (abort_) {
        routes += " or /abort_request";
    }
    if (tokenize_) {
        routes += " or /tokenize or /detokenize";
    }
    throw HttpError("404 Not Found", routes.c_str());
}

ConnectionQueue::NativeRoute ConnectionQueue::native_route(std::string_view path,
                                                           std::string_view method) const {
    if (const auto allowed = responses_ ? responses_method(path) : std::string_view{};
        !allowed.empty()) {
        return {allowed, method == allowed};
    }
    if (path == "/flush_cache" || (session_control_ && session_control_route(path))) {
        return {"GET, POST", method == "GET" || method == "POST"};
    }
    if (generate_ && path == "/generate") {
        return {"POST, PUT", method == "POST" || method == "PUT"};
    }
    if ((tokenize_ && path == anthropic_count_route) ||
        (messages_ && path == anthropic_messages_route) ||
        (decisions_ && path == systemone_route)) {
        return {"POST", method == "POST", true};
    }
    if (path == "/health" || path == "/health_generate" || path == "/ready" ||
        path == "/v1/models" || path == loads_route ||
        model_retrieval_route(path) || server_info_route(path) ||
        (!model_info_.empty() && (path == "/model_info" || path == "/get_model_info")) ||
        (prefix_controls_ && path == "/metrics")) {
        return {"GET", method == "GET"};
    }
    if (path == "/v1/completions" || path == "/v1/chat/completions" ||
        path == separate_reasoning_route || path == parse_function_call_route ||
        (abort_ && path == "/abort_request") || (score_ && path == score_route) ||
        (decisions_ && path == decisions_route) ||
        (tokenize_ && (path == "/tokenize" || path == "/v1/tokenize" ||
                       path == "/detokenize" || path == "/v1/detokenize")) ||
        (prefix_controls_ && (path == "/reset_prefix_cache" ||
                              path == "/start_profile" || path == "/stop_profile"))) {
        return {"POST", method == "POST"};
    }
    return {};
}

void ConnectionQueue::route(Socket& connection, HttpRequest request) {
    const auto native = native_route(request.route_path(), request.method);
    if (discovery_ && (!native.full || native.after_discovery)) {
        const auto earlier_allow = native.after_discovery ? std::string_view{} : native.allow;
        const auto route =
            discovery_->respond_exact(connection, request, native.full, earlier_allow);
        switch (route) {
        case OllamaDiscovery::Route::responded:
            return;
        case OllamaDiscovery::Route::chat:
        case OllamaDiscovery::Route::generate: {
            auto prepared = discovery_->prepare_generate(connection, request, route);
            if (prepared.complete)
                enqueue(connection, std::move(request), false, false, std::move(prepared));
            return;
        }
        case OllamaDiscovery::Route::unmatched:
            break;
        }
    }
    if (native.allow.empty()) {
        if (const std::string_view path = request.path; !path.empty() && path.back() == '/') {
            const auto canonical = path.substr(0, path.find_last_not_of('/') + 1);
            if (!native_route(matchable_route_path(canonical), request.method).allow.empty()) {
                write_redirect(connection, request, canonical);
                return;
            }
        }
        if (discovery_ && discovery_->respond_unmatched(connection, request)) return;
    }
    const auto path = request.route_path();
    if (session_control_ && session_control_route(path)) {
        session_control_->serve_session(connection, request);
        return;
    }
    if (path == loads_route) {
        std::unique_lock lock(mutex_);
        discard_disconnected();
        const LoadSnapshot snapshot{running_, pending_.size(), capacity_, accelerator_};
        lock.unlock();
        write_loads(connection, request, snapshot);
        return;
    }
    if (path == separate_reasoning_route || path == parse_function_call_route) {
        if (request.method == "POST") {
            enqueue_auxiliary(connection, std::move(request));
        } else {
            write_method_not_allowed(connection, "POST", request.method == "HEAD");
        }
        return;
    }
    const bool score = score_ && path == score_route;
    const bool decisions = decisions_ && (path == decisions_route || path == systemone_route);
    const bool count = tokenize_ && path == anthropic_count_route;
    const bool messages = messages_ && path == anthropic_messages_route;
    const bool responses = responses_ && path == responses_route;
    if (responses_ && !native.allow.empty() && response_store_ &&
        response_store_->serve_state(connection, request))
        return;
    if ((count || score || messages || responses || decisions) && request.method != "POST") {
        write_method_not_allowed(connection, "POST", request.method == "HEAD");
        return;
    }
    if (!request.body_is_application_json && (responses || messages || decisions)) {
        constexpr auto reason = "Unsupported Media Type: Only 'application/json' is allowed";
        if (responses) {
            responses_error(connection, 400, reason);
        } else if (messages) {
            write_anthropic_error(connection, "400 Bad Request", "invalid_request_error",
                                  std::string("header.content-type: ") + reason);
        } else {
            write_tokenization_error(connection, "400 Bad Request", "Bad Request", reason);
        }
        return;
    }
    if (server_info_route(request.route_path())) {
        if (request.method == "GET") {
            write_json(connection, "200 OK", server_info_());
        } else {
            write_get_only(connection, request.method == "HEAD");
        }
        return;
    }
    const auto native_route = generate_route(request);
    if (generate_ && native_route == GenerateRoute::method_not_allowed) {
        write_method_not_allowed(connection, "POST, PUT", request.method == "HEAD");
        return;
    }
    const bool generate = generate_ && native_route == GenerateRoute::allowed;
    if (tokenize_ && tokenization_route(request) != TokenizationRoute::none) {
        enqueue_auxiliary(connection, std::move(request));
        return;
    }
    if (!generate && !score && !messages && !responses && !decisions &&
        !completion_route(request)) {
        serve_control(connection, request);
        return;
    }
    enqueue(connection, std::move(request), generate, score || messages || decisions, {});
}

void write_queue_overload(const Socket& connection, bool anthropic) {
    if (anthropic) {
        write_anthropic_error(connection, "503 Service Unavailable", "api_error",
                              "Internal server error");
    } else {
        write_tokenization_error(connection, "503 Service Unavailable", "ServiceUnavailableError",
                                 "Native request queue is full");
    }
}

void ConnectionQueue::enqueue(Socket& connection, HttpRequest request, bool generate,
                              bool tokenization_errors, QueuedGenerate prepared) {
    if (!tokenization_errors) {
        service_.rethrow_failure();
    }
    if (prepared.complete) {
        request.body = std::string{};
    }
    Preparation preparation{*this};
    std::unique_lock lock(mutex_);
    if (closing_ || disconnected(connection)) {
        return;
    }
    discard_disconnected();
    const int idle_workers = resetting_ ? 0 : capacity_ - running_;
    const auto available =
        static_cast<std::size_t>(idle_workers) + static_cast<std::size_t>(capacity_);
    if (pending_.size() >= available) {
        lock.unlock();
        if (!tokenization_errors) {
            throw HttpError("503 Service Unavailable", "Native completion queue is full");
        }
        write_queue_overload(connection, request.route_path() == anthropic_messages_route);
        return;
    }
    if (generate) {
        preparation.active = preparing_ = true;
        lock.unlock();
        prepared = generate_(request.body);
        if (prepared.complete) {
            request.body = std::string{};
        }
        lock.lock();
        if (closing_ || disconnected(connection)) {
            return;
        }
    }
    pending_.push_back({std::move(connection), std::move(request), std::move(prepared)});
    if (preparation.active) {
        preparing_ = false;
        preparation.active = false;
    }
    lock.unlock();
    changed_.notify_one();
}

void ConnectionQueue::enqueue_auxiliary(Socket& connection, HttpRequest request) {
    const bool anthropic = request.route_path() == anthropic_count_route;
    if (!enqueue_control(auxiliary_, auxiliary_ready_, connection, std::move(request))) {
        write_queue_overload(connection, anthropic);
    }
}

void ConnectionQueue::read(const Socket& listener) {
    read_connections(listener, capacity_ + 1, [&](Socket& connection, HttpRequest request) {
        handle_errors(connection, [&] { route(connection, std::move(request)); });
    }, [&](const HttpRequest& request) {
        return discovery_ && discovery_->overrides_count(request);
    });
}

std::optional<Completion> ConnectionQueue::take(Lease& lease) {
    std::unique_lock lock(mutex_);
    for (;;) {
        changed_.wait(lock, [&] {
            return closing_ || reset_ || (!resetting_ && !pending_.empty());
        });
        if (closing_ || stopping()) {
            return std::nullopt;
        }
        if (reset_) {
            Completion work{std::move(*reset_), std::nullopt};
            reset_.reset();
            return work;
        }
        discard_disconnected();
        if (!pending_.empty()) {
            auto work = std::move(pending_.front());
            pending_.pop_front();
            ++running_;
            lease.active = true;
            return work;
        }
    }
}

void ConnectionQueue::run(int slot) {
    for (;;) {
        Lease lease{*this};
        auto work = take(lease);
        if (!work) {
            return;
        }
        if (!work->request) {
            reset_cache(work->connection);
        } else if (!disconnected(work->connection)) {
            const auto& request = *work->request;
            handle_errors(work->connection, [&] {
                const auto path = request.route_path();
                if (work->generate.complete) {
                    const auto generate = std::move(work->generate);
                    generate.complete(work->connection, slot);
                } else if (score_ && path == score_route) {
                    score_(work->connection, slot, request.body);
                } else if (messages_ && path == anthropic_messages_route) {
                    messages_(work->connection, slot, request.body);
                } else if (responses_ && path == responses_route) {
                    responses_(work->connection, slot, request.body);
                } else if (decisions_ && (path == decisions_route || path == systemone_route)) {
                    decisions_(work->connection, slot, request.body, path == systemone_route);
                } else {
                    complete_(work->connection, slot, request.body, chat_route(request));
                }
            });
        }
    }
}

void ConnectionQueue::run_auxiliary() {
    while (auto work = take_control(auxiliary_, auxiliary_ready_)) {
        if (disconnected(work->connection)) {
            continue;
        }
        handle_errors(work->connection, [&] {
            const auto path = work->request.route_path();
            if (path == separate_reasoning_route) {
                separate_reasoning(work->connection, work->request.body);
            } else if (path == parse_function_call_route) {
                parse_function_call(work->connection, work->request.body);
            } else {
                tokenize_(work->connection, tokenization_route(work->request), work->request.body);
            }
        });
    }
}

void ConnectionQueue::run_aborted() {
    for (;;) {
        std::unique_lock lock(mutex_);
        const auto find_aborted = [&] {
            return std::find_if(pending_.begin(), pending_.end(), [](const auto& item) {
                return item.generate.aborted && item.generate.aborted();
            });
        };
        auto found = pending_.end();
        abort_ready_.wait(lock, [&] {
            return closing_ || (found = find_aborted()) != pending_.end();
        });
        if (closing_) {
            return;
        }
        auto work = std::move(*found);
        pending_.erase(found);
        if (pending_.empty()) {
            cache_ready_.notify_one();
        }
        lock.unlock();
        if (!disconnected(work.connection)) {
            handle_errors(work.connection, [&] {
                const auto generate = std::move(work.generate);
                generate.complete(work.connection, std::nullopt);
            });
        }
    }
}

void ConnectionQueue::queue_health(Socket& connection, const HttpRequest& request) {
    if (request.method != "GET") {
        write_get_only(connection, request.method == "HEAD");
        return;
    }
    if (!request.body.empty()) {
        throw HttpError("404 Not Found", "Expected an empty generation health request");
    }
    const auto started = Clock::now();
    if (!service_.ready() || !enqueue_control(health_checks_, health_ready_, connection, started)) {
        write_generation_health(connection, false);
    }
}

void ConnectionQueue::run_health() {
    while (auto work = take_control(health_checks_, health_ready_)) {
        const auto cancelled = [&] { return disconnected(work->connection); };
        const bool healthy = service_.wait_for_generation(
            work->started, work->started + generation_health_timeout, cancelled);
        write_generation_health(work->connection, healthy && service_.ready());
    }
}

void ConnectionQueue::stop() {
    {
        const std::lock_guard lock(mutex_);
        closing_ = true;
        pending_.clear();
        auxiliary_.clear();
        flushes_.clear();
        reset_.reset();
    }
    changed_.notify_all();
    auxiliary_ready_.notify_one();
    abort_ready_.notify_one();
    health_ready_.notify_one();
    cache_ready_.notify_one();
}

}

void serve_connections(const Socket& listener, BatchService& service, int max_active,
                       const BatchCompletion& complete, std::string_view health,
                       bool prefix_controls, const BatchGenerate& generate,
                       const Tokenization& tokenize, std::string_view model_info,
                       const NativeAbort& abort, const ServerInfo& server_info,
                       const BatchScore& score, const OllamaDiscovery* discovery,
                       const BatchMessages& messages, const BatchResponses& responses,
                       ResponseStore* response_store, const BatchDecisions& decisions,
                       std::string_view accelerator, NativeGenerate* session_control) {
    if (max_active < 1 || max_active > decoder_batch_max_lanes) {
        throw std::invalid_argument("Invalid HTTP completion capacity");
    }
    ConnectionQueue queue(service, max_active, complete, health, prefix_controls, generate,
                          tokenize, model_info, abort, server_info, score, discovery, messages,
                          responses, response_store, decisions, accelerator, session_control);
    std::vector<std::thread> workers;
    workers.reserve(max_active + (abort ? 4 : 3));
    std::mutex failure_mutex;
    std::exception_ptr failure;
    const auto stop_on_error = [&] {
        const std::lock_guard lock(failure_mutex);
        if (!failure) {
            failure = std::current_exception();
        }
        stop_serving();
    };
    const auto start_worker = [&](auto member, auto... args) {
        workers.emplace_back([&, member, args...] {
            try {
                (queue.*member)(args...);
            } catch (...) {
                stop_on_error();
            }
        });
    };
    try {
        start_worker(&ConnectionQueue::run_health);
        start_worker(&ConnectionQueue::run_flushes);
        start_worker(&ConnectionQueue::run_auxiliary);
        if (abort) {
            start_worker(&ConnectionQueue::run_aborted);
        }
        for (int slot = 0; slot < max_active; ++slot) {
            start_worker(&ConnectionQueue::run, slot);
        }
        queue.read(listener);
    } catch (...) {
        stop_on_error();
    }
    queue.stop();
    service.stop_cache_restores();
    for (auto& worker : workers) {
        worker.join();
    }
    if (failure) {
        std::rethrow_exception(failure);
    }
}

}
