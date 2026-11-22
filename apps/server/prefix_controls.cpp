#include "sglang/server/prefix_controls.h"
#include "sglang/server/responses.h"

#include "sglang/json.h"
#include "sglang/server/batch_service.h"
#include "sglang/server/chat_output.h"
#include "sglang/server/completions.h"
#include "sglang/server/flush_cache.h"
#include "sglang/server/http.h"
#include "sglang/server/loads.h"
#include "sglang/server/ollama_discovery.h"
#include "sglang/server/parse_function_call.h"
#include "sglang/utf8.h"

#include <algorithm>
#include <array>
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
constexpr std::string_view model_prefix = "/v1/models/";

std::string lowercase_ascii(std::string_view text) {
    std::string result(text);
    for (auto& character : result) {
        if (character >= 'A' && character <= 'Z') {
            character += 'a' - 'A';
        }
    }
    return result;
}

bool return_reasoning_blocks(const json::Value* value) {
    if (!value) {
        return false;
    }
    if (const auto flag = value->boolean()) {
        return *flag;
    }
    if (const auto number = value->number(); number == 0 || number == 1) {
        return number == 1;
    }
    if (const auto text = value->string()) {
        const auto flag = lowercase_ascii(*text);
        if (flag == "1" || flag == "true" || flag == "t" ||
            flag == "yes" || flag == "y" || flag == "on") {
            return true;
        }
        if (flag == "0" || flag == "false" || flag == "f" ||
            flag == "no" || flag == "n" || flag == "off") {
            return false;
        }
    }
    throw std::invalid_argument("return_blocks: Input should be a valid boolean");
}

void write_separated_reasoning(const Socket& connection, const SeparatedReasoning& parsed,
                               bool return_blocks) {
    std::string response = R"({"reasoning_text":)";
    json::append_quoted(response, parsed.reasoning, true);
    response += R"(,"text":)";
    json::append_quoted(response, parsed.text, true);
    if (return_blocks) {
        response += R"(,"reasoning_blocks":[)";
        if (!parsed.reasoning.empty()) {
            json::append_quoted(response, parsed.reasoning, true);
        }
        response += R"(],"text_blocks":[)";
        json::append_quoted(response, parsed.text, true);
        response += R"(],"blocks":[)";
        if (!parsed.reasoning.empty()) {
            response += R"({"type":"reasoning","text":)";
            json::append_quoted(response, parsed.reasoning, true);
            response += "},";
        }
        response += R"({"type":"text","text":)";
        json::append_quoted(response, parsed.text, true);
        response += "}]";
    }
    write_json(connection, "200 OK", response + '}');
}

void separate_reasoning(const Socket& connection, std::string_view body) {
    try {
        if (!valid_utf8(body)) {
            throw std::invalid_argument("JSON decode error: Invalid UTF-8");
        }
        const auto input = body.empty() ? json::Value() :
            json::parse(body, json::Syntax::last_key_wins);
        if (input.is_null()) {
            throw std::invalid_argument("body: Field required");
        }
        const auto* object = input.object();
        if (!object) {
            throw std::invalid_argument("Input should be a valid dictionary");
        }
        for (const auto key : {"rid", "http_worker_ipc", "text", "reasoning_parser"}) {
            const auto* value = json::get(*object, key);
            const bool required = std::string_view(key) == "text" ||
                std::string_view(key) == "reasoning_parser";
            if (!value && required) {
                throw std::invalid_argument(std::string(key) + ": Field required");
            }
            if (value && !value->string() && (required || !value->is_null())) {
                throw std::invalid_argument(std::string(key) + ": Input should be a valid string");
            }
        }
        const bool return_blocks = return_reasoning_blocks(json::get(*object, "return_blocks"));
        const auto parser = *json::get_string(*object, "reasoning_parser");
        constexpr std::array<std::string_view, 4> adapter_parsers{
            "deepseek-v4", "deepseekv4", "deepseek-v41", "deepseekv41"};
        if (std::find(adapter_parsers.begin(), adapter_parsers.end(), parser) !=
            adapter_parsers.end()) {
            write_tokenization_error(connection, "400 Bad Request", "400",
                "'" + std::string(parser) + "' output is parsed with its model's tokenizer; "
                "this server does not serve that model");
            return;
        }
        const auto name = lowercase_ascii(parser);
        constexpr std::array<std::string_view, 6> qwen3_parsers{
            "qwen3", "qwen3-thinking", "dots", "interns1", "minimax", "nanbeige"};
        if (std::find(qwen3_parsers.begin(), qwen3_parsers.end(), name) == qwen3_parsers.end()) {
            write_internal_server_error(connection);
            return;
        }
        const auto text = *json::get_string(*object, "text");
        const bool forced = name == "qwen3-thinking" || name == "minimax";
        write_separated_reasoning(connection, ChatOutputParser::separate_qwen3(text, forced),
                                  return_blocks);
    } catch (const json::ParseError& error) {
        write_tokenization_error(connection, "400 Bad Request", "Bad Request",
                                 "JSON decode error: " + std::string(error.what()));
    } catch (const std::invalid_argument& error) {
        write_tokenization_error(connection, "400 Bad Request", "Bad Request", error.what());
    }
}

struct Completion {
    Socket connection;
    std::optional<HttpRequest> request;
    QueuedGenerate generate = QueuedGenerate{};
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
                    std::string_view accelerator)
        : service_(service), capacity_(capacity), complete_(complete), generate_(generate),
          tokenize_(tokenize), abort_(abort), health_(health), model_info_(model_info),
          server_info_(server_info), score_(score), discovery_(discovery), messages_(messages),
          responses_(responses), response_store_(response_store),
          decisions_(decisions), accelerator_(accelerator), prefix_controls_(prefix_controls) {}

    void read(const Socket& listener);
    void run(int slot);
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
    void enqueue(Socket& connection, HttpRequest request, bool generate, bool auxiliary,
                 QueuedGenerate prepared);
    void serve_control(Socket& connection, const HttpRequest& request);
    void queue_health(Socket& connection, const HttpRequest& request);
    void queue_reset(Socket& connection, const HttpRequest& request);
    bool idle_for_cache_reset();
    void reset_cache(const Socket& connection);
    void queue_flush(Socket& connection, double timeout);
    FlushCacheResult flush_cache(double timeout);
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
    const bool prefix_controls_;
    std::mutex mutex_;
    std::condition_variable changed_;
    std::condition_variable abort_ready_;
    std::condition_variable health_ready_;
    std::condition_variable cache_ready_;
    std::list<Completion> pending_;
    std::list<HealthCheck> health_checks_;
    std::list<FlushCache> flushes_;
    std::optional<Socket> reset_;
    int running_ = 0;
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

void ConnectionQueue::discard_disconnected() {
    const bool had_pending = !pending_.empty();
    pending_.remove_if([](const auto& job) { return disconnected(job.connection); });
    if (had_pending && pending_.empty()) {
        cache_ready_.notify_one();
    }
}

bool ConnectionQueue::idle_for_cache_reset() {
    discard_disconnected();
    return pending_.empty() && running_ == 0 && !resetting_;
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

void ConnectionQueue::queue_flush(Socket& connection, double timeout) {
    {
        const std::lock_guard lock(mutex_);
        flushes_.push_back({std::move(connection), timeout});
    }
    cache_ready_.notify_one();
}

FlushCacheResult ConnectionQueue::flush_cache(double timeout) {
    using Seconds = std::chrono::duration<double>;
    const auto started = Clock::now();
    std::unique_lock lock(mutex_);
    while (!closing_ && !stopping()) {
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
        if (timeout == 0.0) {
            return FlushCacheResult::busy;
        }
        const auto remaining = Seconds(timeout) - Seconds(Clock::now() - started);
        if (remaining <= Seconds::zero()) {
            return FlushCacheResult::timed_out;
        }
        const auto wait_limit = idle ? Seconds(flush_probe_retry) : Seconds(60);
        cache_ready_.wait_for(lock, std::min(remaining, wait_limit));
    }
    return FlushCacheResult::busy;
}

void ConnectionQueue::run_flushes() {
    for (;;) {
        std::unique_lock lock(mutex_);
        cache_ready_.wait(lock, [&] { return closing_ || !flushes_.empty(); });
        if (closing_) {
            return;
        }
        auto work = std::move(flushes_.front());
        flushes_.pop_front();
        lock.unlock();
        handle_errors(work.connection, [&] {
            write_flush_cache(work.connection, flush_cache(work.timeout));
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
            std::string body;
            {
                const std::lock_guard lock(mutex_);
                discard_disconnected();
                body = format_metrics(service_.metrics(), running_, pending_.size());
            }
            write_metrics(connection, body);
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
    if (path == "/flush_cache") {
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
    if (path == loads_route) {
        std::unique_lock lock(mutex_);
        discard_disconnected();
        const LoadSnapshot snapshot{running_, pending_.size(), capacity_, accelerator_};
        lock.unlock();
        write_loads(connection, request, snapshot);
        return;
    }
    if (const auto parser = path == separate_reasoning_route ? separate_reasoning
                          : path == parse_function_call_route ? parse_function_call : nullptr) {
        if (request.method == "POST") parser(connection, request.body);
        else write_method_not_allowed(connection, "POST", request.method == "HEAD");
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
    const bool tokenize = tokenize_ && tokenization_route(request) != TokenizationRoute::none;
    if (!generate && !tokenize && !score && !messages && !responses &&
        !decisions && !completion_route(request)) {
        serve_control(connection, request);
        return;
    }
    enqueue(connection, std::move(request), generate, tokenize || score || messages || decisions,
            {});
}

void ConnectionQueue::enqueue(Socket& connection, HttpRequest request, bool generate,
                              bool auxiliary, QueuedGenerate prepared) {
    if (!auxiliary) {
        service_.rethrow_failure();
    }
    {
        std::unique_lock lock(mutex_);
        if (closing_) {
            return;
        }
        discard_disconnected();
        if (pending_.size() >= static_cast<std::size_t>(capacity_)) {
            if (auxiliary) {
                lock.unlock();
                if (request.route_path() == anthropic_count_route ||
                    request.route_path() == anthropic_messages_route) {
                    write_anthropic_error(connection, "503 Service Unavailable", "api_error",
                                          "Internal server error");
                } else {
                    write_tokenization_error(connection, "503 Service Unavailable",
                        "ServiceUnavailableError", "Native request queue is full");
                }
                return;
            }
            throw HttpError("503 Service Unavailable", "Native completion queue is full");
        }
        if (generate) {
            prepared = generate_(request.body);
        }
        if (prepared.complete) {
            request.body = std::string{};
        }
        pending_.push_back({std::move(connection), std::move(request), std::move(prepared)});
    }
    changed_.notify_one();
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
        // Destroy the socket before releasing the active slot.
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
                const auto tokenize = tokenization_route(request);
                if (work->generate.complete) {
                    const auto generate = std::move(work->generate);
                    generate.complete(work->connection, slot);
                } else if (tokenize != TokenizationRoute::none) {
                    tokenize_(work->connection, tokenize, request.body);
                } else if (score_ && request.route_path() == score_route) {
                    score_(work->connection, slot, request.body);
                } else if (messages_ && request.route_path() == anthropic_messages_route) {
                    messages_(work->connection, slot, request.body);
                } else if (responses_ && request.route_path() == responses_route) {
                    responses_(work->connection, slot, request.body);
                } else if (decisions_ && (request.route_path() == decisions_route ||
                                          request.route_path() == systemone_route)) {
                    decisions_(work->connection, slot, request.body,
                               request.route_path() == systemone_route);
                } else {
                    complete_(work->connection, slot, request.body, chat_route(request));
                }
            });
        }
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
    {
        const std::lock_guard lock(mutex_);
        health_checks_.remove_if([](const auto& check) {
            return disconnected(check.connection);
        });
        if (!stopping() && service_.ready() &&
            health_checks_.size() < static_cast<std::size_t>(capacity_)) {
            health_checks_.push_back({std::move(connection), started});
            health_ready_.notify_one();
            return;
        }
    }
    write_generation_health(connection, false);
}

void ConnectionQueue::run_health() {
    for (;;) {
        std::unique_lock lock(mutex_);
        health_ready_.wait(lock, [&] { return closing_ || !health_checks_.empty(); });
        if (health_checks_.empty()) {
            return;
        }
        auto work = std::move(health_checks_.front());
        health_checks_.pop_front();
        lock.unlock();

        const auto cancelled = [&] { return disconnected(work.connection); };
        const bool healthy = service_.wait_for_generation(
            work.started, work.started + generation_health_timeout, cancelled);
        write_generation_health(work.connection, healthy && service_.ready());
    }
}

void ConnectionQueue::stop() {
    {
        const std::lock_guard lock(mutex_);
        closing_ = true;
        pending_.clear();
        flushes_.clear();
        reset_.reset();
    }
    changed_.notify_all();
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
                       std::string_view accelerator) {
    if (max_active < 1 || max_active > decoder_batch_max_lanes) {
        throw std::invalid_argument("Invalid HTTP completion capacity");
    }
    ConnectionQueue queue(service, max_active, complete, health, prefix_controls, generate,
                          tokenize, model_info, abort, server_info, score, discovery, messages,
                          responses, response_store, decisions, accelerator);
    std::vector<std::thread> workers;
    workers.reserve(max_active + (abort ? 3 : 2));
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
        workers.emplace_back([&] {
            try {
                queue.run_health();
            } catch (...) {
                stop_on_error();
            }
        });
        workers.emplace_back([&] {
            try {
                queue.run_flushes();
            } catch (...) {
                stop_on_error();
            }
        });
        if (abort) {
            workers.emplace_back([&] {
                try {
                    queue.run_aborted();
                } catch (...) {
                    stop_on_error();
                }
            });
        }
        for (int slot = 0; slot < max_active; ++slot) {
            workers.emplace_back([&, slot] {
                try {
                    queue.run(slot);
                } catch (...) {
                    stop_on_error();
                }
            });
        }
        queue.read(listener);
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
