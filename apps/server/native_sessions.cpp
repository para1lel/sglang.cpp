#include "sglang/server/native_generate.h"

#include "sglang/server/http.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <limits>

namespace sglang::server {
namespace {

using Clock = std::chrono::steady_clock;
constexpr std::size_t max_sessions = 256;
constexpr std::size_t max_session_turns = 256;
constexpr std::size_t max_session_tokens = 4 * 1024 * 1024;

template <typename Type>
std::optional<Type> session_field(const json::Object& object, const char* key,
                                  std::optional<Type> (json::Value::*read)() const) {
    const auto* field = json::get(object, key);
    if (!field || field->is_null()) {
        return std::nullopt;
    }
    const auto value = std::invoke(read, *field);
    if (!value) {
        throw std::invalid_argument(std::string("Invalid session field: ") + key);
    }
    return value;
}

std::optional<std::int64_t> session_integer(const json::Object& object, const char* key) {
    const auto* field = json::get(object, key);
    if (field && !field->is_null()) {
        if (const auto value = field->unsigned_integer()) {
            constexpr auto limit = std::numeric_limits<std::int64_t>::max();
            return static_cast<std::int64_t>(std::min(*value, static_cast<std::uint64_t>(limit)));
        }
    }
    return session_field(object, key, &json::Value::signed_integer);
}

void session_error(const Socket& connection, std::string_view status, std::string_view message) {
    std::string body = status == "422 Unprocessable Entity" ? R"({"detail":)"
                                                            : R"({"error":{"message":)";
    json::append_quoted(body, message);
    body += status == "422 Unprocessable Entity" ? "}" : "}}";
    write_json(connection, status, body);
}

}

struct NativeGenerate::Session {
    struct Turn {
        std::vector<int> tokens;
        std::size_t prompt_size = 0;
        std::size_t charged = 0;
        bool complete = false;
    };
    std::map<std::string, Turn, std::less<>> turns;
    std::optional<double> timeout;
    Clock::time_point last_admitted = Clock::now();
    bool closed = false;
};

bool session_control_route(std::string_view path) {
    return path == "/open_session" || path == "/close_session";
}

void NativeGenerate::close_session(Session& session) {
    for (const auto& [id, turn] : session.turns) {
        session_tokens_ -= turn.charged;
    }
    session.turns.clear();
    session.closed = true;
}

void NativeGenerate::expire_sessions() {
    const auto now = Clock::now();
    for (auto entry = sessions_.begin(); entry != sessions_.end();) {
        auto& session = *entry->second;
        if (session.timeout &&
            std::chrono::duration<double>(now - session.last_admitted).count() > *session.timeout) {
            close_session(session);
            entry = sessions_.erase(entry);
        } else {
            ++entry;
        }
    }
}

void NativeGenerate::serve_session(const Socket& connection, const HttpRequest& request) {
    if (request.method != "GET" && request.method != "POST") {
        write_method_not_allowed(connection, "GET, POST", request.method == "HEAD");
        return;
    }
    constexpr auto invalid = "422 Unprocessable Entity";
    json::Value parsed;
    const bool opening = request.route_path() == "/open_session";
    std::optional<std::string_view> requested;
    std::optional<double> timeout;
    bool streaming = false;
    try {
        if (!request.body_is_json.value_or(true)) {
            throw std::invalid_argument("Expected a JSON session object");
        }
        parsed = json::parse(request.body, json::Syntax::last_key_wins);
        const auto* fields = parsed.object();
        if (!fields) {
            throw std::invalid_argument("Expected a JSON session object");
        }
        requested = session_field(*fields, "session_id", &json::Value::string);
        if (opening) {
            if (!session_integer(*fields, "capacity_of_str_len")) {
                throw std::invalid_argument("Expected integer capacity_of_str_len");
            }
            streaming = session_field(*fields, "streaming", &json::Value::boolean).value_or(false);
            timeout = session_field(*fields, "timeout", &json::Value::number);
            if (timeout && !std::isfinite(*timeout)) {
                throw std::invalid_argument("Expected finite session timeout");
            }
        } else if (!requested) {
            throw std::invalid_argument("Expected string session_id");
        }
    } catch (const json::ParseError&) {
        session_error(connection, invalid, "Expected a JSON session object");
        return;
    } catch (const std::invalid_argument& error) {
        session_error(connection, invalid, error.what());
        return;
    }
    if (streaming) {
        session_error(connection, "400 Bad Request",
                      "Streaming sessions are disabled. "
                      "Please relaunch with --enable-streaming-session.");
        return;
    }
    std::unique_lock lock(sessions_mutex_);
    expire_sessions();
    if (!opening) {
        const auto existing = sessions_.find(*requested);
        if (existing != sessions_.end()) {
            close_session(*existing->second);
            sessions_.erase(existing);
        }
        lock.unlock();
        write_raw_response(connection, "200 OK", "");
        return;
    }
    if (requested && sessions_.count(*requested)) {
        lock.unlock();
        session_error(connection, "400 Bad Request", "Failed to open the session. "
                      "Check if a session with the same id is still open.");
        return;
    }
    if (sessions_.size() >= max_sessions || (requested && requested->size() > 256)) {
        lock.unlock();
        session_error(connection, "400 Bad Request", "Native session limit exceeded");
        return;
    }
    std::string id;
    if (requested) {
        id = *requested;
    } else {
        std::random_device random;
        do {
            id = random_id(random);
        } while (sessions_.count(id));
    }
    auto session = std::make_shared<Session>();
    session->timeout = timeout;
    sessions_.emplace(id, std::move(session));
    lock.unlock();
    std::string body;
    json::append_quoted(body, id);
    write_json(connection, "200 OK", body);
}

std::shared_ptr<NativeGenerate::Session> NativeGenerate::bind_session(const GenerateInput& input) {
    const auto* value = json::get(input.fields, "session_params");
    if (!value || value->is_null()) {
        return nullptr;
    }
    const auto* parameters = value->object();
    if (!parameters || input.indexed()) {
        throw HttpError("400 Bad Request",
                        "Expected one session_params object with scalar /generate and n: 1");
    }
    const auto id = session_field(*parameters, "id", &json::Value::string);
    if (!id) {
        throw std::invalid_argument("Expected a session id");
    }
    const std::lock_guard lock(sessions_mutex_);
    expire_sessions();
    const auto entry = sessions_.find(*id);
    if (entry == sessions_.end()) {
        throw std::invalid_argument("Invalid request: session id " + std::string(*id) +
                                    " does not exist");
    }
    return entry->second;
}

std::unique_ptr<NativeGenerate::SessionTurn>
NativeGenerate::begin_session(const GenerateInput& input, std::string_view rid,
                              const std::shared_ptr<Session>& session) {
    const auto& params = *json::get(input.fields, "session_params")->object();
    const auto parent = session_field(params, "rid", &json::Value::string);
    const auto offset = session_integer(params, "offset").value_or(0);
    const bool drop = session_field(params, "drop_previous_output", &json::Value::boolean)
                          .value_or(false);
    if (rid.size() > 256) {
        throw std::invalid_argument("Expected a session turn rid of up to 256 bytes");
    }
    if (session_field(params, "replace", &json::Value::boolean).value_or(false)) {
        throw std::invalid_argument("Session replace is not supported. To branch from a turn, "
                                    "append with its rid; to stop a running turn first, abort it.");
    }
    const std::lock_guard lock(sessions_mutex_);
    expire_sessions();
    if (session->closed) {
        throw std::invalid_argument("Session closed before turn admission");
    }
    auto turn = std::make_unique<SessionTurn>(*this, session, std::string(rid));
    if (parent) {
        const auto found = session->turns.find(*parent);
        if (found == session->turns.end()) {
            throw std::invalid_argument("Invalid request session id");
        }
        if (!found->second.complete) {
            throw std::invalid_argument(
                "Session request is appending to a request that hasn't finished.");
        }
        const auto& prior = found->second;
        auto count = static_cast<std::int64_t>(drop ? prior.prompt_size : prior.tokens.size());
        if (offset > 0) {
            count = std::min(count, offset);
        } else if (offset < 0) {
            count = std::max<std::int64_t>(0, count + offset);
        }
        turn->history.assign(prior.tokens.begin(), prior.tokens.begin() + count);
    }
    return turn;
}

void NativeGenerate::SessionTurn::reserve(const BatchPrompt& item) {
    const auto charge = item.prompt.size() + item.options.max_new_tokens;
    Session::Turn stored;
    stored.tokens.reserve(charge);
    stored.tokens.assign(item.prompt.begin(), item.prompt.end());
    stored.prompt_size = item.prompt.size();
    stored.charged = charge;
    const std::lock_guard lock(owner.sessions_mutex_);
    owner.expire_sessions();
    if (session->closed) {
        throw std::invalid_argument("Session closed before turn admission");
    }
    if (session->turns.count(id)) {
        throw std::invalid_argument("Duplicate session turn rid");
    }
    if (session->turns.size() >= max_session_turns ||
        charge > max_session_tokens - owner.session_tokens_) {
        throw std::invalid_argument("Native session history limit exceeded");
    }
    session->turns.emplace(id, std::move(stored));
    owner.session_tokens_ += charge;
    session->last_admitted = Clock::now();
    pending = true;
}

NativeGenerate::SessionTurn::~SessionTurn() {
    if (!pending) {
        return;
    }
    const std::lock_guard lock(owner.sessions_mutex_);
    if (!session->closed) {
        const auto found = session->turns.find(id);
        owner.session_tokens_ -= found->second.charged;
        session->turns.erase(found);
    }
}

void NativeGenerate::SessionTurn::finish(const DecoderResult& result) {
    if (result.finish_reason == DecoderFinishReason::aborted ||
        result.finish_reason == DecoderFinishReason::cancelled) {
        return;
    }
    const std::lock_guard lock(owner.sessions_mutex_);
    if (session->closed) {
        return;
    }
    auto& stored = session->turns.at(id);
    stored.tokens.insert(stored.tokens.end(), result.output_token_ids.begin(),
                         result.output_token_ids.end());
    stored.complete = true;
    pending = false;
}

}
