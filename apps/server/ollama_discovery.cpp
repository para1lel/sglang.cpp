#include "sglang/server/ollama_discovery.h"

#include "sglang/json.h"
#include "sglang/server/http.h"
#include "sglang/tokenizer/tokenizer.h"
#include "sglang/utf8.h"

#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <set>
#include <stdexcept>

namespace sglang::server {
namespace {

constexpr std::string_view default_root = "/";
constexpr std::string_view default_tags = "/api/tags";
constexpr std::string_view default_show = "/api/show";
constexpr auto root_variable = "SGLANG_OLLAMA_ROOT_ROUTE";

std::string_view route_env(const char* variable, std::string_view fallback) {
    const auto* value = std::getenv(variable);
    return value ? std::string_view(value) : fallback;
}

std::string_view tail_name(std::string_view model) {
    const auto slash = model.rfind('/');
    return slash == model.npos ? model : model.substr(slash + 1);
}

std::string timestamp() {
    using namespace std::chrono;
    const auto now = system_clock::now();
    const auto seconds = floor<std::chrono::seconds>(now);
    const auto unix_seconds = system_clock::to_time_t(seconds);
    std::tm utc{};
    std::array<char, 24> date{};
    if (!gmtime_r(&unix_seconds, &utc) ||
        !std::strftime(date.data(), date.size(), "%Y-%m-%dT%H:%M:%S", &utc)) {
        throw std::runtime_error("Cannot format the model metadata timestamp");
    }
    std::array<char, 6> fraction{};
    const auto milliseconds = duration_cast<std::chrono::milliseconds>(now - seconds).count();
    std::snprintf(fraction.data(), fraction.size(), ".%03dZ", static_cast<int>(milliseconds));
    return std::string(date.data()) + fraction.data();
}

void quote(std::string& body, std::string_view value) {
    json::append_quoted(body, value, true);
}

void field(std::string& body, std::string_view key, std::string_view value) {
    quote(body, key);
    body += ':';
    quote(body, value);
}

std::string tags_json() {
    std::string body = R"({"models":[{)";
    field(body, "name", model_name);
    body += ',';
    field(body, "model", model_name);
    body += ',';
    field(body, "modified_at", timestamp());
    body += R"(,"size":0,"digest":)";
    quote(body, "sha256:sglang0000000000000000000000000000000000000000000000000000000000");
    body += R"(,"details":{"format":"sglang",)";
    field(body, "family", tail_name(model_name));
    return body + R"(,"parameter_size":"unknown"}}]})";
}

std::string show_json(std::string_view model) {
    auto family = tail_name(model);
    for (const std::string_view suffix : {"-Instruct", "-Chat", "-Base"}) {
        if (family.size() >= suffix.size() &&
            family.substr(family.size() - suffix.size()) == suffix) {
            family.remove_suffix(suffix.size());
            break;
        }
    }
    const auto context = std::to_string(checkpoint_context);
    const auto parameters = "num_ctx " + context;
    std::string body = R"({"license":"",)";
    field(body, "modelfile", "FROM " + std::string(model) + "\nPARAMETER " + parameters + "\n");
    body += ',';
    field(body, "parameters", parameters);
    body += R"(,"template":"",)";
    field(body, "modified_at", timestamp());
    body += R"(,"details":{"parent_model":"","format":"sglang",)";
    field(body, "family", family);
    body += R"(,"families":[)";
    quote(body, family);
    body += R"(],"parameter_size":"unknown","quantization_level":""},"model_info":{)";
    field(body, "general.architecture", family);
    body += ',';
    field(body, "general.name", model);
    body += R"(,"general.parameter_count":0,)";
    const auto family_key = std::string(family) + '.';
    quote(body, family_key + "context_length");
    body += ':' + context;
    for (const auto key : {"block_count", "embedding_length", "attention.head_count"}) {
        body += ',';
        quote(body, family_key + key);
        body += ":0";
    }
    return body + R"(},"capabilities":["completion"]})";
}

void show(const Socket& connection, const HttpRequest& request) {
    constexpr auto dictionary =
        "Input should be a valid dictionary or object to extract fields from";
    std::string_view error = "body: Field required";
    if (!request.body.empty()) {
        error = dictionary;
        if (request.body_is_json) {
            try {
                if (!valid_utf8(request.body)) {
                    throw json::ParseError("Invalid UTF-8 input");
                }
                const auto input = json::parse(request.body, json::Syntax::http_body);
                if (const auto* object = input.object()) {
                    const auto* model = json::get(*object, "model");
                    if (model && model->string()) {
                        write_json(connection, "200 OK", show_json(*model->string()));
                        return;
                    }
                    error = model ? "model: Input should be a valid string"
                                  : "model: Field required";
                } else if (input.is_null()) {
                    error = "body: Field required";
                }
            } catch (const json::ParseError& parse_error) {
                write_tokenization_error(connection, "400 Bad Request", "Bad Request",
                                         "JSON decode error: " + std::string(parse_error.what()));
                return;
            }
        }
    }
    write_tokenization_error(connection, "400 Bad Request", "Bad Request", error);
}

std::string_view converter(std::string_view name) {
    if (name.empty() || name == "str") {
        return "[^/]+";
    }
    if (name == "path") {
        return ".*";
    }
    if (name == "int") {
        return "[0-9]+";
    }
    if (name == "float") {
        return R"([0-9]+(?:\.[0-9]+)?)";
    }
    if (name == "uuid") {
        return "[0-9a-fA-F]{8}-?[0-9a-fA-F]{4}-?[0-9a-fA-F]{4}-?"
               "[0-9a-fA-F]{4}-?[0-9a-fA-F]{12}";
    }
    throw std::invalid_argument("Unknown Ollama route converter: " + std::string(name));
}

void append_regex_literal(std::string& expression, std::string_view text) {
    for (const auto byte : text) {
        if (std::string_view(R"(\.^$|()[]{}*+?)").find(byte) != text.npos) {
            expression += '\\';
        }
        expression += byte;
    }
}

std::string_view slash_alternate(std::string_view path) {
    const auto tail = path.find_last_not_of('/');
    return tail == path.npos ? std::string_view{} : path.substr(0, tail + 1);
}

}

OllamaDiscovery::Path::Path(std::string_view path) : literal(path) {
    if (path.find('{') == path.npos) {
        return;
    }
    const std::regex parameter(R"(\{([a-zA-Z_][a-zA-Z0-9_]*)(?::([a-zA-Z_][a-zA-Z0-9_]*))?\})");
    std::set<std::string> names;
    std::string expression;
    std::size_t offset = 0;
    for (auto match = std::sregex_iterator(literal.begin(), literal.end(), parameter);
         match != std::sregex_iterator(); ++match) {
        if (!names.insert((*match)[1].str()).second) {
            throw std::invalid_argument("Duplicate Ollama route parameter: " + (*match)[1].str());
        }
        const auto position = static_cast<std::size_t>(match->position());
        append_regex_literal(expression, path.substr(offset, position - offset));
        expression += converter((*match)[2].str());
        offset = position + match->length();
    }
    if (offset != 0) {
        append_regex_literal(expression, path.substr(offset));
        pattern.emplace(expression);
    }
}

bool OllamaDiscovery::Path::matches(std::string_view path) const {
    if (pattern) {
        return std::regex_match(path.begin(), path.end(), *pattern);
    }
    return path == literal;
}

OllamaDiscovery::OllamaDiscovery()
    : ollama_root_(std::getenv(root_variable) != nullptr),
      root_(route_env(root_variable, default_root)),
      tags_(route_env("SGLANG_OLLAMA_TAGS_ROUTE", default_tags)),
      show_(route_env("SGLANG_OLLAMA_SHOW_ROUTE", default_show)) {}

bool OllamaDiscovery::matches(std::string_view path) const {
    return root_.matches(path) || tags_.matches(path) || show_.matches(path);
}

bool OllamaDiscovery::matches_show(const HttpRequest& request) const {
    return show_.matches(request.route_path()) || show_.matches(request.path);
}

bool OllamaDiscovery::respond_exact(const Socket& connection, const HttpRequest& request,
                                    bool later_full, std::string_view earlier_allow) const {
    const auto head = request.method == "HEAD";
    const auto get = request.method == "GET";
    const auto path = request.route_path();
    const auto root = root_.matches(path) || root_.matches(request.path);
    const auto tags = tags_.matches(path) || tags_.matches(request.path);
    const auto metadata = matches_show(request);
    if (root && (head || get)) {
        write_json(connection, "200 OK",
                   ollama_root_ ? R"("Ollama is running")" : R"("SGLang is running")", head);
    } else if (tags && get) {
        write_json(connection, "200 OK", tags_json());
    } else if (metadata && request.method == "POST") {
        show(connection, request);
    } else if (!later_full && (root || tags || metadata)) {
        const auto allow = earlier_allow.empty() ?
            (root ? "HEAD" : tags ? "GET" : "POST") : earlier_allow;
        write_method_not_allowed(connection, allow, head);
    } else {
        return false;
    }
    return true;
}

bool OllamaDiscovery::respond_unmatched(const Socket& connection,
                                        const HttpRequest& request) const {
    const auto path = std::string_view(request.path);
    std::string extended;
    if (!path.empty() && path != default_root) {
        const auto alternate = path.back() == '/' ? slash_alternate(path) :
            std::string_view(extended = std::string(path) + '/');
        if (!alternate.empty() && matches(matchable_route_path(alternate))) {
            write_redirect(connection, request, alternate);
            return true;
        }
    }
    const auto base = matchable_route_path(slash_alternate(path));
    if (path == default_root || base.empty() || base == default_tags || base == default_show) {
        write_json(connection, "404 Not Found", R"({"detail":"Not Found"})",
                   request.method == "HEAD");
        return true;
    }
    return false;
}

}
