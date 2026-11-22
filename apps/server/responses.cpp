#include "sglang/server/responses.h"

#include "sglang/server/chat_template.h"
#include "sglang/server/http.h"

#include <algorithm>
#include <ctime>
#include <iostream>
#include <random>

namespace sglang::server {
namespace {

using json::Array;
using json::OrderedObject;
using json::Value;

constexpr std::string_view response_prefix = "/v1/responses/";
constexpr std::string_view cancel_suffix = "/cancel";
constexpr auto store_disabled = "Response store is disabled. Stateful Responses require "
    "--enable-response-store on a standalone server; response storage is unavailable in PD mode.";
constexpr auto store_limit = "Stored response exceeds the 256 KiB retention limit";
constexpr auto store_full = "Response store is full";
constexpr auto id_conflict = "A stored response already has this request ID";

struct RequestError : std::runtime_error {
    RequestError(unsigned code, std::string message, const char* parameter = nullptr)
        : std::runtime_error(std::move(message)), code(code), parameter(parameter) {}

    unsigned code;
    const char* parameter;
};

const Value& field(const OrderedObject& body, std::string_view key) {
    static const Value empty;
    const auto* value = json::get(body, key);
    return value ? *value : empty;
}

const OrderedObject& object(const Value& value, const char* name) {
    const auto* parsed = value.ordered_object();
    if (!parsed) throw RequestError(400, "Expected an object for " + std::string(name), name);
    return *parsed;
}

bool flag(const OrderedObject& body, const char* key, bool fallback) {
    const auto& value = field(body, key);
    if (value.is_null()) return fallback;
    if (const auto result = value.boolean()) return *result;
    throw RequestError(400, "Expected a boolean", key);
}

Array response_input(const Value& input) {
    if (input.string()) return {OrderedObject{{"role", "user"}, {"content", input}}};
    const auto* items = input.array();
    if (!items || items->empty()) {
        throw RequestError(400, "Expected text or a nonempty array of message items", "input");
    }
    return *items;
}

void merge_assistant(OrderedObject& previous, const char* key, Value content) {
    if (content.is_null() || content.string() == "") return;
    auto existing = std::find_if(previous.begin(), previous.end(), [&](const auto& field) {
        return field.first == key;
    });
    if (existing == previous.end()) {
        previous.emplace_back(key, std::move(content));
        return;
    }
    const auto left = existing->second.string();
    const auto right = content.string();
    if (left && right) {
        const char* separator = std::string_view(key) == "content" ? "\n\n" : "\n";
        existing->second = left->empty() ? std::string(*right) :
            std::string(*left) + separator + std::string(*right);
        return;
    }
    Array parts;
    for (const auto* value : {&existing->second, &content}) {
        if (const auto* array = value->array()) {
            parts.insert(parts.end(), array->begin(), array->end());
        } else if (value->string() && value->string() != "") {
            parts.emplace_back(OrderedObject{{"type", "text"}, {"text", *value}});
        }
    }
    existing->second = std::move(parts);
}

std::string reasoning_text(const OrderedObject& item) {
    for (const auto* key : {"summary", "content"}) {
        std::string text;
        if (const auto* parts = field(item, key).array()) {
            for (const auto& part : *parts) {
                const auto value = field(object(part, "input"), "text").string();
                if (value && !value->empty()) {
                    if (!text.empty()) text += '\n';
                    text += *value;
                }
            }
        }
        if (!text.empty()) return text;
    }
    return {};
}

Value message_content(const OrderedObject& message, bool developer) {
    auto content = field(message, "content");
    constexpr auto label = "Developer instructions:";
    if (const auto text = content.string()) {
        return developer ? Value(std::string(label) + '\n' + std::string(*text)) : content;
    }
    const auto* parts = content.array();
    if (!parts) throw RequestError(400, "Expected text message content", "input");
    Array translated;
    for (const auto& part : *parts) {
        const auto& values = object(part, "input");
        const auto kind = field(values, "type").string().value_or("");
        const auto text = field(values, "text").string();
        if ((kind != "input_text" && kind != "output_text") || !text) {
            throw RequestError(400, "Only text content parts are supported", "input");
        }
        auto value = developer && translated.empty() ?
            std::string(label) + '\n' + std::string(*text) : std::string(*text);
        translated.emplace_back(OrderedObject{{"type", "text"}, {"text", std::move(value)}});
    }
    if (developer && translated.empty()) {
        translated.emplace_back(OrderedObject{{"type", "text"}, {"text", label}});
    }
    return translated;
}

Array messages(const Array& input, const Value& instructions) {
    Array result;
    std::string system;
    if (const auto text = instructions.string()) system = *text;
    for (const auto& item : input) {
        const auto& message = object(item, "input");
        const auto type = field(message, "type").string().value_or("message");
        const auto role = type == "reasoning" ? std::string_view("assistant") :
            field(message, "role").string().value_or("");
        if (type != "reasoning" && (type != "message" ||
            (role != "user" && role != "assistant" && role != "system" && role != "developer"))) {
            throw RequestError(400, "Expected a plain text or reasoning message", "input");
        }
        const char* key = type == "reasoning" ? "reasoning_content" : "content";
        auto content = type == "reasoning" ? Value(reasoning_text(message)) :
            message_content(message, role == "developer");
        if (type == "reasoning" && content.string() == "") continue;
        if (role == "system" || role == "developer") {
            if (!system.empty()) system += "\n\n";
            if (const auto text = content.string()) {
                system += *text;
            } else {
                for (const auto& part : *content.array()) {
                    if (&part != &content.array()->front()) system += "\n\n";
                    system += field(*part.ordered_object(), "text").string().value();
                }
            }
            continue;
        }
        if (role == "assistant" && !result.empty()) {
            auto* prior = result.back().ordered_object();
            if (field(*prior, "role").string() == "assistant") {
                merge_assistant(*prior, key, std::move(content));
                continue;
            }
        }
        result.emplace_back(OrderedObject{{"role", std::string(role)}, {key, std::move(content)}});
    }
    if (!system.empty()) {
        result.insert(result.begin(), OrderedObject{{"role", "system"},
                                                     {"content", std::move(system)}});
    }
    return result;
}

Value count(std::size_t value) {
    return static_cast<std::uint64_t>(value);
}

std::string responses_id(std::string_view prefix) {
    constexpr std::string_view hex = "0123456789abcdef";
    std::random_device random;
    std::string id(prefix);
    for (int part = 0; part < 4; ++part) {
        auto value = static_cast<std::uint32_t>(random());
        for (int digit = 0; digit < 8; ++digit) {
            id += hex[value & 15];
            value >>= 4;
        }
    }
    id[prefix.size() + 12] = '4';
    const auto variant = prefix.size() + 16;
    id[variant] = hex[8 + (hex.find(id[variant]) & 3)];
    return id;
}

bool pending(const OrderedObject& envelope) {
    const auto status = field(envelope, "status").string();
    return status == "queued" || status == "in_progress";
}

}

void responses_error(const Socket& connection, unsigned code, std::string_view message,
                     const char* parameter) {
    const auto status = code == 404 ? "404 Not Found" : code == 409 ? "409 Conflict" :
        code == 413 ? "413 Payload Too Large" : code == 500 ? "500 Internal Server Error" :
        code == 503 ? "503 Service Unavailable" : "400 Bad Request";
    const auto type = code >= 500 ? "server_error" : "invalid_request_error";
    write_json(connection, status, chat_template_json(OrderedObject{{"error", OrderedObject{
        {"message", std::string(message)}, {"type", type},
        {"param", parameter ? Value(parameter) : Value{}}, {"code", count(code)}}}}));
}

ResponseStore::Handle ResponseStore::lookup(std::string_view id) const {
    if (id.substr(0, 5) != "resp_") {
        throw RequestError(400, "Invalid 'response_id': '" + std::string(id) +
                                "'. Expected an ID that begins with 'resp'.", "response_id");
    }
    const auto found = std::find_if(entries_.begin(), entries_.end(), [&](const auto& entry) {
        return entry->id == id;
    });
    if (found == entries_.end()) {
        throw RequestError(404, "Response with id '" + std::string(id) + "' not found.",
                           "response_id");
    }
    return *found;
}

ResponseStore::Handle ResponseStore::insert(const OrderedObject& response, std::string envelope,
                                            std::string history) {
    if (envelope.size() + history.size() > record_bytes) throw std::logic_error(store_limit);
    auto entry = std::make_shared<Entry>();
    entry->id = field(response, "id").string().value();
    entry->active = pending(response);
    entry->envelope = std::move(envelope);
    entry->history = std::move(history);
    const std::lock_guard lock(mutex_);
    if (std::any_of(entries_.begin(), entries_.end(), [&](const auto& stored) {
            return stored->id == entry->id;
        })) {
        throw RequestError(409, id_conflict, "request_id");
    }
    constexpr std::size_t capacity = 128;
    if (entries_.size() == capacity) {
        const auto oldest = std::find_if(entries_.begin(), entries_.end(),
            [](const auto& stored) { return !stored->active; });
        if (oldest == entries_.end()) return {};
        entries_.erase(oldest);
    }
    entries_.push_back(entry);
    return entry;
}

bool ResponseStore::update(const Handle& entry, const OrderedObject& response,
                           std::optional<std::string> history) {
    auto envelope = chat_template_json(response);
    const std::lock_guard lock(mutex_);
    if (entry->cancelled) return true;
    if (envelope.size() + (history ? history->size() : entry->history.size()) > record_bytes) {
        return false;
    }
    if (history) entry->history = std::move(*history);
    entry->envelope = std::move(envelope);
    entry->active = pending(response);
    return true;
}

Array ResponseStore::history(std::string_view id) {
    std::string history;
    {
        const std::lock_guard lock(mutex_);
        history = lookup(id)->history;
    }
    return std::move(*json::parse(history, json::Syntax::ordered).array());
}

std::string_view responses_method(std::string_view path) {
    if (path == responses_route) return "POST";
    if (path.substr(0, response_prefix.size()) != response_prefix) return {};
    path.remove_prefix(response_prefix.size());
    if (path.empty() || path.front() == '/') return {};
    const auto separator = path.find('/');
    if (separator == path.npos) return "GET";
    return path.substr(separator) == cancel_suffix ? "POST" : std::string_view{};
}

bool ResponseStore::serve_state(const Socket& connection, const HttpRequest& request) {
    const auto path = request.route_path();
    if (path == responses_route) return false;
    const auto method = responses_method(path);
    if (method.empty()) return false;
    if (request.method != method) {
        write_method_not_allowed(connection, method, request.method == "HEAD");
        return true;
    }
    const bool cancel = method == "POST";
    auto id = path.substr(response_prefix.size());
    if (cancel) id.remove_suffix(cancel_suffix.size());
    try {
        if (!enabled()) throw RequestError(400, store_disabled, "response_id");
        std::string snapshot;
        {
            const std::lock_guard lock(mutex_);
            const auto entry = lookup(id);
            if (cancel && entry->active) {
                auto envelope = json::parse(entry->envelope, json::Syntax::ordered);
                for (auto& [key, value] : *envelope.ordered_object()) {
                    if (key == "status") value = "cancelled";
                }
                entry->envelope = chat_template_json(envelope);
                entry->active = false;
                entry->cancelled = true;
            }
            snapshot = entry->envelope;
        }
        write_json(connection, "200 OK", snapshot);
    } catch (const RequestError& error) {
        responses_error(connection, error.code, error.what(), error.parameter);
    }
    return true;
}

namespace {

void validate_text_options(const OrderedObject& fields) {
    for (const auto* key : {"max_tool_calls", "session_id", "priority", "extra_key",
                            "bootstrap_host", "bootstrap_port", "bootstrap_room",
                            "routed_dp_rank", "disagg_prefill_dp_rank"}) {
        if (!field(fields, key).is_null()) {
            throw RequestError(400, "Unsupported Responses option: " + std::string(key), key);
        }
    }
    if (const auto* tier = json::get(fields, "service_tier"); tier &&
        tier->string() != "auto" && tier->string() != "default" && tier->string() != "flex" &&
        tier->string() != "scale" && tier->string() != "priority") {
        throw RequestError(400, "Unsupported service tier", "service_tier");
    }
    const auto& tools = field(fields, "tools");
    const auto& include = field(fields, "include");
    const auto& logprobs = field(fields, "top_logprobs");
    const auto& choice = field(fields, "tool_choice");
    if ((json::get(fields, "tools") && (!tools.array() || !tools.array()->empty())) ||
        (!include.is_null() && (!include.array() || !include.array()->empty())) ||
        (!logprobs.is_null() && logprobs.unsigned_integer() != 0) ||
        (json::get(fields, "tool_choice") && choice.string() != "auto" &&
         choice.string() != "none")) {
        throw RequestError(400, "Responses tools, includes and output logprobs are unavailable");
    }
    const auto& truncation = field(fields, "truncation");
    if (!truncation.is_null() && truncation.string() != "disabled") {
        throw RequestError(400, "Only disabled truncation is supported", "truncation");
    }
}

OrderedObject schema_descriptor(const OrderedObject& format) {
    const auto name = field(format, "name").string();
    constexpr std::string_view name_characters =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-";
    if (!name || name->empty() || name->size() > 64 ||
        name->find_first_not_of(name_characters) != std::string_view::npos) {
        throw RequestError(400, "Expected a JSON Schema name of 1 to 64 letters, digits, "
                                 "underscores or hyphens", "text.format.name");
    }
    object(field(format, "schema"), "text.format.schema");
    if (const auto* strict = json::get(format, "strict"); strict && strict->boolean() != true) {
        throw RequestError(400, "Only strict:true is supported", "text.format.strict");
    }
    if (const auto* description = json::get(format, "description");
        description && (!description->string() || description->string()->size() > 256)) {
        throw RequestError(400, "Expected a string of at most 256 bytes",
                           "text.format.description");
    }
    OrderedObject descriptor;
    for (const auto& [key, value] : format) {
        if (key == "type") continue;
        if (key != "name" && key != "schema" && key != "strict" && key != "description") {
            throw RequestError(400, "Unsupported JSON Schema format field: " + key, "text.format");
        }
        descriptor.emplace_back(key, value);
    }
    return descriptor;
}

Value text_options(const Value& text, Value& response_format) {
    if (text.is_null()) return {};
    const auto& config = object(text, "text");
    OrderedObject options;
    const auto& format = field(config, "format");
    if (!format.is_null()) {
        const auto& fields = object(format, "text.format");
        const auto type = field(fields, "type").string();
        if (type == "json_schema") {
            response_format = OrderedObject{{"type", "json_schema"},
                                            {"json_schema", schema_descriptor(fields)}};
            options.emplace_back("format", format);
        } else if (type == "text") {
            options.emplace_back("format", OrderedObject{{"type", "text"}});
        } else {
            throw RequestError(400, "Expected text or json_schema format", "text.format.type");
        }
    }
    const auto& verbosity = field(config, "verbosity");
    if (!verbosity.is_null()) {
        if (verbosity.string() != "low" && verbosity.string() != "medium" &&
            verbosity.string() != "high") {
            throw RequestError(400, "Unsupported text verbosity", "text.verbosity");
        }
        options.emplace_back("verbosity", verbosity);
    }
    return options;
}

std::optional<std::string> request_id(const OrderedObject& fields) {
    const auto* value = json::get(fields, "request_id");
    if (!value) return {};
    const auto id = value->string();
    constexpr std::string_view safe =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-";
    if (!id || id->substr(0, 5) != "resp_" || id->size() <= 5 || id->size() > 128 ||
        id->find_first_not_of(safe, 5) != std::string_view::npos) {
        throw RequestError(400, "Expected a URL-safe response ID beginning with 'resp_' and "
                               "containing at most 128 characters", "request_id");
    }
    return std::string(*id);
}

std::pair<std::optional<std::string_view>, bool> configure_reasoning(
    const Value& reason, OrderedObject& thinking) {
    if (reason.is_null()) return {};
    const auto& reasoning = object(reason, "reasoning");
    const auto* level = json::get(reasoning, "effort");
    const auto effort = level ? level->string() : std::optional<std::string_view>("medium");
    if (level && !level->is_null() && effort != "none" && effort != "minimal" &&
        effort != "low" && effort != "medium" && effort != "high" && effort != "xhigh" &&
        effort != "ultra" && effort != "max") {
        throw RequestError(400, "Unsupported reasoning effort", "reasoning");
    }
    const auto& view = field(reasoning, "summary");
    if (!view.is_null() && view.string() != "auto" && view.string() != "concise" &&
        view.string() != "detailed") {
        throw RequestError(400, "Unsupported reasoning summary", "reasoning");
    }
    const bool disable = effort == "none" || ((!level || level->is_null()) &&
        field(reasoning, "reasoning_effort").string() == "none");
    if (disable && !json::get(thinking, "enable_thinking")) {
        thinking.emplace_back("enable_thinking", false);
    }
    if (effort && !json::get(thinking, "reasoning_effort")) {
        thinking.emplace_back("reasoning_effort", std::string(*effort));
    }
    return {effort, !view.is_null()};
}

void validate_storage(const ResponsesRequest& request) {
    if (!request.store) return;
    constexpr std::size_t envelope_overhead = 4096;
    if (chat_template_json(request.history).size() + chat_template_json(request.fields).size() >
        ResponseStore::record_bytes - envelope_overhead) {
        throw RequestError(413, store_limit, "input");
    }
}

ResponsesRequest parse_responses_request(std::string_view body, ResponseStore& store) {
    const auto parsed = json::parse(body, json::Syntax::python_request);
    const auto& fields = object(parsed, "body");
    validate_text_options(fields);
    const auto& requested_model = field(fields, "model");
    if (!requested_model.is_null() && !requested_model.string()) {
        throw RequestError(400, "Expected a model name", "model");
    }
    if (const auto name = requested_model.string(); name && !name->empty() && *name != model_name) {
        throw RequestError(404, "The model '" + std::string(*name) + "' does not exist", "model");
    }
    const auto& previous = field(fields, "previous_response_id");
    const auto background = flag(fields, "background", false);
    const auto* store_value = json::get(fields, "store");
    const auto persist = flag(fields, "store", !store_value);
    if (!store.enabled() && !previous.is_null()) {
        throw RequestError(400, store_disabled, "previous_response_id");
    }
    if (!store.enabled() && background) {
        throw RequestError(400, store_disabled, "background");
    }
    if (background && !persist) {
        throw RequestError(400, "background=true requires store=true.", "store");
    }
    const auto stream = flag(fields, "stream", false);
    const auto& instructions = field(fields, "instructions");
    const auto& metadata = field(fields, "metadata");
    Value response_format;
    auto text = text_options(field(fields, "text"), response_format);
    const auto& user = field(fields, "user");
    const auto& limit = field(fields, "max_output_tokens");
    const auto& reason = field(fields, "reasoning");
    const auto& kwargs = field(fields, "chat_template_kwargs");
    const auto& choice = field(fields, "tool_choice");
    const auto& truncation = field(fields, "truncation");
    if (!instructions.is_null() && !instructions.string()) {
        throw RequestError(400, "Expected text instructions", "instructions");
    }
    if (!user.is_null() && !user.string()) throw RequestError(400, "Expected a string", "user");
    if (!metadata.is_null()) object(metadata, "metadata");

    OrderedObject thinking = kwargs.is_null() ? OrderedObject{} : object(kwargs,
                                                                           "chat_template_kwargs");
    const auto [effort, summary] = configure_reasoning(reason, thinking);
    auto history = response_input(field(fields, "input"));
    if (!previous.is_null()) {
        const auto id = previous.string();
        if (!id) throw RequestError(400, "Expected a response ID", "previous_response_id");
        auto prior = store.history(*id);
        history.insert(history.begin(), prior.begin(), prior.end());
    }
    auto prompt = messages(history, instructions);
    OrderedObject chat{{"model", std::string(model_name)}, {"messages", std::move(prompt)},
                       {"stream", stream},
                       {"chat_template_kwargs", std::move(thinking)}};
    if (!response_format.is_null()) {
        chat.emplace_back("response_format", std::move(response_format));
    }
    if (!limit.is_null()) {
        const auto tokens = limit.unsigned_integer();
        if (!tokens) {
            throw RequestError(400, "Expected an integer output budget", "max_output_tokens");
        }
        if (*tokens < 2) {
            throw RequestError(400, "max_new_tokens must be at least 0, got " +
                                     std::to_string(static_cast<int>(*tokens) - 2) + ".");
        }
        chat.emplace_back("max_tokens", *tokens - 2);
    }
    for (const auto key : {"temperature", "top_p", "cache_salt", "frequency_penalty",
                           "presence_penalty", "stop", "top_k", "min_p", "repetition_penalty"}) {
        const auto& value = field(fields, key);
        if (!value.is_null()) chat.emplace_back(key, value);
    }
    if (field(fields, "temperature").is_null()) chat.emplace_back("temperature", 0.7);
    OrderedObject echo{{"parallel_tool_calls", flag(fields, "parallel_tool_calls", true)},
        {"tool_choice", choice.is_null() ? Value("auto") : choice}, {"tools", Array{}},
        {"instructions", instructions}, {"max_output_tokens", limit},
        {"previous_response_id", previous},
        {"reasoning", OrderedObject{{"effort", effort && effort != "none" ?
                                      Value(std::string(*effort)) : Value{}}, {"summary", {}}}},
        {"store", store_value && store_value->is_null() ? Value{} : Value(persist)},
        {"temperature", field(fields, "temperature")},
        {"text", std::move(text)}, {"top_p", field(fields, "top_p")},
        {"truncation", json::get(fields, "truncation") ? truncation : Value("disabled")},
        {"user", user}, {"metadata", metadata.is_null() ? Value(OrderedObject{}) : metadata}};
    ResponsesRequest request{std::move(chat), std::move(echo), std::move(history),
                             store.enabled() && persist ? &store : nullptr,
                             background && !stream, summary, request_id(fields)};
    validate_storage(request);
    return request;
}

}

void complete_responses(const Socket& connection, BatchService& service, int slot,
                        const CompletionTokenizer& tokenizer, std::string_view body,
                        ResponseStore& store) {
    try {
        const auto envelope = parse_responses_request(body, store);
        auto request = parse_chat_request(envelope.chat, tokenizer);
        request.responses = &envelope;
        complete_text(connection, request, tokenizer, [&](const auto& prompt, const auto& options,
                                                          const auto& callbacks) {
            return service.generate(slot, prompt, options, callbacks);
        });
    } catch (const RequestError& error) {
        responses_error(connection, error.code, error.what(), error.parameter);
    } catch (const json::ParseError&) {
        responses_error(connection, 400, "Invalid JSON request body");
    } catch (const std::invalid_argument& error) {
        responses_error(connection, 400, error.what());
    } catch (const HttpError& error) {
        responses_error(connection, 400, error.what());
    } catch (const std::exception& error) {
        std::cerr << "Responses request failed: " << error.what() << '\n';
        responses_error(connection, 500, "Internal server error");
    }
}

ResponsesOutput::ResponsesOutput(const Socket& connection, EventStream& stream,
                                 const ResponsesRequest& request, const TextRequest& text)
    : connection_(connection), stream_(stream), request_(request), text_(text),
      id_(request.id ? *request.id : responses_id("resp_")),
      created_(static_cast<std::int64_t>(std::time(nullptr))),
      reasoning_(text.chat_output.separate_reasoning && text.chat_output.prompt_has_open_think) {}

OrderedObject ResponsesOutput::envelope(const char* status, Value usage, Value error) const {
    OrderedObject output{{"id", id_}, {"object", "response"}, {"created_at", created_},
        {"model", std::string(model_name)}, {"output", output_}, {"status", status},
        {"usage", std::move(usage)},
        {"error", std::move(error)}, {"incomplete_details", std::string_view(status) == "incomplete"
            ? Value(OrderedObject{{"reason", "max_output_tokens"}}) : Value{}}};
    const bool text_only = !output_.empty() && std::all_of(output_.begin(), output_.end(),
        [](const auto& item) {
            return field(*item.ordered_object(), "type").string() == "message";
        });
    for (const auto& [key, value] : request_.fields) {
        if (key == "text" && value.is_null() && text_only) {
            output.emplace_back(key, OrderedObject{{"format",
                                                    OrderedObject{{"type", "text"}}}});
        } else {
            output.emplace_back(key, value);
        }
    }
    return output;
}

OrderedObject ResponsesOutput::part(const Item& item, bool summary) const {
    if (item.reasoning) {
        return {{"type", summary ? "summary_text" : "reasoning_text"}, {"text", item.text}};
    }
    return {{"type", "output_text"}, {"text", item.text}, {"annotations", Array{}},
            {"logprobs", {}}};
}

OrderedObject ResponsesOutput::item_json(const Item& item, bool ready) const {
    OrderedObject result{{"id", item.id}, {"type", item.reasoning ? "reasoning" : "message"},
                         {"status", ready ? "completed" : "in_progress"},
                         {"content", ready ? Array{part(item)} : Array{}}};
    if (item.reasoning) {
        result.emplace_back("summary", ready && request_.summary ? Array{part(item, true)} :
                                                                 Array{});
        result.emplace_back("encrypted_content", Value{});
    } else {
        result.emplace_back("role", "assistant");
        result.emplace_back("phase", "final_answer");
    }
    return result;
}

bool ResponsesOutput::event(std::string_view type, OrderedObject fields) {
    fields.emplace_back("type", std::string(type));
    fields.emplace_back("sequence_number", sequence_++);
    return stream_.write(chat_template_json(fields), type);
}

bool ResponsesOutput::lifecycle(std::string_view type, const char* status, Value usage,
                                Value error) {
    return event(type, {{"response", envelope(status, std::move(usage), std::move(error))}});
}

bool ResponsesOutput::start() {
    if (entry_) request_.store->update(entry_, envelope("in_progress"));
    return !text_.stream || stream_.started() ||
        (stream_.start() && lifecycle("response.created", "in_progress") &&
         lifecycle("response.in_progress", "in_progress"));
}

bool ResponsesOutput::queue() {
    const auto queued = envelope("queued");
    const auto serialized = chat_template_json(queued);
    entry_ = request_.store->insert(queued, serialized, chat_template_json(request_.history));
    if (!entry_) {
        responses_error(connection_, 503, store_full);
        return false;
    }
    if (write_json(connection_, "200 OK", serialized)) return true;
    fail();
    return false;
}

bool ResponsesOutput::cancelled() const { return entry_ && entry_->cancelled; }

OrderedObject ResponsesOutput::part_fields() const {
    return {{"item_id", item_->id}, {"output_index", count(output_.size())},
            {item_->reasoning && request_.summary ? "summary_index" : "content_index", 0}};
}

bool ResponsesOutput::open_item(bool reasoning) {
    if (!close_item()) return false;
    item_.emplace(Item{responses_id(reasoning ? "rs_" : "msg_"), {}, reasoning});
    if (!text_.stream) return true;
    auto fields = part_fields();
    fields.emplace_back("part", part(*item_, reasoning && request_.summary));
    return event("response.output_item.added", {{"output_index", count(output_.size())},
                 {"item", item_json(*item_, false)}}) &&
        event(reasoning && request_.summary ? "response.reasoning_summary_part.added" :
                                             "response.content_part.added", std::move(fields));
}

bool ResponsesOutput::close_item() {
    if (!item_) return true;
    auto completed = item_json(*item_, true);
    if (text_.stream) {
        const bool summary = item_->reasoning && request_.summary;
        auto done = part_fields();
        done.emplace_back("text", item_->text);
        if (!item_->reasoning) done.emplace_back("logprobs", Array{});
        auto fields = part_fields();
        fields.emplace_back("part", part(*item_, summary));
        if (!event(item_->reasoning ? (summary ? "response.reasoning_summary_text.done" :
             "response.reasoning_text.done") : "response.output_text.done", std::move(done)) ||
            !event(summary ? "response.reasoning_summary_part.done" :
                             "response.content_part.done", std::move(fields)) ||
            !event("response.output_item.done", {{"output_index", count(output_.size())},
                                                  {"item", completed}})) return false;
    }
    output_.emplace_back(std::move(completed));
    item_.reset();
    return true;
}

bool ResponsesOutput::accept(const ChatOutputEvent& delta) {
    if (delta.text.empty()) return true;
    const bool reasoning = delta.kind == ChatOutputKind::reasoning;
    if ((!item_ || item_->reasoning != reasoning) && !open_item(reasoning)) return false;
    item_->text += delta.text;
    if (!text_.stream) return true;
    auto fields = part_fields();
    fields.emplace_back("delta", std::string(delta.text));
    if (!reasoning) fields.emplace_back("logprobs", Array{});
    return event(reasoning ? (request_.summary ? "response.reasoning_summary_text.delta" :
        "response.reasoning_text.delta") : "response.output_text.delta", std::move(fields));
}

void ResponsesOutput::token(std::string_view piece) {
    if (!text_.chat_output.separate_reasoning) return;
    if (first_token_ && piece == "<think>") reasoning_ = true;
    first_token_ = false;
    if (reasoning_) ++reasoning_tokens_;
    if (piece == "</think>") reasoning_ = false;
}

void ResponsesOutput::finish(const char* reason, std::size_t tokens, std::size_t cached) {
    if (cancelled() || !start() || !close_item()) return;
    const auto prompt = text_.prompt.size();
    const auto incomplete = std::string_view(reason) == "length";
    const auto status = incomplete ? "incomplete" : "completed";
    OrderedObject usage{{"input_tokens", count(prompt)}, {"output_tokens", count(tokens)},
        {"total_tokens", count(prompt + tokens)}, {"input_tokens_details",
            OrderedObject{{"cached_tokens", count(cached)}, {"cache_write_tokens", 0}}},
        {"output_tokens_details", OrderedObject{{"reasoning_tokens", count(reasoning_tokens_)}}}};
    const auto finished = envelope(status, usage);
    const auto serialized = chat_template_json(finished);
    if (request_.store) {
        auto history = request_.history;
        history.insert(history.end(), output_.begin(), output_.end());
        auto stored_history = chat_template_json(history);
        if (entry_) {
            if (!request_.store->update(entry_, finished, std::move(stored_history))) {
                fail("response_store_limit", store_limit, 413);
            }
            return;
        }
        if (serialized.size() + stored_history.size() > ResponseStore::record_bytes) {
            fail("response_store_limit", store_limit, 413);
            return;
        }
        try {
            if (!request_.store->insert(finished, serialized, std::move(stored_history))) {
                fail("response_store_full", store_full, 503);
                return;
            }
        } catch (const RequestError& error) {
            fail("response_id_conflict", error.what(), error.code);
            return;
        }
    }
    if (text_.stream) {
        if (lifecycle(incomplete ? "response.incomplete" : "response.completed", status, usage)) {
            stream_.end();
        }
    } else {
        write_json(connection_, "200 OK", serialized);
    }
}

void ResponsesOutput::fail() {
    fail("server_error", "Internal server error", 500);
}

void ResponsesOutput::fail(const char* code, const char* message, unsigned status) {
    OrderedObject error{{"code", code}, {"message", message}};
    output_.clear();
    if (entry_) {
        request_.store->update(entry_, envelope("failed", {}, std::move(error)));
    } else if (!stream_.started()) {
        responses_error(connection_, status, message);
    } else if (lifecycle("response.failed", "failed", {}, std::move(error))) {
        stream_.end();
    }
}

}
