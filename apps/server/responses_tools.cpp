#include "sglang/server/responses.h"

#include "sglang/server/chat_template.h"

namespace sglang::server {
namespace {

using json::Array;
using json::OrderedObject;
using json::Value;

std::optional<std::string_view> text(const OrderedObject& fields, std::string_view key) {
    const auto* value = json::get(fields, key);
    return value ? value->string() : std::nullopt;
}

bool safe_name(std::string_view name) {
    constexpr std::string_view characters =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-";
    return !name.empty() && name.size() <= 64 &&
        name.find_first_not_of(characters) == std::string_view::npos;
}

OrderedObject named_function(const Array& tools, const Value* choice) {
    const auto* selected = choice ? choice->ordered_object() : nullptr;
    if (tools.size() != 1 || !selected || selected->size() != 2 ||
        text(*selected, "type") != "function") {
        throw std::invalid_argument(
            "Responses tools require one function and a named function choice");
    }
    const auto* tool = tools.front().ordered_object();
    if (!tool || text(*tool, "type") != "function") {
        throw std::invalid_argument("Only function tools are supported in Responses");
    }
    const auto name = text(*tool, "name");
    if (!name || !safe_name(*name) || name != text(*selected, "name")) {
        throw std::invalid_argument("Responses tool_choice must name the declared function");
    }
    const auto* strict = json::get(*tool, "strict");
    const auto* parameters = json::get(*tool, "parameters");
    if ((strict && strict->boolean() != true) || !parameters || !parameters->ordered_object()) {
        throw std::invalid_argument("Responses function tools require strict object parameters");
    }
    OrderedObject function{{"strict", true}};
    for (const auto& [key, value] : *tool) {
        if (key == "type" || key == "strict") continue;
        if (key != "name" && key != "parameters" && key != "description") {
            throw std::invalid_argument("Unsupported Responses function field: " + key);
        }
        if (key == "description" && !value.string()) {
            throw std::invalid_argument("Expected a string function description");
        }
        function.emplace_back(key, value);
    }
    return function;
}

OrderedObject history_call(const OrderedObject& item, std::string_view id) {
    const auto name = text(item, "name");
    const auto arguments = text(item, "arguments");
    if (!name || !safe_name(*name) || !arguments) {
        throw std::invalid_argument("Expected a function name and JSON arguments string");
    }
    Value typed;
    try {
        typed = parse_chat_arguments(*arguments, json::Syntax::ordered);
    } catch (const json::ParseError&) {
        throw std::invalid_argument("Tool-call arguments must contain a JSON object");
    }
    return {{"id", std::string(id)}, {"type", "function"}, {"function",
        OrderedObject{{"name", std::string(*name)}, {"arguments", std::move(typed)}}}};
}

void append_call(Array& messages, OrderedObject call) {
    if (messages.empty() || text(*messages.back().ordered_object(), "role") != "assistant") {
        messages.emplace_back(OrderedObject{{"role", "assistant"}});
    }
    auto& message = *messages.back().ordered_object();
    for (auto& [key, value] : message) {
        if (key != "tool_calls") continue;
        value.array()->emplace_back(std::move(call));
        return;
    }
    message.emplace_back("tool_calls", Array{std::move(call)});
}

}

void configure_responses_tool(const OrderedObject& fields, OrderedObject& chat) {
    const auto* tools = json::get(fields, "tools");
    const auto* choice = json::get(fields, "tool_choice");
    if (!tools || (tools->array() && tools->array()->empty())) {
        if (!choice || choice->string() == "auto" || choice->string() == "none") return;
        throw std::invalid_argument("Responses tool_choice requires a declared function");
    }
    if (!tools->array()) throw std::invalid_argument("Expected an array of Responses tools");
    auto function = named_function(*tools->array(), choice);
    const auto name = std::string(*text(function, "name"));
    chat.emplace_back("tools", Array{OrderedObject{{"type", "function"},
                                                   {"function", std::move(function)}}});
    chat.emplace_back("tool_choice", OrderedObject{{"type", "function"},
                                           {"function", OrderedObject{{"name", name}}}});
}

bool append_responses_tool_history(const OrderedObject& item, Array& messages,
                                   std::map<std::string, bool, std::less<>>& calls) {
    const auto type = text(item, "type");
    if (type != "function_call" && type != "function_call_output") return false;
    const auto id = text(item, "call_id");
    if (!id || id->empty()) throw std::invalid_argument("Expected a function call_id");
    if (type == "function_call") {
        if (!calls.emplace(std::string(*id), false).second) {
            throw std::invalid_argument("Duplicate function call_id");
        }
        append_call(messages, history_call(item, *id));
        return true;
    }
    const auto call = calls.find(*id);
    const auto output = text(item, "output");
    if (call == calls.end() || call->second) {
        throw std::invalid_argument("Expected an unanswered function call for call_id");
    }
    if (!output) throw std::invalid_argument("Expected a string function call output");
    call->second = true;
    messages.emplace_back(OrderedObject{{"role", "tool"}, {"tool_call_id", std::string(*id)},
                                        {"content", std::string(*output)}});
    return true;
}

}
