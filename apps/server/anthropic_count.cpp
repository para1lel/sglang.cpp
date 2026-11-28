#include "sglang/server/anthropic_count.h"

#include "sglang/server/completions.h"
#include "sglang/server/tokenization_validation.h"
#include "sglang/utf8.h"

#include <algorithm>
#include <cmath>

namespace sglang::server {
namespace {

using json::Array;
using json::OrderedObject;
using json::Value;
using namespace std::string_view_literals;
using request_validation::invalid;
using request_validation::optional_bool;

std::string_view text(const OrderedObject& object, std::string_view key) {
    const auto* value = supplied(object, key);
    return value ? value->string().value_or("") : "";
}

template <class Type>
const Type& structure(const Value* value, std::string_view field);

template <>
const OrderedObject& structure(const Value* value, std::string_view field) {
    if (!value) invalid(field, "Field required");
    if (const auto* object = value->ordered_object()) return *object;
    invalid(field, "Input should be a valid dictionary");
}

template <>
const Array& structure(const Value* value, std::string_view field) {
    if (!value) invalid(field, "Field required");
    if (const auto* list = value->array()) return *list;
    invalid(field, "Input should be a valid list");
}

std::string_view required_text(const OrderedObject& object, std::string_view key,
                               std::string_view field) {
    const auto* value = json::get(object, key);
    if (!value) invalid(field, "Field required");
    if (const auto string = value->string()) return *string;
    invalid(field, "Input should be a valid string");
}

void optional_text(const OrderedObject& object, std::string_view key) {
    if (const auto* value = supplied(object, key); value && !value->string()) {
        invalid(key, "Input should be a valid string");
    }
}

double integer(const Value& value, std::string_view field) {
    const auto number = chat_number(value);
    if (!number || !std::isfinite(*number) || std::trunc(*number) != *number) {
        invalid(field, "Input should be a valid integer");
    }
    return *number;
}

int task_budget_sign(const Value& value, std::string_view field) {
    if (const auto text = value.string()) {
        const auto number = detail::scan_integer_text(*text);
        if (!number) {
            invalid(field, "Input should be a valid integer, unable to parse string as an integer");
        }
        if (number->digit_count > 4300) {
            invalid(field, "Unable to parse input string as an integer, exceeded maximum size");
        }
        if (number->negative)
            return -1;
        return static_cast<int>(number->digit_count != 0);
    }
    if (const auto* wide = value.wide_integer()) return wide->digits.front() == '-' ? -1 : 1;
    if (const auto flag = value.boolean()) return *flag;
    if (const auto number = value.number()) {
        if (!std::isfinite(*number)) invalid(field, "Input should be a finite number");
        if (std::trunc(*number) != *number) {
            invalid(field, "Input should be a valid integer, got a number with a fractional part");
        }
        if (!value.signed_integer() && !value.unsigned_integer() &&
            (*number <= -0x1p63 || *number >= 0x1p63)) {
            invalid(field, "Unable to parse input string as an integer, exceeded maximum size");
        }
        return (*number > 0) - (*number < 0);
    }
    invalid(field, "Input should be a valid integer");
}

void validate_content(const Value* content, std::string_view field);

void validate_block(const Value& value, std::string_view field) {
    const auto& block = structure<OrderedObject>(&value, field);
    const auto kind = required_text(block, "type", std::string(field) + ".type");
    const auto path = std::string(field) + "." + std::string(kind);
    if (kind == "text") {
        required_text(block, "text", path + ".text");
    } else if (kind == "image") {
        if (const auto* source = supplied(block, "source");
            source && !source->string() && !source->ordered_object()) {
            invalid(path + ".source", "Expected a string or dictionary");
        }
    } else if (kind == "tool_use") {
        required_text(block, "id", path + ".id");
        required_text(block, "name", path + ".name");
        if (const auto* input = json::get(block, "input")) {
            structure<OrderedObject>(input, path + ".input");
        }
    } else if (kind == "tool_result") {
        optional_text(block, "tool_use_id");
        optional_text(block, "id");
        optional_bool(supplied(block, "is_error"), path + ".is_error");
        if (const auto* content = supplied(block, "content")) {
            validate_content(content, path + ".content");
        }
    } else if (kind == "tool_reference") {
        for (const auto key : {"name"sv, "tool_name"sv, "id"sv}) optional_text(block, key);
    } else if (kind == "search_result") {
        optional_text(block, "title");
        if (const auto* source = supplied(block, "source");
            source && !source->string() && !source->ordered_object()) {
            invalid(path + ".source", "Expected a string or dictionary");
        }
        if (const auto* content = supplied(block, "content")) {
            for (const auto& part : structure<Array>(content, path + ".content")) {
                structure<OrderedObject>(&part, path + ".content");
            }
        }
    } else if (kind == "thinking") {
        required_text(block, "thinking", path + ".thinking");
        optional_text(block, "signature");
    } else if (kind == "redacted_thinking") {
        optional_text(block, "data");
    } else {
        invalid(std::string(field), "Unsupported content block type");
    }
}

void validate_content(const Value* content, std::string_view field) {
    if (content && content->string()) return;
    const auto& parts = structure<Array>(content, field);
    for (std::size_t index = 0; index < parts.size(); ++index) {
        validate_block(parts[index], std::string(field) + "." + std::to_string(index));
    }
}

void append_line(std::string& output, std::string_view line) {
    if (line.empty()) return;
    if (!output.empty()) output += '\n';
    output += line;
}

void append_system(std::string& output, const Value& content, bool trim) {
    if (const auto string = content.string()) {
        if (!trim_chat_text(*string).empty()) append_line(output, trim ? trim_chat_text(*string)
                                                                     : *string);
        return;
    }
    for (const auto& value : *content.array()) {
        const auto& block = *value.ordered_object();
        if (text(block, "type") != "text") continue;
        const auto line = text(block, "text");
        append_line(output, trim ? trim_chat_text(line) : line);
    }
}

Value text_part(std::string value) {
    return OrderedObject{{"type", "text"}, {"text", std::move(value)}};
}

std::optional<Value> image_part(const OrderedObject& block, bool generation) {
    const auto* value = supplied(block, "source");
    const auto* source = value ? value->ordered_object() : nullptr;
    if (generation) {
        if (!source || text(*source, "type") != "base64" ||
            text(*source, "media_type") != "image/png" || text(*source, "data").empty()) {
            invalid("image.source", "Native Messages requires inline base64 image/png");
        }
        return OrderedObject{{"type", "image_url"},
            {"image_url", OrderedObject{
                {"url", "data:image/png;base64," + std::string(text(*source, "data"))}}}};
    }
    if (!source) return std::nullopt;
    const bool base64 = text(*source, "type") == "base64";
    const auto* payload = supplied(*source, base64 ? "data" : "url");
    if (!payload || !chat_truthy(*payload)) return std::nullopt;
    if (!base64 && !payload->string()) {
        invalid("image.source.url", "Input should be a valid string");
    }
    return OrderedObject{{"type", "image_url"},
                         {"image_url", OrderedObject{{"url", "image"}}}};
}

void append_search_value(std::string& output, const Value& value, bool quoted = false) {
    if (const auto string = value.string()) {
        const char quote = string->find('\'') != string->npos && string->find('"') == string->npos
            ? '"' : '\'';
        if (quoted) output += quote;
        for (const unsigned char character : *string) {
            if (!quoted) {
                output += static_cast<char>(character);
            } else if (character == '\n') {
                output += "\\n";
            } else if (character == '\r') {
                output += "\\r";
            } else if (character == '\t') {
                output += "\\t";
            } else if (character < 32 || character == 127) {
                constexpr std::string_view hex = "0123456789abcdef";
                output += "\\x";
                output += hex[character >> 4];
                output += hex[character & 15];
            } else {
                if (character == '\\' || character == quote) output += '\\';
                output += static_cast<char>(character);
            }
        }
        if (quoted) output += quote;
    } else if (const auto flag = value.boolean()) {
        output += *flag ? "True" : "False";
    } else if (value.is_null()) {
        output += "None";
    } else if (const auto* list = value.array()) {
        output += '[';
        for (const auto& item : *list) {
            if (&item != &list->front()) output += ", ";
            append_search_value(output, item, true);
        }
        output += ']';
    } else if (const auto* object = value.ordered_object()) {
        output += '{';
        for (const auto& field : *object) {
            if (&field != &object->front()) output += ", ";
            append_search_value(output, field.first, true);
            output += ": ";
            append_search_value(output, field.second, true);
        }
        output += '}';
    } else if (const auto number = value.number(); number && std::isinf(*number)) {
        output += *number < 0 ? "-inf" : "inf";
    } else {
        output += chat_template_json(value);
    }
}

std::string search_text(const OrderedObject& block) {
    std::string output;
    if (const auto title = text(block, "title"); !title.empty()) {
        append_line(output, "Title: " + std::string(title));
    }
    if (const auto* source = supplied(block, "source")) {
        auto* name = source;
        if (const auto* object = source->ordered_object()) {
            name = supplied(*object, "url");
            if (!name || !chat_truthy(*name)) name = supplied(*object, "text");
        }
        if (name && chat_truthy(*name)) {
            std::string label = "Source: ";
            append_search_value(label, *name);
            append_line(output, label);
        }
    }
    std::string contents;
    if (const auto* content = supplied(block, "content")) {
        for (const auto& part : *content->array()) {
            const auto& object = *part.ordered_object();
            const auto* snippet = supplied(object, "text");
            if (text(object, "type") != "text" || !snippet || !chat_truthy(*snippet)) continue;
            if (!snippet->string()) {
                throw std::invalid_argument("Search result text must be a string");
            }
            append_line(contents, *snippet->string());
        }
    }
    if (!contents.empty()) append_line(output, "Content: " + contents);
    return output;
}

Value collapse(Array parts) {
    if (parts.size() == 1) {
        if (const auto* value = json::get(*parts.front().ordered_object(), "text")) return *value;
    }
    return parts;
}

void emit(Array& messages, std::string_view role, Value content) {
    messages.emplace_back(OrderedObject{{"role", std::string(role)},
                                        {"content", std::move(content)}});
}

struct ToolResult {
    Array contents;
    std::string text;
};

ToolResult tool_result(const Value* content, bool forward_parts, bool generation) {
    if (!content || content->string()) {
        std::string text(content ? *content->string() : "");
        return {{Value(text)}, std::move(text)};
    }
    ToolResult result;
    Array group;
    bool references = false;
    for (const auto& item : *content->array()) {
        const auto& block = *item.ordered_object();
        const auto kind = text(block, "type");
        std::optional<Value> part;
        if (kind == "text" || kind == "search_result") {
            const auto string = kind == "text" ? std::string(text(block, "text"))
                                                : search_text(block);
            if (!string.empty()) {
                append_line(result.text, string);
                if (forward_parts) part = text_part(string);
            }
        } else if (forward_parts && kind == "image") {
            part = image_part(block, generation);
        } else if (forward_parts && kind == "tool_reference") {
            if (generation) invalid("tool_reference", "Deferred tool references are unsupported");
            auto name = text(block, "tool_name");
            if (name.empty()) name = text(block, "name");
            if (!name.empty()) {
                part = OrderedObject{{"type", "tool_reference"}, {"name", std::string(name)}};
            }
        } else if (generation) invalid(kind, "Unsupported tool result content block");
        if (!part) continue;
        const bool reference = kind == "tool_reference";
        if (!group.empty() && references != reference) {
            result.contents.emplace_back(collapse(std::move(group)));
            group.clear();
        }
        references = reference;
        group.push_back(std::move(*part));
    }
    if (!group.empty()) result.contents.emplace_back(collapse(std::move(group)));
    if (forward_parts && result.contents.empty()) result.contents.emplace_back("");
    return result;
}

Value tool_call(const OrderedObject& block) {
    const auto* input = json::get(block, "input");
    const auto name = text(block, "name");
    const auto id = text(block, "id");
    OrderedObject function{{"name", std::string(name)},
                           {"arguments", input ? chat_template_json(*input) : "{}"}};
    return OrderedObject{{"type", "function"}, {"id", std::string(id)},
                         {"function", std::move(function)}};
}

std::string history_thinking(const Array& blocks, bool reasoning_parser) {
    std::string reasoning;
    for (const auto& value : blocks) {
        const auto& block = *value.ordered_object();
        const auto kind = text(block, "type");
        if (kind == "redacted_thinking") {
            throw std::invalid_argument("Anthropic redacted_thinking history is not supported");
        }
        if (kind == "thinking" && reasoning_parser) {
            append_line(reasoning, text(block, "thinking"));
        }
    }
    return reasoning.empty() ? "" : "<think>" + reasoning + "\n</think>";
}

void convert_turn(Array& messages, std::string_view role, const Value& content,
                  bool reasoning_parser, bool generation) {
    if (content.string()) {
        emit(messages, role, content);
        return;
    }
    const auto& blocks = *content.array();
    Array parts;
    Array calls;
    if (role == "assistant") {
        auto reasoning = history_thinking(blocks, reasoning_parser);
        if (!reasoning.empty()) parts.emplace_back(text_part(std::move(reasoning)));
    }
    for (const auto& value : blocks) {
        const auto& block = *value.ordered_object();
        const auto kind = text(block, "type");
        if (kind == "text") {
            parts.emplace_back(text_part(std::string(text(block, "text"))));
        } else if (kind == "image") {
            if (auto part = image_part(block, generation)) parts.push_back(std::move(*part));
        } else if (kind == "search_result") {
            auto search = search_text(block);
            if (!search.empty()) parts.emplace_back(text_part(std::move(search)));
        } else if (kind == "tool_use") {
            if (generation && role != "assistant") {
                invalid(kind, "Tool calls require an assistant turn");
            }
            calls.emplace_back(tool_call(block));
        } else if (kind == "tool_result") {
            auto result = tool_result(supplied(block, "content"), role == "user", generation);
            if (role == "user") {
                if (!parts.empty()) emit(messages, role, collapse(std::move(parts)));
                parts.clear();
                for (auto& result_content : result.contents) {
                    emit(messages, "tool", std::move(result_content));
                    if (generation) {
                        auto id = text(block, "tool_use_id");
                        if (id.empty()) id = text(block, "id");
                        messages.back().ordered_object()->emplace_back("tool_call_id",
                                                                       std::string(id));
                    }
                }
            } else {
                parts.emplace_back(text_part("Tool result: " + result.text));
            }
        } else if (generation && kind != "thinking") {
            invalid(kind, "Unsupported Messages content block");
        } else if (generation && (role != "assistant" || !reasoning_parser)) {
            invalid(kind, "Thinking history requires an assistant turn and a reasoning parser");
        }
    }
    if (role == "user") {
        if (!parts.empty()) emit(messages, role, collapse(std::move(parts)));
        else if (!calls.empty()) invalid("messages.user.content", "Field required");
        return;
    }
    OrderedObject turn{{"role", std::string(role)}};
    if (!parts.empty() || calls.empty()) {
        turn.emplace_back("content", parts.empty() ? Value("") : collapse(std::move(parts)));
    }
    if (!calls.empty()) turn.emplace_back("tool_calls", std::move(calls));
    messages.emplace_back(std::move(turn));
}

Array convert_messages(const OrderedObject& request, bool reasoning_parser, bool generation) {
    Array messages;
    std::string system_text;
    if (const auto* system = supplied(request, "system")) {
        validate_content(system, "system");
        append_system(system_text, *system, false);
    }
    const auto& turns = structure<Array>(json::get(request, "messages"), "messages");
    for (std::size_t index = 0; index < turns.size(); ++index) {
        const auto field = "messages." + std::to_string(index);
        const auto& turn = structure<OrderedObject>(&turns[index], field);
        const auto role = required_text(turn, "role", field + ".role");
        if (role != "user" && role != "assistant" && role != "system") {
            invalid(field + ".role", "Input should be 'user', 'assistant' or 'system'");
        }
        const auto* content = json::get(turn, "content");
        validate_content(content, field + ".content");
        if (role == "system") append_system(system_text, *content, true);
    }
    if (!system_text.empty()) emit(messages, "system", std::move(system_text));
    for (const auto& value : turns) {
        const auto& turn = *value.ordered_object();
        const auto role = text(turn, "role");
        if (role != "system") {
            convert_turn(messages, role, *json::get(turn, "content"), reasoning_parser, generation);
        }
    }
    return messages;
}

bool server_tool_decimal(char32_t scalar) {
    constexpr std::pair<char32_t, char32_t> decimal_ranges[] = {
        {0x30, 0x39}, {0x660, 0x669}, {0x6f0, 0x6f9}, {0x7c0, 0x7c9},
        {0x966, 0x96f}, {0x9e6, 0x9ef}, {0xa66, 0xa6f}, {0xae6, 0xaef},
        {0xb66, 0xb6f}, {0xbe6, 0xbef}, {0xc66, 0xc6f}, {0xce6, 0xcef},
        {0xd66, 0xd6f}, {0xde6, 0xdef}, {0xe50, 0xe59}, {0xed0, 0xed9},
        {0xf20, 0xf29}, {0x1040, 0x1049}, {0x1090, 0x1099}, {0x17e0, 0x17e9},
        {0x1810, 0x1819}, {0x1946, 0x194f}, {0x19d0, 0x19d9}, {0x1a80, 0x1a89},
        {0x1a90, 0x1a99}, {0x1b50, 0x1b59}, {0x1bb0, 0x1bb9}, {0x1c40, 0x1c49},
        {0x1c50, 0x1c59}, {0xa620, 0xa629}, {0xa8d0, 0xa8d9}, {0xa900, 0xa909},
        {0xa9d0, 0xa9d9}, {0xa9f0, 0xa9f9}, {0xaa50, 0xaa59}, {0xabf0, 0xabf9},
        {0xff10, 0xff19}, {0x104a0, 0x104a9}, {0x10d30, 0x10d39}, {0x10d40, 0x10d49},
        {0x11066, 0x1106f}, {0x110f0, 0x110f9}, {0x11136, 0x1113f}, {0x111d0, 0x111d9},
        {0x112f0, 0x112f9}, {0x11450, 0x11459}, {0x114d0, 0x114d9}, {0x11650, 0x11659},
        {0x116c0, 0x116c9}, {0x116d0, 0x116e3}, {0x11730, 0x11739}, {0x118e0, 0x118e9},
        {0x11950, 0x11959}, {0x11bf0, 0x11bf9}, {0x11c50, 0x11c59}, {0x11d50, 0x11d59},
        {0x11da0, 0x11da9}, {0x11f50, 0x11f59}, {0x16130, 0x16139}, {0x16a60, 0x16a69},
        {0x16ac0, 0x16ac9}, {0x16b50, 0x16b59}, {0x16d70, 0x16d79}, {0x1ccf0, 0x1ccf9},
        {0x1d7ce, 0x1d7ff}, {0x1e140, 0x1e149}, {0x1e2f0, 0x1e2f9}, {0x1e4f0, 0x1e4f9},
        {0x1e5f1, 0x1e5fa}, {0x1e950, 0x1e959}, {0x1fbf0, 0x1fbf9},
    };
    const auto last = std::end(decimal_ranges);
    const auto range = std::lower_bound(std::begin(decimal_ranges), last, scalar,
        [](const auto& range, char32_t point) { return range.second < point; });
    return range != last && range->first <= scalar;
}

bool server_tool(const OrderedObject& tool, std::string_view field) {
    const auto kind = text(tool, "type");
    for (const auto family : {"web_search"sv, "computer"sv, "bash"sv, "text_editor"sv}) {
        const auto prefix = std::string(family) + '_';
        if (kind.compare(0, prefix.size(), prefix) != 0) continue;
        const auto suffix = kind.substr(prefix.size());
        const auto digits = suffix.size() >= 8 && suffix.size() <= 32
            ? read_utf8(suffix) : std::u32string();
        if (digits.size() != 8 || !std::all_of(digits.begin(), digits.end(), server_tool_decimal)) {
            invalid("tools.type", "Expected a server tool type with an eight-digit version");
        }
        const auto name = family == "text_editor" || json::get(tool, "name")
            ? required_text(tool, "name", std::string(field) + ".name") : family;
        if (family == "text_editor" ? name != "str_replace_editor" &&
            name != "str_replace_based_edit_tool" : name != family) {
            invalid("tools.name", "Invalid server tool name");
        }
        if (family == "computer") {
            for (const auto key :
                 {"display_width_px"sv, "display_height_px"sv, "display_number"sv}) {
                if (const auto* value = supplied(tool, key)) integer(*value, key);
            }
        }
        if (family == "web_search") {
            if (const auto* value = supplied(tool, "max_uses")) integer(*value, "max_uses");
            for (const auto key : {"allowed_domains"sv, "blocked_domains"sv}) {
                if (const auto* value = supplied(tool, key)) {
                    for (const auto& domain : structure<Array>(value, key)) {
                        if (!domain.string()) invalid(key, "Input should be a valid string");
                    }
                }
            }
        }
        return true;
    }
    return false;
}

ChatTools convert_tools(const OrderedObject& request, bool generation) {
    struct NamedTool {
        std::string_view name;
        Value value;
    };
    std::vector<NamedTool> tools;
    if (const auto* list = supplied(request, "tools")) {
        const auto& values = structure<Array>(list, "tools");
        for (std::size_t index = 0; index < values.size(); ++index) {
            const auto field = "tools." + std::to_string(index);
            const auto& tool = structure<OrderedObject>(&values[index], field);
            optional_text(tool, "type");
            optional_text(tool, "description");
            const auto* defer = supplied(tool, "defer_loading");
            const auto deferred = optional_bool(defer, field + ".defer_loading");
            if (server_tool(tool, field)) continue;
            if (generation && deferred) invalid(field, "Deferred custom tools are unsupported");
            const auto name = required_text(tool, "name", field + ".name");
            if (const auto* type = supplied(tool, "type"); type && type->string() != "custom") {
                invalid(field + ".custom.type", "Input should be 'custom'");
            }
            auto schema = structure<OrderedObject>(json::get(tool, "input_schema"),
                                                   field + ".custom.input_schema");
            if (!json::get(schema, "type")) schema.emplace_back("type", "object");
            const auto strict = optional_bool(json::get(tool, "strict"), field + ".custom.strict");
            OrderedObject function{{"name", std::string(name)},
                                   {"description", std::string(text(tool, "description"))},
                                   {"parameters", std::move(schema)}, {"strict", strict}};
            if (defer) function.emplace_back("defer_loading", deferred);
            OrderedObject converted{{"type", "function"}, {"function", std::move(function)}};
            if (defer) converted.emplace_back("defer_loading", deferred);
            tools.push_back({name, std::move(converted)});
        }
    }
    ChatTools selected;
    std::string_view name;
    std::string_view kind = "auto";
    if (const auto* choice = supplied(request, "tool_choice")) {
        const auto& object = structure<OrderedObject>(choice, "tool_choice");
        kind = required_text(object, "type", "tool_choice.type");
        if (kind != "auto" && kind != "none" && kind != "any" && kind != "tool") {
            invalid("tool_choice.type", "Input should be 'auto', 'any', 'tool' or 'none'");
        }
        optional_text(object, "name");
        const auto parallel = json::get(object, "disable_parallel_tool_use");
        const bool disabled = optional_bool(parallel, "tool_choice.disable_parallel_tool_use");
        if (generation) selected.parallel = !disabled;
        if (kind == "none") return selected;
        if ((kind == "any" || kind == "tool") && tools.empty()) {
            throw std::invalid_argument("tool_choice='" + std::string(kind) +
                "' requires at least one custom tool; all supplied tools were server-side "
                "Anthropic built-ins which the OpenAI-compatible backend cannot invoke");
        }
        name = text(object, "name");
        if (kind == "tool" && (!supplied(object, "name") ||
            std::none_of(tools.begin(), tools.end(),
                         [&](const auto& tool) { return tool.name == name; }))) {
            const auto label = supplied(object, "name") ? "'" + std::string(name) + "'" : "None";
            throw std::invalid_argument("tool_choice references tool " + label +
                " but it is not in the forwarded tools list "
                "(server-side Anthropic tools cannot be selected)");
        }
    }
    if (generation && kind == "any") selected.choice = ChatToolChoice::required;
    if (generation && kind == "tool") selected.choice = ChatToolChoice::named;
    for (auto& tool : tools) {
        if (generation || kind != "tool" || tool.name == name) {
            selected.tools.push_back(std::move(tool.value));
        }
    }
    return selected;
}

bool thinking_enabled(const OrderedObject& request, bool reasoning_parser) {
    const auto* value = supplied(request, "thinking");
    if (!value) return true;
    const auto& thinking = structure<OrderedObject>(value, "thinking");
    const auto type = required_text(thinking, "type", "thinking.type");
    if (type != "enabled" && type != "disabled" && type != "adaptive") {
        invalid("thinking.type", "Input should be 'enabled', 'disabled' or 'adaptive'");
    }
    if (const auto* budget = supplied(thinking, "budget_tokens")) {
        const auto amount = integer(*budget, "thinking.budget_tokens");
        if (type != "enabled") {
            invalid("thinking", "Value error, thinking.budget_tokens is not allowed when "
                                "thinking.type is '" + std::string(type) + "'");
        }
        if (amount < 1024) {
            invalid("thinking", "Value error, thinking.budget_tokens must be >= 1024 (got " +
                std::to_string(static_cast<std::int64_t>(std::max(amount, -0x1p63))) + ")");
        }
    }
    if (const auto* display = supplied(thinking, "display")) {
        if (display->string() != "summarized" && display->string() != "omitted" &&
            display->string() != "updates") {
            invalid("thinking.display", "Input should be 'summarized', 'omitted' or 'updates'");
        }
        if (type == "disabled") {
            invalid("thinking", "Value error, thinking.display is not allowed when thinking.type "
                                "is 'disabled'");
        }
    }
    if (type != "disabled" && !reasoning_parser) {
        throw std::invalid_argument(
            "Anthropic thinking is not supported for models without a reasoning parser");
    }
    return !reasoning_parser || type != "disabled";
}

std::string_view output_effort(const OrderedObject& request) {
    const auto* value = supplied(request, "output_config");
    if (!value) return {};
    const auto& output = structure<OrderedObject>(value, "output_config");
    const auto effort = text(output, "effort");
    if (supplied(output, "effort") && effort != "minimal" && effort != "low" &&
        effort != "medium" && effort != "high" && effort != "xhigh" && effort != "max") {
        invalid("output_config.effort",
                "Input should be 'minimal', 'low', 'medium', 'high', 'xhigh' or 'max'");
    }
    if (const auto* budget = supplied(output, "task_budget")) {
        const auto& task = structure<OrderedObject>(budget, "output_config.task_budget");
        const auto type = required_text(task, "type", "output_config.task_budget.type");
        if (type != "tokens") invalid("output_config.task_budget.type", "Input should be 'tokens'");
        const auto* total = json::get(task, "total");
        if (!total) invalid("output_config.task_budget.total", "Field required");
        if (task_budget_sign(*total, "output_config.task_budget.total") <= 0) {
            invalid("output_config.task_budget.total", "Input should be greater than 0");
        }
        if (const auto* remaining = supplied(task, "remaining");
            remaining && task_budget_sign(*remaining, "output_config.task_budget.remaining") < 0) {
            invalid("output_config.task_budget.remaining",
                    "Input should be greater than or equal to 0");
        }
    }
    if (const auto* configuration = supplied(output, "format")) {
        const auto& format = structure<OrderedObject>(configuration, "output_config.format");
        const auto type = required_text(format, "type", "output_config.format.type");
        if (type != "json_schema") {
            invalid("output_config.format.type", "Input should be 'json_schema'");
        }
        structure<OrderedObject>(json::get(format, "schema"), "output_config.format.schema");
    }
    return effort == "xhigh" ? "max" : effort;
}

AnthropicCountChat convert_chat(const Value& value, bool reasoning_parser, bool generation) {
    const auto* object = value.ordered_object();
    if (!object) {
        throw std::invalid_argument(value.is_null() ? "Field required"
            : "Input should be a valid dictionary or object to extract fields from");
    }
    const auto& request = *object;
    const auto model = required_text(request, "model", "model");
    if (model.empty()) invalid("model", "Value error, Model is required");
    auto messages = convert_messages(request, reasoning_parser, generation);
    if (const auto* betas = supplied(request, "betas")) {
        for (const auto& beta : structure<Array>(betas, "betas")) {
            if (!beta.string()) invalid("betas", "Input should be a valid string");
        }
    }
    const bool thinking = thinking_enabled(request, reasoning_parser);
    const auto effort = output_effort(request);
    auto tools = convert_tools(request, generation);
    OrderedObject options{{"chat_template_kwargs", OrderedObject{{"enable_thinking", thinking}}}};
    if (!effort.empty()) options.emplace_back("reasoning_effort", std::string(effort));
    return {std::move(messages), std::move(tools), std::move(options)};
}

void add_generation_options(OrderedObject& chat, const OrderedObject& request) {
    const auto* budget = json::get(request, "max_tokens");
    if (!budget) invalid("max_tokens", "Field required");
    const auto maximum = integer(*budget, "max_tokens");
    if (maximum <= 0) invalid("max_tokens", "Value error, max_tokens must be positive");
    if (maximum > 262144) invalid("max_tokens", "Expected at most 262144");
    chat.emplace_back("max_tokens", static_cast<int>(maximum));
    const auto stream = optional_bool(supplied(request, "stream"), "stream");
    chat.emplace_back("stream", stream);
    for (const auto key : {"temperature"sv, "top_p"sv}) {
        if (const auto* value = supplied(request, key)) {
            const auto number = chat_number(*value);
            if (!number) invalid(key, "Input should be a valid number");
            chat.emplace_back(key, *number);
        }
    }
    if (const auto* value = supplied(request, "top_k")) {
        const auto number = integer(*value, "top_k");
        if (number < -1 || number >= 0x1p63) invalid("top_k", "Expected a supported top_k");
        chat.emplace_back("top_k", static_cast<std::int64_t>(number));
    }
    if (const auto* stops = supplied(request, "stop_sequences")) {
        for (const auto& stop : structure<Array>(stops, "stop_sequences")) {
            if (!stop.string()) invalid("stop_sequences", "Input should be a valid string");
        }
        chat.emplace_back("stop", *stops);
    }
    if (const auto* metadata = supplied(request, "metadata")) {
        structure<OrderedObject>(metadata, "metadata");
    }
    optional_text(request, "cache_salt");
    if (const auto* salt = supplied(request, "cache_salt")) chat.emplace_back("cache_salt", *salt);
    for (const auto key : {"bootstrap_host"sv, "bootstrap_port"sv, "bootstrap_room"sv}) {
        if (supplied(request, key)) invalid(key, "Disaggregated routing is unsupported");
    }
    const auto* config = supplied(request, "output_config");
    const auto* format = config ? supplied(*config->ordered_object(), "format") : nullptr;
    if (format) {
        chat.emplace_back("response_format", OrderedObject{
            {"type", "json_schema"}, {"json_schema", OrderedObject{
                {"name", "response"}, {"strict", true},
                {"schema", *json::get(*format->ordered_object(), "schema")}}}});
    }
}

}

AnthropicCountChat anthropic_count_chat(const Value& value, bool reasoning_parser) {
    return convert_chat(value, reasoning_parser, false);
}

OrderedObject anthropic_messages_chat(const Value& value, ChatParsers parsers) {
    auto converted = convert_chat(value, parsers.qwen3_reasoning, true);
    if (converted.messages.empty()) throw std::invalid_argument("Messages cannot be empty.");
    const auto& request = *value.ordered_object();
    auto& chat = converted.options;
    chat.emplace_back("model", *json::get(request, "model"));
    chat.emplace_back("messages", std::move(converted.messages));
    add_generation_options(chat, request);
    if (!converted.tools.tools.empty()) {
        if (!parsers.qwen3_coder_tools) invalid("tools", "Requires --tool-call-parser qwen3_coder");
        chat.emplace_back("tools", std::move(converted.tools.tools));
        chat.emplace_back("parallel_tool_calls", converted.tools.parallel);
        if (converted.tools.choice == ChatToolChoice::required) {
            chat.emplace_back("tool_choice", "required");
        } else if (converted.tools.choice == ChatToolChoice::named) {
            const auto name = text(*supplied(request, "tool_choice")->ordered_object(), "name");
            chat.emplace_back("tool_choice", OrderedObject{{"type", "function"},
                {"function", OrderedObject{{"name", std::string(name)}}}});
        }
    }
    return std::move(chat);
}

}
