#include "sglang/server/serving/qwen_parser.h"
#include "sglang/server/serving/qwen_serving_profile.h"

#include "sglang/server/tool_literal.h"
#include "sglang/server/tool_union.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <limits>
#include <locale>
#include <map>
#include <sstream>

namespace sglang::model {
namespace {

constexpr std::string_view tool_open = "<tool_call>";
constexpr std::string_view tool_close = "</tool_call>";
constexpr std::string_view function_open = "<function=";
constexpr std::string_view function_close = "</function>";
constexpr std::string_view parameter_open = "<parameter=";
constexpr std::string_view parameter_close = "</parameter>";
constexpr std::array schema_branches{"anyOf", "oneOf", "allOf"};

const json::Value* field(const json::Value& value, std::string_view key) {
    const auto* object = value.ordered_object();
    return object ? json::get(*object, key) : nullptr;
}

std::string_view string_or(const json::Value* value, std::string_view fallback) {
    return value ? value->string().value_or(fallback) : fallback;
}

bool starts_with(std::string_view text, std::string_view prefix) {
    return text.substr(0, prefix.size()) == prefix;
}

bool ends_with_newline(std::string_view text) {
    return !text.empty() && text.back() == '\n';
}

std::string_view trim(std::string_view text) {
    constexpr std::string_view whitespace = " \t\r\n\f\v";
    const auto first = text.find_first_not_of(whitespace);
    if (first == text.npos) {
        return {};
    }
    return text.substr(first, text.find_last_not_of(whitespace) - first + 1);
}

std::string lowercase(std::string_view text) {
    std::string lower(text);
    for (auto& character : lower) {
        if (character >= 'A' && character <= 'Z') {
            character += 'a' - 'A';
        }
    }
    return lower;
}

enum class Conversion { string, boolean, integer, number, structured, structured_union };

struct Parameter {
    const json::Value* schema;
    const json::Value* root;
    std::optional<Conversion> conversion;
};

using Properties = std::map<std::string_view, Parameter, std::less<>>;
using ToolSchemas = std::map<std::string_view, Properties, std::less<>>;

void insert_properties(Properties& target, const json::OrderedObject& properties,
                       const json::Value& root) {
    for (const auto& [name, schema] : properties) {
        target.try_emplace(name, Parameter{&schema, &root, std::nullopt});
    }
}

void collect_properties(Properties& target, const json::Value& schema, const json::Value& root) {
    if (const auto* properties = field(schema, "properties");
        properties && properties->ordered_object()) {
        insert_properties(target, *properties->ordered_object(), root);
        return;
    }
    for (const auto keyword : schema_branches) {
        if (const auto* branches = field(schema, keyword); branches && branches->array()) {
            for (const auto& branch : *branches->array()) {
                collect_properties(target, branch, root);
            }
        }
    }
}

void optional_string(const json::Value& value, std::string_view key) {
    if (const auto* entry = field(value, key); entry && !entry->is_null() && !entry->string()) {
        throw std::invalid_argument(std::string(key) + ": Input should be a valid string");
    }
}

ToolSchemas read_tools(const json::Array& tools) {
    ToolSchemas schemas;
    for (const auto& tool : tools) {
        const auto* function = field(tool, "function");
        if (!function || !function->ordered_object()) {
            throw std::invalid_argument("tools.function: Input should be a valid dictionary");
        }
        const auto* type = field(tool, "type");
        if (type && !type->string()) {
            throw std::invalid_argument("tools.type: Input should be a valid string");
        }
        optional_string(*function, "name");
        optional_string(*function, "description");
        const auto* parameters = field(*function, "parameters");
        if (parameters && !parameters->is_null() && !parameters->ordered_object()) {
            throw std::invalid_argument("tools.function.parameters: Expected a dictionary");
        }
        const auto* name = field(*function, "name");
        if (string_or(type, "function") != "function" || !name || name->is_null()) {
            continue;
        }
        auto [entry, inserted] = schemas.try_emplace(*name->string());
        if (inserted && parameters && parameters->ordered_object()) {
            auto& properties = entry->second;
            collect_properties(properties, *parameters, *parameters);
            if (properties.empty() && !field(*parameters, "properties")) {
                insert_properties(properties, *parameters->ordered_object(), *parameters);
            }
        }
    }
    return schemas;
}

std::string_view value_type(const json::Value& value) {
    if (value.is_null()) return "null";
    if (value.boolean()) return "boolean";
    if (value.string()) return "string";
    if (value.signed_integer() || value.unsigned_integer() || value.wide_integer()) {
        return "integer";
    }
    if (value.number()) return "number";
    if (value.array()) return "array";
    return "object";
}

std::string_view inferred_type(const json::Value& schema);

std::string_view union_type(const json::Array& branches) {
    std::string_view first;
    std::string_view non_null;
    bool multiple_non_null = false;
    bool has_null = false;
    bool has_string = false;
    for (const auto& branch : branches) {
        const auto type = inferred_type(branch);
        if (type.empty()) {
            continue;
        }
        if (first.empty()) first = type;
        if (type == "null") {
            has_null = true;
        } else {
            multiple_non_null |= !non_null.empty() && non_null != type;
            non_null = type;
            has_string |= type == "string";
        }
    }
    if (has_null && !multiple_non_null && !non_null.empty()) {
        return non_null;
    }
    return has_string ? "string" : first;
}

std::string_view inferred_type(const json::Value& schema) {
    if (const auto* type = field(schema, "type")) {
        if (const auto text = type->string()) {
            return *text;
        }
        if (const auto* types = type->array(); types && !types->empty()) {
            for (const auto& member : *types) {
                if (member.string() != "null") {
                    return member.string().value_or("string");
                }
            }
            return "string";
        }
    }
    const auto* branches = field(schema, "anyOf");
    if (!branches || branches->is_null() || branches->boolean() == false ||
        branches->string() == "" || branches->number() == 0 ||
        (branches->array() && branches->array()->empty()) ||
        (branches->ordered_object() && branches->ordered_object()->empty())) {
        branches = field(schema, "oneOf");
    }
    if (branches && branches->array()) {
        const auto type = union_type(*branches->array());
        if (!type.empty()) return type;
    }
    if (const auto* choices = field(schema, "enum"); choices && choices->array()) {
        const auto& values = *choices->array();
        if (!values.empty()) {
            const auto type = value_type(values.front());
            if (std::all_of(values.begin(), values.end(),
                            [type](const auto& value) { return value_type(value) == type; })) {
                return type;
            }
        }
        return "string";
    }
    if (const auto* all = field(schema, "allOf"); all && all->array()) {
        for (const auto& branch : *all->array()) {
            const auto type = inferred_type(branch);
            if (!type.empty() && type != "string") return type;
        }
        return "string";
    }
    if (field(schema, "properties")) return "object";
    if (field(schema, "items")) return "array";
    return {};
}

Conversion conversion_for(const json::Value& schema) {
    const auto type = lowercase(trim(inferred_type(schema)));
    if (type == "boolean" || type == "bool" || type == "binary") {
        return Conversion::boolean;
    }
    if (starts_with(type, "int") || starts_with(type, "uint") || starts_with(type, "long") ||
        starts_with(type, "short") || starts_with(type, "unsigned")) {
        return Conversion::integer;
    }
    if (starts_with(type, "num") || starts_with(type, "float")) {
        return Conversion::number;
    }
    if (type == "object" || type == "array" || type == "arr" ||
        starts_with(type, "dict") || starts_with(type, "list")) {
        return Conversion::structured;
    }
    if ((type == "string" || type == "str" || type == "text" || type == "varchar" ||
         type == "char" || type == "enum") && (field(schema, "anyOf") || field(schema, "oneOf"))) {
        return Conversion::structured_union;
    }
    return Conversion::string;
}

std::string_view float_spelling(double value, char (&buffer)[64]) {
    const auto magnitude = std::abs(value);
    const auto format = (magnitude == 0 || (magnitude >= 1e-4 && magnitude < 1e16))
        ? std::chars_format::fixed : std::chars_format::scientific;
    const auto result = std::to_chars(buffer, buffer + sizeof(buffer), value, format);
    return {buffer, static_cast<std::size_t>(result.ptr - buffer)};
}

void append_float(std::string& output, double value) {
    if (std::isnan(value)) {
        output += "NaN";
        return;
    }
    if (std::isinf(value)) {
        output += value < 0 ? "-Infinity" : "Infinity";
        return;
    }
    char buffer[64];
    const auto digits = float_spelling(value, buffer);
    output += digits;
    if (digits.find_first_of(".e") == digits.npos) {
        output += ".0";
    }
}

void append_json(std::string& output, const json::Value& value) {
    if (const auto text = value.string()) {
        json::append_quoted(output, *text, true);
    } else if (const auto boolean = value.boolean()) {
        output += *boolean ? "true" : "false";
    } else if (const auto integer = value.signed_integer()) {
        output += std::to_string(*integer);
    } else if (const auto integer = value.unsigned_integer()) {
        output += std::to_string(*integer);
    } else if (const auto* integer = value.wide_integer()) {
        output += integer->digits;
    } else if (const auto number = value.number()) {
        append_float(output, *number);
    } else if (const auto* array = value.array()) {
        output += '[';
        for (const auto& element : *array) {
            if (&element != &array->front()) output += ", ";
            append_json(output, element);
        }
        output += ']';
    } else if (const auto* object = value.ordered_object()) {
        output += '{';
        for (const auto& entry : *object) {
            if (&entry != &object->front()) output += ", ";
            json::append_quoted(output, entry.first, true);
            output += ": ";
            append_json(output, entry.second);
        }
        output += '}';
    } else {
        output += "null";
    }
}

bool digit(char character) {
    return character >= '0' && character <= '9';
}

std::string numeric_text(std::string_view raw) {
    const auto text = trim(raw);
    std::string result;
    for (std::size_t index = 0; index < text.size(); ++index) {
        if (text[index] != '_') {
            result += text[index];
        } else if (index == 0 || index + 1 == text.size() ||
                   !digit(text[index - 1]) || !digit(text[index + 1])) {
            return {};
        }
    }
    return result;
}

bool append_integer(std::string& output, std::string_view text) {
    const bool negative = !text.empty() && text.front() == '-';
    if (negative || (!text.empty() && text.front() == '+')) {
        text.remove_prefix(1);
    }
    if (text.empty() || text.size() > 4300 || !std::all_of(text.begin(), text.end(), digit)) {
        return false;
    }
    const auto first = text.find_first_not_of('0');
    if (first == text.npos) {
        output += '0';
    } else {
        if (negative) output += '-';
        output += text.substr(first);
    }
    return true;
}

bool append_number(std::string& output, std::string_view text) {
    if (!text.empty() && text.front() == '+') {
        text.remove_prefix(1);
        if (!text.empty() && text.front() == '-') return false;
    }
    if (text.empty()) return false;
    double number = 0;
    const auto* end = text.data() + text.size();
    const auto parsed = std::from_chars(text.data(), end, number);
    if (parsed.ptr != end) return false;
    if (parsed.ec == std::errc::result_out_of_range) {
        std::istringstream stream{std::string(text)};
        stream.imbue(std::locale::classic());
        stream >> number;
        if (stream.fail()) {
            number = std::copysign(std::numeric_limits<double>::infinity(), number);
        }
    } else if (parsed.ec != std::errc{}) {
        return false;
    }
    if (std::isnan(number) && text.find('(') != text.npos) {
        return false;
    }
    if (std::isfinite(number) && std::trunc(number) == number &&
        text.find_first_of(".eE") == text.npos) {
        char buffer[320];
        const auto result = std::to_chars(buffer, buffer + sizeof(buffer),
                                          number + 0.0, std::chars_format::fixed, 0);
        output.append(buffer, result.ptr);
    } else {
        append_float(output, number);
    }
    return true;
}

bool append_structured(std::string& output, std::string_view raw,
                       const Parameter* union_parameter) {
    if (union_parameter) {
        const auto start = raw.find_first_not_of(" \t\r\n");
        if (start == raw.npos || (raw[start] != '{' && raw[start] != '[')) {
            return false;
        }
    }
    try {
        const auto value = json::parse(raw, json::Syntax::ordered_last_key_wins);
        if (union_parameter &&
            !server::matches_tool_union(value, *union_parameter->schema, *union_parameter->root)) {
            return false;
        }
        append_json(output, value);
        return true;
    } catch (const json::ParseError&) {
        if (!union_parameter) {
            if (const auto value = server::parse_tool_literal(raw)) {
                append_json(output, *value);
                return true;
            }
        }
        return false;
    }
}

std::string convert(std::string_view raw, Parameter* parameter) {
    std::string output;
    if (raw.size() == 4 && lowercase(raw) == "null") return "null";
    if (parameter && !parameter->conversion) {
        parameter->conversion = conversion_for(*parameter->schema);
    }
    const auto conversion = parameter ? *parameter->conversion : Conversion::string;
    switch (conversion) {
    case Conversion::boolean:
        return raw.size() == 4 && lowercase(raw) == "true" ? "true" : "false";
    case Conversion::integer:
        if (append_integer(output, numeric_text(raw))) return output;
        break;
    case Conversion::number:
        if (append_number(output, numeric_text(raw))) return output;
        break;
    case Conversion::structured:
        if (append_structured(output, raw, nullptr)) return output;
        break;
    case Conversion::structured_union:
        if (append_structured(output, raw, parameter)) return output;
        break;
    case Conversion::string:
        break;
    }
    json::append_quoted(output, raw, true);
    return output;
}

std::string format_parameters(std::string_view text, Properties* properties) {
    std::vector<std::pair<std::string_view, std::string_view>> parameters;
    std::map<std::string_view, std::size_t, std::less<>> positions;
    while (true) {
        const auto start = text.find(parameter_open);
        if (start == text.npos) break;
        text.remove_prefix(start + parameter_open.size());
        auto end = text.find('<');
        while (end != text.npos && !starts_with(text.substr(end), parameter_close) &&
               !starts_with(text.substr(end), parameter_open)) {
            end = text.find('<', end + 1);
        }
        const bool closed = end != text.npos && starts_with(text.substr(end), parameter_close);
        const auto body = text.substr(0, end == text.npos && ends_with_newline(text) ?
                                      text.size() - 1 : end);
        const auto separator = body.find('>');
        if (separator != body.npos) {
            const auto name = body.substr(0, separator);
            auto value = body.substr(separator + 1);
            if (!value.empty() && value.front() == '\n') value.remove_prefix(1);
            if (ends_with_newline(value)) value.remove_suffix(1);
            const auto [entry, inserted] = positions.try_emplace(name, parameters.size());
            if (inserted) {
                parameters.emplace_back(name, value);
            } else {
                parameters[entry->second].second = value;
            }
        }
        if (end == text.npos) break;
        text.remove_prefix(end + (closed ? parameter_close.size() : 0));
    }
    std::string result = "{";
    for (const auto& [name, value] : parameters) {
        if (result.size() != 1) result += ", ";
        json::append_quoted(result, name, true);
        result += ": ";
        Parameter* parameter = nullptr;
        if (properties) {
            const auto found = properties->find(name);
            if (found != properties->end()) parameter = &found->second;
        }
        result += convert(value, parameter);
    }
    return result + '}';
}

void append_calls(std::string& output, std::size_t& index, std::string_view text,
                  ToolSchemas& schemas) {
    while (true) {
        const auto start = text.find(function_open);
        if (start == text.npos) return;
        text.remove_prefix(start + function_open.size());
        const auto end = text.find(function_close);
        const auto body = text.substr(0, end);
        const auto separator = body.find('>');
        if (separator != body.npos) {
            const auto name = body.substr(0, separator);
            const auto schema = schemas.find(name);
            if (index != 0) output += ',';
            output += R"({"tool_index":)" + std::to_string(index++) + R"(,"name":)";
            json::append_quoted(output, name, true);
            output += R"(,"parameters":)";
            json::append_quoted(output, format_parameters(body.substr(separator + 1),
                schema == schemas.end() ? nullptr : &schema->second), true);
            output += '}';
        }
        if (end == text.npos) return;
        text.remove_prefix(end + function_close.size());
    }
}

std::string parse_text(std::string_view text, bool has_tools, ToolSchemas& schemas) {
    const auto first = has_tools ? text.find(tool_open) : text.npos;
    std::string response = R"({"normal_text":)";
    json::append_quoted(response, text.substr(0, first), true);
    response += R"(,"calls":[)";
    if (first != text.npos) {
        auto remaining = text.substr(first);
        bool closed = false;
        std::size_t index = 0;
        while (true) {
            const auto start = remaining.find(tool_open);
            if (start == remaining.npos) break;
            remaining.remove_prefix(start + tool_open.size());
            const auto end = remaining.find(tool_close);
            if (end == remaining.npos) break;
            closed = true;
            append_calls(response, index, remaining.substr(0, end), schemas);
            remaining.remove_prefix(end + tool_close.size());
        }
        if (!closed) append_calls(response, index, text, schemas);
    }
    return response + "]}";
}

}

std::optional<std::string> parse_qwen_tool_calls(std::string_view text, const json::Array* tools,
                                                std::string_view parser) {
    auto schemas = tools ? read_tools(*tools) : ToolSchemas{};
    if (parser != qwen_tool_call_parser && parser != "step3p5" && parser != "nanbeige") {
        return std::nullopt;
    }
    return parse_text(text, tools && !tools->empty(), schemas);
}

}
