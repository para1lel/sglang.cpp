#include "sglang/decoder_constraint.h"

#include "sglang/tokenizer/json_schema_pattern.h"

#include <algorithm>
#include <iterator>
#include <set>
#include <vector>

namespace sglang {
namespace {

constexpr std::size_t max_pattern_bytes = 8192;
constexpr std::string_view integer = R"(-?(?:0|[1-9][0-9]{0,17}))";
constexpr std::string_view fraction = R"((?:\.[0-9]{1,12})?)";
constexpr std::string_view string_value =
    R"json("(?:[^"\\\x00-\x1f]|\\(?:["\\/bfnrt]|u(?:[0-9a-ce-fA-CE-F][0-9a-fA-F]{3}|)json"
    R"json([dD][0-7][0-9a-fA-F]{2}|[dD][89a-bA-B][0-9a-fA-F]{2})json"
    R"json(\\u[dD][c-fC-F][0-9a-fA-F]{2})))*")json";

[[noreturn]] void invalid(std::string_view detail) {
    throw std::invalid_argument("Invalid JSON Schema: " + std::string(detail));
}

void append_pattern(std::string& output, std::string_view input) {
    if (input.size() > max_pattern_bytes - output.size()) {
        invalid("expanded pattern exceeds 8192 bytes");
    }
    output += input;
}

}

std::string tokenizer::regex_literal(std::string_view value) {
    std::string pattern;
    for (const char byte : value) {
        if (std::string_view(R"(\.^$|?*+()[]{})").find(byte) != std::string_view::npos) {
            pattern += '\\';
        }
        pattern += byte;
    }
    return pattern;
}

namespace {

using tokenizer::regex_literal;

using tokenizer::xml_string_pattern;

std::string json_literal(std::string_view value) {
    std::string json;
    json::append_quoted(json, value, true);
    return regex_literal(json);
}

std::string_view type_of(const json::Value& value) {
    if (value.string()) return "string";
    if (value.boolean()) return "boolean";
    if (value.is_null()) return "null";
    if (value.signed_integer() || value.unsigned_integer()) return "integer";
    return {};
}

bool safe_xml_text(std::string_view text) {
    constexpr std::string_view markers[] = {"</parameter>", "<parameter=", "</function>",
                                            "</tool_call>", "<function="};
    return text.find_first_of("\r\n") == text.npos &&
        std::none_of(std::begin(markers), std::end(markers), [&](auto marker) {
            return text.find(marker) != text.npos;
        });
}

std::string enum_value(const json::Value& value, std::string_view type, bool xml) {
    const auto actual = type_of(value);
    if (actual.empty() || (!type.empty() && type != actual &&
                           !(type == "number" && actual == "integer"))) {
        invalid("enum supports strings, integral numbers, booleans, and null of the given type");
    }
    if (const auto text = value.string()) {
        if (text->size() > 128) {
            invalid("enum strings must contain at most 128 UTF-8 bytes");
        }
        if (xml && !safe_xml_text(*text)) {
            invalid("XML string enum contains a structural delimiter or newline");
        }
        return xml ? regex_literal(*text) : json_literal(*text);
    }
    if (const auto number = value.signed_integer()) return std::to_string(*number);
    if (const auto number = value.unsigned_integer()) return std::to_string(*number);
    if (const auto flag = value.boolean()) return *flag ? "true" : "false";
    return "null";
}

std::string enumeration(const json::Value& value, std::string_view type, bool xml) {
    const auto* choices = value.array();
    if (!choices || choices->empty() || choices->size() > 16) {
        invalid("enum must have 1 to 16 scalar choices");
    }
    std::set<std::string> seen;
    std::string pattern = "(?:";
    for (const auto& choice : *choices) {
        const auto entry = enum_value(choice, type, xml);
        if (!seen.insert(entry).second) {
            invalid("enum has duplicate choices");
        }
        if (seen.size() > 1) append_pattern(pattern, "|");
        append_pattern(pattern, entry);
    }
    append_pattern(pattern, ")");
    return pattern;
}

const json::OrderedObject& fields(const json::Value& value) {
    if (const auto* object = value.ordered_object()) return *object;
    invalid("expected an ordered schema object");
}

void check_keywords(const json::OrderedObject& object, std::string_view type, bool xml = false) {
    std::set<std::string_view> names;
    for (const auto& [name, value] : object) {
        if (!names.insert(name).second) invalid("duplicate keyword");
        if (name == "type" || name == "enum") continue;
        if (name == "title" || name == "description") {
            if (!value.string() || value.string()->size() > 256) {
                invalid("title and description must be strings of at most 256 UTF-8 bytes");
            }
        } else if (xml && type == "integer" && (name == "minimum" || name == "maximum")) {
            continue;
        } else if (type == "object" &&
                   (name == "properties" || name == "required" || name == "additionalProperties")) {
            continue;
        } else if (type == "array" &&
                   (name == "items" || name == "minItems" || name == "maxItems")) {
            continue;
        } else {
            invalid("unsupported keyword " + name.substr(0, 64));
        }
    }
}

std::set<std::string_view> required_names(const json::OrderedObject& schema) {
    std::set<std::string_view> names;
    if (const auto* value = json::get(schema, "required")) {
        const auto* array = value->array();
        if (!array || array->size() > 8) invalid("required needs an array of up to eight names");
        for (const auto& entry : *array) {
            if (!entry.string() || !names.insert(*entry.string()).second) {
                invalid("required names must be unique strings");
            }
        }
    }
    return names;
}

std::string alternatives(const std::vector<std::string>& values) {
    std::string pattern = "(?:";
    const char* separator = "";
    for (const auto& value : values) {
        append_pattern(pattern, separator);
        append_pattern(pattern, value);
        separator = "|";
    }
    append_pattern(pattern, ")");
    return pattern;
}

std::string digits(char first, char last, std::size_t tail = 0) {
    std::string pattern;
    if (first == last) {
        pattern += first;
    } else {
        pattern = std::string("[") + first + '-' + last + ']';
    }
    if (tail != 0) {
        pattern += "[0-9]{" + std::to_string(tail) + '}';
    }
    return pattern;
}

std::string decimal_range(std::string_view first, std::string_view last) {
    const auto common = std::mismatch(first.begin(), first.end(), last.begin()).first;
    const std::size_t width = common - first.begin();
    std::string pattern(first.substr(0, width));
    first.remove_prefix(width);
    last.remove_prefix(width);
    if (first.empty()) return pattern;
    if (first.find_first_not_of('0') == first.npos && last.find_first_not_of('9') == last.npos) {
        return pattern + "[0-9]{" + std::to_string(first.size()) + '}';
    }
    if (first.size() == 1) return pattern + digits(first.front(), last.front());
    const auto tail = first.size() - 1;
    std::vector<std::string> choices;
    choices.push_back(first.front() + decimal_range(first.substr(1), std::string(tail, '9')));
    if (first.front() + 1 < last.front()) {
        choices.push_back(digits(first.front() + 1, last.front() - 1, tail));
    }
    choices.push_back(last.front() + decimal_range(std::string(tail, '0'), last.substr(1)));
    return pattern + alternatives(choices);
}

std::string unsigned_range(std::uint64_t first, std::uint64_t last) {
    auto low = std::to_string(first);
    const auto high = std::to_string(last);
    std::vector<std::string> choices;
    for (auto length = low.size(); length < high.size(); ++length) {
        choices.push_back(decimal_range(low, std::string(length, '9')));
        low = '1' + std::string(length, '0');
    }
    choices.push_back(decimal_range(low, high));
    return alternatives(choices);
}

constexpr std::int64_t max_integer = 999999999999999999;

std::int64_t integer_bound(const json::OrderedObject& schema, std::string_view key,
                           std::int64_t fallback) {
    const auto* value = json::get(schema, key);
    if (!value) return fallback;
    auto number = value->signed_integer();
    if (const auto positive = value->unsigned_integer(); !number && positive &&
        *positive <= static_cast<std::uint64_t>(max_integer)) {
        number = static_cast<std::int64_t>(*positive);
    }
    if (!number || *number < -max_integer || *number > max_integer) {
        invalid("XML integer bounds must be signed integers of at most 18 digits");
    }
    return *number;
}

std::string bounded_integer(const json::OrderedObject& schema) {
    const auto minimum = integer_bound(schema, "minimum", -max_integer);
    const auto maximum = integer_bound(schema, "maximum", max_integer);
    if (minimum > maximum) invalid("minimum exceeds maximum");
    if (maximum < 0) return '-' + unsigned_range(-maximum, -minimum);
    if (minimum >= 0) return unsigned_range(minimum, maximum);
    return alternatives({'-' + unsigned_range(1, -minimum), unsigned_range(0, maximum)});
}

struct Member {
    std::string pattern;
    bool required;
};

enum class MemberSyntax { json, xml };

enum class MemberComma { none, before, after };

void optional_member(std::string& pattern, const Member& member, MemberComma comma) {
    append_pattern(pattern, "(?:");
    if (comma == MemberComma::before)
        append_pattern(pattern, ",");
    append_pattern(pattern, member.pattern);
    if (comma == MemberComma::after)
        append_pattern(pattern, ",");
    append_pattern(pattern, ")?");
}

std::string member_sequence(const std::vector<Member>& members, MemberSyntax syntax) {
    const bool xml = syntax == MemberSyntax::xml;
    const std::string_view delimiter = xml ? "" : ",";
    std::string pattern = xml ? "" : R"(\{)";
    if (std::none_of(members.begin(), members.end(),
                     [](const Member& member) { return member.required; })) {
        append_pattern(pattern, xml ? "(?:\n" : "(?:");
        const auto comma = xml ? MemberComma::none : MemberComma::before;
        for (auto first = members.begin(); first != members.end(); ++first) {
            append_pattern(pattern, "|");
            append_pattern(pattern, first->pattern);
            for (auto next = std::next(first); next != members.end(); ++next) {
                optional_member(pattern, *next, comma);
            }
        }
        append_pattern(pattern, ")");
    } else {
        const auto after_required = xml ? MemberComma::none : MemberComma::before;
        auto optional_comma = xml ? MemberComma::none : MemberComma::after;
        std::string_view separator;
        for (const auto& member : members) {
            if (member.required) {
                append_pattern(pattern, separator);
                append_pattern(pattern, member.pattern);
                separator = delimiter;
                optional_comma = after_required;
            } else {
                optional_member(pattern, member, optional_comma);
            }
        }
    }
    append_pattern(pattern, xml ? "" : R"(\})");
    return pattern;
}

int item_count(const json::OrderedObject& schema, std::string_view key, int fallback) {
    if (const auto* value = json::get(schema, key)) {
        const auto count = value->unsigned_integer();
        if (!count || *count > 8) invalid("array item bounds must be integers from 0 to 8");
        return static_cast<int>(*count);
    }
    return fallback;
}

class Compiler {
public:
    std::string node(const json::Value& value, int depth, bool xml = false);
    std::string xml_parameters(const json::Value& value);

private:
    std::string object(const json::OrderedObject& schema, int depth);
    std::string array(const json::OrderedObject& schema, int depth);
    int node_count_ = 0;
};

std::string Compiler::object(const json::OrderedObject& schema, int depth) {
    if (const auto* additional = json::get(schema, "additionalProperties")) {
        if (additional->boolean() != false) invalid("additionalProperties must be false");
    }
    auto required = required_names(schema);
    std::vector<Member> members;
    if (const auto* properties = json::get(schema, "properties")) {
        const auto& values = fields(*properties);
        if (values.size() > 8) invalid("an object supports at most eight properties");
        std::set<std::string_view> seen;
        for (const auto& [name, value] : values) {
            if (name.size() > 64 || !seen.insert(name).second) {
                invalid("property names must be unique and at most 64 UTF-8 bytes");
            }
            auto pattern = json_literal(name) + ':';
            append_pattern(pattern, node(value, depth + 1));
            members.push_back({std::move(pattern), required.erase(name) != 0});
        }
    }
    if (!required.empty()) invalid("required contains a name outside properties");
    return member_sequence(members, MemberSyntax::json);
}

std::string Compiler::array(const json::OrderedObject& schema, int depth) {
    const auto* items = json::get(schema, "items");
    if (!items) invalid("arrays require one items schema");
    const int minimum = item_count(schema, "minItems", 0);
    const int maximum = item_count(schema, "maxItems", std::max(4, minimum));
    if (minimum > maximum) invalid("minItems exceeds maxItems");
    const auto item = node(*items, depth + 1);
    if (maximum == 0) return R"(\[\])";
    std::string pattern = minimum == 0 ? R"(\[(?:)" : R"(\[)";
    append_pattern(pattern, item);
    if (maximum > 1) {
        append_pattern(pattern, "(?:,");
        append_pattern(pattern, item);
        append_pattern(pattern, "){" + std::to_string(std::max(0, minimum - 1)) + ',' +
                                    std::to_string(maximum - 1) + '}');
    }
    if (minimum == 0) append_pattern(pattern, ")?");
    append_pattern(pattern, R"(\])");
    return pattern;
}

std::string Compiler::node(const json::Value& value, int depth, bool xml) {
    if (depth > 5 || ++node_count_ > 64) {
        invalid("schema exceeds five nested levels or 64 nodes");
    }
    const auto& schema = fields(value);
    std::string_view type;
    if (const auto* field = json::get(schema, "type")) {
        if (!field->string() || field->string()->empty()) invalid("type must be a string");
        type = *field->string();
    }
    check_keywords(schema, type, xml);
    const bool bounded = xml && (json::get(schema, "minimum") || json::get(schema, "maximum"));
    if (const auto* choices = json::get(schema, "enum")) {
        if (bounded) invalid("integer enums and bounds cannot be combined");
        return enumeration(*choices, type, xml);
    }
    if (bounded) return bounded_integer(schema);
    if (type == "object") return object(schema, depth);
    if (type == "array") return array(schema, depth);
    if (type == "boolean") return "(?:true|false)";
    if (type == "null") return "null";
    if (type == "string") {
        return std::string(xml ? xml_string_pattern : string_value);
    }
    if (type == "integer") return std::string(integer);
    if (type == "number") return std::string(integer) + std::string(fraction);
    invalid("expected a supported type or a finite scalar enum");
}

std::string Compiler::xml_parameters(const json::Value& value) {
    const auto& schema = fields(value);
    check_keywords(schema, "object");
    const auto* type = json::get(schema, "type");
    const auto* additional = json::get(schema, "additionalProperties");
    if ((type && type->string() != "object") || json::get(schema, "enum") ||
        (additional && additional->boolean() != false)) {
        invalid("tool parameters must be a closed ordered object");
    }
    auto required = required_names(schema);
    std::vector<Member> members;
    std::set<std::string_view> seen;
    if (const auto* properties = json::get(schema, "properties")) {
        const auto& values = fields(*properties);
        if (values.size() > 8) invalid("an object supports at most eight properties");
        for (const auto& [name, property] : values) {
            if (name.empty() || name.size() > 64 || name.find_first_of("<>\r\n") != name.npos ||
                !seen.insert(name).second) {
                invalid("XML parameter names must be unique valid labels of up to 64 bytes");
            }
            auto pattern = regex_literal("<parameter=" + name + ">\n");
            append_pattern(pattern, node(property, 2, true));
            append_pattern(pattern, "\n</parameter>\n");
            members.push_back({std::move(pattern), required.erase(name) != 0});
        }
    }
    if (!required.empty()) invalid("required contains a name outside properties");
    return member_sequence(members, MemberSyntax::xml);
}

}

std::string tokenizer::xml_parameters_pattern(const json::Value* schema) {
    return !schema || schema->is_null() ? "\n" : Compiler().xml_parameters(*schema);
}

std::shared_ptr<const TokenConstraint> TokenConstraint::compile_json_schema(
    const Tokenizer& tokenizer, const json::Value& schema) {
    const auto pattern = Compiler().node(schema, 1);
    try {
        return compile_regular(tokenizer, pattern, Kind::json_schema);
    } catch (const std::invalid_argument& error) {
        throw std::invalid_argument("JSON Schema automaton: " + std::string(error.what()));
    }
}

}
