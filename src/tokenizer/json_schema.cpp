#include "sglang/decoder_constraint.h"

#include "sglang/json.h"

#include <algorithm>
#include <iterator>
#include <set>

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

std::string literal(std::string_view value) {
    std::string pattern;
    for (const char byte : value) {
        if (std::string_view(R"(\.^$|?*+()[]{})").find(byte) != std::string_view::npos) {
            pattern += '\\';
        }
        pattern += byte;
    }
    return pattern;
}

std::string json_literal(std::string_view value) {
    std::string json;
    json::append_quoted(json, value, true);
    return literal(json);
}

std::string_view type_of(const json::Value& value) {
    if (value.string()) return "string";
    if (value.boolean()) return "boolean";
    if (value.is_null()) return "null";
    if (value.signed_integer() || value.unsigned_integer()) return "integer";
    return {};
}

std::string enum_value(const json::Value& value, std::string_view type) {
    const auto actual = type_of(value);
    if (actual.empty() || (!type.empty() && type != actual &&
                           !(type == "number" && actual == "integer"))) {
        invalid("enum supports strings, integral numbers, booleans, and null of the given type");
    }
    if (const auto text = value.string()) {
        if (text->size() > 128) {
            invalid("enum strings must contain at most 128 UTF-8 bytes");
        }
        return json_literal(*text);
    }
    if (const auto number = value.signed_integer()) return std::to_string(*number);
    if (const auto number = value.unsigned_integer()) return std::to_string(*number);
    if (const auto flag = value.boolean()) return *flag ? "true" : "false";
    return "null";
}

std::string enumeration(const json::Value& value, std::string_view type) {
    const auto* choices = value.array();
    if (!choices || choices->empty() || choices->size() > 16) {
        invalid("enum must have 1 to 16 scalar choices");
    }
    std::set<std::string> seen;
    std::string pattern = "(?:";
    for (const auto& choice : *choices) {
        const auto entry = enum_value(choice, type);
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

void check_keywords(const json::OrderedObject& object, std::string_view type) {
    std::set<std::string_view> names;
    for (const auto& [name, value] : object) {
        if (!names.insert(name).second) invalid("duplicate keyword");
        if (name == "type" || name == "enum") continue;
        if (name == "title" || name == "description") {
            if (!value.string() || value.string()->size() > 256) {
                invalid("title and description must be strings of at most 256 UTF-8 bytes");
            }
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

struct Member {
    std::string pattern;
    bool required;
};

void optional_member(std::string& pattern, const Member& member, bool before_first_required) {
    append_pattern(pattern, "(?:");
    if (!before_first_required) append_pattern(pattern, ",");
    append_pattern(pattern, member.pattern);
    if (before_first_required) append_pattern(pattern, ",");
    append_pattern(pattern, ")?");
}

void members_without_required(std::string& pattern, const std::vector<Member>& members) {
    append_pattern(pattern, "(?:");
    for (auto first = members.begin(); first != members.end(); ++first) {
        append_pattern(pattern, "|");
        append_pattern(pattern, first->pattern);
        for (auto next = std::next(first); next != members.end(); ++next) {
            optional_member(pattern, *next, false);
        }
    }
    append_pattern(pattern, ")");
}

std::string member_sequence(const std::vector<Member>& members) {
    std::string pattern = R"(\{)";
    if (std::none_of(members.begin(), members.end(), [](Member const& member) {
            return member.required;
        })) {
        members_without_required(pattern, members);
    } else {
        bool started = false;
        for (const auto& member : members) {
            if (member.required) {
                if (started) append_pattern(pattern, ",");
                append_pattern(pattern, member.pattern);
                started = true;
            } else {
                optional_member(pattern, member, !started);
            }
        }
    }
    append_pattern(pattern, R"(\})");
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
    std::string node(const json::Value& value, int depth);

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
    return member_sequence(members);
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

std::string Compiler::node(const json::Value& value, int depth) {
    if (depth > 5 || ++node_count_ > 64) {
        invalid("schema exceeds five nested levels or 64 nodes");
    }
    const auto& schema = fields(value);
    std::string_view type;
    if (const auto* field = json::get(schema, "type")) {
        if (!field->string() || field->string()->empty()) invalid("type must be a string");
        type = *field->string();
    }
    check_keywords(schema, type);
    if (const auto* choices = json::get(schema, "enum")) return enumeration(*choices, type);
    if (type == "object") return object(schema, depth);
    if (type == "array") return array(schema, depth);
    if (type == "boolean") return "(?:true|false)";
    if (type == "null") return "null";
    if (type == "string") return std::string(string_value);
    if (type == "integer") return std::string(integer);
    if (type == "number") return std::string(integer) + std::string(fraction);
    invalid("expected a supported type or a finite scalar enum");
}

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
