#include "sglang/server/tool_union.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>

namespace sglang::server {
namespace {

enum class Match { yes, no, unknown };

std::string integer_spelling(const json::Value& value) {
    if (const auto integer = value.signed_integer()) return std::to_string(*integer);
    if (const auto integer = value.unsigned_integer()) return std::to_string(*integer);
    if (const auto* integer = value.wide_integer()) return integer->digits;
    if (const auto number = value.number(); number && std::isfinite(*number) &&
        std::trunc(*number) == *number) {
        char buffer[320];
        const auto result = std::to_chars(buffer, buffer + sizeof(buffer),
                                          *number + 0.0, std::chars_format::fixed, 0);
        return std::string(buffer, result.ptr);
    }
    return {};
}

Match equal(const json::Value& value, const json::Value& expected) {
    bool same;
    if (expected.ordered_object()) {
        return value.ordered_object() ? Match::unknown : Match::no;
    }
    if (expected.array()) {
        return value.array() ? Match::unknown : Match::no;
    }
    if (expected.is_null()) {
        same = value.is_null();
    } else if (const auto text = expected.string()) {
        same = value.string() == text;
    } else if (const auto flag = expected.boolean()) {
        same = value.boolean() == flag;
    } else {
        const auto integer = integer_spelling(expected);
        same = !integer.empty() ? integer_spelling(value) == integer :
            !value.wide_integer() && value.number() == expected.number();
    }
    return same ? Match::yes : Match::no;
}

Match type_match(const json::Value& value, std::string_view type) {
    bool matches;
    if (type == "object") matches = value.ordered_object() != nullptr;
    else if (type == "array") matches = value.array() != nullptr;
    else if (type == "string") matches = value.string().has_value();
    else if (type == "boolean") matches = value.boolean().has_value();
    else if (type == "null") matches = value.is_null();
    else if (type == "number") matches = value.wide_integer() || value.number();
    else if (type == "integer") {
        const auto number = value.number();
        matches = value.wide_integer() || (number && std::trunc(*number) == *number);
    } else {
        return Match::unknown;
    }
    return matches ? Match::yes : Match::no;
}

template <typename Check>
Match one_or_more(const json::Array& entries, bool exactly_one, const Check& check) {
    unsigned matches = 0;
    bool unknown = false;
    for (const auto& entry : entries) {
        switch (check(entry)) {
        case Match::yes:
            if (!exactly_one) return Match::yes;
            if (++matches > 1) return Match::no;
            break;
        case Match::unknown:
            unknown = true;
            break;
        case Match::no:
            break;
        }
    }
    if (unknown) return Match::unknown;
    return matches == 1 ? Match::yes : Match::no;
}

Match type_match(const json::Value& value, const json::Value& schema) {
    if (const auto type = schema.string()) return type_match(value, *type);
    if (const auto* types = schema.array()) {
        return one_or_more(*types, false, [&](const json::Value& candidate) {
            const auto type = candidate.string();
            return type ? type_match(value, *type) : Match::unknown;
        });
    }
    return Match::unknown;
}

Match combined(Match left, Match right) {
    if (left == Match::no || right == Match::no) return Match::no;
    return left == Match::yes && right == Match::yes ? Match::yes : Match::unknown;
}

bool supported_dialect(const json::OrderedObject& object) {
    const auto* dialect = json::get(object, "$schema");
    if (!dialect) return true;
    const auto uri = dialect->string();
    return uri == "https://json-schema.org/draft/2020-12/schema" ||
           uri == "https://json-schema.org/draft/2020-12/schema#";
}

std::optional<std::string> pointer_key(std::string_view token) {
    std::string key;
    for (std::size_t index = 0; index < token.size(); ++index) {
        if (token[index] != '~') {
            key += token[index];
        } else {
            if (++index == token.size() || (token[index] != '0' && token[index] != '1')) {
                return std::nullopt;
            }
            key += token[index] == '0' ? '~' : '/';
        }
    }
    return key;
}

class UnionMatch {
public:
    explicit UnionMatch(const json::Value& root) : root_(root) {}
    Match schema(const json::Value& value, const json::Value& rule);

private:
    bool consume(std::size_t count);
    const json::Value* reference(const json::Value& rule);
    Match evaluate(const json::Value& value, const json::Value& rule);
    Match keyword(const json::Value& value, std::string_view key, const json::Value& rule,
                  const json::OrderedObject& enclosing);
    Match properties(const json::OrderedObject& value, const json::OrderedObject& rule);
    Match additional(const json::OrderedObject& value, const json::Value& rule,
                     const json::OrderedObject& enclosing);
    Match required(const json::OrderedObject& value, const json::Array& names);
    const json::Value& root_;
    unsigned depth_ = 0;
    unsigned remaining_ = 16'384;
};

bool UnionMatch::consume(std::size_t count) {
    if (count > remaining_) {
        remaining_ = 0;
        return false;
    }
    remaining_ -= static_cast<unsigned>(count);
    return true;
}

Match UnionMatch::schema(const json::Value& value, const json::Value& rule) {
    if (depth_ >= 32 || !consume(1)) return Match::unknown;
    ++depth_;
    const auto result = evaluate(value, rule);
    --depth_;
    return result;
}

const json::Value* UnionMatch::reference(const json::Value& rule) {
    const auto uri = rule.string();
    if (!uri || !consume(uri->size()) || uri->find('%') != uri->npos) return nullptr;
    if (*uri == "#") return &root_;
    if (uri->substr(0, 2) != "#/") return nullptr;
    auto pointer = uri->substr(1);
    const auto* target = &root_;
    while (!pointer.empty()) {
        pointer.remove_prefix(1);
        const auto separator = pointer.find('/');
        const auto* object = target->ordered_object();
        if (!object || !consume(1 + object->size()) ||
            (target != &root_ && json::get(*object, "$id"))) {
            return nullptr;
        }
        const auto key = pointer_key(pointer.substr(0, separator));
        target = key ? json::get(*object, *key) : nullptr;
        if (!target || separator == pointer.npos) return target;
        pointer.remove_prefix(separator);
    }
    return target;
}

Match UnionMatch::evaluate(const json::Value& value, const json::Value& rule) {
    if (const auto boolean = rule.boolean()) return *boolean ? Match::yes : Match::no;
    const auto* object = rule.ordered_object();
    if (!object || !consume(object->size()) || !supported_dialect(*object) ||
        (&rule != &root_ && json::get(*object, "$id"))) {
        return Match::unknown;
    }
    const auto* type = json::get(*object, "type");
    if (type && type->array() && !consume(type->array()->size())) return Match::unknown;
    auto result = type ? type_match(value, *type) : Match::yes;
    for (const auto& [key, entry] : *object) {
        if (result == Match::no) return Match::no;
        if (key != "type") result = combined(result, keyword(value, key, entry, *object));
    }
    return result;
}

Match UnionMatch::properties(const json::OrderedObject& value, const json::OrderedObject& rule) {
    auto result = Match::yes;
    for (const auto& [name, entry] : value) {
        if (!consume(1 + rule.size())) return Match::unknown;
        if (const auto* constraint = json::get(rule, name)) {
            result = combined(result, schema(entry, *constraint));
            if (result == Match::no) return result;
        }
    }
    return result;
}

Match UnionMatch::additional(const json::OrderedObject& value, const json::Value& rule,
                             const json::OrderedObject& enclosing) {
    if (json::get(enclosing, "patternProperties")) return Match::unknown;
    const auto* declared = json::get(enclosing, "properties");
    const auto* properties = declared ? declared->ordered_object() : nullptr;
    auto result = Match::yes;
    for (const auto& [name, entry] : value) {
        if (!consume(1 + (properties ? properties->size() : 0))) return Match::unknown;
        if (!properties || !json::get(*properties, name)) {
            result = combined(result, schema(entry, rule));
            if (result == Match::no) return result;
        }
    }
    return result;
}

Match UnionMatch::required(const json::OrderedObject& value, const json::Array& names) {
    for (const auto& entry : names) {
        if (!consume(1 + value.size())) return Match::unknown;
        const auto name = entry.string();
        if (!name) return Match::unknown;
        if (!json::get(value, *name)) return Match::no;
    }
    return Match::yes;
}

Match UnionMatch::keyword(const json::Value& value, std::string_view key, const json::Value& rule,
                          const json::OrderedObject& enclosing) {
    if (key == "$ref") {
        const auto* target = reference(rule);
        return target ? schema(value, *target) : Match::unknown;
    }
    if (key == "minItems") {
        const auto* items = value.array();
        if (!items) return Match::yes;
        const auto minimum = rule.number();
        if (!minimum || *minimum < 0 || std::trunc(*minimum) != *minimum) {
            return Match::unknown;
        }
        return items->size() >= *minimum ? Match::yes : Match::no;
    }
    if (key == "const") return equal(value, rule);
    if (key == "enum" || key == "anyOf" || key == "oneOf" || key == "allOf") {
        const auto* entries = rule.array();
        if (!entries) return Match::unknown;
        if (key == "allOf") {
            auto result = Match::yes;
            for (const auto& entry : *entries) {
                result = combined(result, schema(value, entry));
                if (result == Match::no) return result;
            }
            return result;
        }
        return one_or_more(*entries, key == "oneOf", [&](const json::Value& entry) {
            if (key != "enum") return schema(value, entry);
            return consume(1) ? equal(value, entry) : Match::unknown;
        });
    }
    if (key == "properties" || key == "required" || key == "additionalProperties") {
        const auto* object = value.ordered_object();
        if (!object) return Match::yes;
        if (key == "additionalProperties") return additional(*object, rule, enclosing);
        if (key == "properties") {
            const auto* entries = rule.ordered_object();
            return entries ? properties(*object, *entries) : Match::unknown;
        }
        const auto* names = rule.array();
        return names ? required(*object, *names) : Match::unknown;
    }
    if (key == "items") {
        if (json::get(enclosing, "prefixItems")) return Match::unknown;
        auto result = Match::yes;
        if (const auto* items = value.array()) {
            for (const auto& item : *items) {
                result = combined(result, schema(item, rule));
                if (result == Match::no) return result;
            }
        }
        return result;
    }
    constexpr std::array<std::string_view, 14> annotations{
        "$schema", "$id", "$defs", "$anchor", "$comment", "title", "description",
        "default", "examples", "deprecated", "readOnly", "writeOnly", "discriminator", "format"
    };
    return std::find(annotations.begin(), annotations.end(), key) != annotations.end()
        ? Match::yes : Match::unknown;
}

}

bool matches_tool_union(const json::Value& value, const json::Value& schema,
                        const json::Value& root) {
    const auto* object = root.ordered_object();
    return object && supported_dialect(*object) && UnionMatch(root).schema(value, schema) ==
        Match::yes;
}

}
