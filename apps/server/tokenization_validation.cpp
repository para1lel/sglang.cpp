#include "sglang/server/tokenization_validation.h"

#include "sglang/tokenizer/tokenizer.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace sglang::server {
namespace {

using namespace std::string_view_literals;
using json::get;
using json::OrderedObject;
using json::Value;

const Value* present(const OrderedObject& fields, std::string_view key) {
    const auto* value = get(fields, key);
    return value && !value->is_null() ? value : nullptr;
}

[[noreturn]] void invalid(std::string_view field, std::string_view format) {
    throw std::invalid_argument(std::string(field) + " " + std::string(format));
}

bool chat_flag(const OrderedObject& fields, std::string_view key) {
    const auto* value = get(fields, key);
    if (!value) return false;
    if (const auto flag = chat_boolean(*value)) return *flag;
    invalid(key, "must be a boolean");
}

std::string integer_label(const Value& value, double number) {
    if (const auto* integer = value.wide_integer()) return integer->digits;
    if (const auto integer = value.signed_integer()) return std::to_string(*integer);
    if (const auto integer = value.unsigned_integer()) return std::to_string(*integer);
    char buffer[384];
    const auto converted = std::to_chars(buffer, buffer + sizeof(buffer), number,
                                         std::chars_format::fixed, 0);
    return std::string(buffer, converted.ptr);
}

std::optional<double> bounded_integer(const OrderedObject& fields, std::string_view key,
                                      double minimum, double maximum, bool allow_boolean) {
    const auto* value = get(fields, key);
    if (!value || (value->is_null() && key != "n")) return std::nullopt;
    const auto number = chat_number(*value);
    if (!number || std::isnan(*number) || std::trunc(*number) != *number ||
        (!allow_boolean && value->boolean().has_value()) ||
        *number < minimum || *number > maximum) {
        invalid(key, "must be an integer in the supported range");
    }
    return number;
}

void validate_response_format(const OrderedObject& body) {
    const auto* field = present(body, "response_format");
    if (!field) return;
    const auto* format = field->ordered_object();
    const auto* type = format ? present(*format, "type") : nullptr;
    const auto kind = type ? type->string().value_or("") : "";
    if (kind != "text" && kind != "json_object" && kind != "json_schema" &&
        kind != "structural_tag") {
        invalid("response_format", "must specify a supported type");
    }
    const auto* field_schema = present(*format, "json_schema");
    const auto* schema = field_schema ? field_schema->ordered_object() : nullptr;
    if (field_schema && !schema) invalid("json_schema", "must be an object");
    if (schema) {
        const auto* name = present(*schema, "name");
        const auto* description = present(*schema, "description");
        const auto* strict = present(*schema, "strict");
        const auto* fields = present(*schema, "schema");
        if (!name || !name->string() || (description && !description->string()) ||
            (strict && !strict->boolean()) || (fields && !fields->ordered_object())) {
            invalid("json_schema", "contains invalid fields");
        }
        if (fields) return;
    } else if (const auto* legacy = present(*format, "schema");
               legacy && legacy->ordered_object() && !legacy->ordered_object()->empty()) {
        return;
    }
    if (kind == "json_schema") {
        throw std::invalid_argument("schema_ is required for json_schema response format request.");
    }
}

std::string_view schema_type(std::string_view type) {
    auto base = type.substr(0, type.find('('));
    const auto first = base.find_first_not_of(" \t\r\n\f\v");
    if (first == base.npos) return {};
    base = base.substr(first, base.find_last_not_of(" \t\r\n\f\v") - first + 1);
    std::string lower(base);
    for (auto& character : lower) {
        if (character >= 'A' && character <= 'Z') character += 'a' - 'A';
    }
    for (const auto standard : {"null"sv, "string"sv, "boolean"sv, "integer"sv,
                                "number"sv, "array"sv, "object"sv}) {
        if (lower == standard) return standard;
    }
    for (const auto alias : {"str", "text", "varchar", "char", "enum", "uuid", "date",
                             "datetime", "time", "timestamp", "binary", "blob", "bytea",
                             "bytes", "varbinary"}) {
        if (lower == alias) return "string";
    }
    for (const auto alias : {"bool", "bigint", "smallint", "tinyint", "double", "decimal",
                             "real", "numeric", "arr", "tuple", "set", "map"}) {
        if (lower != alias) continue;
        const auto first_letter = lower.front();
        if (lower == "bool") return "boolean";
        if (lower == "map") return "object";
        if (first_letter == 'a' || lower == "tuple" || lower == "set") return "array";
        if (lower == "bigint" || lower == "smallint" || lower == "tinyint") return "integer";
        return "number";
    }
    for (const auto& [prefix, target] : {
         std::pair{"int"sv, "integer"sv}, {"uint"sv, "integer"sv}, {"long"sv, "integer"sv},
         {"short"sv, "integer"sv}, {"unsigned"sv, "integer"sv}, {"num"sv, "number"sv},
         {"float"sv, "number"sv}, {"list"sv, "array"sv}, {"dict"sv, "object"sv}}) {
        if (lower.compare(0, prefix.size(), prefix) == 0 && (lower.size() == prefix.size() ||
            "0123456789[<( \t"sv.find(lower[prefix.size()]) != std::string_view::npos)) {
            return target;
        }
    }
    return {};
}

std::string normalized_type(const Value& field) {
    const auto text = field.string();
    const auto type = text ? schema_type(*text) : std::string_view{};
    if (type.empty()) invalid("type", "must name a JSON Schema type");
    return std::string(type);
}

Value schema_types(const Value& value) {
    const auto* items = value.array();
    if (!items) return normalized_type(value);
    json::Array types;
    for (const auto& item : *items) {
        const auto type = normalized_type(item);
        if (std::none_of(types.begin(), types.end(), [&](const auto& other) {
            return other.string() == type;
        })) types.emplace_back(type);
    }
    if (types.empty()) invalid("type", "must contain at least one JSON Schema type");
    return types;
}

Value schema_tree(const Value& value) {
    if (const auto* items = value.array()) {
        json::Array result;
        for (const auto& item : *items) result.push_back(schema_tree(item));
        return result;
    }
    const auto* object = value.ordered_object();
    if (!object) return value;
    OrderedObject result;
    for (const auto& [key, field] : *object) {
        if (key == "type") {
            result.emplace_back(key, schema_types(field));
        } else if (key == "properties" || key == "patternProperties" || key == "$defs" ||
                   key == "definitions" || key == "dependentSchemas") {
            const auto* children = field.ordered_object();
            if (!children) invalid(key, "must be a schema object");
            OrderedObject normalized;
            for (const auto& [name, child] : *children) {
                normalized.emplace_back(name, schema_tree(child));
            }
            result.emplace_back(key, std::move(normalized));
        } else if (key == "anyOf" || key == "oneOf" || key == "allOf" ||
                   key == "prefixItems" || key == "items" || key == "additionalProperties" ||
                   key == "not" || key == "if" || key == "then" || key == "else" ||
                   key == "contains" || key == "propertyNames" || key == "unevaluatedItems" ||
                   key == "unevaluatedProperties") {
            result.emplace_back(key, schema_tree(field));
        } else {
            if (key == "required" && (!field.array() ||
                !std::all_of(field.array()->begin(), field.array()->end(),
                             [](const auto& item) { return item.string().has_value(); }))) {
                invalid(key, "must be an array of strings");
            }
            result.emplace_back(key, field);
        }
    }
    return result;
}

}

std::optional<bool> chat_boolean(const Value& value) {
    if (const auto flag = value.boolean()) return flag;
    if (const auto number = value.number(); number && (*number == 0 || *number == 1)) {
        return *number == 1;
    }
    if (const auto text = value.string(); text && text->size() <= 5) {
        std::string lower(*text);
        for (auto& character : lower) {
            if (character >= 'A' && character <= 'Z') character += 'a' - 'A';
        }
        for (const auto* word : {"1", "true", "t", "on", "yes", "y"}) {
            if (lower == word) return true;
        }
        for (const auto* word : {"0", "false", "f", "off", "no", "n"}) {
            if (lower == word) return false;
        }
    }
    return std::nullopt;
}

std::optional<double> chat_number(const Value& value) {
    if (const auto number = value.number()) return number;
    if (const auto flag = value.boolean()) return *flag ? 1.0 : 0.0;
    auto text = value.string();
    if (const auto* wide = value.wide_integer()) text = wide->digits;
    if (!text) return std::nullopt;
    const auto first = text->find_first_not_of(" \t\r\n\f\v");
    if (first == text->npos) return std::nullopt;
    const auto last = text->find_last_not_of(" \t\r\n\f\v");
    *text = text->substr(first, last - first + 1);
    if (text->front() == '+') text->remove_prefix(1);
    double result = 0;
    const auto end = text->data() + text->size();
    const auto converted = std::from_chars(text->data(), end, result);
    if (converted.ptr != end) return std::nullopt;
    if (converted.ec == std::errc{}) return result;
    if (value.wide_integer() && converted.ec == std::errc::result_out_of_range) {
        return text->front() == '-' ? -INFINITY : INFINITY;
    }
    return std::nullopt;
}

void validate_chat_generation(const OrderedObject& body) {
    constexpr auto unbounded = std::numeric_limits<double>::infinity();
    const auto maximum = bounded_integer(body, "max_completion_tokens", 0, unbounded, false);
    const auto legacy = bounded_integer(body, "max_tokens", 0, unbounded, false);
    bounded_integer(body, "n", 1, 128, true);
    const auto top = bounded_integer(body, "top_logprobs", 0, unbounded, true);
    if (top && *top > model_vocab_size) {
        throw std::invalid_argument("top_logprobs must be in [0, " +
            std::to_string(model_vocab_size) + "], got " +
            integer_label(*present(body, "top_logprobs"), *top) + ".");
    }
    if (const auto* temperature = present(body, "temperature")) {
        const auto value = chat_number(*temperature);
        if (!value || !std::isfinite(*value) || *value < 0) {
            invalid("temperature", "must be a finite nonnegative number");
        }
    }
    const bool meta = chat_flag(body, "return_meta_info");
    for (const auto flag : {"return_sampling_mask", "return_flat_raw_output_top_logprobs",
                            "return_flat_raw_sampling_mask"}) {
        if (chat_flag(body, flag) && !meta) invalid(flag, "requires return_meta_info=true.");
    }
    const auto budget = maximum ? maximum : legacy;
    if (budget && *budget > checkpoint_context) {
        const auto* value = present(body, maximum ? "max_completion_tokens" : "max_tokens");
        throw std::invalid_argument("max_completion_tokens is too large: " +
            integer_label(*value, *budget) + ".This model supports at most " +
            std::to_string(checkpoint_context) + " completion tokens.");
    }
    validate_response_format(body);
}

Value normalize_tokenization_tool_schema(const Value& value, std::size_t index) {
    try {
        if (!value.ordered_object() && !value.boolean()) {
            invalid("expected", "an object or boolean");
        }
        return schema_tree(value);
    } catch (const std::invalid_argument& error) {
        throw std::invalid_argument("Tool " + std::to_string(index) +
            " function has invalid 'parameters' schema: " + error.what() + ".");
    }
}

}
