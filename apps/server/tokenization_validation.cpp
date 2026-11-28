#include "sglang/server/tokenization_validation.h"

#include "sglang/tokenizer/tokenizer.h"
#include "sglang/utf8.h"

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

bool integer_space(std::string_view scalar) {
    if (scalar.size() == 1) {
        const char character = scalar.front();
        return character == ' ' || (character >= '\t' && character <= '\r');
    }
    if (scalar.size() == 2)
        return scalar == u8"\u0085" || scalar == u8"\u00a0";
    constexpr std::string_view spaces = u8"\u1680\u2000\u2001\u2002\u2003\u2004\u2005"
                                        u8"\u2006\u2007\u2008\u2009\u200a\u2028\u2029"
                                        u8"\u202f\u205f\u3000";
    return scalar.size() == 3 && spaces.find(scalar) != std::string_view::npos;
}

std::string_view trim_integer_text(std::string_view text) {
    while (!text.empty()) {
        const auto size = utf8_lead(static_cast<unsigned char>(text.front())).size;
        if (!integer_space(text.substr(0, size)))
            break;
        text.remove_prefix(size);
    }
    while (!text.empty()) {
        auto start = text.size() - 1;
        while (start > 0 && (static_cast<unsigned char>(text[start]) & 0xc0) == 0x80) {
            --start;
        }
        if (!integer_space(text.substr(start)))
            break;
        text.remove_suffix(text.size() - start);
    }
    return text;
}

bool digit(char character) {
    return character >= '0' && character <= '9';
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
    const auto* field = supplied(body, "response_format");
    if (!field) return;
    const auto* format = field->ordered_object();
    const auto* type = format != nullptr ? supplied(*format, "type") : nullptr;
    const auto kind = type ? type->string().value_or("") : "";
    if (kind != "text" && kind != "json_object" && kind != "json_schema" &&
        kind != "structural_tag") {
        invalid("response_format", "must specify a supported type");
    }
    const auto* field_schema = supplied(*format, "json_schema");
    const auto* schema = field_schema ? field_schema->ordered_object() : nullptr;
    if (field_schema && !schema) invalid("json_schema", "must be an object");
    if (schema) {
        const auto* name = supplied(*schema, "name");
        const auto* description = supplied(*schema, "description");
        const auto* strict = supplied(*schema, "strict");
        const auto* fields = supplied(*schema, "schema");
        if (!name || !name->string() || (description && !description->string()) ||
            (strict && !strict->boolean()) || (fields && !fields->ordered_object())) {
            invalid("json_schema", "contains invalid fields");
        }
        if (fields) return;
    } else if (const auto* legacy = supplied(*format, "schema");
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

void request_validation::invalid(std::string_view field, std::string_view message) {
    throw std::invalid_argument(std::string(field) + ": " + std::string(message));
}

bool request_validation::optional_bool(const Value* value, std::string_view field, bool fallback) {
    if (value == nullptr)
        return fallback;
    if (const auto flag = chat_boolean(*value))
        return *flag;
    invalid(field, "Input should be a valid boolean");
}

const Value* supplied(const OrderedObject& fields, std::string_view key) {
    const auto* value = get(fields, key);
    return value != nullptr && !value->is_null() ? value : nullptr;
}

bool python_whitespace(char32_t point) {
    return (point >= 9 && point <= 13) || (point >= 0x1c && point <= 0x20) || point == 0x85 ||
           point == 0xa0 || point == 0x1680 || (point >= 0x2000 && point <= 0x200a) ||
           point == 0x2028 || point == 0x2029 || point == 0x202f || point == 0x205f ||
           point == 0x3000;
}

std::optional<detail::IntegerText> detail::scan_integer_text(std::string_view text,
                                                             IntegerSeparators separators) {
    text = trim_integer_text(text);
    const bool negative = !text.empty() && text.front() == '-';
    if (!text.empty() && (negative || text.front() == '+'))
        text.remove_prefix(1);
    if (text.empty() || !digit(text.front()))
        return std::nullopt;

    std::size_t start = std::string_view::npos;
    std::size_t digit_count = 0;
    std::size_t end = 0;
    bool separator = false;
    for (; end < text.size() && text[end] != '.'; ++end) {
        const char character = text[end];
        if (digit(character)) {
            if (start == std::string_view::npos && character != '0')
                start = end;
            if (start != std::string_view::npos)
                ++digit_count;
        } else if (character != '_' || (separator && separators == IntegerSeparators::single)) {
            return std::nullopt;
        }
        separator = character == '_';
    }
    if (separator ||
        (end < text.size() && (end + 1 == text.size() ||
                               text.find_first_not_of('0', end + 1) != std::string_view::npos))) {
        return std::nullopt;
    }
    if (digit_count == 0)
        return IntegerText{{}, 0, false};
    return IntegerText{text.substr(start, end - start), digit_count, negative};
}

bool chat_truthy(const Value& value) {
    if (const auto boolean = value.boolean()) return *boolean;
    if (const auto number = value.number()) return *number != 0;
    if (const auto text = value.string()) return !text->empty();
    if (const auto* array = value.array()) return !array->empty();
    if (const auto* object = value.object()) return !object->empty();
    if (const auto* object = value.ordered_object()) return !object->empty();
    return !value.is_null();
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
                                    integer_label(*supplied(body, "top_logprobs"), *top) + ".");
    }
    if (const auto* temperature = supplied(body, "temperature")) {
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
        const auto* value = supplied(body, maximum ? "max_completion_tokens" : "max_tokens");
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
