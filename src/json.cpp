#include "sglang/json.h"

#include "sglang/utf8.h"

#include <charconv>
#include <cmath>
#include <limits>
#include <locale>
#include <sstream>

namespace sglang::json {
namespace {

bool digit(char character) {
    return character >= '0' && character <= '9';
}

class Parser {
public:
    Parser(std::string_view input, Syntax syntax) : input_(input), syntax_(syntax) {}

    Value document();

private:
    char peek() const { return input_.empty() ? '\0' : input_.front(); }
    bool accept(char character);
    void whitespace();
    void literal(std::string_view word);
    [[noreturn]] void fail(const char* message) const { throw ParseError(message); }
    Value value(std::size_t depth, bool ordered);
    Array array(std::size_t depth, bool ordered);
    Value object(std::size_t depth, bool ordered);
    Value number();
    bool digits();
    void strict_number();
    std::string string();
    char32_t hex_quad();
    void unicode(std::string& output);

    std::string_view input_;
    Syntax syntax_;
};

bool Parser::accept(char character) {
    if (input_.empty() || peek() != character) {
        return false;
    }
    input_.remove_prefix(1);
    return true;
}

void Parser::whitespace() {
    const auto next = input_.find_first_not_of(" \t\r\n");
    input_.remove_prefix(next == std::string_view::npos ? input_.size() : next);
}

void Parser::literal(std::string_view word) {
    if (input_.compare(0, word.size(), word) != 0) {
        fail("Invalid JSON literal");
    }
    input_.remove_prefix(word.size());
}

Value Parser::document() {
    auto result = value(0, syntax_ == Syntax::ordered || syntax_ == Syntax::ordered_last_key_wins ||
                           syntax_ == Syntax::tool_history || syntax_ == Syntax::chat_tokenization);
    whitespace();
    if (!input_.empty()) {
        fail("Unexpected data after JSON document");
    }
    return result;
}

Value Parser::value(std::size_t depth, bool ordered) {
    whitespace();
    switch (peek()) {
        case 'n': literal("null"); return {};
        case 't': literal("true"); return true;
        case 'f': literal("false"); return false;
        case '"': return string();
        case '{':
        case '[':
            if (depth == 32) {
                fail("JSON nesting exceeds the limit");
            }
            return accept('{') ? object(depth + 1, ordered) : Value(array(depth + 1, ordered));
        default: return number();
    }
}

Array Parser::array(std::size_t depth, bool ordered) {
    accept('[');
    whitespace();
    Array result;
    if (!accept(']')) {
        do {
            result.push_back(value(depth, ordered));
            whitespace();
        } while (accept(','));
        if (!accept(']')) {
            fail("Expected a JSON array delimiter");
        }
    }
    return result;
}

Value Parser::object(std::size_t depth, bool ordered) {
    whitespace();
    Object result;
    OrderedObject entries;
    std::map<std::string, std::size_t, std::less<>> positions;
    if (!accept('}')) {
        do {
            whitespace();
            auto key = string();
            whitespace();
            if (!accept(':')) {
                fail("Expected a JSON object colon");
            }
            const bool schema = syntax_ == Syntax::completion && depth == 1 &&
                key == "response_format";
            const bool bias = (syntax_ == Syntax::completion || syntax_ == Syntax::native_generate)
                && key == "logit_bias";
            auto entry = value(depth, ordered || schema || bias);
            if (ordered) {
                const auto [position, inserted] = positions.try_emplace(key, entries.size());
                if (inserted) {
                    entries.emplace_back(std::move(key), std::move(entry));
                } else if (syntax_ == Syntax::ordered_last_key_wins ||
                           syntax_ == Syntax::tool_history) {
                    entries[position->second].second = std::move(entry);
                } else {
                    fail("duplicate JSON object key");
                }
            } else if (syntax_ == Syntax::last_key_wins) {
                result.insert_or_assign(std::move(key), std::move(entry));
            } else if (!result.emplace(std::move(key), std::move(entry)).second) {
                fail("duplicate JSON object key");
            }
            whitespace();
        } while (accept(','));
        if (!accept('}')) {
            fail("Expected a JSON object delimiter");
        }
    }
    return ordered ? Value(std::move(entries)) : Value(std::move(result));
}

bool Parser::digits() {
    const auto before = input_.size();
    while (digit(peek())) {
        input_.remove_prefix(1);
    }
    return input_.size() != before;
}

void Parser::strict_number() {
    accept('-');
    if (!accept('0') && !digits()) {
        fail("Invalid JSON number");
    }
    if (accept('.') && !digits()) {
        fail("Expected a digit after the decimal point");
    }
    if (accept('e') || accept('E')) {
        if (!accept('+')) {
            accept('-');
        }
        if (!digits()) {
            fail("Expected a digit in the JSON exponent");
        }
    }
}

std::optional<double> ranged_number(std::string_view text) {
    std::istringstream fallback{std::string(text)};
    fallback.imbue(std::locale::classic());
    double result = 0;
    fallback >> result;
    if (!fallback.eof()) {
        return std::nullopt;
    }
    if (!fallback.fail()) {
        return result;
    }
    if (std::abs(result) == std::numeric_limits<double>::max()) {
        return std::copysign(std::numeric_limits<double>::infinity(), result);
    }
    return std::nullopt;
}

Value Parser::number() {
    const auto before = input_;
    strict_number();
    const auto text = before.substr(0, before.size() - input_.size());
    const auto* end = text.data() + text.size();
    if (text.front() == '-') {
        std::int64_t integer = 0;
        const auto parsed = std::from_chars(text.data(), end, integer);
        if (parsed.ec == std::errc{} && parsed.ptr == end) {
            return integer;
        }
    } else {
        std::uint64_t integer = 0;
        const auto parsed = std::from_chars(text.data(), end, integer);
        if (parsed.ec == std::errc{} && parsed.ptr == end) {
            return integer;
        }
    }
    if ((syntax_ == Syntax::ordered || syntax_ == Syntax::completion ||
         syntax_ == Syntax::ordered_last_key_wins || syntax_ == Syntax::chat_tokenization) &&
        text.find_first_of(".eE") == std::string_view::npos) {
        if (text.size() > 4300 + static_cast<std::size_t>(text.front() == '-')) {
            fail("JSON integer exceeds the digit limit");
        }
        return WideInteger{std::string(text)};
    }
    double result = 0;
    const auto parsed = std::from_chars(text.data(), end, result);
    if (parsed.ec == std::errc{} && parsed.ptr == end && std::isfinite(result)) {
        return result;
    }
    if (parsed.ec == std::errc::result_out_of_range && parsed.ptr == end) {
        if (syntax_ == Syntax::benchmark || syntax_ == Syntax::checkpoint ||
            syntax_ == Syntax::tool_history || syntax_ == Syntax::chat_tokenization) {
            const auto ranged = ranged_number(text);
            if (ranged && (syntax_ == Syntax::checkpoint || *ranged == 0.0)) {
                return *ranged;
            }
        }
        throw NumberRangeError("Invalid JSON number");
    }
    fail("Invalid JSON number");
}

char32_t Parser::hex_quad() {
    if (input_.size() < 4) {
        fail("Incomplete JSON Unicode escape");
    }
    char32_t scalar = 0;
    for (const char character : input_.substr(0, 4)) {
        const auto lower = character | 0x20;
        if (digit(character)) {
            scalar = scalar * 16 + (character - '0');
        } else if (lower >= 'a' && lower <= 'f') {
            scalar = scalar * 16 + (lower - 'a' + 10);
        } else {
            fail("Invalid hex digit in JSON Unicode escape");
        }
    }
    input_.remove_prefix(4);
    return scalar;
}

void Parser::unicode(std::string& output) {
    const auto scalar = hex_quad();
    if (scalar >= 0xd800 && scalar < 0xdc00 && input_.compare(0, 2, "\\u") == 0) {
        input_.remove_prefix(2);
        const auto trailing = hex_quad();
        if (trailing >= 0xdc00 && trailing < 0xe000) {
            append_utf8(output, 0x10000 + (scalar - 0xd800) * 1024 + trailing - 0xdc00);
            return;
        }
    }
    if (scalar >= 0xd800 && scalar < 0xe000) {
        fail("Unpaired surrogate in JSON string");
    }
    append_utf8(output, scalar);
}

std::string Parser::string() {
    if (!accept('"')) {
        fail("Expected a JSON string");
    }
    std::string result;
    while (!input_.empty()) {
        if (accept('"')) {
            return result;
        }
        const unsigned char character = peek();
        input_.remove_prefix(1);
        if (character < 0x20) {
            fail("Unescaped control character in JSON string");
        }
        if (character != '\\') {
            result.push_back(static_cast<char>(character));
            continue;
        }
        const char escape = peek();
        if (input_.empty()) {
            fail("Incomplete JSON escape");
        }
        input_.remove_prefix(1);
        switch (escape) {
            case '"': case '\\': case '/': result.push_back(escape); break;
            case 'b': result.push_back('\b'); break;
            case 'f': result.push_back('\f'); break;
            case 'n': result.push_back('\n'); break;
            case 'r': result.push_back('\r'); break;
            case 't': result.push_back('\t'); break;
            case 'u': unicode(result); break;
            default: fail("Invalid JSON escape");
        }
    }
    fail("Unterminated JSON string");
}

}

std::optional<std::string_view> Value::string() const {
    const auto* value = std::get_if<std::string>(&data_);
    return value ? std::optional<std::string_view>(*value) : std::nullopt;
}

std::optional<std::int64_t> Value::signed_integer() const {
    const auto* value = std::get_if<std::int64_t>(&data_);
    return value ? std::optional(*value) : std::nullopt;
}

std::optional<std::uint64_t> Value::unsigned_integer() const {
    if (const auto* value = std::get_if<std::uint64_t>(&data_)) {
        return *value;
    }
    if (const auto* value = std::get_if<std::int64_t>(&data_); value && *value >= 0) {
        return static_cast<std::uint64_t>(*value);
    }
    return std::nullopt;
}

std::optional<bool> Value::boolean() const {
    const auto* value = std::get_if<bool>(&data_);
    return value ? std::optional<bool>(*value) : std::nullopt;
}

std::optional<double> Value::number() const {
    if (const auto* value = std::get_if<double>(&data_)) {
        return *value;
    }
    if (const auto value = signed_integer()) {
        return static_cast<double>(*value);
    }
    if (const auto value = unsigned_integer()) {
        return static_cast<double>(*value);
    }
    return std::nullopt;
}

Value parse(std::string_view input) {
    return parse(input, Syntax::strict);
}

Value parse(std::string_view input, Syntax syntax) {
    if (!valid_utf8(input)) {
        throw ParseError("Invalid UTF-8 in JSON document");
    }
    return Parser(input, syntax).document();
}

const Value* get(const Object& object, std::string_view key) {
    const auto entry = object.find(key);
    return entry == object.end() ? nullptr : &entry->second;
}

const Value* get(const OrderedObject& object, std::string_view key) {
    for (const auto& [name, value] : object) {
        if (name == key) {
            return &value;
        }
    }
    return nullptr;
}

std::optional<std::string_view> get_string(const Object& object, std::string_view key) {
    const auto* value = get(object, key);
    return value ? value->string() : std::nullopt;
}

std::optional<bool> get_boolean(const Object& object, std::string_view key) {
    const auto* value = get(object, key);
    return value ? value->boolean() : std::nullopt;
}

void append_quoted(std::string& output, std::string_view value) {
    append_quoted(output, value, false);
}

void append_quoted(std::string& output, std::string_view value, bool short_control_escapes) {
    constexpr std::string_view hex = "0123456789abcdef";
    output += '"';
    for (const unsigned char character : value) {
        switch (character) {
            case '"': output += "\\\""; break;
            case '\\': output += "\\\\"; break;
            case '\t': output += "\\t"; break;
            case '\n': output += "\\n"; break;
            case '\r': output += "\\r"; break;
            default:
                if (short_control_escapes && (character == '\b' || character == '\f')) {
                    output += character == '\b' ? "\\b" : "\\f";
                    break;
                }
                if (character < 0x20) {
                    output += "\\u00";
                    output += hex[character >> 4];
                    output += hex[character & 0xf];
                } else {
                    output += static_cast<char>(character);
                }
        }
    }
    output += '"';
}

}
