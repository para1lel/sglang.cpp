#include "sglang/json.h"

#include "sglang/utf8.h"

#include <charconv>
#include <cmath>

namespace sglang::json {
namespace {

bool digit(char character) {
    return character >= '0' && character <= '9';
}

class Parser {
public:
    explicit Parser(std::string_view input) : input_(input) {}

    Value document();

private:
    char peek() const { return input_.empty() ? '\0' : input_.front(); }
    bool accept(char character);
    void whitespace();
    void literal(std::string_view word);
    [[noreturn]] void fail(const char* message) const { throw ParseError(message); }
    Value value(std::size_t depth);
    Array array(std::size_t depth);
    Object object(std::size_t depth);
    Value number();
    bool digits();
    void strict_number();
    std::string string();
    char32_t hex_quad();
    void unicode(std::string& output);

    std::string_view input_;
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
    auto result = value(0);
    whitespace();
    if (!input_.empty()) {
        fail("Unexpected data after JSON document");
    }
    return result;
}

Value Parser::value(std::size_t depth) {
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
            return accept('{') ? Value(object(depth + 1)) : Value(array(depth + 1));
        default: return number();
    }
}

Array Parser::array(std::size_t depth) {
    accept('[');
    whitespace();
    Array result;
    if (!accept(']')) {
        do {
            result.push_back(value(depth));
            whitespace();
        } while (accept(','));
        if (!accept(']')) {
            fail("Expected a JSON array delimiter");
        }
    }
    return result;
}

Object Parser::object(std::size_t depth) {
    whitespace();
    Object result;
    if (!accept('}')) {
        do {
            whitespace();
            auto key = string();
            whitespace();
            if (!accept(':')) {
                fail("Expected a JSON object colon");
            }
            auto entry = value(depth);
            if (!result.emplace(std::move(key), std::move(entry)).second) {
                fail("Duplicate JSON object key");
            }
            whitespace();
        } while (accept(','));
        if (!accept('}')) {
            fail("Expected a JSON object delimiter");
        }
    }
    return result;
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
    double result = 0;
    const auto parsed = std::from_chars(text.data(), end, result);
    if (parsed.ec == std::errc{} && parsed.ptr == end && std::isfinite(result)) {
        return result;
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

Value parse(std::string_view input) {
    if (!valid_utf8(input)) {
        throw ParseError("Invalid UTF-8 in JSON document");
    }
    return Parser(input).document();
}

const Value* get(const Object& object, std::string_view key) {
    const auto entry = object.find(key);
    return entry == object.end() ? nullptr : &entry->second;
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
