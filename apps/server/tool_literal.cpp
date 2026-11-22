#include "sglang/server/tool_literal.h"

#include "sglang/utf8.h"

namespace sglang::server {
namespace {

int hex_digit(char character) {
    if (character >= '0' && character <= '9') return character - '0';
    if (character >= 'a' && character <= 'f') return character - 'a' + 10;
    if (character >= 'A' && character <= 'F') return character - 'A' + 10;
    return -1;
}

bool decimal_digit(char character) {
    return character >= '0' && character <= '9';
}

class Literal {
public:
    explicit Literal(std::string_view text) : text_(text) {}
    std::optional<json::Value> parse();

private:
    char peek() const { return text_.empty() ? '\0' : text_.front(); }
    bool take(char character);
    void whitespace();
    bool value(unsigned depth);
    bool collection(char opener, unsigned depth);
    bool string();
    bool escape(std::string& decoded);
    bool unicode(std::string& decoded, unsigned count);
    bool number();
    void digits(std::string& output);

    std::string_view text_;
    std::string json_;
};

bool Literal::take(char character) {
    if (text_.empty() || peek() != character) return false;
    text_.remove_prefix(1);
    return true;
}

void Literal::whitespace() {
    const auto first = text_.find_first_not_of(" \t\r\n\f");
    text_.remove_prefix(first == text_.npos ? text_.size() : first);
}

std::optional<json::Value> Literal::parse() {
    if (!value(0)) return std::nullopt;
    whitespace();
    if (!text_.empty()) return std::nullopt;
    try {
        return json::parse(json_, json::Syntax::ordered_last_key_wins);
    } catch (const json::ParseError&) {
        return std::nullopt;
    }
}

bool Literal::value(unsigned depth) {
    whitespace();
    switch (const auto character = peek(); character) {
    case '{':
    case '[':
    case '(':
        return depth < 32 && collection(character, depth + 1);
    case '\'':
    case '"':
        return string();
    case 'T':
    case 'F':
    case 'N': {
        const std::string_view word = character == 'T' ? "True" :
            character == 'F' ? "False" : "None";
        if (text_.substr(0, word.size()) != word) return false;
        text_.remove_prefix(word.size());
        json_ += character == 'T' ? "true" : character == 'F' ? "false" : "null";
        return true;
    }
    default:
        return number();
    }
}

bool Literal::collection(char opener, unsigned depth) {
    text_.remove_prefix(1);
    const bool object = opener == '{';
    const char closer = object ? '}' : opener == '[' ? ']' : ')';
    const auto start = json_.size();
    json_ += object ? '{' : '[';
    whitespace();
    bool first = true;
    while (!take(closer)) {
        if (!first) json_ += ',';
        if (object) {
            if (!string()) return false;
            whitespace();
            if (!take(':')) return false;
            json_ += ':';
        }
        if (!value(depth)) return false;
        whitespace();
        if (take(closer)) {
            if (first && opener == '(') {
                json_.erase(start, 1);
                return true;
            }
            break;
        }
        if (!take(',')) return false;
        whitespace();
        first = false;
    }
    json_ += object ? '}' : ']';
    return true;
}

bool Literal::string() {
    const auto quote = peek();
    if (quote != '\'' && quote != '"') return false;
    text_.remove_prefix(1);
    std::string decoded;
    while (!text_.empty()) {
        const auto character = peek();
        text_.remove_prefix(1);
        if (character == quote) {
            json::append_quoted(json_, decoded, true);
            return true;
        }
        if (character == '\\') {
            if (!escape(decoded)) return false;
        } else {
            if (character == '\n' || character == '\r' || character == '\0') return false;
            decoded += character;
        }
    }
    return false;
}

bool Literal::unicode(std::string& decoded, unsigned count) {
    if (text_.size() < count) return false;
    char32_t scalar = 0;
    for (char character : text_.substr(0, count)) {
        const auto digit = hex_digit(character);
        if (digit < 0) return false;
        scalar = scalar * 16 + static_cast<unsigned>(digit);
    }
    if (scalar > 0x10ffff || (scalar >= 0xd800 && scalar <= 0xdfff)) return false;
    text_.remove_prefix(count);
    append_utf8(decoded, scalar);
    return true;
}

bool Literal::escape(std::string& decoded) {
    if (text_.empty()) return false;
    const auto character = peek();
    text_.remove_prefix(1);
    switch (character) {
    case '\\': case '\'': case '"': decoded += character; return true;
    case 'a': decoded += '\a'; return true;
    case 'b': decoded += '\b'; return true;
    case 'f': decoded += '\f'; return true;
    case 'n': decoded += '\n'; return true;
    case 'r': decoded += '\r'; return true;
    case 't': decoded += '\t'; return true;
    case 'v': decoded += '\v'; return true;
    case 'x': return unicode(decoded, 2);
    case 'u': return unicode(decoded, 4);
    case 'U': return unicode(decoded, 8);
    case '\r': take('\n'); return true;
    case '\n': return true;
    case 'N': case '\0': return false;
    default:
        if (character >= '0' && character <= '7') {
            char32_t scalar = character - '0';
            for (unsigned count = 1; count < 3 && peek() >= '0' && peek() <= '7'; ++count) {
                scalar = scalar * 8 + (peek() - '0');
                text_.remove_prefix(1);
            }
            append_utf8(decoded, scalar);
        } else {
            decoded += '\\';
            decoded += character;
        }
        return true;
    }
}

void Literal::digits(std::string& output) {
    while (decimal_digit(peek())) {
        output += peek();
        text_.remove_prefix(1);
        if (peek() == '_' && text_.size() > 1 && decimal_digit(text_[1])) {
            text_.remove_prefix(1);
        }
    }
}

bool Literal::number() {
    const bool negative = take('-');
    if (negative || take('+')) whitespace();
    std::string whole;
    std::string fraction;
    digits(whole);
    const bool has_point = take('.');
    if (has_point) digits(fraction);
    if (whole.empty() && fraction.empty()) return false;
    const bool has_exponent = take('e') || take('E');
    const auto first = whole.find_first_not_of('0');
    if (!has_point && !has_exponent && first != whole.npos && first != 0) return false;
    if (negative) json_ += '-';
    if (first == whole.npos) {
        json_ += '0';
    } else {
        json_ += std::string_view(whole).substr(first);
    }
    if (has_point) {
        json_ += '.';
        json_ += fraction.empty() ? "0" : fraction;
    }
    if (has_exponent) {
        json_ += 'e';
        if (take('-')) json_ += '-';
        else take('+');
        const auto start = json_.size();
        digits(json_);
        if (start == json_.size()) return false;
    }
    return true;
}

}

std::optional<json::Value> parse_tool_literal(std::string_view text) {
    return Literal(text).parse();
}

}
