#include "sglang/tokenizer/regex.h"

#include "sglang/utf8.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace sglang::tokenizer::regex {
namespace {

constexpr std::size_t max_pattern_bytes = 1024;
constexpr int max_group_depth = 32;
constexpr int max_repeat = 64;
constexpr char32_t max_scalar = 0x10ffff;
constexpr char32_t end = max_scalar + 1;
using Kind = Expression::Kind;

[[noreturn]] void invalid() {
    throw std::invalid_argument("Invalid or unsupported regex syntax");
}

bool digit(char32_t value) { return value >= U'0' && value <= U'9'; }

std::vector<Range> normalize(std::vector<Range> ranges, bool negate) {
    std::sort(ranges.begin(), ranges.end(), [](Range left, Range right) {
        return left.first < right.first;
    });
    std::vector<Range> merged;
    for (const auto range : ranges) {
        if (!merged.empty() && range.first <= merged.back().last + 1) {
            merged.back().last = std::max(merged.back().last, range.last);
        } else {
            merged.push_back(range);
        }
    }
    if (!negate) {
        return merged;
    }
    std::vector<Range> complement;
    char32_t first = 0;
    for (const auto range : merged) {
        if (first < range.first) {
            complement.push_back({first, range.first - 1});
        }
        first = range.last + 1;
    }
    if (first <= max_scalar) {
        complement.push_back({first, max_scalar});
    }
    return complement;
}

Expression children(Kind kind, std::vector<Expression> values) {
    if (values.size() == 1) {
        return std::move(values.front());
    }
    return {kind, {}, std::move(values)};
}

class Parser {
public:
    explicit Parser(std::u32string_view input) : input_(input) {}
    Expression pattern();

private:
    char32_t peek() const { return input_.empty() ? end : input_.front(); }
    bool take(char32_t value) {
        if (peek() != value || input_.empty()) {
            return false;
        }
        input_.remove_prefix(1);
        return true;
    }
    char32_t next() {
        const char32_t value = peek();
        if (!take(value)) {
            invalid();
        }
        return value;
    }
    void require(char32_t value) {
        if (!take(value)) {
            invalid();
        }
    }

    Expression alternate(int depth);
    Expression sequence(int depth);
    Expression atom(int depth);
    Expression quantify(Expression value);
    std::vector<Range> character_class();
    std::vector<Range> class_atom();
    std::vector<Range> escape();
    char32_t hex(int digits);
    int count();

    std::u32string_view input_;
};

Expression Parser::pattern() {
    take(U'^');
    auto result = alternate(0);
    take(U'$');
    if (!input_.empty()) {
        invalid();
    }
    return result;
}

Expression Parser::alternate(int depth) {
    std::vector<Expression> values;
    do {
        values.push_back(sequence(depth));
    } while (take(U'|'));
    return children(Kind::alternate, std::move(values));
}

Expression Parser::sequence(int depth) {
    std::vector<Expression> values;
    while (peek() != U'|' && peek() != U')' && peek() != U'$' && peek() != end) {
        values.push_back(quantify(atom(depth)));
    }
    return children(Kind::sequence, std::move(values));
}

Expression Parser::atom(int depth) {
    const char32_t value = next();
    switch (value) {
        case U'(':
            if (depth == max_group_depth) {
                throw std::invalid_argument("regex group nesting exceeds 32");
            }
            if (take(U'?')) {
                require(U':');
            }
            {
                auto result = alternate(depth + 1);
                require(U')');
                return result;
            }
        case U'[': return {Kind::character, character_class(), {}};
        case U'.': return {Kind::character, {{0, U'\n' - 1}, {U'\n' + 1, max_scalar}}, {}};
        case U'\\': return {Kind::character, escape(), {}};
        case U'^': case U'{': case U'}': case U']': case U'+': case U'*': case U'?':
            invalid();
        default: return {Kind::character, {{value, value}}, {}};
    }
}

Expression Parser::quantify(Expression value) {
    int minimum = 0;
    int maximum = 1;
    if (take(U'+')) {
        minimum = 1;
        maximum = -1;
    } else if (take(U'*')) {
        maximum = -1;
    } else if (take(U'{')) {
        minimum = count();
        maximum = minimum;
        if (take(U',')) {
            maximum = peek() == U'}' ? -1 : count();
        }
        require(U'}');
    } else if (!take(U'?')) {
        return value;
    }
    if (maximum >= 0 && maximum < minimum) {
        invalid();
    }
    std::vector<Expression> body;
    body.push_back(std::move(value));
    return {Kind::repeat, {}, std::move(body), minimum, maximum};
}

int Parser::count() {
    if (!digit(peek())) {
        invalid();
    }
    int result = 0;
    while (digit(peek())) {
        result = result * 10 + static_cast<int>(next() - U'0');
        if (result > max_repeat) {
            throw std::invalid_argument("regex repetition bound exceeds 64");
        }
    }
    return result;
}

std::vector<Range> Parser::character_class() {
    const bool negate = take(U'^');
    std::vector<Range> result;
    do {
        auto first = class_atom();
        if (peek() == U'-' && input_.size() > 1 && input_[1] != U']') {
            take(U'-');
            const auto last = class_atom();
            if (first.size() != 1 || last.size() != 1 ||
                first.front().first != first.front().last ||
                last.front().first != last.front().last ||
                first.front().first > last.front().last) {
                invalid();
            }
            first.front().last = last.front().last;
        }
        result.insert(result.end(), first.begin(), first.end());
    } while (!take(U']'));
    return normalize(std::move(result), negate);
}

std::vector<Range> Parser::class_atom() {
    const char32_t value = next();
    if (value == U'\\') {
        return escape();
    }
    if (value == U']' || value == U'[') {
        invalid();
    }
    return {{value, value}};
}

std::vector<Range> Parser::escape() {
    const char32_t value = next();
    switch (value) {
        case U'd': return {{U'0', U'9'}};
        case U'D': return normalize({{U'0', U'9'}}, true);
        case U'w': return {{U'0', U'9'}, {U'A', U'Z'}, {U'_', U'_'}, {U'a', U'z'}};
        case U'W': return normalize({{U'0', U'9'}, {U'A', U'Z'}, {U'_', U'_'}, {U'a', U'z'}},
                                    true);
        case U's': return {{U'\t', U'\r'}, {U' ', U' '}};
        case U'S': return normalize({{U'\t', U'\r'}, {U' ', U' '}}, true);
        default: break;
    }
    constexpr std::u32string_view letters = U"nrtfv";
    constexpr std::u32string_view controls = U"\n\r\t\f\v";
    if (const auto index = letters.find(value); index != letters.npos) {
        return {{controls[index], controls[index]}};
    }
    if (value == U'x' || value == U'u' || value == U'U') {
        const auto scalar = hex(value == U'x' ? 2 : value == U'u' ? 4 : 8);
        return {{scalar, scalar}};
    }
    if (std::u32string_view(U"\\.^$|?*+()[]{}-/").find(value) == std::u32string_view::npos) {
        invalid();
    }
    return {{value, value}};
}

char32_t Parser::hex(int digits) {
    char32_t scalar = 0;
    for (int index = 0; index < digits; ++index) {
        const char32_t value = next();
        scalar *= 16;
        if (digit(value)) {
            scalar += value - U'0';
        } else if (value >= U'a' && value <= U'f') {
            scalar += value - U'a' + 10;
        } else if (value >= U'A' && value <= U'F') {
            scalar += value - U'A' + 10;
        } else {
            invalid();
        }
        if (scalar > max_scalar) {
            invalid();
        }
    }
    if (scalar >= 0xd800 && scalar <= 0xdfff) {
        invalid();
    }
    return scalar;
}

}

Expression parse(std::string_view pattern) {
    if (pattern.empty() || pattern.size() > max_pattern_bytes) {
        throw std::invalid_argument("regex must contain 1 to 1024 UTF-8 bytes");
    }
    const auto scalars = read_utf8(pattern);
    return Parser(scalars).pattern();
}

}
