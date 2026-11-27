#include "qwen_tool_syntax.h"

#include <algorithm>
#include <charconv>

namespace sglang::server::qwen_tool_syntax {

std::string_view trim(std::string_view text, std::string_view whitespace) {
    const auto first = text.find_first_not_of(whitespace);
    return first == std::string_view::npos
               ? std::string_view{}
               : text.substr(first, text.find_last_not_of(whitespace) - first + 1);
}

std::string_view trim_line_feeds(std::string_view text) {
    text.remove_prefix(!text.empty() && text.front() == '\n' ? 1 : 0);
    text.remove_suffix(!text.empty() && text.back() == '\n' ? 1 : 0);
    return text;
}

bool equals_folded(std::string_view text, std::string_view lower) {
    return std::equal(text.begin(), text.end(), lower.begin(), lower.end(), [](char a, char b) {
        const char folded = a >= 'A' && a <= 'Z' ? static_cast<char>(a + ('a' - 'A')) : a;
        return folded == b;
    });
}

bool append_integer(std::string& output, std::string_view text, std::size_t max_digits) {
    const bool negative = !text.empty() && text.front() == '-';
    if (negative || (!text.empty() && text.front() == '+')) {
        text.remove_prefix(1);
    }
    if (text.empty() || text.size() > max_digits ||
        text.find_first_not_of("0123456789") != std::string_view::npos) {
        return false;
    }
    const auto first = text.find_first_not_of('0');
    if (negative && first != std::string_view::npos) {
        output += '-';
    }
    output += first == std::string_view::npos ? "0" : text.substr(first);
    return true;
}

std::errc parse_float(std::string_view& text, double& value) {
    if (text.size() > 1 && text.front() == '+' && text[1] != '-') {
        text.remove_prefix(1);
    }
    if (text.empty()) {
        return std::errc::invalid_argument;
    }
    const auto* begin = text.data();
    const auto* end = begin + text.size();
    const auto parsed = std::from_chars(begin, end, value);
    return parsed.ptr == end ? parsed.ec : std::errc::invalid_argument;
}

void append_float_spelling(std::string& output, std::string_view spelling) {
    output += spelling;
    if (spelling.find_first_of(".eE") == std::string_view::npos) {
        output += ".0";
    }
}

}
