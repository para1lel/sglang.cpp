#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <system_error>

namespace sglang::server::qwen_tool_syntax {

inline constexpr std::string_view tool_open = "<tool_call>";
inline constexpr std::string_view tool_close = "</tool_call>";
inline constexpr std::string_view function_open = "<function=";
inline constexpr std::string_view function_close = "</function>";
inline constexpr std::string_view parameter_open = "<parameter=";
inline constexpr std::string_view parameter_close = "</parameter>";

std::string_view trim(std::string_view text, std::string_view whitespace);
std::string_view trim_line_feeds(std::string_view text);
bool equals_folded(std::string_view text, std::string_view lower);
bool append_integer(std::string& output, std::string_view text,
                    std::size_t max_digits = std::string_view::npos);
std::errc parse_float(std::string_view& text, double& value);
void append_float_spelling(std::string& output, std::string_view spelling);

}
