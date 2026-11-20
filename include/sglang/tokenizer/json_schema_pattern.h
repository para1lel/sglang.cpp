#pragma once

#include "sglang/json.h"

namespace sglang::tokenizer {

std::string regex_literal(std::string_view text);
inline constexpr std::string_view xml_string_pattern = "(?x)";
std::string xml_parameters_pattern(const json::Value* schema);

}
