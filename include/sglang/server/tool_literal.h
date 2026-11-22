#pragma once

#include "sglang/json.h"

namespace sglang::server {

std::optional<json::Value> parse_tool_literal(std::string_view text);

}
