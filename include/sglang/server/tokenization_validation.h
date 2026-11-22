#pragma once

#include "sglang/json.h"

namespace sglang::server {

std::optional<bool> chat_boolean(const json::Value& value);
std::optional<double> chat_number(const json::Value& value);
void validate_chat_generation(const json::OrderedObject& body);
json::Value normalize_tokenization_tool_schema(const json::Value& value, std::size_t index);

}
