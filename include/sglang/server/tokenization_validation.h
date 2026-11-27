#pragma once

#include "sglang/json.h"

namespace sglang::server {

namespace detail {

enum class IntegerSeparators { single, repeated };

struct IntegerText {
    std::string_view significant;
    std::size_t digit_count;
    bool negative;
};

std::optional<IntegerText>
scan_integer_text(std::string_view text, IntegerSeparators separators = IntegerSeparators::single);

}

bool chat_truthy(const json::Value& value);
std::optional<bool> chat_boolean(const json::Value& value);
std::optional<double> chat_number(const json::Value& value);
void validate_chat_generation(const json::OrderedObject& body);
json::Value normalize_tokenization_tool_schema(const json::Value& value, std::size_t index);

}
