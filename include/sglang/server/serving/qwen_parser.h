#pragma once

#include "sglang/server/chat_output.h"

namespace sglang::model {

std::optional<server::SeparatedReasoning> separate_qwen_reasoning(std::string_view text,
                                                               std::string_view parser);
std::optional<std::string> parse_qwen_tool_calls(std::string_view text, const json::Array* tools,
                                                std::string_view parser);

}
