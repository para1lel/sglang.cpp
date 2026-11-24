#pragma once

#include "sglang/runtime/model/serving_profile.h"

namespace sglang::server {

ChatTools select_qwen_chat_tools(const json::OrderedObject& chat);
std::unique_ptr<const ChatTemplate> load_qwen_chat_template(
    const std::filesystem::path& checkpoint);

}

namespace sglang::model {

inline constexpr std::string_view qwen_reasoning_parser = "qwen3";
inline constexpr std::string_view qwen_tool_call_parser = "qwen3_coder";

extern const ServingProfile qwen_serving_profile;

}
