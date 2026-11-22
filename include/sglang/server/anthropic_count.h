#pragma once

#include "sglang/server/chat_template.h"

namespace sglang::server {

struct AnthropicCountChat {
    json::Array messages;
    ChatTools tools;
    json::OrderedObject options;
};

AnthropicCountChat anthropic_count_chat(const json::Value& request, bool reasoning_parser);

}
