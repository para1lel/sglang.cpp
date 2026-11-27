#pragma once

#include "sglang/json.h"

namespace sglang::server {

enum class ChatInput { checkpoint_template, http };

enum class ChatToolChoice { none, automatic, required, named };

struct ChatTools {
    json::Array tools;
    ChatToolChoice choice = ChatToolChoice::automatic;
    bool parallel = true;
    bool strict = false;

    bool forced() const;
};

json::Value parse_chat_arguments(std::string_view arguments, json::Syntax syntax);
std::string chat_template_json(const json::Value& value, bool compact = false);
std::string_view trim_chat_text(std::string_view text);

class ChatTemplate {
public:
    virtual ~ChatTemplate();
    virtual std::string render(const json::OrderedObject& chat, ChatInput input) const = 0;
    virtual std::string render(const json::OrderedObject& chat, ChatInput input,
                               const ChatTools& selection) const = 0;
};

}
