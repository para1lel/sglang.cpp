#pragma once

#include "sglang/json.h"

#include <filesystem>

namespace sglang::server {

enum class ChatInput { checkpoint_template, http };

enum class ChatToolChoice { none, automatic, required, named };

struct ChatTools {
    json::Array tools;
    ChatToolChoice choice = ChatToolChoice::automatic;
    bool parallel = true;
    bool strict = false;

    bool forced() const {
        return choice == ChatToolChoice::required || choice == ChatToolChoice::named;
    }
};

ChatTools chat_tools(const json::OrderedObject& chat);
json::Value parse_chat_arguments(std::string_view arguments, json::Syntax syntax);
std::string chat_template_json(const json::Value& value, bool compact = false);
std::string_view trim_chat_text(std::string_view text);

class ChatTemplate {
public:
    explicit ChatTemplate(const std::filesystem::path& checkpoint);
    std::string render(const json::OrderedObject& chat, ChatInput input) const;
    std::string render(const json::OrderedObject& chat, ChatInput input,
                       const ChatTools& selection) const;

private:
    std::string xhigh_;
    std::string low_;
    std::string tools_intro_;
    std::string tools_instructions_;
};

}
