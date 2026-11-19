#pragma once

#include "sglang/json.h"

#include <filesystem>

namespace sglang::server {

enum class ChatInput { checkpoint_template, http };

class ChatTemplate {
public:
    explicit ChatTemplate(const std::filesystem::path& checkpoint);
    std::string render(const json::OrderedObject& chat, ChatInput input) const;

private:
    std::string xhigh_;
    std::string low_;
    std::string tools_intro_;
    std::string tools_instructions_;
};

}
