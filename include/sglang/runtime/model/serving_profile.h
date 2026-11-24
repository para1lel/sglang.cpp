#pragma once

#include "sglang/json.h"

#include <filesystem>
#include <memory>
#include <optional>

namespace sglang::server {
class ChatOutputParser;
class ChatTemplate;
class CompletionTokenizer;
struct ChatOutputOptions;
struct ChatTools;
struct TextRequest;
}

namespace sglang::model {

struct ServingProfile {
    std::string_view checkpoint_path;
    std::string_view served_name;
    std::string_view model_type;
    std::string_view architecture;
    bool has_image_understanding;
    bool has_audio_understanding;
    bool supports_inline_system;
    std::string_view reasoning_parser;
    std::string_view tool_call_parser;
    std::string_view load_format;
    std::string_view weight_version;
    std::string_view tokenizer_file;
    void (*normalize_chat_options)(json::OrderedObject& chat);
    bool (*is_chat_template_option)(std::string_view key);
    server::ChatTools (*select_chat_tools)(const json::OrderedObject& chat);
    std::unique_ptr<const server::ChatTemplate> (*load_chat_template)(
        const std::filesystem::path& checkpoint);
    void (*prepare_chat_output)(server::TextRequest& request, const json::OrderedObject& chat,
                                std::string_view prompt, const server::CompletionTokenizer& loader,
                                const server::ChatTools& tools);
    void (*start_chat_output)(std::optional<server::ChatOutputParser>& parser,
                              const server::ChatOutputOptions& options);
};

}
