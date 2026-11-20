#pragma once

#include "sglang/tokenizer/tokenizer.h"

#include <list>
#include <mutex>
#include <string>
#include <string_view>

namespace sglang {

struct DecoderOptions;
class TokenConstraint;
enum class TriggeredConstraintMode;

namespace server {

class BatchService;
class ChatTemplate;
class Socket;
class VisionCompletion;
struct ChatTools;

enum class SamplingDefaults { model, openai };

struct ChatParsers {
    bool qwen3_reasoning = false;
    bool qwen3_coder_tools = false;
};

class CompletionTokenizer {
public:
    explicit CompletionTokenizer(const std::filesystem::path& checkpoint,
                                 SamplingDefaults sampling = SamplingDefaults::model,
                                 ChatParsers parsers = ChatParsers());
    ~CompletionTokenizer();
    const Tokenizer& get() const;
    const ChatTemplate& chat_template() const;
    const DecoderOptions& chat_sampling_defaults() const;
    ChatParsers chat_parsers() const { return chat_parsers_; }
    std::shared_ptr<const TokenConstraint> tool_constraint(const ChatTools& tools,
                                                           bool prompt_has_open_think) const;

private:
    struct ToolConstraintEntry {
        std::string pattern;
        TriggeredConstraintMode mode;
        bool parallel;
        bool prompt_has_open_think;
        std::shared_ptr<const TokenConstraint> constraint;
    };

    const std::filesystem::path checkpoint_;
    const SamplingDefaults sampling_mode_;
    const ChatParsers chat_parsers_;
    mutable std::once_flag loaded_;
    mutable std::unique_ptr<const Tokenizer> tokenizer_;
    mutable std::once_flag chat_loaded_;
    mutable std::unique_ptr<const ChatTemplate> chat_template_;
    mutable std::once_flag sampling_loaded_;
    mutable std::unique_ptr<const DecoderOptions> sampling_defaults_;
    mutable std::mutex tool_constraints_mutex_;
    mutable std::list<ToolConstraintEntry> tool_constraints_;
};

void complete(const Socket& connection, BatchService& service, int slot, std::string_view body);
void complete(const Socket& connection, BatchService& service, int slot,
              const CompletionTokenizer& tokenizer, std::string_view body);
void complete_chat(const Socket& connection, BatchService& service, int slot,
                   const CompletionTokenizer& tokenizer, std::string_view body,
                   VisionCompletion* vision = nullptr);
void write_error(const Socket& connection, std::string_view status, const char* message);

}
}
