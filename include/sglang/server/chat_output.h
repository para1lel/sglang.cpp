#pragma once

#include "sglang/json.h"

#include <functional>
#include <memory>
#include <set>

namespace sglang::server {

enum class ChatArgumentType { string, integer, number, boolean, object, array, null };

struct ChatToolProperty {
    std::vector<ChatArgumentType> types;
    std::map<std::string, std::string, std::less<>> enum_values;
};

struct ChatToolSchema {
    std::map<std::string, ChatToolProperty, std::less<>> properties;
    std::set<std::string, std::less<>> required;
    bool additional_properties = true;
    bool strict = false;
};

struct ChatOutputOptions {
    bool separate_reasoning = false;
    bool stream_reasoning = true;
    bool prompt_has_open_think = false;
    bool complete_truncated_calls = false;
    bool parallel_tool_calls = true;
    std::map<std::string, ChatToolSchema, std::less<>> tools;

    bool enabled() const { return separate_reasoning || !tools.empty(); }
    bool accepts_tool_calls(std::size_t calls) const {
        return !tools.empty() && (parallel_tool_calls || calls == 0);
    }
};

void read_chat_tools(ChatOutputOptions& options, const json::Array& tools);
void read_chat_tools(ChatOutputOptions& options, const json::Value* tools);

enum class ChatOutputKind { content, reasoning, tool_call };
enum class ChatEnd { stop, length, cancel };

struct ChatOutputEvent {
    ChatOutputKind kind;
    std::string_view text;
    std::size_t index = 0;
    std::string_view name;
};

struct SeparatedReasoning {
    std::string reasoning;
    std::string text;
};

// An event's views are valid for the duration of the callback.
using ChatOutputSink = std::function<bool(const ChatOutputEvent&)>;
class ChatToolFrame;

class ChatOutputParser {
public:
    explicit ChatOutputParser(const ChatOutputOptions& options);
    ~ChatOutputParser();
    static SeparatedReasoning separate_qwen3(std::string_view text, bool force_reasoning);
    bool push(std::string_view text, const ChatOutputSink& sink);
    bool finish(ChatEnd end, const ChatOutputSink& sink);
    std::size_t calls() const { return calls_; }

private:
    enum class Phase { opening, reasoning, content };

    bool drain(bool final, const ChatOutputSink& sink);
    bool emit_text(std::string_view text, const ChatOutputSink& sink);
    bool flush_reasoning(const ChatOutputSink& sink);

    const ChatOutputOptions& options_;
    Phase phase_;
    std::string pending_;
    std::string reasoning_;
    std::unique_ptr<ChatToolFrame> tool_;
    std::size_t calls_ = 0;
    bool truncated_ = false;
};

}
