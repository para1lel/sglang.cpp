#pragma once

#include "sglang/server/chat_output.h"

namespace sglang::server {

class EventStream;
class Socket;

class AnthropicMessages {
public:
    AnthropicMessages(const Socket& connection, EventStream& stream, bool streaming,
                      std::size_t prompt_tokens);

    void admit(std::size_t cached_tokens);
    bool accept(const ChatOutputEvent& delta);
    void finish(std::string_view reason, std::size_t output_tokens, std::size_t cached_tokens);
    void fail(std::string_view type, const char* message);

private:
    bool start(bool failed = false);
    bool event(std::string_view kind, json::OrderedObject fields = json::OrderedObject{});
    bool close_block();
    bool open_block(ChatOutputKind kind, const json::OrderedObject& block);
    bool tool(const ChatOutputEvent& delta);
    json::OrderedObject message(json::Array content) const;
    json::OrderedObject usage(std::size_t output, std::size_t cached) const;

    const Socket& connection_;
    EventStream& stream_;
    const bool streaming_;
    const std::size_t prompt_tokens_;
    const std::string id_;
    std::string thinking_;
    std::string text_;
    json::Array tools_;
    std::optional<ChatOutputKind> block_;
    bool started_ = false;
    std::size_t index_ = 0;
    std::size_t cached_tokens_ = 0;
};

}
