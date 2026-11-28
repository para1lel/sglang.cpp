#include "sglang/server/anthropic_messages.h"

#include "sglang/server/chat_template.h"
#include "sglang/server/http.h"

#include <algorithm>

namespace sglang::server {
namespace {

using json::OrderedObject;

std::uint64_t count(std::size_t tokens) {
    return static_cast<std::uint64_t>(tokens);
}

}

AnthropicMessages::AnthropicMessages(const Socket& connection, EventStream& stream, bool streaming,
                                     std::size_t prompt_tokens)
    : connection_(connection), stream_(stream), streaming_(streaming),
      prompt_tokens_(prompt_tokens), id_(random_request_id("msg_")) {
}

void AnthropicMessages::admit(std::size_t cached_tokens) { cached_tokens_ = cached_tokens; }

OrderedObject AnthropicMessages::message(json::Array content) const {
    return {{"id", id_}, {"type", "message"}, {"role", "assistant"},
            {"content", std::move(content)}, {"model", "Qwen3.8-27B-FP8"}};
}

OrderedObject AnthropicMessages::usage(std::size_t output, std::size_t cached) const {
    const auto read = std::min(cached, prompt_tokens_);
    OrderedObject fields{{"input_tokens", count(prompt_tokens_ - read)},
                         {"output_tokens", count(output)}};
    if (read) fields.emplace_back("cache_read_input_tokens", count(read));
    return fields;
}

bool AnthropicMessages::event(std::string_view kind, OrderedObject fields) {
    fields.emplace_back("type", std::string(kind));
    return stream_.write(chat_template_json(fields), kind);
}

bool AnthropicMessages::start(bool failed) {
    if (!streaming_ || started_) return true;
    auto initial = message({});
    initial.emplace_back("usage", failed ?
        OrderedObject{{"input_tokens", 0}, {"output_tokens", 0}} : usage(0, cached_tokens_));
    started_ = stream_.start() && event("message_start", {{"message", std::move(initial)}});
    return started_;
}

bool AnthropicMessages::close_block() {
    if (!block_) return true;
    block_.reset();
    return event("content_block_stop", {{"index", count(index_++)}});
}

bool AnthropicMessages::open_block(ChatOutputKind kind, const OrderedObject& block) {
    if (block_ == kind) return true;
    if (!close_block()) return false;
    block_ = kind;
    return event("content_block_start", {{"index", count(index_)}, {"content_block", block}});
}

bool AnthropicMessages::tool(const ChatOutputEvent& delta) {
    const auto id = "call_" + id_.substr(4) + '_' + std::to_string(delta.index);
    OrderedObject block{{"type", "tool_use"}, {"id", id}, {"name", std::string(delta.name)}};
    if (!streaming_) {
        block.emplace_back("input", parse_chat_arguments(delta.text, json::Syntax::ordered));
        tools_.emplace_back(std::move(block));
        return true;
    }
    block.emplace_back("input", OrderedObject{});
    return open_block(delta.kind, block) &&
        event("content_block_delta", {{"index", count(index_)}, {"delta", OrderedObject{
            {"type", "input_json_delta"}, {"partial_json", std::string(delta.text)}}}}) &&
        close_block();
}

bool AnthropicMessages::accept(const ChatOutputEvent& delta) {
    if (streaming_ && !delta.text.empty() && !start()) return false;
    if (delta.kind == ChatOutputKind::tool_call) return tool(delta);
    if (delta.text.empty()) return true;
    const bool thinking = delta.kind == ChatOutputKind::reasoning;
    if (!streaming_) {
        (thinking ? thinking_ : text_) += delta.text;
        return true;
    }
    const char* field = thinking ? "thinking" : "text";
    return open_block(delta.kind, {{"type", field}, {field, ""}}) &&
        event("content_block_delta", {{"index", count(index_)}, {"delta", OrderedObject{
            {"type", thinking ? "thinking_delta" : "text_delta"},
            {field, std::string(delta.text)}}}});
}

void AnthropicMessages::finish(std::string_view reason, std::size_t output, std::size_t cached) {
    const char* stop = reason == "tool_calls" ? "tool_use" :
        reason == "length" ? "max_tokens" : "end_turn";
    if (streaming_) {
        admit(cached);
        if (start() && close_block() &&
            event("message_delta", {{"delta", OrderedObject{{"stop_reason", stop}}},
                  {"usage", OrderedObject{{"output_tokens", count(output)}}}}) &&
            event("message_stop")) {
            stream_.end();
        }
        return;
    }
    json::Array content;
    if (!thinking_.empty()) {
        content.emplace_back(OrderedObject{
            {"type", "thinking"}, {"thinking", std::move(thinking_)}});
    }
    if (!text_.empty() || (content.empty() && tools_.empty())) {
        content.emplace_back(OrderedObject{{"type", "text"}, {"text", std::move(text_)}});
    }
    for (auto& tool : tools_) content.push_back(std::move(tool));
    auto response = message(std::move(content));
    response.emplace_back("stop_reason", stop);
    response.emplace_back("usage", usage(output, cached));
    write_json(connection_, "200 OK", chat_template_json(response));
}

void AnthropicMessages::fail(std::string_view type, const char* message) {
    if (start(true) && close_block() &&
        stream_.write(anthropic_error_body(type, message), "error") &&
        event("message_stop")) {
        stream_.end();
    }
}

}
