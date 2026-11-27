#pragma once

#include "sglang/server/http.h"
#include "sglang/server/text_completion.h"

#include <atomic>
#include <list>
#include <map>

namespace sglang::server {

class ResponseStore {
public:
    static constexpr std::size_t record_bytes = 256 * 1024;

    struct Entry {
        std::string id;
        std::string envelope;
        std::string history;
        bool active;
        std::atomic<bool> cancelled{false};
    };
    using Handle = std::shared_ptr<Entry>;

    explicit ResponseStore(bool enabled);
    bool enabled() const;
    Handle insert(const json::OrderedObject& response, std::string envelope,
                  std::string history);
    bool update(const Handle& entry, const json::OrderedObject& response,
                std::optional<std::string> history = std::nullopt);
    bool serve_state(const Socket& connection, const HttpRequest& request);
    json::Array history(std::string_view id);

private:
    Handle lookup(std::string_view id) const;

    bool enabled_;
    std::mutex mutex_;
    std::list<Handle> entries_;
};

struct ResponsesRequest {
    json::OrderedObject chat;
    json::OrderedObject fields;
    json::Array history;
    ResponseStore* store = nullptr;
    bool background = false;
    bool summary = false;
    std::optional<std::string> id;
    int top_logprobs = -1;
};

void responses_error(const Socket& connection, unsigned code, std::string_view message,
                     const char* parameter = nullptr);
void complete_responses(const Socket& connection, BatchService& service, int slot,
                        const CompletionTokenizer& tokenizer, std::string_view body,
                        ResponseStore& store, VisionCompletion* vision = nullptr);

void configure_responses_tool(const json::OrderedObject& fields, json::OrderedObject& chat);
bool append_responses_tool_history(const json::OrderedObject& item, json::Array& messages,
                                   std::map<std::string, bool, std::less<>>& calls);

class ResponsesOutput {
public:
    ResponsesOutput(const Socket& connection, EventStream& stream,
                    const ResponsesRequest& request, const TextRequest& text);
    bool queue();
    bool start();
    bool cancelled() const;
    bool accept(const ChatOutputEvent& delta);
    void token(std::string_view piece);
    void finish(const char* reason, std::size_t count, const DecoderResult& result,
                const Tokenizer& tokenizer);
    void fail();

private:
    struct Item {
        std::string id;
        std::string text;
        bool reasoning;
        json::Value logprobs{};
    };

    json::OrderedObject envelope(const char* status, json::Value usage = json::Value(),
                                 json::Value error = json::Value()) const;
    json::OrderedObject part(const Item& item, bool summary = false) const;
    json::OrderedObject item_json(const Item& item, bool ready) const;
    bool event(std::string_view type, json::OrderedObject fields);
    bool lifecycle(std::string_view type, const char* status,
                   json::Value usage = json::Value(), json::Value error = json::Value());
    bool open_item(bool reasoning);
    bool close_item();
    bool accept_tool(const ChatOutputEvent& delta);
    void fail(const char* code, const char* message, unsigned status);
    json::OrderedObject part_fields() const;

    const Socket& connection_;
    EventStream& stream_;
    const ResponsesRequest& request_;
    const TextRequest& text_;
    const std::string id_;
    const std::int64_t created_;
    ResponseStore::Handle entry_;
    json::Array output_;
    std::optional<Item> item_;
    int sequence_ = 0;
    std::size_t reasoning_tokens_ = 0;
    bool reasoning_;
    bool first_token_ = true;
};

std::string_view responses_method(std::string_view path);

}
