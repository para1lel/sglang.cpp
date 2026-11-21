#include "sglang/server/native_generate.h"

#include "sglang/server/http.h"
#include "sglang/server/text_completion.h"
#include "sglang/utf8.h"

#include <algorithm>
#include <iostream>
#include <iterator>

namespace sglang::server {
namespace {

std::string_view finish_type(const DecoderResult& result, const DecoderOptions& options) {
    const auto& ids = result.output_token_ids;
    const auto budget = static_cast<std::size_t>(options.max_new_tokens);
    const bool matched_stop = !ids.empty() && options.stops_at(ids.back());
    if (result.finish_reason == DecoderFinishReason::stop && matched_stop &&
        ids.size() <= budget) {
        return "stop";
    }
    if (result.finish_reason == DecoderFinishReason::length && !matched_stop &&
        ids.size() == budget) {
        return "length";
    }
    throw std::logic_error("Decoder returned an inconsistent /generate result");
}

class GenerateResponse {
public:
    GenerateResponse(const BatchPrompt& request, const Tokenizer& tokenizer)
        : request_(request), tokenizer_(tokenizer) {}

    void token(int id);
    bool stream_token(EventStream& stream, int id);
    std::string finish(const DecoderResult& result);
    std::string indexed(std::size_t index, const DecoderResult& result, bool terminal);

private:
    std::string body(const std::vector<int>& ids, const DecoderResult* result,
                     bool terminal) const;

    const BatchPrompt& request_;
    const Tokenizer& tokenizer_;
    Utf8Stream utf8_;
    std::string text_;
    std::vector<int> streamed_ids_;
    std::size_t decoded_tokens_ = 0;
};

std::string GenerateResponse::body(const std::vector<int>& ids,
                                   const DecoderResult* result, bool terminal) const {
    std::string output = R"({"text":)";
    json::append_quoted(output, text_);
    output += R"(,"output_ids":)";
    append_ids(output, ids.data(), ids.size());
    output += R"(,"meta_info":{"id":)";
    json::append_quoted(output, request_.request_id->value);
    output += R"(,"prompt_tokens":)" + std::to_string(request_.prompt.size());
    output += R"(,"completion_tokens":)" + std::to_string(ids.size());
    output += R"(,"finish_reason":)";
    if (terminal && result->finish_reason == DecoderFinishReason::aborted) {
        output += request_.request_id->aborted_waiting()
            ? R"({"type":"abort","message":"Abort in waiting queue"})"
            : R"({"type":"abort","message":"Aborted","status_code":null,"err_type":null})";
    } else if (terminal) {
        const auto type = finish_type(*result, request_.options);
        output += R"({"type":)";
        json::append_quoted(output, type);
        if (type == "stop") {
            output += R"(,"matched":)" + std::to_string(ids.back());
        } else {
            output += R"(,"length":)" + std::to_string(ids.size());
        }
        output += '}';
    } else {
        output += "null";
    }
    if (result) {
        output += R"(,"cached_tokens":)" + std::to_string(result->cached_prompt_tokens);
    }
    return output + "}}";
}

void GenerateResponse::token(int id) {
    if (!request_.options.stops_at(id)) {
        text_ += utf8_.push(tokenizer_.piece(id, true));
    }
}

bool GenerateResponse::stream_token(EventStream& stream, int id) {
    token(id);
    if (request_.options.stops_at(id)) {
        return true;
    }
    streamed_ids_.push_back(id);
    return streamed_ids_.size() == static_cast<std::size_t>(request_.options.max_new_tokens) ||
        (stream.start() && stream.write(body(streamed_ids_, nullptr, false)));
}

std::string GenerateResponse::finish(const DecoderResult& result) {
    text_ += utf8_.finish();
    return body(result.output_token_ids, &result, true);
}

std::string GenerateResponse::indexed(std::size_t index, const DecoderResult& result,
                                      bool terminal) {
    const auto& ids = result.output_token_ids;
    while (decoded_tokens_ < ids.size()) {
        token(ids[decoded_tokens_++]);
    }
    auto output = terminal ? finish(result) : body(ids, &result, false);
    output.pop_back();
    output += R"(,"index":)" + std::to_string(index) + '}';
    return output;
}

void write_stream_error(EventStream& stream, const std::exception& error) {
    std::cerr << "/generate failed: " << error.what() << '\n';
    if (stream.write(R"({"error":{"message":"Native decoder failed",)"
                     R"("type":"server_error","code":500,"retryable":false}})")) {
        stream.end();
    }
}

void complete_single(const Socket& connection, BatchService& service, int slot,
                     const CompletionTokenizer& tokenizer, const BatchPrompt& request,
                     bool streaming) {
    GenerateResponse response(request, tokenizer.get());
    EventStream stream(connection);
    const DecoderCallbacks callbacks{
        [&](int token) {
            if (streaming) {
                return response.stream_token(stream, token);
            }
            response.token(token);
            return true;
        },
        [&] { return disconnected(connection); },
        [&] { return !streaming || stream.start(); },
        [&] { return stream.poll(); },
        streaming,
    };
    try {
        const auto result = service.generate(slot, request.prompt, request.options, callbacks,
                                             {}, 0, request.request_id);
        if (result.finish_reason == DecoderFinishReason::cancelled || disconnected(connection)) {
            return;
        }
        const auto output = response.finish(result);
        if (!streaming) {
            write_json(connection, "200 OK", output);
        } else if (stream.start() && stream.write(output) && stream.write("[DONE]")) {
            stream.end();
        }
    } catch (const std::exception& error) {
        if (!stream.started()) {
            throw;
        }
        write_stream_error(stream, error);
    }
}

void write_batch(const Socket& connection, const Tokenizer& tokenizer,
                 const std::vector<BatchPrompt>& items, const std::vector<DecoderResult>& results) {
    std::string output = "[";
    for (std::size_t index = 0; index < items.size(); ++index) {
        if (disconnected(connection)) {
            return;
        }
        GenerateResponse response(items[index], tokenizer);
        const auto& result = results[index];
        for (int token : result.output_token_ids) {
            response.token(token);
        }
        if (index != 0) {
            output += ',';
        }
        output += response.finish(result);
    }
    output += ']';
    write_json(connection, "200 OK", output);
}

void complete_batch(const Socket& connection, BatchService& service, int slot,
                    const CompletionTokenizer& tokenizer, const std::vector<BatchPrompt>& items) {
    const auto& pieces = tokenizer.get();
    const BatchCallbacks callbacks{[&] { return disconnected(connection); }, {}, {}, {}};
    if (const auto results = service.generate_batch(slot, items, callbacks)) {
        write_batch(connection, pieces, items, *results);
    }
}

void stream_batch(const Socket& connection, BatchService& service, int slot,
                  const CompletionTokenizer& tokenizer, const std::vector<BatchPrompt>& items) {
    const auto& pieces = tokenizer.get();
    EventStream stream(connection);
    std::vector<GenerateResponse> responses;
    responses.reserve(items.size());
    for (const auto& item : items) {
        responses.emplace_back(item, pieces);
    }
    const BatchCallbacks callbacks{
        [&] { return disconnected(connection); },
        [&] { return stream.start(); },
        [&] { return stream.poll(); },
        [&](std::size_t index, const DecoderResult& result, bool terminal) {
            return stream.write(responses[index].indexed(index, result, terminal));
        },
    };
    try {
        if (service.generate_batch(slot, items, callbacks) && stream.write("[DONE]")) {
            stream.end();
        }
    } catch (const std::exception& error) {
        if (!stream.started()) {
            throw;
        }
        write_stream_error(stream, error);
    }
}

void complete_aborted(const Socket& connection, const Tokenizer& tokenizer,
                      const GenerateInput& input, const std::vector<BatchPrompt>& items) {
    DecoderResult result;
    result.finish_reason = DecoderFinishReason::aborted;
    for (const auto& item : items) {
        item.request_id->finish_abort();
    }
    if (!input.stream && input.batch) {
        write_batch(connection, tokenizer, items, std::vector<DecoderResult>(items.size(), result));
        return;
    }
    EventStream stream(connection);
    for (std::size_t index = 0; index < items.size(); ++index) {
        GenerateResponse response(items[index], tokenizer);
        if (!input.stream) {
            write_json(connection, "200 OK", response.finish(result));
            return;
        }
        const auto output = input.batch ? response.indexed(index, result, true)
                                        : response.finish(result);
        if (!stream.start() || !stream.write(output)) {
            return;
        }
    }
    if (stream.write("[DONE]")) {
        stream.end();
    }
}

void complete_generate(const Socket& connection, BatchService& service, std::optional<int> slot,
                       const CompletionTokenizer& tokenizer, const GenerateInput& input,
                       const std::vector<std::unique_ptr<BatchRequestId>>& ids) {
    if (disconnected(connection)) {
        return;
    }
    auto items = parse_generate_prompts(input, tokenizer);
    for (std::size_t index = 0; index < items.size(); ++index) {
        items[index].request_id = ids[index].get();
    }
    if (!slot) {
        complete_aborted(connection, tokenizer.get(), input, items);
    } else if (input.batch && input.stream) {
        stream_batch(connection, service, *slot, tokenizer, items);
    } else if (input.batch) {
        complete_batch(connection, service, *slot, tokenizer, items);
    } else {
        complete_single(connection, service, *slot, tokenizer, items.front(), input.stream);
    }
}

std::vector<std::optional<std::string>> generate_ids(const GenerateInput& input) {
    std::vector<std::optional<std::string>> ids(input.count);
    const auto* rid = json::get(input.fields, "rid");
    if (!rid || rid->is_null()) {
        return ids;
    }
    if (const auto value = rid->string()) {
        for (std::size_t index = 0; index < input.count; ++index) {
            ids[index] = std::string(*value);
            if (input.batch) {
                *ids[index] += '_' + std::to_string(index);
            }
        }
        return ids;
    }
    const auto* list = rid->array();
    if (!input.batch || !list || list->size() != input.count) {
        throw std::invalid_argument("Expected a rid string or one rid string per batch item");
    }
    for (std::size_t index = 0; index < input.count; ++index) {
        const auto value = (*list)[index].string();
        if (!value) {
            throw std::invalid_argument("Expected a rid string for every batch item");
        }
        ids[index] = std::string(*value);
    }
    return ids;
}

}

struct NativeGenerate::Registration {
    explicit Registration(NativeGenerate& owner) : owner(owner) {}
    ~Registration() {
        const std::lock_guard lock(owner.mutex_);
        for (const auto& item : items) {
            const auto found = owner.ids_.find(item->value);
            if (found != owner.ids_.end() && found->second == item.get()) {
                owner.ids_.erase(found);
            }
        }
    }

    NativeGenerate& owner;
    std::vector<std::unique_ptr<BatchRequestId>> items;
};

std::shared_ptr<NativeGenerate::Registration>
NativeGenerate::register_ids(std::vector<std::optional<std::string>> ids) {
    auto registration = std::make_shared<Registration>(*this);
    registration->items.reserve(ids.size());
    const std::lock_guard lock(mutex_);
    for (auto& id : ids) {
        if (!id) {
            do {
                id = "gen-" + std::to_string(next_id_++);
            } while (ids_.count(*id));
        }
        const auto known = ids_.find(*id);
        if (known != ids_.end() && !known->second->finished()) {
            throw std::invalid_argument("Duplicate request ID detected: " + *id);
        }
        auto& item = registration->items.emplace_back(
            std::make_unique<BatchRequestId>(std::move(*id)));
        ids_.insert_or_assign(item->value, item.get());
    }
    return registration;
}

QueuedGenerate NativeGenerate::prepare(std::string_view body) {
    auto input = parse_generate_input(body);
    auto registration = register_ids(generate_ids(input));
    return {
        [this, input = std::move(input), registration](
            const Socket& connection, std::optional<int> slot) {
            complete_generate(connection, service_, slot, tokenizer_, input, registration->items);
        },
        [registration] {
            return std::all_of(registration->items.begin(), registration->items.end(),
                               [](const auto& item) { return item->aborted_waiting(); });
        },
    };
}

void NativeGenerate::abort(const Socket& connection, std::string_view body) {
    json::Value input;
    try {
        input = json::parse(body);
    } catch (const json::ParseError&) {
        throw std::invalid_argument("Expected a JSON object for /abort_request");
    }
    const auto* object = input.object();
    if (!object) {
        throw std::invalid_argument("Expected a JSON object for /abort_request");
    }
    const auto* rid = json::get(*object, "rid");
    const auto* all = json::get(*object, "abort_all");
    const auto name = rid ? rid->string() : std::nullopt;
    const auto abort_all = all ? all->boolean() : std::optional<bool>(false);
    if ((rid && !rid->is_null() && !name) || !abort_all) {
        throw std::invalid_argument("Expected an optional rid string and an abort_all boolean");
    }
    bool changed = false;
    {
        const std::lock_guard lock(mutex_);
        const auto prefix = name.value_or("");
        if (*abort_all) {
            for (const auto& [id, request] : ids_) {
                changed = request->abort() || changed;
            }
        } else if (!prefix.empty()) {
            const auto exact = ids_.find(prefix);
            if (exact != ids_.end() && exact->second->abort()) {
                changed = true;
                for (auto item = std::next(exact); item != ids_.end(); ++item) {
                    if (std::string_view(item->first).substr(0, prefix.size()) != prefix) {
                        break;
                    }
                    item->second->abort(true);
                }
            }
        }
    }
    if (changed) {
        service_.notify_abort();
    }
    write_json(connection, "200 OK", "");
}

}
