#include "sglang/server/native_generate.h"

#include "sglang/server/http.h"
#include "sglang/server/text_completion.h"
#include "sglang/utf8.h"

#include <atomic>
#include <iostream>

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
    GenerateResponse(const Socket& connection, const BatchPrompt& request,
                     const Tokenizer& tokenizer, bool streaming)
        : stream(connection), request_(request), tokenizer_(tokenizer), id_(next_id()),
          streaming_(streaming) {}

    bool token(int id);
    std::string finish(const DecoderResult& result);

    EventStream stream;

private:
    static std::string next_id() {
        static std::atomic<std::uint64_t> id{1};
        return "gen-" + std::to_string(id.fetch_add(1, std::memory_order_relaxed));
    }

    std::string body(const std::vector<int>& ids, const DecoderResult* result) const;

    const BatchPrompt& request_;
    const Tokenizer& tokenizer_;
    const std::string id_;
    const bool streaming_;
    Utf8Stream utf8_;
    std::string text_;
    std::vector<int> streamed_ids_;
};

std::string GenerateResponse::body(const std::vector<int>& ids,
                                   const DecoderResult* result) const {
    std::string output = R"({"text":)";
    json::append_quoted(output, text_);
    output += R"(,"output_ids":)";
    append_ids(output, ids.data(), ids.size());
    output += R"(,"meta_info":{"id":)";
    json::append_quoted(output, id_);
    output += R"(,"prompt_tokens":)" + std::to_string(request_.prompt.size());
    output += R"(,"completion_tokens":)" + std::to_string(ids.size());
    output += R"(,"finish_reason":)";
    if (!result) {
        return output + "null}}";
    }
    const auto type = finish_type(*result, request_.options);
    output += R"({"type":)";
    json::append_quoted(output, type);
    if (type == "stop") {
        output += R"(,"matched":)" + std::to_string(ids.back());
    } else {
        output += R"(,"length":)" + std::to_string(ids.size());
    }
    output += R"(},"cached_tokens":)" + std::to_string(result->cached_prompt_tokens);
    return output + "}}";
}

bool GenerateResponse::token(int id) {
    const bool stop = request_.options.stops_at(id);
    if (!stop) {
        text_ += utf8_.push(tokenizer_.piece(id, true));
    }
    if (!streaming_ || stop) {
        return true;
    }
    streamed_ids_.push_back(id);
    return streamed_ids_.size() == static_cast<std::size_t>(request_.options.max_new_tokens) ||
        (stream.start() && stream.write(body(streamed_ids_, nullptr)));
}

std::string GenerateResponse::finish(const DecoderResult& result) {
    text_ += utf8_.finish();
    return body(result.output_token_ids, &result);
}

void complete_single(const Socket& connection, BatchService& service, int slot,
                     const CompletionTokenizer& tokenizer, const BatchPrompt& request,
                     bool stream) {
    GenerateResponse response(connection, request, tokenizer.get(), stream);
    const DecoderCallbacks callbacks{
        [&](int token) { return response.token(token); },
        [&] { return disconnected(connection); },
        [&] { return !stream || response.stream.start(); },
        [&] { return response.stream.poll(); },
    };
    try {
        const auto result = service.generate(slot, request.prompt, request.options, callbacks);
        if (result.finish_reason == DecoderFinishReason::cancelled || disconnected(connection)) {
            return;
        }
        const auto output = response.finish(result);
        if (!stream) {
            write_json(connection, "200 OK", output);
        } else if (response.stream.start() && response.stream.write(output) &&
                   response.stream.write("[DONE]")) {
            response.stream.end();
        }
    } catch (const std::exception& error) {
        if (!response.stream.started()) {
            throw;
        }
        std::cerr << "/generate failed: " << error.what() << '\n';
        if (response.stream.write(R"({"error":{"message":"Native decoder failed",)"
                                  R"("type":"server_error","code":500,"retryable":false}})")) {
            response.stream.end();
        }
    }
}

void complete_batch(const Socket& connection, BatchService& service, int slot,
                    const CompletionTokenizer& tokenizer, const std::vector<BatchPrompt>& items) {
    const auto& pieces = tokenizer.get();
    const auto results = service.generate_batch(slot, items,
        [&] { return disconnected(connection); });
    if (!results) {
        return;
    }
    std::string output = "[";
    for (std::size_t index = 0; index < items.size(); ++index) {
        if (disconnected(connection)) {
            return;
        }
        GenerateResponse response(connection, items[index], pieces, false);
        const auto& result = (*results)[index];
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

}

void complete_generate(const Socket& connection, BatchService& service, int slot,
                       const CompletionTokenizer& tokenizer, std::string_view body) {
    if (disconnected(connection)) {
        return;
    }
    const auto request = parse_generate_requests(body, tokenizer);
    if (request.batch) {
        complete_batch(connection, service, slot, tokenizer, request.items);
    } else {
        complete_single(connection, service, slot, tokenizer, request.items.front(),
                        request.stream);
    }
}

}
