#include "sglang/server/completions.h"

#include "sglang/decoder.h"
#include "sglang/server/http.h"

#include <llvm/Support/Error.h>
#include <llvm/Support/JSON.h>
#include <llvm/Support/raw_ostream.h>

#include <cstdint>
#include <iostream>
#include <limits>

namespace sglang::server {
namespace {

namespace json = llvm::json;

struct CompletionRequest {
    std::vector<int> prompt;
    DecoderOptions options;
};

std::optional<std::uint64_t> unsigned_integer(const json::Object& object, llvm::StringRef key) {
    const auto* value = object.get(key);
    return value ? value->getAsUINT64() : std::nullopt;
}

CompletionRequest parse_completion(std::string_view body) {
    auto parsed = json::parse(llvm::StringRef(body.data(), body.size()));
    if (!parsed) {
        llvm::consumeError(parsed.takeError());
        throw HttpError("400 Bad Request", "Invalid JSON request body");
    }
    const auto* object = parsed->getAsObject();
    if (!object || object->size() != 11 || object->getString("model") != "benchmark" ||
        object->getNumber("temperature") != 0.0 || unsigned_integer(*object, "seed") != 0 ||
        object->getBoolean("ignore_eos") != true || object->getBoolean("stream") != true ||
        object->getBoolean("return_token_ids") != true ||
        unsigned_integer(*object, "stream_interval") != 1 ||
        object->getBoolean("add_special_tokens") != false) {
        throw HttpError("400 Bad Request", "Expected the frozen benchmark streaming options");
    }
    const auto* stream_options = object->getObject("stream_options");
    if (!stream_options || stream_options->size() != 1 ||
        stream_options->getBoolean("include_usage") != true) {
        throw HttpError("400 Bad Request", "Expected stream_options.include_usage=true");
    }
    const auto output_budget = unsigned_integer(*object, "max_tokens");
    if (!output_budget || *output_budget > std::numeric_limits<int>::max()) {
        throw HttpError("400 Bad Request", "Expected a nonnegative integer max_tokens");
    }
    const auto* prompt = object->getArray("prompt");
    if (!prompt || prompt->empty()) {
        throw HttpError("400 Bad Request", "Expected a nonempty prompt token ID array");
    }
    CompletionRequest request;
    request.options.max_new_tokens = static_cast<int>(*output_budget);
    request.prompt.reserve(prompt->size());
    for (const auto& value : *prompt) {
        const auto id = value.getAsUINT64();
        if (!id || *id > std::numeric_limits<int>::max()) {
            throw HttpError("400 Bad Request", "Expected nonnegative integer prompt token IDs");
        }
        request.prompt.push_back(static_cast<int>(*id));
    }
    return request;
}

template <typename Encode>
std::string encode_json(Encode encode) {
    std::string output;
    llvm::raw_string_ostream stream(output);
    json::OStream writer(stream);
    writer.object([&] { encode(writer); });
    return output;
}

std::string choice_event(std::optional<int> token, const std::vector<int>* prompt, bool finished) {
    return encode_json([&](json::OStream& writer) {
        writer.attributeArray("choices", [&] {
            writer.object([&] {
                writer.attribute("index", 0);
                writer.attributeArray("token_ids", [&] {
                    if (token) {
                        writer.value(*token);
                    }
                });
                if (prompt) {
                    writer.attributeArray("prompt_token_ids", [&] {
                        for (const int id : *prompt) {
                            writer.value(id);
                        }
                    });
                }
                writer.attribute("finish_reason", finished ? json::Value("length") : nullptr);
            });
        });
    });
}

std::string usage_event(std::size_t prompt_tokens, std::size_t completion_tokens) {
    return encode_json([&](json::OStream& writer) {
        writer.attribute("choices", json::Array{});
        writer.attributeObject("usage", [&] {
            writer.attribute("prompt_tokens", static_cast<std::int64_t>(prompt_tokens));
            writer.attribute("completion_tokens", static_cast<std::int64_t>(completion_tokens));
            writer.attributeObject("prompt_tokens_details", [&] {
                writer.attribute("cached_tokens", 0);
            });
        });
    });
}

std::string error_body(const char* message) {
    return encode_json([&](json::OStream& writer) {
        writer.attributeObject("error", [&] { writer.attribute("message", message); });
    });
}

bool begin_stream(const Socket& connection, bool& streaming) {
    if (!streaming) {
        streaming = start_events(connection);
    }
    return streaming;
}

void finish_stream(const Socket& connection, const CompletionRequest& request,
                   const DecoderResult& result, bool& streaming) {
    const auto output_count = result.output_token_ids.size();
    if (result.finish_reason != DecoderFinishReason::length ||
        output_count != static_cast<std::size_t>(request.options.max_new_tokens)) {
        throw std::logic_error("Decoder did not complete the requested output budget");
    }
    const auto* prompt = streaming ? nullptr : &request.prompt;
    if (begin_stream(connection, streaming) &&
        write_event(connection, choice_event(std::nullopt, prompt, true)) &&
        write_event(connection, usage_event(request.prompt.size(), output_count)) &&
        write_event(connection, "[DONE]")) {
        end_events(connection);
    }
}

}

void write_error(const Socket& connection, std::string_view status, const char* message) {
    write_json(connection, status, error_body(message));
}

void complete(const Socket& connection, const Decoder& decoder, std::string_view body) {
    const auto request = parse_completion(body);
    bool streaming = false;
    const DecoderCallbacks callbacks{
        [&](int token) {
            const auto* prompt = streaming ? nullptr : &request.prompt;
            return begin_stream(connection, streaming) &&
                   write_event(connection, choice_event(token, prompt, false));
        },
        [&] { return disconnected(connection); },
    };

    try {
        const auto result = decoder.generate(request.prompt, request.options, callbacks);
        if (result.finish_reason != DecoderFinishReason::cancelled) {
            finish_stream(connection, request, result, streaming);
        }
    } catch (const std::exception& error) {
        if (!streaming) {
            throw;
        }
        std::cerr << "Completion failed: " << error.what() << '\n';
        if (write_event(connection, error_body("Native decoder failed"))) {
            end_events(connection);
        }
    }
}

}
