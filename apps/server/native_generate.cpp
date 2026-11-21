#include "sglang/decoder_constraint.h"
#include "sglang/server/native_generate.h"
#include "sglang/server/http.h"
#include "sglang/server/text_completion.h"
#include "sglang/server/text_stop.h"
#include "sglang/utf8.h"

#include <algorithm>
#include <charconv>
#include <iostream>
#include <iterator>
#include <random>
#include <set>

namespace sglang::server {
namespace {

std::string_view finish_type(const DecoderResult& result, const DecoderOptions& options) {
    const auto& ids = result.output_token_ids;
    const auto budget = static_cast<std::size_t>(options.max_new_tokens);
    const bool matched_stop = !ids.empty() && options.stops_at(ids.back());
    if (result.finish_reason == DecoderFinishReason::constraint &&
        options.constraint && !ids.empty() && ids.size() <= budget) {
        return ids.size() == budget && !matched_stop ? "length" : "stop";
    }
    if (result.finish_reason == DecoderFinishReason::stop &&
        (matched_stop || result.matched_text_stop) &&
        ids.size() <= budget) {
        return "stop";
    }
    if (result.finish_reason == DecoderFinishReason::length && !matched_stop &&
        ids.size() == budget) {
        return "length";
    }
    throw std::logic_error("Decoder returned an inconsistent /generate result");
}

void append_logprob(std::string& output, const DecoderLogprob& score, const Tokenizer* tokenizer) {
    char number[32];
    const auto formatted = std::to_chars(number, number + sizeof(number), score.value);
    output += '[';
    output.append(number, formatted.ptr);
    output += ',' + std::to_string(score.token_id) + ',';
    if (tokenizer) {
        Utf8Stream utf8;
        auto text = utf8.push(tokenizer->piece(score.token_id, false));
        text += utf8.finish();
        json::append_quoted(output, text);
    } else {
        output += "null";
    }
    output += ']';
}

void append_logprobs(std::string& output, const BatchPrompt& request, const Tokenizer& tokenizer,
                     const DecoderResult& result) {
    const auto& scores = result.generated_logprobs;
    if (scores.size() != result.output_token_ids.size()) {
        throw std::logic_error("Generated logprob count differs from output token count");
    }
    const auto* pieces = request.return_text_in_logprobs ? &tokenizer : nullptr;
    output += R"(,"input_token_logprobs":[],"output_token_logprobs":[)";
    for (std::size_t index = 0; index < scores.size(); ++index) {
        if (index) output += ',';
        append_logprob(output, scores[index]->selected, pieces);
    }
    output += R"(],"output_token_logprobs_length":)" + std::to_string(scores.size());
    if (request.options.generated_top_logprobs == 0) {
        return;
    }
    output += R"(,"input_top_logprobs":[],"output_top_logprobs":[)";
    for (std::size_t index = 0; index < scores.size(); ++index) {
        if (index) output += ',';
        output += '[';
        const auto& alternatives = scores[index]->alternatives;
        for (std::size_t rank = 0; rank < alternatives.size(); ++rank) {
            if (rank) output += ',';
            append_logprob(output, alternatives[rank], pieces);
        }
        output += ']';
    }
    output += ']';
}

class GenerateResponse {
public:
    GenerateResponse(const BatchPrompt& request, const Tokenizer& tokenizer)
        : request_(request), tokenizer_(tokenizer) {
        if (request_.text_stops) {
            stop_buffer_.emplace(request_.text_stops->stops);
        }
    }

    void token(int id);
    bool stream_token(EventStream& stream, int id);
    std::string finish(const DecoderResult& result);
    std::string frame(std::optional<std::size_t> index, const DecoderResult& result, bool terminal);

private:
    std::string body(const std::vector<int>& ids, const DecoderResult* result,
                     bool terminal, bool include_cached = true) const;

    const BatchPrompt& request_;
    const Tokenizer& tokenizer_;
    Utf8Stream utf8_;
    std::optional<TextStopBuffer> stop_buffer_;
    std::string text_;
    std::vector<int> streamed_ids_;
    std::size_t decoded_tokens_ = 0;
};

std::string GenerateResponse::body(const std::vector<int>& ids,
                                   const DecoderResult* result, bool terminal,
                                   bool include_cached) const {
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
            output += R"(,"matched":)";
            if (result->matched_text_stop) {
                const auto& stops = request_.text_stops->stops.patterns;
                json::append_quoted(output, stops[*result->matched_text_stop].text);
            } else {
                output += std::to_string(ids.back());
            }
        } else {
            output += R"(,"length":)" + std::to_string(ids.size());
        }
        output += '}';
    } else {
        output += "null";
    }
    if (result && include_cached) {
        output += R"(,"cached_tokens":)" + std::to_string(result->cached_prompt_tokens);
    }
    if (result && request_.options.generated_top_logprobs >= 0) {
        append_logprobs(output, request_, tokenizer_, *result);
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

std::string GenerateResponse::frame(std::optional<std::size_t> index,
                                    const DecoderResult& result, bool terminal) {
    const auto& ids = result.output_token_ids;
    while (decoded_tokens_ < ids.size()) {
        const int id = ids[decoded_tokens_++];
        if (!stop_buffer_) {
            token(id);
        } else if (result.matched_text_stop || !request_.options.stops_at(id)) {
            stop_buffer_->push(utf8_.push(tokenizer_.piece(id, true)));
        }
    }
    if (stop_buffer_) {
        if (terminal) {
            stop_buffer_->push(utf8_.finish());
        }
        text_ += stop_buffer_->take(terminal, result.matched_text_stop);
    }
    auto output = terminal ? finish(result) : body(ids, &result, false, index.has_value());
    if (index) {
        output.pop_back();
        output += R"(,"index":)" + std::to_string(*index) + '}';
    }
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
        if (index != 0) {
            output += ',';
        }
        output += response.frame(std::nullopt, results[index], true);
    }
    output += ']';
    write_json(connection, "200 OK", output);
}

void complete_batch(const Socket& connection, BatchService& service, int slot,
                    const CompletionTokenizer& tokenizer, const std::vector<BatchPrompt>& items,
                    bool indexed) {
    const auto& pieces = tokenizer.get();
    const BatchCallbacks callbacks{[&] { return disconnected(connection); }, {}, {}, {}};
    if (const auto results = service.generate_batch(slot, items, callbacks)) {
        if (indexed) {
            write_batch(connection, pieces, items, *results);
        } else if (!disconnected(connection)) {
            GenerateResponse response(items.front(), pieces);
            write_json(connection, "200 OK", response.frame(std::nullopt, results->front(), true));
        }
    }
}

void stream_batch(const Socket& connection, BatchService& service, int slot,
                  const CompletionTokenizer& tokenizer, const std::vector<BatchPrompt>& items,
                  bool indexed) {
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
            const auto frame_index = indexed ? std::optional{index} : std::nullopt;
            return stream.write(responses[index].frame(frame_index, result, terminal));
        },
        !indexed,
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
    if (!input.stream && input.indexed()) {
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
        const auto output = response.frame(input.indexed() ? std::optional{index} : std::nullopt,
                                           result, true);
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
    const bool scored = std::any_of(items.begin(), items.end(), [](const auto& item) {
        return item.options.generated_top_logprobs >= 0;
    });
    if (scored && !service.supports_generated_logprobs()) {
        throw std::invalid_argument("Generated logprobs require server MTP disabled");
    }
    for (std::size_t index = 0; index < items.size(); ++index) {
        items[index].request_id = ids[index].get();
    }
    if (!slot) {
        complete_aborted(connection, tokenizer.get(), input, items);
    } else if (!input.indexed() && !items.front().text_stops && !scored &&
               !items.front().options.constraint) {
        complete_single(connection, service, *slot, tokenizer, items.front(), input.stream);
    } else if (input.stream) {
        stream_batch(connection, service, *slot, tokenizer, items, input.indexed());
    } else {
        complete_batch(connection, service, *slot, tokenizer, items, input.indexed());
    }
}

std::vector<std::optional<std::string>> generate_ids(const GenerateInput& input) {
    std::vector<std::optional<std::string>> ids(input.item_count());
    const auto* rid = json::get(input.fields, "rid");
    if (!rid || rid->is_null()) {
        return ids;
    }
    if (const auto value = rid->string()) {
        for (std::size_t index = 0; index < ids.size(); ++index) {
            ids[index] = std::string(*value);
            if (input.indexed()) {
                *ids[index] += '_' + std::to_string(index);
            }
        }
        return ids;
    }
    const auto* list = rid->array();
    if (!input.indexed() || !list || list->size() != input.count) {
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

std::string sample_id(std::random_device& random) {
    constexpr std::string_view hex = "0123456789abcdef";
    std::string id(32, '0');
    for (std::size_t offset = 0; offset < id.size(); offset += 8) {
        auto bits = static_cast<std::uint32_t>(random());
        for (std::size_t digit = 0; digit < 8; ++digit) {
            id[offset + digit] = hex[bits & 15];
            bits >>= 4;
        }
    }
    id[12] = '4';
    id[16] = hex[8 + (hex.find(id[16]) & 3)];
    return id;
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
NativeGenerate::register_ids(const GenerateInput& input) {
    auto ids = generate_ids(input);
    auto registration = std::make_shared<Registration>(*this);
    registration->items.reserve(ids.size());
    const std::lock_guard lock(mutex_);
    std::set<std::string, std::less<>> normalized;
    std::optional<std::random_device> random;
    if (input.samples > 1) {
        for (const auto& id : ids) {
            if (!id) {
                continue;
            }
            const auto known = ids_.find(*id);
            if ((known != ids_.end() && !known->second->finished()) ||
                !normalized.insert(*id).second) {
                throw std::invalid_argument("Duplicate request ID detected: " + *id);
            }
        }
        random.emplace();
    }
    for (auto& id : ids) {
        if (random) {
            do {
                id = sample_id(*random);
            } while (ids_.count(*id) || normalized.count(*id));
        } else if (!id) {
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
    auto registration = register_ids(input);
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

void set_generate_constraint(const json::Object& sampling, const Tokenizer& tokenizer,
                             DecoderOptions& options) {
    std::string_view kind;
    std::string_view source;
    for (const std::string_view field : {"regex", "json_schema", "ebnf"}) {
        const auto* value = json::get(sampling, field);
        if (!value) {
            continue;
        }
        const auto text = value->string();
        if (!text) {
            const auto name = field == "regex" ? "regex" : "sampling_params." + std::string(field);
            throw std::invalid_argument(name + " must be a string");
        }
        if (text->empty()) {
            continue;
        }
        if (!kind.empty()) {
            throw std::invalid_argument("Use one of sampling_params.regex, "
                                        "sampling_params.json_schema and sampling_params.ebnf");
        }
        kind = field;
        source = *text;
    }
    if (kind.empty()) {
        return;
    }
    const auto name = kind == "json_schema" ? "JSON Schema" : kind;
    if (options.ignore_eos || json::get(sampling, "stop_token_ids")) {
        throw std::invalid_argument(std::string(name) +
                                    " requires model EOS stops without stop_token_ids overrides");
    }
    if (kind == "json_schema") {
        try {
            const auto parsed = json::parse(source, json::Syntax::ordered);
            options.constraint = TokenConstraint::compile_json_schema(tokenizer, parsed);
        } catch (const json::ParseError&) {
            throw std::invalid_argument("sampling_params.json_schema must contain valid JSON");
        }
    } else if (kind == "regex") {
        options.constraint = TokenConstraint::compile_regex(tokenizer, source);
    } else {
        options.constraint = TokenConstraint::compile(tokenizer, source);
    }
    if (options.max_new_tokens < options.constraint->minimum_tokens()) {
        const auto target = kind == "ebnf" ? "an ebnf choice" : "the " + std::string(name);
        throw std::invalid_argument("max_new_tokens cannot complete " + target);
    }
}

std::size_t generate_sample_count(const GenerateInput& input, const json::Array* params) {
    const auto read_count = [](const json::Value* params) -> std::size_t {
        const auto* object = params ? params->object() : nullptr;
        const auto* value = object ? json::get(*object, "n") : nullptr;
        if (!value || value->is_null()) {
            return 1;
        }
        const auto count = value->unsigned_integer();
        if (!count || *count == 0 || *count > decoder_batch_max_lanes) {
            throw std::invalid_argument("Expected an integer sampling_params.n in [1, 32]");
        }
        return static_cast<std::size_t>(*count);
    };
    const auto samples = read_count(params ? &params->front()
                                          : json::get(input.fields, "sampling_params"));
    if (params && std::any_of(params->begin(), params->end(), [&](const auto& item) {
            return read_count(&item) != samples;
        })) {
        throw std::invalid_argument("sampling_params.n must be the same for all batch items");
    }
    if (samples > decoder_batch_max_lanes / input.count) {
        throw std::invalid_argument("Expanded /generate batch must have at most 32 items");
    }
    if (samples > 1) {
        for (const auto* key : {"return_logprob", "logprob_start_len", "top_logprobs_num"}) {
            const auto* value = json::get(input.fields, key);
            if (value && value->array()) {
                throw std::invalid_argument(std::string(key) + " must be scalar when n > 1");
            }
        }
    }
    return samples;
}

}
