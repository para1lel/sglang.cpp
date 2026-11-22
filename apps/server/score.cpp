#include "sglang/server/score.h"

#include "sglang/json.h"
#include "sglang/server/http.h"
#include "sglang/tokenizer/tokenizer.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <numeric>
#include <set>

namespace sglang::server {
namespace {

using json::Array;
using json::Object;
using json::Value;

constexpr std::size_t max_score_items = 1024;
constexpr std::size_t max_score_prompt_tokens = 4 * checkpoint_context;
constexpr std::size_t max_score_text_bytes = 8 * 1024 * 1024;
constexpr std::size_t max_score_response_values = checkpoint_context;

bool flag(const Object& fields, std::string_view name) {
    const auto* field = json::get(fields, name);
    if (!field) return false;
    if (const auto value = field->boolean()) return *value;
    throw ScoreValidationError(std::string(name) + " must be a boolean");
}

const Value* optional(const Object& fields, std::string_view name) {
    const auto* field = json::get(fields, name);
    return field && !field->is_null() ? field : nullptr;
}

bool integer(const Value& value) {
    return value.signed_integer() || value.unsigned_integer() || value.wide_integer();
}

std::string integer_text(const Value& value) {
    if (const auto number = value.signed_integer()) return std::to_string(*number);
    if (const auto number = value.unsigned_integer()) return std::to_string(*number);
    if (const auto* number = value.wide_integer()) return number->digits;
    throw ScoreValidationError("Expected an integer token ID");
}

bool token_list(const Value& value) {
    const auto* list = value.array();
    return list && std::all_of(list->begin(), list->end(), integer);
}

bool token_lists(const Array& values) {
    return std::all_of(values.begin(), values.end(), token_list);
}

std::shared_ptr<const std::vector<int>> label_ids(const Array& values, bool per_item) {
    if (values.empty()) {
        throw std::invalid_argument(per_item ? "each item must have a nonempty candidate list"
                                             : "label_token_ids must be a nonempty list");
    }
    std::set<int> unique;
    std::vector<int> labels;
    labels.reserve(values.size());
    for (const auto& value : values) {
        if (integer_text(value).front() == '-') {
            throw std::invalid_argument("label_token_ids must contain nonnegative integers");
        }
        const auto id = value.unsigned_integer();
        if (!id || *id >= checkpoint_base_vocab_size) {
            throw std::invalid_argument("Token ID " + integer_text(value) +
                " is out of vocabulary (vocab size: " + std::to_string(checkpoint_base_vocab_size) +
                ")");
        }
        labels.push_back(static_cast<int>(*id));
        if (!unique.insert(labels.back()).second) {
            throw std::invalid_argument("label_token_ids must not contain duplicate tokens");
        }
    }
    return std::make_shared<const std::vector<int>>(std::move(labels));
}

struct ScoreItems {
    const Value* query;
    const Array* list;
    std::optional<std::string_view> single;
    std::size_t size;
    bool token_rows;
};

ScoreItems score_items(const Object& fields) {
    const auto* query = optional(fields, "query");
    if (query && !query->string() && !token_list(*query)) {
        throw ScoreValidationError("query must be text or a list of token IDs");
    }
    const auto* items = optional(fields, "items");
    if (!items) return {query, nullptr, std::nullopt, 0, false};
    if (const auto text = items->string()) {
        return {query, nullptr, text, text->empty() ? 0u : 1u, false};
    }
    if (const auto* list = items->array()) {
        const bool rows = !list->empty() && token_lists(*list);
        const bool texts = std::all_of(list->begin(), list->end(), [](const Value& value) {
            return value.string().has_value();
        });
        if (rows || texts) return {query, list, std::nullopt, list->size(), rows};
    }
    throw ScoreValidationError("items must be text, a list of texts, or token-ID rows");
}

bool token_prompts(const ScoreItems& items) {
    return items.token_rows && (!items.query || items.query->array());
}

std::size_t integer_bytes(const Value& value) {
    if (const auto* wide = value.wide_integer()) return wide->digits.size();
    char buffer[24];
    const auto end = buffer + sizeof(buffer);
    const auto signed_id = value.signed_integer();
    const auto written = signed_id ? std::to_chars(buffer, end, *signed_id)
                                   : std::to_chars(buffer, end, *value.unsigned_integer());
    return written.ptr - buffer;
}

void charge(std::size_t& remaining, std::size_t count, const char* message,
            std::size_t repetitions = 1) {
    if (!score_detail::consume(remaining, count, repetitions)) {
        throw std::invalid_argument(message);
    }
}

void check_label_budget(const Array& labels, bool per_item, std::size_t items, bool raw) {
    constexpr auto message = "Score request exceeds the returned label value budget";
    std::size_t remaining = max_score_response_values / (raw ? 2 : 1);
    if (per_item) {
        for (const auto& row : labels) charge(remaining, row.array()->size(), message);
    } else {
        charge(remaining, labels.size(), message, items);
    }
}

void check_prompt_budget(const ScoreItems& items) {
    constexpr auto token_message = "Score request exceeds the aggregate prompt token budget";
    constexpr auto text_message = "Score request exceeds the aggregate composed text budget";
    if (token_prompts(items)) {
        const auto query_tokens = items.query ? items.query->array()->size() : 0;
        if (query_tokens > checkpoint_context) {
            throw std::invalid_argument("Prompt exceeds the available token budget");
        }
        auto remaining = max_score_prompt_tokens;
        charge(remaining, query_tokens, token_message, items.size);
        for (const auto& row : *items.list) {
            const auto size = row.array()->size();
            if (size > checkpoint_context - query_tokens) {
                throw std::invalid_argument("Prompt exceeds the available token budget");
            }
            charge(remaining, size, token_message);
        }
        return;
    }
    if (items.query && !items.query->string()) {
        throw std::invalid_argument("Invalid combination of query/items types for score_request.");
    }
    auto remaining = max_score_text_bytes;
    if (items.query) charge(remaining, items.query->string()->size(), text_message, items.size);
    if (items.single) {
        charge(remaining, items.single->size(), text_message);
    } else if (!items.token_rows) {
        for (const auto& item : *items.list) {
            charge(remaining, item.string()->size(), text_message);
        }
    } else {
        for (const auto& row : *items.list) {
            const auto& tokens = *row.array();
            charge(remaining, std::max(std::size_t{1}, tokens.size()), text_message, 2);
            for (const auto& id : tokens) charge(remaining, integer_bytes(id), text_message);
        }
    }
}

std::vector<int> prompt_ids(const Array* query, const Array& item, bool item_first) {
    const auto count = (query ? query->size() : 0) + item.size();
    std::vector<int> prompt;
    prompt.reserve(count);
    const auto append = [&](const Array& tokens) {
        for (const auto& token : tokens) {
            const auto id = token.unsigned_integer();
            if (!id || *id >= model_vocab_size) {
                throw std::invalid_argument("Prompt must contain token IDs in [0, 248320)");
            }
            prompt.push_back(static_cast<int>(*id));
        }
    };
    if (item_first) append(item);
    if (query) append(*query);
    if (!item_first) append(item);
    return prompt;
}

std::string list_text(const Array& tokens) {
    std::string text = "[";
    for (const auto& token : tokens) {
        if (text.size() != 1) text += ", ";
        text += integer_text(token);
    }
    text += ']';
    return text;
}

std::vector<int> score_prompt(const ScoreItems& items, std::size_t index, bool item_first,
                              const Tokenizer& tokenizer, std::size_t remaining) {
    if (token_prompts(items)) {
        return prompt_ids(items.query ? items.query->array() : nullptr,
                          *(*items.list)[index].array(), item_first);
    }
    const auto query = items.query ? *items.query->string() : std::string_view{};
    const auto row_text = items.token_rows ? list_text(*(*items.list)[index].array()) : "";
    const auto item = items.single ? *items.single
        : items.token_rows ? std::string_view(row_text) : *(*items.list)[index].string();
    std::string text(item_first ? item : query);
    text += item_first ? query : item;
    try {
        const auto prompt = tokenizer.encode(text, std::min(checkpoint_context, remaining));
        return {prompt.begin(), prompt.end()};
    } catch (const std::length_error&) {
        if (remaining < checkpoint_context) {
            throw std::invalid_argument("Score request exceeds the aggregate prompt token budget");
        }
        throw;
    }
}

std::vector<double> probabilities(const std::vector<double>& logprobs,
                                  const ScoreRequest& request) {
    std::vector<double> values;
    values.reserve(logprobs.size());
    const double center = request.apply_softmax
        ? *std::max_element(logprobs.begin(), logprobs.end()) : 0;
    for (double logprob : logprobs) {
        values.push_back(std::exp((logprob - center) / request.temperature));
    }
    if (request.apply_softmax) {
        const double total = std::accumulate(values.begin(), values.end(), 0.0);
        for (auto& value : values) value /= total;
    }
    return values;
}

void append_row(std::string& output, const std::vector<double>& row) {
    output += '[';
    bool first = true;
    for (double value : row) {
        if (!std::exchange(first, false)) output += ',';
        char buffer[32];
        const auto converted = std::to_chars(buffer, buffer + sizeof(buffer), value);
        if (converted.ec != std::errc{}) throw std::logic_error("Cannot serialize score");
        output.append(buffer, converted.ptr);
    }
    output += ']';
}

}

ScoreRequest parse_score_request(std::string_view body, const Tokenizer& tokenizer) {
    const auto value = json::parse(body, json::Syntax::last_key_wins);
    const auto* fields = value.object();
    if (!fields) throw ScoreValidationError("Input should be a valid dictionary");
    const auto* model = json::get(*fields, "model");
    if (model && !model->string()) throw ScoreValidationError("model must be a string");
    const auto* temperature = json::get(*fields, "temperature");
    const auto scale = temperature ? temperature->number() : std::optional<double>{1};
    if (!scale || !std::isfinite(*scale) || *scale <= 0) {
        throw ScoreValidationError("temperature must be finite and greater than zero");
    }
    ScoreRequest request{{}, model ? std::string(*model->string()) : "default", *scale,
                         flag(*fields, "apply_softmax"), flag(*fields, "return_token_logprobs")};
    const bool item_first = flag(*fields, "item_first");
    const bool pooled = flag(*fields, "return_pooled_hidden_states");
    const auto items = score_items(*fields);
    const auto* labels = optional(*fields, "label_token_ids");
    const bool per_item = labels && labels->array() && !labels->array()->empty() &&
        token_lists(*labels->array());
    if (labels && !per_item && !token_list(*labels)) {
        throw ScoreValidationError("label_token_ids must be a list of token IDs or token-ID rows");
    }
    if (request.temperature != 1 && !request.apply_softmax) {
        throw std::invalid_argument("temperature requires apply_softmax=True");
    }
    if (!labels) {
        throw std::invalid_argument(
            "label_token_ids is required for generation (CausalLM) models.");
    }
    if (!items.list && !items.single) throw std::invalid_argument("items must be provided");
    if (!items.size) return request;
    if (optional(*fields, "score_extraction_token") || optional(*fields, "query_embed_overrides") ||
        optional(*fields, "item_embed_overrides") || optional(*fields, "rid")) {
        throw std::invalid_argument("Native score supports pointwise inputs without embedding "
                                    "overrides or a custom rid");
    }
    if (per_item && labels->array()->size() != items.size) {
        throw std::invalid_argument("label_token_ids must have one list per item");
    }
    if (items.size > max_score_items) {
        throw std::invalid_argument("Score request exceeds the item limit");
    }
    if (pooled) {
        throw std::invalid_argument("return_pooled_hidden_states is not supported for CausalLM "
                                    "models. It requires a model with a task-specific head "
                                    "(e.g. SequenceClassification or RewardModel).");
    }
    check_label_budget(*labels->array(), per_item, items.size, request.return_token_logprobs);
    check_prompt_budget(items);
    const auto shared = per_item ? nullptr : label_ids(*labels->array(), false);
    request.prompts.reserve(items.size);
    auto remaining = max_score_prompt_tokens;
    for (std::size_t index = 0; index < items.size; ++index) {
        DecoderOptions options;
        options.max_new_tokens = 0;
        options.label_token_ids = per_item
            ? label_ids(*(*labels->array())[index].array(), true) : shared;
        auto prompt = score_prompt(items, index, item_first, tokenizer, remaining);
        if (prompt.empty()) throw std::invalid_argument("Prompt must contain at least one token");
        remaining -= prompt.size();
        request.prompts.push_back({std::move(prompt), std::move(options)});
    }
    return request;
}

std::string score_response(const ScoreRequest& request, const std::vector<DecoderResult>& results) {
    if (results.size() != request.prompts.size()) throw std::logic_error("Incomplete score batch");
    std::size_t prompt_tokens = 0;
    std::string response = R"({"scores":[)";
    for (std::size_t index = 0; index < results.size(); ++index) {
        const auto& prompt = request.prompts[index];
        const auto& row = results[index].label_logprobs;
        if (!row || row->size() != prompt.options.label_token_ids->size()) {
            throw std::logic_error("Missing score result");
        }
        prompt_tokens += prompt.prompt.size();
        if (index) response += ',';
        append_row(response, probabilities(*row, request));
    }
    response += R"(],"pooled_hidden_states":null,"token_logprobs":)";
    if (request.return_token_logprobs) {
        response += '[';
        for (std::size_t index = 0; index < results.size(); ++index) {
            if (index) response += ',';
            append_row(response, *results[index].label_logprobs);
        }
        response += ']';
    } else {
        response += "null";
    }
    response += R"(,"model":)";
    json::append_quoted(response, request.model);
    response += R"(,"usage":{"prompt_tokens":)" + std::to_string(prompt_tokens);
    response += R"(,"total_tokens":)" + std::to_string(prompt_tokens);
    response += R"(,"completion_tokens":0,"prompt_tokens_details":null,"reasoning_tokens":0)";
    response += R"(,"completion_tokens_details":null},"object":"scoring"})";
    return response;
}

void serve_score(const Socket& connection, BatchService& service, int slot,
                 const Tokenizer& tokenizer, std::string_view body) {
    try {
        const auto request = parse_score_request(body, tokenizer);
        BatchCallbacks callbacks;
        callbacks.is_cancelled = [&] { return disconnected(connection); };
        const auto results = request.prompts.empty() ? std::vector<DecoderResult>{}
            : service.generate_batch(slot, request.prompts, callbacks);
        if (results) write_json(connection, "200 OK", score_response(request, *results));
    } catch (const ScoreValidationError& error) {
        write_tokenization_error(connection, "400 Bad Request", "Bad Request", error.what());
    } catch (const json::ParseError& error) {
        write_tokenization_error(connection, "400 Bad Request", "Bad Request", error.what());
    } catch (const std::invalid_argument& error) {
        write_tokenization_error(connection, "400 Bad Request", "BadRequestError", error.what());
    } catch (const std::length_error& error) {
        write_tokenization_error(connection, "400 Bad Request", "BadRequestError", error.what());
    } catch (const std::exception&) {
        write_tokenization_error(connection, "500 Internal Server Error", "InternalServerError",
                                 "Score request failed");
    }
}

}
