#include "sglang/server/text_completion.h"

#include "sglang/server/chat_template.h"
#include "sglang/decoder_constraint.h"
#include "sglang/json.h"
#include "sglang/server/http.h"
#include "sglang/server/native_generate.h"
#include "sglang/server/text_stop.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <fstream>
#include <iterator>
#include <limits>
#include <map>

namespace sglang::server {
namespace {

[[noreturn]] void bad_request(const char* message) {
    throw HttpError("400 Bad Request", message);
}

bool read_bool(const json::Object& object, std::string_view key, bool fallback = false) {
    const auto* value = json::get(object, key);
    if (!value) {
        return fallback;
    }
    const auto result = value->boolean();
    if (!result) {
        throw std::invalid_argument("Expected boolean " + std::string(key));
    }
    return *result;
}

int text_budget(const json::Value& value,
                const char* error = "Expected a nonnegative integer max_tokens") {
    const auto budget = value.unsigned_integer();
    if (!budget) {
        bad_request(error);
    }
    if (*budget >= checkpoint_context) {
        bad_request("Prompt and output budget must fit the 262144-token context");
    }
    return static_cast<int>(*budget);
}

void set_stops(const json::Object& object, DecoderOptions& options) {
    options.ignore_eos = read_bool(object, "ignore_eos");
    const bool keep_ids = !options.ignore_eos || options.min_new_tokens > 0;
    if (keep_ids) {
        options.additional_stop_token_ids = {end_of_text, end_of_message};
    }
    if (const auto* value = json::get(object, "stop_token_ids")) {
        const auto* stops = value->array();
        if (!stops || stops->size() > model_vocab_size) {
            bad_request("Expected an array of stop token IDs");
        }
        for (const auto& value : *stops) {
            const auto id = value.unsigned_integer();
            if (!id || *id >= model_vocab_size) {
                bad_request("Stop token ID falls outside the vocabulary");
            }
            if (keep_ids) {
                options.additional_stop_token_ids.insert(static_cast<int>(*id));
            }
        }
    }
}

void validate_minimum(const DecoderOptions& options, const char* error) {
    if (options.min_new_tokens > options.max_new_tokens) {
        bad_request(error);
    }
}

std::vector<std::string> stop_text(const json::Object& object, bool native = false) {
    const auto* value = json::get(object, "stop");
    if (!value || value->is_null()) {
        return {};
    }
    std::vector<std::string> stops;
    const auto append = [&](const json::Value& item) {
        const auto text = item.string();
        if (!text || (!native && text->empty())) {
            bad_request(native ? "Expected a stop string" : "Expected a nonempty stop string");
        }
        stops.emplace_back(*text);
    };
    if (const auto* array = value->array()) {
        if (native && array->size() > 32) {
            bad_request("At most 32 stop strings are allowed");
        }
        for (const auto& item : *array) {
            append(item);
        }
    } else {
        append(*value);
    }
    return stops;
}

std::optional<double> sampling_number(const json::Value& value) {
    if (const auto* wide = value.wide_integer()) {
        double parsed = 0;
        const auto* end = wide->digits.data() + wide->digits.size();
        const auto conversion = std::from_chars(wide->digits.data(), end, parsed);
        if (conversion.ec == std::errc{} && conversion.ptr == end) {
            return parsed;
        }
    }
    return value.number();
}

int bias_token(std::string_view key) {
    constexpr std::string_view whitespace = " \t\n\r\f\v";
    const auto first = key.find_first_not_of(whitespace);
    if (first == std::string_view::npos) {
        bad_request("logit_bias keys must be token IDs in [0, 248320)");
    }
    key = key.substr(first, key.find_last_not_of(whitespace) - first + 1);
    if (key.front() == '+') {
        key.remove_prefix(1);
        if (key.empty() || key.front() < '0' || key.front() > '9') {
            bad_request("logit_bias keys must be token IDs in [0, 248320)");
        }
    }
    int token = -1;
    const auto* end = key.data() + key.size();
    const auto parsed = std::from_chars(key.data(), end, token);
    if (parsed.ec != std::errc{} || parsed.ptr != end || token < 0 ||
        static_cast<TokenId>(token) >= model_vocab_size) {
        bad_request("logit_bias keys must be token IDs in [0, 248320)");
    }
    return token;
}

void set_logit_bias(const json::Object& object, DecoderOptions& options) {
    const auto* value = json::get(object, "logit_bias");
    if (!value || value->is_null()) {
        return;
    }
    std::map<int, float> entries;
    const auto read = [&](const auto& fields) {
        for (const auto& [key, value] : fields) {
            const int token = bias_token(key);
            const auto bias = sampling_number(value);
            if (!bias || !std::isfinite(*bias) || std::abs(*bias) > 1e30) {
                bad_request("logit_bias values must be finite numbers in [-1e30, 1e30]");
            }
            entries[token] = static_cast<float>(*bias);
        }
    };
    if (const auto* fields = value->ordered_object()) {
        read(*fields);
    } else if (const auto* fields = value->object()) {
        read(*fields);
    } else {
        bad_request("logit_bias must be an object mapping token IDs to numbers");
    }
    std::vector<DecoderLogitBias> active;
    for (const auto& [token, bias] : entries) {
        if (bias != 0) {
            active.push_back({token, bias});
        }
    }
    if (!active.empty()) {
        options.logit_bias =
            std::make_shared<const std::vector<DecoderLogitBias>>(std::move(active));
    }
}

double sampling_penalty(const json::Object& object, std::string_view key) {
    const auto* value = json::get(object, key);
    if (!value) {
        return 0.0;
    }
    const auto penalty = sampling_number(*value);
    if (!penalty || !std::isfinite(*penalty) || *penalty < -2 || *penalty > 2) {
        const auto message = std::string(key) + " must be a finite number in [-2, 2]";
        bad_request(message.c_str());
    }
    return *penalty;
}

double repetition_penalty(const json::Object& object, double fallback, bool allow_null) {
    const auto* value = json::get(object, "repetition_penalty");
    if (!value || (allow_null && value->is_null())) {
        return fallback;
    }
    const auto penalty = sampling_number(*value);
    if (!penalty || !(*penalty > 0 && *penalty <= 2)) {
        bad_request("repetition_penalty must be a finite number in (0, 2]");
    }
    return *penalty;
}

void set_sampling(const json::Object& object, DecoderOptions& options,
                  std::string_view seed_key = "seed", bool allow_null_repetition = true) {
    options.frequency_penalty = sampling_penalty(object, "frequency_penalty");
    options.presence_penalty = sampling_penalty(object, "presence_penalty");
    options.repetition_penalty =
        repetition_penalty(object, options.repetition_penalty, allow_null_repetition);
    if (const auto* value = json::get(object, "temperature")) {
        const auto temperature = sampling_number(*value);
        if (!temperature || !std::isfinite(*temperature) || *temperature < 0) {
            bad_request("temperature must be finite and nonnegative");
        }
        options.temperature = *temperature;
    }
    if (const auto* value = json::get(object, "top_p")) {
        const auto top_p = value->number();
        if (!top_p || !(*top_p > 0 && *top_p <= 1)) {
            bad_request("top_p must be in (0, 1]");
        }
        options.top_p = *top_p;
    }
    if (const auto* value = json::get(object, "min_p")) {
        const auto min_p = value->number();
        if (!min_p || !(*min_p >= 0 && *min_p <= 1)) {
            bad_request("min_p must be in [0, 1]");
        }
        options.min_p = *min_p;
    }
    if (const auto* value = json::get(object, "top_k")) {
        const auto positive = value->unsigned_integer();
        const auto* wide = value->wide_integer();
        if (value->signed_integer() == -1) {
            options.top_k = -1;
        } else if (positive && *positive > 0) {
            options.top_k = std::min<std::uint64_t>(*positive, model_vocab_size);
        } else if (wide && wide->digits.front() != '-') {
            options.top_k = model_vocab_size;
        } else {
            bad_request("top_k must be -1 or a positive integer");
        }
    }
    if (const auto* value = json::get(object, seed_key); value && !value->is_null()) {
        const auto positive = value->unsigned_integer();
        options.seed = value->signed_integer();
        if (positive && *positive <= std::numeric_limits<std::int64_t>::max()) {
            options.seed = static_cast<std::int64_t>(*positive);
        }
        if (!options.seed) {
            const auto message = std::string(seed_key) + " must be a signed 64-bit integer";
            bad_request(message.c_str());
        }
    }
}

const json::Value& response_schema(const json::Value& value) {
    const auto* format = value.ordered_object();
    const auto* type = format ? json::get(*format, "type") : nullptr;
    if (!type || type->string() != "json_schema") {
        bad_request("Unsupported response_format: expected type json_schema");
    }
    const auto* config = json::get(*format, "json_schema");
    const auto* descriptor = config ? config->ordered_object() : nullptr;
    if (!descriptor || format->size() != 2) {
        bad_request("Expected response_format.json_schema object");
    }
    for (const auto& [key, field] : *descriptor) {
        if (key == "schema") {
            continue;
        }
        if (key == "strict" && field.boolean() == true) {
            continue;
        }
        if (key == "name" && field.string() && !field.string()->empty() &&
            field.string()->size() <= 64) {
            continue;
        }
        if (key == "description" && field.string() && field.string()->size() <= 256) {
            continue;
        }
        bad_request("Unsupported JSON Schema response format descriptor");
    }
    const auto* schema = json::get(*descriptor, "schema");
    if (!schema) {
        bad_request("Expected response_format.json_schema.schema");
    }
    return *schema;
}

void set_constraint(const json::Object& object, const CompletionTokenizer& tokenizer,
                    DecoderOptions& options) {
    const auto* regex = json::get(object, "regex");
    const auto* ebnf = json::get(object, "ebnf");
    const auto* schema = json::get(object, "response_format");
    if (schema && (regex || ebnf)) {
        bad_request("Use one of response_format, regex and ebnf");
    }
    if (regex && ebnf) {
        bad_request("Use one of regex and ebnf");
    }
    const auto* value = regex ? regex : ebnf;
    if (!value && !schema) {
        return;
    }
    if (value && !value->string()) {
        bad_request(regex ? "regex must be a string" : "ebnf must be a string");
    }
    if (read_bool(object, "ignore_eos") || json::get(object, "stop_token_ids")) {
        if (schema) {
            bad_request("JSON Schema requires model EOS stops without stop_token_ids overrides");
        }
        bad_request(regex ? "regex requires model EOS stops without stop_token_ids overrides"
                          : "ebnf requires model EOS stops without stop_token_ids overrides");
    }
    if (schema) {
        const auto& descriptor = response_schema(*schema);
        options.constraint = TokenConstraint::compile_json_schema(tokenizer.get(), descriptor);
    } else {
        options.constraint = regex
            ? TokenConstraint::compile_regex(tokenizer.get(), *value->string())
            : TokenConstraint::compile(tokenizer.get(), *value->string());
    }
    if (options.max_new_tokens < options.constraint->minimum_tokens()) {
        bad_request(options.constraint->budget_error());
    }
}

int openai_logprob_count(const json::Object& object, bool chat) {
    const auto* enabled = json::get(object, "logprobs");
    const auto* count = chat ? json::get(object, "top_logprobs") : enabled;
    const auto top = logprob_width(count);
    if (!top) {
        bad_request(chat ? "Expected integer top_logprobs in [0, 20]"
                         : "Expected integer logprobs in [0, 20]");
    }
    if (chat) {
        return read_bool(object, "logprobs") ? *top : -1;
    }
    return enabled != nullptr && !enabled->is_null() ? *top : -1;
}

TextRequest text_options(const json::Object& object, const CompletionTokenizer& tokenizer,
                         bool chat) {
    TextRequest request;
    request.chat = chat;
    const auto* choices = json::get(object, "n");
    const auto count = choices ? choices->unsigned_integer().value_or(0) : 1;
    if (count < 1 || count > 4) bad_request("Expected n between 1 and 4");
    request.choices = count;
    request.stream = read_bool(object, "stream");
    request.return_token_ids = read_bool(object, "return_token_ids");
    request.options.temperature = 1.0;
    if (chat) {
        request.options = tokenizer.chat_sampling_defaults();
    }
    const auto* budget = json::get(object, "max_tokens");
    request.options.incremental_kv = chat && (!budget || budget->is_null());
    request.options.max_new_tokens = request.options.incremental_kv ? 0 :
        (budget ? text_budget(*budget) : 16);
    if (const auto* minimum = json::get(object, "min_tokens")) {
        request.options.min_new_tokens =
            text_budget(*minimum, "Expected a nonnegative integer min_tokens");
    }
    if (const auto* value = json::get(object, "stream_options")) {
        const auto* options = value->object();
        if (!request.stream || !options || options->size() != 1 ||
            !json::get_boolean(*options, "include_usage").has_value()) {
            bad_request("Expected stream_options.include_usage for a stream");
        }
        request.include_usage = read_bool(*options, "include_usage");
    }
    set_stops(object, request.options);
    request.stop_text = stop_text(object);
    set_sampling(object, request.options, "seed", chat);
    set_logit_bias(object, request.options);
    set_cache_salt(object, request.options);
    request.options.generated_top_logprobs = openai_logprob_count(object, chat);
    if (request.options.generated_top_logprobs >= 0 && request.options.samples()) {
        bad_request("Generated logprobs require greedy sampling");
    }
    return request;
}

void finish_options(TextRequest& request, const json::Object& object,
                    const CompletionTokenizer& tokenizer) {
    if (request.options.incremental_kv) {
        constexpr int chat_token_limit = checkpoint_context - 2;
        constexpr int chat_prompt_limit = checkpoint_context - 6;
        if (request.prompt.size() >= chat_prompt_limit) {
            bad_request("Prompt must fit within the 262144-token context");
        }
        request.options.max_new_tokens =
            chat_token_limit - static_cast<int>(request.prompt.size());
    }
    validate_minimum(request.options, "min_tokens must not exceed the output token budget");
    set_constraint(object, tokenizer, request.options);
    if (request.options.generated_top_logprobs >= 0 && request.options.constraint) {
        bad_request("Generated logprobs do not support response constraints");
    }
    if (request.options.constraint && !request.stop_text.empty()) {
        bad_request("Text stops cannot be combined with a response constraint");
    }
}

void validate_prompt(std::string_view prompt) {
    if (prompt.empty() || prompt.size() > 1024 * 1024) {
        bad_request("Expected a nonempty prompt string of up to 1048576 bytes");
    }
}

void encode_prompt(std::vector<int>& output, int max_new_tokens, std::string_view prompt,
                   const CompletionTokenizer& tokenizer) {
    try {
        const auto ids = tokenizer.get().encode(prompt, checkpoint_context - max_new_tokens);
        output.assign(ids.begin(), ids.end());
    } catch (const std::length_error&) {
        bad_request("Prompt and output budget must fit the 262144-token context");
    }
}

std::vector<int> token_prompt(const json::Value& input, int max_new_tokens,
                              const char* shape_error, const char* token_error,
                              bool allow_empty = false) {
    const auto* tokens = input.array();
    if (!tokens || (tokens->empty() && !allow_empty)) {
        bad_request(shape_error);
    }
    if (tokens->size() > checkpoint_context - max_new_tokens) {
        bad_request("Prompt and output budget must fit the 262144-token context");
    }
    std::vector<int> prompt;
    prompt.reserve(tokens->size());
    for (const auto& value : *tokens) {
        const auto token = value.unsigned_integer();
        if (!token || *token >= model_vocab_size) {
            bad_request(token_error);
        }
        prompt.push_back(static_cast<int>(*token));
    }
    return prompt;
}

std::vector<int> completion_prompt(const json::Value& input, bool text, int max_new_tokens,
                                   const CompletionTokenizer& tokenizer) {
    if (!text) {
        return token_prompt(input, max_new_tokens, "Expected a nonempty array of prompt token IDs",
                            "Prompt token IDs must be integers in [0, 248320)");
    }
    const auto prompt = input.string().value_or("");
    validate_prompt(prompt);
    std::vector<int> ids;
    encode_prompt(ids, max_new_tokens, prompt, tokenizer);
    return ids;
}

bool is_text_option(std::string_view key) {
    return key == "model" || key == "max_tokens" || key == "min_tokens" ||
        key == "stream" || key == "stream_options" ||
        key == "ignore_eos" || key == "stop_token_ids" || key == "stop" ||
        key == "return_token_ids" || key == "logprobs" ||
        key == "temperature" || key == "top_p" || key == "top_k" || key == "seed" ||
        key == "min_p" || key == "logit_bias" || key == "ebnf" || key == "regex" ||
        key == "frequency_penalty" || key == "presence_penalty" ||
        key == "repetition_penalty" || key == "response_format" || key == "cache_salt";
}

TextCompletionRequest parse_text(const json::Object& object,
                                 const CompletionTokenizer& tokenizer) {
    for (const auto& [key, value] : object) {
        if (key != "prompt" && key != "n" && key != "echo" && !is_text_option(key)) {
            throw std::invalid_argument("Unsupported text completion option: " + key);
        }
    }
    const auto model = tokenizer.serving_profile().served_name;
    if (json::get_string(object, "model") != model) {
        const auto expected = "Expected model " + std::string(model);
        const auto message = json::get_string(object, "prompt") ?
            expected + " for a string prompt" : expected;
        bad_request(message.c_str());
    }
    const auto* prompt = json::get(object, "prompt");
    const auto* list = prompt ? prompt->array() : nullptr;
    if (!prompt || (!prompt->string() && (!list || list->empty()))) {
        bad_request("Expected a nonempty prompt string or array");
    }
    TextCompletionRequest completion;
    auto& request = completion.request;
    request = text_options(object, tokenizer, false);
    const bool echo = read_bool(object, "echo");
    request.options.score_prompt = echo && request.options.generated_top_logprobs > 0;
    const int budget = request.options.max_new_tokens;
    const bool batch = list && (list->front().string() || list->front().array());
    if (!batch) {
        request.prompt = completion_prompt(*prompt, list == nullptr, budget, tokenizer);
        if (echo) request.echo = completion_echo(*prompt, request.prompt, tokenizer.get());
        finish_options(request, object, tokenizer);
        return completion;
    }
    if (request.choices > 1) bad_request("Multiple choices require a single completion prompt");
    if (list->size() > decoder_batch_max_lanes) {
        bad_request("Expected 1 to 32 completion batch items");
    }
    if (list->size() * budget > checkpoint_context) {
        bad_request("Batch output budgets must total at most 262144 tokens");
    }
    const bool text = list->front().string().has_value();
    completion.batch.reserve(list->size());
    for (const auto& item : *list) {
        if (text ? !item.string() : !item.array()) {
            bad_request("Expected a homogeneous list of text prompts or token ID arrays");
        }
        completion.batch.push_back({completion_prompt(item, text, budget, tokenizer), {}});
        if (echo) {
            completion.batch_echoes.push_back(
                completion_echo(item, completion.batch.back().prompt, tokenizer.get()));
        }
    }
    finish_options(request, object, tokenizer);
    for (auto& item : completion.batch) {
        item.options = request.options;
    }
    return completion;
}

json::Object chat_text_options(const json::OrderedObject& chat, bool media,
                                const model::ServingProfile& profile) {
    json::Object options;
    for (const auto& [key, value] : chat) {
        if (key == "stream_options" && value.ordered_object()) {
            const auto& fields = *value.ordered_object();
            options.emplace(key, json::Object(fields.begin(), fields.end()));
        } else if ((key == "temperature" || key == "top_p" || key == "top_k" ||
                    key == "min_p" || key == "seed") && value.is_null()) {
            continue;
        } else if (is_text_option(key) || key == "top_logprobs" || key == "n") {
            options.emplace(key, value);
        } else if (key == "max_completion_tokens") {
            if (!value.is_null()) {
                options.insert_or_assign("max_tokens", value);
            }
        } else if (media && key == "add_special_tokens") {
            if (value.boolean() != false) {
                bad_request("Decoded media chat requires add_special_tokens=false");
            }
        } else if (key != "messages" && key != "tools" && key != "tool_choice" &&
                   key != "add_generation_prompt" && key != "separate_reasoning" &&
                   key != "stream_reasoning" && key != "parallel_tool_calls" &&
                   !profile.is_chat_template_option(key)) {
            throw std::invalid_argument("Unsupported chat completion option: " + key);
        }
    }
    if (json::get_string(options, "model") != profile.served_name) {
        const auto message = "Expected model " + std::string(profile.served_name) +
            " for a chat request";
        bad_request(message.c_str());
    }
    return options;
}

json::Value parse_body(std::string_view body, json::Syntax syntax) {
    try {
        return json::parse(body, syntax);
    } catch (const json::ParseError&) {
        bad_request("Invalid JSON request body");
    }
}

}

CompletionTokenizer::CompletionTokenizer(const std::filesystem::path& checkpoint,
                                         SamplingDefaults sampling, ChatParsers parsers,
                                         const model::ServingProfile& profile)
    : checkpoint_(checkpoint), sampling_mode_(sampling), chat_parsers_(parsers),
      profile_(profile) {}

CompletionTokenizer::~CompletionTokenizer() = default;

ChatParsers CompletionTokenizer::chat_parsers() const { return chat_parsers_; }

const model::ServingProfile& CompletionTokenizer::serving_profile() const { return profile_; }

const ChatTemplate& CompletionTokenizer::chat_template() const {
    std::call_once(chat_loaded_, [this] {
        chat_template_ = profile_.load_chat_template(checkpoint_);
    });
    return *chat_template_;
}

const DecoderOptions& CompletionTokenizer::chat_sampling_defaults() const {
    std::call_once(sampling_loaded_, [this] {
        const auto path = checkpoint_ / "generation_config.json";
        auto defaults = std::make_unique<DecoderOptions>();
        defaults->temperature = 1.0;
        if (sampling_mode_ == SamplingDefaults::openai || !std::filesystem::exists(path)) {
            sampling_defaults_ = std::move(defaults);
            return;
        }
        std::ifstream stream(path, std::ios::binary);
        if (!stream) {
            throw std::runtime_error("Cannot open checkpoint generation config: " + path.string());
        }
        const std::string contents{std::istreambuf_iterator<char>(stream), {}};
        const auto config = json::parse(contents);
        const auto* object = config.object();
        if (!object) {
            throw std::runtime_error("Expected a checkpoint generation config object");
        }
        set_sampling(*object, *defaults);
        defaults->seed.reset();
        sampling_defaults_ = std::move(defaults);
    });
    return *sampling_defaults_;
}

TextCompletionRequest parse_text_request(const json::Object& body,
                                         const CompletionTokenizer& tokenizer) {
    return parse_text(body, tokenizer);
}

std::optional<int> logprob_width(const json::Value* value) {
    if (value == nullptr || value->is_null())
        return 0;
    const auto width = value->unsigned_integer();
    if (!width || *width > decoder_max_generated_top_logprobs)
        return std::nullopt;
    return static_cast<int>(*width);
}

namespace {

void set_cache_salt_value(const json::Value& value, DecoderOptions& options) {
    const auto salt = value.string();
    if (!salt || salt->size() > decoder_prefix_max_salt_bytes) {
        bad_request("Expected a cache_salt string of up to 256 bytes");
    }
    options.prefix_cache_salt = *salt;
}

void validate_generate_fields(const json::Object& object) {
    for (const auto& [key, value] : object) {
        if (key == "text" || key == "input_ids" || key == "sampling_params" ||
            key == "stream" || key == "cache_salt" || key == "rid" || key == "session_params" ||
            key == "return_logprob" || key == "logprob_start_len" || key == "top_logprobs_num" ||
            key == "return_text_in_logprobs") {
            continue;
        }
        if ((key == "return_routed_experts" && (value.is_null() || value.boolean() == false)) ||
            ((key == "lora_path" || key == "token_ids_logprob") && value.is_null())) {
            continue;
        }
        throw std::invalid_argument("Unsupported /generate field: " + key);
    }
}

BatchPrompt generate_options(const json::Value* params, const CompletionTokenizer& tokenizer) {
    BatchPrompt request;
    auto& options = request.options;
    options.max_new_tokens = 128;
    options.temperature = 1.0;
    json::Object sampling;
    if (params && !params->is_null()) {
        const auto* object = params->object();
        if (!object) {
            bad_request("Expected one sampling_params object");
        }
        for (const auto& [key, parameter] : *object) {
            if (key == "max_new_tokens") {
                options.max_new_tokens = text_budget(parameter,
                    "Expected a finite nonnegative integer sampling_params.max_new_tokens");
            } else if (key == "min_new_tokens") {
                if (!parameter.is_null()) {
                    options.min_new_tokens = text_budget(parameter,
                        "Expected a nonnegative integer sampling_params.min_new_tokens");
                }
            } else if (key == "n" || key == "stop") {
                continue;
            } else if (key == "temperature" || key == "top_p" || key == "top_k" ||
                       key == "min_p" || key == "sampling_seed" || key == "ignore_eos" ||
                       key == "stop_token_ids" || key == "logit_bias" ||
                       key == "frequency_penalty" || key == "presence_penalty" ||
                       key == "repetition_penalty" || key == "regex" || key == "json_schema" ||
                       key == "ebnf" || key == "no_stop_trim" || key == "skip_special_tokens") {
                if (!parameter.is_null()) {
                    sampling.emplace(key, parameter);
                }
            } else {
                throw std::invalid_argument("Unsupported /generate sampling parameter: " + key);
            }
        }
    }
    validate_minimum(options, "sampling_params.min_new_tokens must not exceed max_new_tokens");
    set_sampling(sampling, options, "sampling_seed");
    set_logit_bias(sampling, options);
    set_stops(sampling, options);
    set_generate_constraint(sampling, tokenizer.get(), options);
    request.no_stop_trim = read_bool(sampling, "no_stop_trim");
    request.skip_special_tokens = read_bool(sampling, "skip_special_tokens", true);
    return request;
}

std::shared_ptr<const NativeTextStops> generate_stops(const json::Value* params,
                                                     const CompletionTokenizer& tokenizer) {
    const auto* object = params ? params->object() : nullptr;
    if (object) {
        const auto stops = stop_text(*object, true);
        if (!stops.empty()) {
            return std::make_shared<NativeTextStops>(NativeTextStops{tokenizer.get(),
                                                                     TextStops(stops)});
        }
    }
    return nullptr;
}

BatchPrompt generate_prompt(const json::Value& input, bool text, BatchPrompt request,
                            const CompletionTokenizer& tokenizer,
                            std::shared_ptr<const NativeTextStops> stops,
                            const std::vector<int>* history) {
    request.text_stops = std::move(stops);
    const auto history_size = history ? history->size() : 0;
    if (history_size + request.options.max_new_tokens > checkpoint_context) {
        bad_request("Prompt and output budget must fit the 262144-token context");
    }
    const int occupied = request.options.max_new_tokens + static_cast<int>(history_size);
    if (text) {
        const auto prompt = input.string();
        if (!prompt || (prompt->empty() && !history) || prompt->size() > 1024 * 1024) {
            bad_request("Expected one nonempty text string of up to 1048576 bytes");
        }
        if (!prompt->empty()) {
            encode_prompt(request.prompt, occupied, *prompt, tokenizer);
        }
    } else {
        request.prompt = token_prompt(input, occupied,
                                      "Expected one nonempty flat array of input_ids",
                                      "input_ids must contain integers in [0, 248320)", history);
    }
    if (history) {
        request.prompt.insert(request.prompt.begin(), history->begin(), history->end());
        if (request.prompt.empty()) {
            bad_request("A session request must contain input tokens after restoring history.");
        }
    }
    return request;
}

const json::Value* generate_logprob_field(const GenerateInput& input, std::string_view key,
                                         std::size_t index) {
    const auto* value = json::get(input.fields, key);
    if (const auto* list = value ? value->array() : nullptr) {
        if (!input.batch || list->size() != input.count) {
            throw std::invalid_argument("Expected one " + std::string(key) + " per batch item");
        }
        return &(*list)[index];
    }
    return value;
}

void generate_logprob_options(const GenerateInput& input, std::size_t index, BatchPrompt& item) {
    const auto* enabled = generate_logprob_field(input, "return_logprob", index);
    const auto* start = generate_logprob_field(input, "logprob_start_len", index);
    const auto* top = generate_logprob_field(input, "top_logprobs_num", index);
    const auto* text = json::get(input.fields, "return_text_in_logprobs");
    if (enabled && !enabled->is_null() && !enabled->boolean()) {
        bad_request("Expected boolean return_logprob");
    }
    const auto count = logprob_width(top);
    if (!count) {
        bad_request("Expected integer top_logprobs_num in [0, 20]");
    }
    if (text && !text->is_null() && !text->boolean()) {
        bad_request("Expected boolean return_text_in_logprobs");
    }
    if (enabled && enabled->boolean() == true) {
        if (start && !start->is_null() && start->signed_integer() != -1) {
            const auto offset = start->unsigned_integer();
            if (!offset || *offset > item.prompt.size()) bad_request("Invalid logprob_start_len");
            item.options.score_prompt = *offset < item.prompt.size();
            item.options.prompt_logprob_start = static_cast<int>(*offset);
        }
        if (item.options.constraint) {
            bad_request("Generated logprobs do not support response constraints");
        }
        if (item.options.samples()) {
            bad_request("return_logprob requires greedy sampling");
        }
        item.options.generated_top_logprobs = *count;
        item.return_text_in_logprobs = text && text->boolean() == true;
    }
}

}

std::size_t GenerateInput::item_count() const { return count * samples; }

bool GenerateInput::indexed() const { return batch || samples > 1; }

GenerateInput parse_generate_input(std::string_view body) {
    auto parsed = parse_body(body, json::Syntax::native_generate);
    auto* object = parsed.object();
    if (!object) {
        bad_request("Expected a JSON object for /generate");
    }
    validate_generate_fields(*object);
    GenerateInput input;
    if (const auto* stream = json::get(*object, "stream"); stream && !stream->is_null()) {
        input.stream = read_bool(*object, "stream");
    }
    const auto* text = json::get(*object, "text");
    const auto* ids = json::get(*object, "input_ids");
    const bool has_text = text && !text->is_null();
    const bool has_ids = ids && !ids->is_null();
    if (has_text == has_ids) {
        bad_request("Provide exactly one of text or input_ids");
    }
    const auto* list = (has_text ? text : ids)->array();
    input.text = has_text;
    input.batch = list && (has_text || (!list->empty() && list->front().array()));
    if (input.batch) {
        if (list->empty() || list->size() > decoder_batch_max_lanes) {
            bad_request("Expected 1 to 32 /generate batch items");
        }
        input.count = list->size();
    }
    input.fields = std::move(*object);
    const auto* params = json::get(input.fields, "sampling_params");
    const auto* param_list = generate_parameter_list(params, input.count, false);
    input.samples = generate_sample_count(input, param_list);
    if (param_list && !input.indexed()) {
        bad_request("Expected one sampling_params object");
    }
    return input;
}

std::vector<BatchPrompt> parse_generate_prompts(const GenerateInput& input,
                                               const CompletionTokenizer& tokenizer,
                                               const std::vector<int>* history) {
    const auto& prompt = *json::get(input.fields, input.text ? "text" : "input_ids");
    const auto* params = json::get(input.fields, "sampling_params");
    const auto* salt = json::get(input.fields, "cache_salt");
    const auto* list = input.batch ? prompt.array() : nullptr;
    const auto* param_list = params ? params->array() : nullptr;
    const auto* salt_list = input.indexed() ? generate_parameter_list(salt, input.count, true)
                                            : nullptr;
    const auto broadcast = param_list ? BatchPrompt{} : generate_options(params, tokenizer);
    const auto broadcast_stops = param_list ? nullptr : generate_stops(params, tokenizer);
    std::vector<BatchPrompt> items;
    items.reserve(input.item_count());
    const int samples = static_cast<int>(input.samples);
    int remaining = checkpoint_context;
    for (std::size_t index = 0; index < input.count; ++index) {
        auto item = param_list ? generate_options(&(*param_list)[index], tokenizer) : broadcast;
        auto& options = item.options;
        const auto* item_salt = salt_list ? &(*salt_list)[index] : salt;
        if (item_salt && !item_salt->is_null()) {
            set_cache_salt_value(*item_salt, options);
        }
        if (options.max_new_tokens > remaining / samples) {
            bad_request("Batch output budgets must total at most 262144 tokens");
        }
        remaining -= options.max_new_tokens * samples;
        const auto stops = param_list ? generate_stops(&(*param_list)[index], tokenizer)
                                      : broadcast_stops;
        items.push_back(generate_prompt(list ? (*list)[index] : prompt, input.text,
                                        std::move(item), tokenizer, stops, history));
        generate_logprob_options(input, index, items.back());
        items.insert(items.end(), input.samples - 1, items.back());
    }
    return items;
}

void set_cache_salt(const json::Object& object, DecoderOptions& options) {
    if (const auto* salt = json::get(object, "cache_salt")) {
        set_cache_salt_value(*salt, options);
    }
}

TextRequest parse_chat_request(const json::OrderedObject& chat,
                               const CompletionTokenizer& tokenizer,
                               bool media_enabled) {
    auto media = parse_chat_media(chat, media_enabled);
    auto normalized = media ? std::move(media->chat) : chat;
    const auto& profile = tokenizer.serving_profile();
    profile.normalize_chat_options(normalized);
    const auto options = chat_text_options(normalized, media.has_value(), profile);
    auto request = text_options(options, tokenizer, true);
    const auto tools = profile.select_chat_tools(normalized);
    if (request.choices > 1 && (media || !tools.tools.empty())) {
        bad_request("Multiple chat choices support text without tools");
    }
    const auto prompt = tokenizer.chat_template().render(normalized, ChatInput::http, tools);
    validate_prompt(prompt);
    encode_prompt(request.prompt, request.options.max_new_tokens, prompt, tokenizer);
    if (media) {
        auto& prepared = request.media.emplace();
        prepared.template_ids = std::move(request.prompt);
        try {
            request.prompt = expand_chat_media(prepared, media->parts, tokenizer.get());
        } catch (const std::length_error& error) {
            throw HttpError("400 Bad Request", error.what());
        }
        if (request.prompt.size() > checkpoint_context - request.options.max_new_tokens) {
            bad_request("Expanded prompt and output budget must fit the 262144-token context");
        }
    }
    finish_options(request, options, tokenizer);
    profile.prepare_chat_output(request, normalized, prompt, tokenizer, tools);
    if (request.options.generated_top_logprobs >= 0 && request.chat_output.enabled()) {
        bad_request("Generated logprobs do not support chat reasoning or tool output parsers");
    }
    return request;
}

}
