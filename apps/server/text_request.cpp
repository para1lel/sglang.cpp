#include "sglang/server/text_completion.h"

#include "sglang/server/chat_template.h"
#include "sglang/server/chat_tool_constraints.h"
#include "sglang/decoder_constraint.h"
#include "sglang/json.h"
#include "sglang/server/http.h"
#include "sglang/server/native_generate.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <fstream>
#include <iterator>
#include <limits>

namespace sglang::server {
namespace {

[[noreturn]] void bad_request(const char* message) {
    throw HttpError("400 Bad Request", message);
}

bool read_bool(const json::Object& object, std::string_view key) {
    const auto* value = json::get(object, key);
    if (!value) {
        return false;
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
    const bool ignore_eos = read_bool(object, "ignore_eos");
    if (!ignore_eos) {
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
            if (!ignore_eos) {
                options.additional_stop_token_ids.insert(static_cast<int>(*id));
            }
        }
    }
}

std::vector<std::string> stop_text(const json::Object& object) {
    const auto* value = json::get(object, "stop");
    if (!value || value->is_null()) {
        return {};
    }
    std::vector<std::string> stops;
    const auto append = [&](const json::Value& item) {
        const auto text = item.string();
        if (!text || text->empty()) {
            bad_request("Expected a nonempty stop string");
        }
        stops.emplace_back(*text);
    };
    if (const auto* array = value->array()) {
        for (const auto& item : *array) {
            append(item);
        }
    } else {
        append(*value);
    }
    return stops;
}

void set_sampling(const json::Object& object, DecoderOptions& options,
                  std::string_view seed_key = "seed") {
    if (const auto* value = json::get(object, "temperature")) {
        auto temperature = value->number();
        if (const auto* wide = value->wide_integer()) {
            double parsed = 0;
            const auto* end = wide->digits.data() + wide->digits.size();
            const auto conversion = std::from_chars(wide->digits.data(), end, parsed);
            if (conversion.ec == std::errc{} && conversion.ptr == end) {
                temperature = parsed;
            }
        }
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

TextRequest text_options(const json::Object& object, const CompletionTokenizer& tokenizer,
                         bool chat) {
    TextRequest request;
    request.chat = chat;
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
    set_sampling(object, request.options);
    set_cache_salt(object, request.options);
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
    set_constraint(object, tokenizer, request.options);
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

bool is_text_option(std::string_view key) {
    return key == "model" || key == "max_tokens" || key == "stream" || key == "stream_options" ||
        key == "ignore_eos" || key == "stop_token_ids" || key == "stop" ||
        key == "return_token_ids" ||
        key == "temperature" || key == "top_p" || key == "top_k" || key == "seed" ||
        key == "min_p" || key == "ebnf" || key == "regex" || key == "response_format" ||
        key == "cache_salt";
}

TextRequest parse_text(const json::Object& object, const CompletionTokenizer& tokenizer) {
    for (const auto& [key, value] : object) {
        if (key != "prompt" && !is_text_option(key)) {
            throw std::invalid_argument("Unsupported text completion option: " + key);
        }
    }
    if (json::get_string(object, "model") != model_name) {
        bad_request("Expected model Qwen3.8-27B-FP8 for a string prompt");
    }
    const auto prompt = json::get_string(object, "prompt").value_or("");
    validate_prompt(prompt);
    auto request = text_options(object, tokenizer, false);
    encode_prompt(request.prompt, request.options.max_new_tokens, prompt, tokenizer);
    finish_options(request, object, tokenizer);
    return request;
}

json::Object chat_text_options(const json::OrderedObject& chat, bool media) {
    json::Object options;
    for (const auto& [key, value] : chat) {
        if (key == "stream_options" && value.ordered_object()) {
            const auto& fields = *value.ordered_object();
            options.emplace(key, json::Object(fields.begin(), fields.end()));
        } else if (media && key == "ebnf") {
            bad_request("Decoded media chat does not support EBNF constraints");
        } else if (media && key == "regex") {
            bad_request("Decoded media chat does not support regex constraints");
        } else if (media && key == "response_format") {
            bad_request("Decoded media chat does not support JSON Schema constraints");
        } else if (media && key == "cache_salt") {
            bad_request("Decoded media chat does not support cache_salt");
        } else if ((key == "temperature" || key == "top_p" || key == "top_k" ||
                    key == "min_p" || key == "seed") && value.is_null()) {
            continue;
        } else if (is_text_option(key)) {
            options.emplace(key, value);
        } else if (key == "max_completion_tokens") {
            if (!value.is_null()) {
                options.insert_or_assign("max_tokens", value);
            }
        } else if (key == "n") {
            if (value.unsigned_integer() != 1) {
                bad_request("Only n=1 is supported");
            }
        } else if (media && key == "add_special_tokens") {
            if (value.boolean() != false) {
                bad_request("Decoded media chat requires add_special_tokens=false");
            }
        } else if (key != "messages" && key != "tools" && key != "tool_choice" &&
                   key != "add_generation_prompt" && key != "add_vision_id" &&
                   key != "enable_thinking" && key != "preserve_thinking" &&
                   key != "reasoning_effort" && key != "separate_reasoning" &&
                   key != "stream_reasoning" && key != "parallel_tool_calls") {
            throw std::invalid_argument("Unsupported chat completion option: " + key);
        }
    }
    if (json::get_string(options, "model") != model_name) {
        bad_request("Expected model Qwen3.8-27B-FP8 for a chat request");
    }
    return options;
}

void normalize_template_options(json::OrderedObject& chat) {
    const auto kwargs = std::find_if(chat.begin(), chat.end(), [](const auto& field) {
        return field.first == "chat_template_kwargs";
    });
    if (kwargs == chat.end()) {
        return;
    }
    if (kwargs->second.is_null()) {
        chat.erase(kwargs);
        return;
    }
    const auto* object = kwargs->second.ordered_object();
    if (!object) {
        bad_request("Expected a chat_template_kwargs object");
    }
    auto settings = *object;
    chat.erase(kwargs);
    for (auto& [key, value] : settings) {
        if (key != "enable_thinking" && key != "preserve_thinking" &&
            key != "reasoning_effort" && key != "add_vision_id") {
            bad_request("Unsupported chat_template_kwargs option");
        }
        if (!json::get(chat, key)) {
            chat.emplace_back(std::move(key), std::move(value));
        }
    }
}

bool chat_flag(const json::OrderedObject& chat, std::string_view key) {
    const auto* value = json::get(chat, key);
    if (!value) {
        return true;
    }
    if (const auto flag = value->boolean()) {
        return *flag;
    }
    throw std::invalid_argument("Expected a boolean for " + std::string(key));
}

bool model_eos_stops(const DecoderOptions& options) {
    const auto& stops = options.additional_stop_token_ids;
    return (!options.stop_token_id || constraint_eos(*options.stop_token_id)) &&
        stops.count(end_of_text) && stops.count(end_of_message) &&
        std::all_of(stops.begin(), stops.end(), constraint_eos);
}

void check_tool_constraint(const TextRequest& request, const ChatTools& tools,
                           ChatParsers parsers) {
    const bool forced = tools.forced();
    if (!forced && (!tool_constraint_applies(tools) || request.media ||
                    request.options.constraint || !parsers.qwen3_coder_tools)) {
        return;
    }
    if (request.media) {
        bad_request("Decoded media chat does not support tool_choice required or a named tool");
    }
    if (request.options.constraint) {
        bad_request("tool_choice 'required' or a named tool cannot be combined with "
                    "response_format, regex, or ebnf: the tool-call constraint and the output "
                    "constraint cannot both be honored.");
    }
    if (!parsers.qwen3_coder_tools) {
        bad_request("tool_choice 'required' or a named tool needs --tool-call-parser qwen3_coder");
    }
    const std::string subject = forced ? "tool_choice 'required' or a named tool" :
                                         "A strict tool with tool_choice auto";
    if (!model_eos_stops(request.options)) {
        bad_request((subject + " requires model EOS stops without ignore_eos or stop_token_ids "
                    "overrides").c_str());
    }
    if (!request.stop_text.empty()) {
        bad_request((subject + " cannot be combined with text stops").c_str());
    }
}

void set_chat_output(TextRequest& request, const json::OrderedObject& chat,
                     std::string_view prompt, const CompletionTokenizer& tokenizer,
                     const ChatTools& tools) {
    const auto parsers = tokenizer.chat_parsers();
    const bool separate = chat_flag(chat, "separate_reasoning");
    const bool stream = chat_flag(chat, "stream_reasoning");
    check_tool_constraint(request, tools, parsers);
    if (request.media || request.options.constraint) {
        return;
    }
    constexpr std::string_view open_suffix = "<|im_start|>assistant\n<think>\n";
    auto& output = request.chat_output;
    output.separate_reasoning = parsers.qwen3_reasoning && separate;
    output.stream_reasoning = stream;
    output.prompt_has_open_think = prompt.size() >= open_suffix.size() &&
        prompt.substr(prompt.size() - open_suffix.size()) == open_suffix;
    if (!parsers.qwen3_coder_tools) {
        return;
    }
    output.parallel_tool_calls = tools.parallel && tools.choice != ChatToolChoice::named;
    read_chat_tools(output, tools.tools);
    request.options.constraint = tokenizer.tool_constraint(tools, output.prompt_has_open_think);
    output.complete_truncated_calls = request.options.constraint != nullptr;
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
                                         SamplingDefaults sampling, ChatParsers parsers)
    : checkpoint_(checkpoint), sampling_mode_(sampling), chat_parsers_(parsers) {}

CompletionTokenizer::~CompletionTokenizer() = default;

const Tokenizer& CompletionTokenizer::get() const {
    std::call_once(loaded_, [this] {
        tokenizer_ = std::make_unique<const Tokenizer>(checkpoint_ / "tokenizer.json");
    });
    return *tokenizer_;
}

const ChatTemplate& CompletionTokenizer::chat_template() const {
    std::call_once(chat_loaded_, [this] {
        chat_template_ = std::make_unique<const ChatTemplate>(checkpoint_);
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

TextRequest parse_text_request(const json::Object& body, const CompletionTokenizer& tokenizer) {
    return parse_text(body, tokenizer);
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
            key == "stream" || key == "cache_salt") {
            continue;
        }
        if (((key == "return_logprob" || key == "return_routed_experts") &&
             (value.is_null() || value.boolean() == false)) ||
            (key == "logprob_start_len" && (value.is_null() || value.signed_integer() == -1)) ||
            (key == "lora_path" && value.is_null())) {
            continue;
        }
        throw std::invalid_argument("Unsupported /generate field: " + key);
    }
}

DecoderOptions generate_options(const json::Value* params, const json::Value* salt) {
    DecoderOptions options;
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
            } else if (key == "n") {
                if (!parameter.is_null() && parameter.unsigned_integer() != 1) {
                    bad_request("Only sampling_params.n = 1 is supported");
                }
            } else if (key == "temperature" || key == "top_p" || key == "top_k" ||
                       key == "min_p" || key == "sampling_seed" || key == "ignore_eos" ||
                       key == "stop_token_ids") {
                if (!parameter.is_null()) {
                    sampling.emplace(key, parameter);
                }
            } else {
                throw std::invalid_argument("Unsupported /generate sampling parameter: " + key);
            }
        }
    }
    set_sampling(sampling, options, "sampling_seed");
    set_stops(sampling, options);
    if (salt && !salt->is_null()) {
        set_cache_salt_value(*salt, options);
    }
    return options;
}

BatchPrompt generate_prompt(const json::Value& input, bool text, DecoderOptions options,
                            const CompletionTokenizer& tokenizer) {
    BatchPrompt request{{}, std::move(options)};
    if (text) {
        const auto prompt = input.string().value_or("");
        if (prompt.empty() || prompt.size() > 1024 * 1024) {
            bad_request("Expected one nonempty text string of up to 1048576 bytes");
        }
        encode_prompt(request.prompt, request.options.max_new_tokens, prompt, tokenizer);
    } else {
        const auto* tokens = input.array();
        if (!tokens || tokens->empty()) {
            bad_request("Expected one nonempty flat array of input_ids");
        }
        if (tokens->size() > checkpoint_context - request.options.max_new_tokens) {
            bad_request("Prompt and output budget must fit the 262144-token context");
        }
        request.prompt.reserve(tokens->size());
        for (const auto& value : *tokens) {
            const auto token = value.unsigned_integer();
            if (!token || *token >= model_vocab_size) {
                bad_request("input_ids must contain integers in [0, 248320)");
            }
            request.prompt.push_back(static_cast<int>(*token));
        }
    }
    return request;
}

const json::Array* generate_parameter_list(const json::Value* value, std::size_t count,
                                           bool strings) {
    const auto* list = value ? value->array() : nullptr;
    if (list && (list->size() != count ||
        std::any_of(list->begin(), list->end(), [&](const auto& item) {
            return strings ? !item.string() : !item.object();
        }))) {
        bad_request(strings ? "Expected one cache_salt string per batch item"
                            : "Expected one sampling_params object per batch item");
    }
    return list;
}

}

GenerateRequests parse_generate_requests(std::string_view body,
                                         const CompletionTokenizer& tokenizer) {
    const auto parsed = parse_body(body, json::Syntax::strict);
    const auto* object = parsed.object();
    if (!object) {
        bad_request("Expected a JSON object for /generate");
    }
    validate_generate_fields(*object);
    GenerateRequests requests;
    if (const auto* stream = json::get(*object, "stream"); stream && !stream->is_null()) {
        requests.stream = read_bool(*object, "stream");
    }
    const auto* text = json::get(*object, "text");
    const auto* ids = json::get(*object, "input_ids");
    const bool has_text = text && !text->is_null();
    const bool has_ids = ids && !ids->is_null();
    if (has_text == has_ids) {
        bad_request("Provide exactly one of text or input_ids");
    }
    const auto& input = has_text ? *text : *ids;
    const auto* list = input.array();
    requests.batch = list && (has_text || (!list->empty() && list->front().array()));
    const auto* params = json::get(*object, "sampling_params");
    const auto* salt = json::get(*object, "cache_salt");
    if (!requests.batch) {
        requests.items.push_back(generate_prompt(input, has_text,
            generate_options(params, salt), tokenizer));
        return requests;
    }
    if (requests.stream) {
        bad_request("Only non-streaming /generate batches are supported");
    }
    if (list->empty() || list->size() > decoder_batch_max_lanes) {
        bad_request("Expected 1 to 32 /generate batch items");
    }
    const auto* param_list = generate_parameter_list(params, list->size(), false);
    const auto* salt_list = generate_parameter_list(salt, list->size(), true);
    requests.items.reserve(list->size());
    int remaining = checkpoint_context;
    for (std::size_t index = 0; index < list->size(); ++index) {
        auto options = generate_options(param_list ? &(*param_list)[index] : params,
                                        salt_list ? &(*salt_list)[index] : salt);
        remaining -= options.max_new_tokens;
        if (remaining < 0) {
            bad_request("Batch output budgets must total at most 262144 tokens");
        }
        requests.items.push_back(generate_prompt((*list)[index], has_text,
                                                 std::move(options), tokenizer));
    }
    return requests;
}

void set_cache_salt(const json::Object& object, DecoderOptions& options) {
    if (const auto* salt = json::get(object, "cache_salt")) {
        set_cache_salt_value(*salt, options);
    }
}

TextRequest parse_chat_request(std::string_view body, const CompletionTokenizer& tokenizer,
                               bool media_enabled) {
    const auto parsed = parse_body(body, json::Syntax::ordered);
    const auto* chat = parsed.ordered_object();
    if (!chat) {
        bad_request("Expected a JSON object for a chat request");
    }
    auto media = parse_chat_media(*chat, media_enabled);
    auto normalized = media ? std::move(media->chat) : *chat;
    normalize_template_options(normalized);
    const auto options = chat_text_options(normalized, media.has_value());
    auto request = text_options(options, tokenizer, true);
    const auto tools = chat_tools(normalized);
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
    set_chat_output(request, normalized, prompt, tokenizer, tools);
    return request;
}

}
