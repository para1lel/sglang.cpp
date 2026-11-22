#include "sglang/server/decisions.h"

#include "sglang/server/chat_media.h"
#include "sglang/server/chat_template.h"
#include "sglang/server/http.h"
#include "sglang/server/tokenization_validation.h"
#include "sglang/server/vision_completion.h"
#include "sglang/tokenizer/unicode.h"
#include "sglang/utf8.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <set>

namespace sglang::server {
namespace {

using json::Array;
using json::get;
using json::OrderedObject;
using json::Value;
using namespace score_detail;
using namespace std::string_view_literals;

constexpr auto version = 1;

struct Question {
    std::string id;
    std::string kind;
    std::string text;
    std::vector<std::string> names;
    std::vector<std::string> details;
    std::vector<std::string> labels;
    OrderedObject legend;
};

struct Request {
    std::string model = "default";
    std::string input;
    Array image_parts;
    std::vector<Question> questions;
    double temperature = 1;
    bool return_ids = false;
    bool vision_ids = false;
};

[[noreturn]] void invalid(std::string message) {
    throw ScoreValidationError(std::move(message));
}

const OrderedObject& object(const Value* value, std::string_view name) {
    if (value && value->ordered_object()) return *value->ordered_object();
    invalid(std::string(name) + " must be an object");
}

void fields(const OrderedObject& value, std::initializer_list<std::string_view> allowed) {
    for (const auto& [name, item] : value) {
        if (std::find(allowed.begin(), allowed.end(), name) == allowed.end()) {
            invalid("Extra inputs are not permitted: " + name);
        }
    }
}

const Array& list(const Value* value, std::string_view name, std::size_t min, std::size_t max) {
    if (value && value->array() && value->array()->size() >= min &&
        value->array()->size() <= max) return *value->array();
    invalid(std::string(name) + " has an invalid number of items or is not an array");
}

bool whitespace(char32_t point) {
    return (point >= 9 && point <= 13) || (point >= 0x1c && point <= 0x20) ||
        point == 0x85 || point == 0xa0 || point == 0x1680 ||
        (point >= 0x2000 && point <= 0x200a) || point == 0x2028 || point == 0x2029 ||
        point == 0x202f || point == 0x205f || point == 0x3000;
}

std::u32string_view stripped(std::u32string_view value) {
    while (!value.empty() && whitespace(value.front())) value.remove_prefix(1);
    while (!value.empty() && whitespace(value.back())) value.remove_suffix(1);
    return value;
}

bool version_one_string(std::string_view value) {
    const auto scalars = read_utf8(value);
    constexpr auto separators = U"\t\n\v\f\r \u0085\u00a0\u1680\u2000\u2001\u2002\u2003"
                                U"\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u2028\u2029"
                                U"\u202f\u205f\u3000"sv;
    const auto start = scalars.find_first_not_of(separators);
    if (start == std::u32string::npos) return false;
    const auto end = scalars.find_last_not_of(separators);
    auto body = std::u32string_view(scalars).substr(start, end - start + 1);
    if (body.front() == U'+') body.remove_prefix(1);
    const auto dot = body.find(U'.');
    const auto integer = body.substr(0, dot);
    if (integer.empty() || integer.back() != U'1' || integer.front() == U'_') return false;
    const auto leading = integer.substr(0, integer.size() - 1);
    return leading.find_first_not_of(U"0_") == std::u32string_view::npos &&
        (dot == std::u32string_view::npos ||
         (dot + 1 < body.size() && body.find_first_not_of(U'0', dot + 1) == body.npos));
}

bool supported_version(const Value& value) {
    if (value.is_null()) return true;
    if (const auto field = value.string()) return version_one_string(*field);
    return value.number() == version || value.boolean().value_or(false);
}

std::string string(const Value* value, std::string_view name, bool nonblank = false) {
    if (const auto text = value ? value->string() : std::nullopt;
        text && (!nonblank || !stripped(read_utf8(*text)).empty())) return std::string(*text);
    invalid(std::string(name) + (nonblank ? " must be a nonblank string" : " must be a string"));
}

std::string text(const Value* value, std::string_view name, bool required = true) {
    if (!required && (!value || value->is_null())) return {};
    if (value) {
        if (const auto plain = value->string();
            plain && (!required || !stripped(read_utf8(*plain)).empty())) {
            return std::string(*plain);
        }
        if (const auto* items = value->array(); items && (!required || !items->empty())) {
            return chat_template_json(*value, true);
        }
        if (const auto* members = value->ordered_object();
            members && (!required || !members->empty())) return chat_template_json(*value, true);
    }
    invalid(std::string(name) + (required ? " must be nonblank text, an object, or an array"
                                         : " must be text, an object, or an array"));
}

void option_name(std::string_view name, std::set<std::u32string>& seen) {
    const auto scalars = read_utf8(name);
    const auto key = stripped(scalars);
    if (key.empty()) invalid("option names must be nonempty");
    if (std::any_of(scalars.begin(), scalars.end(), [](char32_t point) {
        return point < 0x20 || (point >= 0x7f && point <= 0x9f) ||
            point == 0x2028 || point == 0x2029;
    })) invalid("option names must not contain control or line break characters");
    if (!seen.insert(tokenizer::casefold(key)).second) {
        invalid("option name repeats another option");
    }
}

void choice_options(const Value* value, Question& output) {
    std::set<std::u32string> seen;
    for (const auto& item : list(value, "options", 2, 26)) {
        const auto& option = object(&item, "option");
        fields(option, {"name", "description"});
        auto name = string(get(option, "name"), "option.name");
        option_name(name, seen);
        output.labels.emplace_back(1, static_cast<char>('A' + output.names.size()));
        output.names.push_back(std::move(name));
        output.details.push_back(text(get(option, "description"), "description", false));
    }
}

Question parse_question(const Value& value) {
    const auto& question = object(&value, "question");
    Question output;
    output.id = string(get(question, "id"), "id", true);
    output.kind = string(get(question, "type"), "type");
    output.text = text(get(question, "question"), "question");
    if (output.kind == "choice") {
        fields(question, {"id", "type", "question", "options"});
        choice_options(get(question, "options"), output);
    } else if (output.kind == "score") {
        fields(question, {"id", "type", "question", "levels"});
        for (const auto& item : list(get(question, "levels"), "levels", 2, 10)) {
            output.names.push_back(std::to_string(output.names.size()));
            output.details.push_back(text(&item, "level"));
        }
        output.labels = output.names;
    } else if (output.kind == "yes_no") {
        fields(question, {"id", "type", "question", "yes", "no"});
        output.names = output.labels = {"yes", "no"};
        output.details = {text(get(question, "yes"), "yes", false),
                          text(get(question, "no"), "no", false)};
    } else {
        invalid("type must be choice, score, or yes_no");
    }
    return output;
}

void parse_template(const Value* value, Request& output) {
    if (!value) return;
    const auto& arguments = object(value, "chat_template_kwargs");
    if (const auto* thinking = get(arguments, "enable_thinking");
        thinking && thinking->boolean() != false) {
        throw std::invalid_argument("decisions need enable_thinking false or unset");
    }
    for (const auto key : {"messages", "tokenize", "add_generation_prompt", "tools"}) {
        if (get(arguments, key)) {
            throw std::invalid_argument(std::string("Unsupported chat_template_kwargs: ") + key);
        }
    }
    if (const auto* vision = get(arguments, "add_vision_id")) {
        output.vision_ids = chat_truthy(*vision);
    }
}

Array parse_images(const Value* value) {
    Array images;
    if (!value) return images;
    for (const auto& image : list(value, "images", 0, 4)) {
        OrderedObject payload;
        if (const auto url = image.string()) {
            payload.emplace_back("url", std::string(*url));
        } else {
            payload = object(&image, "image");
            string(get(payload, "url"), "image.url");
            for (const auto name : {"max_dynamic_patch", "min_dynamic_patch",
                                    "max_long_side_pixel", "content_hash"}) {
                if (const auto* option = get(payload, name); option && !option->is_null()) {
                    throw std::invalid_argument("Unsupported decision image option: " +
                                                std::string(name));
                }
            }
            if (const auto* detail = get(payload, "detail");
                detail && !detail->is_null() && detail->string() != "auto" &&
                detail->string() != "default") {
                if (detail->string() != "low" && detail->string() != "high") {
                    invalid("image.detail must be auto, low, high, or null");
                }
                throw std::invalid_argument("Decision images support automatic detail only");
            }
            const auto unsupported = std::remove_if(payload.begin(), payload.end(),
                                                    [](const auto& field) {
                return field.first != "url" && field.first != "height" &&
                       field.first != "width";
            });
            payload.erase(unsupported, payload.end());
        }
        images.emplace_back(OrderedObject{{"type", "image_url"},
                                          {"image_url", std::move(payload)}});
    }
    return images;
}

Request parse_request(std::string_view body) {
    const auto root = json::parse(body, json::Syntax::ordered_last_key_wins);
    const auto& request = object(&root, "request");
    fields(request, {"input", "images", "questions", "temperature", "chat_template_kwargs",
                     "prompt_format_version", "return_prompt_token_ids", "model"});
    Request output;
    output.input = text(get(request, "input"), "input");
    if (const auto* model = get(request, "model")) output.model = string(model, "model");
    if (const auto* scale = get(request, "temperature")) {
        const auto number = scale->number();
        if (!number || !std::isfinite(*number) || *number <= 0) {
            invalid("temperature must be finite and greater than zero");
        }
        output.temperature = *number;
    }
    if (const auto* flag = get(request, "return_prompt_token_ids")) {
        if (!flag->boolean()) invalid("return_prompt_token_ids must be a boolean");
        output.return_ids = *flag->boolean();
    }
    const auto& questions = list(get(request, "questions"), "questions", 1, max_score_items);
    std::set<std::string_view> ids;
    output.questions.reserve(questions.size());
    for (const auto& item : questions) output.questions.push_back(parse_question(item));
    for (const auto& question : output.questions) {
        if (!ids.insert(question.id).second) invalid("question id repeats another question");
    }
    output.image_parts = parse_images(get(request, "images"));
    if (!output.image_parts.empty() && output.return_ids) {
        throw std::invalid_argument(
            "return_prompt_token_ids cannot replay image inputs through /v1/score");
    }
    if (const auto* format = get(request, "prompt_format_version");
        format && !supported_version(*format)) {
        throw std::invalid_argument(
            "prompt_format_version is not served, this server uses version 1");
    }
    parse_template(get(request, "chat_template_kwargs"), output);
    return output;
}

bool blank(const Value* value) {
    return !value || value->is_null() ||
        (value->string() && stripped(read_utf8(*value->string())).empty()) ||
        (value->array() && value->array()->empty()) ||
        (value->ordered_object() && value->ordered_object()->empty());
}

Question parse_systemone_question(std::string_view id, const Value& value) {
    const auto& fields_value = object(&value, "question");
    fields(fields_value, {"type", "instructions", "criteria"});
    Question question;
    question.id = id;
    question.kind = string(get(fields_value, "type"), "type");
    const auto* instructions = get(fields_value, "instructions");
    const auto instruction_text = text(instructions, "instructions", false);
    if (!blank(instructions)) question.text = instruction_text;
    const auto* criteria = get(fields_value, "criteria");
    if (question.kind == "choice") {
        const auto& options = object(criteria, "criteria");
        if (options.empty() || options.size() > 255) {
            invalid("choice criteria must contain between 1 and 255 options");
        }
        std::set<std::u32string> seen;
        for (const auto& [name, detail] : options) {
            option_name(name, seen);
            question.names.push_back(name);
            question.details.push_back(text(&detail, "criterion", false));
        }
        if (options.size() <= 26) {
            for (std::size_t index = 0; index < options.size(); ++index) {
                question.labels.emplace_back(1, static_cast<char>('A' + index));
            }
        }
    } else if (question.kind == "score") {
        for (const auto& level : list(criteria, "criteria", 1, 10)) {
            auto index = std::to_string(question.names.size());
            question.names.push_back(index);
            question.details.push_back(text(&level, "criterion"));
            question.legend.emplace_back(std::move(index), level);
        }
        question.labels = question.names;
    } else if (question.kind == "noul") {
        const OrderedObject empty;
        const auto& conditions = !criteria || criteria->is_null()
            ? empty : object(criteria, "criteria");
        fields(conditions, {"true", "false"});
        const auto* positive = get(conditions, "true");
        const auto* negative = get(conditions, "false");
        question.details = {text(positive, "true", false), text(negative, "false", false)};
        if (blank(instructions) && blank(positive) && blank(negative)) {
            invalid("a noul question needs instructions or a true or false description");
        }
        question.kind = "yes_no";
        question.names = question.labels = {"yes", "no"};
    } else {
        invalid("type must be choice, score, or noul");
    }
    return question;
}

Request parse_systemone(std::string_view body, std::string_view model) {
    const auto root = json::parse(body, json::Syntax::ordered_last_key_wins);
    const auto& request = object(&root, "request");
    for (const auto key : {"temperature", "prompt_format_version", "return_prompt_token_ids"}) {
        if (const auto* field = get(request, key); field && !field->is_null()) {
            invalid(std::string(key) + " is not part of this API, use /v1/decisions for it");
        }
    }
    string(get(request, "model"), "model");
    const auto* state = get(request, "state");
    if (!state || state->is_null()) invalid("state must be text, an object, or an array");
    Request output;
    output.model = model;
    output.input = text(state, "state", false);
    const auto& questions = object(get(request, "questions"), "questions");
    if (questions.empty() || questions.size() > max_score_items) {
        invalid("questions has an invalid number of entries");
    }
    output.questions.reserve(questions.size());
    for (const auto& [id, question] : questions) {
        try {
            output.questions.push_back(parse_systemone_question(id, question));
        } catch (const ScoreValidationError& error) {
            invalid("question '" + id + "': " + error.what());
        }
    }
    if (const auto* images = get(request, "images")) {
        if (!images->array()) invalid("images must be an array");
        if (!images->array()->empty()) {
            throw std::invalid_argument("Vision decision inputs are unavailable");
        }
    }
    parse_template(get(request, "chat_template_kwargs"), output);
    return output;
}

std::string render_question(std::string_view input, const Question& question) {
    std::string output(input);
    output += "\n\n";
    if (question.kind == "yes_no") {
        output += "Is the following true?";
        if (!question.text.empty()) output += " " + question.text;
        output += '\n';
    } else if (!question.text.empty()) {
        output += "Question: " + question.text + '\n';
    }
    for (std::size_t index = 0; index < question.labels.size(); ++index) {
        const auto& detail = question.details[index];
        if (question.kind == "yes_no" && detail.empty()) continue;
        output += question.labels[index] + ": ";
        if (question.kind == "choice") {
            output += question.names[index];
            if (!detail.empty()) output += " - " + detail;
        } else {
            output += detail;
        }
        output += '\n';
    }
    if (question.kind == "choice") output += "Answer with the letter of one option only.";
    else if (question.kind == "score") output += "Answer with the number of one level only.";
    else output += "Answer with yes or no only.";
    return output;
}

std::string render_prompt(const CompletionTokenizer& tokenizer, Value content,
                          bool vision_ids) {
    Array messages{OrderedObject{{"role", "user"}, {"content", std::move(content)}}};
    return tokenizer.chat_template().render(
        {{"messages", std::move(messages)}, {"enable_thinking", false},
         {"add_generation_prompt", true}, {"add_vision_id", vision_ids}}, ChatInput::http);
}

constexpr auto label_suffix = "\n\n"sv;

bool label_suffix_matches(const Tokenizer& tokenizer, std::string_view prompt,
                          const std::vector<TokenId>& ids,
                          const std::vector<TokenId>& suffix_ids) {
    constexpr auto terminator = "</think>"sv;
    const auto marker_ids = tokenizer.encode(terminator);
    return prompt.size() >= terminator.size() + label_suffix.size() &&
        prompt.substr(prompt.size() - label_suffix.size() - terminator.size()) == "</think>\n\n" &&
        marker_ids.size() == 1 && marker_ids.front() >= checkpoint_base_vocab_size &&
        ids.size() >= 1 + suffix_ids.size() &&
        ids[ids.size() - suffix_ids.size() - 1] == marker_ids.front() &&
        std::equal(suffix_ids.rbegin(), suffix_ids.rend(), ids.rbegin());
}

std::optional<int> encode_label(const Tokenizer& tokenizer, std::string_view base,
                                const std::vector<TokenId>& ids, const std::string& label) {
    const auto candidate = tokenizer.encode(std::string(base) + label, ids.size() + 2);
    if (candidate.size() != ids.size() + 1 ||
        !std::equal(ids.begin(), ids.end(), candidate.begin()) ||
        candidate.back() >= checkpoint_base_vocab_size) return std::nullopt;
    return static_cast<int>(candidate.back());
}

std::vector<int> encode_labels(const Tokenizer& tokenizer, std::string_view prompt,
                               const std::vector<TokenId>& ids, const Question& question) {
    const auto suffix_ids = tokenizer.encode(label_suffix);
    const bool split = label_suffix_matches(tokenizer, prompt, ids, suffix_ids);
    const auto& base_ids = split ? suffix_ids : ids;
    const auto base_text = split ? label_suffix : prompt;
    std::vector<int> labels;
    labels.reserve(question.labels.size());
    for (const auto& label : question.labels) {
        const auto candidate = encode_label(tokenizer, base_text, base_ids, label);
        if (!candidate || std::find(labels.begin(), labels.end(), *candidate) != labels.end()) {
            throw std::invalid_argument("question '" + question.id + "': the answer label '" +
                label + "' is not one distinct token after the chat prompt");
        }
        labels.push_back(*candidate);
    }
    return labels;
}

std::vector<std::string> pair_labels(const CompletionTokenizer& completion, bool vision_ids) {
    const auto& tokenizer = completion.get();
    const auto suffix_ids = tokenizer.encode(label_suffix);
    for (const auto content : {"x", "y"}) {
        const auto prompt = render_prompt(completion, content, vision_ids);
        if (!label_suffix_matches(tokenizer, prompt, tokenizer.encode(prompt), suffix_ids)) {
            throw std::invalid_argument("More than 26 options require a stable chat prompt suffix");
        }
    }
    std::set<int> seen;
    std::vector<std::string> labels;
    for (const char first : "ABCDEFGHIJKLMNOPQRSTUVWXYZ"sv) {
        for (const char second : "ABCDEFGHIJKLMNOPQRSTUVWXYZ"sv) {
            std::string label{first, second};
            const auto id = encode_label(tokenizer, label_suffix, suffix_ids, label);
            if (id && seen.insert(*id).second) labels.push_back(std::move(label));
        }
    }
    return labels;
}

std::vector<BatchPrompt> prepare(Request& request, const CompletionTokenizer& completion,
                                 const BatchCallbacks& callbacks, VisionCompletion& vision) {
    auto bytes = max_score_text_bytes;
    auto tokens = max_score_prompt_tokens;
    auto labels = max_score_response_values;
    if (request.input.size() > bytes / request.questions.size()) {
        throw std::length_error("Decision request exceeds the aggregate text byte budget");
    }
    std::vector<BatchPrompt> batch;
    batch.reserve(request.questions.size());
    const auto& tokenizer = completion.get();
    MediaPrompt media;
    if (!request.image_parts.empty()) {
        if (callbacks.is_cancelled()) return {};
        const OrderedObject chat{{"messages",
            Array{OrderedObject{{"role", "user"}, {"content", std::move(request.image_parts)}}}}};
        media.parts = prepare_chat_media(parse_chat_media(chat, true).value().parts, tokenizer);
    }
    std::optional<std::vector<std::string>> pairs;
    for (auto& question : request.questions) {
        if (callbacks.is_cancelled()) return {};
        if (question.labels.empty()) {
            if (!pairs) pairs = pair_labels(completion, request.vision_ids);
            if (question.names.size() > pairs->size()) {
                throw std::invalid_argument("Too many options for distinct two-letter labels");
            }
            question.labels.assign(pairs->begin(), pairs->begin() + question.names.size());
        }
        Value content = render_question(request.input, question);
        if (!media.parts.empty()) {
            Array blocks(media.parts.size(), OrderedObject{{"type", "image"}});
            blocks.emplace_back(OrderedObject{{"type", "text"}, {"text", std::move(content)}});
            content = std::move(blocks);
        }
        const auto prompt = render_prompt(completion, std::move(content), request.vision_ids);
        if (!consume(bytes, prompt.size()) || !consume(labels, question.labels.size())) {
            throw std::length_error("Decision request exceeds the aggregate score budget");
        }
        const auto ids = tokenizer.encode(prompt, std::min(tokens, checkpoint_context - 1));
        std::vector<int> prompt_ids(ids.begin(), ids.end());
        const auto expanded = media.parts.empty()
            ? prompt_ids.size() : expand_prepared_chat_media(prompt_ids, media.parts).size();
        if (expanded >= checkpoint_context || !consume(tokens, expanded)) {
            throw std::length_error("Decision request exceeds the aggregate prompt token budget");
        }
        DecoderOptions options;
        options.max_new_tokens = 0;
        options.label_token_ids = std::make_shared<const std::vector<int>>(
            encode_labels(tokenizer, prompt, ids, question));
        batch.push_back({std::move(prompt_ids), std::move(options)});
        batch.back().expanded_tokens = expanded;
    }
    if (!media.parts.empty()) {
        DecoderCallbacks media_callbacks;
        media_callbacks.is_cancelled = callbacks.is_cancelled;
        const auto features = vision.encode(media, media_callbacks);
        if (!features) return {};
        for (auto& prompt : batch) prompt.media = *features;
    }
    return batch;
}

Array id_array(const std::vector<int>& ids) {
    return Array(ids.begin(), ids.end());
}

Value answer(const Request& request, const Question& question, const BatchPrompt& prompt,
             const DecoderResult& result) {
    if (!result.label_logprobs || result.label_logprobs->size() != question.labels.size()) {
        throw std::logic_error("Missing decision result");
    }
    const auto& logs = *result.label_logprobs;
    const auto values = probabilities(logs, request.temperature, true);
    long double mass = 0;
    long double weighted = 0;
    OrderedObject options;
    for (std::size_t index = 0; index < values.size(); ++index) {
        mass += std::exp(logs[index]);
        weighted += index * values[index];
        options.emplace_back(question.names[index], values[index]);
    }
    OrderedObject output{{"type", question.kind}, {"probabilities", std::move(options)},
                         {"label_mass", static_cast<double>(mass)}};
    if (question.kind == "choice") {
        const auto best = std::max_element(values.begin(), values.end()) - values.begin();
        output.emplace_back("choice", question.names[best]);
    } else if (question.kind == "score") {
        output.emplace_back("score", static_cast<double>(weighted));
    }
    if (request.return_ids) {
        output.emplace_back("prompt_token_ids", id_array(prompt.prompt));
        output.emplace_back("label_token_ids", id_array(*prompt.options.label_token_ids));
    }
    return output;
}

std::string response(const Request& request, const std::vector<BatchPrompt>& prompts,
                     const std::vector<DecoderResult>& results) {
    if (results.size() != prompts.size()) throw std::logic_error("Incomplete decision batch");
    OrderedObject answers;
    std::uint64_t tokens = 0;
    for (std::size_t index = 0; index < prompts.size(); ++index) {
        const auto& question = request.questions[index];
        answers.emplace_back(question.id,
                             answer(request, question, prompts[index], results[index]));
        tokens += prompts[index].prompt_tokens();
    }
    OrderedObject usage{{"prompt_tokens", tokens}, {"total_tokens", tokens},
                        {"completion_tokens", 0}, {"reasoning_tokens", 0}};
    return chat_template_json(OrderedObject{{"object", "decisions"}, {"model", request.model},
        {"prompt_format_version", version}, {"answers", std::move(answers)},
        {"usage", std::move(usage)}}, true);
}

Value systemone_answer(const Question& question, const DecoderResult& result) {
    if (!result.label_logprobs || result.label_logprobs->size() != question.labels.size()) {
        throw std::logic_error("Missing System One result");
    }
    const auto& logs = *result.label_logprobs;
    const auto values = probabilities(logs, 1, true);
    long double mass = 0;
    for (const auto logprob : logs) mass += std::exp(logprob);
    if (question.kind == "yes_no") {
        return OrderedObject{{"type", "noul"}, {"noul", values.front()},
                             {"x_label_mass", static_cast<double>(mass)}};
    }
    const auto best = std::max_element(values.begin(), values.end()) - values.begin();
    const auto count = values.size();
    OrderedObject options;
    long double total = 0;
    long double score = 0;
    long double spread = 0;
    long double uniform_spread = 0;
    for (std::size_t index = 0; index < count; ++index) {
        const auto value = values[index];
        options.emplace_back(question.names[index], value);
        total += value;
        score += index * value;
        spread += value * std::abs(static_cast<double>(index) - best);
        uniform_spread += std::abs(static_cast<double>(index) - (count - 1) / 2.0);
    }
    OrderedObject output{{"type", question.kind}, {"probabilities", std::move(options)},
                         {"x_label_mass", static_cast<double>(mass)}};
    double confidence = 1;
    if (question.kind == "choice") {
        output.emplace_back("choice", question.names[best]);
        if (count > 1) {
            confidence = std::clamp(static_cast<double>(
                (count * (values[best] / total) - 1) / (count - 1)), 0.0, 1.0);
        }
    } else {
        output.emplace_back("score", static_cast<double>(score));
        output.emplace_back("legend", question.legend);
        if (count > 1) {
            confidence = std::max(0.0, static_cast<double>(
                1 - (spread / total) / (uniform_spread / count)));
        }
    }
    output.emplace_back("confidence", confidence);
    return output;
}

std::string systemone_response(const Request& request, const std::vector<BatchPrompt>& prompts,
                               const std::vector<DecoderResult>& results) {
    if (results.size() != prompts.size()) throw std::logic_error("Incomplete System One batch");
    OrderedObject answers;
    std::uint64_t tokens = 0;
    for (std::size_t index = 0; index < prompts.size(); ++index) {
        const auto& question = request.questions[index];
        answers.emplace_back(question.id, systemone_answer(question, results[index]));
        tokens += prompts[index].prompt.size();
    }
    return chat_template_json(OrderedObject{
        {"model", request.model}, {"answers", std::move(answers)},
        {"usage", OrderedObject{{"input_tokens", tokens}, {"output_tokens", 0}}}}, true);
}

void schema_error(const Socket& connection, bool systemone, std::string_view message) {
    if (!systemone) {
        write_tokenization_error(connection, "400 Bad Request", "Bad Request", message);
        return;
    }
    OrderedObject error{{"type", "value_error"}, {"loc", Array{"body"}},
                        {"msg", std::string(message)}};
    write_json(connection, "422 Unprocessable Entity",
               chat_template_json(OrderedObject{{"detail", Array{std::move(error)}}}, true));
}

void serve(const Socket& connection, BatchService& service, int slot,
           const CompletionTokenizer& tokenizer, std::string_view body,
           std::optional<std::string_view> model, VisionCompletion& vision) {
    try {
        auto request = model ? parse_systemone(body, *model) : parse_request(body);
        BatchCallbacks callbacks;
        callbacks.is_cancelled = [&] { return disconnected(connection); };
        auto prompts = prepare(request, tokenizer, callbacks, vision);
        if (prompts.empty()) return;
        const auto results = service.generate_batch(slot, prompts, callbacks);
        if (results) {
            write_json(connection, "200 OK", model ? systemone_response(request, prompts, *results)
                                                   : response(request, prompts, *results));
        }
    } catch (const ScoreValidationError& error) {
        schema_error(connection, model.has_value(), error.what());
    } catch (const json::ParseError& error) {
        schema_error(connection, model.has_value(), error.what());
    } catch (const std::invalid_argument& error) {
        write_tokenization_error(connection, "400 Bad Request", "BadRequestError", error.what());
    } catch (const std::length_error& error) {
        write_tokenization_error(connection, "400 Bad Request", "BadRequestError", error.what());
    } catch (const std::exception&) {
        write_tokenization_error(connection, "500 Internal Server Error", "InternalServerError",
                                 "Decision request failed");
    }
}

}

void serve_decisions(const Socket& connection, BatchService& service, int slot,
                     const CompletionTokenizer& tokenizer, std::string_view body,
                     VisionCompletion& vision) {
    serve(connection, service, slot, tokenizer, body, std::nullopt, vision);
}

void serve_systemone(const Socket& connection, BatchService& service, int slot,
                     const CompletionTokenizer& tokenizer, std::string_view body,
                     std::string_view model, VisionCompletion& vision) {
    serve(connection, service, slot, tokenizer, body, model, vision);
}

}
