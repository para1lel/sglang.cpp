#include "sglang/server/decisions.h"

#include "sglang/server/chat_template.h"
#include "sglang/server/http.h"
#include "sglang/server/tokenization_validation.h"
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
};

struct Request {
    std::string model = "default";
    std::string input;
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
    if (const auto* images = get(request, "images")) {
        if (!images->array()) invalid("images must be an array");
        if (!images->array()->empty()) {
            throw std::invalid_argument("Vision decision inputs are unavailable");
        }
    }
    if (const auto* format = get(request, "prompt_format_version");
        format && !supported_version(*format)) {
        throw std::invalid_argument(
            "prompt_format_version is not served, this server uses version 1");
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

std::string render_prompt(const CompletionTokenizer& tokenizer, const Request& request,
                          const Question& question) {
    Array messages{OrderedObject{{"role", "user"},
                                 {"content", render_question(request.input, question)}}};
    return tokenizer.chat_template().render(
        {{"messages", std::move(messages)}, {"enable_thinking", false},
         {"add_generation_prompt", true}, {"add_vision_id", request.vision_ids}}, ChatInput::http);
}

std::vector<int> encode_labels(const Tokenizer& tokenizer, std::string_view prompt,
                               const std::vector<TokenId>& ids, const Question& question) {
    constexpr auto terminator = "</think>"sv;
    constexpr auto suffix = "\n\n"sv;
    const auto marker_ids = tokenizer.encode(terminator);
    const auto suffix_ids = tokenizer.encode(suffix);
    const bool split = prompt.size() >= terminator.size() + suffix.size() &&
        prompt.substr(prompt.size() - suffix.size() - terminator.size()) == "</think>\n\n" &&
        marker_ids.size() == 1 && marker_ids.front() >= checkpoint_base_vocab_size &&
        ids.size() >= 1 + suffix_ids.size() &&
        ids[ids.size() - suffix_ids.size() - 1] == marker_ids.front() &&
        std::equal(suffix_ids.rbegin(), suffix_ids.rend(), ids.rbegin());
    const auto& base_ids = split ? suffix_ids : ids;
    const auto base_text = split ? suffix : prompt;
    std::vector<int> labels;
    labels.reserve(question.labels.size());
    for (const auto& label : question.labels) {
        auto candidate = tokenizer.encode(std::string(base_text) + label, base_ids.size() + 2);
        if (candidate.size() != base_ids.size() + 1 ||
            !std::equal(base_ids.begin(), base_ids.end(), candidate.begin()) ||
            candidate.back() >= checkpoint_base_vocab_size ||
            std::find(labels.begin(), labels.end(), static_cast<int>(candidate.back()))
                != labels.end()) {
            throw std::invalid_argument("question '" + question.id + "': the answer label '" +
                label + "' is not one distinct token after the chat prompt");
        }
        labels.push_back(static_cast<int>(candidate.back()));
    }
    return labels;
}

std::vector<BatchPrompt> prepare(const Request& request, const CompletionTokenizer& completion,
                                 const BatchCallbacks& callbacks) {
    auto bytes = max_score_text_bytes;
    auto tokens = max_score_prompt_tokens;
    auto labels = max_score_response_values;
    if (request.input.size() > bytes / request.questions.size()) {
        throw std::length_error("Decision request exceeds the aggregate text byte budget");
    }
    std::vector<BatchPrompt> batch;
    batch.reserve(request.questions.size());
    const auto& tokenizer = completion.get();
    for (const auto& question : request.questions) {
        if (callbacks.is_cancelled()) return {};
        const auto prompt = render_prompt(completion, request, question);
        if (!consume(bytes, prompt.size()) || !consume(labels, question.labels.size())) {
            throw std::length_error("Decision request exceeds the aggregate score budget");
        }
        auto ids = tokenizer.encode(prompt, std::min(tokens, checkpoint_context - 1));
        if (!consume(tokens, ids.size())) {
            throw std::length_error("Decision request exceeds the aggregate prompt token budget");
        }
        DecoderOptions options;
        options.max_new_tokens = 0;
        options.label_token_ids = std::make_shared<const std::vector<int>>(
            encode_labels(tokenizer, prompt, ids, question));
        batch.push_back({std::vector<int>(ids.begin(), ids.end()), std::move(options)});
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
        tokens += prompts[index].prompt.size();
    }
    OrderedObject usage{{"prompt_tokens", tokens}, {"total_tokens", tokens},
                        {"completion_tokens", 0}, {"reasoning_tokens", 0}};
    return chat_template_json(OrderedObject{{"object", "decisions"}, {"model", request.model},
        {"prompt_format_version", version}, {"answers", std::move(answers)},
        {"usage", std::move(usage)}}, true);
}

}

void serve_decisions(const Socket& connection, BatchService& service, int slot,
                     const CompletionTokenizer& tokenizer, std::string_view body) {
    try {
        const auto request = parse_request(body);
        BatchCallbacks callbacks;
        callbacks.is_cancelled = [&] { return disconnected(connection); };
        auto prompts = prepare(request, tokenizer, callbacks);
        if (prompts.empty()) return;
        const auto results = service.generate_batch(slot, prompts, callbacks);
        if (results) write_json(connection, "200 OK", response(request, prompts, *results));
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
                                 "Decision request failed");
    }
}

}
