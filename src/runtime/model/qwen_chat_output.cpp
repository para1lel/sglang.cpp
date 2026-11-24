#include "sglang/runtime/model/qwen_parser.h"
#include "sglang/runtime/model/qwen_serving_profile.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <list>

namespace sglang::server {
namespace {

using namespace std::string_view_literals;
using Args = std::vector<ChatArgumentType>;
using Type = ChatArgumentType;

constexpr auto think_open = "<think>"sv;
constexpr auto think_close = "</think>"sv;
constexpr auto tool_open = "<tool_call>"sv;
constexpr auto tool_close = "</tool_call>"sv;
constexpr auto function_open = "<function="sv;
constexpr auto function_close = "</function>"sv;
constexpr auto parameter_open = "<parameter="sv;
constexpr auto parameter_close = "</parameter>"sv;
constexpr auto whitespace = " \t\r\n"sv;

bool starts_with(std::string_view text, std::string_view prefix) {
    return text.substr(0, prefix.size()) == prefix;
}

std::string_view trim(std::string_view text) {
    const auto first = text.find_first_not_of(whitespace);
    return first == text.npos ? std::string_view{} :
        text.substr(first, text.find_last_not_of(whitespace) + 1 - first);
}

const json::Value* field(const json::Value& value, std::string_view key) {
    const auto* object = value.ordered_object();
    return object ? json::get(*object, key) : nullptr;
}

Type argument_type(const json::Value& value) {
    const auto text = value.string();
    constexpr std::array names{"string"sv, "integer"sv, "number"sv, "boolean"sv, "object"sv,
                               "array"sv, "null"sv};
    const auto match = std::find(names.begin(), names.end(), text);
    if (match == names.end()) {
        throw std::invalid_argument("Unsupported tool parameter type");
    }
    return static_cast<Type>(match - names.begin());
}

void read_types(const json::Value& schema, Args& types, int depth = 0) {
    if (!schema.ordered_object() || field(schema, "$ref") || depth > 32) {
        throw std::invalid_argument("Expected a supported tool parameter schema");
    }
    if (const auto* type = field(schema, "type")) {
        if (const auto* array = type->array(); array && !array->empty()) {
            for (const auto& item : *array) {
                types.push_back(argument_type(item));
            }
        } else {
            types.push_back(argument_type(*type));
        }
        return;
    }
    const auto* choices = field(schema, "anyOf");
    if (!choices) {
        choices = field(schema, "oneOf");
    }
    if (!choices) {
        types.push_back(Type::string);
    } else if (const auto* array = choices->array(); array && !array->empty()) {
        for (const auto& item : *array) {
            read_types(item, types, depth + 1);
        }
    } else {
        throw std::invalid_argument("Expected a nonempty tool parameter type union");
    }
}

bool valid_label(std::string_view name) {
    return !name.empty() && name.find_first_of("<>\r\n") == name.npos && trim(name) == name;
}

std::string_view schema_name(const json::Value* value) {
    const auto name = value ? value->string().value_or("") : ""sv;
    if (!valid_label(name)) {
        throw std::invalid_argument("Expected a nonempty tool or tool parameter name");
    }
    return name;
}

void read_enum_values(ChatToolProperty& property, const json::Value* choices) {
    if (!choices || !choices->array()) {
        return;
    }
    for (const auto& value : *choices->array()) {
        std::string spelling;
        if (const auto text = value.string()) {
            json::append_quoted(spelling, *text);
            property.enum_values.emplace(*text, std::move(spelling));
            continue;
        }
        if (const auto integer = value.signed_integer()) {
            spelling = std::to_string(*integer);
        } else if (const auto integer = value.unsigned_integer()) {
            spelling = std::to_string(*integer);
        } else if (const auto flag = value.boolean()) {
            spelling = *flag ? "true" : "false";
        } else if (value.is_null()) {
            spelling = "null";
        } else {
            continue;
        }
        property.enum_values.emplace(spelling, spelling);
    }
}

ChatToolSchema read_schema(const json::Value* value, bool strict) {
    ChatToolSchema schema;
    schema.strict = strict;
    if (!value || value->is_null()) {
        return schema;
    }
    const auto* type = field(*value, "type");
    if (!value->ordered_object() || (type && type->string() != "object") ||
        field(*value, "$ref") || field(*value, "anyOf") || field(*value, "oneOf")) {
        throw std::invalid_argument("Expected an object as the tool parameters schema");
    }
    if (const auto* props = field(*value, "properties")) {
        const auto* object = props->ordered_object();
        if (!object) {
            throw std::invalid_argument("Expected an object for tool parameter properties");
        }
        for (const auto& [name, property] : *object) {
            if (!valid_label(name)) {
                throw std::invalid_argument("Expected a valid tool parameter name");
            }
            auto& output = schema.properties[name];
            read_types(property, output.types);
            if (strict && !field(property, "type")) {
                read_enum_values(output, field(property, "enum"));
            }
        }
    }
    if (const auto* required = field(*value, "required")) {
        const auto* array = required->array();
        if (!array) {
            throw std::invalid_argument("Expected an array for required tool parameters");
        }
        for (const auto& item : *array) {
            schema.required.emplace(schema_name(&item));
        }
    }
    if (const auto* additional = field(*value, "additionalProperties")) {
        if (const auto flag = additional->boolean()) {
            schema.additional_properties = *flag;
        } else {
            throw std::invalid_argument("Expected boolean additionalProperties for tools");
        }
    }
    return schema;
}

bool append_integer(std::string& output, std::string_view text) {
    const bool negative = starts_with(text, "-");
    if (negative || starts_with(text, "+")) {
        text.remove_prefix(1);
    }
    if (text.empty() || text.find_first_not_of("0123456789") != text.npos) {
        return false;
    }
    const auto first = text.find_first_not_of('0');
    if (first == text.npos) {
        output += '0';
    } else {
        if (negative) {
            output += '-';
        }
        output += text.substr(first);
    }
    return true;
}

bool append_number(std::string& output, std::string_view text) {
    if (append_integer(output, text)) {
        return true;
    }
    if (starts_with(text, "+")) {
        text.remove_prefix(1);
        if (starts_with(text, "-") || starts_with(text, "+")) {
            return false;
        }
    }
    if (text.empty()) {
        return false;
    }
    double number;
    const auto* last = text.data() + text.size();
    const auto parsed = std::from_chars(text.data(), last, number);
    if (parsed.ec != std::errc{} || parsed.ptr != last || !std::isfinite(number)) {
        return false;
    }
    char buffer[64];
    const auto formatted = std::to_chars(buffer, buffer + sizeof(buffer), number);
    const std::string_view spelling(buffer, formatted.ptr - buffer);
    output += spelling;
    if (spelling.find_first_of(".eE") == spelling.npos) {
        output += ".0";
    }
    return true;
}

bool equals_folded(std::string_view text, std::string_view lower) {
    return std::equal(text.begin(), text.end(), lower.begin(), lower.end(), [](char a, char b) {
        const char folded = a >= 'A' && a <= 'Z' ? static_cast<char>(a + ('a' - 'A')) : a;
        return folded == b;
    });
}

bool append_boolean(std::string& output, std::string_view text) {
    if (equals_folded(text, "true")) {
        output += "true";
    } else if (equals_folded(text, "false")) {
        output += "false";
    } else {
        return false;
    }
    return true;
}

bool append_collection(std::string& output, std::string_view text, Type type) {
    try {
        const auto value = json::parse(text);
        if (type == Type::object ? value.object() != nullptr : value.array() != nullptr) {
            output += text;
            return true;
        }
    } catch (const json::ParseError&) {
        // A parameter may still match a later type in the union.
    }
    return false;
}

bool append_argument(std::string& output, std::string_view raw, Type type) {
    const auto text = trim(raw);
    switch (type) {
    case Type::string:
        json::append_quoted(output, raw);
        return true;
    case Type::integer:
        return append_integer(output, text);
    case Type::number:
        return append_number(output, text);
    case Type::boolean:
        return append_boolean(output, text);
    case Type::object:
    case Type::array:
        return append_collection(output, text, type);
    case Type::null:
        if (raw != "null") {
            return false;
        }
        output += "null";
        return true;
    }
    return false;
}

}

void read_chat_tools(ChatOutputOptions& options, const json::Array& tools) {
    for (const auto& tool : tools) {
        const auto* type = field(tool, "type");
        const auto* function = field(tool, "function");
        if (!type || type->string() != "function" || !function) {
            throw std::invalid_argument("Only function tools are supported");
        }
        const auto name = schema_name(field(*function, "name"));
        const auto* strict = field(*function, "strict");
        auto schema = read_schema(field(*function, "parameters"),
                                  strict && strict->boolean().value_or(false));
        if (!options.tools.emplace(name, std::move(schema)).second) {
            throw std::invalid_argument("Duplicate tool name");
        }
    }
}

void read_chat_tools(ChatOutputOptions& options, const json::Value* tools) {
    if (!tools || tools->is_null()) {
        return;
    }
    const auto* array = tools->array();
    if (!array) {
        throw std::invalid_argument("Expected an array of function tools");
    }
    read_chat_tools(options, *array);
}

namespace {

enum class FrameAction { next, wait, close, restart, nested, abandon };

struct FrameSpan {
    std::size_t start;
    std::size_t end;
};

struct FrameScan {
    FrameAction action;
    std::size_t position = 0;
};

class ToolFrame {
public:
    explicit ToolFrame(const ChatOutputOptions* options, std::size_t start = 0)
        : options_(options), start_(start), position_(start + tool_open.size()) {}

    FrameScan scan(std::string_view input);
    void resume(FrameScan result, ToolFrame& child);
    bool complete_truncated(std::string_view input);
    void release_recovery(std::list<FrameSpan>& spans) { spans.splice(spans.end(), provisional_); }
    std::string_view name() const {
        return schema_ && syntax_valid_ && arguments_valid_ ? name_ : std::string_view{};
    }
    const std::string& arguments() const { return arguments_; }

private:
    enum class State {
        function, function_name, parameters, parameter_name, value, value_tool, close, recover
    };

    bool marker(std::string_view input, std::string_view tag, State next);
    bool label(std::string_view input, bool function);
    FrameScan value(std::string_view input);
    FrameScan recover(std::string_view input);
    void convert(std::string_view raw, bool partial = false);
    bool ignoring_marker(char ch);

    const ChatOutputOptions* options_;
    const ChatToolSchema* schema_ = nullptr;
    State state_ = State::function;
    std::size_t start_;
    std::size_t position_;
    std::size_t field_start_ = 0;
    std::size_t nested_start_ = 0;
    std::list<FrameSpan> provisional_;
    std::string name_;
    std::string key_;
    std::string arguments_ = "{";
    std::set<std::string, std::less<>> seen_;
    bool syntax_valid_ = true;
    bool arguments_valid_ = true;
    bool value_started_ = false;
    bool json_value_ = false;
    bool quoted_ = false;
    bool escaped_ = false;
};

bool ToolFrame::marker(std::string_view input, std::string_view tag, State next) {
    const auto remaining = input.substr(position_);
    if (starts_with(remaining, tag)) {
        position_ += tag.size();
        field_start_ = position_;
        state_ = next;
        return true;
    }
    if (starts_with(tag, remaining)) {
        return false;
    }
    state_ = State::recover;
    return true;
}

bool ToolFrame::label(std::string_view input, bool function) {
    const auto end = input.find_first_of("<>\r\n", position_);
    if (end == input.npos) {
        position_ = input.size();
        return false;
    }
    const auto parsed = input.substr(field_start_, end - field_start_);
    if (input[end] != '>') {
        state_ = State::recover;
        position_ = end;
        return true;
    }
    syntax_valid_ = syntax_valid_ && valid_label(parsed);
    position_ = end + 1;
    if (function) {
        if (options_ && syntax_valid_) {
            name_ = parsed;
            if (const auto it = options_->tools.find(parsed); it != options_->tools.end()) {
                schema_ = &it->second;
            }
        }
        state_ = State::parameters;
    } else {
        key_ = parsed;
        field_start_ = position_;
        value_started_ = json_value_ = quoted_ = escaped_ = false;
        state_ = State::value;
    }
    return true;
}

void ToolFrame::convert(std::string_view raw, bool partial) {
    if (!schema_ || !syntax_valid_ || !arguments_valid_) {
        return;
    }
    const auto property = schema_->properties.find(key_);
    const bool unknown = property == schema_->properties.end() &&
                         !schema_->additional_properties;
    if (partial && (unknown || seen_.count(key_))) {
        return;
    }
    if (!seen_.emplace(key_).second) {
        arguments_valid_ = false;
        return;
    }
    if (starts_with(raw, "\n")) {
        raw.remove_prefix(1);
    }
    if (!raw.empty() && raw.back() == '\n') {
        raw.remove_suffix(1);
    }
    if (unknown) {
        arguments_valid_ = false;
        return;
    }
    if (seen_.size() != 1) {
        arguments_ += ',';
    }
    json::append_quoted(arguments_, key_);
    arguments_ += ':';
    if (!schema_->strict && equals_folded(raw, "null")) {
        arguments_ += "null";
    } else if (property == schema_->properties.end()) {
        json::append_quoted(arguments_, raw);
    } else {
        const auto& [types, choices] = property->second;
        if (choices.empty()) {
            const auto convert = [&](Type type) { return append_argument(arguments_, raw, type); };
            arguments_valid_ = std::any_of(types.begin(), types.end(), convert);
        } else {
            const auto choice = choices.find(raw);
            arguments_valid_ = choice != choices.end();
            if (arguments_valid_) {
                arguments_ += choice->second;
            }
        }
        if (partial && !arguments_valid_) {
            json::append_quoted(arguments_, raw);
            arguments_valid_ = true;
        }
    }
}

bool ToolFrame::ignoring_marker(char ch) {
    if (!value_started_ && whitespace.find(ch) == whitespace.npos) {
        value_started_ = true;
        json_value_ = ch == '{' || ch == '[';
        if (json_value_ && schema_ && schema_->strict) {
            const auto property = schema_->properties.find(key_);
            if (property != schema_->properties.end()) {
                const auto& types = property->second.types;
                const auto structured = [](Type type) {
                    return type == Type::object || type == Type::array;
                };
                json_value_ = std::any_of(types.begin(), types.end(), structured);
            }
        }
    }
    if (quoted_) {
        if (escaped_) {
            escaped_ = false;
        } else if (ch == '\\') {
            escaped_ = true;
        } else if (ch == '"') {
            quoted_ = false;
        }
        return true;
    }
    quoted_ = json_value_ && ch == '"';
    return ch != '<';
}

FrameScan ToolFrame::value(std::string_view input) {
    constexpr std::array markers{parameter_close, parameter_open, function_close, tool_close,
                                 tool_open};
    for (; position_ < input.size(); ++position_) {
        if (ignoring_marker(input[position_])) {
            continue;
        }
        const auto rest = input.substr(position_);
        for (const auto tag : markers) {
            if (starts_with(rest, tag)) {
                if (tag == parameter_close) {
                    convert(input.substr(field_start_, position_ - field_start_));
                    position_ += tag.size();
                    provisional_.clear();
                    state_ = State::parameters;
                } else if (tag == tool_open) {
                    nested_start_ = position_;
                    position_ += tag.size();
                    state_ = State::value_tool;
                } else if (tag == function_close || tag == tool_close) {
                    return {FrameAction::abandon, position_ + tag.size()};
                } else {
                    provisional_.clear();
                    state_ = State::recover;
                }
                return {FrameAction::next};
            }
            if (starts_with(tag, rest)) {
                return {FrameAction::wait};
            }
        }
    }
    return {FrameAction::wait};
}

FrameScan ToolFrame::recover(std::string_view input) {
    syntax_valid_ = false;
    constexpr std::array markers{tool_close, tool_open, function_open, parameter_open,
                                 parameter_close};
    while (position_ < input.size()) {
        const auto next = input.find('<', position_);
        position_ = next == input.npos ? input.size() : next;
        const auto rest = input.substr(position_);
        for (const auto tag : markers) {
            if (starts_with(tag, rest) && rest.size() < tag.size()) {
                return {FrameAction::wait};
            }
            if (!starts_with(rest, tag)) {
                continue;
            }
            if (tag == tool_close) {
                return {FrameAction::close, position_ + tag.size()};
            }
            if (tag == tool_open || (tag == parameter_close && start_ != 0)) {
                return {FrameAction::restart, position_};
            }
            if (tag == function_open || tag == parameter_open) {
                const auto next_state = tag == function_open ? State::function_name :
                                                               State::parameter_name;
                marker(input, tag, next_state);
                return {FrameAction::next};
            }
        }
        ++position_;
    }
    return {FrameAction::wait};
}

FrameScan ToolFrame::scan(std::string_view input) {
    for (;;) {
        if (state_ == State::function || state_ == State::parameters || state_ == State::close ||
            state_ == State::value_tool) {
            const auto next = input.find_first_not_of(whitespace, position_);
            position_ = next == input.npos ? input.size() : next;
        }
        FrameScan result{FrameAction::next};
        switch (state_) {
        case State::function:
            if (!marker(input, function_open, State::function_name)) {
                return {FrameAction::wait};
            }
            break;
        case State::function_name:
        case State::parameter_name:
            if (!label(input, state_ == State::function_name)) {
                return {FrameAction::wait};
            }
            break;
        case State::parameters: {
            const auto rest = input.substr(position_);
            const bool closing = starts_with(rest, function_close) ||
                                 starts_with(function_close, rest);
            if (!marker(input, closing ? function_close : parameter_open,
                        closing ? State::close : State::parameter_name)) {
                return {FrameAction::wait};
            }
            break;
        }
        case State::value:
            result = value(input);
            break;
        case State::value_tool: {
            const auto rest = input.substr(position_);
            if (starts_with(rest, function_open)) {
                return {FrameAction::nested, nested_start_};
            }
            if (starts_with(function_open, rest)) {
                return {FrameAction::wait};
            }
            state_ = State::value;
            break;
        }
        case State::close:
            if (starts_with(input.substr(position_), tool_close)) {
                if (schema_) {
                    const auto seen = [&](const auto& name) { return seen_.count(name); };
                    arguments_valid_ = arguments_valid_ && std::all_of(
                        schema_->required.begin(), schema_->required.end(), seen);
                    arguments_ += '}';
                }
                return {FrameAction::close, position_ + tool_close.size()};
            }
            if (!marker(input, tool_close, State::recover)) {
                return {FrameAction::wait};
            }
            break;
        case State::recover:
            result = recover(input);
            break;
        }
        if (result.action != FrameAction::next) {
            return result;
        }
    }
}

void ToolFrame::resume(FrameScan result, ToolFrame& child) {
    if (result.action == FrameAction::close && child.syntax_valid_) {
        provisional_.push_back({child.start_, result.position});
    } else if (result.action == FrameAction::abandon) {
        child.release_recovery(provisional_);
    }
    position_ = result.position;
    state_ = State::value;
}

std::string_view without_partial_marker(std::string_view raw) {
    const auto start = raw.rfind('<');
    if (start == raw.npos) {
        return raw;
    }
    const auto tail = raw.substr(start);
    constexpr std::array markers{parameter_close, parameter_open, function_close, tool_close};
    const auto partial = [&](std::string_view tag) {
        return tail.size() < tag.size() && starts_with(tag, tail);
    };
    return std::any_of(markers.begin(), markers.end(), partial) ? raw.substr(0, start) : raw;
}

bool ToolFrame::complete_truncated(std::string_view input) {
    if (!schema_ || !syntax_valid_ || !arguments_valid_) {
        return false;
    }
    if (state_ == State::value || state_ == State::value_tool) {
        convert(without_partial_marker(input.substr(field_start_)), true);
    } else if (state_ != State::parameters && state_ != State::parameter_name &&
               state_ != State::close) {
        return false;
    }
    arguments_ += '}';
    return arguments_valid_;
}

}

class ChatToolFrame {
public:
    explicit ChatToolFrame(const ChatOutputOptions& options) : frames_{ToolFrame(&options)} {}

    std::size_t scan(std::string_view input, bool final, bool truncated = false);
    bool emit(std::string_view input, const ChatOutputOptions& options, std::size_t& calls,
              const ChatOutputSink& sink) const;

private:
    std::string_view name() const {
        return complete_ ? frames_.front().name() : std::string_view{};
    }
    const std::string& arguments() const { return frames_.front().arguments(); }

    std::vector<ToolFrame> frames_;
    std::list<FrameSpan> recovery_;
    bool complete_ = false;
    bool dropped_ = false;
};

std::size_t ChatToolFrame::scan(std::string_view input, bool final, bool truncated) {
    // A closed parameter owns any call-shaped text that it contains.
    for (;;) {
        auto& frame = frames_.back();
        const auto step = frame.scan(input);
        if (step.action == FrameAction::nested) {
            frames_.emplace_back(nullptr, step.position);
        } else if (step.action == FrameAction::wait) {
            if (!final) {
                return 0;
            }
            if (truncated) {
                complete_ = frames_.front().complete_truncated(input);
                dropped_ = !complete_;
                return input.size();
            }
            for (auto& pending : frames_) {
                pending.release_recovery(recovery_);
            }
            return input.size();
        } else if (frames_.size() == 1) {
            complete_ = step.action == FrameAction::close;
            if (step.action == FrameAction::abandon) {
                frame.release_recovery(recovery_);
            }
            return step.position;
        } else {
            frames_[frames_.size() - 2].resume(step, frame);
            frames_.pop_back();
        }
    }
}

bool ChatToolFrame::emit(std::string_view input, const ChatOutputOptions& options,
                         std::size_t& calls, const ChatOutputSink& sink) const {
    const auto content = [&](std::string_view raw) {
        return raw.empty() || sink({ChatOutputKind::content, raw, 0, {}});
    };
    if (!name().empty()) {
        return sink({ChatOutputKind::tool_call, arguments(), calls++, name()});
    }
    if (dropped_) {
        return true;
    }
    std::size_t emitted = 0;
    for (const auto& span : recovery_) {
        if (!options.accepts_tool_calls(calls)) {
            break;
        }
        if (!content(input.substr(emitted, span.start - emitted))) {
            return false;
        }
        const auto raw = input.substr(span.start, span.end - span.start);
        ChatToolFrame child(options);
        child.scan(raw, false);
        const bool ready = child.name().empty() ? content(raw) :
            sink({ChatOutputKind::tool_call, child.arguments(), calls++, child.name()});
        if (!ready) {
            return false;
        }
        emitted = span.end;
    }
    return content(input.substr(emitted));
}

ChatOutputParser::ChatOutputParser(const ChatOutputOptions& options)
    : options_(options), phase_(options.separate_reasoning ?
          (options.prompt_has_open_think ? Phase::reasoning : Phase::opening) : Phase::content) {}

ChatOutputParser::~ChatOutputParser() = default;

}

namespace sglang::model {

std::optional<server::SeparatedReasoning> separate_qwen_reasoning(std::string_view text,
                                                                std::string_view parser_name) {
    std::string name(parser_name);
    for (auto& character : name) {
        if (character >= 'A' && character <= 'Z') character += 'a' - 'A';
    }
    constexpr std::array<std::string_view, 6> parsers{
        qwen_reasoning_parser, "qwen3-thinking", "dots", "interns1", "minimax", "nanbeige"};
    if (std::find(parsers.begin(), parsers.end(), name) == parsers.end()) return std::nullopt;
    const bool force_reasoning = name == "qwen3-thinking" || name == "minimax";
    if (!force_reasoning && text.find(server::think_open) == text.npos) {
        return server::SeparatedReasoning{{}, std::string(text)};
    }
    while (server::starts_with(text, server::think_open)) {
        text.remove_prefix(server::think_open.size());
    }
    server::ChatOutputOptions options;
    options.separate_reasoning = true;
    options.prompt_has_open_think = true;
    server::ChatOutputParser parser(options);
    server::SeparatedReasoning result;
    const auto collect = [&](const server::ChatOutputEvent& event) {
        auto& output = event.kind == server::ChatOutputKind::reasoning ?
            result.reasoning : result.text;
        output += event.text;
        return true;
    };
    const auto tool_start = text.find(server::tool_open);
    parser.push(text.substr(0, tool_start), collect);
    parser.finish(server::ChatEnd::stop, collect);
    if (tool_start != text.npos) {
        result.text += text.substr(tool_start);
    }
    return result;
}

}

namespace sglang::server {

bool ChatOutputParser::emit_text(std::string_view text, const ChatOutputSink& sink) {
    if (text.empty()) {
        return true;
    }
    if (phase_ != Phase::reasoning) {
        return sink({ChatOutputKind::content, text, 0, {}});
    }
    if (options_.stream_reasoning) {
        return sink({ChatOutputKind::reasoning, text, 0, {}});
    }
    reasoning_ += text;
    return true;
}

bool ChatOutputParser::flush_reasoning(const ChatOutputSink& sink) {
    const bool ok = reasoning_.empty() || sink({ChatOutputKind::reasoning, reasoning_, 0, {}});
    reasoning_.clear();
    return ok;
}

bool ChatOutputParser::drain(bool final, const ChatOutputSink& sink) {
    std::string_view input(pending_);
    if (phase_ == Phase::opening) {
        if (starts_with(input, think_open)) {
            input.remove_prefix(think_open.size());
            phase_ = Phase::reasoning;
        } else if (!final && starts_with(think_open, input)) {
            return true;
        } else {
            phase_ = Phase::content;
        }
    }
    while (!input.empty()) {
        const bool truncated = final && truncated_ && options_.complete_truncated_calls;
        if (tool_) {
            const auto length = tool_->scan(input, final, truncated);
            if (length == 0) {
                break;
            }
            if (!tool_->emit(input.substr(0, length), options_, calls_, sink)) {
                return false;
            }
            tool_.reset();
            input.remove_prefix(length);
            continue;
        }
        const auto split = input.find('<');
        if (!emit_text(input.substr(0, split), sink)) {
            return false;
        }
        input.remove_prefix(std::min(split, input.size()));
        if (input.empty()) {
            break;
        }
        const bool reasoning = phase_ == Phase::reasoning;
        const bool tools = options_.accepts_tool_calls(calls_);
        if (reasoning && starts_with(input, think_close)) {
            if (!flush_reasoning(sink)) {
                return false;
            }
            phase_ = Phase::content;
            input.remove_prefix(think_close.size());
        } else if (tools && starts_with(input, tool_open)) {
            if (!flush_reasoning(sink)) {
                return false;
            }
            phase_ = Phase::content;
            tool_ = std::make_unique<ChatToolFrame>(options_);
        } else if (!final && ((reasoning && starts_with(think_close, input)) ||
                              (tools && starts_with(tool_open, input)))) {
            break;
        } else if (truncated && tools && starts_with(tool_open, input)) {
            input = {};
        } else {
            const auto next = input.find('<', 1);
            if (!emit_text(input.substr(0, next), sink)) {
                return false;
            }
            input.remove_prefix(std::min(next, input.size()));
        }
    }
    pending_.erase(0, pending_.size() - input.size());
    return !final || flush_reasoning(sink);
}

bool ChatOutputParser::push(std::string_view text, const ChatOutputSink& sink) {
    if (!options_.enabled()) {
        return emit_text(text, sink);
    }
    pending_.append(text);
    return drain(false, sink);
}

bool ChatOutputParser::finish(ChatEnd end, const ChatOutputSink& sink) {
    if (end == ChatEnd::cancel) {
        pending_.clear();
        reasoning_.clear();
        tool_.reset();
        return true;
    }
    truncated_ = end == ChatEnd::length;
    return drain(true, sink);
}

}
