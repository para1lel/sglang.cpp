#include "sglang/server/chat_template.h"

#include "sglang/tokenizer/digest.h"
#include "sglang/utf8.h"

#include <charconv>
#include <fstream>
#include <set>

namespace sglang::server {
namespace {

using json::get;
using json::OrderedObject;
using json::Value;
using namespace std::string_view_literals;

std::string checked_file(const std::filesystem::path& path, std::size_t size,
                         std::string_view digest) {
    std::ifstream stream(path, std::ios::binary);
    std::string contents(size, '\0');
    if (!stream.read(contents.data(), contents.size()) ||
        stream.peek() != std::char_traits<char>::eof() || tokenizer::sha256(contents) != digest) {
        throw std::runtime_error("Missing or incompatible checkpoint file: " + path.string());
    }
    return contents;
}

std::string checkpoint_literal(std::string_view source, std::string_view prefix) {
    const auto start = source.find(prefix);
    if (start == std::string_view::npos) {
        throw std::logic_error("Missing literal in the pinned chat template");
    }
    source.remove_prefix(start + 1);
    const auto end = source.find(prefix.front());
    if (end == std::string_view::npos) {
        throw std::logic_error("Missing literal delimiter in the pinned chat template");
    }
    std::string result;
    for (std::size_t index = 0; index < end; ++index) {
        if (source[index] == '\\') {
            if (++index == end || source[index] != 'n') {
                throw std::logic_error("Unexpected escape in the pinned chat template");
            }
            result += '\n';
        } else {
            result += source[index];
        }
    }
    return result;
}

bool python_space(char32_t scalar) {
    return (scalar >= 0x09 && scalar <= 0x0d) || (scalar >= 0x1c && scalar <= 0x20) ||
        (scalar >= 0x2000 && scalar <= 0x200a) || scalar == 0x85 || scalar == 0xa0 ||
        scalar == 0x1680 || scalar == 0x2028 || scalar == 0x2029 || scalar == 0x202f ||
        scalar == 0x205f || scalar == 0x3000;
}

bool edge_space(std::string_view scalar) {
    if (scalar.size() == 1) {
        return python_space(static_cast<unsigned char>(scalar.front()));
    }
    return scalar.size() >= 2 && scalar.size() <= 3 && python_space(read_utf8(scalar).front());
}

std::string_view trim(std::string_view text) {
    while (!text.empty()) {
        const auto size = utf8_lead(static_cast<unsigned char>(text.front())).size;
        if (!edge_space(text.substr(0, size))) {
            break;
        }
        text.remove_prefix(size);
    }
    while (!text.empty()) {
        auto start = text.size() - 1;
        while ((static_cast<unsigned char>(text[start]) & 0xc0) == 0x80) {
            --start;
        }
        if (!edge_space(text.substr(start))) {
            break;
        }
        text.remove_suffix(text.size() - start);
    }
    return text;
}

std::optional<std::string_view> string(const OrderedObject& object, std::string_view key) {
    const auto* value = get(object, key);
    return value ? value->string() : std::nullopt;
}

bool boolean(const OrderedObject& object, std::string_view key, bool fallback) {
    const auto* value = get(object, key);
    if (!value) {
        return fallback;
    }
    if (const auto result = value->boolean()) {
        return *result;
    }
    throw std::invalid_argument("Expected boolean " + std::string(key));
}

struct VisionCounts {
    std::size_t images = 0;
    std::size_t videos = 0;
    bool numbered;
};

void append_media(std::string& output, const OrderedObject& part, bool image, bool system,
                  VisionCounts& vision, bool count, ChatInput input) {
    if (system) {
        throw std::invalid_argument(image ? "System message cannot contain images."
                                          : "System message cannot contain videos.");
    }
    if (input == ChatInput::http) {
        for (const auto& [key, value] : part) {
            if (key == "type") {
                continue;
            }
            if (key != "image" && key != "image_url" && key != "video" &&
                key != "video_url" && key != "url") {
                throw std::invalid_argument("Unexpected field in a media placeholder: " + key);
            }
            if (!value.is_null()) {
                throw std::invalid_argument("Media payloads are unsupported by the text decoder");
            }
        }
    }
    auto& number = image ? vision.images : vision.videos;
    number += count;
    if (vision.numbered) {
        output += image ? "Picture " : "Video ";
        output += std::to_string(number) + ": ";
    }
    output += image ? "<|vision_start|><|image_pad|><|vision_end|>"
                    : "<|vision_start|><|video_pad|><|vision_end|>";
}

std::string render_content(const OrderedObject& message, VisionCounts& vision,
                           bool count, bool system, ChatInput input) {
    const auto* value = get(message, "content");
    if (!value || value->is_null()) {
        return {};
    }
    if (const auto text = value->string()) {
        return std::string(trim(*text));
    }
    const auto* parts = value->array();
    if (!parts) {
        throw std::invalid_argument("Unexpected content type.");
    }
    std::string content;
    for (const auto& value : *parts) {
        const auto* part = value.ordered_object();
        if (!part) {
            throw std::invalid_argument("Expected an object in content parts");
        }
        const auto type = string(*part, "type");
        if (input == ChatInput::http && !type && get(*part, "type")) {
            throw std::invalid_argument("Expected a string content part type");
        }
        const bool image = get(*part, "image") || get(*part, "image_url") || type == "image";
        const bool video = get(*part, "video") || type == "video";
        if (image || video) {
            append_media(content, *part, image, system, vision, count, input);
        } else {
            const auto text = string(*part, "text");
            if (!text || (input == ChatInput::http && type && type != "text")) {
                throw std::invalid_argument("Unexpected item type in content.");
            }
            content += *text;
        }
    }
    return std::string(trim(content));
}

void append_float(std::string& output, double number) {
    char buffer[32];
    const auto converted = std::to_chars(buffer, buffer + sizeof(buffer), number,
                                         std::chars_format::scientific);
    if (converted.ec != std::errc{}) {
        throw std::logic_error("Could not format a JSON number");
    }
    std::string_view scientific(buffer, converted.ptr - buffer);
    const auto exponent_at = scientific.find('e');
    const auto exponent_digits = scientific.substr(exponent_at + 2);
    int exponent = 0;
    std::from_chars(exponent_digits.data(), scientific.data() + scientific.size(), exponent);
    if (scientific[exponent_at + 1] == '-') {
        exponent = -exponent;
    }
    if (exponent < -4 || exponent >= 16) {
        output += scientific;
        return;
    }
    auto mantissa = scientific.substr(0, exponent_at);
    if (mantissa.front() == '-') {
        output += '-';
        mantissa.remove_prefix(1);
    }
    std::string digits(mantissa);
    if (digits.size() > 1) {
        digits.erase(1, 1);
    }
    if (exponent < 0) {
        output += "0.";
        output.append(-exponent - 1, '0');
        output += digits;
        return;
    }
    const auto whole = static_cast<std::size_t>(exponent + 1);
    output += digits.substr(0, whole);
    if (digits.size() > whole) {
        output += '.';
        output += digits.substr(whole);
    } else {
        output.append(whole - digits.size(), '0');
        output += ".0";
    }
}

void append_template_json(std::string& output, const Value& value) {
    if (const auto* object = value.ordered_object()) {
        output += '{';
        std::string_view separator;
        for (const auto& [key, field] : *object) {
            output += separator;
            json::append_quoted(output, key, true);
            output += ": ";
            append_template_json(output, field);
            separator = ", ";
        }
        output += '}';
    } else if (const auto* array = value.array()) {
        output += '[';
        std::string_view separator;
        for (const auto& item : *array) {
            output += separator;
            append_template_json(output, item);
            separator = ", ";
        }
        output += ']';
    } else if (const auto text = value.string()) {
        json::append_quoted(output, *text, true);
    } else if (const auto integer = value.signed_integer()) {
        output += std::to_string(*integer);
    } else if (const auto integer = value.unsigned_integer()) {
        output += std::to_string(*integer);
    } else if (const auto* integer = value.wide_integer()) {
        output += integer->digits;
    } else if (const auto number = value.number()) {
        append_float(output, *number);
    } else if (const auto flag = value.boolean()) {
        output += *flag ? "true" : "false";
    } else if (value.is_null()) {
        output += "null";
    } else {
        throw std::logic_error("Expected a value from ordered JSON");
    }
}

void append_arguments(std::string& output, const Value* value, ChatInput input) {
    if (!value || value->string() == "") {
        return;
    }
    Value parsed;
    const auto* args = value->ordered_object();
    if (input == ChatInput::http && value->string()) {
        try {
            parsed = json::parse(*value->string(), json::Syntax::ordered);
            args = parsed.ordered_object();
        } catch (const json::ParseError&) {
            throw std::invalid_argument("Tool-call arguments must contain a JSON object");
        }
    }
    if (!args) {
        throw std::invalid_argument("Tool-call arguments must be an object");
    }
    for (const auto& [name, argument] : *args) {
        output += "<parameter=" + name + ">\n";
        if (const auto text = argument.string()) {
            output += *text;
        } else {
            append_template_json(output, argument);
        }
        output += "\n</parameter>\n";
    }
}

void append_tool_calls(std::string& output, const OrderedObject& message, bool has_content,
                       ChatInput input) {
    const auto* value = get(message, "tool_calls");
    if (!value || value->is_null()) {
        return;
    }
    const auto* calls = value->array();
    if (!calls) {
        throw std::invalid_argument("Expected an array of tool calls");
    }
    std::string_view separator = has_content ? "\n\n" : "";
    for (const auto& call : *calls) {
        const auto* function = call.ordered_object();
        if (const auto* type = function ? get(*function, "type") : nullptr;
            input == ChatInput::http && type && type->string() != "function") {
            throw std::invalid_argument("Only function tool-call history is supported");
        }
        if (const auto* wrapper = function ? get(*function, "function") : nullptr) {
            function = wrapper->ordered_object();
        }
        const auto name = function ? string(*function, "name") : std::nullopt;
        if (!name) {
            throw std::invalid_argument("Expected a tool-call function name");
        }
        output += separator;
        output += "<tool_call>\n<function=";
        output += *name;
        output += ">\n";
        append_arguments(output, get(*function, "arguments"), input);
        output += "</function>\n</tool_call>";
        separator = "\n";
    }
}

std::string_view reasoning(const OrderedObject& message) {
    for (const auto key : {"reasoning_content"sv, "thinking"sv, "reasoning"sv}) {
        if (const auto* value = get(message, key); value && !value->is_null()) {
            return trim(value->string().value_or(""));
        }
    }
    return {};
}

struct Turn {
    std::string_view role;
    const OrderedObject& message;
};

std::vector<Turn> read_turns(const OrderedObject& chat) {
    const auto* value = get(chat, "messages");
    const auto* messages = value ? value->array() : nullptr;
    if (!messages || messages->empty()) {
        throw std::invalid_argument("No messages provided. Expected a nonempty array.");
    }
    std::vector<Turn> turns;
    turns.reserve(messages->size());
    for (const auto& value : *messages) {
        const auto* message = value.ordered_object();
        const auto role = message ? string(*message, "role") : std::nullopt;
        if (!role) {
            throw std::invalid_argument("Expected a message object with a string role");
        }
        const bool instruction = *role == "root" || *role == "developer";
        turns.push_back({instruction ? "system"sv : *role, *message});
    }
    return turns;
}

std::size_t last_query(const std::vector<Turn>& turns, VisionCounts& vision, ChatInput input) {
    for (auto index = turns.size(); index-- > 0;) {
        if (turns[index].role == "user") {
            const auto content = render_content(turns[index].message, vision, false, false, input);
            constexpr auto open = "<tool_response>"sv;
            constexpr auto close = "</tool_response>"sv;
            if (content.compare(0, open.size(), open) != 0 ||
                content.size() < close.size() ||
                content.compare(content.size() - close.size(), close.size(), close) != 0) {
                return index;
            }
        }
    }
    throw std::invalid_argument("No user query found in messages.");
}

void append_turns(std::string& output, const std::vector<Turn>& turns, VisionCounts& vision,
                  bool preserve, std::size_t query, ChatInput input) {
    for (std::size_t index = 0; index < turns.size(); ++index) {
        const auto& [role, message] = turns[index];
        const auto content = render_content(message, vision, true, role == "system", input);
        if (role == "system") {
            if (index != 0) {
                throw std::invalid_argument("System message must be at the beginning.");
            }
        } else if (role == "user") {
            output += "<|im_start|>user\n" + content + "<|im_end|>\n";
        } else if (role == "assistant") {
            output += "<|im_start|>assistant\n";
            if (preserve || index > query) {
                output += "<think>\n";
                output += reasoning(message);
                output += "\n</think>\n\n";
            }
            output += content;
            append_tool_calls(output, message, !content.empty(), input);
            output += "<|im_end|>\n";
        } else if (role == "tool") {
            if (index && turns[index - 1].role != "tool") {
                output += "<|im_start|>user";
            }
            output += "\n<tool_response>\n" + content + "\n</tool_response>";
            if (index + 1 == turns.size() || turns[index + 1].role != "tool") {
                output += "<|im_end|>\n";
            }
        } else {
            throw std::invalid_argument("Unexpected message role.");
        }
    }
}

}

ChatTemplate::ChatTemplate(const std::filesystem::path& checkpoint) {
    checked_file(checkpoint / "tokenizer_config.json", 9984,
                 "207c03c41a06a9c3b0cc527d8a56115159c1eac6b59a4ffae85b646e773d074e");
    const auto source = checked_file(checkpoint / "chat_template.jinja", 9004,
        "af16f654968d47d09a6c604e00370b12dbc21f76456f59c9e98b82e84d9f6375");
    xhigh_ = checkpoint_literal(source, "'Reasoning effort is set to xhigh.");
    low_ = checkpoint_literal(source, "'Reasoning effort is set to low.");
    tools_intro_ = checkpoint_literal(source, "\"# Tools");
    tools_instructions_ = checkpoint_literal(source, "'\\n\\nIf you choose to call");
}

namespace {

const Value* field(const Value& value, std::string_view key) {
    const auto* object = value.ordered_object();
    return object ? get(*object, key) : nullptr;
}

void collect_tools(json::Array& output, const Value* value) {
    if (!value || value->is_null()) return;
    const auto* tools = value->array();
    if (!tools) throw std::invalid_argument("Expected an array of tool schemas");
    output.insert(output.end(), tools->begin(), tools->end());
}

void collect_message_tools(json::Array& output, const OrderedObject& chat) {
    const auto* list = get(chat, "messages");
    if (!list || !list->array()) return;
    for (const auto& message : *list->array()) {
        const auto* tools = field(message, "tools");
        if (!tools || tools->is_null()) continue;
        const auto* role = field(message, "role");
        if (role && (role->string() == "system" || role->string() == "developer" ||
                     role->string() == "root")) {
            collect_tools(output, tools);
        } else if (!tools->array() || !tools->array()->empty()) {
            throw std::invalid_argument("Message tools require a privileged instruction role");
        }
    }
}

ChatToolChoice read_tool_choice(const Value* selector, std::string_view& selected) {
    if (!selector || selector->is_null() || selector->string() == "auto") {
        return ChatToolChoice::automatic;
    }
    if (selector->string() == "none") return ChatToolChoice::none;
    if (selector->string() == "required") return ChatToolChoice::required;
    const auto* type = field(*selector, "type");
    const auto* function = field(*selector, "function");
    const auto* name = function ? field(*function, "name") : nullptr;
    if (!type || type->string() != "function" || !name || !name->string() ||
        name->string()->empty()) {
        throw std::invalid_argument("Unsupported tool_choice. Expected auto, none, required, "
                                    "or a named function.");
    }
    selected = *name->string();
    return ChatToolChoice::named;
}

}

ChatTools chat_tools(const OrderedObject& chat) {
    ChatTools output;
    std::string_view selected;
    output.choice = read_tool_choice(get(chat, "tool_choice"), selected);
    if (const auto* parallel = get(chat, "parallel_tool_calls")) {
        if (!parallel->boolean()) {
            throw std::invalid_argument("Expected boolean parallel_tool_calls");
        }
        output.parallel = *parallel->boolean();
    }
    json::Array effective;
    collect_tools(effective, get(chat, "tools"));
    collect_message_tools(effective, chat);
    if (output.choice == ChatToolChoice::none) return output;
    std::set<std::string> names;
    for (auto& tool : effective) {
        const auto* type = field(tool, "type");
        const auto* function = field(tool, "function");
        const auto* value = function ? field(*function, "name") : nullptr;
        const auto name = value ? value->string().value_or("") : ""sv;
        if (!type || type->string() != "function" || name.empty() || name.size() > 64 ||
            name.find_first_of("<>\r\n") != name.npos || !names.emplace(name).second) {
            throw std::invalid_argument("Expected unique function tools with valid names");
        }
        if (output.choice == ChatToolChoice::named && name != selected) continue;
        const auto* strict = field(*function, "strict");
        if (strict && !strict->is_null()) {
            if (!strict->boolean()) throw std::invalid_argument("Expected boolean function.strict");
            output.strict |= *strict->boolean();
        }
        output.tools.push_back(std::move(tool));
    }
    if (output.forced() && output.tools.empty()) {
        throw std::invalid_argument("Required or named tool_choice needs an advertised function");
    }
    return output;
}

std::string ChatTemplate::render(const OrderedObject& chat, ChatInput input) const {
    return render(chat, input, chat_tools(chat));
}

std::string ChatTemplate::render(const OrderedObject& chat, ChatInput input,
                                 const ChatTools& selection) const {
    auto turns = read_turns(chat);
    const bool thinking = boolean(chat, "enable_thinking", true);
    const bool preserve = boolean(chat, "preserve_thinking", true);
    const bool generation = boolean(chat, "add_generation_prompt", input == ChatInput::http);
    VisionCounts vision{0, 0, boolean(chat, "add_vision_id", false)};
    std::string output;
    std::string_view instructions;
    if (thinking) {
        const auto* value = get(chat, "reasoning_effort");
        const auto effort = value ? value->string() : std::optional("xhigh"sv);
        if (effort == "xhigh") {
            instructions = xhigh_;
        } else if (effort == "low") {
            instructions = low_;
        } else if (effort != "medium") {
            throw std::invalid_argument(
                "Unexpected reasoning effort. Expected xhigh, medium or low.");
        }
    }
    const auto& tools = selection.tools;
    const auto system = turns.front().role == "system"
        ? render_content(turns.front().message, vision, false, true, input) : std::string();
    const bool has_tools = !tools.empty();
    if (!instructions.empty() || has_tools || !system.empty()) {
        output += "<|im_start|>system\n";
        output += instructions;
        if (!instructions.empty() && (has_tools || !system.empty())) {
            output += "\n\n";
        }
        if (has_tools) {
            output += tools_intro_;
            for (const auto& tool : tools) {
                if (!tool.ordered_object()) {
                    throw std::invalid_argument("Expected a tool schema object");
                }
                output += '\n';
                append_template_json(output, tool);
            }
            output += "\n</tools>";
            output += tools_instructions_;
            if (!system.empty()) {
                output += "\n\n";
            }
        }
        output += system + "<|im_end|>\n";
    }
    const auto query = last_query(turns, vision, input);
    append_turns(output, turns, vision, preserve, query, input);
    if (generation) {
        output += "<|im_start|>assistant\n<think>\n";
        if (!thinking) {
            output += "\n</think>\n\n";
        }
    }
    return output;
}

}
