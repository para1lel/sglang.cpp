#include "sglang/server/tokenization.h"

#include "sglang/json.h"
#include "sglang/server/chat_template.h"
#include "sglang/server/completions.h"
#include "sglang/server/http.h"
#include "sglang/server/tokenization_validation.h"
#include "sglang/tokenizer/tokenizer.h"
#include "sglang/utf8.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <iostream>

namespace sglang::server {
namespace {

constexpr std::string_view bad_request = "400 Bad Request";

const json::Value* supplied(const json::OrderedObject& object, std::string_view field) {
    const auto* value = json::get(object, field);
    return value && !value->is_null() ? value : nullptr;
}

[[noreturn]] void invalid_request(const char* message) {
    throw HttpError(bad_request, message);
}

std::string lower_ascii(std::string_view text) {
    std::string lower(text);
    for (auto& character : lower) {
        if (character >= 'A' && character <= 'Z') {
            character += 'a' - 'A';
        }
    }
    return lower;
}

bool boolean_option(const json::Value* value, std::string_view field, bool fallback = true) {
    if (!value) {
        return fallback;
    }
    if (const auto boolean = chat_boolean(*value)) {
        return *boolean;
    }
    invalid_request((std::string(field) + " must be a boolean").c_str());
}

void append_tokens(std::string& output, const std::vector<TokenId>& tokens) {
    output += '[';
    for (const auto& token : tokens) {
        if (&token != &tokens.front()) {
            output += ',';
        }
        output += std::to_string(token);
    }
    output += ']';
}

bool truthy(const json::Value& value) {
    if (const auto boolean = value.boolean()) {
        return *boolean;
    }
    if (const auto number = value.number()) {
        return *number != 0;
    }
    if (const auto text = value.string()) {
        return !text->empty();
    }
    if (const auto* array = value.array()) {
        return !array->empty();
    }
    if (const auto* object = value.object()) {
        return !object->empty();
    }
    if (const auto* object = value.ordered_object()) {
        return !object->empty();
    }
    return !value.is_null();
}

std::string_view media_part(const json::OrderedObject& part, std::string_view type) {
    const auto* media = json::get(part, type);
    if (type == "image_url" && media && media->string()) {
        return "image";
    }
    const auto* media_object = media ? media->ordered_object() : nullptr;
    const auto* url = media_object ? json::get(*media_object, "url") : nullptr;
    if (!url || !url->string()) {
        invalid_request("Expected a string media URL");
    }
    if (type == "image_url") {
        if (const auto* detail = supplied(*media_object, "detail"); detail &&
            detail->string() != "auto" && detail->string() != "default" &&
            detail->string() != "low" && detail->string() != "high") {
            invalid_request("image_url.detail must be auto, low, or high");
        }
        return "image";
    }
    if (const auto* fps = supplied(*media_object, "fps")) {
        const auto number = chat_number(*fps);
        if (!number || !std::isfinite(*number) || *number <= 0) {
            invalid_request("video_url.fps must be a finite positive number");
        }
    }
    return "video";
}

json::Value chat_content(const json::Value* content, std::string_view role) {
    if (!content && role != "user") return "";
    if (content && content->string()) return *content;
    const auto* parts = content ? content->array() : nullptr;
    if (!parts) invalid_request("Expected message content to be a string or an array");
    json::Array normalized;
    std::string tool_text;
    bool all_text = true;
    for (const auto& item : *parts) {
        const auto* part = item.ordered_object();
        const auto* kind = part ? json::get(*part, "type") : nullptr;
        const auto type = kind ? kind->string().value_or("") : "";
        if (type == "text") {
            const auto* text = json::get(*part, "text");
            if (!text || !text->string()) invalid_request("Expected string text in content part");
            normalized.emplace_back(json::OrderedObject{{"type", "text"}, {"text", *text}});
            if (role == "tool") {
                if (normalized.size() > 1) tool_text += ' ';
                tool_text += *text->string();
            }
        } else if (type == "image_url" || type == "video_url") {
            all_text = false;
            normalized.emplace_back(json::OrderedObject{
                {"type", std::string(media_part(*part, type))}});
        } else if (type == "audio_url" || type == "input_audio" || type == "tool_reference") {
            if (type == "tool_reference") {
                const auto* name = supplied(*part, "name");
                if (!name || !name->string()) invalid_request("Expected a tool reference name");
            }
            all_text = false;
            normalized.emplace_back(json::OrderedObject{{"type", std::string(type)}});
        } else {
            invalid_request("Unsupported chat content part type");
        }
    }
    if (role == "tool" && all_text) return tool_text;
    return normalized;
}

void validate_history(const json::Value& value, std::string_view role) {
    const auto* calls = value.array();
    if (!calls) invalid_request("Expected an array of tool calls");
    for (const auto& call : *calls) {
        const auto* wrapper = call.ordered_object();
        const auto* type = wrapper ? json::get(*wrapper, "type") : nullptr;
        const auto* function = wrapper ? json::get(*wrapper, "function") : nullptr;
        const auto* fields = function ? function->ordered_object() : nullptr;
        const auto* name = fields ? supplied(*fields, "name") : nullptr;
        const auto* args = fields ? supplied(*fields, "arguments") : nullptr;
        const auto* id = wrapper ? supplied(*wrapper, "id") : nullptr;
        const auto* index = wrapper ? supplied(*wrapper, "index") : nullptr;
        const auto number = index ? chat_number(*index) : std::nullopt;
        if (!fields || (type && type->string() != "function") ||
            (id && !id->string()) ||
            (index && (!number || !std::isfinite(*number) || std::trunc(*number) != *number)) ||
            (name && !name->string()) || (role == "assistant" &&
                (!name || name->string()->empty())) ||
            (args && !args->string() && !args->ordered_object())) {
            invalid_request("Invalid function tool call in message history");
        }
    }
}

void validate_turn_fields(const json::OrderedObject& message) {
    for (const auto* key : {"name", "tool_call_id", "reasoning_content"}) {
        const auto* field = supplied(message, key);
        if (field && !field->string()) {
            invalid_request((std::string("Expected string ") + key).c_str());
        }
    }
    if (const auto* phase = supplied(message, "phase"); phase &&
        phase->string() != "commentary" && phase->string() != "final_answer") {
        invalid_request("Invalid message phase");
    }
    if (const auto* prefix = supplied(message, "prefix")) boolean_option(prefix, "prefix");
}

json::OrderedObject chat_turn(const json::Value& value) {
    const auto* message = value.ordered_object();
    const auto* field = message ? json::get(*message, "role") : nullptr;
    if (!field || !field->string()) invalid_request("Expected a message with a string role");
    const auto role = lower_ascii(*field->string());
    if (role != "system" && role != "assistant" && role != "tool" && role != "function" &&
        role != "root" && role != "developer" && role != "latest_reminder" &&
        *field->string() != "user") {
        invalid_request("Invalid message role");
    }
    json::OrderedObject turn{{"role", role},
                             {"content", chat_content(supplied(*message, "content"), role)}};
    if (role == "user") return turn;
    validate_turn_fields(*message);
    if (const auto* reasoning = supplied(*message, "reasoning_content")) {
        turn.emplace_back("reasoning_content", *reasoning);
    }
    if (const auto* calls = supplied(*message, "tool_calls")) {
        validate_history(*calls, role);
        turn.emplace_back("tool_calls", *calls);
    }
    if (const auto* tools = supplied(*message, "tools")) {
        if (!tools->array()) invalid_request("Expected an array of message tools");
        if (!tools->array()->empty()) turn.emplace_back("tools", *tools);
    }
    return turn;
}

json::OrderedObject tool_function(const json::OrderedObject& function,
                                  const json::Value* wrapper_defer) {
    json::OrderedObject fields;
    for (const auto& [key, value] : function) {
        if (key == "name" || key == "description" || key == "parameters") {
            if (key == "description" && !value.is_null() && !value.string()) {
                invalid_request("Expected string function description");
            }
            fields.emplace_back(key, value);
        } else if (key == "strict" || key == "defer_loading") {
            if (!value.is_null()) {
                fields.emplace_back(key, boolean_option(&value, key));
            } else if (key == "defer_loading" && wrapper_defer) {
                fields.emplace_back(key, boolean_option(wrapper_defer, key));
            }
        }
    }
    for (const auto* key : {"description", "parameters"}) {
        if (!json::get(fields, key)) fields.emplace_back(key, json::Value{});
    }
    if (wrapper_defer && !json::get(fields, "defer_loading")) {
        fields.emplace_back("defer_loading", boolean_option(wrapper_defer, "defer_loading"));
    }
    return fields;
}

const json::Value* function_name(const json::OrderedObject* wrapper) {
    const auto* function = wrapper ? json::get(*wrapper, "function") : nullptr;
    const auto* object = function ? function->ordered_object() : nullptr;
    return object ? json::get(*object, "name") : nullptr;
}

json::Array tokenize_tools(const json::Value* value) {
    if (!value) return {};
    const auto* tools = value->array();
    if (!tools) invalid_request("Expected an array of tools");
    json::Array normalized;
    for (const auto& tool : *tools) {
        const auto* wrapper = tool.ordered_object();
        const auto* name = function_name(wrapper);
        const auto* type = wrapper ? json::get(*wrapper, "type") : nullptr;
        if (!name || !name->string() || (type && !type->string())) {
            invalid_request("Expected a tool with a string function name and type");
        }
        const auto* defer = supplied(*wrapper, "defer_loading");
        const auto& function = *json::get(*wrapper, "function")->ordered_object();
        json::OrderedObject fields{{"type", type ? *type : json::Value("function")},
                                  {"function", tool_function(function, defer)}};
        if (defer) fields.emplace_back("defer_loading", boolean_option(defer, "defer_loading"));
        normalized.emplace_back(std::move(fields));
    }
    return normalized;
}

void validate_tool_choice(const json::Value* choice) {
    if (!choice) return;
    const auto text = choice->string();
    const auto* object = choice->ordered_object();
    const auto* type = object ? json::get(*object, "type") : nullptr;
    const auto* function = object ? json::get(*object, "function") : nullptr;
    const auto* name = function_name(object);
    if (text == "auto" || text == "none" || text == "required") return;
    if (!function || !function->ordered_object() || (type && type->string() != "function") ||
        (name && !name->is_null() && !name->string())) {
        invalid_request("Invalid tool_choice");
    }
}

ChatTools select_tools(json::Array tools, const json::Value* choice) {
    ChatTools selection;
    if (choice && choice->string() == "required" && tools.empty()) {
        throw std::invalid_argument("Tools cannot be empty if tool choice is set to required.");
    }
    auto selected = tools.end();
    if (choice && choice->ordered_object()) {
        if (tools.empty()) {
            throw std::invalid_argument(
                "Tools cannot be empty if tool choice is set to a specific tool.");
        }
        const auto* name = function_name(choice->ordered_object());
        const auto requested = name ? name->string() : std::nullopt;
        selected = std::find_if(tools.begin(), tools.end(), [&](const auto& tool) {
            return function_name(tool.ordered_object())->string() == requested;
        });
        if (selected == tools.end()) {
            throw std::invalid_argument("Tool '" + std::string(requested.value_or("None")) +
                                        "' not found in tools list.");
        }
    }
    for (std::size_t index = 0; index < tools.size(); ++index) {
        for (auto& [key, function] : *tools[index].ordered_object()) {
            if (key != "function") continue;
            for (auto& [field, value] : *function.ordered_object()) {
                if (field == "parameters" && !value.is_null()) {
                    value = normalize_tokenization_tool_schema(value, index);
                }
            }
        }
    }
    if (choice && choice->string() == "none") return selection;
    if (selected != tools.end()) {
        const auto name = function_name(selected->ordered_object())->string();
        const std::string selected_name(*name);
        tools.erase(std::remove_if(tools.begin(), tools.end(), [&](const auto& tool) {
            return function_name(tool.ordered_object())->string() != selected_name;
        }), tools.end());
    }
    selection.tools = std::move(tools);
    return selection;
}

json::Value history_arguments(const json::Value& value) {
    if (!value.string()) return value;
    try {
        return parse_chat_arguments(*value.string(), json::Syntax::tool_history);
    } catch (const json::ParseError&) {
        throw std::invalid_argument("Assistant tool call function.arguments must be valid JSON.");
    } catch (const std::invalid_argument&) {
        throw std::invalid_argument(
            "Assistant tool call function.arguments must be a JSON object.");
    }
}

void normalize_history(json::OrderedObject& turn) {
    const auto* value = supplied(turn, "tool_calls");
    if (!value) return;
    json::Array calls;
    for (const auto& call : *value->array()) {
        const auto& function = *json::get(*call.ordered_object(), "function")->ordered_object();
        const auto* args = supplied(function, "arguments");
        calls.emplace_back(json::OrderedObject{{"name", *json::get(function, "name")},
            {"arguments", args ? history_arguments(*args) : json::Value{}}});
    }
    for (auto& [key, field] : turn) {
        if (key == "tool_calls") field = std::move(calls);
    }
}

std::optional<json::Value> reasoning_effort(const json::Value* value, bool allow_boolean) {
    if (!value) return std::nullopt;
    auto number = value->number();
    if (allow_boolean && value->boolean() == false) number = 0;
    if (const auto text = value->string()) {
        if (*text == "none" || *text == "minimal" || *text == "low" || *text == "medium" ||
            *text == "high" || *text == "xhigh" || *text == "ultra" || *text == "max") {
            return *value;
        }
        const auto first = text->find_first_not_of(" \t\r\n\f\v");
        if (first != text->npos) {
            const auto last = text->find_last_not_of(" \t\r\n\f\v");
            auto numeric = text->substr(first, last - first + 1);
            if (numeric.front() == '+') numeric.remove_prefix(1);
            double parsed = 0;
            const auto end = numeric.data() + numeric.size();
            const auto result = std::from_chars(numeric.data(), end, parsed);
            if (result.ec == std::errc{} && result.ptr == end) number = parsed;
        }
    }
    if (number && *number >= 0 && *number <= 0.99) return json::Value(*number);
    return std::nullopt;
}

const json::OrderedObject* object_option(const json::OrderedObject& body, std::string_view key) {
    const auto* value = json::get(body, key);
    return value ? value->ordered_object() : nullptr;
}

std::optional<json::Value> resolved_effort(const json::OrderedObject& body) {
    auto effort = reasoning_effort(supplied(body, "reasoning_effort"), true);
    for (const auto* key : {"reasoning", "thinking"}) {
        const auto* config = object_option(body, key);
        if (!config) continue;
        auto* nested = supplied(*config, "effort");
        if (!nested && std::string_view(key) == "reasoning") {
            nested = supplied(*config, "reasoning_effort");
        }
        if (nested) {
            effort = reasoning_effort(nested, false);
            if (!effort) throw std::invalid_argument(std::string("Invalid ") + key + " effort.");
        }
    }
    return effort;
}

std::string effort_label(const json::Value& value) {
    if (const auto text = value.string()) return std::string(*text);
    if (value.is_null()) return "None";
    if (const auto boolean = value.boolean()) return *boolean ? "True" : "False";
    if (const auto number = value.number()) {
        char buffer[32];
        const auto converted = std::to_chars(buffer, buffer + sizeof(buffer), *number);
        if (converted.ec == std::errc{}) {
            std::string label(buffer, converted.ptr);
            if (!value.signed_integer() && !value.unsigned_integer() &&
                label.find_first_of(".e") == label.npos) {
                label += ".0";
            }
            return label;
        }
    }
    return "value";
}

json::OrderedObject template_options(const json::OrderedObject& body,
                                    const json::OrderedObject& kwargs,
                                    const std::optional<json::Value>& effort, bool generation) {
    if (json::get(kwargs, "add_generation_prompt") || json::get(kwargs, "tools")) {
        throw std::invalid_argument("Duplicate chat template arguments");
    }
    for (const auto* argument : {"tokenize", "return_dict", "conversation"}) {
        if (json::get(kwargs, argument)) {
            throw std::invalid_argument(
                "transformers.tokenization_utils_base.PreTrainedTokenizerBase."
                "apply_chat_template() got multiple values for " +
                std::string(argument == std::string_view("conversation") ? "" : "keyword ") +
                "argument '" + argument + "'");
        }
    }
    if (json::get(kwargs, "messages")) {
        throw std::invalid_argument(
            "jinja2.environment.Template.render() got multiple values for keyword argument "
            "'messages'");
    }
    const auto* mask = json::get(kwargs, "return_assistant_tokens_mask");
    if (mask && truthy(*mask)) {
        throw std::invalid_argument(
            "`return_assistant_tokens_mask=True` requires `return_dict=True` and `tokenize=True`");
    }
    const auto* thinking = json::get(kwargs, "enable_thinking");
    const auto* preserve = json::get(kwargs, "preserve_thinking");
    const auto* vision = json::get(kwargs, "add_vision_id");
    const auto* config = object_option(body, "thinking");
    const auto* type = config ? supplied(*config, "type") : nullptr;
    const bool default_thinking = type ? type->string() != "disabled"
                                       : !effort || effort->string() != "none";
    const bool enable = thinking ? thinking->boolean().value_or(true) : default_thinking;
    json::OrderedObject options{{"add_generation_prompt", generation},
                                {"enable_thinking", enable},
                                {"preserve_thinking", !preserve || preserve->boolean() == true},
                                {"add_vision_id", vision && truthy(*vision)}};
    if (thinking && !thinking->boolean().has_value()) {
        options.emplace_back("reasoning_effort", "medium");
    } else if (enable) {
        const auto* effective = json::get(kwargs, "reasoning_effort");
        if (!effective && effort) effective = &*effort;
        if (effective) {
            const auto tier = effective->string();
            if (tier != "xhigh" && tier != "medium" && tier != "low") {
                throw std::invalid_argument("Unexpected reasoning effort " +
                    effort_label(*effective) +
                    ". Supported types are xhigh (default), medium, and low.");
            }
            options.emplace_back("reasoning_effort", *effective);
        }
    }
    return options;
}

constexpr std::string_view continuation_tag = "CONTINUE_FINAL_MESSAGE_TAG";

std::string mark_continuation(json::Value& value) {
    if (const auto text = value.string()) {
        std::string original(*text);
        value = original + std::string(continuation_tag) + ' ';
        return original;
    }
    if (auto* parts = value.array()) {
        for (auto item = parts->rbegin(); item != parts->rend(); ++item) {
            for (auto& [key, field] : *item->ordered_object()) {
                if (key == "text") return mark_continuation(field);
            }
        }
    }
    throw std::invalid_argument(
        "continue_final_message is set but we could not find any text to continue "
        "in the final message!");
}

std::optional<std::string> nested_continuation(json::OrderedObject& last,
                                               const json::OrderedObject& kwargs,
                                               bool generation) {
    const auto* setting = json::get(kwargs, "continue_final_message");
    if (!setting || !truthy(*setting)) return std::nullopt;
    if (generation) {
        throw std::invalid_argument(
            "continue_final_message and add_generation_prompt are not compatible. Use "
            "continue_final_message when you want the model to continue the final message, "
            "and add_generation_prompt when you want to add a header that will prompt it "
            "to start a new assistant message instead.");
    }
    const auto requested = setting->string().value_or("content");
    const auto key = requested == "reasoning" || requested == "thinking"
        ? std::string_view("reasoning_content") : requested;
    for (auto& [name, value] : last) {
        if (name != key || value.is_null()) continue;
        if (key == "content" || key == "reasoning_content") return mark_continuation(value);
        throw std::invalid_argument("continue_final_message is set to \"" +
            std::string(requested) + "\" but this is not an accepted field in the chat_template");
    }
    throw std::invalid_argument("continue_final_message is set but the final message has no \"" +
                                std::string(requested) + "\" to continue!");
}

void finish_continuation(std::string& prompt, std::string_view original) {
    const auto tag = prompt.rfind(continuation_tag);
    const auto trimmed = trim_chat_text(original);
    if (tag == prompt.npos || prompt.find(trimmed) == prompt.npos) {
        throw std::invalid_argument(
            "continue_final_message is set but the final message does not appear in the chat "
            "after applying the chat template! This can happen if the chat template deletes "
            "portions of the final message. Please verify the chat template and final message "
            "in your chat to ensure they are compatible.Final message to continue: " +
            std::string(trimmed) + "\nRendered chat:\n" + prompt);
    }
    const auto next = tag + continuation_tag.size();
    const bool preserve = next < prompt.size() && prompt[next] == ' ';
    prompt.resize(tag);
    if (!preserve) {
        const auto retained = trim_chat_text(prompt);
        prompt.resize(retained.empty() ? 0 : retained.data() - prompt.data() + retained.size());
    }
}

struct ChatTokenization {
    std::vector<json::OrderedObject> turns;
    json::Array tools;
    std::optional<bool> generation;
    bool continuation;
};

ChatTokenization read_chat_tokenization(const json::OrderedObject& body, const json::Value& value) {
    const auto* messages = value.array();
    if (!messages) invalid_request("Expected messages to be an array");
    const auto* effort = supplied(body, "reasoning_effort");
    if (effort && !reasoning_effort(effort, true)) invalid_request("Invalid reasoning_effort");
    const auto* generation = supplied(body, "add_generation_prompt");
    const auto* kwargs = supplied(body, "chat_template_kwargs");
    if (kwargs && !kwargs->ordered_object()) {
        invalid_request("Expected a chat_template_kwargs object");
    }
    validate_tool_choice(supplied(body, "tool_choice"));
    ChatTokenization chat{{}, tokenize_tools(supplied(body, "tools")),
        generation ? std::optional(boolean_option(generation, "add_generation_prompt"))
                   : std::nullopt,
        boolean_option(json::get(body, "continue_final_message"), "continue_final_message", false)};
    chat.turns.reserve(messages->size());
    for (const auto& message : *messages) {
        chat.turns.push_back(chat_turn(message));
    }
    return chat;
}

std::vector<TokenId> tokenize_chat(const CompletionTokenizer& loader,
                                    const json::OrderedObject& body, ChatTokenization chat) {
    if (const auto* stream = json::get(body, "stream"); stream && stream->boolean() != false &&
        stream->string() != "0" && stream->number() != 0) {
        throw std::invalid_argument("stream must be a boolean");
    }
    const auto effort = resolved_effort(body);
    if (chat.turns.empty()) throw std::invalid_argument("Messages cannot be empty.");
    validate_chat_generation(body);
    const json::OrderedObject defaults;
    const auto* settings = object_option(body, "chat_template_kwargs");
    const auto& kwargs = settings ? *settings : defaults;
    if (json::get(kwargs, "chat_template")) {
        throw std::invalid_argument(
            "Request-supplied chat_template is not supported by this server.");
    }
    const auto tools = select_tools(std::move(chat.tools), supplied(body, "tool_choice"));
    if (chat.continuation && chat.generation == false) {
        throw std::invalid_argument("continue_final_message needs add_generation_prompt "
                                    "to open the assistant turn.");
    }
    if (supplied(body, "input_ids")) {
        throw std::invalid_argument("Chat tokenization does not support input_ids.");
    }
    auto& last = chat.turns.back();
    std::string prefix;
    for (auto& turn : chat.turns) {
        if (json::get(turn, "tools")) {
            throw std::invalid_argument("Chat tokenization does not support message-level tools.");
        }
        const auto role = json::get(turn, "role")->string();
        if (role == "assistant") normalize_history(turn);
    }
    if ((!chat.generation || chat.continuation) &&
        json::get(last, "role")->string() == "assistant") {
        if (const auto text = json::get(last, "content")->string()) {
            if (chat.continuation) {
                prefix = *text;
                chat.turns.pop_back();
            } else {
                last = {{"role", "user"}, {"content", std::string(*text)}};
            }
        }
    }
    if (chat.turns.empty()) {
        throw std::invalid_argument(
            "Cannot apply chat template to an empty conversation. Provide at least one message.");
    }
    const bool generation = chat.generation.value_or(true);
    auto options = template_options(body, kwargs, effort, generation);
    const auto continuation = nested_continuation(chat.turns.back(), kwargs, generation);
    json::Array messages;
    for (auto& turn : chat.turns) {
        const auto role = json::get(turn, "role")->string();
        if (role == "root" || role == "developer") {
            throw std::invalid_argument("Unexpected message role.");
        }
        messages.emplace_back(std::move(turn));
    }
    options.emplace_back("messages", std::move(messages));
    auto prompt = loader.chat_template().render(options, ChatInput::http, tools);
    if (continuation) finish_continuation(prompt, *continuation);
    const auto& tokenizer = loader.get();
    auto ids = tokenizer.encode(prompt);
    if (!prefix.empty()) {
        const auto tail = tokenizer.encode(prefix);
        ids.insert(ids.end(), tail.begin(), tail.end());
    }
    return ids;
}

void tokenize(const Socket& connection, const CompletionTokenizer& loader,
              const json::OrderedObject& body) {
    const auto* prompt = supplied(body, "prompt");
    const auto* messages = supplied(body, "messages");
    if ((prompt == nullptr) == (messages == nullptr)) {
        invalid_request("Exactly one of 'prompt' or 'messages' must be provided.");
    }
    const auto text = prompt ? prompt->string() : std::nullopt;
    const auto* texts = prompt ? prompt->array() : nullptr;
    if (prompt && !text && (!texts || !std::all_of(texts->begin(), texts->end(),
        [](const auto& item) { return item.string().has_value(); }))) {
        invalid_request("Expected prompt to be a string or an array of strings");
    }
    boolean_option(json::get(body, "add_special_tokens"), "add_special_tokens");
    auto chat = messages ? std::optional(read_chat_tokenization(body, *messages)) : std::nullopt;
    if (const auto* stream = json::get(body, "stream"); stream && truthy(*stream)) {
        write_tokenization_error(connection, "501 Not Implemented", "NotImplementedError",
                                 "OpenAIServingTokenize does not support streaming requests");
        return;
    }
    std::string output = R"({"tokens":)";
    std::string count;
    if (texts) {
        const auto& tokenizer = loader.get();
        output += '[';
        count += '[';
        for (const auto& item : *texts) {
            if (&item != &texts->front()) {
                output += ',';
                count += ',';
            }
            const auto tokens = tokenizer.encode(*item.string());
            append_tokens(output, tokens);
            count += std::to_string(tokens.size());
        }
        output += ']';
        count += ']';
    } else {
        std::vector<TokenId> tokens;
        if (chat) {
            try {
                tokens = tokenize_chat(loader, body, std::move(*chat));
            } catch (const std::invalid_argument& error) {
                write_tokenization_error(connection, bad_request, "BadRequestError", error.what());
                return;
            }
        } else {
            tokens = loader.get().encode(*text);
        }
        append_tokens(output, tokens);
        count = std::to_string(tokens.size());
    }
    output += R"(,"count":)" + count;
    output += R"(,"max_model_len":)" + std::to_string(checkpoint_context) + '}';
    write_json(connection, "200 OK", output);
}

bool integer_space(char32_t scalar) {
    return (scalar >= U'\t' && scalar <= U'\r') || scalar == U' ' || scalar == 0x85 ||
           scalar == 0xa0 || scalar == 0x1680 || (scalar >= 0x2000 && scalar <= 0x200a) ||
           scalar == 0x2028 || scalar == 0x2029 || scalar == 0x202f || scalar == 0x205f ||
           scalar == 0x3000;
}

bool digit(char32_t scalar) {
    return scalar >= U'0' && scalar <= U'9';
}

[[noreturn]] void invalid_ids() {
    throw HttpError(bad_request, "Expected tokens to be integers in a flat or nested array");
}

std::string integer_text(std::string_view text) {
    const auto scalars = read_utf8(text);
    std::u32string_view input(scalars);
    while (!input.empty() && integer_space(input.front())) {
        input.remove_prefix(1);
    }
    while (!input.empty() && integer_space(input.back())) {
        input.remove_suffix(1);
    }
    const bool negative = !input.empty() && input.front() == U'-';
    if (!input.empty() && (negative || input.front() == U'+')) {
        input.remove_prefix(1);
    }
    if (input.empty() || !digit(input.front())) {
        invalid_ids();
    }
    const auto point = input.find(U'.');
    const auto whole = input.substr(0, point);
    if (point != std::u32string_view::npos) {
        const auto fraction = input.substr(point + 1);
        if (fraction.empty() || fraction.find_first_not_of(U'0') != std::u32string_view::npos) {
            invalid_ids();
        }
    }
    std::string digits;
    for (std::size_t index = 0; index < whole.size(); ++index) {
        const auto scalar = whole[index];
        if (digit(scalar)) {
            if (scalar != U'0' || !digits.empty()) {
                digits += static_cast<char>(scalar);
            }
        } else if (scalar != U'_' || !digit(whole[index - 1]) ||
                   index + 1 == whole.size() || !digit(whole[index + 1])) {
            invalid_ids();
        }
    }
    if (digits.size() > 4300) {
        invalid_ids();
    }
    if (digits.empty()) {
        return "0";
    }
    if (negative) {
        digits.insert(digits.begin(), '-');
    }
    return digits;
}

TokenId checked_id(const json::Value& value, std::optional<std::string>& invalid) {
    if (const auto integer = value.unsigned_integer(); integer && *integer < model_vocab_size) {
        return static_cast<TokenId>(*integer);
    }
    if (const auto boolean = value.boolean()) {
        return *boolean;
    }
    std::string digits;
    if (const auto integer = value.signed_integer()) {
        digits = std::to_string(*integer);
    } else if (const auto integer = value.unsigned_integer()) {
        digits = std::to_string(*integer);
    } else if (const auto* wide = value.wide_integer()) {
        digits = wide->digits;
    } else if (const auto text = value.string()) {
        digits = integer_text(*text);
    } else if (const auto number = value.number(); number && std::trunc(*number) == *number &&
               *number > -0x1p63 && *number < 0x1p63) {
        const auto integer = static_cast<std::int64_t>(*number);
        if (integer >= 0 && integer < model_vocab_size) {
            return static_cast<TokenId>(integer);
        }
        digits = std::to_string(integer);
    } else {
        invalid_ids();
    }
    TokenId id = 0;
    const auto* end = digits.data() + digits.size();
    const auto parsed = std::from_chars(digits.data(), end, id);
    if (parsed.ec == std::errc{} && parsed.ptr == end && id < model_vocab_size) {
        return id;
    }
    if (!invalid) {
        invalid = std::move(digits);
    }
    return 0;
}

std::vector<TokenId> read_ids(const json::Array& values, std::optional<std::string>& invalid) {
    std::vector<TokenId> ids;
    ids.reserve(values.size());
    for (const auto& value : values) {
        ids.push_back(checked_id(value, invalid));
    }
    return ids;
}

void append_text(std::string& output, const Tokenizer& tokenizer,
                 const std::vector<TokenId>& ids, bool skip_special) {
    Utf8Stream utf8;
    std::string text;
    for (const auto id : ids) {
        text += utf8.push(tokenizer.piece(id, skip_special));
    }
    text += utf8.finish();
    json::append_quoted(output, text);
}

void detokenize(const Socket& connection, const CompletionTokenizer& loader,
                const json::Object& body) {
    const auto* tokens = json::get(body, "tokens");
    const auto* values = tokens ? tokens->array() : nullptr;
    if (!values) {
        invalid_ids();
    }
    const bool skip_special = boolean_option(json::get(body, "skip_special_tokens"),
                                             "skip_special_tokens");
    if (values->empty()) {
        write_json(connection, "200 OK", R"({"text":""})");
        return;
    }
    const bool nested = values->front().array() != nullptr;
    std::optional<std::string> invalid;
    std::vector<std::vector<TokenId>> items;
    if (nested) {
        items.reserve(values->size());
        for (const auto& value : *values) {
            const auto* item = value.array();
            if (!item) {
                invalid_ids();
            }
            items.push_back(read_ids(*item, invalid));
        }
    } else {
        items.push_back(read_ids(*values, invalid));
    }
    if (invalid) {
        write_tokenization_error(connection, bad_request, "BadRequestError",
            "Invalid input: token id " + *invalid + " is out of vocabulary; valid range is [0, " +
            std::to_string(model_vocab_size) + ").");
        return;
    }
    const auto& tokenizer = loader.get();
    std::string output = R"({"text":)";
    if (nested) {
        output += '[';
    }
    for (const auto& ids : items) {
        if (&ids != &items.front()) {
            output += ',';
        }
        append_text(output, tokenizer, ids, skip_special);
    }
    if (nested) {
        output += ']';
    }
    output += '}';
    write_json(connection, "200 OK", output);
}

json::Value read_tokenize_body(std::string_view body) {
    try {
        return json::parse(body, json::Syntax::ordered);
    } catch (const json::NumberRangeError&) {
        auto value = json::parse(body, json::Syntax::chat_tokenization);
        const auto* object = value.ordered_object();
        if (!object || !supplied(*object, "messages") || supplied(*object, "prompt")) {
            throw;
        }
        return value;
    }
}

}

void serve_tokenization(const Socket& connection, TokenizationRoute route,
                        const CompletionTokenizer& tokenizer, std::string_view body) {
    try {
        const bool encode = route == TokenizationRoute::tokenize;
        const auto value = encode ? read_tokenize_body(body)
                                  : json::parse(body, json::Syntax::completion);
        const auto* ordered = value.ordered_object();
        const auto* object = value.object();
        if (!ordered && !object) {
            throw HttpError(bad_request, "Expected a JSON request object");
        }
        const auto* model = encode ? json::get(*ordered, "model") : json::get(*object, "model");
        if (model && !model->string()) {
            throw HttpError(bad_request, "model must be a string");
        }
        if (encode) {
            tokenize(connection, tokenizer, *ordered);
        } else {
            detokenize(connection, tokenizer, *object);
        }
    } catch (const HttpError& error) {
        write_tokenization_error(connection, error.status, "Bad Request", error.what());
    } catch (const json::ParseError&) {
        write_tokenization_error(connection, bad_request, "Bad Request",
                                 "Invalid JSON request body");
    } catch (const std::exception& error) {
        std::cerr << "Tokenization request failed: " << error.what() << '\n';
        write_tokenization_error(connection, "500 Internal Server Error", "InternalServerError",
                                 "Checkpoint tokenization failed");
    }
}

}
