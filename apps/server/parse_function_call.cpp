#include "sglang/server/parse_function_call.h"

#include "sglang/json.h"
#include "sglang/server/http.h"
#include "sglang/server/serving/qwen_parser.h"
#include "sglang/utf8.h"

#include <optional>
#include <utility>

namespace sglang::server {
namespace {

constexpr std::string_view bad_request = "400 Bad Request";

std::optional<std::string> unsupported_tokenizer(std::string_view parser) {
    if (parser == "deepseek-v4" || parser == "deepseekv4" || parser == "deepseek-v41" ||
        parser == "deepseekv41") {
        return "'" + std::string(parser) +
               "' output is parsed with its model's tokenizer; "
               "this server does not serve that model";
    }
    return std::nullopt;
}

std::string parser_error(std::string_view parser) {
    if (parser == "qwen25") return "qwen25 was removed in SGLang 1.0; use qwen";
    if (parser == "glm45") return "glm45 was removed in SGLang 1.0; use glm";
    if (auto message = unsupported_tokenizer(parser))
        return std::move(*message);
    return "Unsupported tool_call_parser: " + std::string(parser);
}

std::string lowercase_ascii(std::string_view text) {
    std::string result(text);
    for (auto& character : result) {
        if (character >= 'A' && character <= 'Z') {
            character += 'a' - 'A';
        }
    }
    return result;
}

bool return_reasoning_blocks(const json::Value* value) {
    if (value == nullptr) {
        return false;
    }
    if (const auto flag = value->boolean()) {
        return *flag;
    }
    if (const auto number = value->number(); number == 0 || number == 1) {
        return number == 1;
    }
    if (const auto text = value->string()) {
        const auto flag = lowercase_ascii(*text);
        if (flag == "1" || flag == "true" || flag == "t" || flag == "yes" || flag == "y" ||
            flag == "on") {
            return true;
        }
        if (flag == "0" || flag == "false" || flag == "f" || flag == "no" || flag == "n" ||
            flag == "off") {
            return false;
        }
    }
    throw std::invalid_argument("return_blocks: Input should be a valid boolean");
}

void write_separated_reasoning(const Socket& connection, const SeparatedReasoning& parsed,
                               bool return_blocks) {
    std::string response = R"({"reasoning_text":)";
    json::append_quoted(response, parsed.reasoning, true);
    response += R"(,"text":)";
    json::append_quoted(response, parsed.text, true);
    if (return_blocks) {
        response += R"(,"reasoning_blocks":[)";
        if (!parsed.reasoning.empty()) {
            json::append_quoted(response, parsed.reasoning, true);
        }
        response += R"(],"text_blocks":[)";
        json::append_quoted(response, parsed.text, true);
        response += R"(],"blocks":[)";
        if (!parsed.reasoning.empty()) {
            response += R"({"type":"reasoning","text":)";
            json::append_quoted(response, parsed.reasoning, true);
            response += "},";
        }
        response += R"({"type":"text","text":)";
        json::append_quoted(response, parsed.text, true);
        response += "}]";
    }
    write_json(connection, "200 OK", response + '}');
}
}

void parse_function_call(const Socket& connection, std::string_view body) {
    try {
        constexpr std::string_view utf8_bom = "\xef\xbb\xbf";
        if (body.substr(0, utf8_bom.size()) == utf8_bom) {
            body.remove_prefix(utf8_bom.size());
            if (body.empty()) {
                throw json::ParseError("Empty JSON document after UTF-8 BOM");
            }
        }
        const auto input = body.empty() ? json::Value{} :
            json::parse(body, json::Syntax::ordered_last_key_wins);
        if (input.is_null()) throw std::invalid_argument("body: Field required");
        const auto* object = input.ordered_object();
        if (!object) throw std::invalid_argument("body: Input should be a valid dictionary");
        const auto* text = json::get(*object, "text");
        if (!text) throw std::invalid_argument("text: Field required");
        if (!text->string()) throw std::invalid_argument("text: Input should be a valid string");
        for (const auto key : {"tool_call_parser", "rid", "http_worker_ipc"}) {
            if (const auto* value = json::get(*object, key);
                value && !value->is_null() && !value->string()) {
                throw std::invalid_argument(std::string(key) + ": Input should be a valid string");
            }
        }
        const auto* tools = json::get(*object, "tools");
        if (tools && !tools->array()) {
            throw std::invalid_argument("tools: Input should be a valid list");
        }
        const auto* parser_field = json::get(*object, "tool_call_parser");
        const auto parser = parser_field ? parser_field->string().value_or("None") : "None";
        if (const auto response = model::parse_qwen_tool_calls(*text->string(),
                                                               tools ? tools->array() : nullptr,
                                                               parser)) {
            write_json(connection, "200 OK", *response);
        } else {
            write_tokenization_error(connection, bad_request, "400", parser_error(parser));
        }
    } catch (const json::ParseError& error) {
        if (valid_utf8(body)) {
            write_tokenization_error(connection, bad_request, "Bad Request",
                                     std::string("JSON decode error: ") + error.what());
        } else {
            write_tokenization_error(connection, bad_request, "400",
                                     "There was an error parsing the body");
        }
    } catch (const std::invalid_argument& error) {
        write_tokenization_error(connection, bad_request, "Bad Request", error.what());
    }
}

void separate_reasoning(const Socket& connection, std::string_view body) {
    try {
        if (!valid_utf8(body)) {
            throw std::invalid_argument("JSON decode error: Invalid UTF-8");
        }
        const auto input =
            body.empty() ? json::Value{} : json::parse(body, json::Syntax::last_key_wins);
        if (input.is_null()) {
            throw std::invalid_argument("body: Field required");
        }
        const auto* object = input.object();
        if (object == nullptr) {
            throw std::invalid_argument("Input should be a valid dictionary");
        }
        for (const std::string_view key : {"rid", "http_worker_ipc"}) {
            if (const auto* value = json::get(*object, key);
                value != nullptr && !value->string() && !value->is_null()) {
                throw std::invalid_argument(std::string(key) + ": Input should be a valid string");
            }
        }
        const auto required_string = [object](std::string_view key) {
            const auto* value = json::get(*object, key);
            if (value == nullptr) {
                throw std::invalid_argument(std::string(key) + ": Field required");
            }
            if (const auto string = value->string()) {
                return *string;
            }
            throw std::invalid_argument(std::string(key) + ": Input should be a valid string");
        };
        const auto text = required_string("text");
        const auto parser = required_string("reasoning_parser");
        const bool return_blocks = return_reasoning_blocks(json::get(*object, "return_blocks"));
        if (const auto message = unsupported_tokenizer(parser)) {
            write_tokenization_error(connection, bad_request, "400", *message);
            return;
        }
        const auto result = model::separate_qwen_reasoning(text, parser);
        if (!result) {
            write_internal_server_error(connection);
            return;
        }
        write_separated_reasoning(connection, *result, return_blocks);
    } catch (const json::ParseError& error) {
        write_tokenization_error(connection, bad_request, "Bad Request",
                                 "JSON decode error: " + std::string(error.what()));
    } catch (const std::invalid_argument& error) {
        write_tokenization_error(connection, bad_request, "Bad Request", error.what());
    }
}
}
