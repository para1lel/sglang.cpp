#include "sglang/server/parse_function_call.h"

#include "sglang/json.h"
#include "sglang/server/http.h"
#include "sglang/server/serving/qwen_parser.h"
#include "sglang/utf8.h"

namespace sglang::server {
namespace {

constexpr std::string_view bad_request = "400 Bad Request";

std::string parser_error(std::string_view parser) {
    if (parser == "qwen25") return "qwen25 was removed in SGLang 1.0; use qwen";
    if (parser == "glm45") return "glm45 was removed in SGLang 1.0; use glm";
    if (parser == "deepseek-v4" || parser == "deepseekv4" ||
        parser == "deepseek-v41" || parser == "deepseekv41") {
        return "'" + std::string(parser) + "' output is parsed with its model's tokenizer; "
               "this server does not serve that model";
    }
    return "Unsupported tool_call_parser: " + std::string(parser);
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

}
