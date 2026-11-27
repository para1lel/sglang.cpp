#include "sglang/server/serving/qwen_serving_profile.h"

#include "sglang/decoder_constraint.h"
#include "sglang/runtime/model/qwen_text_model.h"
#include "sglang/runtime/model/qwen_tokenizer.h"
#include "sglang/server/chat_template.h"
#include "sglang/server/chat_tool_constraints.h"
#include "sglang/server/http.h"
#include "sglang/server/text_completion.h"

#include <algorithm>

namespace sglang::server {
namespace {

[[noreturn]] void bad_request(const char* message) {
    throw HttpError("400 Bad Request", message);
}

bool is_qwen_chat_template_option(std::string_view key) {
    return key == "enable_thinking" || key == "preserve_thinking" ||
        key == "reasoning_effort" || key == "add_vision_id";
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
        if (!is_qwen_chat_template_option(key)) {
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

void start_qwen_chat_output(std::optional<ChatOutputParser>& parser,
                            const ChatOutputOptions& options) {
    parser.emplace(options);
}

}
}

namespace sglang::model {

const ServingProfile qwen_serving_profile{
    "/data0/shared/Qwen3.8-27B-FP8",
    qwen_served_name,
    decoder::QwenTextModel::model_type,
    decoder::QwenTextModel::architecture,
    true,
    false,
    false,
    qwen_reasoning_parser,
    qwen_tool_call_parser,
    "safetensors",
    "default",
    qwen_tokenizer_file,
    server::normalize_template_options,
    server::is_qwen_chat_template_option,
    server::select_qwen_chat_tools,
    server::load_qwen_chat_template,
    server::set_chat_output,
    server::start_qwen_chat_output,
};

}
