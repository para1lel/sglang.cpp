#include "sglang/server/chat_tool_constraints.h"

#include "sglang/decoder_constraint.h"
#include "sglang/server/chat_template.h"
#include "sglang/tokenizer/json_schema_pattern.h"

#include <stdexcept>

namespace sglang::server {
namespace {

constexpr std::string_view tool_trigger = "<tool_call>";
constexpr std::string_view think_close = "</think>";
constexpr std::size_t max_cached_tool_constraints = 8;

std::string loose_parameters() {
    std::string pattern = "(?:\n|(?:<parameter=[^<>\\x00-\\x1f]+>\n";
    pattern += tokenizer::xml_string_pattern;
    pattern += "\n</parameter>\n)+)";
    return pattern;
}

std::string function_pattern(const json::Value& tool) {
    const auto& function = *get_ordered(tool, "function");
    const auto name = get_ordered(function, "name")->string();
    if (!name) {
        throw std::invalid_argument("Expected unique function tools with valid names");
    }
    const auto* strict = get_ordered(function, "strict");
    const bool constrained = strict && strict->boolean().value_or(false);
    auto pattern = tokenizer::regex_literal("\n<function=" + std::string(*name) + ">\n");
    try {
        const auto* parameters = get_ordered(function, "parameters");
        pattern += constrained ? tokenizer::xml_parameters_pattern(parameters) : loose_parameters();
    } catch (const std::invalid_argument& error) {
        throw std::invalid_argument("Strict tool " + std::string(*name) + ": " + error.what());
    }
    pattern += tokenizer::regex_literal("</function>\n</tool_call>");
    return pattern;
}

TriggeredConstraintMode tool_mode(ChatToolChoice choice) {
    switch (choice) {
    case ChatToolChoice::named: return TriggeredConstraintMode::single;
    case ChatToolChoice::required: return TriggeredConstraintMode::required;
    default: return TriggeredConstraintMode::optional;
    }
}

std::string tool_call_pattern(const ChatTools& tools) {
    std::string pattern = "(?:";
    const char* separator = "";
    for (const auto& tool : tools.tools) {
        pattern += separator;
        pattern += function_pattern(tool);
        separator = "|";
    }
    return pattern + ')';
}

}

bool tool_constraint_applies(const ChatTools& tools) {
    const bool constrained = tools.forced() ||
        (tools.choice == ChatToolChoice::automatic && tools.strict);
    return constrained && !tools.tools.empty();
}

std::shared_ptr<const TokenConstraint> CompletionTokenizer::tool_constraint(
    const ChatTools& tools, bool prompt_has_open_think) const {
    if (!tool_constraint_applies(tools)) {
        return nullptr;
    }
    auto pattern = tool_call_pattern(tools);
    const auto mode = tool_mode(tools.choice);
    const bool parallel = tools.parallel && tools.choice != ChatToolChoice::named;
    const std::lock_guard lock(tool_constraints_mutex_);
    const auto match = std::find_if(tool_constraints_.begin(), tool_constraints_.end(),
        [&](const ToolConstraintEntry& entry) {
            return entry.mode == mode && entry.parallel == parallel &&
                entry.prompt_has_open_think == prompt_has_open_think && entry.pattern == pattern;
        });
    if (match != tool_constraints_.end()) {
        tool_constraints_.splice(tool_constraints_.begin(), tool_constraints_, match);
        return match->constraint;
    }
    auto constraint = TokenConstraint::compile_triggered(
        get(), pattern, tool_trigger, mode, parallel,
        prompt_has_open_think ? think_close : std::string_view{});
    tool_constraints_.push_front(
        {std::move(pattern), mode, parallel, prompt_has_open_think, constraint});
    if (tool_constraints_.size() > max_cached_tool_constraints) {
        tool_constraints_.pop_back();
    }
    return constraint;
}

}
