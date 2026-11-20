#include "sglang/server/chat_output.h"
#include "sglang/tokenizer/tokenizer.h"
#include "sglang/utf8.h"

#include <algorithm>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace {

using namespace sglang::server;
using Kind = ChatOutputKind;
using Row = std::tuple<Kind, std::string, std::size_t, std::string>;
using Rows = std::vector<Row>;
using Chunks = std::vector<std::string_view>;

struct Case {
    std::string name;
    std::string input;
    Rows expected;
    ChatEnd ending = ChatEnd::stop;
    bool prompt_has_open_think = false;
    bool streamed_reasoning = true;
};

Row content(std::string text) { return {Kind::content, std::move(text), 0, {}}; }
Row reasoning(std::string text) { return {Kind::reasoning, std::move(text), 0, {}}; }
Row call(std::size_t index, std::string name, std::string args) {
    return {Kind::tool_call, std::move(args), index, std::move(name)};
}

const std::string ping = "<tool_call><function=ping></function></tool_call>";
const std::string half = "<tool_call><function=ping>";
const std::string complex = R"(<tool_call>
<function=complex>
<parameter=note>
café 😊
a "quote" and \ slash
</parameter>
<parameter=integer>03</parameter>
<parameter=amount>2e0</parameter>
<parameter=ok>TrUe</parameter>
<parameter=object>{"nested":["雪", "x\"</parameter>"]}</parameter>
<parameter=array>[1,{"a":false}]</parameter>
<parameter=union>03</parameter>
<parameter=maybe>null</parameter>
</function>
</tool_call>)";
const std::string converted =
    R"({"note":"café 😊\na \"quote\" and \\ slash","integer":3,"amount":2.0,"ok":true,)"
    R"("object":{"nested":["雪", "x\"</parameter>"]},"array":[1,{"a":false}],)"
    R"("union":"03","maybe":null})";

std::vector<Case> cases() {
    const auto truncated = std::string("<tool_call><function=complex><parameter=note>unfinished");
    const auto malformed = half + "</function><function=ping><parameter=key>"
        R"({"value":"</tool_call>)" + ping + R"("}</parameter></function></tool_call>)";
    const auto literal = "<tool_call><function=complex><parameter=integer>1</parameter>"
        "<parameter=note>keep " + ping + " as text</parameter></function></tool_call>";
    const auto double_sign = "<tool_call><function=complex><parameter=integer>1</parameter>"
        "<parameter=amount>+-1</parameter></function></tool_call>";
    return {
        {"plain", "café 😊 <tool_caX><function=ping>literal", {
            content("café 😊 <tool_caX><function=ping>literal")}},
        {"prompt-reasoning", "café</think>\n\nvisible 😊", {
            reasoning("café"), content("\n\nvisible 😊")}, ChatEnd::stop, true},
        {"empty-reasoning", "</think>answer", {content("answer")}, ChatEnd::stop, true},
        {"self-open", "<think>café</think>visible", {reasoning("café"), content("visible")}},
        {"late-open", "prefix<think>hidden</think>answer", {
            content("prefix<think>hidden</think>answer")}},
        {"implicit-close", "checking" + complex, {
            reasoning("checking"), call(0, "complex", converted)}, ChatEnd::stop, true},
        {"buffered", "<think>checking</think>reply", {
            reasoning("checking"), content("reply")}, ChatEnd::stop, false, false},
        {"parallel", "begin" + ping + "\n中" + complex + "end", {
            content("begin"), call(0, "ping", "{}"), content("\n中"),
            call(1, "complex", converted), content("end")}},
        {"unknown", "<tool_call><function=ghost></function></tool_call>" + ping, {
            content("<tool_call><function=ghost></function></tool_call>"), call(0, "ping", "{}")}},
        {"two-functions-in-frame", half + "</function><function=ping></function></tool_call>" +
            ping, {content(half + "</function><function=ping></function></tool_call>"),
                   call(0, "ping", "{}")}},
        {"bad-types", "<tool_call><function=complex><parameter=integer>03e1</parameter>"
                      "</function></tool_call>", {content(
                      "<tool_call><function=complex><parameter=integer>03e1</parameter>"
                      "</function></tool_call>")}},
        {"duplicate", "<tool_call><function=complex><parameter=integer>1</parameter>"
                      "<parameter=integer>2</parameter></function></tool_call>", {content(
                      "<tool_call><function=complex><parameter=integer>1</parameter>"
                      "<parameter=integer>2</parameter></function></tool_call>")}},
        {"unknown-then-valid", "<tool_call><function=ghost><parameter=object>"
            R"({"tag":"</tool_call><tool_call><function=ping></function></tool_call>"})"
            "</parameter></function></tool_call>" + ping, {content(
            "<tool_call><function=ghost><parameter=object>"
            R"({"tag":"</tool_call><tool_call><function=ping></function></tool_call>"})"
            "</parameter></function></tool_call>"), call(0, "ping", "{}")}},
        {"malformed-quoted-markers", malformed + ping, {
            content(malformed), call(0, "ping", "{}")}},
        {"quoted-argument-marker", literal, {
            call(0, "complex", R"({"integer":1,"note":"keep )" + ping + " as text\"}")}},
        {"nested-in-unclosed-frame", half + ping, {content(half), call(0, "ping", "{}")}},
        {"nested-after-truncated-value", truncated + ping, {
            content(truncated), call(0, "ping", "{}")}},
        {"cancel-with-ambiguous-call", truncated + ping, {}, ChatEnd::cancel},
        {"bad-number-double-sign", double_sign + ping, {
            content(double_sign), call(0, "ping", "{}")}},
        {"single-positive-sign", "<tool_call><function=complex><parameter=integer>+03</parameter>"
            "<parameter=amount>+2.5</parameter></function></tool_call>", {
                call(0, "complex", R"({"integer":3,"amount":2.5})")}},
        {"case-insensitive-typed-null", "<tool_call><function=complex>"
            "<parameter=integer>nUlL</parameter><parameter=amount>NULL</parameter>"
            "<parameter=note>NuLl</parameter><parameter=ok>nuLL</parameter>"
            "<parameter=object>nULL</parameter></function></tool_call>", {
                call(0, "complex", R"({"integer":null,"amount":null,"note":null,"ok":null,)"
                                   R"("object":null})")}},
        {"missing-parameter-close", "<tool_call><function=complex><parameter=integer>1"
                                    "</function></tool_call>" + ping, {content(
                                    "<tool_call><function=complex><parameter=integer>1"
                                    "</function></tool_call>"), call(0, "ping", "{}")}},
        {"unknown-parameter", "<tool_call><function=loose><parameter=extra>raw \"雪\"</parameter>"
                              "</function></tool_call>", {
                              call(0, "loose", R"({"extra":"raw \"雪\""})")}},
        {"length-partial-after-call", ping + half, {
            call(0, "ping", "{}"), content(half)}, ChatEnd::length},
        {"stop-partial-tool", half, {content(half)}},
        {"length-partial-think", "checking</thi", {reasoning("checking</thi")},
           ChatEnd::length, true, false},
        {"cancel-marker", "<tool_ca", {}, ChatEnd::cancel},
        {"cancel-open-reasoning", "checking</th", {reasoning("checking")},
           ChatEnd::cancel, true},
        {"cancel-buffered-reasoning", "checking</th", {}, ChatEnd::cancel, true, false},
        {"cancel-after-call", ping + half, {call(0, "ping", "{}")}, ChatEnd::cancel},
    };
}

void add_parameter_boundary_cases(std::vector<Case>& fixtures) {
    const std::string bad_name = "<tool_call><function=></function><parameter=note>unfinished";
    fixtures.push_back({"invalid-name-then-call", bad_name + ping,
                        {content(bad_name), call(0, "ping", "{}")}});
    const std::string open_quote = "<tool_call><function=complex><parameter=object>{\"note\":\"";
    fixtures.push_back({"open-json-quote-at-eof", open_quote + ping, {content(open_quote + ping)}});
    for (const std::string_view prior : {"1", "invalid"}) {
        const std::string open = "<tool_call><function=complex><parameter=integer>" +
            std::string(prior) + "</parameter><parameter=note>before";
        for (const std::string_view fence : {"", "</function>", "</tool_call>"}) {
            const std::string prefix = "parameter-boundary/" + std::string(prior) + "/" +
                std::string(fence);
            const std::string rest = "after" + std::string(fence);
            fixtures.push_back({prefix + "/open", open + ping + rest,
                                {content(open), call(0, "ping", "{}"), content(rest)}});
            const std::string literal_wrapper = open + ping + "after</parameter>" +
                std::string(fence);
            fixtures.push_back({prefix + "/closed", literal_wrapper, {content(literal_wrapper)}});
        }
        const std::string complete = open + ping + "after</parameter></function></tool_call>";
        Rows expected = {content(complete), call(0, "ping", "{}")};
        if (prior == "1") {
            expected = {call(0, "complex", R"({"integer":1,"note":"before)" + ping + "after\"}"),
                        call(1, "ping", "{}")};
        }
        fixtures.push_back({"complete-parameter/" + std::string(prior), complete + ping, expected});
    }
}

ChatOutputOptions options(const Case& test) {
    const auto tools = sglang::json::parse(R"([
        {"type":"function","function":{"name":"ping","parameters":{
            "type":"object","additionalProperties":false}}},
        {"type":"function","function":{"name":"loose"}},
        {"type":"function","function":{"name":"complex","parameters":{
            "type":"object","required":["integer"],"additionalProperties":false,"properties":{
                "note":{"type":"string"},"integer":{"type":"integer"},
                "amount":{"type":"number"},"ok":{"type":"boolean"},
                "object":{"type":"object"},"array":{"type":"array"},
                "union":{"anyOf":[{"type":"string"},{"type":"integer"}]},
                "maybe":{"type":["null","string"]}}}}}
    ])", sglang::json::Syntax::ordered);
    ChatOutputOptions output;
    output.separate_reasoning = true;
    output.prompt_has_open_think = test.prompt_has_open_think;
    output.stream_reasoning = test.streamed_reasoning;
    read_chat_tools(output, &tools);
    return output;
}

void append(Rows& rows, const ChatOutputEvent& event) {
    if (event.kind != Kind::tool_call && !rows.empty() &&
        std::get<0>(rows.back()) == event.kind) {
        std::get<1>(rows.back()) += event.text;
    } else {
        rows.emplace_back(event.kind, event.text, event.index, event.name);
    }
}

void verify(const Case& test, const ChatOutputOptions& config, const Chunks& chunks,
            std::string_view partition) {
    ChatOutputParser parser(config);
    sglang::Utf8Stream utf8;
    Rows rows;
    std::string reconstructed;
    const auto sink = [&](const ChatOutputEvent& event) {
        if (!sglang::valid_utf8(event.text)) {
            throw std::runtime_error(test.name + " emitted invalid UTF-8");
        }
        append(rows, event);
        return true;
    };
    for (const auto chunk : chunks) {
        reconstructed += chunk;
        if (!parser.push(utf8.push(chunk), sink)) {
            throw std::runtime_error(test.name + " stopped accepting text");
        }
    }
    if (reconstructed != test.input || !utf8.finish().empty()) {
        throw std::runtime_error(test.name + " invalid fixture chunks");
    }
    const auto count = std::count_if(test.expected.begin(), test.expected.end(),
        [](const Row& row) { return std::get<0>(row) == Kind::tool_call; });
    if (!parser.finish(test.ending, sink) || rows != test.expected ||
        parser.calls() != static_cast<std::size_t>(count)) {
        for (const auto& [kind, text, index, name] : rows) {
            std::cerr << static_cast<int>(kind) << ' ' << index << ' ' << name
                      << ' ' << text << '\n';
        }
        throw std::runtime_error(test.name + " mismatched events in " + std::string(partition));
    }
}

std::size_t verify_partitions(const Case& test, const sglang::Tokenizer& tokenizer) {
    const auto config = options(test);
    const std::string_view input(test.input);
    verify(test, config, {input}, "whole");
    Chunks single, tokens;
    for (std::size_t index = 0; index < input.size(); ++index) {
        single.push_back(input.substr(index, 1));
    }
    verify(test, config, single, "every byte");
    for (const auto token : tokenizer.encode(input)) {
        tokens.push_back(tokenizer.piece(token));
    }
    verify(test, config, tokens, "checkpoint token pieces");
    for (std::size_t index = 0; index <= input.size(); ++index) {
        verify(test, config, {input.substr(0, index), {}, input.substr(index)}, "two-way split");
    }
    std::mt19937 random(638763);
    for (int sample = 0; sample < 32; ++sample) {
        Chunks chunks;
        for (std::size_t position = 0; position < input.size();) {
            const auto piece = input.substr(position, 1 + random() % 19);
            chunks.push_back(piece);
            position += piece.size();
        }
        verify(test, config, chunks, "random widths");
    }
    return 36 + input.size();
}

void verify_cancelled_sink() {
    const auto config = options({"cancelled-sink", {}, {}});
    ChatOutputParser parser(config);
    int called = 0;
    const auto reject = [&](const ChatOutputEvent& event) {
        ++called;
        if (event.kind != Kind::tool_call || event.index != 0) {
            throw std::runtime_error("Missed the expected first call before cancellation");
        }
        return false;
    };
    if (parser.push(ping + ping + "suffix", reject) || !parser.finish(ChatEnd::cancel, reject) ||
        called != 1) {
        throw std::runtime_error("Parser continued after cancelled sink");
    }
}

}

int main(int argc, char** argv) {
    try {
        if (argc != 2) {
            throw std::runtime_error("Pass the Qwen checkpoint tokenizer.json path");
        }
        const sglang::Tokenizer tokenizer(argv[1]);
        auto fixtures = cases();
        add_parameter_boundary_cases(fixtures);
        std::size_t partitions = 0;
        int failures = 0;
        for (const auto& test : fixtures) {
            try {
                partitions += verify_partitions(test, tokenizer);
            } catch (const std::runtime_error& error) {
                std::cerr << "Chat output chunks: " << error.what() << '\n';
                ++failures;
            }
        }
        verify_cancelled_sink();
        if (failures != 0) {
            return 1;
        }
        std::cout << fixtures.size() << " chat output cases passed across " << partitions
                  << " partitions and the cancelled sink\n";
    } catch (const std::exception& error) {
        std::cerr << "Chat output chunks: " << error.what() << '\n';
        return 1;
    }
}
