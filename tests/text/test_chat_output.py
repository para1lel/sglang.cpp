"""Check opted-in chat output on the native HTTP endpoint."""

import argparse
import http.client
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import json
from pathlib import Path
import re
import socket
import struct
from threading import Barrier

from run_chat_layer1 import save, server
from test_native_chat import (
    CHAT, CHECKPOINT, GREEDY, HEADERS, MODEL, event, json_response, metadata, one_choice, request,
    usage,
)

REASONING = ("--reasoning-parser", "qwen3")
TOOLS = ("--tool-call-parser", "qwen3_coder")
BOTH = REASONING + TOOLS
COMMON = ("--mtp-steps", "0")


def fold_calls(calls, deltas):
    for delta in deltas:
        index = delta["index"]
        function = delta["function"]
        fragment = function.get("arguments", "")
        assert type(index) is int and type(fragment) is str, delta
        if index == len(calls):
            assert delta["type"] == "function" and function["name"], delta
            assert function["arguments"] == "", delta
            assert type(delta["id"]) is str and delta["id"], delta
            assert all(call["id"] != delta["id"] for call in calls), delta
            calls.append({"id": delta["id"], "name": function["name"], "arguments": fragment})
        else:
            assert index == len(calls) - 1 and calls, delta
            assert delta.get("id") is None and delta.get("type") is None, delta
            assert function.get("name") is None and fragment, delta
            calls[index]["arguments"] += fragment


def normalize_calls(calls):
    return [(call["name"], call["arguments"]) for call in calls]


def fold_json(packet, diagnostics):
    metadata(packet, "chat.completion")
    choice = one_choice(packet)
    message = choice["message"]
    assert message["role"] == "assistant", message
    assert set(message) <= {"role", "content", "reasoning_content", "tool_calls"}, message
    calls = message.get("tool_calls") or []
    assert all(call["type"] == "function" and call["id"] for call in calls), calls
    assert len({call["id"] for call in calls}) == len(calls), calls
    if calls and not message["content"]:
        assert message["content"] is None, message
    if diagnostics:
        usage(packet, len(choice["prompt_token_ids"]), len(choice["token_ids"]))
    else:
        assert "prompt_token_ids" not in choice and "token_ids" not in choice, choice
    return {
        "content": message.get("content") or "",
        "reasoning": message.get("reasoning_content") or "",
        "calls": normalize_calls([call["function"] for call in calls]),
        "finish": choice["finish_reason"],
        "ids": choice.get("token_ids", []),
        "prompt": choice.get("prompt_token_ids"),
    }


def stream_data(response, identity, diagnostics):
    texts = defaultdict(list)
    calls, frames, ids, prompts = [], [], [], []
    while True:
        packet = event(response)
        frames.append(packet)
        assert metadata(packet, "chat.completion.chunk") == identity and "usage" not in packet
        choice = one_choice(packet)
        delta = choice["delta"]
        if "prompt_token_ids" in choice:
            prompts.append(choice["prompt_token_ids"])
        if diagnostics:
            ids.extend(choice.get("token_ids", []))
        else:
            assert "prompt_token_ids" not in choice and "token_ids" not in choice, choice
        reason = choice["finish_reason"]
        if reason is not None:
            assert delta == {} and not choice.get("token_ids"), choice
            break
        assert set(delta) <= {"content", "reasoning_content", "tool_calls"}, delta
        for channel in ("content", "reasoning_content"):
            if channel in delta and delta[channel] is not None:
                assert type(delta[channel]) is str, delta
                texts[channel].append(delta[channel])
        fold_calls(calls, delta.get("tool_calls") or [])
    assert len(prompts) == int(diagnostics), (prompts, diagnostics)
    return frames, {
        "content": "".join(texts["content"]),
        "reasoning": "".join(texts["reasoning_content"]),
        "calls": normalize_calls(calls),
        "finish": reason,
        "ids": ids,
        "prompt": prompts[0] if prompts else None,
    }


def fold_sse(port, body, include_usage):
    diagnostics = body.get("return_token_ids", False)
    stream = body | {"stream": True}
    if include_usage:
        stream["stream_options"] = {"include_usage": True}
    connection, response = request(port, CHAT, stream)
    with closing(connection), response:
        assert response.status == 200, (response.status, response.read())
        assert response.headers.get_content_type() == "text/event-stream"
        role = event(response)
        identity = metadata(role, "chat.completion.chunk")
        assert one_choice(role) == {
            "index": 0, "delta": {"role": "assistant"}, "finish_reason": None,
        }, role
        frames, folded = stream_data(response, identity, diagnostics)
        frames.insert(0, role)
        if include_usage:
            packet = event(response)
            frames.append(packet)
            assert metadata(packet, "chat.completion.chunk") == identity
            assert packet["choices"] == []
            if diagnostics:
                usage(packet, len(folded["prompt"]), len(folded["ids"]))
        assert event(response) == "[DONE]" and response.read(1) == b""
        frames.append("[DONE]")
    return frames, folded


def pair(port, out, name, body, *, include_usage=True):
    response = json_response(port, CHAT, body)
    frames, folded = fold_sse(port, body, include_usage)
    save(out / f"{name}.json", {"request": body, "response": response, "sse": frames})
    assert fold_json(response, body.get("return_token_ids", False)) == folded, (name, folded)
    assert response["id"] != frames[0]["id"], name
    return folded


def expect(folded, content, *, reasoning="", calls=(), finish="stop"):
    assert folded["content"] == content, folded
    assert folded["reasoning"] == reasoning, folded
    actual = [(name, json.loads(arguments)) for name, arguments in folded["calls"]]
    assert actual == list(calls), folded
    assert folded["finish"] == finish, folded


def native_cases():
    reasoning = {"messages": [{"role": "user", "content":
                              "Compute 17 × 23. Give just the final number."}], "max_tokens": 128}
    plain = {"messages": [{"role": "user", "content":
                          "Give the word café and a smiling emoji, with no explanation."}],
             "enable_thinking": False, "max_tokens": 24}
    tool = {"messages": [{"role": "user", "content":
                          "Use the get_weather tool to look up the weather in Paris. "
                          "Call the tool now."}],
            "tools": [{"type": "function", "function": {
                "name": "get_weather", "description": "Get the weather for a city.",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                               "required": ["city"]},
            }}], "tool_choice": "auto", "enable_thinking": False, "max_tokens": 64}
    literal = "<think>café</think><tool_call>literal"
    constraint = {"messages": [{"role": "user", "content": "Emit the allowed literal."}],
                  "max_tokens": 40, "ebnf": "root ::= " + json.dumps(literal, ensure_ascii=False)}
    return {"reasoning": reasoning, "plain": plain, "tool": tool, "raw_constraint": constraint}


def check_tool_choice(port, out, cases, isolated):
    tool = native_body(cases["tool"])
    omitted = {key: value for key, value in tool.items() if key != "tool_choice"}
    for label, body in (("omitted", omitted), ("null", tool | {"tool_choice": None})):
        assert pair(port, out, f"tools-{label}", body) == isolated["tool"], label

    plain = native_body(cases["plain"])
    for label, value in (("auto", "auto"), ("null", None), ("none", "none")):
        body = plain | {"tool_choice": value}
        assert pair(port, out, f"no-tools-{label}", body) == isolated["plain"], label

    messages = cases["tool"]["messages"] + [
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": "prior_call", "type": "function", "function": {
                "name": "get_weather", "arguments": '{"city":"Paris"}',
            },
        }]},
        {"role": "tool", "tool_call_id": "prior_call", "content": "Paris is sunny."},
        {"role": "user", "content": "Use the weather result to answer in one short sentence."},
    ]
    history = native_body({"messages": messages, "max_tokens": 128})
    expected = pair(port, out, "history-no-tools", history)
    disabled = history | {"tools": tool["tools"], "tool_choice": "none"}
    actual = pair(port, out, "history-tools-none", disabled)
    assert actual == expected and actual["reasoning"] and not actual["calls"], actual
    compare = pair(port, out, "history-tools-auto", history | {"tools": tool["tools"]})
    assert compare["prompt"] != actual["prompt"], compare
    plain_history = history | {"messages": [messages[0], messages[-1]]}
    without_history = pair(port, out, "history-removed", plain_history)
    assert without_history["prompt"] != actual["prompt"], without_history


def native_body(body):
    return GREEDY | body | {"model": MODEL, "return_token_ids": True}


def keep_ids(actual, default):
    assert actual["ids"] == default["ids"] and actual["prompt"] == default["prompt"], (
        actual, default,
    )


def check_default(port, out, cases):
    outputs = {name: pair(port, out, name, native_body(body)) for name, body in cases.items()}
    for name, folded in outputs.items():
        assert len(folded["ids"]) > 1, (name, folded)
        expect(folded, folded["content"])
        receipt = json.loads((out / f"{name}.json").read_text())
        message = one_choice(receipt["response"])["message"]
        assert message == {"role": "assistant", "content": folded["content"]}, message
    assert "</think>" in outputs["reasoning"]["content"]
    assert "<tool_call>" in outputs["tool"]["content"]
    assert outputs["plain"]["content"] == "café 😊", outputs
    assert outputs["raw_constraint"]["content"] == "<think>café</think><tool_call>literal", outputs
    return outputs


def check_native(port, out, cases, default):
    outputs = {name: pair(port, out, name, native_body(body)) for name, body in cases.items()}
    for name, result in outputs.items():
        keep_ids(result, default[name])
    internal, answer = default["reasoning"]["content"].split("</think>", 1)
    expect(outputs["reasoning"], answer, reasoning=internal)
    expect(outputs["plain"], default["plain"]["content"])
    expect(outputs["tool"], "", calls=[("get_weather", {"city": "Paris"})], finish="tool_calls")
    expect(outputs["raw_constraint"], default["raw_constraint"]["content"])
    body = native_body(cases["reasoning"])
    disabled = pair(port, out, "reasoning-raw", body | {"separate_reasoning": False})
    assert disabled == default["reasoning"], disabled
    buffered = pair(port, out, "reasoning-buffered", body | {"stream_reasoning": False})
    assert buffered == outputs["reasoning"], buffered
    for name in ("reasoning", "tool"):
        body = cases[name] | {"model": MODEL, **GREEDY}
        public = pair(port, out, name + "-public", body, include_usage=False)
        assert public == outputs[name] | {"prompt": None, "ids": []}, public
    return outputs


def check_native_reset(port, out, body, expected):
    with closing(http.client.HTTPConnection("127.0.0.1", port, timeout=30)) as connection:
        connection.request("POST", CHAT, json.dumps(body | {"stream": True}), HEADERS)
        with connection.sock.dup() as control, connection.getresponse() as response:
            assert response.status == 200
            role, first = event(response), event(response)
            assert metadata(role, "chat.completion.chunk") == (
                metadata(first, "chat.completion.chunk")
            )
            choice = one_choice(first)
            assert choice["finish_reason"] is None and len(choice["token_ids"]) == 1, first
            assert choice["prompt_token_ids"] == expected["prompt"], first
            control.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
    save(out / "cancel.json", [role, first])
    assert pair(port, out, "after-cancel", body) == expected


def run_overlap(port, out, cases, isolated):
    names = ("reasoning", "tool", "plain", "raw_constraint")
    barrier = Barrier(len(names), timeout=15)

    def lane(name):
        barrier.wait()
        actual = pair(port, out, "overlap-" + name, native_body(cases[name]))
        assert actual == isolated[name], (name, actual, isolated[name])
        return actual

    with ThreadPoolExecutor(max_workers=len(names)) as pool:
        results = list(pool.map(lane, names))
    assert len(results) == len(names)
    trace = (out / "server.log").read_text()
    assert re.search(r"batch width=[2-4] decode=[2-4] request_bytes=", trace), trace[-2000:]


def verify_native(args):
    cases = native_cases()
    settings = (args.server, args.checkpoint)
    with server(*settings, args.evidence / "native-default", 1, flags=COMMON) as port:
        default = check_default(port, args.evidence / "native-default", cases)
    with server(*settings, args.evidence / "native-width-1", 1, flags=COMMON + BOTH) as port:
        width_one = args.evidence / "native-width-1"
        isolated = check_native(port, width_one, cases, default)
        check_tool_choice(port, width_one, cases, isolated)
        check_native_reset(port, width_one, native_body(cases["reasoning"]), isolated["reasoning"])
    with server(*settings, args.evidence / "native-width-4", 4, flags=COMMON + BOTH) as port:
        run_overlap(port, args.evidence / "native-width-4", cases, isolated)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    args.evidence.mkdir()
    verify_native(args)
    print("Native chat output width 1 and concurrent width 4 passed")


if __name__ == "__main__":
    main()
