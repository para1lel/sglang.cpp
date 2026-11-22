"""Exercise the Messages endpoint and compare generation with native chat."""

import argparse
import base64
from contextlib import closing
import http.client
import json
import os
from pathlib import Path
import re
import socket
import struct
import sys
from unittest.mock import patch
import zlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "text"))

from run_chat_layer1 import server
from test_native_chat import CHAT, CHECKPOINT, MODEL, json_response, request


ROUTE = "/v1/messages"
COUNT = ROUTE + "/count_tokens"
PARSERS = ("--reasoning-parser", "qwen3", "--tool-call-parser", "qwen3_coder")
SYSTEM = "Follow the instructions. Answer briefly."
USER = {"role": "user", "content": "What is the capital of France? Reply with the city only."}
TOOLS = [{"name": "lookup", "description": "Look up a city", "input_schema": {
    "type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"],
}}]


def body(**fields):
    return {
        "model": MODEL, "max_tokens": 24, "temperature": 0, "messages": [USER],
        "system": SYSTEM, "thinking": {"type": "disabled"},
    } | fields


def openai(messages):
    payload = {key: messages[key] for key in
               ("model", "max_tokens", "temperature", "top_p", "top_k", "cache_salt")
               if key in messages}
    return payload | {
        "messages": [{"role": "system", "content": messages["system"]}, *messages["messages"]],
        "enable_thinking": messages.get("thinking", {}).get("type") != "disabled",
        "stop": messages.get("stop_sequences", []),
    }


def frames(response):
    while line := response.readline():
        if line.startswith(b":"):
            assert response.readline() == b"\n"
            continue
        assert line.startswith(b"event: "), line
        name = line.removeprefix(b"event: ").strip().decode()
        data = response.readline()
        assert data.startswith(b"data: "), data
        value = json.loads(data.removeprefix(b"data: "))
        assert value["type"] == name
        assert response.readline() == b"\n"
        yield value


def stream(port, payload):
    connection, response = request(port, ROUTE, payload | {"stream": True})
    with closing(connection), response:
        assert response.status == 200, (response.status, response.read())
        assert response.headers.get_content_type() == "text/event-stream"
        return list(frames(response))


def replay(events):
    assert events[0]["type"] == "message_start" and events[-1] == {"type": "message_stop"}
    result = events[0]["message"]
    assert result["content"] == [] and result["usage"]["output_tokens"] == 0
    assert "stop_reason" not in result and "stop_sequence" not in result
    result = result | {"content": [], "usage": dict(result["usage"])}
    block = None
    for packet in events[1:-2]:
        kind = packet["type"]
        index = len(result["content"])
        assert packet["index"] == index
        if kind == "content_block_start":
            assert block is None
            block = dict(packet["content_block"])
            arguments = ""
        elif kind == "content_block_delta":
            delta = packet["delta"]
            assert block is not None
            if block["type"] == "tool_use":
                assert delta["type"] == "input_json_delta"
                arguments += delta["partial_json"]
            else:
                field = block["type"]
                assert delta["type"] == field + "_delta"
                block[field] += delta[field]
        else:
            assert kind == "content_block_stop" and block is not None
            if block["type"] == "tool_use":
                assert block["input"] == {}
                if arguments:
                    block["input"] = json.loads(arguments)
            result["content"].append(block)
            block = None
    assert block is None
    terminal = events[-2]
    assert terminal["type"] == "message_delta"
    assert set(terminal["usage"]) == {"output_tokens"}
    assert set(terminal["delta"]) == {"stop_reason"}
    result.update(terminal["delta"])
    result["usage"].update(terminal["usage"])
    return result


def normalize(payload):
    assert re.fullmatch(r"msg_[a-f0-9]{32}", payload["id"])
    result = {key: value for key, value in payload.items() if key != "id"}
    result["content"] = [
        {key: value for key, value in block.items() if key != "id"}
        if block["type"] == "tool_use" else block for block in result["content"]
    ]
    return result


def paired(port, messages, chat):
    reference = json_response(port, CHAT, chat)
    connection, reply = request(port, CHAT, chat | {
        "stream": True, "stream_options": {"include_usage": True},
    })
    with closing(connection), reply:
        assert reply.status == 200, (reply.status, reply.read())
        chat_stream = reply.read().decode()
    message = json_response(port, ROUTE, messages)
    events = stream(port, messages)
    folded = replay(events)
    expected = normalize(message)
    if not expected["content"][0].get("text", True) and not folded["content"]:
        folded["content"] = [{"type": "text", "text": ""}]
    assert normalize(folded) == expected, (folded, message)
    choice, = reference["choices"]
    first = choice["message"]
    assert message["usage"]["input_tokens"] == reference["usage"]["prompt_tokens"]
    assert message["usage"]["output_tokens"] == reference["usage"]["completion_tokens"]
    assert message["stop_reason"] == {
        "stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use",
    }[choice["finish_reason"]]
    for kind, field in (("text", "content"), ("thinking", "reasoning_content")):
        actual = "".join(block[kind] for block in message["content"] if block["type"] == kind)
        assert actual == (first.get(field) or ""), (kind, actual, first)
    calls = [block for block in expected["content"] if block["type"] == "tool_use"]
    assert calls == [
        {"type": "tool_use", "name": call["function"]["name"],
         "input": json.loads(call["function"]["arguments"])}
        for call in first.get("tool_calls", [])
    ]
    return {"response": message, "events": events, "chat": reference, "chat_stream": chat_stream}


def invalid_cases():
    yield "model-missing", {key: val for key, val in body().items() if key != "model"}
    yield "budget-missing", {key: val for key, val in body().items() if key != "max_tokens"}
    yield "empty-history", body(messages=[], system="")
    yield "negative-budget", body(max_tokens=-1)
    yield "bad-role", body(messages=[{"role": "tool", "content": "x"}])
    yield "bad-thinking-budget", body(thinking={"type": "enabled", "budget_tokens": 12})
    yield "unknown-tool", body(tools=TOOLS, tool_choice={"type": "tool", "name": "missing"})
    yield "server-tool-required", body(
        tools=[{"type": "web_search_20250305", "name": "web_search"}], tool_choice={"type": "any"},
    )


def invalid(port):
    receipts = {}
    for name, payload in invalid_cases():
        value = json_response(port, ROUTE, payload | {"stream": True}, 400)
        assert value["type"] == "error" and value["error"]["type"] == "invalid_request_error"
        assert value["error"]["message"]
        receipts[name] = value
    receipts["malformed"] = json_response(port, ROUTE, b"{", 400)
    with closing(http.client.HTTPConnection("127.0.0.1", port, timeout=30)) as connection:
        connection.request("POST", ROUTE, "{}", {"Content-Type": "text/plain"})
        with connection.getresponse() as response:
            assert response.status == 400
            receipts["media-type"] = json.load(response)
        connection.request("GET", ROUTE)
        with connection.getresponse() as response:
            assert response.status == 405 and response.headers["Allow"] == "POST"
            response.read()
    for value in receipts.values():
        assert value["error"]["type"] == "invalid_request_error"
    return receipts


def cancel(port):
    connection, response = request(port, ROUTE, body(max_tokens=2048, stream=True))
    assert response.status == 200
    packets = frames(response)
    assert next(packets)["type"] == "message_start"
    assert next(packets)["type"] == "content_block_start"
    response.fp.raw._sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
    connection.close()
    response.close()
    assert json_response(port, ROUTE, body(max_tokens=1))["usage"]["output_tokens"] == 1
    return {"reset_after": "content_block_start", "next_request_output_tokens": 1}


def visual_case():
    def chunk(kind, data):
        crc = struct.pack(">I", zlib.crc32(kind + data))
        return struct.pack(">I", len(data)) + kind + data + crc

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", 32, 32, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress((b"\0" + b"\xff\0\0" * 32) * 32))
    png += chunk(b"IEND", b"")
    data = base64.b64encode(png).decode()
    text = {"type": "text", "text": "What color is this image? Answer with one word."}
    image = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}}
    payload = body(max_tokens=4, messages=[{"role": "user", "content": [image, text]}])
    chat = openai(payload)
    chat["messages"][-1] = {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + data}}, text,
    ]}
    return payload, chat


def tool_history_case():
    call = {"type": "tool_use", "id": "call_1", "name": "lookup", "input": {"city": "Paris"}}
    result = {"type": "tool_result", "tool_use_id": "call_1", "content": "France"}
    follow = {"type": "text", "text": "Give the result only."}
    payload = body(messages=[USER, {"role": "assistant", "content": [call]},
                             {"role": "user", "content": [result, follow]}])
    chat = openai(payload)
    chat["messages"] = [chat["messages"][0], USER, {"role": "assistant", "tool_calls": [{
        "type": "function", "id": "call_1", "function": {
            "name": "lookup", "arguments": '{"city":"Paris"}',
        },
    }]}, {"role": "tool", "tool_call_id": "call_1", "content": "France"},
        {"role": "user", "content": follow["text"]}]
    return payload, chat


def schema_case():
    schema = {"type": "object", "properties": {"city": {"type": "string"}},
              "required": ["city"], "additionalProperties": False}
    format_config = {"type": "json_schema", "schema": schema}
    payload = body(max_tokens=32, output_config={"format": format_config})
    chat = openai(payload) | {"response_format": {"type": "json_schema", "json_schema": {
        "name": "response", "strict": True, "schema": schema,
    }}}
    return payload, chat


def exercise(port, mode):
    plain = body()
    short = body(max_tokens=1)
    multi = body(messages=[{"role": "user", "content": "Say Paris."},
                           {"role": "assistant", "content": "Paris"},
                           {"role": "user", "content": "Repeat the city."}],
                 top_k=1, top_p=1, stop_sequences=["\n"])
    reasoning = body(thinking={"type": "enabled", "budget_tokens": 1024})
    default = body()
    del default["thinking"]
    checks = {"plain": plain, "short": short, "history-stop": multi, "thinking": reasoning}
    checks["thinking-default"] = default
    result = {name: paired(port, payload, openai(payload)) for name, payload in checks.items()}
    result["tool-history"] = paired(port, *tool_history_case())
    count = {key: val for key, val in plain.items() if key not in {"max_tokens", "temperature"}}
    assert json_response(port, COUNT, count)["input_tokens"] == (
        result["plain"]["response"]["usage"]["input_tokens"]
    )
    result["errors"] = invalid(port)
    if mode == "live":
        forced = body(max_tokens=96, tools=TOOLS, tool_choice={"type": "tool", "name": "lookup"})
        chat = openai(forced) | {"tools": [{"type": "function", "function": {
            "name": "lookup", "description": "Look up a city",
            "parameters": TOOLS[0]["input_schema"], "strict": False,
        }}], "tool_choice": {"type": "function", "function": {"name": "lookup"}}}
        result["tool"] = paired(port, forced, chat)
        assert result["tool"]["response"]["stop_reason"] == "tool_use"
        result["image"] = paired(port, *visual_case())
        result["schema"] = paired(port, *schema_case())
        text = result["schema"]["response"]["content"][0]["text"]
        assert isinstance(json.loads(text)["city"], str)
        result["cancel"] = cancel(port)
    return result


def cache(port):
    payload = body(max_tokens=2, cache_salt="messages-1", messages=[{
        "role": "user", "content": "blue green orange pink " * 45 + "Say one color.",
    }])
    cold = json_response(port, ROUTE, payload)["usage"]
    hot = stream(port, payload)[0]["message"]["usage"]
    different = json_response(port, ROUTE, payload | {"cache_salt": "messages-2"})["usage"]
    assert hot["cache_read_input_tokens"] > 0 and hot["output_tokens"] == 0
    assert cold == different
    assert hot["input_tokens"] + hot["cache_read_input_tokens"] == cold["input_tokens"]
    return {"cold": cold, "hot_stream_start": hot, "different_salt": different}


def runtime_errors(binary, out):
    receipts = {}
    for accepted in (0, 2):
        label = f"failure-after-{accepted}"
        os.environ["SGLANG_TEST_MESSAGES_FAILURE_AFTER"] = str(accepted)
        try:
            with server(binary, CHECKPOINT, out / label, 1, flags=PARSERS) as port:
                events = stream(port, body())
        finally:
            del os.environ["SGLANG_TEST_MESSAGES_FAILURE_AFTER"]
        names = [event["type"] for event in events]
        assert names[0] == "message_start" and names[-2:] == ["error", "message_stop"]
        if accepted:
            assert names[1] == "content_block_start" and names[-3] == "content_block_stop"
        else:
            assert len(names) == 3 and events[0]["message"]["usage"]["input_tokens"] == 0
        assert events[-2]["error"] == {"type": "api_error", "message": "Internal server error"}
        receipts[label] = events
    return receipts


def ollama_collision(binary, out, *, graph=False, mtp=False):
    payload = {"model": MODEL, "messages": [{"role": "user", "content": "Say Paris."}],
               "stream": False, "options": {"temperature": 0, "num_predict": 3}}
    with patch.dict(os.environ, {"SGLANG_OLLAMA_CHAT_ROUTE": ROUTE}):
        with server(binary, CHECKPOINT, out, 1, graph=graph, mtp=mtp) as port:
            reply = json_response(port, ROUTE, payload)
            counted = json_response(port, COUNT, body())
    assert reply["model"] == MODEL and reply["done"] is True
    assert reply["message"]["role"] == "assistant"
    assert counted["input_tokens"] > 0
    return {"chat": reply, "count_tokens": counted}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("binary", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--mtp", action="store_true")
    parser.add_argument("--prefix", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    flags = PARSERS + (("--prefix-cache-entries", "2", "--fp8-prefill", "exact")
                       if args.prefix else ())
    with server(args.binary, CHECKPOINT, args.out / "server", 1,
                flags=flags, mtp=args.mtp, graph=args.live) as port:
        result = {"cache": cache(port)} if args.prefix else exercise(
            port, "live" if args.live else "cpu",
        )
    if not args.live:
        result["runtime-errors"] = runtime_errors(args.binary, args.out)
    if not args.prefix:
        result["ollama-collision"] = ollama_collision(
            args.binary, args.out / "ollama-collision", graph=args.live, mtp=args.mtp)
    (args.out / "wire.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print("Verified " + ", ".join(result))


if __name__ == "__main__":
    main()
