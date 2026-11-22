"""Check the named Responses function contract and retain raw HTTP bodies."""

import argparse
import http.client
import json
from pathlib import Path

from responses_schema_http import FORMAT, REQUEST, answer, events


ROUTE = "/v1/responses"
TOOL = {
    "type": "function",
    "name": "get_weather",
    "description": "Get weather for a city.",
    "strict": True,
    "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
        "additionalProperties": False,
    },
}
NAMED = {
    **REQUEST,
    "input": "Call get_weather for Paris.",
    "max_output_tokens": 96,
    "tools": [TOOL],
    "tool_choice": {"type": "function", "name": "get_weather"},
}


def exchange(args, name, payload=None, *, method="POST", path=ROUTE, headers=None):
    connection = http.client.HTTPConnection("127.0.0.1", args.port, timeout=90)
    try:
        request_headers = {"Content-Type": "application/json"} | (headers or {})
        raw = json.dumps(payload) if payload is not None else None
        connection.request(method, path, raw, request_headers)
        response = connection.getresponse()
        body = response.read().decode("utf-8")
        row = dict(name=name, method=method, path=path, request=payload,
                   request_headers=request_headers, status=response.status,
                   headers=response.getheaders(), body=body)
        with args.receipt.open("a") as output:
            output.write(json.dumps(row) + "\n")
        print(name, response.status, flush=True)
        return response.status, response.getheader("Content-Type", ""), body
    finally:
        connection.close()


def packet(reply, status=200):
    assert reply[0] == status and reply[1].startswith("application/json"), reply
    return json.loads(reply[2])


def function(response, expected, name="get_weather"):
    assert response["status"] == "completed", response
    item, = response["output"]
    assert item["type"] == "function_call" and item["name"] == name, item
    assert item["status"] == "completed" and item["id"].startswith("fc_"), item
    assert item["call_id"].startswith("call_"), item
    arguments = json.loads(item["arguments"])
    assert arguments == expected, item
    assert all(type(arguments[key]) is type(value) for key, value in expected.items()), item
    usage = response["usage"]
    assert usage["input_tokens"] > 0 and usage["output_tokens"] > 0, usage
    assert usage["total_tokens"] == usage["input_tokens"] + usage["output_tokens"], usage
    return item


def stream_function(reply, expected):
    assert reply[0] == 200 and reply[1].startswith("text/event-stream"), reply
    history = list(events(reply[2]))
    assert [event["sequence_number"] for event in history] == list(range(len(history))), history
    assert [event["type"] for event in history] == [
        "response.created", "response.in_progress", "response.output_item.added",
        "response.function_call_arguments.delta", "response.function_call_arguments.done",
        "response.output_item.done", "response.completed",
    ], history
    terminal = history[-1]["response"]
    item = function(terminal, expected)
    added, delta, done, item_done = history[2:6]
    assert added["item"] == item | {"arguments": "", "status": "in_progress"}, added
    assert delta["item_id"] == done["item_id"] == item["id"], history
    assert delta["delta"] == done["arguments"] == item["arguments"], history
    assert done["name"] == item["name"] and item_done["item"] == item, history
    assert all(event["output_index"] == 0 for event in history[2:6]), history
    assert history[0]["response"]["id"] == history[1]["response"]["id"] == terminal["id"], history
    return terminal


def failures(args, parser_enabled):
    call = {"type": "function_call", "name": "get_weather", "call_id": "call_manual",
            "arguments": '{"city":"Paris"}'}
    for name, changes, message in (
        ("auto", {"tool_choice": "auto"}, "named function choice"),
        ("required", {"tool_choice": "required"}, "named function choice"),
        ("wrong_name", {"tool_choice": {"type": "function", "name": "other"}}, "declared function"),
        ("hosted", {"tools": [{"type": "web_search"}]}, "Only function tools"),
        ("multiple", {"tools": [TOOL, TOOL]}, "one function"),
        ("non_strict", {"tools": [TOOL | {"strict": False}]}, "strict object parameters"),
        ("no_declaration", {"tools": []}, "declared function"),
        ("unknown_call", {"input": [{"type": "function_call_output", "call_id": "call_unknown",
                                     "output": "21 C"}]}, "unanswered function call"),
        ("duplicate_call", {"input": [call, call]}, "Duplicate function call_id"),
        ("bad_arguments", {"input": [call | {"arguments": "[1]"}]}, "must be an object"),
    ):
        error = packet(exchange(args, name, NAMED | changes | {"stream": True}), 400)
        assert message in error["error"]["message"], error
    if parser_enabled:
        for name, changes, message in (
            ("schema_conflict", {"text": {"format": FORMAT}}, "cannot be combined"),
            ("logprobs_conflict", {"include": ["message.output_text.logprobs"]}, "unconstrained"),
            ("stop_conflict", {"stop": ["Paris"]}, "cannot be combined with text stops"),
        ):
            error = packet(exchange(args, name, NAMED | changes), 400)
            assert message in error["error"]["message"], error
    else:
        error = packet(exchange(args, "parser_disabled", NAMED | {"stream": True}), 400)
        assert "--tool-call-parser qwen3_coder" in error["error"]["message"], error


def continuations(args, stored, item):
    route = f"{ROUTE}/{stored['id']}"
    assert packet(exchange(args, "retrieved", method="GET", path=route)) == stored
    result = {"type": "function_call_output", "call_id": item["call_id"], "output": "21 C"}
    request = REQUEST | {"instructions": "Answer with just the number from the tool result."}
    followup = request | {"input": [result], "previous_response_id": stored["id"]}
    resumed = packet(exchange(args, "resumed", followup))
    manual = packet(exchange(args, "manual", request | {"input": [
        {"type": "message", "role": "user", "content": NAMED["input"]}, item, result,
    ]}))
    assert answer(resumed).strip() == answer(manual).strip() == "21", (resumed, manual)
    assert resumed["usage"] == manual["usage"], (resumed, manual)
    assert resumed["previous_response_id"] == stored["id"], resumed
    error = packet(exchange(args, "duplicate_output", followup | {"input": [result, result]}), 400)
    assert "unanswered function call" in error["error"]["message"], error


def run(args):
    failures(args, not args.parser_disabled)
    if args.parser_disabled:
        return
    expected = {"city": "Paris"}
    blocking = packet(exchange(args, "pinned_request", NAMED))
    first = function(blocking, expected)
    assert blocking["tools"] == [TOOL] and blocking["tool_choice"] == NAMED["tool_choice"], blocking
    reply = exchange(args, "stream", NAMED | {"stream": True, "store": True},
                     headers={"Authorization": "Bearer local-probe"})
    stored = stream_function(reply, expected)
    item = stored["output"][0]
    assert first["call_id"] != item["call_id"] and first["id"] != item["id"], (first, item)
    continuations(args, stored, item)
    typed_tool = TOOL | {"name": "set_alarm", "description": "Set an alarm.",
                         "parameters": {
                             "type": "object", "additionalProperties": False,
                             "required": ["hour", "repeat"],
                             "properties": {"hour": {"type": "integer"},
                                            "repeat": {"type": "boolean"}},
                         }}
    typed = NAMED | {"input": "Call set_alarm with hour 7 and repeat true.",
                     "tools": [typed_tool], "parallel_tool_calls": False,
                     "tool_choice": {"type": "function", "name": "set_alarm"}}
    function(packet(exchange(args, "typed", typed)), {"hour": 7, "repeat": True}, "set_alarm")
    schema = packet(exchange(args, "schema_regression", REQUEST | {"text": {"format": FORMAT}}))
    assert json.loads(answer(schema)) == {"answer": 42}, schema
    scored = packet(exchange(args, "logprobs_regression", REQUEST | {
        "include": ["message.output_text.logprobs"], "top_logprobs": 2,
    }))
    assert scored["output"][0]["content"][0]["logprobs"], scored
    status, _, _ = exchange(args, "wrong_method", method="GET")
    assert status == 405, status


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--parser-disabled", action="store_true")
    run(parser.parse_args())
