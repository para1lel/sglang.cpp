"""Check Responses schema output against the checkpoint and retain raw HTTP bodies."""

import argparse
import http.client
import json
from pathlib import Path


MODEL = "Qwen3.8-27B-FP8"
REQUEST = {
    "model": MODEL,
    "input": "What is 6 times 7? Answer only with the number.",
    "reasoning": {"effort": "none"},
    "temperature": 0,
    "max_output_tokens": 48,
    "store": False,
}
SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "integer"}},
    "required": ["answer"],
    "additionalProperties": False,
}
FORMAT = {"type": "json_schema", "name": "answer", "strict": True, "schema": SCHEMA}


def exchange(args, name, request, route="/v1/responses"):
    connection = http.client.HTTPConnection("127.0.0.1", args.port, timeout=75)
    try:
        connection.request("POST", route, json.dumps(request), {"Content-Type": "application/json"})
        response = connection.getresponse()
        body = response.read().decode("utf-8")
        with args.receipt.open("a") as output:
            row = dict(name=name, route=route, request=request, status=response.status,
                       headers=response.getheaders(), body=body)
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(name, response.status, flush=True)
        return response.status, response.getheader("Content-Type", ""), body
    finally:
        connection.close()


def responses(args, name, descriptor, **fields):
    request = {**REQUEST, "text": {"format": descriptor}, **fields}
    return exchange(args, name, request)


def rejected(result, statuses, parameter=None):
    status, content_type, body = result
    assert status in statuses and content_type.startswith("application/json"), result
    error = json.loads(body)
    assert "error" in error or "detail" in error, error
    if parameter is not None:
        assert error["error"]["param"] == parameter, error


def answer(response):
    assert response["status"] == "completed", response
    messages = [item for item in response["output"] if item["type"] == "message"]
    assert len(messages) == 1, response
    part, = messages[0]["content"]
    assert part["type"] == "output_text" and messages[0]["role"] == "assistant", response
    return part["text"]


def events(body):
    for frame in body.replace("\r\n", "\n").strip().split("\n\n"):
        lines = frame.splitlines()
        data = "\n".join(line.removeprefix("data:").lstrip()
                         for line in lines if line.startswith("data:"))
        if not data or data == "[DONE]":
            continue
        event = json.loads(data)
        for line in lines:
            if line.startswith("event:"):
                assert line.removeprefix("event:").strip() == event["type"], frame
        yield event


def check_schema(format_descriptor, raw, expected):
    from schema_oracle import validate

    value = validate(format_descriptor["schema"], raw)
    assert value == expected, value


def check_output(args, name, descriptor, expected):
    status, content_type, body = responses(args, name, descriptor)
    assert status == 200 and content_type.startswith("application/json"), body
    response = json.loads(body)
    blocking = answer(response)
    if args.mode == "native":
        check_schema(descriptor, blocking, expected)
        assert [item["type"] for item in response["output"]] == ["message"], response
        assert response["text"]["format"] == descriptor, response
    else:
        assert json.loads(blocking) == expected, blocking

    status, content_type, body = responses(args, name + "_stream", descriptor, stream=True)
    assert status == 200 and content_type.startswith("text/event-stream"), body
    stream = list(events(body))
    sequence = [event["sequence_number"] for event in stream]
    assert len(sequence) > 1 and sequence == sorted(set(sequence)), stream
    assert stream[0]["type"] == "response.created", stream
    assert stream[-1]["type"] == "response.completed", stream
    deltas = [event["delta"] for event in stream if event["type"] == "response.output_text.delta"]
    done = [event["text"] for event in stream if event["type"] == "response.output_text.done"]
    final = stream[-1]["response"]
    assert deltas and done == ["".join(deltas)] == [answer(final)], stream
    assert answer(final) == blocking, stream
    if args.mode == "native":
        assert [item["type"] for item in final["output"]] == ["message"], final
        assert final["text"]["format"] == descriptor, final
    return blocking


def run(args):
    if args.mode == "baseline":
        rejected(responses(args, "baseline_schema", FORMAT), {400}, "text")
        return

    raw = check_output(args, "schema", FORMAT, {"answer": 42})
    for key in ("name", "schema"):
        descriptor = {name: value for name, value in FORMAT.items() if name != key}
        rejected(responses(args, f"missing_{key}_stream", descriptor, stream=True), {400},
                 f"text.format.{key}" if args.mode == "native" else None)
    if args.mode == "pinned":
        return

    ordered = {
        "type": "object",
        "properties": {"z": {"enum": [1]}, "a": {"enum": [2]}},
        "required": ["z", "a"],
        "additionalProperties": False,
    }
    descriptor = {"type": "json_schema", "name": "ordered-answer",
                  "description": "A pair in declaration order.", "schema": ordered}
    check_output(args, "ordered_without_strict", descriptor, {"z": 1, "a": 2})
    for name, changes, parameter in [
        ("bad_name", {"name": "bad name"}, "name"),
        ("oversized_name", {"name": "a" * 65}, "name"),
        ("not_strict", {"strict": False}, "strict"),
        ("null_strict", {"strict": None}, "strict"),
        ("bad_description", {"description": 17}, "description"),
        ("unknown_key", {"json_schema": {}}, ""),
        ("non_object_schema", {"schema": []}, "schema"),
        ("unsupported_schema", {"schema": {"$ref": "#"}}, None),
        ("unsupported_format", {"type": "json_object"}, "type"),
    ]:
        if parameter is not None:
            parameter = f"text.format.{parameter}" if parameter else "text.format"
        rejected(responses(args, name, {**FORMAT, **changes}, stream=True), {400}, parameter)
    rejected(responses(args, "insufficient_budget", FORMAT, max_output_tokens=2, stream=True),
             {400})
    rejected(responses(args, "text_stop_conflict", FORMAT, stop=["42"], stream=True), {400})

    status, content_type, body = responses(args, "plain_text", {"type": "text"})
    assert status == 200 and content_type.startswith("application/json"), body
    assert answer(json.loads(body)).strip() == "42", body
    chat = {key: REQUEST[key] for key in ("model", "temperature")}
    chat.update(messages=[{"role": "user", "content": REQUEST["input"]}], max_tokens=46,
                chat_template_kwargs={"enable_thinking": False},
                response_format={"type": "json_schema", "json_schema": {
                    key: value for key, value in FORMAT.items() if key != "type"}})
    status, content_type, body = exchange(args, "chat_schema_control", chat, "/v1/chat/completions")
    assert status == 200 and content_type.startswith("application/json"), body
    assert json.loads(body)["choices"][0]["message"]["content"] == raw, body


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("port", type=int)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--mode", choices=["native", "pinned", "baseline"], default="native")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
