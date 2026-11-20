"""Exercise HTTP chat parsing with scripted checkpoint token IDs and real transport."""

import argparse
from contextlib import closing
import http.client
import json
import os
from pathlib import Path
import socket
import struct
import time

from run_chat_layer1 import save, server
from test_chat_output import BOTH, COMMON, REASONING, TOOLS, expect, native_body, pair
from test_native_chat import CHAT, CHECKPOINT, HEADERS, MODEL, RAW, event, json_response, one_choice
from test_native_chat import tokenizer_pieces

IM_END = 248046
WEATHER = {"type": "function", "function": {"name": "get_weather", "parameters": {
    "type": "object", "properties": {"city": {"type": "string"}, "day": {"type": "integer"},
                                     "amount": {"type": "number"}, "data": {"type": "object"}},
    "required": ["city"], "additionalProperties": False,
}}}
PING = {"type": "function", "function": {"name": "ping", "parameters": {
    "type": "object", "properties": {}, "additionalProperties": False,
}}}
RAW_CALL = ("<tool_call><function=get_weather><parameter=city>Paris</parameter>"
            "</function></tool_call>")
PING_CALL = "<tool_call><function=ping></function></tool_call>"
HALF_CALL = "<tool_call><function=ping>"
TRUNCATED = "<tool_call><function=get_weather><parameter=city>unfinished"


def request_body(**options):
    return native_body({"messages": [{"role": "user", "content": "Say hello."}],
                        "enable_thinking": False, **options})


def encode_inputs(args, inputs):
    pieces = tokenizer_pieces(args.checkpoint)
    singles = {piece[0]: token for token, piece in pieces.items() if len(piece) == 1}
    sequences = {}
    for name, source in inputs.items():
        text, terminal = source
        ids = [singles[byte] for byte in text.encode("utf-8")]
        if terminal:
            ids.append(IM_END)
        sequences[name] = (1001 + len(sequences), ids)
    scripts = args.evidence / "token-ids.txt"
    scripts.write_text("".join(f"{seed} {' '.join(map(str, ids))}\n"
                               for seed, ids in sequences.values()))
    os.environ["SGLANG_TEST_CHAT_OUTPUT_IDS"] = str(scripts.resolve())
    return sequences


def check_cancel(port, out, body, name, held_text):
    body = body | {"stream": True, "stream_options": {"include_usage": True}}
    cancels = (out / "server.log").read_text().count("cpu cancel slot=")
    with closing(http.client.HTTPConnection("127.0.0.1", port, timeout=10)) as connection:
        connection.request("POST", CHAT, json.dumps(body), HEADERS)
        with connection.sock.dup() as control, connection.getresponse() as response:
            assert response.status == 200
            frames = [event(response)]
            for _ in range(len(held_text.encode("utf-8"))):
                frames.append(event(response))
                choice = one_choice(frames[-1])
                assert len(choice["token_ids"]) == 1, choice
                delta = choice["delta"]
                assert not any(delta.get(field) for field in ("tool_calls", "reasoning_content",
                                                              "content")), frames[-1]
                assert choice["finish_reason"] is None
            control.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
    deadline = time.monotonic() + 6
    while (out / "server.log").read_text().count("cpu cancel slot=") == cancels:
        assert time.monotonic() < deadline
        time.sleep(0.01)
    save(out / f"{name}.json", frames)


def check_validation(port, out, body):
    strict = {"type": "function", "function": PING["function"] | {"strict": True}}
    ref = {"type": "function", "function": {"name": "ping", "parameters": {"$ref": "#"}}}
    invalid = {
        "strict": body | {"tools": [strict]},
        "ref": body | {"tools": [ref]},
        "duplicate-tool": body | {"tools": [PING, PING]},
        "required": body | {"tool_choice": "required"},
        "named": body | {"tool_choice": {"type": "function", "function": {"name": "ping"}}},
        "parallel-policy": body | {"parallel_tool_calls": False},
        "logprobs": body | {"logprobs": True},
        "top-logprobs": body | {"top_logprobs": 1},
        "stream-reasoning": body | {"stream_reasoning": "true"},
        "separate-reasoning": body | {"separate_reasoning": None},
    }
    errors = {}
    for name, request in invalid.items():
        for stream in (False, True):
            returned = json_response(port, CHAT, request | {"stream": stream}, status=400)
            assert returned["error"]["message"], returned
            errors[f"{name}-{stream}"] = returned
    save(out / "validation.json", errors)


def recoveries():
    literal = f"before {PING_CALL} after"
    malformed = (HALF_CALL + '</function><function=ping><parameter=key>{"value":"</tool_call>'
                 + PING_CALL + '"}</parameter></function></tool_call>')
    double_sign = RAW_CALL.replace("</function>", "<parameter=amount>+-1</parameter></function>")
    nulls = RAW_CALL.replace(">Paris<", ">nUlL<").replace(
        "</function>", "<parameter=day>NULL</parameter><parameter=amount>NuLL</parameter>"
        "</function>",
    )
    quoted_object = {"nested": ['雪\\"</parameter></tool_call>' + PING_CALL]}
    data = json.dumps(quoted_object, ensure_ascii=False)
    object_wrapper = RAW_CALL.replace(
        "</function>", f"<parameter=data>{data}</parameter></function>",
    )
    ping = [("ping", {})]
    cases = {
        "unclosed-wrapper": (HALF_CALL + PING_CALL, HALF_CALL, ping),
        "truncated-parameter": (TRUNCATED + PING_CALL, TRUNCATED, ping),
        "malformed-quoted-marker": (malformed + PING_CALL, malformed, ping),
        "double-sign-number": (double_sign + PING_CALL, double_sign, ping),
        "plain-marker": (RAW_CALL.replace("Paris", literal), "", [
            ("get_weather", {"city": literal}),
        ]),
        "json-marker": (object_wrapper, "", [
            ("get_weather", {"city": "Paris", "data": quoted_object}),
        ]),
        "typed-nulls": (nulls, "", [("get_weather", {"city": None, "day": None, "amount": None})]),
    }
    for prior in ("1", "invalid"):
        opened = (f"<tool_call><function=get_weather><parameter=day>{prior}</parameter>"
                  "<parameter=city>before")
        for label, fence in (("eof", ""), ("function", "</function>"), ("tool", "</tool_call>")):
            prefix = f"parameter-boundary-{prior}-{label}"
            cases[f"{prefix}-open"] = (opened + PING_CALL + "after" + fence,
                                      opened + "after" + fence, ping)
            closed = opened + PING_CALL + "after</parameter>" + fence
            cases[f"{prefix}-closed"] = (closed, closed, [])
        closed = opened + PING_CALL + "after</parameter></function></tool_call>"
        weather = ("get_weather", {"day": 1, "city": f"before{PING_CALL}after"})
        expected = ([weather, *ping] if prior == "1" else ping)
        cases[f"complete-parameter-{prior}"] = (closed + PING_CALL,
                                               "" if prior == "1" else closed, expected)
    open_quote = '<tool_call><function=get_weather><parameter=data>{"note":"' + PING_CALL
    cases["open-json-quote-at-eof"] = (open_quote, open_quote, [])
    bad_name = "<tool_call><function=></function><parameter=city>unfinished"
    cases["invalid-name-then-call"] = (bad_name + PING_CALL, bad_name, ping)
    return cases


def check_both(port, out, sequences):
    tools = [WEATHER, PING]

    def run(name, expected_tokens=None, **options):
        seed, ids = sequences[name]
        body = request_body(seed=seed, max_tokens=len(ids), tools=tools) | options
        result = pair(port, out, name, body)
        limit = body["max_tokens"] if expected_tokens is None else expected_tokens
        assert result["ids"] == ids[:limit], result
        return result

    prefix = "before café 😊"
    suffix = " between 雪 and after"
    expected = [("get_weather", {"city": "Paris"}), ("ping", {})]
    expect(run("parallel"), prefix + suffix, calls=expected, finish="tool_calls")
    expect(run("initial-thinking", enable_thinking=True), "visible", reasoning="hidden 雪")
    expect(run("empty-generation-suffix", enable_thinking=True, add_generation_prompt=False),
           "visible")
    expect(run("self-open"), "visible", reasoning="hidden 雪")
    expect(run("late-open"), "prefix<think>hidden</think>visible")
    parsed = run("implicit-tool", enable_thinking=True)
    expect(parsed, "", reasoning="checking", calls=[("ping", {})], finish="tool_calls")
    seed, ids = sequences["implicit-tool"]
    body = request_body(seed=seed, max_tokens=len(ids), tools=tools, enable_thinking=True)
    for label, choice in (("auto", "auto"), ("null", None)):
        assert pair(port, out, f"implicit-{label}", body | {"tool_choice": choice}) == parsed
    disabled = pair(port, out, "implicit-none", body | {"tool_choice": "none"})
    no_tools = pair(port, out, "implicit-no-tools", body | {"tools": []})
    assert disabled == no_tools and no_tools["prompt"] != parsed["prompt"], disabled
    expect(disabled, "", reasoning="checking" + PING_CALL)
    expect(run("trailing-held"), "plain <tool_ca", finish="length")
    expect(run("tool-length"), "", calls=[("ping", {})], finish="length")
    expect(run("complete-then-cut"), HALF_CALL, calls=[("ping", {})], finish="length")
    expect(run("raw-stop", len(HALF_CALL + "MATCH"), stop=["MATCH"]), HALF_CALL)
    recovery = [("ping", {})]
    expect(run("recovery-stop", len(TRUNCATED + PING_CALL + "MATCH"), stop=["MATCH"]),
           TRUNCATED, calls=recovery, finish="tool_calls")
    expect(run("recovery-length"), TRUNCATED, calls=recovery, finish="length")
    expect(run("unknown"), PING_CALL.replace("ping", "ghost"))
    multi = HALF_CALL + "</function><function=ping></function></tool_call>"
    expect(run("multiple-functions"), multi, calls=[("ping", {})], finish="tool_calls")
    unknown_param = RAW_CALL.replace("=city", "=country")
    expect(run("unknown-param"), unknown_param)
    invalid_value = RAW_CALL.replace("</function>", "<parameter=day>NaN</parameter></function>")
    expect(run("invalid-value"), invalid_value)
    for name, (_, visible, calls) in recoveries().items():
        expect(run(name), visible, calls=calls, finish="tool_calls" if calls else "stop")
    expect(run("no-tools", tools=[]), RAW_CALL)
    expect(run("zero", max_tokens=0), "", finish="length")
    check_validation(port, out, request_body(tools=tools, max_tokens=32))
    for name, prefix in (("cancel", HALF_CALL), ("cancel-ambiguous", TRUNCATED + PING_CALL)):
        cancel_seed, _ = sequences[name]
        body = request_body(seed=cancel_seed, tools=tools, max_tokens=512)
        check_cancel(port, out, body, name, prefix)
        expect(run("tool-length"), "", calls=[("ping", {})], finish="length")
    raw_seed, raw_ids = sequences["no-tools"]
    response = json_response(port, RAW, {"model": MODEL, "prompt": "Say hello.",
                                         "seed": raw_seed, "max_tokens": len(raw_ids)})
    assert one_choice(response)["text"] == RAW_CALL, response


def check_independent(port, out, flags, sequences):
    seed, ids = sequences["implicit-tool"]
    body = request_body(seed=seed, max_tokens=len(ids), tools=[PING], enable_thinking=True)
    actual = pair(port, out, "independent", body)
    if flags == REASONING:
        expect(actual, "", reasoning="checking" + PING_CALL)
    elif flags == TOOLS:
        expect(actual, "checking", calls=[("ping", {})], finish="tool_calls")
    else:
        expect(actual, "checking" + PING_CALL)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    args.evidence.mkdir()
    cases = {
        "parallel": ("before café 😊" + RAW_CALL + " between 雪" + PING_CALL + " and after", True),
        "initial-thinking": ("hidden 雪</think>visible", True),
        "empty-generation-suffix": ("visible", True),
        "self-open": ("<think>hidden 雪</think>visible", True),
        "late-open": ("prefix<think>hidden</think>visible", True),
        "implicit-tool": ("checking" + PING_CALL, True),
        "trailing-held": ("plain <tool_ca", False),
        "tool-length": (PING_CALL, False),
        "complete-then-cut": (PING_CALL + HALF_CALL, False),
        "raw-stop": (HALF_CALL + "MATCH" + PING_CALL, True),
        "recovery-stop": (TRUNCATED + PING_CALL + "MATCH" + RAW_CALL, True),
        "recovery-length": (TRUNCATED + PING_CALL, False),
        "unknown": (PING_CALL.replace("ping", "ghost"), True),
        "multiple-functions": (HALF_CALL + "</function><function=ping></function></tool_call>"
                               + PING_CALL, True),
        "unknown-param": (RAW_CALL.replace("=city", "=country"), True),
        "invalid-value": (RAW_CALL.replace("</function>", "<parameter=day>NaN</parameter>"
                                           "</function>"), True),
        "no-tools": (RAW_CALL, True),
        "zero": ("unused", True),
        "cancel": (HALF_CALL + "x" * 512, False),
        "cancel-ambiguous": (TRUNCATED + PING_CALL + "x" * 512, False),
    }
    cases.update({name: (text, True) for name, (text, _, _) in recoveries().items()})
    sequences = encode_inputs(args, cases)
    with server(args.server, args.checkpoint, args.evidence / "both", 1,
                flags=COMMON + BOTH) as port:
        check_both(port, args.evidence / "both", sequences)
    for name, flags in (("reasoning", REASONING), ("tools", TOOLS), ("off", ())):
        out = args.evidence / name
        with server(args.server, args.checkpoint, out, 1, flags=COMMON + flags) as port:
            check_independent(port, out, flags, sequences)
    print("Scripted chat HTTP output cases passed")


if __name__ == "__main__":
    main()
