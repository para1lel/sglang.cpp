#!/usr/bin/env python3
"""Exercise bounded EBNF through the native text and chat HTTP routes."""

import argparse
import codecs
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
from threading import Barrier

from run_chat_layer1 import save, server
from test_native_chat import (
    CHAT, CHECKPOINT, RAW, chat_body, chat_json, chat_sse, fixtures, json_response,
    one_choice, raw_body, tokenizer_pieces,
)
from test_native_text import EOS, events, expected_choices, legacy, stream_options, usage


def grammar(*choices):
    return "root ::= " + " | ".join(json.dumps(value, ensure_ascii=True) for value in choices)


def valid_cases():
    return (
        ("tight_prefix", ' \troot ::= "hello" \n| "hello!" ', ("hello",), 1, {}),
        ("prefix", grammar("hello", "hello!"), ("hello", "hello!"), 8, {}),
        ("split_scalar", grammar("\U0010fffd"), ("\U0010fffd",), 4, {}),
        ("escapes", r'root ::= "\"\\\n\u0000" | "\uD83D\uDE42"',
         ('"\\\n\0', "🙂"), 8, {}),
        ("literal_bytes", 'root ::= "e\u0301"', ("e\u0301",), 5, {}),
        ("sampled", grammar("42+17=59", "42+17=591", "42−17=25"),
         ("42+17=59", "42+17=591", "42−17=25"), 12,
         {"temperature": 1.3, "top_p": 0.9, "top_k": 4, "seed": -(2**63)}),
    )


def body_for(route, case, budget, **options):
    body = (chat_body if route == CHAT else raw_body)(case, budget)
    return body | {"ignore_eos": False} | options


def result(port, route, case, body):
    if route == CHAT:
        _, ids, text, finish = chat_json(port, body, case["ids"])
        return ids, text, finish
    packet = json_response(port, route, body)
    choice = one_choice(packet)
    ids = choice["token_ids"]
    assert choice["prompt_token_ids"] == case["ids"]
    assert packet["usage"] == usage(case["ids"], ids)
    return ids, choice["text"], choice["finish_reason"]


def verify_valid(port, route, case, pieces, spec):
    name, ebnf, choices, budget, sampling = spec
    body = body_for(route, case, budget, ebnf=ebnf, **sampling)
    ids, text, finish = result(port, route, case, body)
    assert 0 < len(ids) <= budget and finish == "stop", (name, ids, finish)
    assert text in choices, (name, text, choices)
    assert not any(token in EOS for token in ids[:-1]), (name, ids)
    decoded_pieces = [pieces[token] for token in ids]
    assert all(piece or token in EOS for token, piece in zip(ids, decoded_pieces, strict=True))
    assert b"".join(decoded_pieces) == text.encode(), (name, ids, text)
    if name == "split_scalar":
        assert len(ids) > 1 and all(decoded_pieces), ids
    if ids[-1] in EOS:
        assert any(other != text and other.startswith(text) for other in choices), (name, ids)
    elif len(ids) < budget:
        assert not any(other != text and other.startswith(text) for other in choices), (name, ids)
    if route == CHAT:
        frames = []
        _, stream_ids, stream_text, stream_finish = chat_sse(
            port, body | stream_options(), case["ids"], frames=frames
        )
        assert (stream_ids, stream_text, stream_finish) == (ids, text, finish), name
        decoder = codecs.getincrementaldecoder("utf-8")("strict")
        for frame, token, piece in zip(frames[1:-3], ids, decoded_pieces, strict=True):
            choice = one_choice(frame)
            assert choice["token_ids"] == [token]
            assert choice["delta"]["content"] == decoder.decode(piece), (name, choice)
        assert decoder.decode(b"", final=True) == ""
    else:
        frames = events(port, body | stream_options())
        assert frames == expected_choices(case["ids"], ids, decoded_pieces, finish) + [
            {"choices": [], "usage": usage(case["ids"], ids)}
        ], name
    return {"ids": ids, "text": text, "finish": finish}


def verify_invalid(port, route, case):
    invalid = (
        ("null", {"ebnf": None}), ("number", {"ebnf": 1}), ("object", {"ebnf": {}}),
        ("empty", {"ebnf": ""}), ("wrong_rule", {"ebnf": 'other ::= "x"'}),
        ("bare_word", {"ebnf": "root ::= x"}), ("empty_literal", {"ebnf": grammar("")}),
        ("missing_alt", {"ebnf": 'root ::= "x" |'}),
        ("concat", {"ebnf": 'root ::= "x" "y"'}),
        ("recursion", {"ebnf": 'root ::= "x" | root'}),
        ("repetition", {"ebnf": 'root ::= "x"+'}),
        ("regex", {"ebnf": "root ::= [a-z]"}),
        ("additional_rule", {"ebnf": 'root ::= "x"\nother ::= "y"'}),
        ("bad_escape", {"ebnf": r'root ::= "\q"'}),
        ("bad_surrogate", {"ebnf": r'root ::= "\uD800"'}),
        ("raw_control", {"ebnf": 'root ::= "x\0"'}),
        ("too_many", {"ebnf": grammar(*(str(index) for index in range(33)))}),
        ("too_long", {"ebnf": grammar("x" * 257)}),
        ("total_bytes", {"ebnf": grammar(*("x" * 256 for index in range(8)), "z")}),
        ("grammar_bytes", {"ebnf": " " * 16384 + grammar("a")}),
        ("budget", {"ebnf": grammar("\U0010fffd"), "max_tokens": 1}),
        ("ignore_eos", {"ebnf": grammar("yes"), "ignore_eos": True}),
        ("stop_override", {"ebnf": grammar("yes"), "stop_token_ids": [248044]}),
    )
    for name, options in invalid:
        packet = json_response(port, route, body_for(route, case, 12, **options), status=400)
        assert isinstance(packet.get("error"), dict), (name, packet)
    if route == RAW:
        packet = json_response(port, route, legacy(case["ids"], 4, ebnf=grammar("yes")), status=400)
        assert isinstance(packet.get("error"), dict), packet
    return [name for name, _ in invalid]


def verify_overlap(port, case, pieces):
    text = "ZX: 76543210987654321098765432109876543210"
    spec = ("overlap", grammar(text), (text,), 48, {"temperature": 1.2, "seed": 19})
    barrier = Barrier(3)

    def constrained(route):
        barrier.wait(timeout=10)
        return verify_valid(port, route, case, pieces, spec)

    def ordinary():
        barrier.wait(timeout=10)
        return result(port, RAW, case, raw_body(case, 32))

    with ThreadPoolExecutor(max_workers=3) as workers:
        raw = workers.submit(constrained, RAW)
        chat = workers.submit(constrained, CHAT)
        greedy = workers.submit(ordinary)
        left, right, normal = raw.result(), chat.result(), greedy.result()
    assert left == right
    return left, normal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    args.evidence.mkdir()
    pieces = tokenizer_pieces(args.checkpoint)
    assert "\U0010fffd".encode() not in pieces.values()
    case = fixtures()[0]
    sampled = {"temperature": 1.1, "seed": -19, "top_k": 8, "top_p": 1.0, "min_p": 0.0}
    normal = (("greedy", {}), ("sample", sampled))
    with server(args.control, args.checkpoint, args.evidence / "control", 1) as port:
        expected = {
            name: result(port, RAW, case, raw_body(case, 12) | options)
            for name, options in normal
        }
        expected_long = result(port, RAW, case, raw_body(case, 32))
    reference = None
    for name, width, graph, mtp in (
        ("serial", 1, False, False), ("continuous", 4, True, False),
        ("serial-mtp", 1, False, True), ("continuous-mtp", 4, True, True),
    ):
        output = args.evidence / name
        with server(args.server, args.checkpoint, output, width, graph=graph, mtp=mtp) as port:
            cases = [verify_valid(port, RAW, case, pieces, spec) for spec in valid_cases()]
            chat = [verify_valid(port, CHAT, case, pieces, spec) for spec in valid_cases()]
            assert cases == chat, name
            if reference is None:
                reference = cases
            assert cases == reference, name
            invalid = {route: verify_invalid(port, route, case) for route in (RAW, CHAT)}
            for route in (RAW, CHAT):
                for key, options in normal:
                    body = body_for(route, case, 12, ignore_eos=True) | options
                    assert result(port, route, case, body) == expected[key], (name, route, key)
            overlap = None
            if width > 1:
                overlap, greedy = verify_overlap(port, case, pieces)
                assert greedy == expected_long, name
            save(output / "results.json", {
                "valid": dict(zip((spec[0] for spec in valid_cases()), cases, strict=True)),
                "invalid": invalid,
                "overlap": overlap,
                "normal": expected,
            })
        log = (output / "server.log").read_text()
        if width > 1:
            assert re.search(r"batch width=[23] decode=[23] request_bytes=", log), name
        if mtp and width > 1:
            assert re.search(r"drafted=[1-4] accepted=", log), name
        print(f"PASS {name}: finite choices, UTF-8 wire, errors, and normal decoding", flush=True)


if __name__ == "__main__":
    main()
