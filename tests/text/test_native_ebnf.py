"""Check native finite-choice EBNF and its OpenAI route boundaries on the checkpoint."""

import argparse
import itertools
import json
from pathlib import Path

from run_chat_layer1 import save, server
from test_native_chat import CHAT, CHECKPOINT, RAW, json_response, request, tokenizer_pieces
from test_native_constraint import grammar
from test_native_text import EOS, event
from test_sampling_routes_cpu import body


GENERATE = "/generate"
SPLIT = "\U0010fffd"


def sampling(ebnf, budget=8, **options):
    return {"ebnf": ebnf, "max_new_tokens": budget, "temperature": 0, **options}


def exchange(port, route, payload, status=200):
    connection, response = request(port, route, payload)
    try:
        with response:
            assert response.status == status, (route, payload, response.status, response.read())
            media = response.headers.get_content_type()
            if payload.get("stream") and status == 200:
                assert media == "text/event-stream", media
                data = list(iter(lambda: event(response), "[DONE]"))
                assert response.read() == b""
                return data
            assert media == "application/json", (route, media)
            data = json.load(response)
            if status == 400:
                assert "error" in data, data
            return data
    finally:
        connection.close()


def check_choice(packet, choices, budget, pieces, prompt_tokens=1):
    ids = packet["output_ids"]
    text = packet["text"]
    meta = packet["meta_info"]
    assert text in choices and 0 < len(ids) <= budget, packet
    assert (meta["prompt_tokens"], meta["completion_tokens"]) == (prompt_tokens, len(ids)), packet
    assert b"".join(pieces[token] for token in ids) == text.encode(), packet
    assert all(token not in EOS for token in ids[:-1]), packet
    if len(ids) == budget and ids[-1] not in EOS:
        assert meta["finish_reason"] == {"type": "length", "length": budget}, packet
    else:
        assert meta["finish_reason"] == {"type": "stop", "matched": ids[-1]}, packet
    if ids[-1] in EOS:
        assert any(other != text and other.startswith(text) for other in choices), packet


def check_formats(port, payload, evidence):
    plain = exchange(port, GENERATE, payload)
    frames = exchange(port, GENERATE, payload | {"stream": True})
    batch = isinstance(plain, list)
    final = plain if batch else [plain]
    previous = {index: {"text": "", "output_ids": []} for index in range(len(final))}
    finished = set()
    rids = {}
    partial = set()
    for frame in frames:
        index = frame["index"] if batch else 0
        assert ("index" in frame) == batch and index in previous and index not in finished, frame
        assert frame["meta_info"]["id"] == rids.setdefault(index, frame["meta_info"]["id"])
        assert frame["text"].startswith(previous[index]["text"]), frame
        ids = frame["output_ids"]
        earlier = previous[index]["output_ids"]
        assert len(ids) > len(earlier) and ids[:len(earlier)] == earlier, frame
        assert frame["meta_info"]["completion_tokens"] == len(ids), frame
        previous[index] = frame
        if frame["meta_info"]["finish_reason"] is not None:
            finished.add(index)
        else:
            partial.add(index)
    assert finished == previous.keys(), frames
    assert len(set(rids.values())) == len(final), rids
    for index, packet in enumerate(final):
        observed = previous[index].copy()
        observed.pop("index", None)
        observed["meta_info"] = observed["meta_info"] | {"id": packet["meta_info"]["id"]}
        assert observed == packet, (observed, packet)
    evidence.update({"json": plain, "sse": frames})
    return final, partial


def valid(port, pieces):
    records = {}
    for name, choices, budget in (
        ("leaf", ("A",), 4),
        ("cap", ("A",), 1),
        ("shared_prefix", ("hello", "hello!"), 1),
        ("escapes", ('"\\\n\0',), 8),
        ("split_utf8", (SPLIT,), 8),
    ):
        packets, partial = check_formats(port, {
            "input_ids": [32], "sampling_params": sampling(grammar(*choices), budget),
        }, records.setdefault(name, {}))
        allowed = ("hello",) if name == "shared_prefix" else choices
        check_choice(packets[0], allowed, budget, pieces)
        if name == "split_utf8":
            assert partial == {0} and len(packets[0]["output_ids"]) > 1
            assert records[name]["sse"][0]["text"] == "", records[name]

    options = sampling(grammar(SPLIT + "A"))
    packets, partial = check_formats(port, {
        "input_ids": [[32], [33]], "sampling_params": options,
    }, records.setdefault("broadcast", {}))
    assert partial == {0, 1}, records["broadcast"]
    for packet in packets:
        check_choice(packet, (SPLIT + "A",), 8, pieces)

    packets, partial = check_formats(port, {
        "input_ids": [[32], [33], [34], [35]],
        "sampling_params": [
            sampling(grammar(SPLIT + "B"), regex="", json_schema=None),
            {"regex": "C", "ebnf": None, "max_new_tokens": 8, "temperature": 0},
            {"json_schema": '{"enum":["D"]}', "ebnf": "", "max_new_tokens": 8, "temperature": 0},
            sampling("", 2, json_schema="", regex=None),
        ],
        "return_logprob": [False, False, False, True],
    }, records.setdefault("mixed", {}))
    for packet, choice in zip(packets[:3], (SPLIT + "B", "C", '"D"'), strict=True):
        check_choice(packet, (choice,), 8, pieces)
    assert 0 in partial and [len(packet["meta_info"].get("output_token_logprobs", []))
                             for packet in packets] == [0, 0, 0, 2], records["mixed"]

    maximum = grammar("A", *("x" * 256 for _ in range(7)), "x" * 255)
    limits = grammar(*("A" for _ in range(32)))
    for name, pattern in (
        ("byte_limit", " " * (16384 - len(limits)) + limits),
        ("decoded_limit", maximum),
    ):
        packet = exchange(port, GENERATE, {"text": "A", "sampling_params": sampling(pattern, 1)})
        check_choice(packet, ("A",), 1, pieces)
        records[name] = packet
    packets, _ = check_formats(port, {
        "input_ids": [32], "sampling_params": sampling(grammar("XYZ"), stop="Y"),
    }, records.setdefault("text_stop", {}))
    assert packets[0]["text"] == "X", packets
    assert packets[0]["meta_info"]["finish_reason"] == {"type": "stop", "matched": "Y"}
    return records


def invalid(port):
    choice = grammar("A")
    active = {"ebnf": choice, "regex": "A", "json_schema": '{"enum":["A"]}'}
    cases = {
        **{f"conflict_{left}_{right}": {left: active[left], right: active[right]}
           for left, right in itertools.combinations(active, 2)},
        "all_active": active,
        "number": {"ebnf": 1},
        "object": {"ebnf": {}},
        "empty_literal": {"ebnf": grammar("")},
        "wrong_rule": {"ebnf": 'other ::= "A"'},
        "missing_literal": {"ebnf": 'root ::= "A" |'},
        "rule_reference": {"ebnf": 'root ::= "A" | root'},
        "concat": {"ebnf": 'root ::= "A" "B"'},
        "repeat": {"ebnf": 'root ::= "A"+'},
        "additional_rule": {"ebnf": 'root ::= "A"\nother ::= "B"'},
        "bad_escape": {"ebnf": r'root ::= "\q"'},
        "bad_surrogate": {"ebnf": r'root ::= "\uD800"'},
        "whitespace": {"ebnf": " "},
        "choice_limit": {"ebnf": grammar(*("A" for _ in range(33)))},
        "literal_limit": {"ebnf": grammar("A" * 257)},
        "decoded_limit": {"ebnf": grammar(*("A" * 256 for _ in range(8)), "B")},
        "byte_limit": {"ebnf": " " * 16384 + choice},
        "budget": {"ebnf": grammar(SPLIT), "max_new_tokens": 1},
        "zero_budget": sampling(choice, 0),
        "ignore_eos": sampling(choice, ignore_eos=True),
        "override": sampling(choice, stop_token_ids=[min(EOS)]),
        "override_empty": sampling(choice, stop_token_ids=[]),
    }
    records = {}
    for name, options in cases.items():
        records[name] = exchange(port, GENERATE, {
            "input_ids": [[32], [33]], "stream": True,
            "sampling_params": [sampling(choice), options],
        }, 400)
    records["same_item_scores"] = exchange(port, GENERATE, {
        "input_ids": [[32], [33]], "stream": True, "return_logprob": [False, True],
        "sampling_params": [sampling(None), sampling(choice)],
    }, 400)
    records["top_level"] = exchange(port, GENERATE, {
        "input_ids": [32], "stream": True, "ebnf": choice,
    }, 400)
    return records


def openai(port):
    evidence = {}
    for route in (RAW, CHAT):
        fields = body(route, max_tokens=1, temperature=0, ebnf=grammar("A"))
        packet = json_response(port, route, fields)
        choice = packet["choices"][0]
        text = choice["message"]["content"] if route == CHAT else choice["text"]
        assert (text, choice["finish_reason"]) == ("A", "stop"), packet
        reject = exchange(port, route, body(route, stream=True, sampling_params=sampling(
            grammar("A"))), 400)
        evidence[route] = {"top_level": packet, "nested_rejection": reject}
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    args.evidence.mkdir()
    pieces = tokenizer_pieces(args.checkpoint)
    assert SPLIT.encode() not in pieces.values()
    with server(args.server, args.checkpoint, args.evidence / "listener", 4,
                flags=("--mtp-steps", "0")) as port:
        save(args.evidence / "valid.json", valid(port, pieces))
        save(args.evidence / "invalid.json", invalid(port))
        save(args.evidence / "openai.json", openai(port))
    print("Native EBNF JSON/SSE, batch isolation, rejections, and OpenAI wire checks passed")


if __name__ == "__main__":
    main()
