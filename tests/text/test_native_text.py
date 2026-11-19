"""Run against a pinned native server with --port and --tokenizer.

Pass --width 2 or --width 4 to match the server's --max-active setting.
"""

import argparse
import codecs
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import hashlib
import http.client
import json
from pathlib import Path
import socket
import struct
from threading import Barrier


MODEL = "Qwen3.8-27B-FP8"
FIXTURE = Path(__file__).with_name("cases.json")
HEADERS = {"Content-Type": "application/json", "Connection": "close"}
EOS = {248044, 248046}


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def raw(prompt, budget, **options):
    return {"model": MODEL, "prompt": prompt, "max_tokens": budget,
            "return_token_ids": True, "ignore_eos": True, "temperature": 0.0, "seed": 0,
            **options}


def legacy(prompt, budget, **options):
    return {"model": "benchmark", "prompt": prompt, "max_tokens": budget, "stream": True,
            "return_token_ids": True, "stream_interval": 1, "ignore_eos": True,
            "add_special_tokens": False, "temperature": 0.0, "seed": 0,
            "stream_options": {"include_usage": True}, **options}


def stream_options():
    return {"stream": True, "stream_options": {"include_usage": True}}


def start(connection, body, headers=HEADERS):
    if not isinstance(body, bytes):
        body = json.dumps(body, ensure_ascii=False).encode("utf-8")
    connection.request("POST", "/v1/completions", body, headers)
    return connection.getresponse()


def exchange(port, body, status=200, headers=HEADERS):
    with closing(http.client.HTTPConnection("127.0.0.1", port, timeout=60)) as connection:
        with start(connection, body, headers) as response:
            data = response.read()
            check(response.status == status,
                  f"expected HTTP {status}, got {response.status}: {data[:240]!r}")
            return response.headers.get_content_type(), data


def event(response):
    line = response.readline()
    check(line.startswith(b"data: ") and line.endswith(b"\n"), f"invalid SSE event: {line!r}")
    check(response.readline() == b"\n", "missing SSE event separator")
    return "[DONE]" if line == b"data: [DONE]\n" else json.loads(line[6:].decode("utf-8"))


def events(port, body):
    with closing(http.client.HTTPConnection("127.0.0.1", port, timeout=60)) as connection:
        with start(connection, body) as response:
            if response.status != 200:
                raise AssertionError(
                    f"expected HTTP 200 SSE, got {response.status}: {response.read(240)!r}"
                )
            check(response.headers.get_content_type() == "text/event-stream",
                  "invalid SSE content type")
            result = []
            while (message := event(response)) != "[DONE]":
                result.append(message)
            check(response.read(1) == b"", "SSE data after [DONE]")
            return result


def usage(prompt, generated):
    return {"prompt_tokens": len(prompt), "completion_tokens": len(generated),
            "total_tokens": len(prompt) + len(generated),
            "prompt_tokens_details": {"cached_tokens": 0}}


def choice(text, ids, reason=None):
    return {"choices": [{"index": 0, "text": text, "token_ids": ids, "finish_reason": reason}]}


def expected_choices(prompt, ids, pieces, reason):
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    output = [
        choice(decoder.decode(piece), [token]) for token, piece in zip(ids, pieces, strict=True)
    ]
    if tail := decoder.decode(b"", final=True):
        output.append(choice(tail, []))
    output.append(choice("", [], reason))
    output[0]["choices"][0]["prompt_token_ids"] = prompt
    return output


def checkpoint_pieces(path, digest):
    data = path.read_bytes()
    check(hashlib.sha256(data).hexdigest() == digest, "tokenizer fixture SHA-256 mismatch")
    tokenizer = json.loads(data)
    ordinary = set(range(33, 127)) | set(range(161, 173)) | set(range(174, 256))
    escaped = iter(range(256, 512))
    inverse = {chr(byte if byte in ordinary else next(escaped)): byte for byte in range(256)}
    pieces = {token: bytes(inverse[char] for char in symbol)
              for symbol, token in tokenizer["model"]["vocab"].items()}
    pieces.update({entry["id"]: b"" if entry["special"] else entry["content"].encode("utf-8")
                   for entry in tokenizer["added_tokens"]})
    return pieces


def generated_ids(port, prompt, budget):
    received = events(port, legacy(prompt, budget))
    payloads, terminal, counts = received[:-2], received[-2], received[-1]
    ids = []
    for index, payload in enumerate(payloads):
        entry, = payload["choices"]
        echo = prompt if index == 0 else None
        check(entry["finish_reason"] is None and len(entry["token_ids"]) == 1 and
              entry.get("prompt_token_ids") == echo, "legacy token event")
        ids.extend(entry["token_ids"])
    entry, = terminal["choices"]
    echo = prompt if budget == 0 else None
    check(entry["finish_reason"] == "length" and entry["token_ids"] == [] and
          entry.get("prompt_token_ids") == echo, "legacy terminal event")
    expected = usage(prompt, ids)
    del expected["total_tokens"]
    check(len(ids) == budget and counts == {"choices": [], "usage": expected}, "legacy usage")
    return ids


def verify_raw(port, sample, ids, pieces, *, reason="length", overrides=None):
    prompt = sample["ids"]
    body = raw(sample["text"], sample["budget"], **(overrides or {}))
    expected = expected_choices(prompt, ids, pieces, reason)
    counts = usage(prompt, ids)
    expected_stream = expected + [{"choices": [], "usage": counts}]
    check(events(port, {**body, **stream_options()}) == expected_stream,
          f"raw SSE differs for {sample['text'][:40]!r}")

    media_type, data = exchange(port, body)
    text = "".join(message["choices"][0]["text"] for message in expected)
    result = choice(text, ids, reason)
    result["choices"][0]["prompt_token_ids"] = prompt
    result["usage"] = counts
    check(media_type == "application/json" and json.loads(data.decode("utf-8")) == result,
          f"raw JSON differs for {sample['text'][:40]!r}")


def verify_errors(port, tokenizer_ids):
    invalid = (
        raw("", 0), raw([220], 0), raw("x", -1), raw("a", 0, stop_token_ids=[-1]),
        raw("a", 0, stop_token_ids=[248320]), legacy([-1], 0), legacy([248320], 0),
        legacy([220], 0, stop_token_ids=[248320]), raw("x y z", 262143),
        raw("é" * 524289, 0), raw("x", 0, temperature=-1),
        raw("x", 0, stream_options={"include_usage": True}),
        b'{"model":"Qwen3.8-27B-FP8","prompt":"\xc0\xaf"}',
        b'{"model":"Qwen3.8-27B-FP8","prompt":"\\ud800"}',
        b'{"model":"Qwen3.8-27B-FP8","prompt":',
        b'{"model":"Qwen3.8-27B-FP8","prompt":"x","prompt":"y"}',
    )
    for body in invalid:
        media_type, data = exchange(port, body, 400)
        check(media_type == "application/json" and "error" in json.loads(data),
              "invalid error body")
    media_type, data = exchange(port, b"", 413, {**HEADERS, "Content-Length": "2097153"})
    check(media_type == "application/json" and "error" in json.loads(data), "oversize HTTP body")
    generated_ids(port, [248319], 0)
    verify_raw(port, {"text": "248320 -1", "ids": tokenizer_ids, "budget": 0}, [], [])


def verify_sampling(port, sample, greedy_ids, pieces):
    for options in ({"temperature": 0}, {"temperature": 0.0000009, "seed": -1},
                    {"temperature": 3, "top_k": 1, "top_p": 0.1}):
        _, data = exchange(port, raw(sample["text"], sample["budget"], **options))
        selected = json.loads(data)["choices"][0]["token_ids"]
        check(selected == greedy_ids, f"greedy IDs changed with {options!r}")

    sampling = {"temperature": 1.1, "top_p": 0.91, "top_k": 24}
    for seed in (-(1 << 63), 0, (1 << 63) - 1):
        body = raw(sample["text"], sample["budget"], **sampling, seed=seed)
        _, data = exchange(port, body)
        packet = json.loads(data)
        selected = packet["choices"][0]["token_ids"]
        check(len(selected) == sample["budget"], f"sampled length for seed {seed}")
        token_pieces = [pieces.get(token, b"") for token in selected]
        expected = expected_choices(sample["ids"], selected, token_pieces, "length")
        streamed = events(port, body | stream_options())
        check(streamed == expected + [{"choices": [], "usage": usage(sample["ids"], selected)}],
              f"sampled JSON/SSE IDs differ for seed {seed}")
        check(packet["choices"][0]["text"] ==
              "".join(frame["choices"][0]["text"] for frame in expected),
              f"sampled UTF-8 text differs for seed {seed}")

    for options in ({"seed": (1 << 63)}, {"seed": -(1 << 63) - 1}, {"seed": 4.0},
                    {"top_k": 1.5}, {"top_p": 0}, {"temperature": "1"}):
        exchange(port, raw(sample["text"], 0, **options), 400)


def pending_start(ids, pieces):
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    for index, token in enumerate(ids):
        was_empty = not decoder.getstate()[0]
        delta = decoder.decode(pieces.get(token, b""))
        if was_empty and not delta and decoder.getstate()[0] and token not in ids[:index]:
            return index
    raise AssertionError("fixture did not generate the start of a split UTF-8 scalar")


def verify_terminal(port, cases, generated, pieces):
    han = cases["han_partial"]
    han_ids = generated["han_partial"]
    split = pending_start(han_ids, pieces)
    ids = han_ids[:split + 1]
    chunks = [pieces.get(token, b"") for token in ids]
    verify_raw(port, {**han, "budget": 0}, [], [])
    verify_raw(port, {**han, "budget": split + 1}, ids, chunks)
    verify_raw(port, {**han, "budget": split + 1}, ids, [*chunks[:-1], b""], reason="stop",
               overrides={"ignore_eos": False, "stop_token_ids": [ids[-1]]})

    chat_ids = generated["chatml"]
    eos = next(index for index, token in enumerate(chat_ids) if token in EOS)
    eos_ids = chat_ids[:eos + 1]
    verify_raw(port, cases["chatml"], eos_ids, [pieces.get(token, b"") for token in eos_ids],
               reason="stop", overrides={"ignore_eos": False})
    without_ids = raw(han["text"], 0, return_token_ids=False)
    check(events(port, {**without_ids, "stream": True}) ==
          [{"choices": [{"index": 0, "text": "", "finish_reason": "length"}]}],
          "optional SSE metadata must be absent")
    media_type, data = exchange(port, without_ids)
    check(media_type == "application/json" and json.loads(data) ==
          {"choices": [{"index": 0, "text": "", "finish_reason": "length"}],
           "usage": usage(han["ids"], [])}, "optional JSON metadata must be absent")


def cancel_pending(port, sample, ids, pieces):
    split = pending_start(ids, pieces)
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    with closing(http.client.HTTPConnection("127.0.0.1", port, timeout=60)) as connection:
        connection.connect()
        transport = connection.sock
        with start(connection, raw(sample["text"], 4096, **stream_options())) as response:
            check(response.status == 200, "expected cancellation SSE stream")
            for index, token in enumerate(ids[:split + 1]):
                expected = choice(decoder.decode(pieces.get(token, b"")), [token])
                if index == 0:
                    expected["choices"][0]["prompt_token_ids"] = sample["ids"]
                check(event(response) == expected, f"unexpected pre-cancel token at {index}")
            check(decoder.getstate()[0], "no pending UTF-8 at cancellation")
            transport.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
    return {"token_ids": ids[:split + 1], "pending_hex": decoder.getstate()[0].hex()}


def verify_concurrent(port, width, cases, generated, pieces):
    keys = ("han_partial", "emoji_partial", "cjk", "combining")[:width]
    barrier = Barrier(width, timeout=10)

    def run(key):
        case, ids = cases[key], generated[key]
        barrier.wait()
        if key == "han_partial":
            return cancel_pending(port, case, ids, pieces)
        chunks = [pieces.get(token, b"") for token in ids]
        expected = expected_choices(case["ids"], ids, chunks, "length")
        expected.append({"choices": [], "usage": usage(case["ids"], ids)})
        check(events(port, raw(case["text"], case["budget"], **stream_options())) == expected,
              f"concurrent response differs for {key}")

    with ThreadPoolExecutor(max_workers=width) as pool:
        cancellation, *_ = pool.map(run, keys)
    ids = generated["han_partial"]
    verify_raw(port, cases["han_partial"], ids, [pieces.get(token, b"") for token in ids])
    return cancellation


def verify_unavailable_tokenizer(port, fixture):
    sample = {**fixture, "budget": 1}
    expected = generated_ids(port, sample["ids"], sample["budget"])
    content_type, data = exchange(port, raw(sample["text"], sample["budget"]), 500)
    check(content_type == "application/json" and "error" in json.loads(data),
          "tokenizer failure must return an HTTP error")
    check(generated_ids(port, sample["ids"], sample["budget"]) == expected,
          "tokenizer failure changed generation from loaded weights")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--tokenizer", type=Path, help="pinned tokenizer.json for positive checks")
    mode.add_argument("--expect-tokenizer-error", action="store_true",
                      help="check a server started with a missing or corrupt tokenizer")
    parser.add_argument("--width", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--evidence", type=Path)
    args = parser.parse_args()
    fixture = json.loads(FIXTURE.read_bytes())
    cases = fixture["cases"]
    if args.expect_tokenizer_error:
        verify_unavailable_tokenizer(args.port, cases["cjk"])
        print("Native text tokenizer error and loaded-weight checks passed")
        return
    pieces = checkpoint_pieces(args.tokenizer, fixture["tokenizer_sha256"])
    generated = {}
    for name, case in cases.items():
        ids = generated_ids(args.port, case["ids"], case["budget"])
        verify_raw(args.port, case, ids, [pieces.get(token, b"") for token in ids])
        generated[name] = ids
    verify_terminal(args.port, cases, generated, pieces)
    verify_errors(args.port, fixture["numeric_text_ids"])
    verify_sampling(args.port, cases["cjk"], generated["cjk"], pieces)
    cancellation = verify_concurrent(args.port, args.width, cases, generated, pieces)
    if args.evidence:
        result = {"width": args.width, "generated_ids": generated, "cancellation": cancellation}
        args.evidence.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Native text HTTP passed for {len(cases)} prompt cases at width {args.width}")


if __name__ == "__main__":
    main()
