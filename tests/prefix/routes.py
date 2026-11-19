"""Run against a native batch server with a fresh two-entry cache."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import gzip
import hashlib
import http.client
import json
from pathlib import Path
import socket
import struct
from threading import Barrier


MODEL = "Qwen3.8-27B-FP8"
ROW_BYTES = 8 + 2 * 248320
HEADERS = {"Content-Type": "application/json", "Connection": "close"}
FIXTURE = Path(__file__).with_name("route-cases.json")


def oracle(path):
    raw = path.read_bytes()
    assert raw and len(raw) % ROW_BYTES == 0, path
    results = []
    for offset in range(0, len(raw), ROW_BYTES):
        token, count = struct.unpack_from("<ii", raw, offset)
        assert 0 <= token < 248320 and count == 248320, path
        results.append(token)
    return results


def benchmark(prompt, count, **fields):
    return {"model": "benchmark", "prompt": prompt, "max_tokens": count, "stream": True,
            "return_token_ids": True, "stream_interval": 1, "ignore_eos": True,
            "add_special_tokens": False, "temperature": 0.0, "seed": 0,
            "stream_options": {"include_usage": True}, **fields}


class Routes:
    def __init__(self, port, directory):
        self.port = port
        self.directory = directory

    def raw(self, name, body, code=200, route="/v1/completions"):
        encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
        with closing(http.client.HTTPConnection("127.0.0.1", self.port, timeout=90)) as connection:
            connection.request("POST", route, encoded, HEADERS)
            with connection.getresponse() as response:
                raw = response.read()
                status = response.status
                content = response.headers.get_content_type()
        (self.directory / f"{name}.request.json").write_bytes(encoded)
        with gzip.open(self.directory / f"{name}.response.gz", "wb") as stream:
            stream.write(raw)
        assert status == code, (name, status, raw[:240])
        if code != 200:
            assert content == "application/json" and "error" in json.loads(raw), name
        return raw, content

    def benchmark(self, name, prompt, ids, cached, salt, reason="length", budget=None):
        body = benchmark(prompt, len(ids) if budget is None else budget, cache_salt=salt)
        if reason == "stop":
            body["stop_token_ids"] = ids[-1:]
        raw, content = self.raw(name, body)
        assert content == "text/event-stream", name
        events = raw.rstrip(b"\n").split(b"\n\n")
        assert events.pop() == b"data: [DONE]", name
        messages = [json.loads(event.removeprefix(b"data: ")) for event in events]
        self.usage(name, messages.pop()["usage"], len(prompt), len(ids), cached)
        terminal, = messages.pop()["choices"]
        assert terminal["token_ids"] == [] and terminal["finish_reason"] == reason, name
        if not ids:
            assert terminal["prompt_token_ids"] == prompt, name
        for index, (message, token) in enumerate(zip(messages, ids, strict=True)):
            choice, = message["choices"]
            assert choice["token_ids"] == [token] and choice["finish_reason"] is None, name
            assert choice.get("prompt_token_ids") == (prompt if index == 0 else None), name

    def text(self, name, body, ids, cached, route="/v1/completions"):
        raw, content = self.raw(name, body, route=route)
        assert content == "application/json", name
        result = json.loads(raw)
        choice, = result["choices"]
        prompt = choice["prompt_token_ids"]
        assert choice["token_ids"] == ids and choice["finish_reason"] == "length", name
        self.usage(name, result["usage"], len(prompt), len(ids), cached)
        return prompt, choice["token_ids"]

    @staticmethod
    def usage(name, usage, prompt, completion, cached):
        assert usage["prompt_tokens"] == prompt and usage["completion_tokens"] == completion, name
        assert usage["prompt_tokens_details"] == {"cached_tokens": cached}, (name, usage)
        if "total_tokens" in usage:
            assert usage["total_tokens"] == prompt + completion, name

    def cancel(self, name, prompt, token, salt, barrier):
        body = json.dumps(benchmark(prompt, 4096, cache_salt=salt)).encode("utf-8")
        (self.directory / f"{name}.request.json").write_bytes(body)
        with closing(http.client.HTTPConnection("127.0.0.1", self.port, timeout=90)) as connection:
            connection.connect()
            transport = connection.sock
            barrier.wait()
            connection.request("POST", "/v1/completions", body, HEADERS)
            with connection.getresponse() as response:
                line = response.readline()
                assert response.status == 200 and line.startswith(b"data: "), name
                choice, = json.loads(line[6:])["choices"]
                assert choice["token_ids"] == [token], name
                transport.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
                (self.directory / f"{name}.first-event").write_bytes(line)


def test_boundaries(routes, inputs, disabled):
    if inputs is None:
        fixture = json.loads(FIXTURE.read_text())
        data = {int(size): (fixture["prompt"][:int(size)], ids)
                for size, ids in fixture["output"].items()}
    else:
        data = {}
        for size in (127, 128, 129, 255, 256, 257):
            path = inputs / f"page-{size}.ids"
            data[size] = ([int(token) for token in path.read_text().split()],
                          oracle(path.with_suffix(".ids.bf16")))
    hit = 0 if disabled else 128
    cases = ((127, 0), (128, 0), (128, hit), (129, hit), (255, hit),
             (256, hit), (256, hit * 2), (257, hit * 2))
    for index, (size, cached) in enumerate(cases):
        prompt, ids = data[size]
        routes.benchmark(f"page-{index}-{size}", prompt, ids, cached, "boundaries")
    prompt, ids = data[128]
    routes.benchmark("exact-stop", prompt, ids[:1], hit, "boundaries", "stop", budget=4)
    routes.benchmark("zero-budget", prompt, [], 0, "boundaries")
    routes.benchmark("repeat", prompt, ids, hit, "boundaries")
    routes.benchmark("isolated-salt", prompt, ids, 0, "boundaries\0different")
    return data


def test_cancellation(routes, data, disabled):
    prompt, ids = data[128]
    peer_prompt, peer_ids = data[129]
    replacement_prompt, replacement_ids = data[255]
    hit = 0 if disabled else 128
    routes.benchmark("cancel-prime", prompt, ids, 0, "cancel")
    barrier = Barrier(2, timeout=10)

    def peer():
        barrier.wait()
        routes.benchmark("cancel-peer", peer_prompt, peer_ids, hit, "cancel")

    with ThreadPoolExecutor(max_workers=2) as pool:
        aborted = pool.submit(routes.cancel, "cancel-hit", prompt, ids[0], "cancel", barrier)
        completed = pool.submit(peer)
        aborted.result()
        completed.result()
    routes.benchmark("cancel-replacement", replacement_prompt, replacement_ids, hit, "cancel")


def test_salts(routes, disabled):
    hit = 0 if disabled else 128
    text = {"model": MODEL, "prompt": "Prefix cache fixture. " * 70,
            "max_tokens": 1, "return_token_ids": True, "ignore_eos": True}
    for index, salt in enumerate((None, 123, [], {}, "é" * 129, "x" * 257)):
        routes.raw(f"legacy-invalid-{index}", benchmark([220], 0, cache_salt=salt), 400)
        routes.raw(f"text-invalid-{index}", {**text, "cache_salt": salt}, 400)
    expected, output = text_oracle(routes, "text-prime", {**text, "cache_salt": "é" * 128})
    assert len(expected) >= 128, len(expected)
    checkpoint = len(expected) // 128 * hit
    hit_prompt, _ = routes.text("text-hit", {**text, "cache_salt": "é" * 128}, output, checkpoint)
    assert hit_prompt == expected
    routes.text("text-miss", {**text, "cache_salt": "é" * 127}, output, 0)
    chat = {"model": MODEL, "messages": [{"role": "user", "content": text["prompt"]}],
            "max_tokens": 1, "ignore_eos": True, "return_token_ids": True,
            "cache_salt": "chat"}
    route = "/v1/chat/completions"
    prompt, output = text_oracle(routes, "chat-prime", chat, route)
    assert len(prompt) >= 128, len(prompt)
    routes.text("chat-hit", chat, output, len(prompt) // 128 * hit, route)
    routes.raw("chat-invalid", {**chat, "cache_salt": False}, 400, route)


def text_oracle(routes, name, body, route="/v1/completions"):
    raw, content = routes.raw(name, body, route=route)
    assert content == "application/json", name
    result = json.loads(raw)
    choice, = result["choices"]
    prompt = choice["prompt_token_ids"]
    ids = choice["token_ids"]
    assert len(ids) == body["max_tokens"] and choice["finish_reason"] == "length", name
    routes.usage(name, result["usage"], len(prompt), len(ids), 0)
    return prompt, ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--disabled", action="store_true")
    args = parser.parse_args()
    args.evidence.mkdir()
    routes = Routes(args.port, args.evidence)
    data = test_boundaries(routes, args.inputs, args.disabled)
    test_cancellation(routes, data, args.disabled)
    test_salts(routes, args.disabled)
    artifact = args.evidence / "passed.json"
    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sorted(args.evidence.iterdir())}
    artifact.write_text(json.dumps({"disabled": args.disabled, "files": hashes}, indent=2) + "\n")
    print(f"Verified {len(hashes)} prefix route artifacts")


if __name__ == "__main__":
    main()
