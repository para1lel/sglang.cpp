"""Exercise decoded media on the native loopback server."""

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, contextmanager
from copy import deepcopy
import http.client
import json
from pathlib import Path
import select
import socket
import struct
import subprocess
import time


CHAT = "/v1/chat/completions"
TEXT = "/v1/completions"
MODEL = "Qwen3.8-27B-FP8"
HEADERS = {"Content-Type": "application/json", "Connection": "close"}
RGB8 = "data:application/x-sglang-rgb8;base64,"


def exchange(port, route, body=None, *, timeout=90):
    with closing(http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)) as client:
        client.request("GET" if body is None else "POST", route,
                       None if body is None else json.dumps(body), HEADERS)
        with client.getresponse() as response:
            return response.status, response.headers.get_content_type(), response.read()


def wait_ready(process, port):
    deadline = time.monotonic() + 60
    while process.poll() is None and time.monotonic() < deadline:
        try:
            if exchange(port, "/health")[0] == 200:
                return
        except OSError:
            time.sleep(0.1)
    raise AssertionError(("server did not become ready", process.poll()))


@contextmanager
def serve(args, label, *options):
    with closing(socket.socket()) as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    command = [str(args.server.resolve()), "--checkpoint", str(args.checkpoint.resolve()),
               "--device", "0", "--port", str(port), "--trace-batches", *options]
    log = args.evidence / f"{label}.server.log"
    (args.evidence / f"{label}.argv.json").write_text(json.dumps(command) + "\n")
    with log.open("wb") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
        try:
            wait_ready(process, port)
            yield port, log
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)


def visual(kind):
    frames = 2 if kind == "video_url" else 1
    pixels = bytes((index * 29 + index // 3072 * 113) % 256
                   for index in range(32 * 32 * 3 * frames))
    payload = {"url": RGB8 + base64.b64encode(pixels).decode(), "height": 32, "width": 32}
    if frames == 2:
        payload.update({"source_fps": 4, "source_frame_count": 5, "frame_indices": [0, 4]})
    return {"type": kind, kind: payload}


def chat(content, budget=4, stream=False):
    result = {"model": MODEL, "messages": [{"role": "user", "content": content}],
              "max_tokens": budget, "temperature": 0, "ignore_eos": True,
              "return_token_ids": True, "stream": stream}
    if stream:
        result["stream_options"] = {"include_usage": True}
    return result


def capture(args, name, port, route, body, status=200):
    code, content, raw = exchange(port, route, body)
    (args.evidence / f"{name}.request.json").write_text(json.dumps(body) + "\n")
    (args.evidence / f"{name}.body").write_bytes(raw)
    (args.evidence / f"{name}.http").write_text(f"{code} {content}\n")
    mime = "text/event-stream" if status == 200 and body.get("stream") else "application/json"
    assert code == status and content == mime, (name, code, content, raw[:300])
    return raw if mime == "text/event-stream" else json.loads(raw)


def negative(args, label, port, body, message):
    response = capture(args, label, port, CHAT, body, 400)
    assert message.lower() in response["error"]["message"].lower(), (label, response)


def malformed(args, port):
    image = visual("image_url")
    video = visual("video_url")
    for index, (part, field, value, message) in enumerate([
        (image, "width", 15, "width"),
        (image, "url", RGB8 + "AB==", "byte count"),
        (image, "url", RGB8 + "!" + image["image_url"]["url"][len(RGB8) + 1:],
         "canonical base64"),
        (video, "frame_indices", [3, 1], "nondecreasing"),
        (video, "source_fps", 0, "source_fps"),
    ]):
        changed = deepcopy(part)
        changed[part["type"]][field] = value
        negative(args, f"invalid-{index}", port, chat([image, changed], stream=True), message)
    for kind, mime in (("image_url", "image/png"), ("video_url", "video/mp4")):
        body = chat([{"type": kind, kind: {"url": f"data:{mime};base64,AAAA"}}])
        negative(args, f"codec-disabled-{kind}", port, body, "enabled inline media data URL")
    for field, value in (("cache_salt", "media"), ("ebnf", 'root ::= "Hello"')):
        negative(args, f"media-{field}", port, chat([image]) | {field: value}, field)
    negative(args, "too-many-parts", port, chat([image] * 5), "up to four parts")


def usage(body, budget):
    assert body["prompt_tokens"] > 0 and body["completion_tokens"] == budget, body
    assert body["total_tokens"] == body["prompt_tokens"] + budget, body
    assert body["prompt_tokens_details"] == {"cached_tokens": 0}, body


def received(reply, budget):
    choice, = reply["choices"]
    assert choice["finish_reason"] == "length" and len(choice["token_ids"]) == budget
    usage(reply["usage"], budget)
    return choice["token_ids"], reply["usage"]


def stream_result(raw, budget):
    frames = raw.rstrip(b"\n").split(b"\n\n")
    assert frames.pop() == b"data: [DONE]"
    assert all(frame.startswith(b"data: ") for frame in frames)
    messages = [json.loads(frame[6:]) for frame in frames]
    stats = messages.pop()["usage"]
    usage(stats, budget)
    assert messages[0]["choices"][0]["delta"] == {"role": "assistant"}
    choices = [message["choices"][0] for message in messages]
    assert all(choice["finish_reason"] is None for choice in choices[:-1])
    assert choices[-1]["finish_reason"] == "length"
    ids = [token for choice in choices for token in choice.get("token_ids", [])]
    assert len(ids) == budget
    return ids, stats, messages[0].get("prompt_token_ids")


def mixed_content():
    return [visual("image_url"), {"type": "text", "text": "Compare the picture"},
            visual("video_url"), {"type": "text", "text": " and the clip."}]


def ordinary(args, port, label, content):
    reply = capture(args, f"{label}-json", port, CHAT, chat(content))
    expected = received(reply, 4)
    raw = capture(args, f"{label}-sse", port, CHAT, chat(content, stream=True))
    ids, stats, prompt = stream_result(raw, 4)
    assert (ids, stats) == expected
    assert prompt == reply.get("prompt_token_ids")
    return expected, prompt


def start(port, body, *, fragment=False):
    client = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        client.connect()
        client.sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
        payload = json.dumps(body).encode()
        client.putrequest("POST", CHAT)
        for name, value in HEADERS.items():
            client.putheader(name, value)
        client.putheader("Content-Length", str(len(payload)))
        client.endheaders(payload[:-1] if fragment else payload)
    except BaseException:
        client.close()
        raise
    return client


def wait_retired(log, offset):
    deadline = time.monotonic() + 20
    with log.open("rb") as trace:
        while time.monotonic() < deadline:
            trace.seek(offset)
            if b"batch width=0 decode=0 request_bytes=0\n" in trace.read():
                return
            time.sleep(0.01)
    raise AssertionError("The reset media request did not retire with zero request bytes")


def wait_pending(args, port):
    deadline = time.monotonic() + 2
    running = b'vllm:num_requests_running{engine="0"} 1'
    waiting = b'vllm:num_requests_waiting{engine="0"} 1'
    while (remaining := deadline - time.monotonic()) > 0:
        status, _, raw = exchange(port, "/metrics", timeout=remaining)
        assert status == 200, (status, raw)
        lines = raw.splitlines()
        if running in lines and waiting in lines:
            (args.evidence / "fragmented-pending.prom").write_bytes(raw)
            return
    raise AssertionError("The complete request did not enter the pending queue")


def reject_overflow(args, label, first, second):
    ready, _, _ = select.select([first.sock, second.sock], [], [], 5)
    assert len(ready) == 1, (label, "expected one response while the worker was occupied")
    first_rejected = ready[0] is first.sock
    rejected, pending = (first, second) if first_rejected else (second, first)
    (args.evidence / f"{label}-queue-full-peer.txt").write_text(
        "first\n" if first_rejected else "second\n"
    )
    with rejected.getresponse() as response:
        raw = response.read()
        mime = response.headers.get_content_type()
        (args.evidence / f"{label}-queue-full.body").write_bytes(raw)
        (args.evidence / f"{label}-queue-full.http").write_text(f"{response.status} {mime}\n")
        assert (response.status, mime) == (503, "application/json"), (label, response.status, raw)
        assert json.loads(raw)["error"]["message"] == "Native completion queue is full", raw
    assert not select.select([pending.sock], [], [], 0)[0], (label, "pending request responded")
    return first_rejected


def cancel_and_replace(args, port, log, content, expected, *, fragment=False):
    label = "fragmented" if fragment else "whole"
    body = chat(content, 1024, True)
    with closing(start(port, body)) as active:
        transport = active.sock
        with active.getresponse() as response:
            assert response.status == 200
            for line in response:
                if line.startswith(b"data: ") and line.rstrip() != b"data: [DONE]":
                    choices = json.loads(line[6:])["choices"]
                    if choices and choices[0].get("token_ids"):
                        (args.evidence / f"{label}-active-first-token.sse").write_bytes(line)
                        break
            else:
                raise AssertionError("The active media stream ended before its first token")
            offset = log.stat().st_size
            with closing(start(port, body, fragment=fragment)) as first:
                with closing(start(port, body)) as second:
                    if fragment:
                        wait_pending(args, port)
                        first.send(b"}")
                    first_rejected = reject_overflow(args, label, first, second)
                    assert not fragment or first_rejected
            transport.close()
    wait_retired(log, offset)
    survivor = capture(args, f"{label}-survivor", port, CHAT, chat(content, stream=True))
    assert stream_result(survivor, 4) == expected


def main(args):
    args.evidence.mkdir()
    content = mixed_content()
    with serve(args, "width-one", "--max-active", "1", "--mtp-steps", "0") as (port, log):
        malformed(args, port)
        response = capture(args, "zero-budget", port, CHAT, chat(content, 0))
        received(response, 0)
        assert "batch width=" not in log.read_text()
        expected, prompt = ordinary(args, port, "mixed", content)
        assert len(prompt) == expected[1]["prompt_tokens"]
        cancel_and_replace(args, port, log, content, (*expected, prompt))
        ordinary(args, port, "plain-chat", "Say hello.")
        received(capture(args, "text-completion", port, TEXT,
                         {"model": MODEL, "prompt": "Hello", "max_tokens": 4,
                          "return_token_ids": True, "ignore_eos": True}), 4)
    with serve(args, "mtp-configured", "--max-active", "1", "--mtp-steps", "1") as (port, _):
        assert ordinary(args, port, "media-k0", content) == (expected, prompt)
    for label, options in (("continuous", ["--max-active", "2"]),
                           ("cached", ["--max-active", "1", "--prefix-cache-entries", "1"])):
        with serve(args, label, *options) as (port, log):
            assert ordinary(args, port, f"{label}-media", content) == (expected, prompt)
            if label == "cached":
                cancel_and_replace(args, port, log, content, (*expected, prompt), fragment=True)
            ordinary(args, port, f"{label}-plain", "Say hello.")
    print("Decoded media HTTP route contract passed.")


def mixed_service(args):
    args.evidence.mkdir()
    prompt = json.loads((Path(__file__).parents[2] / "benchmarks/streaming/W1.json").read_text())[
        "prompt_token_ids"]
    description = {"type": "text", "text": "Describe the visual content in detail. " * 16}
    queries = {
        "image": (CHAT, chat([visual("image_url"), description])),
        "video": (CHAT, chat([visual("video_url"), description])),
        "text": ("/generate", {"input_ids": prompt, "sampling_params": {
            "max_new_tokens": 32, "temperature": 0, "ignore_eos": True}}),
        "score": ("/v1/score", {"query": prompt[:127], "items": [[17], [24, 31]],
                                "label_token_ids": [17, 23, 31], "return_token_logprobs": True}),
    }

    def observe(port, label, name):
        route, request = queries[name]
        reply = capture(args, f"{label}-{name}", port, route, request)
        if name in ("image", "video"):
            result = received(reply, 4)
            assert result[1]["prompt_tokens"] > 128
            return result, reply["prompt_token_ids"]
        if name == "text":
            assert len(reply["output_ids"]) == 32
            return reply["output_ids"]
        assert len(reply["scores"]) == 2 and reply["usage"]["completion_tokens"] == 0
        return reply

    with serve(args, "mixed-service") as (port, _):
        observed = {name: observe(port, "serial", name) for name in queries}
        with ThreadPoolExecutor(max_workers=len(queries)) as executor:
            concurrent = list(executor.map(
                lambda name: observe(port, "concurrent", name), queries))
        assert dict(zip(queries, concurrent)) == observed
    print("Default mixed media, text, and label-score route contract passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("server", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--mixed-service", action="store_true")
    args = parser.parse_args()
    (mixed_service if args.mixed_service else main)(args)
