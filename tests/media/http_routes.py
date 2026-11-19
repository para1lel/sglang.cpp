"""Exercise decoded media on the native loopback server."""

import argparse
import base64
from contextlib import closing, contextmanager
from copy import deepcopy
import http.client
import json
from pathlib import Path
import socket
import struct
import subprocess
import time


CHAT = "/v1/chat/completions"
TEXT = "/v1/completions"
MODEL = "Qwen3.8-27B-FP8"
HEADERS = {"Content-Type": "application/json", "Connection": "close"}
RGB8 = "data:application/x-sglang-rgb8;base64,"


def exchange(port, route, body=None):
    with closing(http.client.HTTPConnection("127.0.0.1", port, timeout=90)) as client:
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
               "--device", "0", "--port", str(port), "--fp8-prefill", "exact",
               "--trace-batches", *options]
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
        negative(args, f"codec-disabled-{kind}", port, body, "decoded")
    for field, value in (("cache_salt", "media"), ("ebnf", 'root ::= "Hello"')):
        negative(args, f"media-{field}", port, chat([image]) | {field: value}, field)
    negative(args, "too-many-parts", port, chat([image] * 5), "at most four")


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


def start(port, body):
    client = http.client.HTTPConnection("127.0.0.1", port, timeout=90)
    try:
        client.connect()
        client.sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
        client.request("POST", CHAT, json.dumps(body), HEADERS)
    except BaseException:
        client.close()
        raise
    return client


def cancel_and_replace(args, port, log, content, expected):
    with closing(start(port, chat(content, 1024, True))) as active:
        transport = active.sock
        with active.getresponse() as response:
            assert response.status == 200
            while True:
                line = response.readline()
                assert line
                if line.startswith(b"data: ") and b'"token_ids":[' in line:
                    (args.evidence / "active-first-token.sse").write_bytes(line)
                    break
            with closing(start(port, chat(content, 1024, True))) as cancelled:
                with closing(start(port, chat(content, stream=True))) as survivor:
                    cancelled.close()
                    transport.close()
                    response.close()
                    with survivor.getresponse() as survived:
                        assert survived.status == 200
                        raw = survived.read()
                        (args.evidence / "survivor.sse").write_bytes(raw)
    assert stream_result(raw, 4) == expected
    assert "batch width=0 decode=0" in log.read_text()


def main(args):
    args.evidence.mkdir()
    content = mixed_content()
    with serve(args, "serial", "--max-active", "1", "--mtp-steps", "0") as (port, log):
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
        with serve(args, label, *options) as (port, _):
            negative(args, f"{label}-media", port, chat(content), "serial")
            ordinary(args, port, f"{label}-plain", "Say hello.")
    print("Decoded media HTTP route contract passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("server", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("evidence", type=Path)
    main(parser.parse_args())
