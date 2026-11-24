"""Exercise decoded media on the native loopback server."""

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, contextmanager
from copy import deepcopy
import http.client
import json
import math
from pathlib import Path
import select
import socket
import struct
import subprocess
import time


CHAT = "/v1/chat/completions"
TEXT = "/v1/completions"
GENERATE = "/generate"
SCORE = "/v1/score"
MODEL = "Qwen3.8-27B-FP8"
HEADERS = {"Content-Type": "application/json", "Connection": "close"}
RGB8 = "data:application/x-sglang-rgb8;base64,"
CACHE_PAGE_TOKENS = 128
CHECKPOINT_INTERVAL = 8192


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


def visual(kind, width=32, height=32):
    frames = 2 if kind == "video_url" else 1
    pixels = bytes((index * 29 + index // 3072 * 113) % 256
                   for index in range(width * height * 3 * frames))
    payload = {"url": RGB8 + base64.b64encode(pixels).decode(), "height": height, "width": width}
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
    negative(args, "too-many-parts", port, chat([image] * 5), "up to four parts")


def usage(body, budget, cached=0):
    assert body["prompt_tokens"] > 0 and body["completion_tokens"] == budget, body
    assert body["total_tokens"] == body["prompt_tokens"] + budget, body
    assert body["prompt_tokens_details"] == {"cached_tokens": cached}, body


def received(reply, budget, cached=0):
    choice, = reply["choices"]
    assert choice["finish_reason"] == "length" and len(choice["token_ids"]) == budget
    usage(reply["usage"], budget, cached)
    return choice["token_ids"], reply["usage"]


def stream_frames(raw):
    frames = raw.rstrip(b"\n").split(b"\n\n")
    assert frames.pop() == b"data: [DONE]"
    assert all(frame.startswith(b"data: ") for frame in frames)
    return [json.loads(frame[6:]) for frame in frames]


def stream_result(raw, budget, cached=0):
    messages = stream_frames(raw)
    stats = messages.pop()["usage"]
    usage(stats, budget, cached)
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
    prompt = reply.get("prompt_token_ids")
    raw = capture(args, f"{label}-sse", port, CHAT, chat(content, stream=True))
    ids, stats, streamed_prompt = stream_result(raw, 4)
    assert (ids, without_cached_tokens(stats)) == (expected[0], without_cached_tokens(expected[1]))
    assert streamed_prompt == prompt
    return expected, prompt


def without_cached_tokens(stats):
    return {key: value for key, value in stats.items() if key != "prompt_tokens_details"}


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


def first_stream_token(response):
    assert response.status == 200
    for line in response:
        if line.startswith(b"data: ") and line.rstrip() != b"data: [DONE]":
            choices = json.loads(line[6:])["choices"]
            if choices and choices[0].get("token_ids"):
                return line
    raise AssertionError("The active media stream ended before its first token")


def cancel_and_replace(args, port, log, content, expected, *, fragment=False, cached=0):
    label = "fragmented" if fragment else "whole"
    body = chat(content, 1024, True)
    with closing(start(port, body)) as active:
        transport = active.sock
        with active.getresponse() as response:
            line = first_stream_token(response)
            (args.evidence / f"{label}-active-first-token.sse").write_bytes(line)
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
    ids, stats, prompt = stream_result(survivor, 4, cached)
    assert (ids, without_cached_tokens(stats), prompt) == (
        expected[0], without_cached_tokens(expected[1]), expected[2])


def main(args):
    args.evidence.mkdir()
    content = mixed_content()
    with serve(args, "width-one", "--max-active", "1", "--mtp-steps", "0",
               "--prefix-cache-entries", "0") as (port, log):
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
    with serve(args, "mtp-configured", "--max-active", "1", "--mtp-steps", "1",
               "--prefix-cache-entries", "0") as (port, _):
        assert ordinary(args, port, "media-k1", content) == (expected, prompt)
    for label, options in (("continuous", ["--max-active", "2", "--prefix-cache-entries", "0"]),
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

    with serve(args, "mixed-service", "--prefix-cache-entries", "0") as (port, _):
        observed = {name: observe(port, "serial", name) for name in queries}
        with ThreadPoolExecutor(max_workers=len(queries)) as executor:
            concurrent = list(executor.map(
                lambda name: observe(port, "concurrent", name), queries))
        assert dict(zip(queries, concurrent)) == observed
    print("Default mixed media, text, and label-score route contract passed.")


def native_text(ids, budget=8):
    return {"input_ids": ids, "sampling_params": {
        "max_new_tokens": budget, "temperature": 0, "ignore_eos": True}}


def native_result(reply, prompt, budget, cached, start=None):
    meta = dict(reply["meta_info"])
    assert meta.pop("prompt_tokens") == len(prompt), reply
    assert meta.pop("cached_tokens") == cached, reply
    assert meta.pop("completion_tokens") == len(reply["output_ids"]) == budget, reply
    assert meta.pop("finish_reason") == {"type": "length", "length": budget}, reply
    meta.pop("id")
    if start is not None:
        selected = meta["input_token_logprobs"]
        assert [row[1] for row in selected] == (prompt[start:] if start >= 0 else []), reply
        assert meta["output_token_logprobs_length"] == budget, reply
        output = meta["output_token_logprobs"]
        assert [row[1] for row in output] == reply["output_ids"], reply
        assert all(math.isfinite(row[0]) for row in output), reply
        if selected:
            assert selected[0][0] is None and meta["input_top_logprobs"][0] is None, reply
            assert all(math.isfinite(row[0]) for row in selected[1:]), reply
        for rows in (meta["input_top_logprobs"][1:], meta["output_top_logprobs"]):
            assert all(len(row) == 5 and all(math.isfinite(score[0]) for score in row)
                       for row in rows), reply
    return reply["output_ids"], reply["text"], meta


def score_body(prompt, start):
    return native_text(prompt, 4) | {
        "return_logprob": True, "logprob_start_len": start, "top_logprobs_num": 5}


def server_policy(args, port, label, entries):
    status, _, raw = exchange(port, "/server_info")
    (args.evidence / f"{label}-server-info.json").write_bytes(raw)
    info = json.loads(raw)
    expected = {"dp_size": 1, "tp_size": 1, "pp_size": 1, "mtp_steps": 4,
                "max_running_requests": 8, "max_num_batched_tokens": 32768,
                "prefix_cache_entries": entries, "enable_multimodal": True,
                "fp8_prefill": "tensor-core", "paired_prefill": True,
                "target_gqa": "grouped-split", "batch_cuda_graphs": True, "eager": False}
    assert status == 200 and {key: info[key] for key in expected} == expected, info


def feature_media(args, port, log, request, image_or_video, media, controls):
    for name in ("image", "video"):
        output, prompt = controls[name]
        assert len(prompt) > CHECKPOINT_INTERVAL, len(prompt)
        assert image_or_video(port, f"default-cold-{name}", media[name], 0) == (output, prompt)
        branch = f"{name}-branch"
        branch_prompt = controls[branch][1]
        assert branch_prompt != prompt
        assert branch_prompt[:CHECKPOINT_INTERVAL] == prompt[:CHECKPOINT_INTERVAL]
        for stream in (True, False):
            assert image_or_video(port, f"default-{branch}-{stream}", media[branch],
                                  CHECKPOINT_INTERVAL, stream=stream) == controls[branch]
    assert controls["image"][1] == controls["changed-pixels"][1]
    assert controls["image"][1] == controls["changed-grid"][1]
    assert controls["video"][1] != controls["changed-positions"][1]
    for name, base in (("changed-pixels", "image"), ("changed-grid", "image"),
                       ("changed-positions", "video"), ("salt-distinct", "image")):
        assert request(port, f"reset-before-{name}", "/reset_prefix_cache", {}) == {"success": True}
        assert image_or_video(port, f"prime-before-{name}", media[base], 0) == controls[base]
        assert image_or_video(port, f"default-{name}", media[name], 0) == controls[name]
    cancelled = media["short"] | {"max_tokens": 1024, "stream": True, "cache_salt": "cancelled"}
    with closing(start(port, cancelled)) as active:
        transport = active.sock
        with active.getresponse() as response:
            first = args.evidence / "default-cancel-first-token.sse"
            first.write_bytes(first_stream_token(response))
            offset = log.stat().st_size
            transport.close()
    wait_retired(log, offset)
    replacement = media["short"] | {"cache_salt": "cancelled"}
    assert image_or_video(port, "default-after-cancel", replacement, 0,
                          stream=True) == controls["short"]
    wait_retired(log, 0)


def check_label_order(got, reversed_labels):
    assert got["token_logprobs"] == [row[::-1] for row in reversed_labels["token_logprobs"]]
    for scores, logprobs, reverse in zip(got["scores"], got["token_logprobs"],
                                        reversed_labels["scores"], strict=True):
        weights = [math.exp(value - max(logprobs)) for value in logprobs]
        total = math.fsum(weights)
        for actual, weight, other in zip(scores, weights, reversed(reverse), strict=True):
            assert math.isclose(actual, weight / total, rel_tol=1e-12), (scores, logprobs)
            assert math.isclose(actual, other, rel_tol=1e-12), (scores, reverse)


def checkpoint_media_fixtures():
    image = visual("image_url", 32, 64)
    video = visual("video_url", 32, 64)
    changed_pixels = deepcopy(image)
    changed_pixels["image_url"]["url"] = RGB8 + base64.b64encode(
        bytes((195, 30, 80)) * 2048).decode()
    changed_positions = deepcopy(video)
    changed_positions["video_url"]["frame_indices"] = [0, 3]
    stem = "Describe the visual content in detail. " * 1250

    def media_body(visual_part, suffix=""):
        return chat([visual_part, {"type": "text", "text": stem + suffix}], budget=8)

    return {"image": media_body(image), "video": media_body(video),
            "image-branch": media_body(image, "Explain the result in French."),
            "video-branch": media_body(video, "Describe the last frame."),
            "changed-pixels": media_body(changed_pixels),
            "changed-grid": media_body(visual("image_url", 64, 32)),
            "changed-positions": media_body(changed_positions),
            "salt-distinct": media_body(image) | {"cache_salt": "media\0a"},
            "short": chat([image, {"type": "text", "text": "Describe the image."}], budget=8)}


def feature_complete(args):
    args.evidence.mkdir()
    fixture = Path(__file__).resolve().parents[2] / "benchmarks/streaming/W2.json"
    ids = json.loads(fixture.read_text())["prompt_token_ids"]
    prefix = ids * 4
    extension = prefix + ids[:1024]
    scoring = ids[:132]
    media = checkpoint_media_fixtures()
    labels = [[31, 17, 23], [29, 23, 17]]
    score = {"query": scoring[:128], "items": [scoring[128:129], scoring[129:]],
             "label_token_ids": labels, "return_token_logprobs": True, "apply_softmax": True}
    timeline = args.evidence / "feature-timeline.jsonl"

    def request(port, name, route, body):
        start = time.monotonic_ns()
        try:
            return capture(args, name, port, route, body)
        finally:
            record = {"case": name, "port": port, "begin_ns": start, "end_ns": time.monotonic_ns()}
            with timeline.open("a") as stream:
                stream.write(json.dumps(record) + "\n")

    def text(port, name, prompt, cached, *, start=None, stream=False, budget=8):
        body = native_text(prompt, budget) if start is None else score_body(prompt, start)
        body["sampling_params"]["max_new_tokens"] = budget if start is None or start == -1 else 4
        budget = body["sampling_params"]["max_new_tokens"]
        reply = request(port, name, GENERATE, body | {"stream": stream})
        if stream:
            *partial, reply = stream_frames(reply)
            assert all(frame["meta_info"]["finish_reason"] is None for frame in partial), partial
            assert all(frame["output_ids"] == reply["output_ids"][:len(frame["output_ids"])]
                       for frame in partial), partial
        return native_result(reply, prompt, budget, cached, start)

    def image_or_video(port, name, body, cached, *, stream=False):
        budget = body["max_tokens"]
        if stream:
            raw = request(port, name, CHAT, body | {"stream": True,
                                                   "stream_options": {"include_usage": True}})
            output, stats, prompt = stream_result(raw, budget, cached)
        else:
            reply = request(port, name, CHAT, body)
            output, stats = received(reply, budget, cached)
            prompt = reply["prompt_token_ids"]
        assert len(prompt) == stats["prompt_tokens"], (prompt, stats)
        return output, prompt

    with serve(args, "feature-cache-off", "--prefix-cache-entries", "0") as (port, _):
        server_policy(args, port, "cache-off", 0)
        text_controls = {len(prompt): text(port, f"off-text-{len(prompt)}", prompt, 0)
                         for prompt in (prefix, extension)}
        boundary_control = text(port, "off-prompt-score-127", scoring, 0, start=127)
        controls = {name: image_or_video(port, f"off-{name}", body, 0)
                    for name, body in media.items()}

    with serve(args, "feature-split-score", "--prefix-cache-entries", "0",
               "--max-num-batched-tokens", str(CACHE_PAGE_TOKENS)) as (port, _):
        split_control = text(port, "off-split-prompt-score-128", scoring, 0, start=128)
        label_controls = [request(port, f"off-split-labels-{index}", SCORE, score | {
            "items": [item], "label_token_ids": [labels[index]],
        }) for index, item in enumerate(score["items"])]

    with serve(args, "feature-default") as (port, log):
        server_policy(args, port, "default", 4)
        for name, prompt, hits in (("cold", prefix, 0), ("exact", prefix, len(prefix)),
                                   ("partial", extension, len(prefix))):
            assert text(port, f"default-text-{name}", prompt, hits, stream=name != "cold") == (
                text_controls[len(prompt)])
        text(port, "prime-prompt-scores", scoring[:128], 0, start=-1, budget=1)
        for score_start, hits, control in ((128, 128, split_control), (127, 0, boundary_control)):
            for stream in (False, True):
                assert text(port, f"default-prompt-score-{score_start}-{stream}", scoring, hits,
                            start=score_start, stream=stream) == control
        got = request(port, "default-labels", SCORE, score)
        assert got["token_logprobs"] == [row["token_logprobs"][0] for row in label_controls], got
        assert got["scores"] == [row["scores"][0] for row in label_controls], got
        assert got["usage"]["completion_tokens"] == 0, got
        assert got["usage"]["prompt_tokens"] == 260 and got["usage"]["total_tokens"] == 260, got
        reversed_labels = request(port, "default-labels-reversed", SCORE,
                                  score | {"label_token_ids": [row[::-1] for row in labels]})
        check_label_order(got, reversed_labels)
        feature_media(args, port, log, request, image_or_video, media, controls)
    print("Default B200 cache, decoded media, and exact score route contract passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("server", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("evidence", type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--mixed-service", action="store_true")
    modes.add_argument("--feature-complete", action="store_true")
    args = parser.parse_args()
    if args.feature_complete:
        feature_complete(args)
    elif args.mixed_service:
        mixed_service(args)
    else:
        main(args)
