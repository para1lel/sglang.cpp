"""Check live native prefix controls with two active slots and two cache entries."""

import argparse
from contextlib import ExitStack, closing
import http.client
import json
from pathlib import Path
import socket
import struct
import time

from routes import FIXTURE, HEADERS, MODEL, Routes, benchmark


RUNNING = "vllm:num_requests_running"
WAITING = "vllm:num_requests_waiting"
CACHED = "vllm:prefix_cache_hits_total"
COMPUTED = "vllm:request_prefill_kv_computed_tokens_sum"
PREFILLS = "vllm:request_prefill_kv_computed_tokens_count"
PROMPT = "vllm:prompt_tokens_total"
GENERATED = "vllm:generation_tokens_total"
EXTERNAL = "vllm:external_prefix_cache_hits_total"
ENTRIES = "sglang_cpp:prefix_cache_entries"
PAGES = "sglang_cpp:prefix_cache_pages"
OCCUPANCY = "vllm:kv_cache_usage_perc"
DEVICE_BYTES = "sglang_cpp:prefix_cache_device_bytes"
HITS = "sglang_cpp:prefix_cache_hits_total"
STORES = "sglang_cpp:prefix_cache_stores_total"
EVICTIONS = "sglang_cpp:prefix_cache_evictions_total"


class Controls(Routes):
    def control(self, name, route, body=None, status=200):
        with closing(http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)) as client:
            client.request("GET" if body is None else "POST", route, body, HEADERS)
            with client.getresponse() as response:
                raw = response.read()
                observed = response.status
                content = response.headers.get_content_type()
        (self.directory / f"{name}.http").write_bytes(
            f"{observed} {content}\n".encode() + raw
        )
        assert observed == status, (name, observed, raw)
        expected = "text/plain" if route == "/metrics" else "application/json"
        assert content == expected, (name, content)
        return raw

    def metrics(self, name):
        raw = self.control(name, "/metrics")
        assert raw.startswith(b"# Native sglang.cpp metrics"), raw
        metrics = {}
        for line in raw.decode().splitlines()[1:]:
            key, separator, value = line.partition('{engine="0"} ')
            assert separator and key not in metrics, line
            metrics[key] = float(value)
        return metrics

    def wait_for(self, name, running, waiting, generated=0):
        deadline = time.monotonic() + 15
        while True:
            metrics = self.metrics(name)
            if ((metrics[RUNNING], metrics[WAITING]) == (running, waiting)
                    and metrics[GENERATED] >= generated):
                return metrics
            if time.monotonic() >= deadline:
                raise AssertionError((name, metrics))
            time.sleep(0.02)

    def reset(self, name, status=200):
        result = self.control(name, "/reset_prefix_cache", b"{}", status)
        if status == 200:
            assert json.loads(result) == {"success": True}, result

    def start(self, name, limit=64, status=200):
        body = json.dumps({"profile_prefix": name, "max_iterations": limit,
                           "profiler_kwargs": {"with_stack": False}}).encode()
        result = self.control(f"{name}-start", "/start_profile", body, status)
        if status == 200:
            assert json.loads(result) == {"success": True, "profiler": "native-batch-steps"}

    def stop(self, name):
        response = json.loads(self.control(f"{name}-stop", "/stop_profile", b"{}"))
        assert response["success"] is True and response["profiler"] == "native-batch-steps"
        assert response["profile_prefix"] == name, response
        for step in response["steps"]:
            assert set(step) == {"elapsed_ns", "width", "prefill_rows", "decode_lanes",
                                 "graph_slots"}, step
            assert all(type(value) is int and value >= 0 for value in step.values()), step
            assert 0 < step["width"] <= 2 and step["decode_lanes"] <= step["width"]
            assert step["elapsed_ns"] > 0 and step["graph_slots"] < 4, step
            assert step["graph_slots"].bit_count() <= step["decode_lanes"], step
        return response["steps"]

    def open_stream(self, name, prompt, salt, count=4096):
        client = http.client.HTTPConnection("127.0.0.1", self.port, timeout=90)
        body = json.dumps(benchmark(prompt, count, cache_salt=salt)).encode()
        (self.directory / f"{name}.request.json").write_bytes(body)
        try:
            client.request("POST", "/v1/completions", body, HEADERS)
            client.sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
        except BaseException:
            client.close()
            raise
        return client


def output_ids(controls, name, raw, prompt, count, cached):
    frames = raw.rstrip(b"\n").split(b"\n\n")
    assert frames.pop() == b"data: [DONE]", name
    messages = [json.loads(frame.removeprefix(b"data: ")) for frame in frames]
    controls.usage(name, messages.pop()["usage"], len(prompt), count, cached)
    terminal, = messages.pop()["choices"]
    assert terminal["finish_reason"] == "length" and terminal["token_ids"] == [], terminal
    ids = [token for message in messages for token in message["choices"][0]["token_ids"]]
    assert len(ids) == count, (name, ids)
    return ids


def verify_step(controls, prompt, output, name, cached, computed):
    before = controls.wait_for(f"{name}-before", 0, 0)
    controls.start(name)
    controls.benchmark(name, prompt, output, cached, "branch")
    after = controls.wait_for(f"{name}-after", 0, 0)
    steps = controls.stop(name)
    assert after[CACHED] - before[CACHED] == cached, (name, before, after)
    assert after[COMPUTED] - before[COMPUTED] == computed, (name, before, after)
    assert after[PREFILLS] - before[PREFILLS] == 1, (name, before, after)
    assert after[PROMPT] - before[PROMPT] == len(prompt), (name, before, after)
    assert after[GENERATED] - before[GENERATED] == len(output), (name, before, after)
    assert after[EXTERNAL] == before[EXTERNAL] == 0, (name, before, after)
    assert sum(step["prefill_rows"] for step in steps) == computed, (name, steps)


def check_profile_controls(controls, prompt, output):
    controls.control("bad-profile", "/start_profile", b'{"profile_prefix":"invalid"}', 400)
    controls.start("too-many-iterations", 65, 400)
    controls.control("bad-stop", "/stop_profile", b'{"unexpected":true}', 400)
    controls.control("stop-without-profile", "/stop_profile", b"{}", 409)
    controls.start("bounded", 2)
    controls.start("duplicate", 2, 409)
    controls.benchmark("bounded-generation", prompt, output, len(prompt), "branch")
    controls.start("at-profile-limit", 2, 409)
    steps = controls.stop("bounded")
    assert len(steps) == 2, steps


def check_live_graph(controls):
    controls.start("live-graph", 4)
    before = controls.metrics("live-graph-before")
    controls.wait_for("live-graph-progress", 2, 2, before[GENERATED] + 32)
    steps = controls.stop("live-graph")
    assert len(steps) == 4, steps
    assert all(step["width"] == 2 and step["decode_lanes"] == 2
               and step["prefill_rows"] == 0 for step in steps), steps
    assert all(step["graph_slots"] == 0 for step in steps), steps


def live_reset(controls, prompt, output):
    with ExitStack() as streams:
        with ExitStack() as active:
            for index in (1, 2):
                client = active.enter_context(closing(
                    controls.open_stream(f"live-{index}", prompt, "branch")
                ))
                response = active.enter_context(client.getresponse())
                assert response.status == 200
                line = response.readline()
                (controls.directory / f"live-{index}.first-event").write_bytes(line)
                choice, = json.loads(line.removeprefix(b"data: "))["choices"]
                assert choice["token_ids"] == output[:1], choice
            controls.wait_for("active", 2, 0)
            queued = streams.enter_context(closing(
                controls.open_stream("queued", prompt, "queued", len(output))
            ))
            survivor = streams.enter_context(closing(
                controls.open_stream("survivor", prompt, "branch", len(output))
            ))
            busy = controls.wait_for("busy", 2, 2)
            assert all(busy[key] > 0 for key in (ENTRIES, PAGES, OCCUPANCY, DEVICE_BYTES)), busy
            controls.control("live-health", "/health")
            controls.reset("busy-reset", 409)
            controls.raw("queue-full", benchmark(prompt, 1), 503)
            check_live_graph(controls)
            queued.close()
            controls.wait_for("queued-cancelled", 2, 1)
        with survivor.getresponse() as response:
            raw = response.read()
            (controls.directory / "survivor.sse").write_bytes(raw)
            assert response.status == 200
            assert response.headers.get_content_type() == "text/event-stream"
        assert output_ids(controls, "survivor", raw, prompt, len(output), len(prompt)) == output
    controls.wait_for("retired", 0, 0)
    controls.benchmark("after-cancel", prompt, output, len(prompt), "branch")
    controls.benchmark("cancelled-queue-miss", prompt, output, 0, "queued")
    before = controls.wait_for("before-reset", 0, 0)
    controls.reset("idle-reset")
    after = controls.metrics("after-reset")
    assert after[ENTRIES] == after[PAGES] == after[OCCUPANCY] == 0, after
    assert after[DEVICE_BYTES] == before[DEVICE_BYTES] > 0, (before, after)
    for counter in (CACHED, COMPUTED, PREFILLS, PROMPT, GENERATED,
                    EXTERNAL, HITS, STORES, EVICTIONS):
        assert before[counter] == after[counter], (counter, before, after)
    verify_step(controls, prompt, output, "after-reset-miss", 0, len(prompt))


def check_wrong_prefix_and_eviction(controls, prefix, prefix_output):
    wrong = [*prefix[:64], (prefix[64] + 1) % 248320, *prefix[65:]]
    before = controls.metrics("wrong-before")
    raw, content = controls.raw(
        "wrong-prefix", benchmark(wrong, len(prefix_output), cache_salt="branch")
    )
    assert content == "text/event-stream"
    wrong_output = output_ids(controls, "wrong-prefix", raw, wrong, len(prefix_output), 0)
    after = controls.wait_for("wrong-after", 0, 0)
    assert after[CACHED] == before[CACHED] and after[COMPUTED] - before[COMPUTED] == 128
    controls.benchmark("wrong-now-warm", wrong, wrong_output, 128, "branch")
    controls.benchmark("another-salt", prefix, prefix_output, 0, "other")
    after = controls.wait_for("evicted", 0, 0)
    assert after[ENTRIES] == 2 and after[EVICTIONS] == before[EVICTIONS] + 1, after
    verify_step(controls, prefix, prefix_output, "evicted-miss", 0, 128)


def run(controls):
    fixture = json.loads(FIXTURE.read_text())
    prefix = fixture["prompt"][:128]
    prefix_output = fixture["output"]["128"]
    suffix = fixture["prompt"][:129]
    suffix_output = fixture["output"]["129"]
    check_health(controls)
    for index, body in enumerate((b"[]", b"{", b'{"unexpected":true}')):
        controls.control(f"bad-reset-{index}", "/reset_prefix_cache", body, 400)
    controls.control("reset-with-get", "/reset_prefix_cache", status=404)
    controls.reset("initial-reset")
    verify_step(controls, prefix, prefix_output, "cold", 0, 128)
    verify_step(controls, prefix, prefix_output, "exact", 128, 0)
    verify_step(controls, suffix, suffix_output, "partial", 128, 1)

    check_profile_controls(controls, prefix, prefix_output)
    check_wrong_prefix_and_eviction(controls, prefix, prefix_output)
    live_reset(controls, prefix, prefix_output)
    check_text_stops(controls)


def check_health(controls):
    health = json.loads(controls.control("health", "/health"))
    assert isinstance(health, dict), health
    if health == {}:
        controls.control("legacy-ready", "/ready", status=404)
        return
    active = health.get("max_active")
    tokens = health.get("max_num_batched_tokens")
    assert type(active) is int and active == 2, health
    assert type(tokens) is int and 1 <= tokens <= 32768, health
    assert json.loads(controls.control("ready", "/ready")) == {"ready": True}


def check_text_reply(controls, name, request, expected_text, ids, reason, cached):
    before = controls.wait_for(f"{name}-before", 0, 0)
    raw, content = controls.raw(name, request)
    after = controls.wait_for(f"{name}-after", 0, 0)
    assert after[CACHED] - before[CACHED] == cached, (name, before, after)
    assert after[COMPUTED] - before[COMPUTED] == 161 - cached, (name, before, after)
    if request.get("stream"):
        assert content == "text/event-stream", name
        frames = raw.split(b"\n\n")
        assert frames[-2:] == [b"data: [DONE]", b""], name
        messages = [json.loads(frame.removeprefix(b"data: ")) for frame in frames[:-2]]
        usage = messages.pop()
        assert usage["choices"] == [], name
        terminal, = messages.pop()["choices"]
        assert terminal["finish_reason"] == reason and terminal["token_ids"] == [], name
        choices = [message["choices"][0] for message in messages]
        assert [choice["token_ids"] for choice in choices] == [[token] for token in ids], name
        assert all(choice["finish_reason"] is None for choice in choices), name
        text = "".join(choice["text"] for choice in choices) + terminal["text"]
        assert text == expected_text, name
    else:
        assert content == "application/json", name
        usage = json.loads(raw)
        choice, = usage["choices"]
        assert (choice["text"], choice["token_ids"], choice["finish_reason"]) == (
            expected_text, ids, reason), (name, choice)
    assert usage["usage"]["total_tokens"] == 161 + len(ids), name
    return usage["usage"]


def check_text_stops(controls):
    # The exact-prefill continuation is " seven, eight,". Both stops trigger on its first comma.
    prompt = ("Prefix cache fixture. " * 36
              + "\n\nThe sequence starts at one, two, three, four, five, six,")
    body = {"model": MODEL, "prompt": prompt, "max_tokens": 4, "return_token_ids": True,
            "temperature": 0, "seed": 0, "cache_salt": "warm-stop-repro"}
    stream = {"stream": True, "stream_options": {"include_usage": True}}
    string_stop = {"stop": [","]}
    token_stop = {"stop_token_ids": [11]}
    stopped_ids = [7840, 11]
    full_ids = [*stopped_ids, 7810, 11]
    cases = (
        ("cold-string-json", string_stop, " seven", stopped_ids, "stop", 0),
        ("warm-no-stop-json", {}, " seven, eight,", full_ids, "length", 128),
        ("warm-token-json", token_stop, " seven", stopped_ids, "stop", 128),
        ("warm-string-json", string_stop, " seven", stopped_ids, "stop", 128),
        ("warm-token-sse", stream | token_stop, " seven", stopped_ids, "stop", 128),
        ("warm-string-sse", stream | string_stop, " seven", stopped_ids, "stop", 128),
        ("warm-after-stop-sse", stream, " seven, eight,", full_ids, "length", 128),
    )
    usages = [check_text_reply(controls, name, body | options, text, ids, reason, cached)
              for name, options, text, ids, reason, cached in cases]
    for (name, _, _, ids, _, cached), usage in zip(cases, usages, strict=True):
        controls.usage(name, usage, 161, len(ids), cached)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    args.evidence.mkdir()
    run(Controls(args.port, args.evidence))
    (args.evidence / "passed.json").write_text('{"status":"passed"}\n')


if __name__ == "__main__":
    main()
