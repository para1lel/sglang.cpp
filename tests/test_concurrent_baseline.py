import copy
import json
import statistics
import threading
from contextlib import contextmanager
from http.client import HTTPResponse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

import pytest

from benchmarks.concurrent import client
from benchmarks.concurrent.client import (
    capture_cohort,
    gate,
    summarize,
    thresholds,
    validate_cohort,
)
from benchmarks.concurrent.reference import ROWS, digest, goldens, lane_manifest, reference
from benchmarks.streaming.client import json_bytes, request_body


def wire_fixture(output_tokens=3):
    lanes = []
    for lane in range(2):
        case = {"prompt_token_ids": [100 + lane, 200], "output_tokens": output_tokens}
        body = json_bytes(request_body(case))
        lanes.append(({"lane": lane, "request_sha256": digest(body)}, body, case))
    return lanes


@contextmanager
def loopback(lanes, status_oom=False, *, cached_tokens=0):
    submitted = threading.Barrier(len(lanes), timeout=3)
    emitted = threading.Barrier(len(lanes), timeout=3)
    received = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            received.append((self.path, body))
            request = json.loads(body)
            prompt = request["prompt"]
            output = list(range(1, request["max_tokens"] + 1))
            lane = prompt[0] - 100
            submitted.wait()
            if status_oom and lane == 0:
                self.send_error(503, "CUDA out of memory")
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.end_headers()
            meta = {"id": f"lane-{lane}", "model": "benchmark", "object": "text_completion"}
            echo = {"index": 0, "token_ids": [], "prompt_token_ids": prompt,
                    "text": "", "finish_reason": None}
            first = {"index": 0, "token_ids": output[:2], "text": "é", "finish_reason": None}
            frames = b"".join(b"data: " + json_bytes(meta | {"choices": [choice]}) + b"\r\n\r\n"
                              for choice in (echo, first))
            split = frames.index("é".encode()) + 1
            self.wfile.write(frames[:split])
            self.wfile.flush()
            self.wfile.write(frames[split:])
            self.wfile.flush()
            if not status_oom:
                emitted.wait()
                threading.Event().wait(.025)
            last = {"index": 0, "token_ids": output[2:], "text": "", "finish_reason": "length"}
            usage = {"prompt_tokens": len(prompt), "completion_tokens": len(output),
                     "total_tokens": len(prompt) + len(output),
                     "prompt_tokens_details": {"cached_tokens": cached_tokens}}
            last_choice = json_bytes(meta | {"choices": [last]})
            usage_frame = json_bytes(meta | {"choices": [], "usage": usage})
            self.wfile.write(b"data: " + last_choice + b"\n\n"
                             b"data: " + usage_frame + b"\n\n"
                             b"data: [DONE]\n\n")

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(
            target=server.serve_forever, kwargs={"poll_interval": .01}, daemon=True,
        )
        thread.start()
        try:
            yield urlsplit(f"http://127.0.0.1:{server.server_port}"), received
        finally:
            server.shutdown()
            thread.join()


@pytest.fixture
def captured():
    lanes = wire_fixture()
    policy = reference()["policy"]
    with loopback(lanes) as (url, received):
        raw = capture_cohort(url, "W3", lanes, "measured-01", policy)
    expected_requests = sorted(("/v1/completions", item[1]) for item in lanes)
    assert sorted(received) == expected_requests
    return raw, lanes, [[1, 2, 3]] * 2, policy


def test_overlapping_streams_and_timing(captured):
    raw, lanes, _, policy = captured
    report = validate_cohort(raw, lanes, policy)
    assert report["submission_span_ms"] < 50
    assert report["all_streams_overlap_ms"] > 0
    assert report["all_generating_overlap_ms"] > 0
    records = raw["requests"]
    duration = (max(row["done_ns"] for row in records) - min(row["submit_ns"] for row in records))
    assert report["output_tokens_per_second"] == 6 / (duration / 1e9)
    assert report["aggregate_tokens_per_second"] == 10 / (duration / 1e9)
    for record in records:
        assert [count for _, count in record["token_chunks"]] == [2, 1]
        assert record["metrics"]["tpot_ms"] == (record["last_ns"] - record["first_ns"]) / 2e6


@pytest.fixture
def coalesced_reads(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(HTTPResponse, "read1", HTTPResponse.read)


@pytest.mark.usefixtures("coalesced_reads")
def test_prefix_cohort_replays_one_token_and_checks_cache_range() -> None:
    identity, body, case = wire_fixture(output_tokens=1)[0]
    lanes = [(identity, body, case | {"cached_token_range": (1, 2)})]
    policy = reference()["policy"]
    with loopback(lanes, cached_tokens=2) as (url, _):
        raw = capture_cohort(url, "W3", lanes, "prime", policy)
    assert {event["read"] for event in raw["requests"][0]["events"]} == {1}
    report, records = client.replay_cohort(raw, lanes, policy)
    record, = records
    assert record["output_ids"] == [1]
    assert record["usage"]["prompt_tokens_details"]["cached_tokens"] == 2
    assert report["request_latencies"][0]["tpot_ms"] is None
    assert report["all_generating_overlap_ms"] == 0
    with pytest.raises(ValueError, match="Unexpected cached prompt rows"):
        validate_cohort(raw, [(identity, body, case | {"cached_token_range": (0, 0)})], policy)


@pytest.mark.usefixtures("coalesced_reads")
def test_coalesced_reads_do_not_satisfy_multi_lane_overlap() -> None:
    lanes, policy = wire_fixture(output_tokens=1), reference()["policy"]
    with loopback(lanes) as (url, _):
        raw = capture_cohort(url, "W3", lanes, "measured-01", policy)
    for record, lane in zip(raw["requests"], lanes, strict=True):
        assert {event["read"] for event in record["events"]} == {1}
        client.replay_lane(record, lane)
    with pytest.raises(ValueError, match="overlap bounds"):
        validate_cohort(raw, lanes, policy)


def test_oom_records_failed_and_successful_requests():
    lanes, policy = wire_fixture(), reference()["policy"]
    with loopback(lanes, status_oom=True) as (url, _):
        raw = capture_cohort(url, "W3", lanes, "warmup-01", policy)
    failed, finished = raw["requests"]
    assert failed["http_status"] == 503 and "out of memory" in failed["http_error_body"]
    assert finished["output_ids"] == [1, 2, 3] and finished["finish_reason"] == "length"
    with pytest.raises(ValueError, match="HTTP 503"):
        validate_cohort(raw, lanes, policy)


def mutate_payload(record, event, field, value):
    payload = json.loads(record["events"][event]["lines"][0].removeprefix("data: "))
    target = payload if field == "id" else payload["usage"] if event == 3 else payload["choices"][0]
    target[field] = value
    record["events"][event]["lines"][0] = "data: " + json_bytes(payload).decode()


@pytest.mark.parametrize(("change", "message"), [
    ((0, "prompt_token_ids", [999]), "Prompt IDs"),
    ((1, "id", "different"), "Completion ID changed"),
    ((2, "stop_reason", 13), "stop reason"),
    ((2, "finish_reason", "stop"), "Expected length completion"),
    ((2, "token_ids", []), "Expected length completion"),
    ((3, "prompt_tokens_details", {"cached_tokens": 1}), "zero reused"),
    ((3, "total_tokens", 6), "Usage total"),
])
def test_wire_rejects_changed_transcript(captured, change, message):
    raw, lanes, _, policy = captured
    mutate_payload(raw["requests"][0], *change)
    with pytest.raises(ValueError, match=message):
        validate_cohort(raw, lanes, policy)


def test_rejects_missing_done_and_submission_skew(captured):
    raw, lanes, _, policy = captured
    lost_done = copy.deepcopy(raw)
    lost_done["requests"][0]["events"].pop()
    with pytest.raises(ValueError, match=r"before \[DONE\]"):
        validate_cohort(lost_done, lanes, policy)
    with pytest.raises(ValueError, match="overlap bounds"):
        validate_cohort(raw, lanes, policy | {"max_submission_span_ms": -1})


def test_cross_engine_wire_accepts_different_ids_but_same_engine_rejects(captured):
    raw, lanes, expected, policy = captured
    original = validate_cohort(raw, lanes, policy)
    first = raw["requests"][0]
    mutate_payload(first, 1, "token_ids", [1, 99])
    first["output_ids"][1] = 99
    assert validate_cohort(raw, lanes, policy) == original
    with pytest.raises(ValueError, match="output 1 differs from warmup"):
        client.verify_round_ids(raw, expected)


def test_wire_accepts_optional_openai_envelope(captured):
    raw, lanes, _, policy = captured
    for record in raw["requests"]:
        for event in record["events"][:-1]:
            payload = json.loads(event["lines"][0].removeprefix("data: "))
            for key in ("id", "model", "object"):
                del payload[key]
            if "usage" in payload:
                del payload["usage"]["total_tokens"]
                del record["usage"]["total_tokens"]
            event["lines"][0] = "data: " + json_bytes(payload).decode()
    assert validate_cohort(raw, lanes, policy)["all_generating_overlap_ms"] > 0


def test_rate_margin_and_independent_latency_budgets():
    measured = [
        {"output_tokens_per_second": rate, "requests_per_second": rate / 10,
         "aggregate_tokens_per_second": rate * 2,
         "request_latencies": [{"ttft_ms": first, "tpot_ms": 1, "e2e_ms": 101}]}
        for rate, first in zip([90, 100, 100, 100, 110], [0, 100, 100, 100, 100])
    ]
    stats = summarize(measured)
    limits = thresholds(stats)
    assert limits["strict_median_output_tokens_per_second_gt"] == (
        100 * (1 + 2 * statistics.stdev([90, 100, 100, 100, 110]) / 100)
    )
    caps = limits["p95_latency_ceilings_ms"]
    assert caps["ttft_ms"] == 100 + 2 * statistics.stdev([0, 100, 100, 100, 100])
    assert caps["e2e_ms"] == 102.01
    stats["latency_ms"]["ttft_ms"]["p95"] = caps["ttft_ms"]
    stats["throughput"]["output_tokens_per_second"]["p50"] = (
        limits["strict_median_output_tokens_per_second_gt"]
    )
    assert gate(stats, limits) == {
        "throughput": False, "latency": {"ttft_ms": True, "tpot_ms": True, "e2e_ms": True},
    }


def test_portable_goldens_and_condition_identity():
    baseline, manifest = reference(), lane_manifest()
    assert sum(row["measured_requests"] for row in baseline["rows"].values()) == 280
    assert sum(row["measured_output_ids"] for row in baseline["rows"].values()) == 778240
    assert baseline["tokenizer_sha256"] == manifest["tokenizer_sha256"]
    for name, (_, output, concurrency) in ROWS.items():
        ids = goldens(name)
        assert len(ids) == concurrency and all(len(lane) == output for lane in ids)
        row = baseline["rows"][name]
        assert row["thresholds"]["strict_throughput_ratio_gt"] == 1.01
        assert row["observed_cached_prompt_tokens"] == 0


def test_current_concurrent_runs_each_round_once(tmp_path, monkeypatch):
    previous = client.reference()
    previous["rows"]["W3"]["thresholds"]["strict_median_output_tokens_per_second_gt"] = 1e12
    monkeypatch.setattr(client, "reference", lambda: previous)
    lanes = wire_fixture()
    monkeypatch.setattr(client, "load_lanes", lambda *_: lanes)
    output = tmp_path / "capture"
    with loopback(lanes) as (url, received):
        assert client.run("W3", url.geturl(), tmp_path, output) == 1
        assert len(received) == 2 * len(lanes)
    report = json.loads((output / "summary.json").read_bytes())
    assert report["protocol"] == client.SINGLE_ROUND_PROTOCOL
    assert [outcome["tag"] for outcome in report["request_outcomes"]] == [
        "warmup-01", "measured-01"]
    assert report["throughput"]["output_tokens_per_second"]["sample_stdev"] is None
    assert report["wire_and_timing_gates"]["throughput"] is False
