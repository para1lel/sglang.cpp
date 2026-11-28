"""Capture and replay the paired output-dependent W6 requests."""

from __future__ import annotations

import gzip
import json
from hashlib import sha256
from http import HTTPStatus
from typing import TYPE_CHECKING

import cold_cache
from identity import verify_stream_end
from makespan import cohort_makespan_ns
from saved_evidence import require

from benchmarks.concurrent import client as concurrent
from benchmarks.serial import fixture
from benchmarks.streaming import client

if TYPE_CHECKING:
    from pathlib import Path
    from urllib.parse import SplitResult


TAGS = ("warmup-01", "measured-01")


def request(segment: list[int], history: list[int]) -> tuple[dict, bytes]:
    prompt = history + segment
    case = {
        "prompt_token_ids": prompt,
        "output_tokens": fixture.OUTPUT_TOKENS,
        "cached_token_range": (0, len(history)),
    }
    return case, client.json_bytes(client.request_body(case))


def capture(
    url: SplitResult, segments: list[list[int]], tag: str, engine: str, binding: dict | None
) -> dict:
    receipt = cold_cache.before_cohort(url, engine)
    requests, history = [], []
    for segment in segments:
        case, body = request(segment, history)
        record = client.measure(url, case, body)
        requests.append(record)
        if "error" in record:
            break
        history = case["prompt_token_ids"] + record["output_ids"]
    return {
        "row": fixture.ROW,
        "tag": tag,
        "binding": binding,
        "serial_protocol": fixture.NAME,
        "requests": requests,
        "cold_cache": receipt,
    }


def replay_short(record: dict, item: tuple[dict, bytes, dict]) -> tuple[int, list[list[int]], dict]:
    lane, _, case = item
    require(
        "error" not in record
        and record["http_status"] == HTTPStatus.OK
        and record["request_sha256"] == lane["request_sha256"],
        f"Invalid HTTP outcome or request identity: {record.get('error')}",
    )
    require(
        any(
            name.lower() == "content-type" and value.split(";")[0].strip() == "text/event-stream"
            for name, value in record["response_headers"]
        ),
        "Expected text/event-stream",
    )
    replay = {key: record[key] for key in ("submit_ns", "headers_ns")}
    completion = client.Completion()
    concurrent.consume_events(iter(record["events"]), replay, case, completion)
    concurrent.finish_record(replay, completion)
    verify_stream_end(record)
    require(
        replay["submit_ns"]
        <= replay["headers_ns"]
        <= replay["first_ns"]
        <= replay["last_ns"]
        <= replay["done_ns"]
        <= record["response_end"]["ns"],
        "Request timestamps are out of order",
    )
    require(
        all(
            replay[key] == record[key]
            for key in ("done_ns", "output_ids", "finish_reason", "usage")
        )
        and replay["token_chunks"] == [tuple(chunk) for chunk in record["token_chunks"]]
        and all(replay["metrics"][key] == record["metrics"][key] for key in concurrent.LATENCIES),
        "Saved stream differs from SSE replay",
    )
    duration = cohort_makespan_ns([replay])
    return (
        duration,
        [replay["output_ids"]],
        {
            "output_tokens_per_second": len(replay["output_ids"]) * 1e9 / duration,
            "request_latencies": [replay["metrics"]],
        },
    )


def replay(raw: dict, segments: list[list[int]], tag: str, binding: dict | None) -> dict:
    require(
        (raw["row"], raw["tag"], raw["binding"], raw["serial_protocol"])
        == (fixture.ROW, tag, binding, fixture.NAME),
        "W6 receipt identity mismatch",
    )
    records = raw["requests"]
    require(len(records) == len(segments), "Wrong W6 request count")
    expected = fixture.PROMPT_TOKENS if tag == TAGS[1] else (fixture.PROMPT_TOKENS[-1],)
    history, latencies, handoffs, cached = [], [], [], []
    previous_end = None
    for record, segment, prompt_tokens in zip(records, segments, expected, strict=True):
        case, body = request(segment, history)
        require(len(case["prompt_token_ids"]) == prompt_tokens, "Wrong W6 prompt geometry")
        require(
            all(
                type(record[key]) is int and record[key] > 0
                for key in ("submit_ns", "headers_ns", "done_ns")
            ),
            "Invalid W6 clock",
        )
        if previous_end is not None:
            require(previous_end <= record["submit_ns"], "Overlapping or reordered W6 streams")
            handoffs.append(record["submit_ns"] - previous_end)
        for _ in client.completion_frames(record["events"]):
            pass
        lane = {"request_sha256": sha256(body).hexdigest()}
        _, (emitted,), metrics = replay_short(record, (lane, body, case))
        history = case["prompt_token_ids"] + emitted
        previous_end = record["response_end"]["ns"]
        latencies.extend(metrics["request_latencies"])
        cached.append(record["usage"]["prompt_tokens_details"]["cached_tokens"])
    start = records[0]["submit_ns"]
    end = records[-1]["response_end"]["ns"]
    duration = end - start
    require(duration > 0, "Expected a positive W6 makespan")
    return {
        "makespan_ns": duration,
        "submit_ns": start,
        "response_end_ns": end,
        "output_tokens_per_second": len(records) * fixture.OUTPUT_TOKENS * 1e9 / duration,
        "request_latencies": latencies,
        "handoff_ns": handoffs,
        "cached_tokens": cached,
        "request_sha256": [record["request_sha256"] for record in records],
    }


def handle(
    directory: Path,
    inputs: fixture.Inputs,
    engine: str,
    binding: dict | None,
    *,
    url: SplitResult | None = None,
) -> tuple[dict, dict]:
    if url is None:
        require(
            {path.name for path in directory.glob("*.json.gz")}
            == {f"{tag}.json.gz" for tag in TAGS},
            "Unexpected W6 receipt set",
        )
    timings, cache = {}, {}
    for tag, segments in zip(TAGS, ([inputs["warmup"]], inputs["fresh"]), strict=True):
        path = directory / f"{tag}.json.gz"
        if url is None:
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                raw = json.load(stream)
        else:
            raw = capture(url, segments, tag, engine, binding)
            try:
                cold_cache.after_cohort(url, engine, raw["cold_cache"])
            finally:
                with gzip.open(path, "xt", compresslevel=3, encoding="utf-8") as stream:
                    json.dump(raw, stream, separators=(",", ":"), allow_nan=False)
        if tag == TAGS[1]:
            require(
                timings[TAGS[0]]["response_end_ns"] <= raw["cold_cache"]["reset_started_ns"],
                "W6 measured reset overlaps the independent warmup",
            )
        timings[tag] = replay(raw, segments, tag, binding)
        cache[tag] = cold_cache.replay(raw, engine, serial=(tag == TAGS[1]))
    return timings, cache
