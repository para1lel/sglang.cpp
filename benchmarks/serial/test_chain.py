from __future__ import annotations

import gzip
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

import cold_cache
import pytest
from makespan import K4_MAKESPAN_CRITERION, compare

from benchmarks.serial import chain, fixture

if TYPE_CHECKING:
    from pathlib import Path


MEASURED_REQUESTS = 20
TOTAL_REQUESTS = MEASURED_REQUESTS + 1


def metric_names(engine: str) -> dict[str, int]:
    common = {cold_cache.RUNNING: 0, cold_cache.WAITING: 0, cold_cache.HITS: 0}
    if engine == "vllm":
        return common
    return (
        common
        | dict.fromkeys(cold_cache.LMCACHE_POOL_CAPACITIES, 4)
        | dict.fromkeys(
            (
                cold_cache.ENTRIES,
                cold_cache.PAGES,
                cold_cache.PENDING_STORES,
                *cold_cache.LMCACHE_REUSE,
                *cold_cache.LMCACHE_READ_ACTIVITY,
                *cold_cache.LMCACHE_RANK_READS,
            ),
            0,
        )
    )


def sse(prompt: list[int], output: list[int], previous: int) -> bytes:
    payloads = [
        {
            "choices": [
                {
                    "index": 0,
                    "token_ids": output,
                    "prompt_token_ids": prompt,
                    "text": "unrelated decoded text",
                    "finish_reason": "length",
                }
            ]
        },
        {
            "choices": [],
            "usage": {
                "prompt_tokens": len(prompt),
                "completion_tokens": len(output),
                "total_tokens": len(prompt) + len(output),
                "prompt_tokens_details": {"cached_tokens": previous},
            },
        },
    ]
    frames = [f"data: {json.dumps(payload)}\n\n" for payload in payloads]
    return "".join([*frames, "data: [DONE]\n\n"]).encode()


def serve_chain(
    address: tuple[str, int],
    engine: str,
    output_base: int,
    requests: list[tuple[list[int], list[int]]],
    resets: list[int],
) -> HTTPServer:
    metrics = metric_names(engine)

    class Handler(BaseHTTPRequestHandler):
        def send(self, body: bytes, content_type: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            assert self.path == "/metrics"
            self.send(
                "".join(f"{name} {value}\n" for name, value in metrics.items()).encode(),
                "text/plain",
            )

        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path == "/reset_prefix_cache":
                resets.append(len(requests))
                if engine != "vllm":
                    metrics[cold_cache.ENTRIES] = metrics[cold_cache.PAGES] = 0
                self.send(b'{"success":true}', "application/json")
                return
            assert self.path == "/v1/completions"
            assert body["return_token_ids"] is True
            assert body["add_special_tokens"] is False
            prompt = body["prompt"]
            output = [output_base + len(requests)] * body["max_tokens"]
            previous = 0 if resets[-1] == len(requests) else len(requests[-1][0])
            metrics[cold_cache.HITS] += previous
            if engine != "vllm":
                metrics[cold_cache.ENTRIES] = metrics[cold_cache.PAGES] = 1
            requests.append((prompt, output))
            self.send(sse(prompt, output, previous), "text/event-stream")

    return HTTPServer(address, Handler)


def test_w6_pinned_geometry() -> None:
    observed = (fixture.K, fixture.OUTPUT_TOKENS, fixture.FRESH_TOKENS, fixture.MAX_CONTEXT)
    assert observed == (1024, 1024, 9216, 204800)
    observed_prompts = fixture.PROMPT_TOKENS
    assert observed_prompts == tuple(range(9216, 203777, 10240))
    assert fixture.metadata()["total_requests"] == TOTAL_REQUESTS


@pytest.mark.parametrize(
    ("engine", "output_base"),
    [("vllm", 1000), (cold_cache.LMCACHE_POOL_TP2, 2000)],
)
def test_w6_wire_and_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, engine: str, output_base: int
) -> None:
    monkeypatch.setattr(fixture, "OUTPUT_TOKENS", 2)
    monkeypatch.setattr(fixture, "PROMPT_TOKENS", tuple(range(3, 103, 5)))
    inputs: fixture.Inputs = {
        "warmup": [100] * 98,
        "fresh": [[turn] * 3 for turn in range(MEASURED_REQUESTS)],
    }
    requests, resets = [], []
    with serve_chain(("127.0.0.1", 0), engine, output_base, requests, resets) as server:
        thread = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
        thread.start()
        try:
            endpoint = urlsplit(f"http://127.0.0.1:{server.server_port}")
            captured = chain.handle(tmp_path, inputs, engine, {}, url=endpoint)
        finally:
            server.shutdown()
            thread.join()
    assert resets == [0, 1]
    assert len(requests) == TOTAL_REQUESTS
    assert requests[0][0] == inputs["warmup"]
    expected = []
    for (prompt, output), segment in zip(requests[1:], inputs["fresh"], strict=True):
        expected += segment
        assert prompt == expected
        expected += output
    assert captured == chain.handle(tmp_path, inputs, engine, {})
    report = captured[0]["measured-01"]
    assert report["makespan_ns"] == report["response_end_ns"] - report["submit_ns"]
    assert len(report["handoff_ns"]) == MEASURED_REQUESTS - 1
    with gzip.open(tmp_path / "measured-01.json.gz", "rt", encoding="utf-8") as stream:
        raw = json.load(stream)
    raw["requests"][1]["request_sha256"] = raw["requests"][0]["request_sha256"]
    with pytest.raises(ValueError, match="request identity"):
        chain.replay(raw, inputs["fresh"], "measured-01", {})


def test_pool_cold_receipt_allows_no_backups_and_rejects_rank_restore() -> None:
    engine = cold_cache.LMCACHE_POOL_TP2
    samples = {stage: metric_names(engine) for stage in ("before", "cleared", "after")}
    samples["after"][cold_cache.ENTRIES] = 1
    cold_cache.verify_pool(samples, engine, serial=False)
    samples["after"][cold_cache.LMCACHE_RANK_READS[0]] = 1
    with pytest.raises(ValueError, match="restored state during a cold cohort"):
        cold_cache.verify_pool(samples, engine, serial=False)
    cold_cache.verify_pool(samples, engine, serial=True)


def test_serial_ratio_gate_accepts_exact_boundary_and_rejects_overage() -> None:
    for native, passed in ((110, True), (111, False)):
        result = compare(
            {"W6": [native]}, {"W6": [100]}, eligible=True, criterion=K4_MAKESPAN_CRITERION
        )
        assert result["passed"] is passed
