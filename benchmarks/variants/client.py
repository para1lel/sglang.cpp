"""Capture W3 feature variants against a local text-completion service."""

import argparse
import gzip
import http.client
import json
import socket
import threading
from concurrent.futures import ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter_ns, time_ns
from urllib.parse import urlsplit

from benchmarks.concurrent.client import consume_events, summarize
from benchmarks.concurrent.reference import digest, goldens
from benchmarks.concurrent.replay import verify_greedy_ids
from benchmarks.streaming.client import Completion, json_bytes, sse_frames
from benchmarks.variants.fixtures import POLICY, ROWS, load_requests


class VariantCompletion(Completion):
    def validate(self, case):
        usage = self.usage or {}
        cached = usage.get("prompt_tokens_details", {}).get("cached_tokens")
        if self.prompt != case["prompt_token_ids"]:
            raise ValueError("Prompt IDs absent or changed")
        if len(self.ids) != case["output_tokens"] or self.finish != "length":
            raise ValueError(f"Expected length stop, got {len(self.ids)} IDs and {self.finish!r}")
        if (usage.get("prompt_tokens"), usage.get("completion_tokens"), cached) != (
            len(case["prompt_token_ids"]), case["output_tokens"], case["cached_tokens"]
        ):
            raise ValueError(f"Usage or reused prompt tokens differ from fixture: {usage}")


def control(url, route, body=None, timeout=30):
    connection = http.client.HTTPConnection(url.hostname, url.port, timeout=timeout)
    try:
        connection.request("GET" if body is None else "POST", f"/{route}", body,
                           {"Content-Type": "application/json", "Connection": "close"})
        with connection.getresponse() as response:
            return {"status": response.status, "body": response.read().decode("utf-8"),
                    "ack_ns": perf_counter_ns()}
    finally:
        connection.close()


def require_http(result):
    if result["status"] != 200:
        raise ValueError(f"Control request failed: {result}")
    return result


def capture_lane(item, url, barrier, sockets, ready):
    lane, body, _ = item
    number = lane["lane"]
    result = {"lane": number, "request_sha256": lane["request_sha256"], "events": []}
    connection = http.client.HTTPConnection(
        url.hostname, url.port, timeout=POLICY["socket_timeout_seconds"]
    )
    try:
        connection.connect()
        sockets[number] = connection.sock
        result["connect_ns"] = perf_counter_ns()
        barrier.wait(POLICY["barrier_timeout_seconds"])
        result["submit_ns"] = perf_counter_ns()
        connection.request("POST", "/v1/completions", body, {
            "Content-Type": "application/json", "Accept": "text/event-stream",
            "Connection": "close",
        })
        result["sent_ns"] = perf_counter_ns()
        with connection.getresponse() as response:
            result["headers_ns"] = perf_counter_ns()
            result["http_status"] = response.status
            result["response_headers"] = response.getheaders()
            if response.status != 200 or response.headers.get_content_type() != "text/event-stream":
                result["http_error_body"] = response.read().decode("utf-8", errors="replace")
                raise ValueError(f"Unexpected streaming HTTP {response.status}")
            for event in sse_frames(response):
                result["events"].append(event)
                if ready is not None and not ready.is_set():
                    data = "\n".join(line[5:].removeprefix(" ") for line in event["lines"]
                                     if line.startswith("data:"))
                    if data and data != "[DONE]":
                        choices = json.loads(data).get("choices", [])
                        if choices and choices[0].get("token_ids"):
                            ready.set()
            result["eof_ns"] = perf_counter_ns()
    except (OSError, http.client.HTTPException, ValueError, KeyError,
            TypeError, threading.BrokenBarrierError) as error:
        if "submit_ns" not in result:
            barrier.abort()
        result["failure_ns"] = perf_counter_ns()
        result["error"] = f"{type(error).__name__}: {error}"
    finally:
        connection.close()
    return result


def start_profile(url, tag, phase, directory):
    data = {"profile_prefix": f"{tag}-{phase}", "max_iterations": 4,
            "profiler_kwargs": {"with_stack": False}}
    result = control(url, "start_profile", json_bytes(data))
    write_json(directory / f"{phase}-profile-start.json", result)
    return require_http(result)


def capture_cohort(url, lanes, tag, directory, profile=False):
    started_utc = datetime.now(timezone.utc).isoformat()
    monotonic, epoch = perf_counter_ns(), time_ns()
    deadline = monotonic + POLICY["cohort_deadline_seconds"] * 10**9
    released, sockets = [], [None] * len(lanes)
    barrier = threading.Barrier(len(lanes), action=lambda: released.append(perf_counter_ns()))
    ready = [threading.Event() if profile else None for _ in lanes]
    instrumentation = {}
    if profile:
        instrumentation["prefill"] = start_profile(url, tag, "prefill", directory)
    with ThreadPoolExecutor(len(lanes)) as pool:
        jobs = [pool.submit(capture_lane, lane, url, barrier, sockets, signal)
                for lane, signal in zip(lanes, ready, strict=True)]
        if profile:
            try:
                for signal in ready:
                    if not signal.wait(max(0, deadline - perf_counter_ns()) / 1e9):
                        raise ValueError("All lanes did not begin generating before the deadline")
                instrumentation["all_lanes_emitting_ns"] = perf_counter_ns()
                instrumentation["prefill_stop"] = require_http(control(url, "stop_profile", b"{}"))
                instrumentation["decode"] = start_profile(url, tag, "decode", directory)
            except (OSError, ValueError, http.client.HTTPException) as error:
                instrumentation["error"] = str(error)
        _, pending = wait(jobs, timeout=max(0, deadline - perf_counter_ns()) / 1e9)
        if pending:
            instrumentation["deadline_exceeded"] = True
            barrier.abort()
            for sock in sockets:
                if sock is not None:
                    try:
                        sock.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
        requests = [job.result() for job in jobs]
    if profile:
        try:
            instrumentation["decode_stop"] = require_http(control(url, "stop_profile", b"{}"))
        except (OSError, ValueError, http.client.HTTPException) as error:
            instrumentation["stop_error"] = str(error)
    return {"tag": tag, "started_utc": started_utc,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "clock_reference": {"perf_counter_ns": monotonic, "time_ns": epoch},
            "barrier_ns": released[0] if released else None,
            "instrumentation": instrumentation, "requests": requests}


def replay_lane(raw, item):
    lane, _, case = item
    if ("error" in raw or raw["http_status"] != 200
            or raw["lane"] != lane["lane"] or raw["request_sha256"] != lane["request_sha256"]):
        raise ValueError(f'Invalid request outcome for lane {lane["lane"]}: {raw.get("error")}')
    if not any(key.lower() == "content-type" and value.split(";")[0].strip() == "text/event-stream"
               for key, value in raw["response_headers"]):
        raise ValueError("Expected text/event-stream")
    completion = VariantCompletion()
    record = {key: raw[key] for key in ("connect_ns", "submit_ns", "sent_ns",
                                        "headers_ns", "eof_ns")}
    consume_events(raw["events"], record, case, completion)
    first, last = completion.token_chunks[0][0], completion.token_chunks[-1][0]
    if not (record["connect_ns"] <= record["submit_ns"] <= record["sent_ns"]
            <= record["headers_ns"] <= first <= last <= record["done_ns"] <= record["eof_ns"]):
        raise ValueError("Invalid request clock order")
    record.update(first_ns=first, last_ns=last, output_ids=completion.ids,
                  finish_reason=completion.finish, usage=completion.usage,
                  token_chunks=completion.token_chunks)
    record["metrics"] = {
        "ttft_ms": (first - record["submit_ns"]) / 1e6,
        "tpot_ms": (last - first) / (len(completion.ids) - 1) / 1e6
        if len(completion.ids) > 1 else None,
        "e2e_ms": (record["done_ns"] - record["submit_ns"]) / 1e6,
    }
    return record


def replay_cohort(raw, lanes):
    if any(key in raw["instrumentation"] for key in ("error", "stop_error", "deadline_exceeded")):
        raise ValueError(f"Invalid cohort instrumentation: {raw['instrumentation']}")
    records = [replay_lane(request, lane)
               for request, lane in zip(raw["requests"], lanes, strict=True)]
    stream_ids = [record["stream_id"] for record in records if record["stream_id"] is not None]
    if len(stream_ids) != len(set(stream_ids)):
        raise ValueError("Duplicate completion IDs across lanes")
    earliest_submit = min(record["submit_ns"] for record in records)
    submit_span = (max(record["submit_ns"] for record in records) - earliest_submit) / 1e6
    max_headers = max(record["headers_ns"] for record in records)
    max_first = max(record["first_ns"] for record in records)
    earliest_done = min(record["done_ns"] for record in records)
    latest_done = max(record["done_ns"] for record in records)
    if (type(raw["barrier_ns"]) is not int
            or not max(r["connect_ns"] for r in records) <= raw["barrier_ns"] <= earliest_submit
            or not max(r["sent_ns"] for r in records) < min(r["first_ns"] for r in records)
            or max(max_headers, max_first) >= earliest_done
            or submit_span > POLICY["max_submission_span_ms"]):
        raise ValueError("Cohort submissions or open streams failed overlap bounds")
    lifecycle = raw["instrumentation"]
    if lifecycle and not (
        lifecycle["prefill"]["ack_ns"] < earliest_submit <= max_first
        <= lifecycle["all_lanes_emitting_ns"] <= lifecycle["prefill_stop"]["ack_ns"]
        <= lifecycle["decode"]["ack_ns"] < earliest_done
        <= latest_done <= lifecycle["decode_stop"]["ack_ns"]
    ):
        raise ValueError("Diagnostic profiler acknowledgements are outside the stream windows")
    denominator = (latest_done - earliest_submit) / 1e9
    output = sum(len(record["output_ids"]) for record in records)
    prompt = sum(record["usage"]["prompt_tokens"] for record in records)
    return {"output_tokens_per_second": output / denominator,
            "requests_per_second": len(records) / denominator,
            "aggregate_tokens_per_second": (prompt + output) / denominator,
            "submission_span_ms": submit_span,
            "all_streams_overlap_ms": (earliest_done - max_headers) / 1e6,
            "all_generating_overlap_ms": (earliest_done - max_first) / 1e6,
            "request_latencies": [record["metrics"] for record in records],
            "requests": records}


def write_json(path, data):
    path.write_bytes(json_bytes(data) + b"\n")


def verify_same_engine_greedy(report, expected):
    requests = [{"lane": lane, "output_ids": record["output_ids"]}
                for lane, record in enumerate(report["requests"])]
    verify_greedy_ids(requests, expected)


def capture(url, lanes, tag, directory, same_engine_greedy=None, profile=False):
    raw = capture_cohort(url, lanes, tag, directory, profile)
    with gzip.open(directory / f"{tag}.json.gz", "wt", compresslevel=3) as output:
        json.dump(raw, output, separators=(",", ":"), allow_nan=False)
    report = replay_cohort(raw, lanes)
    if same_engine_greedy is not None:
        verify_same_engine_greedy(report, same_engine_greedy)
    write_json(directory / f"{tag}-replayed.json", report)
    return report


def metrics(url, directory, stem):
    result = control(url, "metrics", timeout=10)
    (directory / f"{stem}.prom").write_text(result["body"], encoding="utf-8")
    return require_http(result)


def reset(url, controls, tag):
    result = control(url, "reset_prefix_cache", b"{}")
    write_json(controls / f"{tag}-cache-reset.json", result)
    require_http(result)


SINGLE_ROUND_PROTOCOL = "variant-single-round-v2"
SINGLE_ROUND_TAGS = ("warmup-01", "measured-01")


def run_vllm(row, endpoint, fixtures, directory):
    url = urlsplit(endpoint)
    if (url.scheme != "http" or url.hostname not in {"127.0.0.1", "localhost", "::1"}
            or url.path not in ("", "/") or url.query or url.fragment or url.username is not None):
        raise ValueError("Provide a loopback HTTP origin")
    directory.mkdir()
    is_prefix = row == "W3-prefix"
    lanes = load_requests(fixtures, "prefix" if is_prefix else "mtp")
    controls = directory / "controls"
    controls.mkdir()
    summary = {"protocol": SINGLE_ROUND_PROTOCOL, "row": row,
               "status": "in_progress", "attempted": [],
               "fixture_manifest_sha256": digest((fixtures / "manifest.json").read_bytes())}
    measured = []
    try:
        if is_prefix:
            summary["attempted"].append("cold-control")
            reset(url, controls, "cold-control")
            cold = capture(url, load_requests(fixtures, "cold"), "cold-control", controls)
            expected = [record["output_ids"] for record in cold["requests"]]
        else:
            expected = goldens("W3")
        write_json(directory / "output-ids.json", expected)
        for tag in SINGLE_ROUND_TAGS:
            summary["attempted"].append(tag)
            if is_prefix:
                reset(url, controls, tag)
                capture(url, load_requests(fixtures, "prime"), f"{tag}-prime", controls)
            metrics(url, controls, f"{tag}-before")
            report = capture(url, lanes, tag, directory, expected, tag == "warmup-01")
            metrics(url, controls, f"{tag}-after")
            if tag.startswith("measured"):
                measured.append(report)
            print(f"{row} {tag}: {report['output_tokens_per_second']:.4f} IDs/s", flush=True)
        summary.update(summarize(measured), status="captured")
    except (ValueError, OSError, TypeError, KeyError, http.client.HTTPException) as error:
        summary.update(status="invalid", error=f"{type(error).__name__}: {error}")
        print(f"{row}: {summary['error']}", flush=True)
    write_json(directory / "summary.json", summary)
    return summary["status"] != "captured"


def main():
    parser = argparse.ArgumentParser(description="Capture the pinned vLLM variant baselines.")
    parser.add_argument("row", choices=ROWS)
    parser.add_argument("endpoint")
    parser.add_argument("fixtures", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    return run_vllm(args.row, args.endpoint, args.fixtures, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
