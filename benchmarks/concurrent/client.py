"""Measure or replay synchronized W3–W6 SSE cohorts."""

import argparse
import gzip
import http.client
import json
import socket
import statistics
import threading
from concurrent.futures import ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter_ns, time_ns
from urllib.parse import urlsplit

from benchmarks.concurrent.reference import ROWS, load_lanes, reference
from benchmarks.streaming.client import Completion, percentile, sse_frames


RATES = ("output_tokens_per_second", "requests_per_second", "aggregate_tokens_per_second")
LATENCIES = ("ttft_ms", "tpot_ms", "e2e_ms")


def consume_events(events, record, case, completion):
    read_number, previous_ns = 0, record["headers_ns"]
    for event in events:
        if "unterminated" in event:
            raise ValueError("Incomplete SSE frame")
        event_ns, current = event["ns"], event["read"]
        if (event_ns < previous_ns or current < read_number
                or (current == read_number and event_ns != previous_ns)):
            raise ValueError("SSE frame timestamps are inconsistent with read boundaries")
        read_number, previous_ns = current, event_ns
        data = "\n".join(line[5:].removeprefix(" ") for line in event["lines"]
                         if line.startswith("data:"))
        if not data:
            continue
        if "done_ns" in record:
            raise ValueError("Data after [DONE]")
        if data == "[DONE]":
            completion.validate(case)
            usage = completion.usage
            counts = (usage["prompt_tokens"], usage["completion_tokens"],
                      usage["prompt_tokens_details"]["cached_tokens"])
            if any(type(count) is not int for count in counts):
                raise ValueError("Usage counts must be integers")
            total = len(case["prompt_token_ids"]) + case["output_tokens"]
            if "total_tokens" in usage and (type(usage["total_tokens"]) is not int
                                            or usage["total_tokens"] != total):
                raise ValueError("Usage total differs from the fixture")
            record["done_ns"] = event_ns
            continue
        payload = json.loads(data)
        if "error" in payload:
            raise ValueError(f"Server error: {payload['error']}")
        if (payload.get("model", "benchmark") != "benchmark"
                or payload.get("object", "text_completion") != "text_completion"):
            raise ValueError("Unexpected completion model or object")
        stream_id = payload.get("id")
        if "id" in payload and (not isinstance(stream_id, str) or not stream_id):
            raise ValueError("Invalid completion ID")
        if record.setdefault("stream_id", stream_id) != stream_id:
            raise ValueError("Completion ID changed within the stream")
        choices, usage = payload["choices"], payload.get("usage")
        if (choices and usage is not None) or (not choices and usage is None):
            raise ValueError("Expected a choice event or a terminal usage event")
        if choices and (type(choices[0]["index"]) is not int
                        or choices[0].get("stop_reason") is not None):
            raise ValueError("Invalid choice index or stop reason")
        completion.consume(payload, event_ns)
    if "done_ns" not in record:
        raise ValueError("Stream ended before [DONE]")


def finish_record(record, completion):
    record.update(output_ids=completion.ids, finish_reason=completion.finish,
                  usage=completion.usage, token_chunks=completion.token_chunks)
    if "error" in record:
        return
    first, last = completion.token_chunks[0][0], completion.token_chunks[-1][0]
    record["first_ns"], record["last_ns"] = first, last
    record["metrics"] = {
        "ttft_ms": (first - record["submit_ns"]) / 1e6,
        "tpot_ms": (last - first) / (len(completion.ids) - 1) / 1e6
        if len(completion.ids) > 1 else None,
        "e2e_ms": (record["done_ns"] - record["submit_ns"]) / 1e6,
    }


def capture_events(response, record, case, completion):
    for event in sse_frames(response):
        record["events"].append(event)
        if "unterminated" in event:
            raise ValueError("Incomplete SSE frame")
        data = "\n".join(line[5:].removeprefix(" ") for line in event["lines"]
                         if line.startswith("data:"))
        if not data:
            continue
        if "done_ns" in record:
            raise ValueError("Data after [DONE]")
        if data == "[DONE]":
            completion.validate(case)
            record["done_ns"] = event["ns"]
        else:
            completion.consume(json.loads(data), event["ns"])
    if "done_ns" not in record:
        raise ValueError("Stream ended before [DONE]")


def capture_lane(item, url, barrier, sockets, policy):
    lane, body, case = item
    number = lane["lane"]
    result = {"lane": number, "request_sha256": lane["request_sha256"], "events": []}
    connection = http.client.HTTPConnection(
        url.hostname, url.port, timeout=policy["socket_timeout_seconds"]
    )
    completion = Completion()
    try:
        connection.connect()
        sockets[number] = connection.sock
        result["connect_ns"] = perf_counter_ns()
        barrier.wait(policy["barrier_timeout_seconds"])
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
                raise ValueError(f"Unexpected streaming response HTTP {response.status}")

            capture_events(response, result, case, completion)
            result["eof_ns"] = perf_counter_ns()
    except (OSError, http.client.HTTPException, ValueError, TypeError, KeyError,
            IndexError, AttributeError, threading.BrokenBarrierError) as error:
        if "submit_ns" not in result:
            barrier.abort()
        result["failure_ns"] = perf_counter_ns()
        result["error"] = f"{type(error).__name__}: {error}"
    finally:
        connection.close()
    finish_record(result, completion)
    return result


def capture_cohort(url, name, lanes, tag, policy):
    started = datetime.now(timezone.utc).isoformat()
    clock_ns, wall_ns = perf_counter_ns(), time_ns()
    released = []
    barrier = threading.Barrier(len(lanes), action=lambda: released.append(perf_counter_ns()))
    sockets = [None] * len(lanes)
    instrumentation = {}
    with ThreadPoolExecutor(len(lanes)) as pool:
        futures = [pool.submit(capture_lane, lane, url, barrier, sockets, policy) for lane in lanes]
        deadline = clock_ns + policy["cohort_deadline_seconds"] * 1e9
        _, pending = wait(futures, timeout=max(0, deadline - perf_counter_ns()) / 1e9)
        if pending:
            instrumentation["deadline_exceeded"] = True
            barrier.abort()
            for sock in sockets:
                if sock is not None:
                    try:
                        sock.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
        requests = [task.result() for task in futures]
    return {"row": name, "tag": tag, "started_utc": started,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "clock_reference": {"perf_counter_ns": clock_ns, "time_ns": wall_ns},
            "barrier_ns": released[0] if released else None,
            "instrumentation": instrumentation, "requests": requests,
            "errors": [item["error"] for item in requests if "error" in item]}


def replay_lane(source, item):
    lane, _, case = item
    if (source["lane"] != lane["lane"] or source["request_sha256"] != lane["request_sha256"]
            or source["http_status"] != 200 or "error" in source):
        raise ValueError(f'Invalid HTTP outcome or request identity for lane {lane["lane"]}')
    if not any(key.lower() == "content-type" and value.split(";")[0].strip() == "text/event-stream"
               for key, value in source["response_headers"]):
        raise ValueError("Expected text/event-stream")
    completion = Completion()
    record = {key: source[key] for key in ("connect_ns", "submit_ns", "sent_ns",
                                          "headers_ns", "eof_ns")}
    consume_events(source["events"], record, case, completion)
    finish_record(record, completion)
    if not (record["connect_ns"] <= record["submit_ns"] <= record["sent_ns"]
            <= record["headers_ns"] <= record["first_ns"] <= record["last_ns"]
            <= record["done_ns"] <= record["eof_ns"]):
        raise ValueError("Request timestamps are out of order")
    for key in ("done_ns", "first_ns", "last_ns", "output_ids", "finish_reason", "usage"):
        if record[key] != source[key]:
            raise ValueError(f"Recorded {key} differs from SSE replay")
    if record["token_chunks"] != [tuple(chunk) for chunk in source["token_chunks"]]:
        raise ValueError("Recorded chunks differ from SSE replay")
    if record["metrics"] != source["metrics"]:
        raise ValueError("Recorded latencies differ from SSE replay")
    return record


def replay_cohort(raw, lanes, policy):
    instrumentation = raw["instrumentation"]
    if raw["errors"] or any(key in instrumentation
                            for key in ("error", "stop_error", "deadline_exceeded")):
        raise ValueError(f'Cohort {raw["tag"]} failed: {raw["errors"] or instrumentation}')
    records = [replay_lane(source, item)
               for source, item in zip(raw["requests"], lanes, strict=True)]
    stream_ids = [item["stream_id"] for item in records if item["stream_id"] is not None]
    if len(set(stream_ids)) != len(stream_ids):
        raise ValueError("Completion IDs must be distinct across the cohort")
    earliest = min(item["submit_ns"] for item in records)
    span = (max(item["submit_ns"] for item in records) - earliest) / 1e6
    earliest_done = min(item["done_ns"] for item in records)
    latest_done = max(item["done_ns"] for item in records)
    latest_headers = max(item["headers_ns"] for item in records)
    earliest_first = min(item["first_ns"] for item in records)
    latest_first = max(item["first_ns"] for item in records)
    if (type(raw["barrier_ns"]) is not int
            or not max(item["connect_ns"] for item in records) <= raw["barrier_ns"] <= earliest
            or not max(item["sent_ns"] for item in records) < earliest_first
            or max(latest_headers, latest_first) >= earliest_done
            or span > policy["max_submission_span_ms"]):
        raise ValueError("Cohort submissions or open streams failed overlap bounds")
    duration = (latest_done - earliest) / 1e9
    prompt = sum(record["usage"]["prompt_tokens"] for record in records)
    output = sum(len(record["output_ids"]) for record in records)
    return {"output_tokens_per_second": output / duration,
            "requests_per_second": len(records) / duration,
            "aggregate_tokens_per_second": (prompt + output) / duration,
            "submission_span_ms": span,
            "all_streams_overlap_ms": (earliest_done - latest_headers) / 1e6,
            "all_generating_overlap_ms": (earliest_done - latest_first) / 1e6,
            "request_latencies": [record["metrics"] for record in records]}, records


def validate_cohort(raw, lanes, policy):
    report, _ = replay_cohort(raw, lanes, policy)
    return report


def describe(values):
    return {"n": len(values), "p50": statistics.median(values),
            "p95": percentile(values, .95), "min": min(values), "max": max(values),
            "sample_stdev": statistics.stdev(values) if len(values) > 1 else None,
            "samples": values}


def summarize(measured):
    return {
        "throughput": {key: describe([cohort[key] for cohort in measured]) for key in RATES},
        "latency_ms": {key: describe([request[key] for cohort in measured
                                     for request in cohort["request_latencies"]])
                       for key in LATENCIES},
    }


def thresholds(statistics_by_metric):
    rate = statistics_by_metric["throughput"]["output_tokens_per_second"]
    ratio = 1 + max(.01, 2 * rate["sample_stdev"] / rate["p50"])
    return {"strict_throughput_ratio_gt": ratio,
            "strict_median_output_tokens_per_second_gt": rate["p50"] * ratio,
            "p95_latency_ceilings_ms": {
                key: stat["p95"] + max(.01 * stat["p95"], 2 * stat["sample_stdev"])
                for key, stat in statistics_by_metric["latency_ms"].items()}}


def gate(measured, limits):
    latency = measured["latency_ms"]
    return {
        "throughput": measured["throughput"]["output_tokens_per_second"]["p50"]
        > limits["strict_median_output_tokens_per_second_gt"],
        "latency": {key: latency[key]["p95"] <= limit
                    for key, limit in limits["p95_latency_ceilings_ms"].items()},
    }


def tags(policy):
    return [f"{phase}-{index:02d}" for phase in ("warmup", "measured")
            for index in range(1, policy[f"{phase}_cohorts"] + 1)]


def verify_round_ids(raw, warmup_ids):
    for request, expected in zip(raw["requests"], warmup_ids, strict=True):
        actual = request["output_ids"]
        if actual == expected:
            continue
        for offset, (before, after) in enumerate(zip(expected, actual, strict=True)):
            if before != after:
                raise ValueError(f"Lane {request['lane']} output {offset} differs from warmup: "
                                 f"expected {before}, observed {after}")


SINGLE_ROUND_PROTOCOL = "concurrent-single-round-v2"


def run(name, endpoint, fixtures, directory, *, validate=validate_cohort, enforce_gate=True):
    url = urlsplit(endpoint)
    if (url.scheme != "http" or url.hostname not in {"127.0.0.1", "localhost", "::1"}
            or url.path not in ("", "/") or url.query or url.fragment or url.username is not None):
        raise ValueError("Provide a loopback HTTP origin")
    baseline = reference()
    policy = baseline["policy"] | {"warmup_cohorts": 1, "measured_cohorts": 1}
    lanes = load_lanes(fixtures, name)
    directory.mkdir()
    attempts, measured = [], []
    warmup_ids = None
    report = {"protocol": SINGLE_ROUND_PROTOCOL, "row": name,
              "request_outcomes": attempts, "status": "in_progress",
              "fixture_manifest_sha256": baseline["fixture_manifest_sha256"],
              "baseline_bundle_sha256": baseline["archive"]["bundle_sha256"]}
    for tag in tags(policy):
        raw = capture_cohort(url, name, lanes, tag, policy)
        path = directory / f"{tag}.json.gz"
        with gzip.open(path, "wt", compresslevel=3) as stream:
            json.dump(raw, stream, separators=(",", ":"), allow_nan=False)
        outcome = {"tag": tag, "raw_file": path.name, "attempted": len(raw["requests"]),
                   "successful_http_completions":
                       sum("error" not in item for item in raw["requests"])}
        attempts.append(outcome)
        try:
            result = validate(raw, lanes, policy)
            if warmup_ids is None:
                warmup_ids = [request["output_ids"] for request in raw["requests"]]
            else:
                verify_round_ids(raw, warmup_ids)
        except (ValueError, KeyError, TypeError) as error:
            outcome["error"] = str(error)
            report["status"] = "invalid_cohort"
            break
        outcome.update({key: value for key, value in result.items() if key != "request_latencies"})
        if tag.startswith("measured"):
            measured.append(result)
    else:
        report.update(summarize(measured), status="complete")
        report["wire_and_timing_gates"] = gate(report, baseline["rows"][name]["thresholds"])
    (directory / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"{name}: {report['status']} ({directory / 'summary.json'})")
    if report["status"] != "complete":
        return 1
    verdict = report["wire_and_timing_gates"]
    return int(enforce_gate and not (verdict["throughput"] and all(verdict["latency"].values())))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("row", choices=ROWS)
    parser.add_argument("endpoint")
    parser.add_argument("fixtures", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    raise SystemExit(run(args.row, args.endpoint, args.fixtures, args.output))
