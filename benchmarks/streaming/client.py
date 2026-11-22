import argparse
import hashlib
import http.client
import json
import platform
import statistics
import sys
from pathlib import Path
from time import perf_counter_ns
from urllib.parse import urlsplit


SHAPES = {"W1": (128, 128), "W2": (2048, 512)}
WARMUPS = 3
REPEATS = 9
PUBLISHED_REFERENCE_SHA256 = "798aeebc2688a9d5744892524c1995e6060612640656fde03b6e2609ac4c4bc7"


def json_bytes(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def load_cases(directory):
    cases = {}
    for name, (input_count, output_count) in SHAPES.items():
        raw = (directory / f"{name}.json").read_bytes()
        case = json.loads(raw)
        if (case["workload"] != name or len(case["prompt_token_ids"]) != input_count
                or case["output_tokens"] != output_count
                or not token_ids(case["prompt_token_ids"])):
            raise ValueError(f"{name} fixture does not meet its contract")
        cases[name] = (case, hashlib.sha256(raw).hexdigest())
    return cases


def token_ids(ids):
    return isinstance(ids, list) and all(type(value) is int and value >= 0 for value in ids)


def request_body(case):
    return {
        "model": "benchmark",
        "prompt": case["prompt_token_ids"],
        "max_tokens": case["output_tokens"],
        "temperature": 0.0,
        "seed": 0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
        "return_token_ids": True,
        "stream_interval": 1,
        "add_special_tokens": False,
    }


def sse_frames(response, result=None):
    pending = b""
    lines = []
    read_index = 0
    body_bytes = 0
    while data := response.read1(65536):
        now = perf_counter_ns()
        read_index += 1
        body_bytes += len(data)
        parts = (pending + data).split(b"\n")
        pending = parts.pop()
        for part in parts:
            line = part.removesuffix(b"\r")
            if line:
                lines.append(line.decode("utf-8"))
            else:
                yield {"ns": now, "read": read_index, "lines": lines}
                lines = []
    if response.length:
        raise http.client.IncompleteRead(b"", response.length)
    if pending or lines:
        yield {"ns": perf_counter_ns(), "read": read_index, "lines": lines,
               "unterminated": pending.hex()}
    if result is not None:
        result["response_end"] = {"ns": perf_counter_ns(), "body_bytes": body_bytes}


class Completion:
    def __init__(self):
        self.ids = []
        self.prompt = None
        self.usage = None
        self.finish = None
        self.token_chunks = []

    def consume(self, payload, ns):
        if "error" in payload:
            raise ValueError(f"Server error: {payload['error']}")
        choices = payload["choices"]
        usage = payload.get("usage")
        if not choices:
            if usage is not None:
                if self.usage is not None or self.finish is None:
                    raise ValueError("Unexpected usage event")
                self.usage = usage
            return
        choice, = choices
        if choice["index"] != 0 or self.finish is not None:
            raise ValueError("Unexpected completion choice")
        ids = choice.get("token_ids")
        if not token_ids(ids):
            raise ValueError("Completion choice must carry explicit delta token IDs")
        prompt = choice.get("prompt_token_ids")
        if self.prompt is None:
            if not token_ids(prompt):
                raise ValueError("First completion choice must echo prompt token IDs")
            self.prompt = prompt
        elif prompt is not None:
            raise ValueError("Prompt IDs repeated")
        if ids:
            self.ids.extend(ids)
            self.token_chunks.append((ns, len(ids)))
        self.finish = choice.get("finish_reason")

    def validate(self, case):
        counts = (len(case["prompt_token_ids"]), case["output_tokens"])
        if not token_ids(self.prompt) or self.prompt != case["prompt_token_ids"]:
            raise ValueError("Prompt IDs absent or changed")
        if len(self.ids) != counts[1] or self.finish != "length":
            raise ValueError(f"Expected length completion of {counts[1]} IDs, got "
                             f"{len(self.ids)} IDs and finish_reason={self.finish!r}")
        usage = self.usage or {}
        if (usage.get("prompt_tokens"), usage.get("completion_tokens")) != counts:
            raise ValueError("Server usage counts differ from the fixture")
        if usage.get("prompt_tokens_details", {}).get("cached_tokens") != 0:
            raise ValueError("Server must report zero reused prompt tokens")


def completion_frames(events, recorded=None):
    done = False
    trailing_data = False
    for event in events:
        if recorded is not None:
            recorded.append(event)
        if "unterminated" in event:
            raise ValueError("Incomplete SSE frame")
        values = [line[5:].removeprefix(" ") for line in event["lines"]
                  if line == "data" or line.startswith("data:")]
        if done and values:
            trailing_data = True
        data = "\n".join(values)
        done |= data == "[DONE]"
        if not trailing_data:
            yield event, data
    if not done:
        raise ValueError("Stream ended before [DONE]")
    if trailing_data:
        raise ValueError("Data after [DONE]")


def record_stream(response, result, completion):
    if response.status != 200:
        result["http_error_body"] = response.read().decode("utf-8", errors="replace")
        raise ValueError(f"HTTP {response.status}")
    if response.headers.get_content_type() != "text/event-stream":
        result["http_error_body"] = response.read().decode("utf-8", errors="replace")
        raise ValueError("Expected text/event-stream")
    for event, data in completion_frames(sse_frames(response, result), result["events"]):
        if data == "[DONE]":
            result["done_ns"] = event["ns"]
        elif data:
            completion.consume(json.loads(data), event["ns"])


def measure(url, case, body):
    conn = StreamingConnection(url.hostname, url.port, timeout=120)
    result = {"events": [], "request_sha256": hashlib.sha256(body).hexdigest()}
    completion = Completion()
    try:
        result["submit_ns"] = perf_counter_ns()
        conn.request("POST", "/v1/completions", body, {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "Connection": "close",
        })
        with conn.getresponse() as response:
            result["headers_ns"] = perf_counter_ns()
            result["http_status"] = response.status
            result["response_headers"] = response.getheaders()
            record_stream(response, result, completion)
        completion.validate(case)
    except (
        OSError, http.client.HTTPException, ValueError, TypeError, KeyError, AttributeError
    ) as error:
        result["failure_ns"] = perf_counter_ns()
        result["error"] = f"{type(error).__name__}: {error}"
    finally:
        conn.close()
    result.update(output_ids=completion.ids, token_chunks=completion.token_chunks,
                  finish_reason=completion.finish, usage=completion.usage)
    if "error" not in result:
        first, last = completion.token_chunks[0][0], completion.token_chunks[-1][0]
        duration = result["done_ns"] - result["submit_ns"]
        count = len(completion.ids)
        result["metrics"] = {
            "output_tokens_per_second": count * 1e9 / duration,
            "ttft_ms": (first - result["submit_ns"]) / 1e6,
            "tpot_ms": (last - first) / (count - 1) / 1e6,
            "e2e_ms": duration / 1e6,
            "requests_per_second": 1e9 / duration,
            "aggregate_tokens_per_second": (count + len(completion.prompt)) * 1e9 / duration,
        }
    return result


class ChunkedResponse(http.client.HTTPResponse):
    def _get_chunk_left(self):
        if self.chunk_left == 0:
            if self._safe_read(2) != b"\r\n":
                raise http.client.HTTPException("Invalid chunk data separator")
            self.chunk_left = None
        return super()._get_chunk_left()

    def _read_and_discard_trailer(self):
        max_headers = getattr(self, "_max_headers", None)
        if max_headers is None:
            max_headers = http.client._MAXHEADERS
        for _ in range(max_headers + 1):
            line = self.fp.readline(http.client._MAXLINE + 1)
            if len(line) > http.client._MAXLINE:
                raise http.client.LineTooLong("trailer line")
            if not line.endswith(b"\r\n"):
                raise http.client.HTTPException("Missing CRLF in chunk trailer")
            if line == b"\r\n":
                return
        raise http.client.HTTPException(f"got more than {max_headers} trailers")


class StreamingConnection(http.client.HTTPConnection):
    response_class = ChunkedResponse


def percentile(values, fraction):
    values = sorted(values)
    rank = fraction * (len(values) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (rank - lower)


def summarize(results, freeze_baseline):
    measured = [item for item in results if item["phase"] == "measured"]
    good = [item for item in measured if "error" not in item]
    first = next((item for item in results if "error" not in item), None)
    report = {
        "successful": len(good),
        "attempted": len(measured),
        "warmups_successful": sum("error" not in item and item["phase"] == "warmup"
                                  for item in results),
        "greedy_output_ids": first["output_ids"] if first else None,
        "valid": all("error" not in item for item in results),
    }
    if not good:
        return report
    for key in good[0]["metrics"]:
        values = [item["metrics"][key] for item in good]
        report[key] = {
            "n": len(values), "p50": statistics.median(values), "p95": percentile(values, .95),
            "min": min(values), "max": max(values),
            "sample_stdev": statistics.stdev(values) if len(values) > 1 else None,
            "samples": values,
        }
    if freeze_baseline and report["valid"] and len(good) == REPEATS:
        throughput = report["output_tokens_per_second"]
        margin = max(0.01, 2 * throughput["sample_stdev"] / throughput["p50"])
        report["thresholds"] = {
            "strict_throughput_ratio_gt": 1 + margin,
            "strict_median_output_tokens_per_second_gt": throughput["p50"] * (1 + margin),
            "p95_latency_ceilings_ms": {
                key: report[key]["p95"] + max(.01 * report[key]["p95"],
                                              2 * report[key]["sample_stdev"])
                for key in ("ttft_ms", "tpot_ms", "e2e_ms")
            },
        }
    return report


def compare(report, baseline):
    limits = baseline["thresholds"]
    if not report["valid"]:
        return {"passed": False, "error": "Stream validation failed"}
    ratio = (report["output_tokens_per_second"]["p50"]
             / baseline["output_tokens_per_second"]["p50"])
    latency = {
        key: {
            "difference_ms": report[key]["p95"] - baseline[key]["p95"],
            "passed": report[key]["p95"] <= ceiling,
        }
        for key, ceiling in limits["p95_latency_ceilings_ms"].items()
    }
    throughput_passed = ratio > limits["strict_throughput_ratio_gt"]
    return {
        "passed": throughput_passed and all(metric["passed"] for metric in latency.values()),
        "throughput_ratio": ratio,
        "throughput_passed": throughput_passed,
        "p95_latency": latency,
    }


def run(args):
    url = urlsplit(args.url)
    if (url.scheme != "http" or not url.hostname or url.username is not None
            or url.path not in ("", "/") or url.query or url.fragment):
        raise ValueError("Use an HTTP origin without credentials, query or path")
    cases = load_cases(args.fixtures)
    previous_bytes = args.reference.read_bytes() if args.reference else None
    previous = json.loads(previous_bytes) if previous_bytes is not None else None
    schedule = [("warmup", WARMUPS), ("measured", REPEATS)]
    if args.probe:
        schedule = [("probe", 1)]
        cases = {args.probe: cases[args.probe]}
    bodies = {name: json_bytes(request_body(case)) for name, (case, _) in cases.items()}
    manifest = {
        "url": args.url, "command": sys.argv, "python": sys.version, "host": platform.platform(),
        "clock": "perf_counter_ns", "schedule": schedule, "workload_order": list(cases),
        "fixture_sha256": {name: digest for name, (_, digest) in cases.items()},
        "request_sha256": {name: hashlib.sha256(body).hexdigest() for name, body in bodies.items()},
        "client_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    expected = {}
    if previous is not None:
        published = hashlib.sha256(previous_bytes).hexdigest() == PUBLISHED_REFERENCE_SHA256
        for name in cases:
            baseline = previous["workloads"][name]
            for field in ("fixture_sha256", "request_sha256", "client_sha256"):
                actual = manifest[field]
                frozen = previous["manifest"][field]
                if isinstance(actual, dict):
                    actual, frozen = actual[name], frozen[name]
                if actual != frozen and (field != "client_sha256" or not published):
                    raise ValueError(f"{name} differs from baseline {field}")
            if not baseline["valid"] or "thresholds" not in baseline:
                raise ValueError(f"{name} reference requires a successful baseline")
            expected[name] = baseline["greedy_output_ids"]
    args.output.mkdir(parents=True)
    (args.output / "client.json").write_bytes(json_bytes(manifest) + b"\n")
    for name, body in bodies.items():
        (args.output / f"{name}.request.json").write_bytes(body)
    results = {name: [] for name in cases}
    with (args.output / "events.jsonl").open("wb") as raw:
        for phase, count in schedule:
            for cohort in range(1, count + 1):
                for name, (case, _) in cases.items():
                    item = measure(url, case, bodies[name])
                    item.update(workload=name, phase=phase, cohort=cohort)
                    if "error" not in item:
                        reference = expected.setdefault(name, item["output_ids"])
                        if item["output_ids"] != reference:
                            item["error"] = "Greedy output IDs differ from the reference"
                    results[name].append(item)
                    raw.write(json_bytes(item) + b"\n")
                    raw.flush()
                    if "error" in item:
                        print(f"{name} {phase} {cohort}: {item['error']}", file=sys.stderr)
    summary = {name: summarize(items, args.freeze_baseline) for name, items in results.items()}
    if previous and not args.probe:
        for name, report in summary.items():
            report["comparison"] = compare(report, previous["workloads"][name])
    (args.output / "summary.json").write_text(
        json.dumps({"manifest": manifest, "workloads": summary}, indent=2) + "\n", encoding="utf-8"
    )
    return 0 if all(item["valid"] and item.get("comparison", {}).get("passed", True)
                    for item in summary.values()) else 1


def main():
    parser = argparse.ArgumentParser(description="Measure the fixed W1/W2 token-ID SSE contract")
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, default=Path(__file__).parent)
    parser.add_argument("--reference", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--freeze-baseline", action="store_true")
    mode.add_argument("--probe", choices=SHAPES)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
