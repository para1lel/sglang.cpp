import argparse
import base64
import collections
import hashlib
import http.client
import json
import statistics
from pathlib import Path
from time import perf_counter_ns
from urllib.parse import urlsplit

from benchmarks.streaming.check_vllm import audit


HERE = Path(__file__).resolve().parent
NAMES = ("W1-image", "W1-video")
ARCHIVED_COUNTS = (3, 9)
SINGLE_ROUND_COUNTS = (1, 1)
SINGLE_ROUND_PROTOCOL = "media-single-round-v2"
TOKEN_LIMIT = 128
ENDPOINT = "/v1/chat/completions"
TOKENIZE_KEYS = {
    "model", "messages", "chat_template_kwargs", "add_special_tokens", "add_generation_prompt",
    "media_io_kwargs", "mm_processor_kwargs",
}


def load(path):
    return json.loads((HERE / path).read_bytes())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def connection(url):
    address = urlsplit(url)
    assert address.scheme == "http" and address.hostname == "127.0.0.1"
    return http.client.HTTPConnection(address.hostname, address.port, timeout=180)


def json_post(url, path, body):
    conn = connection(url)
    start = perf_counter_ns()
    raw = json.dumps(body, separators=(",", ":")).encode()
    try:
        conn.request("POST", path, raw, {"Content-Type": "application/json"})
        response = conn.getresponse()
        data = response.read()
        elapsed_ns = perf_counter_ns() - start
        if response.status != 200:
            detail = data[:1000].decode(errors="replace")
            raise ValueError(f"{path}: HTTP {response.status}: {detail}")
        return json.loads(data), elapsed_ns
    finally:
        conn.close()


def read_request(name):
    case = load(f"{name}.case.json")
    data = (HERE / f"{name}.http-body").read_bytes()
    assert hashlib.sha256(data).hexdigest() == case["request_sha256"]
    assert len(data) == case["request_bytes"]
    return case, data


def request(url, phase, cohort, name, raw):
    events = []
    record = {"name": name, "phase": phase, "cohort": cohort, "events": events}
    conn = connection(url)
    record["submission_ns"] = perf_counter_ns()
    try:
        conn.request(
            "POST", ENDPOINT, raw,
            {"Content-Type": "application/json", "Accept": "text/event-stream"},
        )
        response = conn.getresponse()
        record["status"] = response.status
        record["content_type"] = response.getheader("Content-Type")
        if response.status != 200 or not (record["content_type"] or "").startswith(
            "text/event-stream"
        ):
            raise ValueError(response.read(65536).decode(errors="replace"))
        pending = b""
        while True:
            block = response.read1(65536)
            now = perf_counter_ns()
            if not block:
                raise ValueError("stream closed before [DONE]")
            pending += block
            while b"\n\n" in pending:
                frame, pending = pending.split(b"\n\n", 1)
                frame = frame.decode("utf-8")
                events.append([now, frame])
                if frame == "data: [DONE]":
                    assert not pending.strip(), pending
                    return record
    except Exception as error:
        record["error"] = f"{type(error).__name__}: {error}"
        return record
    finally:
        conn.close()


def validate(record, case, expected=None):
    if "error" in record:
        raise ValueError(record["error"])
    assert record["status"] == 200 and record["content_type"].startswith("text/event-stream")
    ids = []
    times = []
    usage = None
    finish = None
    choices_seen = 0
    previous = record["submission_ns"]
    done = None
    chunks = collections.Counter()
    for timestamp, frame in record["events"]:
        assert previous <= timestamp and done is None
        previous = timestamp
        assert frame.startswith("data: ")
        if frame == "data: [DONE]":
            assert usage is not None and finish == "length"
            done = timestamp
            continue
        data = json.loads(frame.removeprefix("data: "))
        assert isinstance(data["choices"], list)
        echo = data.get("prompt_token_ids")
        if not data["choices"]:
            assert echo is None and finish == "length" and usage is None
            usage = data["usage"]
            cached = usage["prompt_tokens_details"]["cached_tokens"]
            assert cached == 0
            assert usage["prompt_tokens"] == usage["completion_tokens"] == TOKEN_LIMIT
            assert usage["total_tokens"] == 2 * TOKEN_LIMIT
            continue
        assert len(data["choices"]) == 1 and finish is None and usage is None
        assert echo == (case["effective_input_ids"] if choices_seen == 0 else None)
        choices_seen += 1
        choice, = data["choices"]
        assert choice["index"] == 0
        tokens = choice.get("token_ids") or []
        assert type(tokens) is list and all(type(token) is int and token >= 0 for token in tokens)
        chunks[len(tokens)] += 1
        ids.extend(tokens)
        times.extend([timestamp] * len(tokens))
        finish = choice["finish_reason"]
        assert finish in (None, "length") and choice.get("stop_reason") is None
    assert done and len(ids) == TOKEN_LIMIT and usage is not None
    assert len(case["effective_input_ids"]) == TOKEN_LIMIT
    if expected is not None and ids != expected:
        index = next(
            index for index, (actual, reference) in enumerate(zip(ids, expected))
            if actual != reference
        )
        raise ValueError(
            f"first divergent output at step {index + 1}: "
            f"expected {expected[index]}, saw {ids[index]}"
        )
    submission = record["submission_ns"]
    return {
        "output_ids": ids, "finish_reason": finish, "usage": usage, "token_chunks": dict(chunks),
        "first_output_ns": times[0], "last_output_ns": times[-1], "done_ns": done,
        **timing_metrics(times[0] - submission, times[-1] - submission, done - submission),
    }


def timing_metrics(first_ns, last_ns, e2e_ns):
    return {
        "throughput_tps": TOKEN_LIMIT * 1e9 / e2e_ns,
        "ttft_ms": first_ns / 1e6,
        "tpot_ms": (last_ns - first_ns) / (TOKEN_LIMIT - 1) / 1e6,
        "e2e_ms": e2e_ns / 1e6,
    }


def percentile(values, probability):
    ordered = sorted(values)
    index = (len(ordered) - 1) * probability
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (index - lower) * (ordered[upper] - ordered[lower])


def summarize(samples, output_ids, counts=ARCHIVED_COUNTS):
    if counts == SINGLE_ROUND_COUNTS:
        sample, = samples
        return {"requests": {"warmup": 1, "measured": 1}, "output_ids": output_ids,
                "finish_reason": "length", "measurement": {key: sample[key]
                    for key in ("throughput_tps", "ttft_ms", "tpot_ms", "e2e_ms")}}
    durations = {}
    for metric in ("throughput_tps", "ttft_ms", "tpot_ms", "e2e_ms"):
        values = [result[metric] for result in samples]
        durations[metric] = {
            "min": min(values), "max": max(values), "p50": percentile(values, 0.5),
            "p95": percentile(values, 0.95), "sample_stdev": statistics.stdev(values),
        }
    throughput = durations["throughput_tps"]
    margin = max(0.01, 2 * throughput["sample_stdev"] / throughput["p50"])
    ceilings = {
        metric: stats["p95"] + max(0.01 * stats["p95"], 2 * stats["sample_stdev"])
        for metric, stats in durations.items() if metric != "throughput_tps"
    }
    return {
        "requests": dict(zip(("warmup", "measured"), counts, strict=True)),
        "output_ids": output_ids, "finish_reason": "length", "statistics": durations,
        "thresholds": {
            "median_throughput_ratio_strictly_greater_than": 1 + margin,
            "median_throughput_tps_strictly_greater_than": throughput["p50"] * (1 + margin),
            "p95_latency_at_most": ceilings,
        },
    }


def verify():
    reference = load("gpu0-reference.json")
    timings = load("gpu0-timings.json")
    assert not reference["failures"] and set(reference["rows"]) == set(NAMES)
    columns = ("first_output_ns", "last_output_ns", "done_ns")
    assert timings["submission_relative_columns"] == list(columns)
    _, measured = ARCHIVED_COUNTS
    for name in NAMES:
        case, raw = read_request(name)
        media, text = json.loads(raw)["messages"][0]["content"]
        url = media[media["type"]]["url"]
        prefix = f"data:{case['mime']};base64,"
        assert url.startswith(prefix) and text["text"] == "".join(case["text_segments"])
        assert text["text"] in "".join(case["template_rendered_segments"])
        payload = base64.b64decode(url.removeprefix(prefix), validate=True)
        assert len(payload) == case["media_bytes"]
        assert hashlib.sha256(payload).hexdigest() == case["media_sha256"]
        input_ids = case["effective_input_ids"]
        canonical_ids = json.dumps(input_ids, separators=(",", ":")).encode()
        assert hashlib.sha256(canonical_ids).hexdigest() == case["effective_input_sha256"]
        assert len(input_ids) == case["max_output_tokens"] == TOKEN_LIMIT
        expected = reference["rows"][name]
        output_ids = expected["output_ids"]
        assert len(output_ids) == TOKEN_LIMIT
        assert all(type(token) is int and 0 <= token < 248320 for token in input_ids + output_ids)
        assert len(timings[name]) == measured
        for first_ns, last_ns, e2e_ns in timings[name]:
            assert 0 < first_ns <= last_ns <= e2e_ns
        samples = [timing_metrics(*row) for row in timings[name]]
        assert summarize(samples, output_ids) == expected
        print(f"{name}: verified media, request, input IDs, and {measured} timing rows")


def replay(path):
    cases = {name: read_request(name)[0] for name in NAMES}
    raw = [json.loads(line) for line in path.read_text().splitlines()]
    versions = {row.get("protocol") for row in raw}
    if versions == {SINGLE_ROUND_PROTOCOL}:
        counts = SINGLE_ROUND_COUNTS
    elif versions == {None}:
        counts = ARCHIVED_COUNTS
    else:
        raise ValueError("Unknown or mixed media protocols")
    schedule = [(phase, cohort, name)
                for phase, rounds in zip(("warmup", "measure"), counts, strict=True)
                for cohort in range(1, rounds + 1) for name in NAMES]
    assert [(row["phase"], row["cohort"], row["name"]) for row in raw] == schedule
    expected = {}
    failure = []
    measurements = {name: [] for name in NAMES}
    for row in raw:
        name = row["name"]
        try:
            result = validate(row, cases[name], expected.get(name))
            expected.setdefault(name, result["output_ids"])
            if row["phase"] == "measure":
                measurements[name].append(result)
        except Exception as error:
            failure.append({"name": name, "phase": row["phase"], "cohort": row["cohort"],
                            "error": f"{type(error).__name__}: {error}"})
    report = {"failures": failure, "rows": {}}
    if counts == SINGLE_ROUND_COUNTS:
        report["protocol"] = SINGLE_ROUND_PROTOCOL
    _, measured = counts
    for name in NAMES:
        samples = measurements[name]
        if len(samples) == measured and not any(item["name"] == name for item in failure):
            report["rows"][name] = summarize(samples, expected[name], counts)
    return report


def measure(url, output):
    output.mkdir()
    state, _ = json_post(url, "/collective_rpc", {"method": "media_snapshot", "timeout": 90})
    write(output / "effective-worker.json", state)
    audit(state)
    media = state["results"][0]["media"]
    assert media["model_class"] == "Qwen3_5ForConditionalGeneration"
    assert media["vision_class"] == "Qwen3_VisionTransformer" and media["vision_depth"] > 0
    assert media["vision_dtype"] == "torch.bfloat16"
    assert media["mm_processor_cache_gb"] == 0 and media["max_num_seqs"] == 1
    fixtures = {}
    for name in NAMES:
        case, raw = read_request(name)
        fixtures[name] = (case, raw)
        body = {key: value for key, value in json.loads(raw).items() if key in TOKENIZE_KEYS}
        for repetition in (1, 2):
            result, elapsed = json_post(url, "/tokenize", body)
            write(output / f"{name}.tokenize.{repetition}.json", {
                "body": result, "localhost_round_trip_ns": elapsed,
            })
            assert result["tokens"] == case["effective_input_ids"]
            assert result["count"] == TOKEN_LIMIT
    stream_path = output / "streams.jsonl"
    with stream_path.open("x", encoding="utf-8") as rows:
        for phase in ("warmup", "measure"):
            for name, (case, raw) in fixtures.items():
                record = request(url, phase, 1, name, raw)
                record["protocol"] = SINGLE_ROUND_PROTOCOL
                rows.write(json.dumps(record, separators=(",", ":")) + "\n")
                rows.flush()
                try:
                    result = validate(record, case)
                    rate = result["throughput_tps"]
                    print(name, phase, 1, f"{rate:.2f} tok/s", flush=True)
                except Exception as error:
                    detail = f"{type(error).__name__}: {error}"
                    print(name, phase, 1, "failed:", detail, flush=True)
    report = replay(stream_path)
    write(output / "summary.json", report)
    if report["failures"]:
        raise SystemExit(f"{len(report['failures'])} streams failed replay")
    assert len(report["rows"]) == len(NAMES)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("verify", "measure", "replay"))
    parser.add_argument("--url", default="http://127.0.0.1:17462")
    parser.add_argument("--output", type=Path)
    options = parser.parse_args()
    if options.mode == "verify":
        verify()
    elif options.output is None:
        parser.error("--output is required for measure and replay")
    elif options.mode == "measure":
        measure(options.url, options.output)
    else:
        report = replay(options.output / "streams.jsonl")
        print(json.dumps(report, indent=2, allow_nan=False))
        if report["failures"]:
            raise SystemExit(1)
