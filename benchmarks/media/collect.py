import argparse
import base64
import collections
import hashlib
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
NAMES = ("W1-image", "W1-video")
ARCHIVED_COUNTS = (3, 9)
TOKEN_LIMIT = 128
ENDPOINT = "/v1/chat/completions"


def load(path):
    return json.loads((HERE / path).read_bytes())


def read_request(name):
    case = load(f"{name}.case.json")
    data = (HERE / f"{name}.http-body").read_bytes()
    assert hashlib.sha256(data).hexdigest() == case["request_sha256"]
    assert len(data) == case["request_bytes"]
    return case, data


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


def summarize(samples: list[dict[str, float]], output_ids: list[int]) -> dict[str, object]:
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
        "requests": dict(zip(("warmup", "measured"), ARCHIVED_COUNTS, strict=True)),
        "output_ids": output_ids,
        "finish_reason": "length",
        "statistics": durations,
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("verify",))
    parser.parse_args()
    verify()
