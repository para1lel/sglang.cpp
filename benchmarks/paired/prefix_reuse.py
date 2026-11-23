"""Capture a separate, identically primed W3 prefix-reuse diagnostic."""

import argparse
import gzip
import json
from os import SEEK_END
from pathlib import Path
from urllib.parse import urlsplit

import cold_cache
import k4
from benchmarks.concurrent.reference import digest
from benchmarks.variants import client, fixtures


COMMON_PREFIX = 8192
BRANCH_SUFFIX = 1792
DRAFT_CACHED_ROWS = "sglang_cpp:prefix_cache_draft_cached_rows_total"


def workload(root):
    original = k4.fixture("W3", root)
    prompts = [case["prompt_token_ids"] for _, _, case in original]
    prefix = prompts[0][:COMMON_PREFIX]
    branch_tokens = {prompt[0] for prompt in prompts}
    k4.require(len(prefix) == COMMON_PREFIX and len(branch_tokens) == len(prompts),
               "W3 fixture lacks a complete common prefix or distinct branch tokens")
    off_branch = next(token for token in prompts[0] if token not in branch_tokens)
    branches = [prefix + prompt[:BRANCH_SUFFIX] for prompt in prompts]

    def request(lane, ids, count):
        body = fixtures.make_request(ids, count)
        return ({"lane": lane, "request_sha256": digest(body)}, body,
                {"prompt_token_ids": ids, "output_tokens": count})

    return (original, [request(0, prefix + [off_branch], 1)],
            [request(lane, ids, fixtures.OUTPUT) for lane, ids in enumerate(branches)])


def cached_case(raw, item, lower, upper):
    cached = []
    for frame in raw["events"]:
        data = "\n".join(line[5:].removeprefix(" ") for line in frame["lines"]
                         if line.startswith("data:"))
        if data and data != "[DONE]":
            usage = json.loads(data).get("usage")
            if usage is not None:
                cached.append(usage["prompt_tokens_details"]["cached_tokens"])
    value, = cached
    k4.require(type(value) is int and lower <= value <= upper,
               f"Unexpected cached prompt rows {value}, expected {lower}..{upper}")
    lane, body, case = item
    return lane, body, {**case, "cached_tokens": value}


def capture(url, engine, label, directory, lanes, *, reset, lower=0, upper=0, trace=None):
    receipt = cold_cache.before_cohort(url, engine) if reset else None
    before = receipt["cleared"] if receipt else cold_cache.idle_metrics(url, engine)
    for lane, body, _ in lanes:
        (directory / f'{label}-{lane["lane"]:02d}.request.json').write_bytes(body)
    if trace is not None:
        trace.seek(0, SEEK_END)
    raw = client.capture_cohort(url, lanes, label, directory)
    try:
        if receipt is not None:
            raw["cold_cache"] = receipt
            cold_cache.after_cohort(url, engine, receipt)
            after = receipt["after"]
        else:
            after = cold_cache.idle_metrics(url, engine)
        raw.update(metrics_before=before, metrics_after=after)
    finally:
        with gzip.open(directory / f"{label}.json.gz", "wt", compresslevel=3) as stream:
            json.dump(raw, stream, separators=(",", ":"), allow_nan=False)
    cases = [cached_case(record, item, lower, upper)
             for record, item in zip(raw["requests"], lanes, strict=True)]
    report = client.replay_cohort(raw, cases)
    if receipt is not None:
        cold_cache.replay(raw, engine)
    metrics = [cold_cache.metrics(snapshot, engine) for snapshot in (before, after)]
    if engine == "vllm" and lower:
        for moment, snapshot in (("before", before), ("after", after)):
            (directory / f"metrics-{moment}.txt").write_text(snapshot)
        report["mtp_activity"] = k4.read_activity(directory, engine)
    if trace is not None:
        warm_trace = trace.read()
        (directory / "native-warm.log").write_bytes(warm_trace)
        report["mtp_activity"] = k4.native_activity(warm_trace.decode(), require_groups=True)
        draft_rows = metrics[1][DRAFT_CACHED_ROWS] - metrics[0][DRAFT_CACHED_ROWS]
        k4.require(draft_rows > 0, "No native draft prefix rows restored for warm branches")
        report["draft_cached_rows_delta"] = draft_rows
    report["cached_tokens"] = [case["cached_tokens"] for _, _, case in cases]
    report["prefix_hits_delta"] = metrics[1][cold_cache.HITS] - metrics[0][cold_cache.HITS]
    k4.require(report["prefix_hits_delta"] > 0 if lower else report["prefix_hits_delta"] == 0,
               f"Unexpected prefix hit counter for {label}")
    return report


def run(engine, endpoint, root, directory, *, native_trace=None):
    url = urlsplit(endpoint)
    k4.require(url.scheme == "http" and url.hostname == "127.0.0.1" and url.port
               and url.path in ("", "/") and not (url.query or url.fragment or url.username),
               "Expected a loopback HTTP origin")
    k4.require((engine == "native") == (native_trace is not None),
               "Native reuse requires the dedicated server trace log")
    original, prime, branches = workload(root)
    directory.mkdir()
    cold = capture(url, engine, "cold-W3", directory, original, reset=True)
    first = capture(url, engine, "prime", directory, prime, reset=True)
    if native_trace is None:
        warm = capture(url, engine, "warm-W3-branches", directory, branches, reset=False,
                       lower=1, upper=COMMON_PREFIX)
    else:
        with native_trace.open("rb") as trace:
            warm = capture(url, engine, "warm-W3-branches", directory, branches, reset=False,
                           lower=COMMON_PREFIX, upper=COMMON_PREFIX, trace=trace)
    control = capture(url, engine, "prime-after-reset", directory, prime, reset=True)
    k4.require(first["requests"][0]["output_ids"] == control["requests"][0]["output_ids"],
               "Prime changed output after clearing prefix state")
    summary = {"protocol": "matched-k4-prefix-reuse-v2", "engine": engine,
               "common_prefix_tokens": COMMON_PREFIX,
               "cold-W3": cold, "prime": first, "warm-W3-branches": warm,
               "prime-after-reset": control}
    k4.save(directory / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("engine", choices=("native", "vllm"))
    parser.add_argument("endpoint")
    parser.add_argument("fixtures", type=Path)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--native-trace", type=Path, help="Native --trace-batches server log")
    args = parser.parse_args()
    run(args.engine, args.endpoint, args.fixtures, args.directory, native_trace=args.native_trace)


if __name__ == "__main__":
    main()
