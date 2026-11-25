"""Capture or independently replay one frozen TP2 row. Use the paired driver's PYTHONPATH."""

import argparse
import gzip
import json
from pathlib import Path
from unittest.mock import patch
from urllib.parse import urlsplit

from benchmarks.concurrent import client, reference
from benchmarks.streaming.client import StreamingConnection
import cache_activity
import cold_cache
import k4
import lifecycle
from makespan import compare, K4_MAKESPAN_CRITERION, validate_cohort
from saved_evidence import digest, load, require
from tp2_corpus import load_lanes
from workload_suite import ROWS


def binding(run):
    return {"ready_sha256": digest(run / "ready.json"),
            "campaign_sha256": digest(run.parent.parent / "identity.json"),
            "arm": run.name, "row": run.parent.name}


def handle(run, fixtures, endpoint=None, *, probe=False):
    row = "W1" if probe else run.parent.name
    lanes = load_lanes(fixtures, row)
    engine = "vllm" if run.name == "vllm-k4" else cold_cache.LMCACHE_DRAIN
    require(run.name in ("vllm-k4", "native-k4", "native-k4-trace"), "Unknown TP2 arm")
    policy = reference.reference()["policy"]
    measured = {}
    previous = prior_ranks = warmup = None
    tags = ["trace"] if run.name == "native-k4-trace" else ["warmup-01", "measured-01"]
    if probe:
        identity = load(run.parent.parent / "identity.json")
        require(run.name == "native-k4" and identity["mode"] == "qualify",
                "Functional probes require a separate native qualification arm")
        lanes = lanes[:1]
        tags = ["probe"]
    if not endpoint:
        require({p.name for p in run.glob("*.json.gz")} == {f"{tag}.json.gz" for tag in tags},
                "Unexpected or missing frozen cohort")
    for tag in tags:
        path = run / f"{tag}.json.gz"
        if endpoint:
            url = urlsplit(endpoint)
            require(url.hostname == "127.0.0.1" and url.scheme == "http" and url.port,
                    "Expected loopback endpoint")
            receipt = cold_cache.before_cohort(url, engine)
            with patch.object(client.http.client, "HTTPConnection", StreamingConnection):
                raw = client.capture_cohort(url, row, lanes, tag, policy)
            raw.update(binding=binding(run), cold_cache=receipt)
            try:
                cold_cache.after_cohort(url, engine, receipt)
            finally:
                with gzip.open(path, "xt", compresslevel=3) as stream:
                    json.dump(raw, stream, separators=(",", ":"), allow_nan=False)
        else:
            with gzip.open(path, "rt") as stream:
                raw = json.load(stream)
        require((raw["row"], raw["tag"], raw["binding"]) == (row, tag, binding(run)),
                "Cohort belongs to a different row, arm or campaign")
        if probe:
            require(len(raw["requests"]) == 1 and not raw["errors"], "Functional request failed")
            client.replay_lane(raw["requests"][0], lanes[0])
        else:
            measured[tag] = validate_cohort(raw, lanes, policy)
        previous = cold_cache.replay(raw, engine, previous)
        if engine == cold_cache.LMCACHE_DRAIN:
            prior_ranks = cache_activity.replay(raw["cold_cache"], previous, prior_ranks)
            if probe:
                before = cold_cache.parse_metrics(raw["cold_cache"]["cleared"], prior_ranks)
                return {"request_sha256": lanes[0][0]["request_sha256"], "binding": binding(run),
                        "rank_deltas": {k: value - before[k] for k, value in prior_ranks.items()}}
        if tag == "trace":
            baseline = run.parent / "native-k4" / "warmup-01.json.gz"
            with gzip.open(baseline, "rt") as stream:
                timed_warmup = json.load(stream)
            require(timed_warmup["binding"] == binding(baseline.parent)
                    and timed_warmup["tag"] == "warmup-01", "Wrong trace warmup baseline")
            client.verify_round_ids(raw, [req["output_ids"] for req in timed_warmup["requests"]])
        if tag == "warmup-01":
            warmup = [record["output_ids"] for record in raw["requests"]]
        if tag == "measured-01":
            client.verify_round_ids(raw, warmup)
    return measured


def assess(directory, fixtures):
    row = directory.name
    identity = lifecycle.campaign(directory.parent)
    arms = ["native-k4", "vllm-k4"]
    if row == "W1":
        arms.append("native-k4-trace")
    require(row in ROWS and {p.name for p in directory.iterdir() if p.is_dir()} == set(arms),
            "Wrong campaign row or arm set")
    samples, diagnostics = {}, {}
    for arm in arms:
        run = directory / arm
        lifecycle.verify_arm(run, identity, fixtures)
        require(load(run / "completed.json") == {"binding": binding(run), "retired": True},
                "Arm lifecycle binding mismatch")
        data = handle(run, fixtures)
        if arm == "native-k4-trace":
            k4.read_activity(run, "native", require_groups=True)
            continue
        if arm == "vllm-k4":
            k4.read_activity(run, "vllm")
        result = data["measured-01"]
        samples[arm] = {row: [result["makespan_ns"]]}
        diagnostics[arm] = result | {"all_lanes_generating_before_first_done":
                                     result["all_generating_overlap_ms"] > 0}
    return compare(samples["native-k4"], samples["vllm-k4"], True,
                   criterion=K4_MAKESPAN_CRITERION) | {"diagnostics": diagnostics}


def comparison(base, fixtures):
    rows = {}
    for name in ROWS:
        row = assess(base / name, fixtures)
        require(row == load(base / name / "comparison.json"), f"Changed {name} row verdict")
        rows[name] = row["workloads"][name]
    return {"passed": all(row["passed"] for row in rows.values()), "workloads": rows,
            "makespan_criterion": K4_MAKESPAN_CRITERION}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    actions = ("capture", "replay", "assess", "compare", "binding", "probe", "replay-probe")
    parser.add_argument("action", choices=actions)
    parser.add_argument("directory", type=Path)
    parser.add_argument("fixtures", type=Path)
    parser.add_argument("--endpoint")
    opts = parser.parse_args()
    live = opts.action in ("capture", "probe")
    require(bool(opts.endpoint) == live, "A live capture or probe requires an endpoint")
    if opts.action == "compare":
        result = comparison(opts.directory, opts.fixtures)
    elif opts.action == "assess":
        result = assess(opts.directory, opts.fixtures)
    elif opts.action == "binding":
        result = binding(opts.directory)
    else:
        result = handle(opts.directory, opts.fixtures, opts.endpoint,
                        probe=opts.action in ("probe", "replay-probe"))
    print(json.dumps(result, indent=2))
