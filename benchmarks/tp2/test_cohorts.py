"""Replay the same workload with independent warmup, measured and trace outputs."""

import gzip
import json
from copy import deepcopy

import pytest

import cache_activity
import cold_cache
import cohorts
from benchmarks.concurrent.client import capture_cohort
from benchmarks.concurrent.reference import reference
from benchmarks.paired.test_lmcache_defaults import drain_metric_text
from tests.test_concurrent_baseline import loopback, mutate_payload, wire_fixture


def metrics(rounds):
    values = {cold_cache.STORES: 2, cold_cache.LMCACHE_STORES: 2,
              cold_cache.GQA_WRITTEN_TOKENS: 8, cold_cache.LMCACHE_WRITE_NS: 1}
    for rank in (0, 1):
        for counter in cache_activity.COUNTERS:
            values[cache_activity.rank_name(rank, counter)] = 0
        values[cache_activity.rank_name(rank, "stores")] = 2
        for family in cache_activity.FAMILIES:
            count = int(family != "draft" or rank == 0)
            counter = f"{family}_written_bytes"
            values[cache_activity.rank_name(rank, counter)] = count
            name = f"sglang_cpp:lmcache_{counter}_total"
            values[name] = values.get(name, 0) + count
    return drain_metric_text(**{name: value * rounds for name, value in values.items()})


@pytest.fixture
def timing_row(tmp_path, monkeypatch):
    lanes = wire_fixture()
    with loopback(lanes) as (url, _):
        sample = capture_cohort(url, "W1", lanes, "warmup-01", reference()["policy"])
    monkeypatch.setattr(cohorts, "load_lanes", lambda *_: lanes)
    monkeypatch.setattr(cohorts.lifecycle, "campaign", lambda *_: {})
    monkeypatch.setattr(cohorts.lifecycle, "verify_arm", lambda *_: None)
    monkeypatch.setattr(cohorts.k4, "read_activity", lambda *_, **__: None)
    (tmp_path / "identity.json").write_text("{}")
    row = tmp_path / "W1"
    records = sample["requests"]
    start = min(r["submit_ns"] for r in records)
    end = max(r["done_ns"] for r in records)
    eof = max(r["eof_ns"] for r in records)
    receipt = {"reset": {"status": 200, "body": '{"success":true}'},
               "reset_started_ns": start - 3, "reset_completed_ns": start - 2,
               "ready_ns": start - 1, "capture_finished_ns": eof, "settled_ns": eof + 1}
    for arm_index, arm in enumerate(("native-k4", "vllm-k4", "native-k4-trace")):
        run = row / arm
        run.mkdir(parents=True)
        (run / "ready.json").write_text("{}")
        binding = cohorts.binding(run)
        (run / "completed.json").write_text(json.dumps({"retired": True, "binding": binding}))
        tags = ["trace"] if arm.endswith("-trace") else ["warmup-01", "measured-01"]
        for index, tag in enumerate(tags):
            raw = deepcopy(sample)
            before = metrics(index)
            raw.update(binding=binding, tag=tag, cold_cache=receipt | {
                "before": before, "cleared": before, "after": metrics(index + 1)})
            first = raw["requests"][0]
            ids = [10 * (arm_index + 1) + index, 2]
            mutate_payload(first, 1, "token_ids", ids)
            first["output_ids"][:2] = ids
            (run / f"{tag}.json.gz").write_bytes(gzip.compress(json.dumps(raw).encode()))
    return row, end - start


def test_changed_round_and_trace_outputs_preserve_paired_timing(timing_row):
    row, duration = timing_row
    assert cohorts.handle(row / "native-k4-trace", None)["trace"]["makespan_ns"] == duration
    report = cohorts.assess(row, None)["workloads"]["W1"]
    assert report["native"]["samples_ns"] == report["vllm"]["samples_ns"] == [duration]
    assert report["passed"] and report["native_to_vllm_median_ratio"] == 1.0


def test_independent_trace_replays_without_timed_warmup(timing_row):
    row, duration = timing_row
    (row / "native-k4" / "warmup-01.json.gz").unlink()
    trace = cohorts.handle(row / "native-k4-trace", None)
    assert trace["trace"]["makespan_ns"] == duration


@pytest.mark.parametrize("defect", ["short-output", "missing-done", "wrong-end", "cache-hit"])
def test_timing_still_requires_a_complete_cold_workload(timing_row, defect):
    row, _ = timing_row
    run = row / "vllm-k4"
    path = run / "measured-01.json.gz"
    raw = json.loads(gzip.decompress(path.read_bytes()))
    first = raw["requests"][0]
    if defect == "short-output":
        mutate_payload(first, 1, "token_ids", [1])
        del first["output_ids"][1]
    elif defect == "missing-done":
        first["events"].pop()
    elif defect == "wrong-end":
        first["done_ns"] += 1
    else:
        after = raw["cold_cache"]["after"]
        raw["cold_cache"]["after"] = after.replace("hits_total 0", "hits_total 1")
    path.write_bytes(gzip.compress(json.dumps(raw).encode()))
    with pytest.raises(ValueError):
        cohorts.handle(run, None)
