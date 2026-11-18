"""Replay the frozen GPU2 archive without a running server or GPU."""

import argparse
import gzip
import json
from pathlib import Path

from benchmarks.concurrent.client import summarize, tags, thresholds, validate_cohort
from benchmarks.concurrent.profile import verify_profile
from benchmarks.concurrent.reference import ROWS, digest, goldens, load_lanes, reference


def checked_bytes(root, path, identity):
    data = (root / path).read_bytes()
    if digest(data) != identity:
        raise ValueError(f"{path}: evidence digest mismatch")
    return data


def archive_manifest(root, baseline):
    archive = baseline["archive"]
    checked_bytes(root, "run-summary.json", archive["run_summary_sha256"])
    data = checked_bytes(root, "files.sha256", archive["files_manifest_sha256"])
    return {path: checksum for checksum, path in
            (line.split("  ", 1) for line in data.decode().splitlines())}


def verify_greedy_ids(requests, expected):
    for request, output in zip(requests, expected, strict=True):
        actual = request["output_ids"]
        if actual != output:
            position = next((index for index, (left, right) in enumerate(zip(actual, output))
                             if left != right), min(len(actual), len(output)))
            raise ValueError(f'Lane {request["lane"]} greedy mismatch at output index {position}')


def replay_cohort(root, files, name, tag, lanes, expected, policy):
    path = f"{name}/{tag}.json.gz"
    payload = json.loads(gzip.decompress(checked_bytes(root, path, files[path])))
    if (payload["row"], payload["tag"]) != (name, tag):
        raise ValueError(f"{path}: cohort identity changed")
    result = validate_cohort(payload, lanes, policy)
    verify_greedy_ids(payload["requests"], expected)
    return payload, result


def replay_row(root, files, baseline, name):
    row, policy = baseline["rows"][name], baseline["policy"]
    concurrency = ROWS[name][2]
    lanes, expected = load_lanes(root / "fixtures", name), goldens(name)
    original = json.loads(checked_bytes(root, row["per_lane_output_ids_path"],
                                        row["per_lane_output_ids_sha256"]))
    if isinstance(original, dict):
        original = [original[str(index)] for index in range(concurrency)]
    if original != expected:
        raise ValueError(f"{name}: compact goldens differ from original IDs")
    measured = []
    for tag in tags(policy):
        _, report = replay_cohort(root, files, name, tag, lanes, expected, policy)
        if tag.startswith("measured"):
            measured.append(report)
    stats = summarize(measured)
    compact = {group: {key: {field: value for field, value in described.items()
                             if field != "samples"} for key, described in metrics.items()}
               for group, metrics in stats.items()}
    if any(values != row[group] for group, values in compact.items()):
        raise ValueError(f"{name}: replayed statistics differ from frozen evidence")
    if thresholds(stats) != row["thresholds"]:
        raise ValueError(f"{name}: replayed gates differ from frozen evidence")
    diagnostic, _ = replay_cohort(root, files, name, "diagnostic", lanes, expected, policy)
    spans = diagnostic["instrumentation"]
    budget = baseline["observed_vllm_settings"]["config"]["max_num_batched_tokens"]
    if set(spans) != {"prefill", "all_lanes_emitting_ns", "decode"} or not (
            spans["prefill"]["ack_time_ns"] < min(r["submit_ns"] for r in diagnostic["requests"])
            <= max(r["first_ns"] for r in diagnostic["requests"]) <= spans["all_lanes_emitting_ns"]
            < spans["decode"]["ack_time_ns"] < min(r["done_ns"] for r in diagnostic["requests"])):
        raise ValueError(f"{name}: diagnostic profiles were collected outside their cohort windows")
    for phase in ("prefill", "decode"):
        trace = row["route_profiles"][phase]
        prefix = f"{name}-diagnostic-{phase}"
        if spans[phase]["prefix"] != prefix or not Path(trace["trace"]).name.startswith(prefix):
            raise ValueError(f"{name}: {phase} profile is from a different cohort")
        verify_profile(root, trace, phase, concurrency, budget)
    return len(measured) * concurrency


def replay_exploration(root, files, baseline):
    record = baseline["explored_w3_v1"]
    prefix = "exploratory-v1"
    checked_bytes(root, f"{prefix}/policy.json", record["policy_sha256"])
    checked_bytes(root, f"{prefix}/disposition.txt", record["disposition_sha256"])
    warmups = []
    for index in range(1, record["warmup_cohorts_completed"] + 1):
        path = f"{prefix}/W3/warmup-{index:02d}.json.gz"
        data = checked_bytes(root, path, files[path])
        warmups.append(json.loads(gzip.decompress(data)))
    first, second = warmups
    if any(row["errors"] for row in warmups):
        raise ValueError("Exploratory W3 warmup transport failed")
    matches = 0
    for a, b in zip(first["requests"], second["requests"], strict=True):
        left, right = a["output_ids"], b["output_ids"]
        differences = [index for index, (actual, other) in
                       enumerate(zip(left, right, strict=True)) if actual != other]
        if not differences:
            matches += 1
        elif (a["lane"] != b["lane"] or a["lane"] != record["lane"]
              or len(left) != record["total_positions"]
              or len(differences) != record["different_positions"]
              or differences[0] != record["first_differing_zero_based_index"]
              or [left[differences[0]], right[differences[0]]]
              != record["ids_at_first_difference"]):
            raise ValueError("Exploratory W3 divergence differs from frozen evidence")
    if matches != record["matching_lanes"]:
        raise ValueError("Exploratory W3 matching lane count changed")


def main(root, rows):
    baseline = reference()
    files = archive_manifest(root, baseline)
    if "W3" in rows:
        replay_exploration(root, files, baseline)
        print("Exploratory W3 divergence verified")
    for name in rows:
        count = replay_row(root, files, baseline, name)
        print(f"{name}: {count} timed requests, exact IDs, stats, gates, and dispatch verified")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive_root", type=Path)
    parser.add_argument("--row", action="append", choices=ROWS)
    args = parser.parse_args()
    main(args.archive_root, args.row or ROWS)
