"""Verify saved benchmark identity before launch, then label completed comparisons."""

import argparse
import gzip
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from benchmarks.concurrent import client as concurrent
from benchmarks.concurrent.reference import ROWS, load_lanes
from benchmarks.streaming import client as streaming
from makespan import cohort_makespan_ns, compare, validate_cohort
from saved_evidence import (
    arg, artifact_hashes, command, device, digest, fields, load, observed_device, require,
    saved_worker,
)


COLLECTOR = "measure_concurrent.py"
PAIRED_COLLECTOR = "makespan.py"
LEGACY_COLLECTOR = "00d702fb1efe7749a3bb0b03aa1fd61169d248114283b32145c5304d87e51c05"
LEGACY_STREAMING_CLIENT = "c8fdfde28f6ce35aaea79fff4c2cac2d386d7be2c611ee8e3660ae11b803afcd"
DRAINING_STREAMING_CLIENT = "160ec28c95941f3a5d55de4ced6f2470bfc80f9ca0e290dcd5a167fafa5295d9"
STREAMING = ("streaming/client.py", "streaming/W1.json", "streaming/W2.json")
CONCURRENT = ("streaming/client.py", "concurrent/client.py", "concurrent/reference.py",
              "concurrent/fixture-manifest.json", "concurrent/gpu2-reference.json", COLLECTOR)
FROZEN_PROTOCOL = dict(zip(CONCURRENT, (
    LEGACY_STREAMING_CLIENT,
    "e8f53f54b3135318508eb8052ef015437c508499b3b738632e2fea0e4af66ee8",
    "65aeb33607a74512374c2f4f04fe5d7d408e465762e208613d693b2155e636df",
    "08caa19062d29c896c13e94bda164fbe247daf0735f3a174c6e58e1d2719b44c",
    "8221e70bdda0fd06f8c8a3f3bb4f6fdd274fc1b76380b268b2dcdcd1f7593abe",
    "b9f6699abece4b64de7879880daf42b5172c09f69077467cc386e4a05f963dfe",
), strict=True))
ACTIVE_FROZEN_PROTOCOL = FROZEN_PROTOCOL | {"streaming/client.py": DRAINING_STREAMING_CLIENT}
DEFAULT_TOKEN_BUDGET = 8192
STREAMING_COUNTS = (3, 9)
CONCURRENT_COUNTS = (2, 5)


def token_budget(value):
    require(type(value) is int and 1 <= value <= 32768,
            "Expected --max-num-batched-tokens in [1, 32768]")
    return value


def current_protocol(group, protocol, *, frozen=False):
    paths = STREAMING if group == "W1-W2" else CONCURRENT
    if group != "W1-W2" and not frozen:
        paths += (PAIRED_COLLECTOR,)
    return {name: digest(Path(__file__).with_name(name) if "/" not in name
                         else protocol / "benchmarks" / name) for name in paths}


def stream_identity(protocol, hashes):
    cases = streaming.load_cases(protocol / "benchmarks" / "streaming")
    return {
        "clock": "perf_counter_ns", "schedule": [["warmup", 3], ["measured", 9]],
        "workload_order": ["W1", "W2"], "client_sha256": hashes["streaming/client.py"],
        "fixture_sha256": {name: sha for name, (_, sha) in cases.items()},
        "request_sha256": {name: hashlib.sha256(
            streaming.json_bytes(streaming.request_body(case))).hexdigest()
                           for name, (case, _) in cases.items()},
    }


def summary(run, group, hashes, protocol):
    require((run / "client-exit-code.txt").read_text().strip() == "0",
            f"{run}: client did not exit successfully")
    report = load(run / "measurement" / "summary.json")
    if group == "W1-W2":
        expected = stream_identity(protocol, hashes)
        require(all(report["manifest"][key] == value for key, value in expected.items()),
                f"{run}: streaming client, fixtures, request bytes or schedule differ")
        require(set(report["workloads"]) == {"W1", "W2"} and all(
            row["valid"] is True and row["warmups_successful"] == 3 and row["successful"] == 9
            for row in report["workloads"].values()), f"{run}: incomplete streaming results")
        return report, STREAMING_COUNTS
    baseline = concurrent.reference()
    require(report["status"] == "complete" and report["row"] == group
            and report["fixture_manifest_sha256"] == hashes["concurrent/fixture-manifest.json"]
            and report["baseline_bundle_sha256"] == baseline["archive"]["bundle_sha256"],
            f"{run}: wrong concurrent row, incomplete results or fixture/reference mismatch")
    tags = [outcome["tag"] for outcome in report["request_outcomes"]]
    counts = (sum(tag.startswith("warmup-") for tag in tags),
              sum(tag.startswith("measured-") for tag in tags))
    policy = {"warmup_cohorts": counts[0], "measured_cohorts": counts[1]}
    lanes = ROWS[group][2]
    require(counts[0] >= 1 and counts[1] >= 2 and tags == concurrent.tags(policy) and all(
        outcome["attempted"] == lanes and outcome["successful_http_completions"] == lanes
        and outcome["raw_file"] == outcome["tag"] + ".json.gz" and "error" not in outcome
        and (run / "measurement" / outcome["raw_file"]).is_file()
        for outcome in report["request_outcomes"]), f"{run}: invalid or incomplete cohort schedule")
    return report, counts


def replay_cohorts(run, group, report, lanes, collector):
    policy = concurrent.reference()["policy"]
    paired = collector == PAIRED_COLLECTOR
    validate = validate_cohort if paired else concurrent.validate_cohort
    first_outputs = None
    measured = []
    samples = []
    for outcome in report["request_outcomes"]:
        with gzip.open(run / "measurement" / outcome["raw_file"], "rt") as stream:
            raw = json.load(stream)
        tag = outcome["tag"]
        require(raw["row"] == group and raw["tag"] == tag, f"{run}: wrong raw cohort identity")
        result = validate(raw, lanes, policy)
        if paired:
            require(outcome["makespan_ns"] == result["makespan_ns"],
                    f"{run}: recorded makespan differs from raw cohort replay at {tag}")
        outputs = [request["output_ids"] for request in raw["requests"]]
        if first_outputs is None:
            first_outputs = outputs
        require(outputs == first_outputs, f"{run}: greedy IDs differ across cohorts at {tag}")
        if tag.startswith("measured-"):
            measured.append(result)
            samples.append(cohort_makespan_ns(raw["requests"]))
    require(all(report[key] == value for key, value in concurrent.summarize(measured).items()),
            f"{run}: summary metrics differ from raw cohort replay")
    return {group: samples}


def verify_stream_end(record):
    end = record.get("response_end", {})
    require(isinstance(end, dict)
            and type(end.get("ns")) is int and end["ns"] >= record["events"][-1]["ns"]
            and type(end.get("body_bytes")) is int and end["body_bytes"] > 0,
            "Missing or invalid HTTP response end receipt")
    headers = {name.lower(): value for name, value in reversed(record["response_headers"])}
    if headers.get("transfer-encoding", "").lower() == "chunked":
        return
    try:
        length = int(headers.get("content-length", "-1"))
    except ValueError:
        return
    require(length < 0 or length == end["body_bytes"],
            "HTTP response body bytes differ from Content-Length")


def replay_streams(run, report, protocol, freeze_baseline):
    measurement = run / "measurement"
    cases = streaming.load_cases(protocol / "benchmarks" / "streaming")
    names = report["manifest"]["workload_order"]
    requests = report["manifest"]["request_sha256"]
    require(all(digest(measurement / f"{name}.request.json") == requests[name] for name in names),
            f"{run}: saved streaming request digest differs")
    results = {name: [] for name in names}
    samples = {name: [] for name in names}
    schedule = [(name, phase, cohort) for phase, count in report["manifest"]["schedule"]
                for cohort in range(1, count + 1) for name in names]
    with (measurement / "events.jsonl").open() as source:
        for line, (name, phase, cohort) in zip(source, schedule, strict=True):
            record = json.loads(line)
            require((record["workload"], record["phase"], record["cohort"])
                    == (name, phase, cohort) and record["http_status"] == 200
                    and record["request_sha256"] == requests[name] and "error" not in record,
                    f"{run}: wrong saved streaming request or schedule")
            replay = {"headers_ns": record["headers_ns"], "submit_ns": record["submit_ns"]}
            completion = streaming.Completion()
            events = (event for event, _ in streaming.completion_frames(record["events"]))
            concurrent.consume_events(events, replay, cases[name][0], completion)
            concurrent.finish_record(replay, completion)
            if report["manifest"]["client_sha256"] != LEGACY_STREAMING_CLIENT:
                verify_stream_end(record)
            require(replay["headers_ns"] >= replay["submit_ns"]
                    and replay["output_ids"] == record["output_ids"]
                    == report["workloads"][name]["greedy_output_ids"]
                    and all(replay["metrics"][key] == record["metrics"][key]
                            for key in concurrent.LATENCIES),
                    f"{run}: streaming record differs from raw wire replay")
            duration = cohort_makespan_ns([replay])
            count = len(completion.ids)
            record["metrics"].update(output_tokens_per_second=count * 1e9 / duration,
                requests_per_second=1e9 / duration,
                aggregate_tokens_per_second=(count + len(completion.prompt)) * 1e9 / duration)
            results[name].append(record)
            if phase == "measured":
                samples[name].append(duration)
    require(all(streaming.summarize(items, freeze_baseline) == report["workloads"][name]
                for name, items in results.items()),
            f"{run}: streaming summary differs from replay")
    return samples


def replay_arm(run, group, cpus, counts, report, protocol, lanes, hashes, freeze_baseline):
    client_args = command(run, "client", cpus)
    if group == "W1-W2":
        require(("--freeze-baseline" in client_args) == freeze_baseline and any(
            item.endswith("/benchmarks/streaming/client.py") for item in client_args),
            f"{run}: unexpected streaming client")
        return replay_streams(run, report, protocol, freeze_baseline)
    collector = PAIRED_COLLECTOR if PAIRED_COLLECTOR in hashes else COLLECTOR
    require(any(Path(item).name == collector and client_args[index + 1] == group
                for index, item in enumerate(client_args[:-1]))
            and (int(arg(client_args, "--warmups")), int(arg(client_args, "--measured")))
            == counts, f"{run}: client command differs from reported cohort schedule")
    return replay_cohorts(run, group, report, lanes, collector)


def saved_arm(args, group, hashes, lanes, native_device, native_cpus):
    capture = args.saved / f"vllm-{group}"
    run = capture / group
    baseline = concurrent.reference()
    observed = device(capture)
    require(observed == native_device,
            f"{run}: saved device {observed} differs from {native_device}")
    cpus = fields(capture / "cpus.txt")
    require(cpus == native_cpus, f"{run}: CPU/NUMA placement differs from candidate")
    source_receipt = capture / "vllm-source-commit.txt"
    require(source_receipt.is_file(), f"{run}: missing vLLM source commit receipt")
    source_commit = source_receipt.read_text().strip()
    require(source_commit == baseline["vllm_pin"], f"{run}: unexpected vLLM source commit")
    arm = saved_worker(capture, run, cpus, observed["uuid"], group)
    worker = arm["effective"]
    pinned = baseline["observed_vllm_settings"]
    require(Path(arm["model_path"]).is_absolute()
            and Path(arm["model_path"]).name == baseline["model"],
            f"{run}: unexpected saved model path")
    require(all(worker[key] == pinned[key] for key in ("vllm", "torch", "torch_cuda")),
            f"{run}: unexpected vLLM runtime identity")
    require(worker["compilation"] == pinned["compilation"],
            f"{run}: effective vLLM graph configuration differs")
    require(worker["scheduler"]["max_num_batched_tokens"] == args.max_num_batched_tokens,
            f"{run}: saved vLLM scheduler cap differs from {args.max_num_batched_tokens}")
    recorded = artifact_hashes(capture / "protocol.sha256")
    manifest = "concurrent/fixture-manifest.json"
    reconstructed = []
    if "max_num_scheduled_tokens_effective" not in worker["scheduler"]:
        reconstructed.append("scheduled token cap from pinned scheduler fallback")
    if group == "W1-W2":
        if recorded.get("streaming/client.py") == LEGACY_STREAMING_CLIENT:
            require(args.max_num_batched_tokens != DEFAULT_TOKEN_BUDGET,
                    f"{run}: formal W1/W2 needs a fresh capture through response end")
            hashes = hashes | {"streaming/client.py": LEGACY_STREAMING_CLIENT}
            reconstructed.append("legacy streaming capture ends at first [DONE]")
    else:
        if PAIRED_COLLECTOR not in recorded:
            require(all(hashes[name] == sha for name, sha in ACTIVE_FROZEN_PROTOCOL.items()),
                    f"{run}: active sources differ from the historical collector protocol")
            hashes = FROZEN_PROTOCOL
    report, counts = summary(run, group, hashes, args.protocol)
    if group != "W1-W2":
        if manifest not in recorded:
            recorded[manifest] = report["fixture_manifest_sha256"]
            reconstructed.append("fixture manifest from summary and raw request hashes")
        if recorded.get(COLLECTOR) == LEGACY_COLLECTOR:
            reconstructed.append("legacy collector repeatability from raw cohorts")
    for name, expected in hashes.items():
        acceptable = {expected}
        if name == COLLECTOR and expected == FROZEN_PROTOCOL[COLLECTOR]:
            acceptable.add(LEGACY_COLLECTOR)
        require(recorded.get(name) in acceptable, f"{run}: protocol digest differs for {name}")
    samples = replay_arm(run, group, cpus, counts, report, args.protocol, lanes, hashes,
                         freeze_baseline=True)
    arm.update(device=observed, cpus=cpus, counts=counts, source_commit=source_commit,
               summary_sha256=digest(
        run / "measurement" / "summary.json"), protocol={name: recorded[name] for name in hashes},
        reconstructed=reconstructed, makespan_samples_ns=samples)
    return arm


def frozen_arm(hashes):
    require(hashes == ACTIVE_FROZEN_PROTOCOL,
            "Active concurrent protocol differs from frozen publication")
    reference = concurrent.reference()
    policy = reference["policy"]
    return {"device": {"uuid": reference["physical_gpu_uuid"].lower(),
                       "driver_version": reference["host_driver"]},
            "model_path": reference["model"],
            "effective": reference["observed_vllm_settings"],
            "counts": [policy["warmup_cohorts"], policy["measured_cohorts"]],
            "archive": reference["archive"], "protocol": FROZEN_PROTOCOL}


def comparison_label(mode, group, arm, hashes, native_device, counts, budget):
    if mode == "frozen":
        require(arm == frozen_arm(hashes) and tuple(arm["counts"]) == tuple(counts),
                "Frozen comparison arm differs from the published reference")
        differences = []
        if arm["device"]["uuid"] != native_device["uuid"]:
            differences.append("cross-device")
        if arm["effective"]["config"]["max_num_batched_tokens"] != budget:
            differences.append("cross-cap")
        label = "published frozen reference"
        return label + (" (" + ", ".join(differences) + ")" if differences else "")
    scheduler = arm["effective"]["scheduler"]
    require(arm["source_commit"] == concurrent.reference()["vllm_pin"],
            "Matched comparison requires the pinned vLLM source commit")
    require("archive" not in arm and arm["device"] == native_device
            and scheduler["max_num_batched_tokens"] == budget,
            "Matched comparison requires a same-device saved arm with the native scheduler cap")
    if tuple(arm["counts"]) != tuple(counts):
        return "exploratory (count mismatch)"
    if budget != DEFAULT_TOKEN_BUDGET:
        return f"exploratory (non-protocol cap {budget})"
    required_counts = STREAMING_COUNTS if group == "W1-W2" else CONCURRENT_COUNTS
    if tuple(counts) != required_counts:
        return "exploratory (non-protocol counts)"
    require(scheduler["max_num_scheduled_tokens_configured"] == budget
            and scheduler["max_num_scheduled_tokens_effective"] == budget
            and scheduler["async_scheduling"] is True,
            "Formal comparison requires observed 8192-token vLLM scheduling")
    return "same-device matched-cap"


def plan(args):
    budget = token_budget(args.max_num_batched_tokens)
    require((args.mode == "matched") == (args.saved is not None),
            "Use matched mode with saved vLLM results or frozen mode with the published reference")
    require(args.mode != "frozen" or "W1-W2" not in args.groups,
            "The frozen GPU2 concurrent reference covers W3–W6")
    native_device = device(args.output)
    cpus = fields(args.output / "cpus.txt")
    baseline = concurrent.reference()
    checkpoint_hash = digest(args.checkpoint_manifest)
    require(checkpoint_hash == baseline["checkpoint_manifest_sha256"],
            "Checkpoint manifest differs from the published model")
    if args.mode == "matched":
        saved_manifest = args.saved / "checkpoint.sha256"
        if not saved_manifest.is_file():
            saved_manifest = args.saved / "meta" / "checkpoint.sha256"
        require(digest(saved_manifest) == checkpoint_hash,
                "Saved checkpoint manifest differs from the candidate model")
    rows = {}
    checkpoints = {args.checkpoint.resolve()}
    for group in args.groups:
        hashes = current_protocol(group, args.protocol, frozen=args.mode == "frozen")
        lanes = None if group == "W1-W2" else load_lanes(args.fixtures, group)
        expected = STREAMING_COUNTS if group == "W1-W2" else (args.warmups, args.measured)
        if args.mode == "matched":
            arm = saved_arm(args, group, hashes, lanes, native_device, cpus)
            checkpoints.add(Path(arm["model_path"]).resolve())
        else:
            arm = frozen_arm(hashes)
        matched_counts = tuple(arm["counts"]) == expected
        require(matched_counts or (args.mode == "matched" and args.allow_count_mismatch),
                f"{group}: saved warmup/measured counts {arm['counts']} differ from {expected}")
        label = comparison_label(args.mode, group, arm, hashes, native_device, expected, budget)
        rows[group] = {"planned_comparison": label, "vllm": arm,
                       "protocol": hashes, "counts": expected}
    for checkpoint in sorted(checkpoints):
        subprocess.run(["sha256sum", "--check", "--strict", "--quiet",
                        str(args.checkpoint_manifest.resolve())], cwd=checkpoint, check=True)
    return {"status": "planned", "mode": args.mode, "device": native_device, "cpus": cpus,
            "checkpoint": str(args.checkpoint.resolve()), "max_num_batched_tokens": budget,
            "fixtures": str(args.fixtures.resolve()),
            "checkpoint_manifest_sha256": checkpoint_hash, "groups": rows}


def finish(args):
    plan_bytes = (args.output / "comparison-plan.json").read_bytes()
    require(hashlib.sha256(plan_bytes).hexdigest() == args.plan_sha256,
            "Comparison plan changed after preflight")
    planned = json.loads(plan_bytes)
    require(planned["status"] == "planned" and planned["mode"] == args.mode,
            "Comparison plan mode differs from the requested comparison")
    budget = token_budget(planned["max_num_batched_tokens"])
    row = planned["groups"][args.group]
    run = args.output / args.group
    require(device(args.output) == planned["device"]
            and fields(args.output / "cpus.txt") == planned["cpus"],
            f"{run}: native device or placement receipts changed after preflight")
    pid = (run / "server-pid.txt").read_text().strip()
    require(pid.isdecimal(), f"{run}: invalid native process ID")
    observed_device(run, pid, planned["device"]["uuid"])
    launch = command(run, "server", planned["cpus"])
    health = load(run / "health.json")
    require(health["max_active"] == 32 and health["max_num_batched_tokens"] == budget,
            f"{run}: effective native scheduler settings differ")
    settings = {"device": "0", "max-active": "32", "max-num-batched-tokens": str(budget),
                "fp8-prefill": "tensor-core", "prefix-cache-entries": "0", "mtp-steps": "0"}
    require(Path(arg(launch, "--checkpoint")).resolve() == Path(planned["checkpoint"])
            and all(arg(launch, "--" + name) == value for name, value in settings.items())
            and "--batch-cuda-graphs" in launch
            and not {"--eager", "--mtp-k", "--trace-batches"}.intersection(launch),
            f"{run}: native command differs from planned settings")
    environment = fields(args.output / "native-environment.txt")
    require(environment["CUDA_VISIBLE_DEVICES"].lower() == planned["device"]["uuid"]
            and not any(key.startswith("SGLANG_") for key in environment),
            f"{run}: unexpected CUDA visibility or native route override")
    hashes = current_protocol(args.group, args.protocol, frozen=args.mode == "frozen")
    require(hashes == row["protocol"], f"{run}: candidate protocol changed during the run")
    label = comparison_label(args.mode, args.group, row["vllm"], hashes, planned["device"],
                             row["counts"], budget)
    require(label == row["planned_comparison"], f"{run}: comparison label disagrees with the arms")
    recorded = artifact_hashes(args.output / "protocol.sha256")
    require(all(recorded.get(name) == sha for name, sha in hashes.items()),
            f"{run}: native protocol receipt differs from the plan")
    report, counts = summary(run, args.group, hashes, args.protocol)
    require(counts == tuple(row["counts"]),
            f"{run}: native measurement differs from planned counts")
    lanes = (None if args.group == "W1-W2"
             else load_lanes(Path(planned["fixtures"]), args.group))
    samples = replay_arm(run, args.group, planned["cpus"], counts, report, args.protocol, lanes,
                         hashes, freeze_baseline=False)
    performance = compare(samples, row["vllm"].get("makespan_samples_ns", {}),
                          eligible=label == "same-device matched-cap")
    return {"comparison": label, "mode": args.mode, "group": args.group,
            "comparison_plan_sha256": args.plan_sha256, "vllm": row["vllm"],
            "makespan": performance,
            "quality": {"native_wire_and_repeat_ids_replayed": True,
                        "saved_vllm_wire_and_repeat_ids_replayed": args.mode == "matched",
                        "cross_engine_ids_compared": False},
            "checkpoint_manifest_sha256": planned["checkpoint_manifest_sha256"],
            "protocol": hashes, "native": {"device": planned["device"], "cpus": planned["cpus"],
                "worker_pid": int(pid), "effective_scheduler": health, "launch_settings": settings,
                "batch_cuda_graphs_requested": True, "counts": counts,
                "summary_sha256": digest(run / "measurement" / "summary.json")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("protocol", type=Path)
    parser.add_argument("output", type=Path)
    modes = parser.add_subparsers(dest="action", required=True)
    preflight = modes.add_parser("plan")
    preflight.add_argument("mode", choices=("matched", "frozen"))
    preflight.add_argument("checkpoint", type=Path)
    preflight.add_argument("checkpoint_manifest", type=Path)
    preflight.add_argument("fixtures", type=Path)
    preflight.add_argument("warmups", type=int)
    preflight.add_argument("measured", type=int)
    preflight.add_argument("groups", nargs="+", choices=["W1-W2", *ROWS])
    preflight.add_argument("--saved", type=Path)
    preflight.add_argument("--max-num-batched-tokens", type=int, default=DEFAULT_TOKEN_BUDGET)
    preflight.add_argument("--allow-count-mismatch", action="store_true")
    complete = modes.add_parser("finish")
    complete.add_argument("mode", choices=("matched", "frozen"))
    complete.add_argument("group", choices=["W1-W2", *ROWS])
    complete.add_argument("plan_sha256")
    args = parser.parse_args()
    try:
        result = plan(args) if args.action == "plan" else finish(args)
    except (OSError, ValueError, TypeError, KeyError, IndexError, subprocess.CalledProcessError) \
            as error:
        print(f"Paired identity {args.action} failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, allow_nan=False))
    return 3 if args.action == "finish" and result["makespan"]["passed"] is False else 0


if __name__ == "__main__":
    sys.exit(main())
