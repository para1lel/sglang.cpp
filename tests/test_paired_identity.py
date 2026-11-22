"""Check paired-control receipts offline using the fixed W1 and W2 requests."""

import gzip
import hashlib
import json
import os
import runpy
import shlex
import subprocess
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks" / "paired"))
import identity
import k4


GPU = "GPU-7e25f95a-29b7-9f94-ce9a-2a2871cf7c0c"
OTHER_GPU = "GPU-3c08cbc3-d8b3-c723-c086-a775f6a17f2a"
PLACEMENT = ["numactl", "--localalloc", "taskset", "--cpu-list"]
GROUP = "W1-W2"


def write_json(path, value):
    path.write_text(json.dumps(value) + "\n")


def write_command(run, role, cpus, args):
    (run / f"{role}-command.sh").write_text(shlex.join([*PLACEMENT, cpus, *args]))


def write_receipts(capture, index, uuid):
    (capture / "gpu.txt").write_text(f"{index} {uuid}\n")
    (capture / "cpus.txt").write_text("server=24-33\nclient=34-35\nmemory=localalloc\n")
    (capture / "devices.csv").write_text(
        f"index, uuid, name, driver_version\n{index}, {uuid}, NVIDIA B200, 590.48.01\n")


def write_compute(run, uuid, pid):
    for sample in ("ready", "measurement-end"):
        (run / f"compute-{sample}.csv").write_text(f"{uuid}, {pid}, worker, 80000 MiB\n")


def synthetic_cohort():
    """Send both requests before either completes, with no generating overlap."""
    lanes, requests = [], []
    for lane, (submit, sent, headers) in enumerate(((5, 7, 10), (9, 24, 80))):
        case = {"prompt_token_ids": [100 + lane], "output_tokens": 3}
        request = identity.streaming.json_bytes(identity.streaming.request_body(case))
        lane_manifest = {"lane": lane, "request_sha256": hashlib.sha256(request).hexdigest()}
        stamps = tuple(headers + delta for delta in (10, 12, 14, 20))
        raw = stream_record(case, lane_manifest["request_sha256"], "W3", "measured", 1, stamps)
        raw.update(lane=lane, connect_ns=1, submit_ns=submit, sent_ns=sent, headers_ns=headers,
                   eof_ns=stamps[-1] + 1000,
                   response_headers=[["Content-Type", "text/event-stream"]])
        completion = identity.streaming.Completion()
        identity.concurrent.consume_events(raw["events"], raw, case, completion)
        identity.concurrent.finish_record(raw, completion)
        requests.append(raw)
        lanes.append((lane_manifest, request, case))
    return {"tag": "measured-01", "row": "W3", "barrier_ns": 4,
            "instrumentation": {}, "errors": [], "requests": requests}, lanes


def test_paired_concurrent_replays_late_headers_and_first_token(tmp_path):
    raw, lanes = synthetic_cohort()
    policy = identity.concurrent.reference()["policy"]
    result = identity.validate_cohort(raw, lanes, policy)
    assert result["makespan_ns"] == 95
    assert result["all_streams_overlap_ms"] < 0
    assert result["all_generating_overlap_ms"] < 0
    with pytest.raises(ValueError, match="overlap bounds"):
        identity.concurrent.validate_cohort(raw, lanes, policy)

    measurement = tmp_path / "measurement"
    measurement.mkdir()
    measured, outcomes = [], []
    for tag in ("warmup-01", "measured-01", "measured-02"):
        path = f"{tag}.json.gz"
        raw["tag"] = tag
        with gzip.open(measurement / path, "wt") as stream:
            json.dump(raw, stream)
        outcomes.append({"tag": tag, "raw_file": path, "makespan_ns": 95})
        if tag.startswith("measured"):
            measured.append(result)
    report = identity.concurrent.summarize(measured) | {"request_outcomes": outcomes}
    assert identity.replay_cohorts(tmp_path, "W3", report, lanes, identity.PAIRED_COLLECTOR) == {
        "W3": [95, 95]}
    outcomes[-1]["makespan_ns"] += 1
    with pytest.raises(ValueError, match="recorded makespan differs"):
        identity.replay_cohorts(tmp_path, "W3", report, lanes, identity.PAIRED_COLLECTOR)


@pytest.mark.parametrize(("change", "error"), [
    ({"submit_ns": 30, "sent_ns": 31}, "overlap bounds"),
    ({"sent_ns": 30}, "overlap bounds"),
    ({"request_sha256": "wrong"}, "request identity"),
    ({"output_ids": [1, 2, 3]}, "output_ids differs"),
])
def test_paired_concurrent_rejects_serial_submission_or_corrupt_evidence(change, error):
    raw, lanes = synthetic_cohort()
    request = raw["requests"][1]
    request.update(change)
    if "submit_ns" in change:
        request["metrics"]["ttft_ms"] = (request["first_ns"] - request["submit_ns"]) / 1e6
        request["metrics"]["e2e_ms"] = (request["done_ns"] - request["submit_ns"]) / 1e6
    with pytest.raises(ValueError, match=error):
        identity.validate_cohort(raw, lanes, identity.concurrent.reference()["policy"])


def test_paired_collector_keeps_diagnostic_failures_out_of_exit_status(tmp_path, monkeypatch):
    raw, lanes = synthetic_cohort()
    baseline = identity.concurrent.reference()
    baseline["rows"]["W3"]["thresholds"]["strict_median_output_tokens_per_second_gt"] = 1e12
    collector = ROOT / "benchmarks" / "paired" / identity.PAIRED_COLLECTOR
    destination = tmp_path / "measurement"

    def capture(_url, _name, _lanes, tag, _policy):
        return raw | {"tag": tag}

    monkeypatch.setattr(identity.concurrent, "reference", lambda: baseline)
    monkeypatch.setattr(identity.concurrent, "load_lanes", lambda _fixtures, _name: lanes)
    monkeypatch.setattr(identity.concurrent, "capture_cohort", capture)
    monkeypatch.setattr(sys, "argv",
                        [str(collector), "W3", "http://127.0.0.1", str(tmp_path), str(destination)])
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path(str(collector), run_name="__main__")
    assert exit_info.value.code == 0
    report = identity.load(destination / "summary.json")
    assert report["wire_and_timing_gates"]["throughput"] is False
    assert identity.replay_cohorts(
        tmp_path, "W3", report, lanes, identity.PAIRED_COLLECTOR) == {"W3": [95] * 5}


def test_paired_comparison_uses_medians_and_rejects_ties_and_slower_workloads():
    result = identity.compare(
        {"W3": [30, 10, 12, 9, 8], "W4": [12] * 5, "W5": [13] * 5},
        {"W3": [10, 11, 12, 11, 11], "W4": [12] * 5, "W5": [12] * 5},
        eligible=True)
    assert [row["passed"] for row in result["workloads"].values()] == [True, False, False]
    assert result["passed"] is False
    with pytest.raises(ValueError, match="matching workload and measured sample counts"):
        identity.compare({"W3": [10, 11]}, {"W3": [10, 11, 12]}, eligible=True)


@pytest.mark.parametrize(("scenario", "status", "receipt", "next_group"), [
    ("win", 0, True, True),
    ("timing-miss", 1, True, False),
    ("identity-error", 1, False, False),
    ("measurement-error", 1, False, False),
    ("health-error", 1, False, False),
    ("startup-error", 1, False, False),
])
def test_native_driver_stops_at_first_failed_gate(tmp_path, scenario, status, receipt, next_group):
    paired = ROOT / "benchmarks" / "paired"
    driver = tmp_path / "run_native.sh"
    driver.write_bytes((paired / driver.name).read_bytes())
    (tmp_path / "common.sh").write_text(f'source {shlex.quote(str(paired / "common.sh"))}\n' + r"""
prepare_run() { mkdir "$2"; output=$2; gpu_uuid=mocked; }
prepare_fixtures() { :; }
plan_comparison() { comparison_plan_sha256=mocked; }
start_group() {
    if [[ ${1##*/} == W5 ]]; then
        case "$SCENARIO" in
            startup-error) return 1 ;;
            health-error) printf '{}\n' >"$1/health.json"; return ;;
        esac
    fi
    printf '{"max_active":32,"max_num_batched_tokens":%s}\n' "$token_budget" >"$1/health.json"
}
measure_group() { [[ $1 != W5 || $SCENARIO != measurement-error ]]; }
comparison_identity() {
    printf '{"group":"%s"}\n' "$3"
    if [[ $3 == W5 ]]; then
        case "$SCENARIO" in
            timing-miss) return 3 ;;
            identity-error) return 2 ;;
        esac
    fi
}
stop_group() { :; }
record_compute() { : >"$1"; }
""")
    binary = tmp_path / "bin" / "sglang-server"
    binary.parent.mkdir()
    binary.symlink_to("/bin/true")
    output = tmp_path / "output"
    env = dict(os.environ, SCENARIO=scenario, PAIRED_COMPARISON_MODE="matched",
               PAIRED_VLLM_RESULTS=str(tmp_path), PAIRED_MAX_NUM_BATCHED_TOKENS="8192")
    result = subprocess.run(["bash", str(driver), str(tmp_path), "7", str(output), "W5", "W6"],
                            env=env, capture_output=True, text=True)
    assert result.returncode == status, result.stderr
    assert (output / "finished-utc.txt").is_file()
    assert (output / "W5" / "compute-after.csv").is_file()
    assert (output / "W5" / "comparison.json").is_file() == receipt
    assert not (output / "W5" / "comparison.tmp").exists()
    assert (output / "W6").exists() == next_group


def stream_record(case, request_hash, name, phase, cohort, stamps=(20, 50, 60, 70)):
    tokens = list(range(case["output_tokens"]))
    prompt = case["prompt_token_ids"]
    usage = {"prompt_tokens": len(prompt), "completion_tokens": len(tokens),
             "prompt_tokens_details": {"cached_tokens": 0}}
    payloads = [
        {"choices": [{"index": 0, "token_ids": tokens[:1], "prompt_token_ids": prompt}]},
        {"choices": [{"index": 0, "token_ids": tokens[1:], "finish_reason": "length"}]},
        {"choices": [], "usage": usage},
    ]
    data = [json.dumps(payload) for payload in payloads] + ["[DONE]"]
    events = [{"lines": ["data: " + payload], "ns": stamp, "read": index}
              for index, (payload, stamp) in enumerate(zip(data, stamps), start=1)]
    body_bytes = len("".join("data: " + payload + "\n\n" for payload in data).encode())
    first, last, _, duration = stamps
    aggregate_rate = (len(tokens) + len(prompt)) * 1e9 / duration
    return {"workload": name, "phase": phase, "cohort": cohort, "http_status": 200,
            "submit_ns": 0, "headers_ns": 10, "request_sha256": request_hash,
            "response_headers": [["Content-Length", str(body_bytes)]],
            "response_end": {"ns": duration + 1, "body_bytes": body_bytes},
            "events": events, "output_ids": tokens,
            "metrics": {"ttft_ms": first / 1e6,
                        "tpot_ms": (last - first) / (len(tokens) - 1) / 1e6,
                        "e2e_ms": duration / 1e6,
                        "output_tokens_per_second": len(tokens) * 1e9 / duration,
                        "requests_per_second": 1e9 / duration,
                        "aggregate_tokens_per_second": aggregate_rate}}


def write_opt_in_identity(directory, profile, receipt_version=0):
    identity = {
        "protocol": "matched-k4-opt-in-v1",
        "campaign_id": "01234567-89ab-cdef-0123-456789abcdef",
        "native_profile": profile,
        "native_config": {
            "fp8_prefill": "tensor-core-mtp-target", "mtp_int8_head": True, "mtp_steps": 4,
        },
        "native_binary_sha256": "b" * 64,
    }
    if profile == k4.SPLIT_PROFILE:
        identity["native_config"][k4.SPLIT_FIELD] = True
    if profile == k4.LONG_SPLIT_PROFILE:
        identity["native_config"][k4.LONG_SPLIT_FIELD] = True
    if receipt_version:
        identity["replay_receipts"] = receipt_version
        identity.update(server_cpus="24-33", memory="localalloc")
    path = directory / "identity.json"
    path.write_text(json.dumps(identity) + "\n")
    return identity, hashlib.sha256(path.read_bytes()).hexdigest()


def write_opt_in_arm(run, arm, identity):
    if arm == "vllm-k4":
        return
    info = dict(identity["native_config"], batch_cuda_graphs=True, eager=False, device=0,
                host="127.0.0.1", trace_batches=arm == "native-k4-trace",
                max_running_requests=32, max_num_batched_tokens=8192, prefix_cache_entries=0)
    if arm in ("native-k0", "native-exact-k0"):
        info.update(mtp_steps=0, mtp_int8_head=False)
    if arm == "native-exact-k0":
        info["fp8_prefill"] = "exact"
        info.pop(k4.SPLIT_FIELD, None)
        info.pop(k4.LONG_SPLIT_FIELD, None)
    (run / "server-info.json").write_text(json.dumps(info))
    (run / "running-binary.sha256").write_text(
        identity["native_binary_sha256"] + "  /proc/123/exe\n")


def replace_stream_ids(record, offset):
    record["output_ids"] = [token + offset for token in record["output_ids"]]
    for event, tokens in zip(record["events"][:2],
                             (record["output_ids"][:1], record["output_ids"][1:])):
        payload = json.loads(event["lines"][0][6:])
        payload["choices"][0]["token_ids"] = tokens
        event["lines"] = ["data: " + json.dumps(payload)]


def write_k4_row(directory, native_ns=70, fault=None, trailing_line=None, opt_in=False,
                 profile=k4.ORDERED_PROFILE, receipt_version=0, row="W1"):
    (lane, body, case), = k4.fixture(row, directory)
    arms = (*k4.ARM_ORDER, "native-exact-k0") if opt_in else k4.ARM_ORDER
    if row != "W1":
        arms = tuple(arm for arm in arms if arm != "native-k4-trace")
    if opt_in:
        identity, fingerprint = write_opt_in_identity(directory, profile, receipt_version)
    elif receipt_version:
        identity = {"protocol": k4.PROTOCOL, "replay_receipts": receipt_version,
                    "server_cpus": "24-33", "memory": "localalloc",
                    "native_binary_sha256": "b" * 64}
        path = directory / "identity.json"
        write_json(path, identity)
        fingerprint = hashlib.sha256(path.read_bytes()).hexdigest()
        identity["native_config"] = {"fp8_prefill": "exact", "mtp_steps": 4, "mtp_int8_head": False}
    for arm in arms:
        measurement = directory / arm / "measurement"
        measurement.mkdir(parents=True)
        if opt_in or receipt_version:
            write_opt_in_arm(measurement.parent, arm, identity)
    if receipt_version:
        write_k4_runtime_receipts(directory, arms, opt_in=opt_in, receipt_version=receipt_version)
    for arm in arms:
        measurement = directory / arm / "measurement"
        (measurement / "00.request.json").write_bytes(body)
        for index, tag in enumerate(k4.schedule(row, arm)):
            duration = 80 if arm == "vllm-k4" else native_ns
            record = stream_record(case, lane["request_sha256"], row, arm, index,
                                   (20, 50, 60, duration))
            record["response_headers"] = [["Content-Type", "text/event-stream"],
                                          ["Transfer-Encoding", "chunked"]]
            record.update(finish_reason="length", done_ns=duration,
                          usage=json.loads(record["events"][2]["lines"][0][6:])["usage"],
                          token_chunks=[[20, 1], [50, case["output_tokens"] - 1]])
            if arm in ("vllm-k4", "native-exact-k0") or (
                    fault == "native-drift" and arm == "native-k4"):
                replace_stream_ids(record, index + 1)
            if arm == "vllm-k4" and tag == "measured-01":
                if trailing_line is not None:
                    record["events"].append({"lines": [trailing_line], "ns": duration + 1,
                                             "read": record["events"][-1]["read"] + 1})
                if fault == "missing-done":
                    record["events"].pop()
                if fault == "missing-drain":
                    del record["response_end"]
                if fault == "short":
                    payload = json.loads(record["events"][1]["lines"][0][6:])
                    payload["choices"][0]["token_ids"].pop()
                    record["events"][1]["lines"] = ["data: " + json.dumps(payload)]
                if fault == "malformed":
                    record["events"][1]["lines"] = ["data: {not JSON}"]
            raw = {"row": row, "tag": tag, "requests": [record]}
            if opt_in or receipt_version:
                raw["binding"] = {"campaign_sha256": fingerprint, "arm": arm}
            if receipt_version and arm != "vllm-k4":
                argv = (measurement.parent / "live-argv.bin").read_bytes()
                raw["binding"]["native_argv_sha256"] = hashlib.sha256(argv).hexdigest()
                if receipt_version == 2:
                    dispatch = (measurement.parent / "live-dispatch.bin").read_bytes()
                    raw["binding"]["native_dispatch_sha256"] = hashlib.sha256(dispatch).hexdigest()
            with gzip.open(measurement / f"{tag}.json.gz", "wt") as stream:
                json.dump(raw, stream)


def write_k4_runtime_receipts(directory, arms, *, opt_in, receipt_version=1):
    for arm in arms:
        run = directory / arm
        (run / "client-exit-code.txt").write_text("0\n")
        if arm == "vllm-k4":
            for tag, value in (("before", 17), ("after", 33)):
                (run / f"metrics-{tag}.txt").write_text(
                    f'vllm:spec_decode_num_draft_tokens_total{{model="benchmark"}} {value}\n'
                    f'vllm:spec_decode_num_verified_tokens_per_pos_total{{position="3"}} {value}\n')
            k4.activity(run, "vllm")
            continue
        info_path = run / "server-info.json"
        info = json.loads(info_path.read_bytes())
        info.update(model_path="/model", port=17424)
        write_json(info_path, info)
        write_json(run / "server-check.txt", True)
        (run / "server-pid.txt").write_text("123\n")
        args = ["/bin/sglang-server", "--checkpoint", "/model", "--device", "0",
                "--port", "17424", "--batch-cuda-graphs", "--prefix-cache-entries", "0",
                "--fp8-prefill", info["fp8_prefill"],
                "--max-active", "32", "--max-num-batched-tokens", "8192"]
        if arm == "native-k4-trace":
            args += ["--trace-batches"]
            (run / "server.log").write_text(
                "batch verify lanes=1 rows=5 peak_request_bytes=4096\n"
                "batch mtp slot=0 drafted=4 accepted=2 verified=5 replayed=3 boundary=0\n")
            k4.activity(run, "native")
        if opt_in or info["mtp_steps"] == 0:
            args += ["--mtp-steps", str(info["mtp_steps"])]
        if info["mtp_int8_head"]:
            args += ["--mtp-int8-head"]
        env = ["env", "CUDA_CACHE_PATH=/archive/cuda-cache"]
        dispatch = ""
        if info.get(k4.SPLIT_FIELD):
            dispatch = "SGLANG_EXPERIMENTAL_TARGET_SPLIT_GQA=1"
        if info.get(k4.LONG_SPLIT_FIELD):
            dispatch = "SGLANG_EXPERIMENTAL_TARGET_SPLIT_GQA=long"
        if dispatch:
            env.append(dispatch)
        write_command(run, "server", "24-33", [*env, *args])
        (run / "live-argv.bin").write_bytes(b"\0".join(map(os.fsencode, args)) + b"\0")
        if receipt_version == 2:
            for name in ("live-dispatch.bin", "live-dispatch-end.bin"):
                (run / name).write_bytes(dispatch.encode() + b"\0" if dispatch else b"")


@pytest.mark.parametrize(("file", "old", "new", "message"), [
    ("native-k4/live-argv.bin", b"--mtp-steps\0" b"4\0", b"--mtp-steps\0" b"0\0",
     "live process arguments differ"),
    ("native-k4/server-command.sh", b"--mtp-steps 4", b"--mtp-steps 0",
     "command differs from effective native profile"),
    ("native-k4/server-info.json", b'"port"', b'"experimental_long_target_split_gqa": true, "port"',
     "wrong native server configuration"),
    ("native-k4/server-check.txt", b"true", b"false", "server configuration check failed"),
    ("native-k4-trace/server.log", b"drafted=4", b"drafted=3", "Invalid native MTP round"),
    ("native-k4-trace/server.log", b"verified=5", b"verified=4", "Incomplete verification group"),
    ("native-k4-trace/mtp-activity.json", b'"depth_four_rounds": 1',
     b'"depth_four_rounds": 2', "saved MTP activity differs from raw"),
    ("vllm-k4/metrics-after.txt", b'position="3"} 33', b'position="3"} 17',
     "No fresh vLLM K4 activity"),
    ("native-k4/client-exit-code.txt", b"0", b"2", "client did not succeed"),
    ("exit-code.txt", b"0", b"1", "Saved row exit status differs from replay"),
    ("comparison.json", b'"passed": true', b'"passed": false', "Saved comparison differs"),
])
def test_k4_replay_rejects_contradictory_saved_receipts(tmp_path, file, old, new, message):
    write_k4_row(tmp_path, opt_in=True, receipt_version=1)
    result = k4.assess("W1", tmp_path, tmp_path)
    write_json(tmp_path / "comparison.json", result)
    (tmp_path / "exit-code.txt").write_text("0\n")
    target = tmp_path / file
    original = target.read_bytes()
    assert old in original
    target.write_bytes(original.replace(old, new, 1))
    with pytest.raises(ValueError, match=message):
        k4.assess("W1", tmp_path, tmp_path)


@pytest.mark.parametrize("opt_in", [False, True])
def test_k4_receipt_version_requires_argv_and_activity(tmp_path, opt_in):
    write_k4_row(tmp_path, opt_in=opt_in, profile=k4.SPLIT_PROFILE, receipt_version=1)
    assert k4.assess("W1", tmp_path, tmp_path)["passed"]
    for name in ("native-k4/live-argv.bin", "native-k4-trace/server.log",
                 "vllm-k4/metrics-after.txt"):
        path = tmp_path / name
        original = path.read_bytes()
        path.unlink()
        with pytest.raises(FileNotFoundError):
            k4.assess("W1", tmp_path, tmp_path)
        path.write_bytes(original)


@pytest.mark.parametrize(("native_ns", "passed"), [(70, True), (80, False), (90, False)])
def test_k4_timing_gate_with_different_valid_ids(tmp_path, monkeypatch, native_ns, passed):
    write_k4_row(tmp_path, native_ns)
    monkeypatch.setattr(sys, "argv", ["k4.py", "assess", "W1", str(tmp_path), str(tmp_path)])
    assert k4.main() == int(not passed)
    result = json.loads((tmp_path / "comparison.json").read_bytes())
    assert result["passed"] is passed
    assert result["workloads"]["W1"]["native"]["samples_ns"] == [native_ns] * 9
    assert result["workloads"]["W1"]["vllm"]["samples_ns"] == [80] * 9


@pytest.mark.parametrize(("native_ns", "passed"), [(70, True), (80, False), (90, False)])
@pytest.mark.parametrize("profile", [k4.ORDERED_PROFILE, k4.SPLIT_PROFILE])
def test_k4_opt_in_labels_exact_compatibility_separately(tmp_path, native_ns, passed, profile):
    write_k4_row(tmp_path, native_ns, opt_in=True, profile=profile)
    result = k4.assess("W1", tmp_path, tmp_path)
    assert result["protocol"] == "matched-k4-opt-in-v1"
    assert result["native_profile"] == profile
    assert (k4.SPLIT_FIELD in result["native_config"]) is (profile == k4.SPLIT_PROFILE)
    assert result["passed"] is passed
    assert result["native_k0_parity"] is True
    assert result["exact_k0_compatibility"]["diagnostic_only"] is True
    assert result["exact_k0_compatibility"]["all_ids_match"] is False
    assert result["exact_k0_compatibility"]["lanes"][0]["different_ids"] == 128
    first = result["exact_k0_compatibility"]["lanes"][0]["first_difference"]
    assert first == {"output_index": 0, "exact_id": 1,
                     ("policy_id" if profile == k4.SPLIT_PROFILE else "ordered_id"): 0}
    assert result["workloads"]["W1"]["native"]["samples_ns"] == [native_ns] * 9


@pytest.mark.parametrize(("arm", "tag", "other_arm", "other_tag"), [
    ("native-k0", "control", "native-exact-k0", "control"),
    ("native-k4", "measured-01", "vllm-k4", "measured-01"),
    ("native-k4-trace", "control", "native-k0", "control"),
])
def test_k4_opt_in_rejects_swapped_arm_receipts(tmp_path, arm, tag, other_arm, other_tag):
    write_k4_row(tmp_path, opt_in=True)
    target = tmp_path / arm / "measurement" / f"{tag}.json.gz"
    foreign = tmp_path / other_arm / "measurement" / f"{other_tag}.json.gz"
    target.write_bytes(foreign.read_bytes())
    with pytest.raises(ValueError, match="Wrong raw cohort campaign or arm identity"):
        k4.assess("W1", tmp_path, tmp_path)


@pytest.mark.parametrize("replacement", [None, "matched-k4-v1", "foreign-campaign"])
def test_k4_opt_in_rejects_changed_campaign(tmp_path, replacement):
    write_k4_row(tmp_path, opt_in=True)
    path = tmp_path / "identity.json"
    if replacement is None:
        path.unlink()
    else:
        identity_data = json.loads(path.read_bytes())
        if replacement == "matched-k4-v1":
            identity_data["protocol"] = replacement
        else:
            identity_data["campaign_id"] = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        path.write_text(json.dumps(identity_data) + "\n")
    message = ("Exact campaign cannot select" if replacement == "matched-k4-v1"
               else "Wrong raw cohort campaign or arm identity")
    with pytest.raises(ValueError, match=message):
        k4.assess("W1", tmp_path, tmp_path)


@pytest.mark.parametrize(("arm", "field", "value"), [
    ("native-k0", "fp8_prefill", "exact"),
    ("native-k4", "mtp_int8_head", False),
    ("native-exact-k0", "fp8_prefill", "tensor-core-mtp-target"),
])
def test_k4_opt_in_rejects_wrong_live_config(tmp_path, arm, field, value):
    write_k4_row(tmp_path, opt_in=True)
    path = tmp_path / arm / "server-info.json"
    info = json.loads(path.read_bytes())
    info[field] = value
    path.write_text(json.dumps(info))
    with pytest.raises(ValueError, match="wrong native server configuration"):
        k4.assess("W1", tmp_path, tmp_path)


@pytest.mark.parametrize(("profile", "arm", "active"), [
    (k4.ORDERED_PROFILE, "native-k4", True),
    (k4.SPLIT_PROFILE, "native-k0", None),
    (k4.SPLIT_PROFILE, "native-k4-trace", False),
    (k4.SPLIT_PROFILE, "native-k4", 1),
    (k4.SPLIT_PROFILE, "native-exact-k0", True),
])
def test_k4_split_rejects_wrong_live_policy(tmp_path, profile, arm, active):
    write_k4_row(tmp_path, opt_in=True, profile=profile)
    path = tmp_path / arm / "server-info.json"
    info = json.loads(path.read_bytes())
    if active is None:
        del info[k4.SPLIT_FIELD]
    else:
        info[k4.SPLIT_FIELD] = active
    write_json(path, info)
    with pytest.raises(ValueError, match=f"{arm}: wrong native server configuration"):
        k4.assess("W1", tmp_path, tmp_path)


@pytest.mark.parametrize("active", [False, 1, "true"])
def test_k4_split_requires_boolean_activation_in_campaign(tmp_path, active):
    config, _ = write_opt_in_identity(tmp_path, k4.SPLIT_PROFILE)
    config["native_config"][k4.SPLIT_FIELD] = active
    write_json(tmp_path / "identity.json", config)
    with pytest.raises(ValueError, match="Unknown opt-in native campaign configuration"):
        k4.campaign(tmp_path)


@pytest.mark.parametrize("profile", [k4.ORDERED_PROFILE, k4.SPLIT_PROFILE])
def test_k4_opt_in_rejects_policy_k4_drift(tmp_path, profile):
    write_k4_row(tmp_path, fault="native-drift", opt_in=True, profile=profile)
    with pytest.raises(ValueError, match="native K4 differs from native K0"):
        k4.assess("W1", tmp_path, tmp_path)


def test_k4_opt_in_captures_exact_diagnostic_before_timing(tmp_path, monkeypatch):
    write_k4_row(tmp_path, opt_in=True)
    directory = tmp_path / "native-exact-k0" / "measurement"
    receipt = k4.read_raw(directory, "control")["requests"][0]
    directory.rename(directory.with_name("synthetic-input"))
    monkeypatch.setattr(k4.streaming, "measure", lambda *_: receipt)
    k4.capture("W1", "native-exact-k0", "http://127.0.0.1:1", tmp_path, directory,
               tmp_path / "native-k0" / "measurement")
    diagnostic = json.loads((tmp_path / "exact-k0-compatibility.json").read_bytes())
    assert diagnostic["diagnostic_only"] is True
    assert diagnostic["all_ids_match"] is False
    assert diagnostic["lanes"][0]["different_ids"] == 128
    assert diagnostic["campaign_sha256"] == k4.campaign(tmp_path)[1]


@pytest.mark.parametrize(("fault", "message"), [
    ("missing-done", "Stream ended before"),
    ("missing-drain", "HTTP response end"),
    ("short", "Expected length completion"),
    ("malformed", "Expecting property name"),
    ("native-drift", "native K4 differs from native K0"),
])
def test_k4_rejects_invalid_evidence(tmp_path, fault, message):
    write_k4_row(tmp_path, fault=fault)
    with pytest.raises(ValueError, match=message):
        k4.assess("W1", tmp_path, tmp_path)


@pytest.mark.parametrize("trailing_line", ["data:", "data: ", "data", "data: [DONE]"])
def test_k4_rejects_trailing_data_frames(tmp_path, trailing_line):
    write_k4_row(tmp_path, trailing_line=trailing_line)
    with pytest.raises(ValueError, match=r"Data after \[DONE\]"):
        k4.assess("W1", tmp_path, tmp_path)


def test_k4_concurrent_rejects_empty_data_after_done():
    raw, lanes = synthetic_cohort()
    policy = identity.concurrent.reference()["policy"]
    assert k4.replay(raw, "W3", "measured-01", lanes, policy)[0] == 95
    events = raw["requests"][1]["events"]
    events.append({"lines": ["data:"], "ns": events[-1]["ns"] + 1,
                   "read": events[-1]["read"] + 1})
    with pytest.raises(ValueError, match=r"Data after \[DONE\]"):
        k4.replay(raw, "W3", "measured-01", lanes, policy)


def write_measurement(run, hashes, freeze_baseline=True, stamps_by_name=None):
    measurement = run / "measurement"
    measurement.mkdir(exist_ok=True)
    manifest = identity.stream_identity(ROOT, hashes)
    cases = identity.streaming.load_cases(ROOT / "benchmarks" / "streaming")
    records = {name: [] for name in cases}
    for name, (case, _) in cases.items():
        request = identity.streaming.request_body(case)
        (measurement / f"{name}.request.json").write_bytes(identity.streaming.json_bytes(request))
    with (measurement / "events.jsonl").open("w") as events:
        for phase, count in manifest["schedule"]:
            for cohort in range(1, count + 1):
                for name, (case, _) in cases.items():
                    stamps = (stamps_by_name or {}).get(name, (20, 50, 60, 70))
                    record = stream_record(case, manifest["request_sha256"][name],
                                           name, phase, cohort, stamps)
                    events.write(json.dumps(record) + "\n")
                    records[name].append(record)
    write_json(measurement / "summary.json", {"manifest": manifest, "workloads": {
        name: identity.streaming.summarize(items, freeze_baseline)
        for name, items in records.items()}})
    (run / "client-exit-code.txt").write_text("0\n")


def write_saved_worker(run, checkpoint, reference, cap, *, historical=False):
    worker = reference["observed_vllm_settings"]
    worker.update(
        scheduler={"max_model_len": 2560, "max_num_batched_tokens": cap, "max_num_seqs": 32,
                   "enable_chunked_prefill": True},
        parallel={"tp": 1, "dp": 1, "pp": 1}, quantization="fp8", skip_tokenizer_init=True,
        flashinfer="0.7.4",
        kv_transfer=False, kv_offloading_size=None, cpu_offload="cpu_offload_gb=0.0, "
        "offload_group_size=0, ")
    worker["cache"]["kv_cache_memory_bytes"] = 10 << 30
    if not historical:
        worker["scheduler"].update(max_num_scheduled_tokens_configured=cap,
                                   max_num_scheduled_tokens_effective=cap,
                                   async_scheduling=True)
    scheduler_flags = [] if historical else [
        "--max-num-scheduled-tokens", str(cap), "--async-scheduling",
    ]
    write_json(run / "effective-worker.json", {"results": [worker]})
    write_command(run, "server", "24-33", [
        "python", "-m", "vllm.entrypoints.openai.api_server", "--model", str(checkpoint),
        "--max-model-len", "2560", "--max-num-seqs", "32", "--max-num-batched-tokens", str(cap),
        *scheduler_flags,
        "--generation-config", "vllm", "--seed", "0", "--stream-interval", "1",
        "--language-model-only", "--no-enable-prefix-caching"])
    (run / "server-pid.txt").write_text("1000\n")
    (run / "server.log").write_text(
        "(APIServer pid=1000) Starting engine\n"
        f"(EngineCore pid=1001) Initializing a V1 LLM engine with config: model='{checkpoint}'\n")
    write_compute(run, GPU, 1001)


class PairedIdentityTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        saved = self.root / "saved"
        output = self.root / "native"
        capture = saved / f"vllm-{GROUP}"
        self.saved_run = capture / GROUP
        self.saved_run.mkdir(parents=True)
        output.mkdir()
        self.reference = identity.concurrent.reference()
        checkpoint = self.root / self.reference["model"]
        checkpoint.mkdir()
        weight = checkpoint / "weights.safetensors"
        weight.write_bytes(b"fixed checkpoint fixture")
        manifest = saved / "checkpoint.sha256"
        manifest.write_text(f"{identity.digest(weight)}  {weight.name}\n")
        self.args = SimpleNamespace(
            mode="matched", saved=saved, output=output, protocol=ROOT, groups=[GROUP], group=GROUP,
            checkpoint=checkpoint, checkpoint_manifest=manifest, warmups=2, measured=5,
            fixtures=self.root / "fixtures", allow_count_mismatch=False,
            max_num_batched_tokens=8192)
        self.reference["checkpoint_manifest_sha256"] = identity.digest(manifest)
        patcher = patch.object(identity.concurrent, "reference", side_effect=self.fixture_reference)
        patcher.start()
        self.addCleanup(patcher.stop)
        write_receipts(capture, 2, GPU)
        write_receipts(output, 0, GPU)
        (capture / "vllm-source-commit.txt").write_text(self.reference["vllm_pin"] + "\n")
        (capture / "effective-environment.txt").write_text(f"CUDA_VISIBLE_DEVICES={GPU}\n")
        hashes = identity.current_protocol(GROUP, ROOT)
        (capture / "protocol.sha256").write_text("".join(
            f"{sha}  /saved/benchmarks/{name}\n" for name, sha in hashes.items()))
        write_saved_worker(self.saved_run, checkpoint, self.fixture_reference(), 8192)
        write_command(self.saved_run, "client", "34-35", [
            "python", str(ROOT / "benchmarks/streaming/client.py"), "--freeze-baseline"])
        write_measurement(self.saved_run, hashes)

    def fixture_reference(self):
        return deepcopy(self.reference)

    def test_same_uuid_can_have_different_logical_index(self):
        plan = identity.plan(self.args)
        self.assertEqual(plan["groups"][GROUP]["planned_comparison"], "same-device matched-cap")
        self.assertEqual(plan["groups"][GROUP]["vllm"]["device"]["uuid"], GPU.lower())
        self.assertEqual(plan["groups"][GROUP]["vllm"]["source_commit"], self.reference["vllm_pin"])

    def test_reject_unverified_vllm_source_commit(self):
        receipt = self.saved_run.parent / "vllm-source-commit.txt"
        receipt.write_text("0" * 40 + "\n")
        with self.assertRaisesRegex(ValueError, "unexpected vLLM source commit"):
            identity.plan(self.args)
        receipt.unlink()
        with self.assertRaisesRegex(ValueError, "missing vLLM source commit receipt"):
            identity.plan(self.args)

    def test_reject_legacy_stream_capture_for_formal_gate(self):
        receipt = self.saved_run.parent / "protocol.sha256"
        current = identity.current_protocol(GROUP, ROOT)
        legacy = identity.LEGACY_STREAMING_CLIENT
        receipt.write_text(receipt.read_text().replace(current["streaming/client.py"], legacy))
        with self.assertRaisesRegex(ValueError, "fresh capture through response end"):
            identity.plan(self.args)

    def test_replay_rejects_stream_data_after_reported_done(self):
        path = self.saved_run / "measurement" / "events.jsonl"
        original = path.read_text().splitlines()
        for trailing_line in ("data: [DONE]", 'data: {"error":"failed after DONE"}', "data:"):
            with self.subTest(trailing_line=trailing_line):
                record = json.loads(original[0])
                terminal = record["events"][-1]
                record["events"].append({**terminal, "lines": [trailing_line]})
                path.write_text("\n".join([json.dumps(record), *original[1:]]) + "\n")
                with self.assertRaisesRegex(ValueError, r"Data after \[DONE\]"):
                    identity.plan(self.args)

    def test_replay_requires_complete_http_body(self):
        path = self.saved_run / "measurement" / "events.jsonl"
        original = path.read_text().splitlines()
        for short_body in (False, True):
            with self.subTest(short_body=short_body):
                record = json.loads(original[0])
                if short_body:
                    record["response_end"]["body_bytes"] -= 1
                    message = "HTTP response body bytes differ from Content-Length"
                else:
                    record.pop("response_end")
                    message = "Missing or invalid HTTP response end receipt"
                path.write_text("\n".join([json.dumps(record), *original[1:]]) + "\n")
                with self.assertRaisesRegex(ValueError, message):
                    identity.plan(self.args)

    def test_w5_override_counts_cannot_earn_formal_verdict(self):
        plan = identity.plan(self.args)
        arm = plan["groups"][GROUP]["vllm"]
        for counts, expected in (
            ((1, 2), "exploratory (non-protocol counts)"),
            ((2, 5), "same-device matched-cap"),
        ):
            with self.subTest(counts=counts):
                label = identity.comparison_label(
                    "matched", "W5", arm | {"counts": list(counts)}, {}, plan["device"],
                    counts, 8192)
                self.assertEqual(label, expected)
                samples = counts[1]
                result = identity.compare({"W5": [50] * samples}, {"W5": [60] * samples},
                                          eligible=label == "same-device matched-cap")
                self.assertIs(result["passed"], True if counts == (2, 5) else None)

    def test_equal_counts_cannot_hide_device_mismatch(self):
        self.args.allow_count_mismatch = True
        write_receipts(self.args.output, 2, OTHER_GPU)
        with self.assertRaisesRegex(ValueError, "saved device"):
            identity.plan(self.args)

    def test_compare_process_uuid_at_measurement_end(self):
        (self.saved_run / "compute-measurement-end.csv").write_text(f"{OTHER_GPU}, 1001, worker\n")
        with self.assertRaisesRegex(ValueError, "compute PID 1001"):
            identity.plan(self.args)

    def test_compare_saved_request_body(self):
        with (self.saved_run / "measurement" / "W1.request.json").open("ab") as request:
            request.write(b" ")
        with self.assertRaisesRegex(ValueError, "saved streaming request digest"):
            identity.plan(self.args)

    def test_compare_observed_scheduler_cap(self):
        receipt = self.saved_run / "effective-worker.json"
        worker = identity.load(receipt)
        worker["results"][0]["scheduler"]["max_num_scheduled_tokens_effective"] = 32768
        write_json(receipt, worker)
        with self.assertRaisesRegex(ValueError, "effective vLLM scheduler"):
            identity.plan(self.args)

    def test_reject_unobserved_protocol_cap(self):
        receipt = self.saved_run / "effective-worker.json"
        worker = identity.load(receipt)
        del worker["results"][0]["scheduler"]["max_num_scheduled_tokens_effective"]
        write_json(receipt, worker)
        with self.assertRaisesRegex(ValueError, "effective vLLM scheduler"):
            identity.plan(self.args)

    def test_historical_cap_is_exploratory_even_with_identical_counts(self):
        write_saved_worker(self.saved_run, self.args.checkpoint, self.fixture_reference(),
                           32768, historical=True)
        with self.assertRaisesRegex(ValueError, "vLLM scheduler cap"):
            identity.plan(self.args)
        self.args.max_num_batched_tokens = 32768
        self.prepare_native()
        result = identity.finish(self.args)
        self.assertEqual(result["comparison"], "exploratory (non-protocol cap 32768)")

    def test_reject_vllm_eager_command_with_graph_receipt(self):
        command = self.saved_run / "server-command.sh"
        command.write_text(command.read_text() + " --enforce-eager")
        with self.assertRaisesRegex(ValueError, "effective vLLM scheduler"):
            identity.plan(self.args)

    def test_compare_protocol_digest(self):
        receipt = self.saved_run.parent / "protocol.sha256"
        contents = receipt.read_text().splitlines()
        receipt.write_text("0" * 64 + contents[0][64:] + "\n" + "\n".join(contents[1:]) + "\n")
        with self.assertRaisesRegex(ValueError, "protocol digest"):
            identity.plan(self.args)

    def test_replay_detects_summary_metric_changes_with_equal_counts(self):
        receipt = self.saved_run / "measurement" / "summary.json"
        report = identity.load(receipt)
        report["workloads"]["W1"]["ttft_ms"]["p95"] = 1234
        write_json(receipt, report)
        with self.assertRaisesRegex(ValueError, "streaming summary differs from replay"):
            identity.plan(self.args)

    def test_check_actual_checkpoint_bytes(self):
        (self.args.checkpoint / "weights.safetensors").write_bytes(b"different checkpoint fixture")
        with self.assertRaises(identity.subprocess.CalledProcessError):
            identity.plan(self.args)

    def test_published_comparison_has_cross_cap_and_device_labels(self):
        self.args.mode = "frozen"
        self.args.saved = None
        self.args.groups = ["W5"]
        with patch.object(identity, "load_lanes") as fixtures:
            first = identity.plan(self.args)["groups"]["W5"]
            write_receipts(self.args.output, 2, OTHER_GPU)
            second = identity.plan(self.args)["groups"]["W5"]
            self.args.max_num_batched_tokens = 32768
            third = identity.plan(self.args)["groups"]["W5"]
            write_receipts(self.args.output, 2, GPU)
            fourth = identity.plan(self.args)["groups"]["W5"]
        self.assertEqual(fixtures.call_count, 4)
        self.assertEqual(first["planned_comparison"], "published frozen reference")
        self.assertEqual(second["planned_comparison"],
                         "published frozen reference (cross-device)")
        self.assertEqual(third["planned_comparison"],
                         "published frozen reference (cross-device, cross-cap)")
        self.assertEqual(fourth["planned_comparison"], "published frozen reference (cross-cap)")

    def prepare_native(self):
        output = self.args.output
        plan_path = output / "comparison-plan.json"
        write_json(plan_path, identity.plan(self.args))
        self.args.plan_sha256 = identity.digest(plan_path)
        run = output / GROUP
        run.mkdir()
        (run / "server-pid.txt").write_text("2000\n")
        write_compute(run, GPU, 2000)
        write_command(run, "server", "24-33", [
            "sglang-server", "--checkpoint", str(self.args.checkpoint), "--device", "0",
            "--max-active", "32", "--max-num-batched-tokens",
            str(self.args.max_num_batched_tokens),
            "--fp8-prefill", "tensor-core", "--prefix-cache-entries", "0", "--mtp-steps", "0",
            "--batch-cuda-graphs"])
        write_command(run, "client", "34-35", [
            "python", str(ROOT / "benchmarks/streaming/client.py")])
        (output / "native-environment.txt").write_text(f"CUDA_VISIBLE_DEVICES={GPU}\n")
        write_json(run / "health.json", {"max_active": 32,
                                        "max_num_batched_tokens": self.args.max_num_batched_tokens})
        hashes = identity.current_protocol(GROUP, ROOT)
        (output / "protocol.sha256").write_text("".join(
            f"{sha}  /native/benchmarks/{name}\n" for name, sha in hashes.items()))
        write_measurement(run, hashes, freeze_baseline=False)
        return run

    def test_publish_only_after_native_process_is_observed(self):
        run = self.prepare_native()
        write_compute(run, OTHER_GPU, 2000)
        with self.assertRaisesRegex(ValueError, "compute PID 2000"):
            identity.finish(self.args)
        write_compute(run, GPU, 2000)
        result = identity.finish(self.args)
        self.assertEqual(result["comparison"], "same-device matched-cap")
        self.assertFalse(result["quality"]["cross_engine_ids_compared"])

    def test_finish_uses_each_workload_makespan_and_preserves_tie_receipt(self):
        run = self.prepare_native()
        hashes = identity.current_protocol(GROUP, ROOT)
        write_measurement(run, hashes, freeze_baseline=False,
                          stamps_by_name={"W1": (40, 50, 60, 65)})
        result = identity.finish(self.args)
        makespan = result["makespan"]
        self.assertFalse(makespan["passed"])
        self.assertTrue(makespan["workloads"]["W1"]["passed"])
        self.assertFalse(makespan["workloads"]["W2"]["passed"])
        self.assertEqual(makespan["workloads"]["W1"]["native"],
                         {"samples_ns": [65] * 9, "median_ns": 65})
        argv = ["identity.py", str(ROOT), str(self.args.output), "finish", "matched", GROUP,
                self.args.plan_sha256]
        with patch.object(sys, "argv", argv), patch("sys.stdout") as stdout:
            self.assertEqual(identity.main(), 3)
        receipt = json.loads(stdout.write.call_args_list[0].args[0])
        self.assertEqual(receipt["makespan"], makespan)
        self.assertEqual(receipt["comparison"], "same-device matched-cap")

        write_measurement(run, hashes, freeze_baseline=False,
                          stamps_by_name={"W1": (40, 50, 60, 65), "W2": (30, 50, 60, 69)})
        self.assertTrue(identity.finish(self.args)["makespan"]["passed"])

    def test_reject_native_health_cap_mismatch(self):
        run = self.prepare_native()
        write_json(run / "health.json", {"max_active": 32, "max_num_batched_tokens": 32768})
        with self.assertRaisesRegex(ValueError, "effective native scheduler settings"):
            identity.finish(self.args)

    def test_reject_vllm_effective_cap_mismatch_at_finish(self):
        self.prepare_native()
        path = self.args.output / "comparison-plan.json"
        plan = identity.load(path)
        scheduler = plan["groups"][GROUP]["vllm"]["effective"]["scheduler"]
        scheduler["max_num_scheduled_tokens_effective"] = 32768
        write_json(path, plan)
        self.args.plan_sha256 = identity.digest(path)
        with self.assertRaisesRegex(ValueError, "requires observed 8192-token vLLM scheduling"):
            identity.finish(self.args)

    def test_reject_relabelled_comparison_plan(self):
        self.prepare_native()
        path = self.args.output / "comparison-plan.json"
        plan = identity.load(path)
        plan["groups"][GROUP]["planned_comparison"] = "published frozen reference (cross-cap)"
        write_json(path, plan)
        with self.assertRaisesRegex(ValueError, "changed after preflight"):
            identity.finish(self.args)
        self.args.plan_sha256 = identity.digest(path)
        with self.assertRaisesRegex(ValueError, "label disagrees"):
            identity.finish(self.args)

    def test_reject_finish_in_another_mode(self):
        self.prepare_native()
        self.args.mode = "frozen"
        with self.assertRaisesRegex(ValueError, "plan mode differs"):
            identity.finish(self.args)

    def test_reject_wrong_native_client(self):
        run = self.prepare_native()
        write_command(run, "client", "34-35", ["python", "client.py"])
        with self.assertRaisesRegex(ValueError, "unexpected streaming client"):
            identity.finish(self.args)

    def test_reject_native_mtp_alias(self):
        run = self.prepare_native()
        command = run / "server-command.sh"
        command.write_text(command.read_text() + " --mtp-k 1")
        with self.assertRaisesRegex(ValueError, "native command differs"):
            identity.finish(self.args)

    def test_replay_detects_native_summary_change(self):
        run = self.prepare_native()
        receipt = run / "measurement" / "summary.json"
        report = identity.load(receipt)
        report["workloads"]["W1"]["ttft_ms"]["p95"] = 1234
        write_json(receipt, report)
        with self.assertRaisesRegex(ValueError, "streaming summary differs from replay"):
            identity.finish(self.args)
