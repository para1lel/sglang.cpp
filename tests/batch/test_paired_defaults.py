"""Exercise default-only launch evidence and preserve distinct K4 protocols."""

import gzip
import json
import os
import shlex
import sys
from pathlib import Path

import pytest

from tests.test_paired_identity import GPU, OTHER_GPU, k4, write_command, write_compute, write_json
from tests.test_paired_identity import (
    synthetic_cohort, write_k4_row, write_receipts, write_saved_worker,
)
from makespan import K4_MAKESPAN_CRITERION, compare


def write_default_vllm(run, checkpoint, *, cache_on=False, max_seqs=32):
    write_saved_worker(run, checkpoint, k4.concurrent.reference(), 8192)
    args = shlex.split((run / "server-command.sh").read_text())[6:]
    argv = ["/venv/bin/python", "-B", *args, "--host", "127.0.0.1",
            "--port", "17424", "--served-model-name", "benchmark",
            "--speculative-config", '{"method":"mtp","num_speculative_tokens":4}']
    snapshot = run / "effective-worker.json"
    data = json.loads(snapshot.read_bytes())
    argv[argv.index("--max-num-seqs") + 1] = str(max_seqs)
    data["results"][0]["scheduler"]["max_num_seqs"] = max_seqs
    cache = {"enable_prefix_caching": True, "mamba_cache_mode": "align",
             "block_size": 1600, "mamba_block_size": 1600}
    if cache_on:
        argv[argv.index("--no-enable-prefix-caching")] = "--enable-prefix-caching"
        argv.append("--enable-prompt-tokens-details")
        data["results"][0]["cache"].update(cache)
    scale = {"tensor": 1.0, "dtype": "torch.float32", "host": 1.0}
    gqa = {"backend": "FLASHINFER", "kernel_cache_dtype": "torch.float8_e4m3fn",
           "scales": dict.fromkeys("qkv", scale)}
    gdn = {"convolution_dtype": "torch.bfloat16", "recurrence_dtype": "torch.float32"}
    data["results"][0].update(speculative=True, gqa=dict.fromkeys(range(16), gqa),
                              gdn=dict.fromkeys(range(48), gdn))
    write_json(snapshot, data)
    with (run / "server.log").open("a") as stream:
        stream.write("FlashInfer resolved query dtypes: prefill=torch.bfloat16, "
                     "decode=torch.bfloat16, decode_backend=trtllm-gen, "
                     "kv=torch.float8_e4m3fn\n")
    write_json(run / "server-info.json", {"vllm_config": {
        "speculative_config": {"method": "mtp", "num_speculative_tokens": 4,
                               "model": checkpoint},
        "model_config": {"model": checkpoint},
        **({"cache_config": cache} if cache_on else {}),
        "attention_config": {"use_trtllm_attention": True,
                             "disable_flashinfer_q_quantization": True},
    }})
    return argv


def write_default_native(run, arm, *, cache_on=False, max_seqs=32):
    flags = ["--port", "17424"]
    if arm == "native-k4-trace":
        flags.append("--trace-batches")
    info = json.loads((run / "server-info.json").read_bytes())
    info.update(launch_command=" ".join(["sglang-server", *flags]), dp_size=1, tp_size=1, pp_size=1,
                prefix_cache_entries=4 if cache_on else 0, max_running_requests=max_seqs)
    write_json(run / "server-info.json", info)
    write_compute(run, GPU, 123)
    return ["/bin/sglang-server", *flags]


def write_cold_cache_receipt(raw, arm):
    cache = k4.cold_cache
    request = raw["requests"][0]
    for field in ("submit_ns", "headers_ns", "done_ns"):
        request[field] += 100
    request["response_end"]["ns"] += 100
    for event in request["events"]:
        event["ns"] += 100
    for chunk in request["token_chunks"]:
        chunk[0] += 100
    submit = request["submit_ns"]
    eof = request["response_end"]["ns"]
    metrics = f'{cache.RUNNING}{{engine="0"}} 0.0\n{cache.WAITING} 0\n{cache.HITS} 40\n'
    occupied = metrics
    if arm != "vllm-k4":
        metrics += "".join(f"{name} 0\n" for name in cache.NATIVE_METRICS)
        occupied = metrics.replace(f"{cache.ENTRIES} 0", f"{cache.ENTRIES} 1")
    return {"before": occupied, "cleared": metrics, "after": occupied,
            "reset": {"status": 200, "body": '{"success":true}'},
            "reset_started_ns": submit - 30, "reset_completed_ns": submit - 20,
            "ready_ns": submit - 10, "capture_finished_ns": eof + 10}


def write_default_row(root, native_ns=70, *, cache_on=False, criterion=None, current=False):
    if current:
        cache_on = True
        criterion = K4_MAKESPAN_CRITERION
    max_seqs = 8 if current else 32
    protocol_name = k4.defaults.CACHE_ON_PROTOCOL if cache_on else k4.defaults.PROTOCOL
    if current:
        protocol_name = k4.workload_suite.PROTOCOL
    row = root / "W1"
    row.mkdir()
    write_k4_row(row, native_ns, receipt_version=1)
    protocol = row / "protocol.sha256"
    protocol.write_text("captured protocol manifest\n")
    checkpoint = "/model"
    write_json(row / "identity.json", {
        "protocol": protocol_name,
        **({"participating_gpus": [GPU], "max_running_requests": max_seqs} if current else {}),
        **({"cache_policy": k4.defaults.CACHE_POLICY} if cache_on else {}),
        **({"makespan_criterion": criterion} if criterion else {}),
        "native_profile": k4.defaults.PROFILE,
        "replay_receipts": 2, "campaign_id": "01234567-89ab-cdef-0123-456789abcdef",
        "checkpoint_path": checkpoint, "port": 17424, "gpu_uuid": GPU,
        "server_cpus": "24-33", "client_cpus": "34-35", "memory": "localalloc",
        "native_binary_sha256": "b" * 64, "protocol_sha256": k4.digest(protocol),
    })
    reservation = row / "reservation"
    reservation.mkdir()
    write_receipts(reservation, 2, GPU)
    selected = k4.campaign(row)
    for arm in k4.ARM_ORDER[1:]:
        run = row / arm
        environment = dict.fromkeys(k4.defaults.NATIVE_ENVIRONMENT, "/archive")
        environment.update(CUDA_VISIBLE_DEVICES=GPU, CUDA_DEVICE_ORDER="PCI_BUS_ID")
        if arm == "vllm-k4":
            argv = write_default_vllm(run, checkpoint, cache_on=cache_on, max_seqs=max_seqs)
        else:
            argv = write_default_native(run, arm, cache_on=cache_on, max_seqs=max_seqs)
        assignments = [f"{key}={value}" for key, value in sorted(environment.items())]
        (run / "effective-environment.txt").write_text("\n".join(assignments) + "\n")
        for name in ("live-environment.bin", "live-environment-end.bin"):
            (run / name).write_bytes(b"\0".join(map(os.fsencode, assignments)) + b"\0")
        (run / "live-argv.bin").write_bytes(b"\0".join(map(os.fsencode, argv)) + b"\0")
        (run / "live-cpus.txt").write_text("24-33\n")
        write_command(run, "server", "24-33", ["env", "-i", *assignments, *argv])
        write_command(run, "client", "34-35", [
            "/bin/python", "k4.py", "capture", "W1", arm, "http://127.0.0.1:17424",
            str(row), str(run / "measurement"),
        ])
        if arm == "vllm-k4":
            k4.check_vllm_worker("W1", run, reservation, Path(checkpoint),
                                before_measurement=True)
            k4.check_vllm_worker("W1", run, reservation, Path(checkpoint))
        for tag in k4.schedule("W1", arm):
            measurement = run / "measurement"
            raw = k4.read_raw(measurement, tag)
            raw["binding"] = k4.arm_binding(measurement, arm, selected, 2)
            if cache_on:
                raw["cold_cache"] = write_cold_cache_receipt(raw, arm)
            with gzip.open(measurement / f"{tag}.json.gz", "wt") as stream:
                json.dump(raw, stream)
    return row


@pytest.mark.parametrize(("native_ns", "passed", "strict"), [
    (79, True, True), (80, True, False), (88, True, False), (89, False, False),
])
def test_k4_current_criterion_and_prior_row_chain(tmp_path, monkeypatch, native_ns, passed, strict):
    row = write_default_row(tmp_path, native_ns, cache_on=True,
                            criterion=K4_MAKESPAN_CRITERION)
    root_identity = row.joinpath("identity.json").read_bytes()
    tmp_path.joinpath("identity.json").write_bytes(root_identity)
    monkeypatch.setattr(sys, "argv", ["k4.py", "assess", "W1", str(row), str(row)])
    assert k4.main() == int(not passed)
    comparison = row / "comparison.json"
    result = k4.load(comparison)
    assert result["protocol"] == k4.defaults.CACHE_ON_PROTOCOL
    assert result["passed"] is passed
    assert result["makespan_criterion"] == K4_MAKESPAN_CRITERION
    workload = result["workloads"]["W1"]
    assert workload["native"]["median_ns"] == native_ns
    assert workload["vllm"]["median_ns"] == 80
    assert workload["native_to_vllm_median_ratio"] == native_ns / 80
    assert workload["strict_passed"] is strict
    row.joinpath("exit-code.txt").write_text(f"{int(not passed)}\n")
    assert k4.assess("W1", row, row) == result

    if passed:
        k4.check_prior_rows("W2", tmp_path)
    else:
        with pytest.raises(ValueError, match="W1 must pass"):
            k4.check_prior_rows("W2", tmp_path)

    if passed:
        for changed_field in ("makespan_criterion", "campaign_sha256"):
            write_json(comparison, {key: value for key, value in result.items()
                                    if key != changed_field})
            with pytest.raises(ValueError, match="Saved comparison differs from replay"):
                k4.assess("W1", row, row)
            with pytest.raises(ValueError, match="W1 must pass"):
                k4.check_prior_rows("W2", tmp_path)


@pytest.mark.parametrize("criterion", [None, 1.1, "unknown"])
def test_k4_rejects_unknown_criterion(tmp_path, criterion):
    write_k4_row(tmp_path)
    write_json(tmp_path / "identity.json", {"protocol": k4.PROTOCOL, "replay_receipts": 2,
                                          "makespan_criterion": criterion})
    with pytest.raises(ValueError, match="Unknown K4 makespan criterion"):
        k4.assess("W1", tmp_path, tmp_path)


@pytest.mark.parametrize(("native", "passed"), [([115, 116], True), ([116, 116], False)])
def test_k4_accepts_fractional_median_at_boundary(native, passed):
    result = compare({"W1": native}, {"W1": [104, 106]}, eligible=True,
                     criterion=K4_MAKESPAN_CRITERION)
    assert result["passed"] is passed


def test_k4_replays_legacy_receipt_with_strict_verdict(tmp_path):
    write_k4_row(tmp_path, 80, receipt_version=2)
    native = {"samples_ns": [80] * 9, "median_ns": 80}
    result = {"protocol": k4.PROTOCOL, "row": "W1", "passed": False,
              "workloads": {"W1": {"native": native, "vllm": native, "passed": False}}}
    write_json(tmp_path / "comparison.json", result)
    (tmp_path / "exit-code.txt").write_text("1\n")
    assert k4.assess("W1", tmp_path, tmp_path) == result


@pytest.mark.parametrize(("native_ns", "passed"), [(70, True), (80, False)])
def test_native_defaults_k4_uses_timing_and_live_evidence(tmp_path, native_ns, passed):
    row = write_default_row(tmp_path, native_ns)
    (row / "native-k0/measurement/control.json.gz").unlink()
    result = k4.assess("W1", row, row)
    assert result["protocol"] == "matched-k4-native-defaults-v1"
    assert result["vllm_role"] == "timing-only"
    assert "native_k0_parity" not in result
    assert result["passed"] is passed


@pytest.mark.parametrize(("name", "old", "new", "message"), [
    ("native-k4/live-environment.bin", b"LANG=", b"SGLANG_EXPERIMENTAL_TARGET_SPLIT_GQA=",
     "unexpected or missing settings"),
    ("native-k4/live-argv.bin", b"--port\0", b"--mtp-steps\0", "live process arguments differ"),
    ("native-k4/server-info.json", b'"mtp_steps": 4', b'"mtp_steps": 0',
     "wrong resolved native defaults"),
    ("native-k4/live-cpus.txt", b"24-33", b"34-35", "live API process CPU affinity differs"),
    ("vllm-k4/server-info.json", b'"num_speculative_tokens": 4', b'"num_speculative_tokens": 0',
     "Unexpected resolved vLLM model, MTP, or attention configuration"),
    ("vllm-k4/compute-measurement-end.csv", GPU.encode(), OTHER_GPU.encode(), "must be alone"),
    ("vllm-k4/live-environment-end.bin", b"LANG=", b"SGLANG_OPTION=",
     "vLLM environment changed"),
    ("protocol.sha256", b"captured", b"altered", "Saved protocol manifest differs"),
])
def test_native_defaults_k4_rejects_conflicting_evidence(tmp_path, name, old, new, message):
    row = write_default_row(tmp_path)
    path = row / name
    raw = path.read_bytes()
    assert old in raw
    path.write_bytes(raw.replace(old, new, 1))
    with pytest.raises(ValueError, match=message):
        k4.assess("W1", row, row)


def test_cache_on_replays_every_cohort_and_keeps_zero_hit_contract(tmp_path):
    row = write_default_row(tmp_path, cache_on=True)
    result = k4.assess("W1", row, row)
    assert result["protocol"] == k4.defaults.CACHE_ON_PROTOCOL
    assert result["cache_policy"] == k4.defaults.CACHE_POLICY
    assert result["passed"]
    for arm, receipts in result["cache_receipts"].items():
        assert set(receipts) == set(k4.schedule("W1", arm))


@pytest.mark.parametrize(("change", "message"), [
    ("missing", "Missing cold cache control receipt"),
    ("declined", "Prefix cache reset was not acknowledged"),
    ("busy", "active or queued request"),
    ("late", "overlaps the timed cohort"),
    ("hit-counter", "Prefix cache hit counter advanced"),
    ("occupied", "Native cache is occupied after reset"),
    ("wire-hit", "zero reused prompt tokens"),
])
def test_cache_on_rejects_invalid_cohort_isolation(tmp_path, change, message):
    row = write_default_row(tmp_path, cache_on=True)
    directory = row / "native-k4/measurement"
    tag = k4.schedule("W1", "native-k4")[0]
    raw = k4.read_raw(directory, tag)
    receipt = raw["cold_cache"]
    cache = k4.cold_cache
    if change == "missing":
        del raw["cold_cache"]
    elif change == "declined":
        receipt["reset"]["body"] = '{"success":false}'
    elif change == "busy":
        receipt["cleared"] = receipt["cleared"].replace(f"{cache.WAITING} 0", f"{cache.WAITING} 1")
    elif change == "late":
        receipt["ready_ns"] = raw["requests"][0]["submit_ns"] + 1
    elif change == "hit-counter":
        receipt["after"] = receipt["after"].replace(f"{cache.HITS} 40", f"{cache.HITS} 41")
    elif change == "wire-hit":
        request = raw["requests"][0]
        usage = request["events"][-2]["lines"]
        usage[0] = usage[0].replace('"cached_tokens": 0', '"cached_tokens": 1')
    else:
        receipt["cleared"] = receipt["cleared"].replace(f"{cache.ENTRIES} 0", f"{cache.ENTRIES} 1")
    with gzip.open(directory / f"{tag}.json.gz", "wt") as stream:
        json.dump(raw, stream)
    with pytest.raises(ValueError, match=message):
        k4.assess("W1", row, row)


def test_cache_on_rejects_disagreeing_hybrid_block_geometry(tmp_path):
    row = write_default_row(tmp_path, cache_on=True)
    path = row / "vllm-k4/server-info.json"
    config = json.loads(path.read_bytes())
    config["vllm_config"]["cache_config"]["mamba_block_size"] += 1
    write_json(path, config)
    with pytest.raises(ValueError, match="API and worker disagree"):
        k4.assess("W1", row, row)


def test_current_workload_geometry_and_frozen_lane_selection():
    suite = k4.workload_suite
    assert tuple(suite.ROWS) == ("W1", "W2", "W3", "W4", "W5")
    shapes = [(128, 128, 1), (2048, 512, 1), (8192, 1024, 4),
              (32768, 4096, 8), (131072, 1024, 4)]
    for row, (prompt, output, per_gpu) in zip(suite.ROWS, shapes, strict=True):
        assert suite.geometry(row, 3) == {"prompt_tokens": prompt, "output_tokens": output,
                                          "requests_per_gpu": per_gpu,
                                          "concurrent_requests": 3 * per_gpu}
    assert suite.serving_capacity(3) == 24
    original = suite.reference.lane_manifest()["rows"]
    assert suite.concurrent_metadata("W4", 1) == original["W5"]["lanes"][:8]
    assert suite.concurrent_metadata("W5", 1) == original["W6"]["lanes"]
    assert suite.concurrent_metadata("W4", 3) == original["W5"]["lanes"][:24]


def test_current_replay_checks_effective_default_capacity(tmp_path):
    row = write_default_row(tmp_path, current=True)
    identity, _ = k4.campaign(row)
    result = k4.assess("W1", row, row)
    assert result["passed"] is True
    assert result["protocol"] == k4.workload_suite.PROTOCOL
    assert result["workload"] == k4.workload_evidence("W1", identity)
    native = row / "native-k4/server-info.json"
    info = json.loads(native.read_bytes())
    write_json(native, {**info, "max_running_requests": 32})
    with pytest.raises(ValueError, match="wrong resolved native defaults"):
        k4.assess("W1", row, row)
    write_json(native, info)
    worker = row / "vllm-k4/effective-worker.json"
    settings = json.loads(worker.read_bytes())
    settings["results"][0]["scheduler"]["max_num_seqs"] = 32
    write_json(worker, settings)
    with pytest.raises(ValueError, match="scheduler"):
        k4.assess("W1", row, row)


@pytest.mark.parametrize("devices, cap", [([OTHER_GPU], 8), ([GPU, OTHER_GPU], 16), ([GPU], 32)])
def test_current_identity_rejects_mismatched_execution_group(tmp_path, devices, cap):
    row = write_default_row(tmp_path, current=True)
    identity, _ = k4.campaign(row)
    write_json(row / "identity.json", {**identity, "participating_gpus": devices,
                                      "max_running_requests": cap})
    with pytest.raises(ValueError, match="participating GPU|serving concurrency"):
        k4.campaign(row)


def test_current_w5_requires_exact_current_w4_workload(tmp_path):
    first = write_default_row(tmp_path, current=True)
    identity, fingerprint = k4.campaign(first)
    root_identity = first / "identity.json"
    (tmp_path / "identity.json").write_bytes(root_identity.read_bytes())
    for name in ("W1", "W2", "W3", "W4"):
        row = tmp_path / name
        row.mkdir(exist_ok=True)
        (row / "identity.json").write_bytes(root_identity.read_bytes())
        (row / "exit-code.txt").write_text("0\n")
        write_json(row / "comparison.json", {"row": name, "protocol": identity["protocol"],
                   "campaign_sha256": fingerprint, "passed": True,
                   "makespan_criterion": K4_MAKESPAN_CRITERION,
                   "workload": k4.workload_evidence(name, identity)})
    k4.check_prior_rows("W5", tmp_path)
    with pytest.raises(ValueError, match="W6 is unavailable"):
        k4.check_prior_rows("W6", tmp_path)
    with pytest.raises(ValueError, match="W6 is unavailable"):
        k4.fixture("W6", tmp_path, identity)
    comparison = tmp_path / "W4/comparison.json"
    current = json.loads(comparison.read_bytes())
    for rejected in ({**current, "protocol": k4.defaults.CACHE_ON_PROTOCOL},
                     {**current, "workload": {**current["workload"], "output_tokens": 1024}},
                     {**current, "workload": {**current["workload"],
                                              "request_sha256": ["0" * 64]}}):
        write_json(comparison, rejected)
        with pytest.raises(ValueError, match="W4 must pass"):
            k4.check_prior_rows("W5", tmp_path)


def test_current_wire_replay_retains_staggered_first_outputs():
    raw, lanes = synthetic_cohort()
    policy = k4.concurrent.reference()["policy"]
    duration, ids, metrics = k4.replay_cohort(raw, "W3", "measured-01", lanes, policy, None)
    assert duration == 95
    assert ids == [request["output_ids"] for request in raw["requests"]]
    assert metrics["all_generating_overlap_ms"] < 0
    raw["requests"][1]["sent_ns"] = raw["requests"][0]["done_ns"]
    with pytest.raises(ValueError, match="overlap bounds"):
        k4.replay_cohort(raw, "W3", "measured-01", lanes, policy, None)
