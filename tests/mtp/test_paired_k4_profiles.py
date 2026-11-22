"""Exercise new K4 dispatch receipts and historical profile replay offline."""

import gzip
import hashlib
import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_paired_identity import ROOT, k4, replace_stream_ids, synthetic_cohort, write_json
from test_paired_identity import write_k4_row, write_k4_runtime_receipts, write_opt_in_arm
from test_paired_identity import write_opt_in_identity


@pytest.mark.parametrize(("row", "profile"), [
    ("W1", None), ("W1", k4.ORDERED_PROFILE), ("W1", k4.SPLIT_PROFILE),
    ("W1", k4.LONG_SPLIT_PROFILE), ("W2", k4.LONG_SPLIT_PROFILE),
])
def test_k4_v2_replays_new_profiles(tmp_path, row, profile):
    write_k4_row(tmp_path, opt_in=profile is not None, profile=profile, receipt_version=2, row=row)
    result = k4.assess(row, tmp_path, tmp_path)
    assert result["passed"]
    if profile is not None:
        config = result["native_config"]
        assert result["native_profile"] == profile
        assert result["native_k0_parity"]
        assert not result["exact_k0_compatibility"]["all_ids_match"]
        assert (k4.SPLIT_FIELD in config) is (profile == k4.SPLIT_PROFILE)
        assert (k4.LONG_SPLIT_FIELD in config) is (profile == k4.LONG_SPLIT_PROFILE)


@pytest.mark.parametrize("version", [0, 1])
def test_k4_long_profile_requires_v2(tmp_path, version):
    write_opt_in_identity(tmp_path, k4.LONG_SPLIT_PROFILE, version)
    with pytest.raises(ValueError, match="Long split requires live dispatch receipts"):
        k4.campaign(tmp_path)


@pytest.mark.parametrize("other", [k4.ORDERED_PROFILE, k4.SPLIT_PROFILE, "exact"])
def test_k4_long_rejects_substituted_campaign_profile(tmp_path, other):
    identity, _ = write_opt_in_identity(tmp_path, k4.LONG_SPLIT_PROFILE, 2)
    identity["native_profile"] = other
    write_json(tmp_path / "identity.json", identity)
    with pytest.raises(ValueError, match="Unknown opt-in native campaign configuration"):
        k4.campaign(tmp_path)


@pytest.mark.parametrize(("arm", "file", "wrong", "error"), [
    ("native-k4", "server-command.sh", b"1", "wrong native dispatch environment"),
    ("native-k0", "live-dispatch.bin", b"1", "live native dispatch differs"),
    ("native-k4-trace", "live-dispatch-end.bin", b"1", "live native dispatch differs"),
    ("native-k4", "live-dispatch.bin", b"", "live native dispatch differs"),
    ("native-exact-k0", "live-dispatch.bin", b"long", "live native dispatch differs"),
    ("native-k4", "live-dispatch.bin", b"long\0SGLANG_EXPERIMENTAL_TARGET_SPLIT_GQA=long",
     "live native dispatch differs"),
])
def test_k4_long_rejects_substituted_process_mode(tmp_path, arm, file, wrong, error):
    write_k4_row(tmp_path, opt_in=True, profile=k4.LONG_SPLIT_PROFILE, receipt_version=2)
    path = tmp_path / arm / file
    prefix = b"SGLANG_EXPERIMENTAL_TARGET_SPLIT_GQA="
    if file.endswith(".sh"):
        path.write_bytes(path.read_bytes().replace(prefix + b"long", prefix + wrong))
    else:
        path.write_bytes(prefix + wrong + b"\0" if wrong else b"")
    with pytest.raises(ValueError, match=error):
        k4.assess("W1", tmp_path, tmp_path)


def test_k4_bounded_rejects_long_process_mode(tmp_path):
    write_k4_row(tmp_path, opt_in=True, profile=k4.SPLIT_PROFILE, receipt_version=2)
    path = tmp_path / "native-k0" / "live-dispatch.bin"
    path.write_bytes(b"SGLANG_EXPERIMENTAL_TARGET_SPLIT_GQA=long\0")
    with pytest.raises(ValueError, match="live native dispatch differs"):
        k4.assess("W1", tmp_path, tmp_path)


@pytest.mark.parametrize("fault", ["bounded", "both", "missing", "integer"])
def test_k4_long_rejects_wrong_effective_mode(tmp_path, fault):
    write_k4_row(tmp_path, opt_in=True, profile=k4.LONG_SPLIT_PROFILE, receipt_version=2)
    path = tmp_path / "native-k4" / "server-info.json"
    info = json.loads(path.read_bytes())
    if fault in ("bounded", "both"):
        info[k4.SPLIT_FIELD] = True
    if fault in ("bounded", "missing"):
        del info[k4.LONG_SPLIT_FIELD]
    if fault == "integer":
        info[k4.LONG_SPLIT_FIELD] = 1
    write_json(path, info)
    with pytest.raises(ValueError, match="wrong native server configuration"):
        k4.assess("W1", tmp_path, tmp_path)


@pytest.mark.parametrize("receipt", ["live-dispatch.bin", "live-dispatch-end.bin", "cohort"])
def test_k4_long_requires_dispatch_binding(tmp_path, receipt):
    write_k4_row(tmp_path, opt_in=True, profile=k4.LONG_SPLIT_PROFILE, receipt_version=2)
    run = tmp_path / "native-k4"
    if receipt != "cohort":
        (run / receipt).unlink()
        with pytest.raises(FileNotFoundError):
            k4.assess("W1", tmp_path, tmp_path)
    else:
        path = run / "measurement" / "measured-01.json.gz"
        cohort = k4.read_raw(path.parent, "measured-01")
        del cohort["binding"]["native_dispatch_sha256"]
        with gzip.open(path, "wt") as stream:
            json.dump(cohort, stream)
        with pytest.raises(ValueError, match="Wrong raw cohort campaign or arm identity"):
            k4.assess("W1", tmp_path, tmp_path)


@pytest.mark.parametrize("mode", [None, "1", "long", ""])
def test_k4_native_dispatch_cli_reads_only_the_named_live_setting(mode):
    name = "SGLANG_EXPERIMENTAL_TARGET_SPLIT_GQA"
    env = {name + "_UNRELATED": "unused", "OTHER": "unused"}
    if mode is not None:
        env[name] = mode
    with subprocess.Popen(["/bin/cat"], env=env, stdin=subprocess.PIPE) as child:
        result = subprocess.run(
            [sys.executable, "-B", str(ROOT / "benchmarks/paired/k4.py"), "native-dispatch",
             str(child.pid)], cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT)},
            capture_output=True, check=True)
    assert result.stdout == (f"{name}={mode}".encode() + b"\0" if mode is not None else b"")


@pytest.mark.parametrize(("row", "profile", "fault"), [
    ("W3", profile, fault)
    for profile in (k4.ORDERED_PROFILE, k4.SPLIT_PROFILE, k4.LONG_SPLIT_PROFILE)
    for fault in (None, "native-drift", "missing-eof", "trailing-data")
] + [(row, k4.LONG_SPLIT_PROFILE, None) for row in ("W4", "W5", "W6")])
def test_k4_opt_in_replays_concurrent_lanes(tmp_path, monkeypatch, row, profile, fault):
    cohort, lanes = synthetic_cohort()
    monkeypatch.setattr(k4.concurrent, "load_lanes", lambda *_: lanes)
    version = 2 if profile == k4.LONG_SPLIT_PROFILE else 0
    identity_data, fingerprint = write_opt_in_identity(tmp_path, profile, version)
    arms = ("native-k0", "native-exact-k0", "native-k4", "vllm-k4")
    for arm in arms:
        directory = tmp_path / arm / "measurement"
        directory.mkdir(parents=True)
        write_opt_in_arm(directory.parent, arm, identity_data)
    if version:
        write_k4_runtime_receipts(tmp_path, arms, opt_in=True, receipt_version=version)
    for arm in arms:
        directory = tmp_path / arm / "measurement"
        binding = {"campaign_sha256": fingerprint, "arm": arm}
        if version and arm != "vllm-k4":
            for key, name in (("argv", "live-argv.bin"), ("dispatch", "live-dispatch.bin")):
                binding[f"native_{key}_sha256"] = hashlib.sha256(
                    (directory.parent / name).read_bytes()).hexdigest()
        for lane, body, _ in lanes:
            (directory / f'{lane["lane"]:02d}.request.json').write_bytes(body)
        for tag in k4.schedule(row, arm):
            raw = deepcopy(cohort)
            raw.update(row=row, tag=tag, binding=binding)
            if arm in ("native-exact-k0", "vllm-k4"):
                replace_stream_ids(raw["requests"][1], 25)
            if arm == "native-k4" and tag == "measured-01" and fault == "native-drift":
                replace_stream_ids(raw["requests"][1], 50)
            if arm == "vllm-k4" and tag == "measured-01":
                record = raw["requests"][1]
                if fault == "missing-eof":
                    del record["eof_ns"]
                if fault == "trailing-data":
                    record["events"].append({"ns": 101, "read": 5, "lines": ["data:"]})
            with gzip.open(directory / f"{tag}.json.gz", "wt") as stream:
                json.dump(raw, stream)
    if fault:
        errors = {"native-drift": (ValueError, "native K4 differs from native K0"),
                  "missing-eof": (KeyError, "eof_ns"),
                  "trailing-data": (ValueError, "Data after")}
        kind, message = errors[fault]
        with pytest.raises(kind, match=message):
            k4.assess(row, tmp_path, tmp_path)
        return
    result = k4.assess(row, tmp_path, tmp_path)
    assert result["passed"] is False
    assert result["workloads"][row]["native"]["median_ns"] == 95
    assert result["workloads"][row]["vllm"]["median_ns"] == 95
    lanes = result["exact_k0_compatibility"]["lanes"]
    assert [lane["different_ids"] for lane in lanes] == [0, 3]
