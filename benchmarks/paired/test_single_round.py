"""Check live single-round qualification and archived schedule dispatch."""

import gzip
import json
import os
import subprocess
from pathlib import Path

import pytest

import k4
import workload_suite as suite
from makespan import K4_MAKESPAN_CRITERION, compare


ROOT = Path(__file__).resolve().parents[2]
GPU = "GPU-00000000-0000-0000-0000-000000000000"
IDENTITY = {"protocol": suite.PROTOCOL_V5, "gpu_uuid": GPU, "participating_gpus": [GPU],
            "max_running_requests": 8, "max_num_batched_tokens": 32768, "replay_receipts": 2,
            "sampling": k4.SINGLE_ROUND_SAMPLING,
            "makespan_criterion": K4_MAKESPAN_CRITERION}
TAGS = ["warmup-01", "measured-01"]


def test_current_rounds_and_archived_rounds():
    for row in suite.ROWS:
        for arm in ("native-k4", "vllm-k4"):
            assert k4.schedule(row, arm, IDENTITY) == TAGS
            assert len(k4.schedule(row, arm, {"protocol": suite.PROTOCOL_V4})) == (
                12 if row in ("W1", "W2") else 7)
    for arm in ("native-k0", "native-k4-trace", k4.EXACT_CONTROL):
        assert k4.schedule("W1", arm, IDENTITY) == ["control"]
        assert k4.schedule("W1", arm, {"protocol": suite.PROTOCOL_V4}) == ["control"]
    for protocol in (k4.PROTOCOL, k4.OPT_IN_PROTOCOL, *k4.defaults.DEFAULT_PROTOCOLS):
        if protocol == suite.PROTOCOL_V5:
            continue
        identity = {"protocol": protocol, "sampling": k4.SINGLE_ROUND_SAMPLING}
        for arm in ("native-k4", "vllm-k4"):
            assert k4.schedule("W1", arm, identity) == TAGS
            assert k4.schedule("W4", arm, identity) == TAGS


@pytest.mark.parametrize("profile,row", [
    ("server-defaults", "W4"), ("server-defaults-cache-on-v3", "W4"),
    ("server-defaults-cache-on-v2", "W6"), ("server-defaults-cache-off", "W6"),
    ("exact", "W6"), ("ordered-tc-int8", "W6"),
    ("ordered-tc-int8-split-gqa", "W6"), ("ordered-tc-int8-long-split-gqa", "W6"),
])
def test_profiles_still_reach_shared_round_validation(tmp_path, profile, row):
    manifest = tmp_path / "manifest"
    manifest.touch()
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(("SGLANG_", "PAIRED_"))}
    environment.update(PAIRED_K4_NATIVE_PROFILE=profile, PAIRED_WARMUPS="2",
                       PAIRED_NATIVE_SOURCE="1" * 40, PAIRED_NATIVE_SHA256="2" * 64,
                       PAIRED_CHECKPOINT_MANIFEST=str(manifest))
    output = tmp_path / "out"
    result = subprocess.run(["bash", str(ROOT / "benchmarks/paired/run_k4.sh"),
                             "/usr/bin/true", "7", str(output), row], env=environment,
                            capture_output=True, text=True)
    assert result.returncode == 2 and "exactly one warmup" in result.stderr
    assert not output.exists()


@pytest.mark.parametrize("different_ids", [False, True])
def test_v5_capture_requests_one_warm_and_one_measured(tmp_path, monkeypatch, different_ids):
    monkeypatch.setattr(k4, "campaign", lambda _: (IDENTITY, "fixture"))
    monkeypatch.setattr(k4, "receipt_identity", lambda *_: IDENTITY)
    monkeypatch.setattr(k4, "arm_binding", lambda *_: {})
    monkeypatch.setattr(k4, "fixture", lambda *_: [({"lane": 0}, b"fixture", {})])
    resets, requests = [], []
    monkeypatch.setattr(k4.cold_cache, "before_cohort", lambda *_: resets.append(1) or {})
    monkeypatch.setattr(k4.cold_cache, "after_cohort", lambda *_: None)
    monkeypatch.setattr(k4.cold_cache, "replay", lambda *_: None)
    monkeypatch.setattr(k4.streaming, "measure", lambda *args: requests.append(args) or {})

    def replay(_raw, _row, tag, *_args):
        return 100, [[1 if tag == "warmup-01" or not different_ids else 2]]

    monkeypatch.setattr(k4, "replay", replay)
    output = tmp_path / "measurement"
    if different_ids:
        with pytest.raises(ValueError, match="differ from the warmup"):
            k4.capture("W1", "native-k4", "http://127.0.0.1:1", tmp_path, output)
    else:
        k4.capture("W1", "native-k4", "http://127.0.0.1:1", tmp_path, output)
    assert len(resets) == len(requests) == 2
    assert sorted(p.name for p in output.glob("*.json.gz")) == [
        "measured-01.json.gz", "warmup-01.json.gz"]
    for tag in TAGS:
        assert json.loads(gzip.decompress((output / f"{tag}.json.gz").read_bytes()))["tag"] == tag


def test_v5_notice_and_first_miss_stop_next_row(tmp_path):
    identity = tmp_path / "identity.json"
    k4.save(identity, IDENTITY)
    k4.check_prior_rows("W4", tmp_path)
    w4 = tmp_path / "W4"
    w4.mkdir()
    (w4 / "identity.json").write_bytes(identity.read_bytes())
    common = IDENTITY | {"row": "W4", "campaign_sha256": k4.digest(identity),
                         "workload": k4.workload_evidence("W4", IDENTITY)}
    for sample, status in ((111, "1\n"), (110, "0\n")):
        result = compare({"W4": [sample]}, {"W4": [100]}, eligible=True,
                         criterion=K4_MAKESPAN_CRITERION)
        (w4 / "exit-code.txt").write_text(status)
        k4.save(w4 / "comparison.json", common | result)
        if status == "1\n":
            with pytest.raises(ValueError, match="W4 must pass"):
                k4.check_prior_rows("W5", tmp_path)
            (w4 / "exit-code.txt").write_text("0\n")
            k4.save(w4 / "comparison.json", common | result | {"passed": True})
            with pytest.raises(ValueError, match="makespan does not certify a pass"):
                k4.check_prior_rows("W5", tmp_path)
        else:
            k4.check_prior_rows("W5", tmp_path)


def test_live_collectors_reject_extra_rounds_before_launch(tmp_path):
    environment = os.environ | {"PYTHONPATH": str(ROOT), "PAIRED_WARMUPS": "2"}
    script = ROOT / "benchmarks/paired/measure_concurrent.py"
    result = subprocess.run(["python3", str(script), "W3", "http://127.0.0.1:1", str(tmp_path),
                             str(tmp_path / "out"), "--measured", "2"], env=environment,
                            capture_output=True, text=True)
    assert result.returncode == 2 and "invalid choice" in result.stderr
    result = subprocess.run(["bash", str(ROOT / "benchmarks/paired/run_vllm.sh"), "2",
                             str(tmp_path / "out"), "W3"], env=environment,
                            capture_output=True, text=True)
    assert result.returncode == 2 and "exactly one warmup" in result.stderr
    assert not (tmp_path / "out").exists()
