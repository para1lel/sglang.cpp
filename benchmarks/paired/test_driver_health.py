"""Reject measured cohorts whose server log records serious CUDA driver launch failures."""

import gzip
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
for entry in (str(ROOT), str(HERE), str(ROOT / "benchmarks" / "tp2")):
    if entry not in sys.path:
        sys.path.insert(0, entry)
import cohorts
import driver_health
from driver_health import driver_launch_failures, require_healthy_driver_log


ORACLE_LINE = ("CUDA Error: CUDA_ERROR_INVALID_HANDLE at ../fmha/fmhaKernels.cuh 526\n")
ORDINARY_LOG = ("sglang-server: tensor parallel ranks=2\n"
                "tp2_transport status=mnnvl ranks=2\n"
                "Listening on http://127.0.0.1:17424\n")


def test_counts_each_marker_form():
    text = (ORACLE_LINE + "CUDA_ERROR_INVALID_HANDLE\n"
            "cudaErrorInvalidResourceHandle cudaErrorInvalidHandle\n")
    assert driver_health.driver_launch_failures(text) == {
        "CUDA_ERROR_INVALID_HANDLE": 2, "cudaErrorInvalidResourceHandle": 1,
        "cudaErrorInvalidHandle": 1}


def test_benign_configuration_classes_pass():
    text = ("CUDA_ERROR_INVALID_VALUE during configuration probe\n"
            "cudaErrorLaunchOutOfResources from capacity probe\n"
            "warning: error ignored by caller\n")
    assert driver_health.driver_launch_failures(text) == {}


def test_rejection_message_carries_marker_and_counts():
    with pytest.raises(ValueError, match="native-k4: server log records driver launch failures"):
        driver_health.require_healthy_driver_log(
            "cudaErrorIllegalAddress cudaErrorIllegalAddress", "native-k4")


def test_healthy_text_passes():
    assert driver_health.require_healthy_driver_log(ORDINARY_LOG, "native-k4") is None


def arm(tmp_path, log=ORDINARY_LOG):
    run = tmp_path / "W1" / "native-k4"
    run.mkdir(parents=True)
    (run.parent.parent / "identity.json").write_text("{}")
    (run / "ready.json").write_text("{}")
    (run / "server.log").write_text(log)
    for tag in ("warmup-01", "measured-01"):
        raw = {"row": "W1", "tag": tag, "binding": cohorts.binding(run),
               "requests": [], "cold_cache": {}}
        with gzip.open(run / f"{tag}.json.gz", "xt") as stream:
            json.dump(raw, stream)
    return run


def replay_mocks(monkeypatch):
    monkeypatch.setattr(cohorts, "load_lanes", lambda *_: [])
    monkeypatch.setattr(cohorts, "validate_cohort", lambda raw, *_: {"makespan_ns": 1})
    monkeypatch.setattr(cohorts.cold_cache, "replay", lambda *_: None)
    monkeypatch.setattr(cohorts.cache_activity, "replay", lambda *_: None)


def test_replay_rejects_driver_launch_failures(tmp_path, monkeypatch):
    replay_mocks(monkeypatch)
    run = arm(tmp_path, ORDINARY_LOG + ORACLE_LINE * 3968)
    with pytest.raises(ValueError,
                       match="driver launch failures.*CUDA_ERROR_INVALID_HANDLE.: 3968"):
        cohorts.handle(run, fixtures=None)


def test_replay_requires_server_log(tmp_path, monkeypatch):
    replay_mocks(monkeypatch)
    run = arm(tmp_path)
    (run / "server.log").unlink()
    with pytest.raises(ValueError, match="native-k4: missing server log"):
        cohorts.handle(run, fixtures=None)


def test_replay_accepts_healthy_log(tmp_path, monkeypatch):
    replay_mocks(monkeypatch)
    assert cohorts.handle(arm(tmp_path), fixtures=None) == {
        "warmup-01": {"makespan_ns": 1}, "measured-01": {"makespan_ns": 1}}


def capture_mocks(monkeypatch, raw):
    monkeypatch.setattr(cohorts, "load_lanes", lambda *_: [])
    monkeypatch.setattr(cohorts.cold_cache, "before_cohort", lambda *_: {})
    monkeypatch.setattr(cohorts.cold_cache, "after_cohort", lambda *_: None)
    monkeypatch.setattr(cohorts.client, "capture_cohort",
                        lambda url, row, lanes, tag, policy: raw | {"tag": tag})
    monkeypatch.setattr(cohorts, "validate_cohort", lambda *_: {"makespan_ns": 1})
    monkeypatch.setattr(cohorts.cold_cache, "replay", lambda *_: None)
    monkeypatch.setattr(cohorts.cache_activity, "replay", lambda *_: None)


def test_capture_rejects_failures_flushed_during_cohort_and_preserves_raw(tmp_path, monkeypatch):
    run = arm(tmp_path)
    for tag in ("warmup-01", "measured-01"):
        (run / f"{tag}.json.gz").unlink()
    raw = {"row": "W1", "tag": "warmup-01", "requests": [], "errors": []}
    capture_mocks(monkeypatch, raw)
    healthy = run / "server.log"
    healthy.write_text(ORDINARY_LOG)

    def flush_during_capture(url, row, lanes, tag, policy):
        healthy.write_text(ORDINARY_LOG + ORACLE_LINE)
        return raw | {"tag": tag}

    monkeypatch.setattr(cohorts.client, "capture_cohort", flush_during_capture)
    with pytest.raises(ValueError, match="native-k4/warmup-01: server log records driver"):
        cohorts.handle(run, fixtures=None, endpoint="http://127.0.0.1:19790")
    assert (run / "warmup-01.json.gz").is_file()
    with gzip.open(run / "warmup-01.json.gz", "rt") as stream:
        assert json.load(stream)["tag"] == "warmup-01"


def test_capture_accepts_healthy_log(tmp_path, monkeypatch):
    run = arm(tmp_path)
    for tag in ("warmup-01", "measured-01"):
        (run / f"{tag}.json.gz").unlink()
    capture_mocks(monkeypatch, {"row": "W1", "requests": [], "errors": []})
    measured = cohorts.handle(run, fixtures=None, endpoint="http://127.0.0.1:19790")
    assert set(measured) == {"warmup-01", "measured-01"}


def test_probe_covers_early_return_path(tmp_path, monkeypatch):
    run = arm(tmp_path)
    (run / "warmup-01.json.gz").unlink()
    (run / "measured-01.json.gz").unlink()
    (run.parent.parent / "identity.json").write_text(json.dumps({"mode": "qualify"}))
    raw = {"row": "W1", "tag": "probe", "requests": [{}], "errors": [],
           "cold_cache": {"cleared": ""}}
    capture_mocks(monkeypatch, raw)
    monkeypatch.setattr(cohorts, "load_lanes", lambda *_: [({"request_sha256": "0" * 64}, {}, {})])
    monkeypatch.setattr(cohorts.cold_cache, "before_cohort", lambda *_: {"cleared": ""})
    monkeypatch.setattr(cohorts.cache_activity, "replay", lambda *_: {"k": 1})
    monkeypatch.setattr(cohorts.client, "replay_lane", lambda *_: None)
    monkeypatch.setattr(cohorts.cold_cache, "parse_metrics", lambda *_: {"k": 1})
    result = cohorts.handle(run, fixtures=None, endpoint="http://127.0.0.1:19790", probe=True)
    assert result["binding"] == cohorts.binding(run)
    assert result["rank_deltas"] == {"k": 0}
    (run / "server.log").write_text(ORDINARY_LOG + ORACLE_LINE)
    with pytest.raises(ValueError, match="native-k4: server log records driver"):
        cohorts.handle(run, fixtures=None, endpoint="http://127.0.0.1:19790", probe=True)

V2_BAD_LAUNCH = ("sglang: trtllm prefill launch failed: CUresult 1 (CUDA_ERROR_INVALID_VALUE)"
                 " at /data1/shared/dongwu.chen/environments/vllm-81a8cb063e62/venv/lib/"
                 "python3.12/site-packages/flashinfer/data/include/flashinfer/trtllm/fmha/"
                 "fmhaKernels.cuh:526\n")
CAPABILITY_NOISE = ("probe: CUDA_ERROR_INVALID_VALUE not supported by device, using fallback\n"
                    "cudaFuncGetAttributes returned CUDA_SUCCESS (0)\n"
                    "no launch failures recorded for this phase\n"
                    "rejected CUDA_ERROR_INVALID_VALUE_FOO variant name\n"
                    "capability query hit CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES, skipped fused path\n")


def test_real_v2_bad_launch_log_rejected():
    assert driver_launch_failures(V2_BAD_LAUNCH) == {"CUDA_ERROR_INVALID_VALUE": 1}


def test_launch_record_recognized_regardless_of_code():
    text = ("launch failed: CUresult 701 (CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES) at k.cu:88\n"
            "launch failed: CUresult 999 (CUDA_ERROR_SOME_NEW_CODE) at x.cu:1\n")
    assert driver_launch_failures(text) == {"CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES": 1,
                                            "CUDA_ERROR_SOME_NEW_CODE": 1}


def test_launch_record_counted_once_per_line():
    text = "launch failed: CUresult 719 (CUDA_ERROR_LAUNCH_FAILED) at a.cu:1\n"
    assert driver_launch_failures(text) == {"CUDA_ERROR_LAUNCH_FAILED": 1}


def test_capability_probe_noise_passes():
    assert driver_launch_failures(CAPABILITY_NOISE) == {}
    require_healthy_driver_log(CAPABILITY_NOISE, "noise")


def test_v2_bad_launch_rejected_through_replay(tmp_path, monkeypatch):
    replay_mocks(monkeypatch)
    run = arm(tmp_path, ORDINARY_LOG + V2_BAD_LAUNCH)
    with pytest.raises(ValueError, match="CUDA_ERROR_INVALID_VALUE.: 1"):
        cohorts.handle(run, fixtures=None)
