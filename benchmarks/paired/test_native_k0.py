"""Check current native K0 receipts and the historical explicit-policy contract."""

import json
from pathlib import Path

import pytest

from identity import NATIVE_K0_PROTOCOL, native_k0_launch


CHECKPOINT = Path("/weights")
LAUNCH = ["sglang-server", "--checkpoint", str(CHECKPOINT), "--device", "0", "--port",
          "17393", "--batch-cuda-graphs", "--mtp-steps", "0", "--prefix-cache-entries",
          "0", "--max-active", "32", "--max-num-batched-tokens", "8192"]
SERVER_INFO = {
    "model_path": str(CHECKPOINT), "launch_command": " ".join(LAUNCH), "mtp_steps": 0,
    "host": "127.0.0.1", "device": 0, "port": 17393, "max_running_requests": 32,
    "max_num_batched_tokens": 8192, "prefix_cache_entries": 0, "batch_cuda_graphs": True,
    "eager": False, "trace_batches": False, "fp8_prefill": "tensor-core",
    "target_gqa": "grouped-split", "paired_prefill": False,
}


@pytest.mark.parametrize(("fp8", "target"), [("tensor-core", "grouped-split"), ("exact", "exact")])
def test_automatic_k0_replays_observed_policy(tmp_path, fp8, target):
    info = SERVER_INFO | {"fp8_prefill": fp8, "target_gqa": target}
    (tmp_path / "server-info.json").write_text(json.dumps(info))
    settings, policy = native_k0_launch(tmp_path, LAUNCH, CHECKPOINT, 8192, NATIVE_K0_PROTOCOL)
    assert "fp8-prefill" not in settings
    assert policy["protocol"] == NATIVE_K0_PROTOCOL
    assert policy["effective_policy"] == {
        "fp8_prefill": fp8, "target_gqa": target, "paired_prefill": False,
    }
    assert len(policy["server_info_sha256"]) == 64


@pytest.mark.parametrize(("field", "value"), [
    ("mtp_steps", 4), ("port", 17394), ("paired_prefill", True),
    ("fp8_prefill", "tensor-core-mtp-target"), ("target_gqa", None),
    ("launch_command", "sglang-server --fp8-prefill tensor-core"),
])
def test_automatic_k0_rejects_wrong_server_info(tmp_path, field, value):
    (tmp_path / "server-info.json").write_text(json.dumps(SERVER_INFO | {field: value}))
    with pytest.raises(ValueError, match="unexpected resolved native K0 settings"):
        native_k0_launch(tmp_path, LAUNCH, CHECKPOINT, 8192, NATIVE_K0_PROTOCOL)


def test_historical_k0_replays_explicit_launch_without_inventing_policy(tmp_path):
    historical = [*LAUNCH, "--fp8-prefill", "tensor-core"]
    settings, policy = native_k0_launch(tmp_path, historical, CHECKPOINT, 8192, None)
    assert settings["fp8-prefill"] == "tensor-core"
    assert not policy
    with pytest.raises(ValueError, match="retired native FP8 override"):
        native_k0_launch(tmp_path, historical, CHECKPOINT, 8192, NATIVE_K0_PROTOCOL)
