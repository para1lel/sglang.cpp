"""Exercise default-only launch evidence and preserve distinct K4 protocols."""

import gzip
import json
import os
import shlex
from pathlib import Path

import pytest

from tests.test_paired_identity import GPU, OTHER_GPU, k4, write_command, write_compute, write_json
from tests.test_paired_identity import write_k4_row, write_receipts, write_saved_worker


def write_default_vllm(run, checkpoint):
    write_saved_worker(run, checkpoint, k4.concurrent.reference(), 8192)
    args = shlex.split((run / "server-command.sh").read_text())[6:]
    argv = ["/venv/bin/python", "-B", *args, "--host", "127.0.0.1",
            "--port", "17424", "--served-model-name", "benchmark",
            "--speculative-config", '{"method":"mtp","num_speculative_tokens":4}']
    snapshot = run / "effective-worker.json"
    data = json.loads(snapshot.read_bytes())
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
        "attention_config": {"use_trtllm_attention": True,
                             "disable_flashinfer_q_quantization": True},
    }})
    return argv


def write_default_native(run, arm):
    flags = ["--port", "17424"]
    if arm == "native-k4-trace":
        flags.append("--trace-batches")
    info = json.loads((run / "server-info.json").read_bytes())
    info.update(launch_command=" ".join(["sglang-server", *flags]), dp_size=1, tp_size=1, pp_size=1)
    write_json(run / "server-info.json", info)
    write_compute(run, GPU, 123)
    return ["/bin/sglang-server", *flags]


def write_default_row(root, native_ns=70):
    row = root / "W1"
    row.mkdir()
    write_k4_row(row, native_ns, receipt_version=1)
    protocol = row / "protocol.sha256"
    protocol.write_text("captured protocol manifest\n")
    checkpoint = "/model"
    write_json(row / "identity.json", {
        "protocol": k4.defaults.PROTOCOL, "native_profile": k4.defaults.PROFILE,
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
        argv = (write_default_vllm(run, checkpoint) if arm == "vllm-k4"
                else write_default_native(run, arm))
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
            with gzip.open(measurement / f"{tag}.json.gz", "wt") as stream:
                json.dump(raw, stream)
    return row


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
