"""Exercise rank receipt reconciliation independently of expensive server startup."""

from hashlib import sha256

import pytest

from tp2_vllm import rank_identities, verify_checkpoint, verify_dispatch


DEVICES = ["gpu-00000000-0000-0000-0000-000000000001", "gpu-00000000-0000-0000-0000-000000000002"]


def identity(rank):
    return {"identity": {"rank": rank, "tp_rank": rank, "local_rank": rank, "cuda_device": rank,
                         "uuid": DEVICES[rank][4:], "device_name": "NVIDIA B200", "pid": 51 + rank}}


def test_rank_receipts_follow_identity_and_detect_end_of_run_changes(tmp_path):
    first, second = identity(0), identity(1)
    ready = tmp_path / "ready.csv"
    end = tmp_path / "end.csv"
    ready.write_text("".join(f"{uuid}, {51 + rank}, python, 100 MiB\n"
                             for rank, uuid in enumerate(DEVICES)))
    end.write_bytes(ready.read_bytes())
    assert rank_identities([second, first], DEVICES, [ready, end]) == [
        first["identity"], second["identity"]]
    with pytest.raises(ValueError, match="two observed"):
        rank_identities([first], DEVICES, [ready])
    with pytest.raises(ValueError, match="distinct matching"):
        rank_identities([first, first], DEVICES, [ready])
    with pytest.raises(ValueError, match="physical device"):
        rank_identities([first, second], DEVICES[::-1], [ready])
    end.write_text(ready.read_text().replace(", 52,", ", 93,"))
    with pytest.raises(ValueError, match="expected exactly compute PIDs"):
        rank_identities([first, second], DEVICES, [ready, end])


def test_collected_checkpoint_identifies_actual_model_bytes(tmp_path):
    model = tmp_path / "part"
    model.write_bytes(b"model")
    manifest = tmp_path / "checkpoint.sha256"
    manifest.write_text(f"{sha256(b'model').hexdigest()}  part\n")
    expected = sha256(manifest.read_bytes()).hexdigest()
    verify_checkpoint(tmp_path, manifest, expected)
    model.write_bytes(b"other")
    with pytest.raises(ValueError, match="digest mismatch"):
        verify_checkpoint(tmp_path, manifest, expected)


def test_worker_dispatch_requires_rank_local_dtype_and_draft_coverage():
    dispatch = {"prefill_dtype": "torch.bfloat16", "decode_dtype": "torch.bfloat16",
                "decode_backend": "trtllm-gen", "kv_dtype": "torch.float8_e4m3fn",
                "layers": [f"layer.{i}" for i in range(17)]}
    draft = dispatch["layers"][-1:]
    worker = identity(1) | {"draft_attention_layers": draft, "flashinfer_dispatch": {
        "runner": [dispatch], "speculator": [dispatch | {"layers": draft}]}}
    verify_dispatch(worker)
    dispatch["decode_backend"] = "xqa"
    with pytest.raises(ValueError, match="Rank 1"):
        verify_dispatch(worker)
    dispatch["decode_backend"] = "trtllm-gen"
    worker["flashinfer_dispatch"]["speculator"].clear()
    with pytest.raises(ValueError, match="one draft"):
        verify_dispatch(worker)
