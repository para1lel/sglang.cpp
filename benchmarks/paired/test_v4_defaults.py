import json
import subprocess
from pathlib import Path

import pytest

import default_evidence as defaults
import k4
import workload_suite as suite
from makespan import K4_MAKESPAN_CRITERION
from saved_evidence import vllm_settings


GPU = "GPU-00000000-0000-0000-0000-000000000000"
BUDGET = suite.V4_TOKEN_BUDGET
IDENTITY = {
    "protocol": suite.PROTOCOL_V4, "gpu_uuid": GPU, "participating_gpus": [GPU],
    "max_running_requests": 8, "max_num_batched_tokens": BUDGET,
    "makespan_criterion": K4_MAKESPAN_CRITERION,
}


def test_v4_w4_can_start_and_only_its_certificate_opens_w5(tmp_path):
    identity = tmp_path / "identity.json"
    k4.save(identity, IDENTITY)
    k4.check_prior_rows("W4", tmp_path)
    with pytest.raises(ValueError, match="W4 must pass"):
        k4.check_prior_rows("W5", tmp_path)
    with pytest.raises(ValueError, match="W1 must pass"):
        k4.check_prior_rows("W2", tmp_path)
    with pytest.raises(ValueError, match="unavailable"):
        k4.check_prior_rows("W6", tmp_path)

    w4 = tmp_path / "W4"
    w4.mkdir()
    (w4 / "identity.json").write_bytes(identity.read_bytes())
    (w4 / "exit-code.txt").write_text("0\n")
    comparison = IDENTITY | {"row": "W4", "passed": True,
                             "campaign_sha256": k4.digest(identity),
                             "workload": k4.workload_evidence("W4", IDENTITY)}
    for invalid in ({"protocol": suite.PROTOCOL}, {"max_num_batched_tokens": 8192},
                    {"workload": {}}, {"passed": False}):
        k4.save(w4 / "comparison.json", comparison | invalid)
        with pytest.raises(ValueError, match="W4 must pass"):
            k4.check_prior_rows("W5", tmp_path)
    k4.save(w4 / "comparison.json", comparison)
    k4.check_prior_rows("W5", tmp_path)


def test_v4_identity_requires_budget():
    suite.validate_identity(IDENTITY)
    with pytest.raises(ValueError, match="Wrong v4 campaign token budget"):
        suite.validate_identity(IDENTITY | {"max_num_batched_tokens": 8192})
    assert defaults.token_budget(IDENTITY) == 32768
    assert defaults.token_budget(IDENTITY | {"protocol": suite.PROTOCOL}) == 8192


def test_v4_requires_explicit_observed_vllm_scheduled_cap(tmp_path):
    scheduler = {"max_num_batched_tokens": BUDGET, "max_num_seqs": 8,
                 "max_num_scheduled_tokens_configured": None}
    args = ["--max-num-batched-tokens", str(BUDGET), "--async-scheduling"]
    path = tmp_path / "effective-worker.json"
    k4.save(path, {"results": [{"scheduler": scheduler}]})
    with pytest.raises(ValueError, match="explicit 32768 scheduled cap"):
        vllm_settings(tmp_path, args, "W4", paired_budget=BUDGET)
    scheduler["max_num_scheduled_tokens_configured"] = BUDGET
    k4.save(path, {"results": [{"scheduler": scheduler}]})
    with pytest.raises(ValueError, match="differs from worker observation"):
        vllm_settings(tmp_path, [*args, "--max-num-scheduled-tokens", "8192"],
                      "W4", paired_budget=BUDGET)


def test_v4_reconciles_api_and_worker_graph_and_scheduler():
    scheduler = {"max_num_batched_tokens": BUDGET, "max_num_scheduled_tokens": BUDGET,
                 "max_num_seqs": 8, "enable_chunked_prefill": True, "async_scheduling": True}
    graph = {"cudagraph_capture_sizes": [1, 8, 8192], "max_cudagraph_capture_size": 8192,
             "max_prefill_cudagraph_capture_tokens": 8192,
             "prefill_cudagraph_capture_token_sizes": [8, 8192],
             "max_decode_cudagraph_capture_requests": 8,
             "max_decode_cudagraph_capture_tokens": 8192,
             "decode_cudagraph_capture_request_sizes": [1, 8]}
    api = {"scheduler_config": scheduler, "compilation_config": graph | {"mode": 0,
                                                                        "cudagraph_mode": [2, 1]}}
    worker = {"scheduler": scheduler.copy(), "compilation": graph.copy()}
    defaults.vllm_budget(api, worker, BUDGET)
    del worker["compilation"]["max_prefill_cudagraph_capture_tokens"]
    with pytest.raises(ValueError, match="CUDA graph settings"):
        defaults.vllm_budget(api, worker, BUDGET)
    worker["compilation"] = graph | {"cudagraph_capture_sizes": [1, 8]}
    with pytest.raises(ValueError, match="CUDA graph settings"):
        defaults.vllm_budget(api, worker, BUDGET)
    scheduler["async_scheduling"] = False
    with pytest.raises(ValueError, match="scheduler settings"):
        defaults.vllm_budget(api, worker, BUDGET)


def test_v4_rejects_graph_overrides_from_flags_or_environment():
    args = ["--compilation-config", json.dumps(defaults.GRAPH_CONFIG)]
    environment = dict.fromkeys(defaults.VLLM_ENVIRONMENT, "1")
    defaults.verify_vllm_graph_launch(args, environment)
    with pytest.raises(ValueError, match="graph launch or environment override"):
        defaults.verify_vllm_graph_launch(
            [*args, "--max-cudagraph-capture-size", "40"], environment)
    with pytest.raises(ValueError, match="graph launch or environment override"):
        defaults.verify_vllm_graph_launch(args, environment | {"VLLM_USE_BREAKABLE_CUDAGRAPH": "0"})


@pytest.mark.parametrize("proxy", ["", "http://127.0.0.1:7890"])
def test_v4_comparator_proxy_is_explicit_and_recorded(tmp_path, monkeypatch, proxy):
    monkeypatch.setenv("PAIRED_VLLM_HTTP_PROXY", proxy)
    monkeypatch.setenv("HTTP_PROXY", "http://inherited.invalid:9999")
    monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", "1")
    common = Path(__file__).with_name("common.sh")
    script = r'''
set -euo pipefail
source "$1"
seed_cache=$2/seed
mkdir "$seed_cache"
native_defaults=true cache_on=true context=2560 kv_bytes=10737418240 max_seqs=8
client=(true) uv=/usr/bin/env python=/usr/bin/python3
export CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=PCI_BUS_ID
start_group() {
    local run=$1
    shift
    local environment=()
    while [[ $1 != "$python" ]]; do
        environment+=("$1")
        shift
    done
    "${environment[@]}" /usr/bin/env -0 >"$run/live-environment.bin"
}
start_vllm "$2"
'''
    subprocess.run(["bash", "-c", script, "bash", str(common), str(tmp_path)],
                   check=True, timeout=15)
    live = dict(item.split(b"=", 1) for item in
                (tmp_path / "live-environment.bin").read_bytes().split(b"\0") if item)
    receipt = dict(line.split(b"=", 1) for line in
                   (tmp_path / "effective-environment.txt").read_bytes().splitlines())
    assert live == receipt
    assert b"PAIRED_VLLM_HTTP_PROXY" not in live
    assert b"VLLM_USE_BREAKABLE_CUDAGRAPH" not in live
    keys = (b"http_proxy", b"https_proxy", b"all_proxy",
            b"HTTP_PROXY", b"HTTPS_PROXY", b"ALL_PROXY")
    if proxy:
        assert all(live[key] == proxy.encode() for key in keys)
        assert live[b"no_proxy"] == live[b"NO_PROXY"] == b"localhost,127.0.0.1,::1"
    else:
        assert all(key not in live for key in (*keys, b"no_proxy", b"NO_PROXY"))
