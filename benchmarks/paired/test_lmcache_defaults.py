"""Check LMCache evidence without changing the historical APC protocol."""

import json
import os
import socket
import struct
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import cold_cache
import default_evidence
import k4
import lmcache_evidence
import workload_suite as suite


GPU = "GPU-00000000-0000-0000-0000-000000000000"
IDENTITY = {"protocol": suite.LMCACHE_PROTOCOL, "gpu_uuid": GPU, "participating_gpus": [GPU],
            "max_running_requests": 8, "max_num_batched_tokens": 32768, "port": 17425,
            "lmcache_port": 17524, "lmcache_version": "0.5.8", "sampling": "single-round-v1"}
DIRECT = IDENTITY | {"protocol": suite.LMCACHE_GPU_PROTOCOL, "lmcache_capacity_bytes": 64 << 30,
                     "lmcache_uid": os.geteuid(), "lmcache_python": "/frozen/bin/python",
                     "lmcache_pythonpath": "/frozen/site-packages", "client_cpus": "10-11",
                     "memory": "localalloc"}
OTHER_GPU = GPU[:-1] + "1"


def test_lmcache_timing_uses_new_protocol_and_original_round_contract():
    suite.validate_identity(IDENTITY)
    assert k4.cache_engine(IDENTITY, "native-k4") == "lmcache"
    assert k4.cache_engine(IDENTITY, "vllm-k4") == "vllm"
    assert k4.cache_engine(IDENTITY | {"protocol": suite.PROTOCOL_V5}, "native-k4") == "native"
    for arm in ("native-k4", "vllm-k4"):
        assert k4.schedule("W4", arm, IDENTITY) == ["warmup-01", "measured-01"]
    for bad in ({"lmcache_port": 17425}, {"lmcache_version": "0.5.7"}, {"sampling": None}):
        with pytest.raises(ValueError, match="LMCache campaign"):
            suite.validate_identity(IDENTITY | bad)


def cold_receipt(**after_changes):
    values = dict.fromkeys((cold_cache.RUNNING, cold_cache.WAITING, cold_cache.HITS,
                           cold_cache.STORES, *cold_cache.LMCACHE_METRICS), 0)
    after = values | {cold_cache.STORES: 1} | after_changes

    def text(sample):
        return "".join(f"{name} {value}\n" for name, value in sample.items())

    return {"requests": [{"submit_ns": 40, "response_end": {"ns": 50}}], "cold_cache": {
        "before": text(values), "cleared": text(values), "after": text(after),
        "reset": {"status": 200, "body": '{"success": true}'},
        "reset_started_ns": 10, "reset_completed_ns": 20,
        "ready_ns": 30, "capture_finished_ns": 60}}


def test_cold_lmcache_replay_detects_errors_reuse_and_unacknowledged_stores():
    assert cold_cache.replay(cold_receipt(), "lmcache")["after"][cold_cache.STORES] == 1
    for counter in (cold_cache.HITS, *cold_cache.LMCACHE_REUSE):
        with pytest.raises(ValueError, match="hit counter|reused state"):
            cold_cache.replay(cold_receipt(**{counter: 1}), "lmcache")
    with pytest.raises(ValueError, match="did not acknowledge"):
        cold_cache.replay(cold_receipt(**{cold_cache.STORES: 0}), "lmcache")


def test_pinned_vllm_must_have_local_apc_without_offload():
    config = {"cache_config": {"enable_prefix_caching": True, "kv_offloading_size": None},
              "kv_transfer_config": None}
    default_evidence.require_vllm_local_apc(config, ["vllm", "serve"])
    candidates = [config | {"kv_transfer_config": {}}, config | {"cache_config": {
        "enable_prefix_caching": True, "kv_offloading_size": 8}}, {"cache_config": {
        "enable_prefix_caching": True, "kv_offloading_size": None}}]
    for wrong in candidates:
        with pytest.raises(ValueError, match="without|no KV offload"):
            default_evidence.require_vllm_local_apc(wrong, ["vllm", "serve"])
    with pytest.raises(ValueError, match="no KV offload"):
        default_evidence.require_vllm_local_apc(config, ["--kv-offloading-size=1"])


def test_lmcache_evidence_binds_native_environment_and_own_daemon(tmp_path):
    run = tmp_path / "native-k4"
    run.mkdir()
    endpoint = tmp_path / "private" / lmcache_evidence.SOCKET_NAME
    environment = {"XDG_RUNTIME_DIR": str(endpoint.parent)}
    identity = IDENTITY | {"client_cpus": "10-11", "memory": "localalloc"}
    k4.save(run / "lmcache-ready.json", {"socket": str(endpoint), "version": "0.5.8",
                                       "namespace": "sglang", "group_pid": 35, "peer_pid": 35})
    (run / "lmcache-pid.txt").write_text("35\n")
    (run / "lmcache-cuda-environment.txt").write_text("CUDA_VISIBLE_DEVICES=\n")
    (run / "lmcache.log").write_text(
        f"sglang LMCache at {endpoint}, L1 capacity {64 << 30} bytes\n")
    listener = run / "lmcache-listener.txt"
    listener.write_text('LISTEN 0 128 127.0.0.1:17524 0.0.0.0:* users:(("python",pid=35,fd=7))\n')
    launcher = Path("/frozen/tools/lmcache/serve.sh")
    digest = "2" * 64
    (run / "lmcache-launcher.sha256").write_text(f"{digest}  {launcher}\n")
    (tmp_path / "protocol.sha256").write_text(f"{digest}  tools/lmcache/serve.sh\n")
    (run / "lmcache-command.sh").write_text(
        f"numactl --localalloc taskset --cpu-list 10-11 bash {launcher} --socket {endpoint} "
        "--namespace sglang --memory-gib 64 --port 17524\n")
    lmcache_evidence.verify(run, environment, identity)
    with pytest.raises(ValueError, match="endpoint differs"):
        lmcache_evidence.verify(run, {"XDG_RUNTIME_DIR": "/different"}, identity)
    listener.write_text(listener.read_text().replace("pid=35", "pid=36"))
    with pytest.raises(ValueError, match="exclusively own"):
        lmcache_evidence.verify(run, environment, identity)


@pytest.mark.parametrize("magic", [b"SLC2", b"SLC1"])
def test_direct_ready_checks_an_actual_socket_frame(tmp_path, magic):
    endpoint = tmp_path / "daemon.sock"
    frame = lmcache_evidence.FRAME
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
        listener.settimeout(2)
        listener.bind(str(endpoint))
        listener.listen()

        def serve():
            connection, _ = listener.accept()
            with connection, connection.makefile("rb") as request:
                assert frame.unpack(request.read(frame.size)) == (b"SLC2", 1, 8)
                assert request.read(8) == struct.pack("!H", 6) + b"sglang"
                connection.sendall(frame.pack(magic, 0, 5) + b"0.5.8")

        with ThreadPoolExecutor(max_workers=1) as pool:
            response = pool.submit(serve)
            if magic == b"SLC2":
                ready = lmcache_evidence.ready(endpoint, os.getpid())
                assert ready["peer_uid"] == os.geteuid()
                assert ready["peer_pid"] == os.getpid() and ready["peer_start_ticks"] > 0
            else:
                with pytest.raises(ValueError, match="Unexpected LMCache readiness response"):
                    lmcache_evidence.ready(endpoint, os.getpid())
            response.result(timeout=2)


def test_direct_ready_rejects_actual_wrong_socket_owner(tmp_path):
    endpoint = tmp_path / "other.sock"
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
        listener.bind(str(endpoint))
        listener.listen()
        with pytest.raises(ValueError, match="peer differs"):
            lmcache_evidence.ready(endpoint, os.getpid() + 1)


def process_line(pid, uuid=GPU):
    return f"{uuid}, {pid}, process, 1 MiB\n"


@pytest.fixture
def direct_receipt(tmp_path):
    run = tmp_path / "native-k4"
    run.mkdir()
    endpoint = tmp_path / "runtime" / lmcache_evidence.SOCKET_NAME
    source = Path("/frozen/tools/lmcache")
    daemon = {"socket": str(endpoint), "version": "0.5.8", "namespace": "sglang",
              "group_pid": 35, "peer_pid": 35, "peer_uid": DIRECT["lmcache_uid"],
              "peer_start_ticks": 2, "frame": (lmcache_evidence.FRAME.pack(b"SLC2", 0, 5)
                                                + b"0.5.8").hex()}
    k4.save(tmp_path / "lmcache-wheel.json", {
        "version": "0.5.8", "sha256": lmcache_evidence.WHEEL_SHA256,
        "python": DIRECT["lmcache_python"], "pythonpath": DIRECT["lmcache_pythonpath"],
        "package_files": 791, "module": "/frozen/site-packages/lmcache/__init__.py"})
    for end in ("", "-end"):
        k4.save(run / f"lmcache-ready{end}.json", daemon)
    k4.save(run / "lmcache-retired.json", {"socket": str(endpoint), "runtime_absent": True,
                                          "socket_absent": True})
    (tmp_path / "protocol.sha256").write_text("".join(
        f"{'2' * 64}  tools/lmcache/{name}\n" for name in lmcache_evidence.DAEMON_SOURCES))
    args = [DIRECT["lmcache_python"], "-m", "lmcache.v1.multiprocess.server",
            "--host", "127.0.0.1", "--port", "17524", "--supported-transfer-mode",
            "lmcache_driven", "--no-run-script-api-enabled", "--disable-metrics", "--l1-size-gb",
            "64", "--l1-init-size-gb", "64", "--eviction-policy", "LRU", "--l1-read-ttl-seconds",
            "60", "--l1-write-ttl-seconds", "60", "--server-module",
            json.dumps({"module_path": "sglang_lmcache", "config": {
                "socket": str(endpoint), "namespace": "sglang"}})]
    for end in ("", "-end"):
        (run / f"lmcache-live-argv{end}.bin").write_bytes(
            b"\0".join(map(os.fsencode, args)) + b"\0")
    files = {
        "lmcache-pid.txt": "35\n", "server-pid.txt": "36\n",
        "lmcache-cuda-environment.txt": (
            f"CUDA_DEVICE_ORDER=PCI_BUS_ID\nCUDA_VISIBLE_DEVICES={GPU}\n"),
        "lmcache-python-environment.txt": (
            "LMCACHE_PYTHON=/frozen/bin/python\n"
            "PYTHONPATH=/frozen/tools/lmcache:/frozen/site-packages\n"),
        "lmcache.log": f"sglang GPU LMCache at {endpoint}, device {'0' * 32}, "
                       f"L1 capacity {64 << 30} bytes (sglang_lmcache.py:440:sglang_lmcache)\n",
        "lmcache-listener.txt": 'LISTEN 0 128 127.0.0.1:17524 *:* users:(("python",pid=35,fd=7))\n',
        "lmcache-source.sha256": "".join(f"{'2' * 64}  {source / name}\n"
                                         for name in lmcache_evidence.DAEMON_SOURCES),
        "lmcache-command.sh": (
            f"numactl --localalloc taskset --cpu-list 10-11 bash {source / 'serve.sh'} "
            f"--socket {endpoint} --namespace sglang --memory-gib 64 --port 17524\n"),
        "compute-before-daemon.csv": process_line(40, OTHER_GPU),
        "compute-before.csv": process_line(35),
        "compute-ready.csv": process_line(35) + process_line(36),
        "compute-measurement-end.csv": process_line(35) + process_line(36),
        "compute-after.csv": process_line(40, OTHER_GPU),
        "listener-after.txt": "", "lmcache-listener-after.txt": "",
    }
    for name, text in files.items():
        (run / name).write_text(text)
    return run, {"XDG_RUNTIME_DIR": str(endpoint.parent)}


def test_direct_receipt_distinguishes_history_and_validates_lifecycle(direct_receipt):
    run, environment = direct_receipt
    suite.validate_identity(DIRECT)
    assert k4.cache_engine(DIRECT, "native-k4") == "lmcache"
    lmcache_evidence.verify(run, environment, DIRECT)
    lmcache_evidence.completed(run, DIRECT)
    with pytest.raises(ValueError, match="Incorrect LMCache storage"):
        lmcache_evidence.verify(run, environment, DIRECT | {"protocol": suite.LMCACHE_PROTOCOL})


def test_direct_replay_rejects_lazy_initial_allocation(direct_receipt):
    run, environment = direct_receipt
    path = run / "lmcache-live-argv.bin"
    path.write_bytes(path.read_bytes().replace(b"--l1-init-size-gb\x0064\x00",
                                               b"--l1-init-size-gb\x000\x00"))
    with pytest.raises(ValueError, match="Unexpected official LMCache daemon arguments"):
        lmcache_evidence.verify(run, environment, DIRECT)


@pytest.mark.parametrize(("name", "text", "error"), [
    ("compute-before-daemon.csv", process_line(35), "exactly compute PIDs"),
    ("compute-before.csv", process_line(37), "exactly compute PIDs"),
    ("compute-ready.csv", process_line(35) + process_line(36) + process_line(37),
     "exactly compute PIDs"),
    ("compute-ready.csv", process_line(35, OTHER_GPU) + process_line(36), "exactly compute PIDs"),
    ("compute-measurement-end.csv", process_line(36), "exactly compute PIDs"),
    ("compute-after.csv", process_line(35), "exactly compute PIDs"),
    ("lmcache-listener-after.txt", "unexpected listener", "listener remained"),
    ("lmcache-source.sha256", f"{'3' * 64}  /frozen/tools/lmcache/serve.sh\n", "daemon sources"),
    ("lmcache-cuda-environment.txt",
     f"CUDA_DEVICE_ORDER=PCI_BUS_ID\nCUDA_VISIBLE_DEVICES={OTHER_GPU}\n",
     "Incorrect direct LMCache device"),
])
def test_direct_replay_rejects_wrong_ownership_and_teardown(direct_receipt, name, text, error):
    run, environment = direct_receipt
    (run / name).write_text(text)
    with pytest.raises(ValueError, match=error):
        lmcache_evidence.verify(run, environment, DIRECT)
        lmcache_evidence.completed(run, DIRECT)


@pytest.mark.parametrize("change", [{"peer_pid": 37}, {"peer_uid": -1}, {"frame": "534c4331"}])
def test_direct_replay_rejects_wrong_peer_receipt(direct_receipt, change):
    run, environment = direct_receipt
    path = run / "lmcache-ready.json"
    k4.save(path, k4.load(path) | change)
    with pytest.raises(ValueError, match="SLC2 daemon identity"):
        lmcache_evidence.verify(run, environment, DIRECT)


def test_direct_pinned_replay_requires_solo_worker(direct_receipt):
    run = direct_receipt[0].with_name("vllm-k4")
    direct_receipt[0].rename(run)
    (run / "compute-before.csv").write_text("")
    for phase in ("ready", "measurement-end"):
        (run / f"compute-{phase}.csv").write_text(process_line(36))
    lmcache_evidence.pinned(run, DIRECT, 36)
    (run / "compute-measurement-end.csv").write_text(process_line(36) + process_line(35))
    with pytest.raises(ValueError, match="exactly compute PIDs"):
        lmcache_evidence.pinned(run, DIRECT, 36)
