"""Check LMCache evidence without changing the historical APC protocol."""

import json
import os
import socket
import struct
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory

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
SLC3 = DIRECT | {"protocol": suite.LMCACHE_GPU_SLC3_PROTOCOL, "lmcache_wire": "SLC3",
                 "lmcache_ready_version": "0.5.8+native-v3"}
DRAIN = SLC3 | {"protocol": suite.LMCACHE_DRAIN_PROTOCOL}
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


def test_drain_campaign_keeps_existing_schedules_and_pinned_apc():
    suite.validate_identity(DRAIN)
    assert k4.cache_engine(DRAIN, "native-k4") == cold_cache.LMCACHE_DRAIN
    assert k4.cache_engine(DRAIN, "native-k4-trace") == cold_cache.LMCACHE_DRAIN
    assert k4.cache_engine(SLC3, "native-k4") == "lmcache"
    assert k4.cache_engine(DRAIN, "vllm-k4") == "vllm"
    assert k4.schedule("W4", "native-k4", DRAIN) == ["warmup-01", "measured-01"]


def drain_metric_text(**changes):
    values = dict.fromkeys((cold_cache.RUNNING, cold_cache.WAITING, cold_cache.HITS,
                           cold_cache.STORES, *cold_cache.LMCACHE_METRICS,
                           *cold_cache.LMCACHE_DRAIN_METRICS), 0)
    return "".join(f"{name} {value}\n" for name, value in (values | changes).items())


def drain_receipt(**changes):
    # The first final cursor lands on a periodic checkpoint. The other needs two stores.
    requests = [{"submit_ns": 40 + lane, "done_ns": 48 + lane, "eof_ns": 50 + lane,
                 "usage": {"prompt_tokens": 8192, "completion_tokens": completion}}
                for lane, completion in enumerate((1, 4))]
    outcomes = {cold_cache.STORES: 3, cold_cache.LMCACHE_STORES: 3,
                cold_cache.GQA_WRITTEN_TOKENS: 16387, cold_cache.LMCACHE_WRITE_NS: 17,
                **dict.fromkeys(cold_cache.LMCACHE_WRITTEN_BYTES, 12)}
    receipt = {"before": drain_metric_text(), "cleared": drain_metric_text(),
               "after": drain_metric_text(**(outcomes | changes)),
               "reset": {"status": 200, "body": '{"success":true}'},
               "reset_started_ns": 10, "reset_completed_ns": 20, "ready_ns": 30,
               "capture_finished_ns": 60, "settled_ns": 85}
    return {"cold_cache": receipt, "requests": requests}


def test_native_drain_checks_settled_writes_without_extending_wire_makespan():
    raw = drain_receipt()
    snapshots = cold_cache.replay(raw, cold_cache.LMCACHE_DRAIN)
    assert snapshots["after"][cold_cache.LMCACHE_STORES] == 3
    assert k4.cohort_makespan_ns(raw["requests"]) == 9
    with pytest.raises(ValueError, match="between settled cohorts"):
        cold_cache.replay(raw, cold_cache.LMCACHE_DRAIN, snapshots)


@pytest.mark.parametrize(("change", "error"), [
    ({cold_cache.PENDING_STORES: 1}, "pending LMCache stores"),
    ({cold_cache.LMCACHE_STORES: 2}, "expected number"),
    ({cold_cache.GQA_WRITTEN_TOKENS: 16386}, "every completed token"),
    ({cold_cache.LMCACHE_WRITTEN_BYTES[-1]: 0}, "all three transfer families"),
    ({cold_cache.DECLINED_STORES: 1}, "declined a checkpoint"),
    ({cold_cache.LMCACHE_REUSE[0]: 1}, "reported an I/O error"),
    *(({counter: 1}, "issued a restore") for counter in cold_cache.LMCACHE_READ_ACTIVITY),
])
def test_native_drain_rejects_incomplete_or_failed_transfers(change, error):
    with pytest.raises(ValueError, match=error):
        cold_cache.replay(drain_receipt(**change), cold_cache.LMCACHE_DRAIN)


def test_native_drain_rejects_absent_gauge_and_premature_settlement():
    with pytest.raises(ValueError, match="Missing cache metrics"):
        cold_cache.replay(cold_receipt(), cold_cache.LMCACHE_DRAIN)
    raw = drain_receipt()
    raw["cold_cache"]["settled_ns"] = 49
    with pytest.raises(ValueError, match="overlaps the timed cohort"):
        cold_cache.replay(raw, cold_cache.LMCACHE_DRAIN)


def test_native_drain_waits_before_reset_and_after_done(monkeypatch):
    expected_requests = iter([
        ("GET", "/metrics", drain_metric_text(**{cold_cache.PENDING_STORES: 2})),
        ("GET", "/metrics", drain_metric_text()),
        ("POST", "/reset_prefix_cache", '{"success":true}'),
        ("GET", "/metrics", drain_metric_text()),
        ("GET", "/metrics", drain_metric_text(**{cold_cache.PENDING_STORES: 1})),
        ("GET", "/metrics", drain_receipt()["cold_cache"]["after"]),
    ])

    def request(url, method, path, body=None):
        expected_method, expected_path, response = next(expected_requests)
        assert (method, path) == (expected_method, expected_path)
        return {"status": 200, "body": response}

    monkeypatch.setattr(cold_cache, "request", request)
    monkeypatch.setattr(cold_cache, "sleep", lambda seconds: None)
    receipt = cold_cache.before_cohort(None, cold_cache.LMCACHE_DRAIN)
    cold_cache.after_cohort(None, cold_cache.LMCACHE_DRAIN, receipt)
    assert receipt["settled_ns"] > receipt["capture_finished_ns"] > receipt["ready_ns"]
    assert cold_cache.metrics(receipt["after"], cold_cache.LMCACHE_DRAIN)[cold_cache.STORES] == 3
    assert next(expected_requests, None) is None


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


@pytest.mark.parametrize(("wire", "magic", "payload", "matches"), [
    ("SLC2", b"SLC2", b"0.5.8", True),
    ("SLC2", b"SLC1", b"0.5.8", False),
    ("SLC2", b"SLC3", b"0.5.8+native-v3", False),
    ("SLC3", b"SLC3", b"0.5.8+native-v3", True),
    ("SLC3", b"SLC2", b"0.5.8", False),
    ("SLC3", b"SLC3", b"0.5.8", False),
])
def test_direct_ready_checks_an_actual_socket_frame(tmp_path, wire, magic, payload, matches):
    endpoint = tmp_path / "daemon.sock"
    frame = lmcache_evidence.FRAME
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
        listener.settimeout(2)
        listener.bind(str(endpoint))
        listener.listen()

        def serve():
            connection, _ = listener.accept()
            with connection, connection.makefile("rb") as request:
                assert frame.unpack(request.read(frame.size)) == (wire.encode(), 1, 8)
                assert request.read(8) == struct.pack("!H", 6) + b"sglang"
                connection.sendall(frame.pack(magic, 0, len(payload)) + payload)

        with ThreadPoolExecutor(max_workers=1) as pool:
            response = pool.submit(serve)
            if matches:
                ready = lmcache_evidence.ready(endpoint, os.getpid(), wire)
                assert ready["version"] == "0.5.8"
                if wire == "SLC3":
                    assert (ready["wire"], ready["ready_version"]) == (wire, payload.decode())
                else:
                    assert "wire" not in ready and "ready_version" not in ready
                assert ready["peer_uid"] == os.geteuid()
                assert ready["peer_pid"] == os.getpid() and ready["peer_start_ticks"] > 0
            else:
                with pytest.raises(ValueError, match="Unexpected LMCache readiness response"):
                    lmcache_evidence.ready(endpoint, os.getpid(), wire)
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


def test_slc3_replay_binds_package_version_and_ready_frame(direct_receipt):
    run, environment = direct_receipt
    suite.validate_identity(SLC3)
    for wrong in ({"lmcache_version": "0.5.8+native-v3"}, {"lmcache_wire": "SLC2"},
                  {"lmcache_ready_version": "0.5.8"}):
        with pytest.raises(ValueError, match="pinned daemon|SLC3 campaign"):
            suite.validate_identity(SLC3 | wrong)
    with pytest.raises(ValueError, match="Incorrect SLC3 READY identity"):
        lmcache_evidence.verify(run, environment, SLC3)

    ready = k4.load(run / "lmcache-ready.json")
    payload = b"0.5.8+native-v3"
    ready.update(wire="SLC3", ready_version=payload.decode())
    k4.save(run / "lmcache-ready.json", ready)
    with pytest.raises(ValueError, match="launched SLC3 daemon identity"):
        lmcache_evidence.verify(run, environment, SLC3)

    ready["frame"] = (lmcache_evidence.FRAME.pack(b"SLC3", 0, len(payload)) + payload).hex()
    for end in ("", "-end"):
        k4.save(run / f"lmcache-ready{end}.json", ready)
    lmcache_evidence.verify(run, environment, SLC3)
    lmcache_evidence.completed(run, SLC3)
    with pytest.raises(ValueError, match="launched SLC2 daemon identity"):
        lmcache_evidence.verify(run, environment, DIRECT)


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


def test_process_owns_receipt_before_starting_command(tmp_path):
    paired = Path(__file__).resolve().parent
    script = '''
set -eu
source "$1/common.sh"
launch_session /dev/full touch "$2/started" &
pid=$!
if wait "$pid"; then exit 1; fi
[[ ! -e $2/started ]]
launch_session "$2/pid" bash -c '
    trap "wait; exit" TERM
    sleep 30 &
    echo "$!" > "$1/child"
    wait
' bash "$2" &
pid=$!
trap 'stop_session "$pid"' EXIT
for ((i = 0; i < 100; ++i)); do
    [[ ! -s $2/child ]] || break
    sleep .02
done
read -r receipt < "$2/pid"
read -r child < "$2/child"
[[ $receipt == "$pid" && $(ps -o sid= "$child") -eq $pid ]]
stop_session "$pid"
trap - EXIT
! kill -0 -- "-$pid" 2>/dev/null
'''
    subprocess.run(["bash", "-c", script, "bash", str(paired), str(tmp_path)],
                   check=True, timeout=18)


@pytest.mark.parametrize("before_setsid", [False, True])
def test_stop_session_before_command_exec(tmp_path, before_setsid):
    paired = Path(__file__).resolve().parent
    receipt = tmp_path / "pid"
    environment = os.environ.copy()
    if before_setsid:
        wrapper = tmp_path / "python"
        wrapper.write_text('#!/bin/sh\necho waiting > "$PAUSE_FIFO.ready"\n'
                           'read -r _ < "$PAUSE_FIFO"\n')
        wrapper.chmod(0o700)
        environment.update(PAIRED_CLIENT_PYTHON=str(wrapper), PAUSE_FIFO=str(receipt))
    os.mkfifo(receipt)
    script = '''
set -eu
source "$1/common.sh"
launch_session "$2/pid" touch "$2/started" &
pid=$!
trap 'stop_session "$pid"' EXIT
for ((i = 0; i < 100; ++i)); do
    sid=$(ps -o sid= "$pid")
    if [[ -n ${PAUSE_FIFO:-} && -s $2/pid.ready ]] || [[ $sid -eq $pid ]]; then break; fi
    sleep .02
done
if [[ -n ${PAUSE_FIFO:-} ]]; then
    [[ -s $2/pid.ready && $sid -ne $pid ]]
else
    [[ $sid -eq $pid ]]
fi
stop_session "$pid"
trap - EXIT
[[ ! -d /proc/$pid && ! -e $2/started ]]
'''
    subprocess.run(["bash", "-c", script, "bash", str(paired), str(tmp_path)],
                   check=True, env=environment, timeout=10)


def test_lmcache_path_resolution_survives_arm_directory_change(tmp_path):
    paired = Path(__file__).resolve().parent
    for directory, value in [("deps", "verified"), ("arm/deps", "wrong")]:
        package = tmp_path / directory / "lmcache"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text(f"identity = '{value}'\n")
    script = '''
set -eu
source "$1/common.sh"
source "$1/lmcache_arm.sh"
[[ -z $(resolve_lmcache_pythonpath "") ]]
path=$(resolve_lmcache_pythonpath "deps::/opt/site")
[[ $path == "$PWD/deps:$PWD:/opt/site" ]]
cd arm
PYTHONPATH="$path" "$client_python" -sPB -c 'import lmcache; assert lmcache.identity == "verified"'
'''
    subprocess.run(["bash", "-c", script, "bash", str(paired)], cwd=tmp_path,
                   check=True, timeout=10)


@pytest.mark.parametrize("unexpected", [False, True])
def test_stop_lmcache_retires_private_runtime_lock(unexpected):
    paired = Path(__file__).resolve().parent
    listener = '''
import signal
import socketserver
import sys
sys.path.insert(0, sys.argv[1])
from sglang_socket import OwnedUnixStreamServer
server = OwnedUnixStreamServer(sys.argv[2], socketserver.BaseRequestHandler)
print("ready", flush=True)
signal.pause()
'''
    script = '''
set -eu
source "$1/common.sh"
source "$1/lmcache_arm.sh"
lmcache_runtime=$(mktemp -d "$2/arm.XXXXXXXX")
endpoint=$lmcache_runtime/sglang-lmcache.sock
listener=("$client_python" -IB -c "$3" "$project/tools/lmcache" "$endpoint")
launch_session "$2/pid" "${listener[@]}" >"$2/ready" &
lmcache_pid=$!
trap 'if [[ -n $lmcache_pid ]]; then stop_session "$lmcache_pid"; fi' EXIT
for ((i = 0; i < 100; ++i)); do
    [[ -s $2/ready ]] && break
    sleep .02
done
[[ -s $2/ready ]]
lmcache_socket_id=$(stat -c '%d:%i' -- "$endpoint")
runtime=$lmcache_runtime
if [[ $4 == keep ]]; then
    touch "$runtime/retain"
    if stop_lmcache; then exit 1; fi
    [[ -f $runtime/retain && ! -e $endpoint ]]
else
    stop_lmcache
    [[ ! -e $runtime && -z $lmcache_runtime ]]
fi
[[ -z $lmcache_pid && -z $lmcache_socket_id ]]
'''
    with TemporaryDirectory(dir="/tmp") as work:
        subprocess.run(["bash", "-c", script, "bash", str(paired), work, listener,
                        "keep" if unexpected else "clean"], check=True, timeout=20)
