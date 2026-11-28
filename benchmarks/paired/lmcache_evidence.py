"""Bind the native benchmark arm to its private LMCache daemon."""

import argparse
import hashlib
import json
import os
import re
import socket
import struct
import subprocess
from pathlib import Path
from time import monotonic, sleep
from zipfile import ZipFile

from saved_evidence import command, compute_owners, digest, hash_manifest, load, require
from workload_suite import (
    LMCACHE_DRAIN_SUFFIX_PROTOCOL,
    LMCACHE_GPU_PROTOCOLS,
    LMCACHE_NATIVE_READY,
    LMCACHE_SLC3_PROTOCOLS,
    LMCACHE_SLC3_READY,
    LMCACHE_VERSION,
)

FRAME = struct.Struct("!4sIQ")
SOCKET_NAME = "sglang-lmcache.sock"
CAPACITY_BYTES = 64 << 30
DAEMON_SOURCES = ("serve.sh", "sglang_lmcache.py", "sglang_gpu_transfer.py", "requirements.txt")
WHEEL_SHA256 = "4ec775bda3a8c5f1f5b1ef0bbc441b666db86b57c3c2a7bcba47a4f6550b07cd"


def ready(path, pid, wire="SLC2", namespace="sglang"):
    expected_magic = wire.encode()
    expected = (LMCACHE_NATIVE_READY if wire == "SLC3" else LMCACHE_VERSION).encode()
    before = os.lstat(path)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(1)
        connection.connect(str(path))
        peer, uid, _ = struct.unpack("3i", connection.getsockopt(
            socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i")))
        if uid != os.geteuid() or peer != pid:
            raise ValueError("LMCache Unix socket peer differs from the launched daemon")
        encoded = namespace.encode("ascii")
        body = struct.pack("!H", len(encoded)) + encoded
        connection.sendall(FRAME.pack(expected_magic, 1, len(body)) + body)
        with connection.makefile("rb") as reply:
            header = reply.read(FRAME.size)
            require(len(header) == FRAME.size, "Incomplete LMCache readiness header")
            magic, status, size = FRAME.unpack(header)
            payload = reply.read(size) if size == len(expected) else b""
            require((magic, status, size) == (expected_magic, 0, len(expected))
                    and payload == expected and reply.read(1) == b"",
                    "Unexpected LMCache readiness response")
    require(os.path.samestat(before, os.lstat(path)), "LMCache readiness endpoint was replaced")
    stat = (Path("/proc") / str(peer) / "stat").read_text().rsplit(")", 1)[1].split()
    receipt = {"socket": str(path), "version": LMCACHE_VERSION, "namespace": namespace,
               "group_pid": pid, "peer_pid": peer, "peer_uid": uid,
               "peer_start_ticks": int(stat[19]), "frame": (header + payload).hex()}
    if wire == "SLC3":
        receipt.update(wire=wire, ready_version=payload.decode(),
                       socket_id=f"{before.st_dev}:{before.st_ino}")
    return receipt


def wait_ready(path, pid, wire="SLC2", namespace="sglang"):
    deadline = monotonic() + 90
    while True:
        os.kill(pid, 0)
        try:
            return ready(path, pid, wire, namespace)
        except (FileNotFoundError, ConnectionRefusedError, TimeoutError):
            if monotonic() >= deadline:
                raise
            sleep(0.1)


def verify(run, environment, identity):
    daemon = load(run / "lmcache-ready.json")
    direct_gpu = identity["protocol"] in LMCACHE_GPU_PROTOCOLS
    directory = Path(environment["XDG_RUNTIME_DIR"])
    require(directory.is_absolute() and daemon["socket"] == str(directory / SOCKET_NAME)
            and daemon["version"] == LMCACHE_VERSION and daemon["namespace"] == "sglang"
            and all(type(daemon[key]) is int and daemon[key] > 0
                    for key in ("group_pid", "peer_pid"))
            and daemon["group_pid"] == int((run / "lmcache-pid.txt").read_text()),
            "Native default endpoint differs from the private LMCache daemon")
    port = str(identity["lmcache_port"])
    cpus = {"lmcache": identity["client_cpus"], "memory": identity["memory"]}
    argv = command(run, "lmcache", cpus)
    require(len(argv) == 10 and argv[0] == "bash"
            and Path(argv[1]).is_absolute() and Path(argv[1]).parts[-3:] == (
                "tools", "lmcache", "serve.sh")
            and argv[2:] == ["--socket", daemon["socket"], "--namespace", "sglang",
                             "--memory-gib", "64", "--port", port],
            "Unexpected private LMCache daemon launch")
    listeners = (run / "lmcache-listener.txt").read_text().splitlines()
    require(len(listeners) == 1 and listeners[0].split()[3] == f"127.0.0.1:{port}"
            and f'pid={daemon["group_pid"]},' in listeners[0],
            "LMCache daemon does not exclusively own its loopback listener")
    launch_log = (run / "lmcache.log").read_text()
    if direct_gpu:
        verify_gpu(run, identity, daemon, argv, launch_log)
        return
    require(f'sglang LMCache at {daemon["socket"]}, L1 capacity {CAPACITY_BYTES} bytes'
            in launch_log and (run / "lmcache-cuda-environment.txt").read_text()
            == "CUDA_VISIBLE_DEVICES=\n", "Incorrect LMCache storage capacity or CUDA visibility")
    digest, launcher = (run / "lmcache-launcher.sha256").read_text().strip().split(maxsplit=1)
    require(launcher == argv[1] and re.fullmatch(r"[0-9a-f]{64}", digest) is not None
            and f"{digest}  tools/lmcache/serve.sh" in (
                run.parent / "protocol.sha256").read_text().splitlines(),
            "LMCache launcher differs from the campaign source manifest")


def verify_wheel(wheel, python, pythonpath):
    require(digest(wheel) == WHEEL_SHA256, "LMCache wheel differs from official 0.5.8")
    script = """
from importlib.metadata import distribution
from importlib.util import find_spec
import json
dist = distribution('lmcache')
print(json.dumps([dist.version, str(dist.locate_file('')), find_spec('lmcache').origin]))
"""
    version, site, origin = json.loads(subprocess.check_output(
        [python, "-sPB", "-c", script], env={"PATH": "/usr/bin:/bin", "PYTHONPATH": pythonpath},
        text=True))
    require(version == LMCACHE_VERSION and Path(origin).resolve() ==
            (Path(site) / "lmcache/__init__.py").resolve(),
            "LMCache package import differs from the installed 0.5.8 distribution")
    with ZipFile(wheel) as archive:
        members = [entry for entry in archive.infolist() if not entry.is_dir()
                   and (entry.filename.startswith("lmcache/") or
                        entry.filename.endswith((".dist-info/METADATA", ".dist-info/WHEEL")))]
        for entry in members:
            require(hashlib.sha256(archive.read(entry)).hexdigest() ==
                    digest(Path(site) / entry.filename),
                    f"Installed LMCache differs from the official wheel: {entry.filename}")
    return {"version": version, "sha256": WHEEL_SHA256, "python": python,
            "pythonpath": pythonpath, "package_files": len(members), "module": origin}


def wheel_receipt(directory, identity):
    record = load(directory / "lmcache-wheel.json")
    require(record["sha256"] == WHEEL_SHA256 and record["version"] == LMCACHE_VERSION
            and record["python"] == identity["lmcache_python"]
            and record["pythonpath"] == identity["lmcache_pythonpath"]
            and record["package_files"] == 791
            and Path(record["module"]).is_absolute()
            and Path(record["module"]).parts[-2:] == ("lmcache", "__init__.py"),
            "Invalid official LMCache wheel and installation receipt")


def verify_gpu(run, identity, daemon, argv, launch_log):
    pid = daemon["group_pid"]
    native_pid = (run / "server-pid.txt").read_text().strip()
    uid = identity["lmcache_uid"]
    gpu = identity["gpu_uuid"]
    wheel_receipt(run.parent, identity)
    if identity["protocol"] in LMCACHE_SLC3_PROTOCOLS:
        expected = (
            LMCACHE_NATIVE_READY
            if identity["protocol"] == LMCACHE_DRAIN_SUFFIX_PROTOCOL
            else LMCACHE_SLC3_READY
        )
        require(
            daemon.get("wire") == identity.get("lmcache_wire") == "SLC3"
            and daemon.get("ready_version") == identity.get("lmcache_ready_version") == expected,
            "Incorrect SLC3 READY identity",
        )
        magic, version = b"SLC3", expected.encode()
    else:
        magic, version = b"SLC2", LMCACHE_VERSION.encode()
    frame = FRAME.pack(magic, 0, len(version)) + version
    require(daemon["peer_pid"] == pid and type(daemon["peer_uid"]) is int
            and daemon["peer_uid"] == uid and type(daemon["peer_start_ticks"]) is int
            and daemon["peer_start_ticks"] > 0 and daemon["frame"] == frame.hex(),
            f"LMCache readiness does not prove the launched {magic.decode()} daemon identity")
    device_hex = gpu.removeprefix("GPU-").replace("-", "").lower()
    log_entries = [re.sub(r"\x1b\[[0-9;]*m", "", line) for line in launch_log.splitlines()
                   if "sglang GPU LMCache at" in line]
    message = (f'sglang GPU LMCache at {daemon["socket"]}, device {device_hex}, '
               f"L1 capacity {CAPACITY_BYTES} bytes")
    require(len(log_entries) == 1 and (message + " (") in log_entries[0]
            and (run / "lmcache-cuda-environment.txt").read_text() ==
            f"CUDA_DEVICE_ORDER=PCI_BUS_ID\nCUDA_VISIBLE_DEVICES={gpu}\n",
            "Incorrect direct LMCache device, storage capacity or CUDA visibility")

    launcher = Path(argv[1])
    pythonpath = os.pathsep.join(filter(
        None, (str(launcher.parent), identity["lmcache_pythonpath"])))
    require((run / "lmcache-python-environment.txt").read_text() ==
            f'LMCACHE_PYTHON={identity["lmcache_python"]}\nPYTHONPATH={pythonpath}\n',
            "LMCache daemon Python environment differs from the verified installation")
    source = hash_manifest(run / "lmcache-source.sha256")
    expected = hash_manifest(run.parent / "protocol.sha256")
    sources = DAEMON_SOURCES
    if "tools/lmcache/sglang_socket.py" in expected:
        sources += ("sglang_socket.py",)
    require(source == {str(launcher.with_name(name)): expected.get(f"tools/lmcache/{name}")
                       for name in sources},
            "LMCache daemon sources differ from the campaign source manifest")
    verify_process(run, identity["lmcache_python"], identity["lmcache_port"], daemon["socket"])
    compute_owners(run / "compute-before-daemon.csv", gpu, ())
    compute_owners(run / "compute-before.csv", gpu, (pid,))
    compute_owners(run / "compute-ready.csv", gpu, (native_pid, pid))


def verify_process(
    run: Path, python: str, port: int, endpoint: str, *, memory_gib: int = 64
) -> None:
    process = (run / "lmcache-live-argv.bin").read_bytes().split(b"\0")
    args = list(map(os.fsdecode, process[:-1]))
    require(process[-1] == b"" and args[:4] == [python, "-m",
                                               "lmcache.v1.multiprocess.server", "--host"],
            "LMCache daemon is not running the official multiprocess server")
    expected_flags = [
        "127.0.0.1",
        "--port",
        str(port),
        "--supported-transfer-mode",
        "lmcache_driven",
        "--no-run-script-api-enabled",
        "--disable-metrics",
        "--l1-size-gb",
        str(memory_gib),
        "--l1-init-size-gb",
        str(memory_gib),
        "--eviction-policy",
        "LRU",
        "--l1-read-ttl-seconds",
        "60",
        "--l1-write-ttl-seconds",
        "60",
        "--server-module",
    ]
    require(
        args[4:-1] == expected_flags
        and json.loads(args[-1])
        == {"module_path": "sglang_lmcache", "config": {"socket": endpoint, "namespace": "sglang"}},
        "Unexpected official LMCache daemon arguments",
    )


def completed(run, identity):
    daemon = load(run / "lmcache-ready.json")
    require(load(run / "lmcache-ready-end.json") == daemon and
            (run / "lmcache-live-argv-end.bin").read_bytes() ==
            (run / "lmcache-live-argv.bin").read_bytes(),
            "LMCache daemon identity changed during measurement")
    compute_owners(run / "compute-measurement-end.csv", identity["gpu_uuid"],
                   ((run / "server-pid.txt").read_text().strip(), daemon["group_pid"]))
    retired(run, identity)


def pinned(run, identity, pid):
    gpu = identity["gpu_uuid"]
    compute_owners(run / "compute-before.csv", gpu, ())
    compute_owners(run / "compute-ready.csv", gpu, (pid,))
    compute_owners(run / "compute-measurement-end.csv", gpu, (pid,))
    retired(run, identity)


def retired(run, identity):
    compute_owners(run / "compute-after.csv", identity["gpu_uuid"], ())
    for name in ("listener-after.txt", "lmcache-listener-after.txt"):
        require(not (run / name).read_bytes(), f"{run}: listener remained after shutdown")
    if run.name != "vllm-k4":
        endpoint = Path(load(run / "lmcache-ready.json")["socket"])
        require(load(run / "lmcache-retired.json") == {
            "socket": str(endpoint), "runtime_absent": True, "socket_absent": True},
            "LMCache private runtime or endpoint remained after shutdown")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    actions = parser.add_subparsers(dest="action", required=True)
    readiness = actions.add_parser("ready")
    readiness.add_argument("socket", type=Path)
    readiness.add_argument("pid", type=int)
    readiness.add_argument("--once", action="store_true")
    readiness.add_argument("--wire", choices=("SLC2", "SLC3"), default="SLC2")
    readiness.add_argument("--namespace", default="sglang")
    ownership = actions.add_parser("owners")
    ownership.add_argument("csv", type=Path)
    ownership.add_argument("gpu")
    ownership.add_argument("pids", nargs="*")
    wheel = actions.add_parser("verify-wheel")
    wheel.add_argument("wheel", type=Path)
    wheel.add_argument("python")
    wheel.add_argument("pythonpath")
    args = parser.parse_args()
    if args.action == "ready":
        probe = ready if args.once else wait_ready
        print(json.dumps(probe(args.socket, args.pid, args.wire, args.namespace)))
    elif args.action == "owners":
        compute_owners(args.csv, args.gpu, args.pids)
    else:
        print(json.dumps(verify_wheel(args.wheel, args.python, args.pythonpath)))
