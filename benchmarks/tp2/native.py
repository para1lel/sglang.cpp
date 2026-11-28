"""Reconcile the native single-process TP2 group with its shared two-device daemon."""

import argparse
import csv
import json
import re
from pathlib import Path

import default_evidence as defaults
import lmcache_evidence as cache
from saved_evidence import UUID, command, compute_owners, hash_manifest, load, require
from workload_suite import HBM_POOL_DEFAULTS, TP2_POOL_PROTOCOL

RANK = re.compile(r"tp2 rank=(\d+) device=(\d+) uuid=([0-9a-fA-F-]+) pid=(\d+) nccl=ok "
                  r"weights=(\d+) lm_head=([01]) mtp=([01])")
TP2_RANKS = 2
COMPUTE_FIELDS = 4


def owners(
    path: Path,
    devices: list[str],
    daemon: int,
    server: int | None = None,
    *,
    before_transfer: bool = False,
) -> None:
    processes = (daemon,) if server is None else (daemon, server)
    require(
        len(set(processes)) == len(processes)
        and all(type(pid) is int and pid > 0 for pid in processes),
        "Invalid process ownership identities",
    )
    selected = {gpu.lower() for gpu in devices}
    require(
        len(devices) == len(selected) == TP2_RANKS and all(UUID.fullmatch(gpu) for gpu in selected),
        "Invalid GPU ownership identities",
    )
    pids = {str(pid) for pid in processes}
    allowed = {(gpu, pid) for gpu in selected for pid in pids}
    required = allowed if not before_transfer else {row for row in allowed if row[1] != str(daemon)}
    with path.open(encoding="utf-8", newline="") as stream:
        rows = [row for row in csv.reader(stream, skipinitialspace=True, strict=True) if row]
    require(
        all(
            len(row) == COMPUTE_FIELDS
            and UUID.fullmatch(row[0].strip())
            and re.fullmatch(r"[1-9][0-9]*", row[1].strip())
            and row[2].strip()
            and row[3].strip()
            for row in rows
        ),
        f"{path}: malformed compute inventory",
    )
    observed = {(row[0].strip().lower(), row[1].strip()) for row in rows}
    require(len(observed) == len(rows), f"{path}: duplicate compute ownership rows")
    found = {(gpu, pid) for gpu, pid in observed if gpu in selected or pid in pids}
    require(
        required <= found <= allowed,
        f"{path}: required ownership {sorted(required)}, allowed {sorted(allowed)}, "
        f"observed {sorted(found)}",
    )


def rank_table(text, devices, pid):
    ranks = {}
    require([line for line in text.splitlines() if line.startswith("tp2 world=")]
            == ["tp2 world=2 ranks=2"],
            "Missing established native two-rank communicator")
    for line in text.splitlines():
        if not line.startswith("tp2 rank="):
            continue
        match = RANK.fullmatch(line)
        require(match, "Malformed native rank line")
        rank, device, uuid, actual_pid, weight_bytes, head, mtp = match.groups()
        rank = int(rank)
        require(rank in (0, 1) and rank not in ranks and int(device) == rank
                and uuid.lower().replace("-", "") == devices[rank][4:].lower().replace("-", "")
                and int(actual_pid) == pid and int(weight_bytes) > 0
                and int(head) == int(mtp) == (rank == 0), "Incorrect native TP rank identity")
        ranks[rank] = {"rank": rank, "cuda_device": rank, "uuid": devices[rank], "pid": pid,
                       "nccl": True, "weight_bytes": int(weight_bytes), "lm_head": bool(int(head)),
                       "mtp": bool(int(mtp))}
    require(set(ranks) == {0, 1}, "Expected two populated native ranks")
    return [ranks[0], ranks[1]]


def verify_daemon(run: Path, identity: dict, runtime: Path, *, completed: bool) -> int:
    directory = run / "daemon"
    daemon = load(directory / "lmcache-ready.json")
    version = identity["lmcache_ready_version"].encode()
    namespace = "sglang"
    port = identity["daemon_port"]
    capacity = identity["lmcache_capacity_bytes"]
    require(capacity == 128 << 30, "Wrong shared LMCache capacity")
    pid = int((directory / "lmcache-pid.txt").read_text())
    endpoint = str(runtime / cache.SOCKET_NAME)
    frame = cache.FRAME.pack(identity["lmcache_wire"].encode(), 0, len(version)) + version
    require(
        daemon["socket"] == endpoint
        and daemon["namespace"] == namespace
        and daemon["wire"] == identity["lmcache_wire"]
        and daemon["ready_version"].encode() == version
        and daemon["frame"] == frame.hex()
        and daemon["peer_pid"] == daemon["group_pid"] == pid
        and daemon["peer_start_ticks"] > 0
        and daemon["peer_uid"] == identity["lmcache_uid"],
        "Incorrect shared cache peer",
    )
    gpus = ",".join(identity["devices"])
    require(
        (directory / "lmcache-cuda-environment.txt").read_text()
        == f"CUDA_DEVICE_ORDER=PCI_BUS_ID\nCUDA_VISIBLE_DEVICES={gpus}\n",
        "Wrong daemon GPU visibility",
    )
    uuids = ",".join(gpu[4:].replace("-", "").lower() for gpu in identity["devices"])
    marker = f"sglang GPU LMCache at {endpoint}, devices {uuids}, L1 capacity {capacity} bytes"
    require(
        marker in (directory / "lmcache.log").read_text(),
        "Wrong shared daemon transfer devices or capacity",
    )
    (listener,) = (directory / "lmcache-listener.txt").read_text().splitlines()
    require(
        listener.split()[3] == f"127.0.0.1:{port}" and f"pid={pid}," in listener,
        "Wrong shared daemon listener",
    )
    cache.verify_process(directory, identity["lmcache_python"], port, endpoint, memory_gib=128)
    sources = hash_manifest(directory / "lmcache-source.sha256")
    project = Path(__file__).resolve().parents[2]
    manifest = hash_manifest(run.parent.parent / "protocol.sha256")
    expected = {str(project / name): value for name, value in manifest.items()
                if name.startswith("tools/lmcache/")}
    require(sources == expected and sources, "Daemon code differs from the fixed campaign sources")
    paths = (str(project / "tools/lmcache"), identity["lmcache_pythonpath"])
    pythonpath = ":".join(filter(None, paths))
    require((directory / "lmcache-python-environment.txt").read_text() ==
            f'LMCACHE_PYTHON={identity["lmcache_python"]}\nPYTHONPATH={pythonpath}\n',
            "Daemon imported a different interpreter or package path")
    if completed:
        require(load(directory / "lmcache-ready-end.json") == daemon and
                (directory / "lmcache-live-argv.bin").read_bytes() ==
                (directory / "lmcache-live-argv-end.bin").read_bytes(), "Cache peer changed")
    return pid


def graph_policy(info, base):
    graphs = info.get("batch_cuda_graphs")
    require(type(graphs) is bool, "Expected a Boolean native batch graph policy")
    for ready in base.glob("W[1-5]/native-k4*/ready.json"):
        if ready.stat().st_size:
            require(load(ready)["config"]["batch_cuda_graphs"] is graphs,
                    "Native graph policy changed across campaign arms")
    return graphs


def verify(run, completed=False):
    base = run.parent.parent
    identity = load(base / "identity.json")
    devices = identity["devices"]
    info = load(run / ("server-info-end.json" if completed else "server-info.json"))
    traced = run.name == "native-k4-trace"
    flags = ["--port", str(identity["port"]), "--tp", "2", "--max-active", "16"]
    if traced:
        flags.append("--trace-batches")
    required = {"device": 0, "tp_size": 2, "dp_size": 1, "pp_size": 1, "mtp_steps": 4,
                "max_running_requests": 16, "max_num_batched_tokens": 32768, "eager": False,
                "batch_cuda_graphs": graph_policy(info, base), "trace_batches": traced,
                "prefix_cache_entries": 0,
                "lmcache_enabled": True, "lmcache_available": True, "port": identity["port"],
                "host": "127.0.0.1", "model_path": identity["checkpoint_path"]}
    if identity["protocol"] == TP2_POOL_PROTOCOL:
        required.update(HBM_POOL_DEFAULTS)
    require(all(type(info.get(k)) is type(v) and info[k] == v for k, v in required.items())
            and info["launch_command"].split() == ["sglang-server", *flags],
            "Unexpected native config")
    raw = (run / "live-environment.bin").read_bytes()
    env = defaults.process_environment(raw, lmcache=True)
    cpus = {"server": identity["server_cpus"], "memory": "localalloc"}
    wrapper = command(run, "server", cpus)
    args = [identity["binary"], *flags]
    require(wrapper[:2] == ["env", "-i"] and wrapper[-len(args):] == args
            and sorted(wrapper[2:-len(args)]) == sorted(f"{k}={v}" for k, v in env.items())
            and env["CUDA_VISIBLE_DEVICES"] == ",".join(devices)
            and env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID",
            "Native clean environment differs from the launch")
    defaults.verify_argv(run, args)
    defaults.verify_cpu(run, cpus["server"])
    pid = int((run / "server-pid.txt").read_text())
    require((run / "running-binary.sha256").read_text().split() ==
            [identity["binary_sha256"], f"/proc/{pid}/exe"], "Native running binary mismatch")
    cache.wheel_receipt(base, identity)
    ranks = rank_table((run / "server.log").read_text(), devices, pid)
    daemon = verify_daemon(run, identity, Path(env["XDG_RUNTIME_DIR"]), completed=completed)
    for gpu in devices:
        compute_owners(run / "compute-before-daemons.csv", gpu, ())
    owners(run / "compute-before.csv", devices, daemon, before_transfer=True)
    owners(run / "compute-ready.csv", devices, daemon, pid, before_transfer=True)
    if completed:
        owners(run / "compute-measurement-end.csv", devices, daemon, pid)
        require((run / "live-environment-end.bin").read_bytes() == raw,
                "Native environment changed")
    return {
        "stage": "completed" if completed else "ready",
        "ranks": ranks,
        "daemon_pid": daemon,
        "config": required,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    phases = parser.add_mutually_exclusive_group()
    phases.add_argument("--preflight", action="store_true")
    phases.add_argument("--completed", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        devices = load(args.run.parent.parent / "identity.json")["devices"]
        daemon = int((args.run / "daemon/lmcache-pid.txt").read_text(encoding="utf-8"))
        owners(args.run / "compute-before.csv", devices, daemon, before_transfer=True)
    else:
        print(json.dumps(verify(args.run, args.completed), indent=2))
