"""Check both rank transfers and generated IDs through a single physical LMCache daemon."""

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, contextmanager
from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic, sleep

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "tests"), str(ROOT / "tests" / "prefix"),
                str(ROOT / "tests" / "text")]
lmcache = import_module("lmcache_relay")
http = import_module("media.http_routes")

PREFIX = [760, 6511, 314, 9338, 369, 220, 17, 478, 220, 17, 283, 13, 198]
PROMPT = [*PREFIX, 760, 6511, 314, 6124, 369]


def free_port():
    with closing(socket.socket()) as sock:
        sock.bind(("127.0.0.1", 0))
        return str(sock.getsockname()[1])


@contextmanager
def daemon(args: argparse.Namespace, work: Path) -> Generator[tuple[Path, int]]:
    socket_path = work / "daemon.sock"
    log_path = args.evidence / "daemon.log"
    env = dict(
        os.environ,
        XDG_RUNTIME_DIR=str(work),
        PYTHONPATH=os.environ.get("LMCACHE_EXTRA_PYTHONPATH", ""),
    )
    command = [
        "bash",
        str(ROOT / "tools/lmcache/serve.sh"),
        "--socket",
        str(socket_path),
        "--port",
        free_port(),
        "--memory-gib",
        "2",
    ]
    with log_path.open("wb") as out:
        proc = subprocess.Popen(command, env=env, stdout=out, stderr=subprocess.STDOUT,
                                start_new_session=True)
        try:
            deadline = monotonic() + 60
            while monotonic() < deadline and proc.poll() is None:
                try:
                    reply = lmcache.control(socket_path, "sglang", lmcache.READY)
                    if reply == (0, lmcache.WIRE_VERSION):
                        break
                except (FileNotFoundError, ConnectionRefusedError):
                    pass
                sleep(0.1)
            else:
                raise RuntimeError(f"Shared daemon failed readiness, see {log_path}")
            yield socket_path, proc.pid
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()


def capture(args, port, phase, prompt, count, cached, salt):
    body = http.native_text(prompt, count) | {"cache_salt": salt}
    result = http.capture(args, phase, port, http.GENERATE, body)
    return http.native_result(result, prompt, count, cached)[:2]


def receipt(events, rank, gpu, opcode, endpoint):
    got = [entry for entry in events if entry["opcode"] == opcode]
    assert got and all(e["status"] == 0 for e in got), (rank, opcode, got)
    total = [sum(e["receipt"][i] for e in got) for i in range(4)]
    expected = [
        16896 * endpoint,
        76992512,
        (endpoint - 1) * 2112 if rank == 0 else 0,
        endpoint if opcode == lmcache.RANK_PUT else 0,
    ]
    assert total == expected, (rank, opcode, total, expected)
    families = {key[0] for e in got for key in e["keys"]}
    target = {lmcache.GQA, lmcache.GDN}
    assert families == (target | {lmcache.DRAFT} if rank == 0 else target), (rank, families)
    for e in got:
        assert e["device"] == gpu
        assert e["namespace"] == "sglang"
        assert e["allocations"] > 0 and e["forwarded"]
    return total


def metrics(port):
    status, _, raw = http.exchange(port, "/metrics")
    assert status == 200
    marker = '{engine="0"} '
    values = {}
    for line in raw.decode().splitlines():
        if line.startswith("sglang_cpp:lmcache_") and marker in line:
            key, value = line.removeprefix("sglang_cpp:lmcache_").split(marker)
            values[key] = int(value)
    return values


def pending(port):
    return metrics(port)["pending_stores"]


def check_counters(before, after, totals, action):
    direction = "read" if action == "get" else "written"
    operation = "restores" if action == "get" else "stores"
    for rank in range(2):
        key = f"rank{rank}_{operation}_total"
        assert after[key] - before[key] == 1, (key, before[key], after[key])
        for i, family in enumerate(("gqa", "gdn", "draft")):
            key = f"rank{rank}_{family}_{direction}_bytes_total"
            expected = totals[f"rank{rank}_{action}"][i]
            assert after[key] - before[key] == expected, (key, expected, before[key], after[key])
    for i, family in enumerate(("gqa", "gdn", "draft")):
        key = f"{family}_{direction}_bytes_total"
        expected = sum(totals[f"rank{rank}_{action}"][i] for rank in range(2))
        assert after[key] - before[key] == expected, (key, expected, before[key], after[key])


def drain(port):
    deadline = monotonic() + 20
    while monotonic() < deadline:
        if pending(port) == 0:
            return
        sleep(0.02)
    raise TimeoutError("Pending LMCache stores did not drain")


def delayed_store(
    args: argparse.Namespace, port: int, relay: lmcache.Relay, devices: list[str]
) -> tuple[int, dict[str, int], dict[str, int | dict[str, int]]]:
    offset = len(relay.since())
    baseline = metrics(port)
    barrier = relay.arm(lmcache.RANK_PUT, devices[1])
    with ThreadPoolExecutor(max_workers=1) as worker:
        prime = worker.submit(capture, args, port, "prime", PREFIX, 1, 0, "subject")
        try:
            barrier.wait()
            deadline = monotonic() + 10
            while monotonic() < deadline:
                completed = [e for e in relay.since(offset) if e.get("device") == devices[0]]
                midpoint = metrics(port)
                if (
                    any(e["opcode"] == lmcache.RANK_PUT and e["forwarded"] for e in completed)
                    and midpoint["rank0_draft_written_bytes_total"]
                    > baseline["rank0_draft_written_bytes_total"]
                ):
                    break
                sleep(0.02)
            else:
                raise AssertionError("Rank 0 store did not retire before rank 1")
            physical = receipt(completed, 0, devices[0], lmcache.RANK_PUT, len(PREFIX))
            deltas = {key: value - baseline[key] for key, value in midpoint.items()}
            expected = {"pending_stores": 1, "rank0_stores_total": 1, "rank1_stores_total": 0,
                        "stores_total": 0, "gqa_written_tokens_total": 0,
                        "declined_stores_total": 0, "io_errors_total": 0}
            for index, family in enumerate(("gqa", "gdn", "draft")):
                expected[f"rank0_{family}_written_bytes_total"] = physical[index]
                expected[f"rank1_{family}_written_bytes_total"] = 0
                expected[f"{family}_written_bytes_total"] = 0
            observed = {key: deltas[key] for key in expected}
            assert observed == expected, (observed, expected)
        finally:
            barrier.release.set()
        prime.result()
    drain(port)
    return (
        offset,
        baseline,
        {
            "held": midpoint["pending_stores"],
            "drained": pending(port),
            "deltas": observed,
        },
    )


def exercise(
    args: argparse.Namespace, work: Path, devices: list[str], daemon_socket: Path, daemon_pid: int
) -> None:
    path = work / "relay.sock"
    with lmcache.Relay(path, daemon_socket) as relay:
        try:
            roundtrip(args, devices, daemon_socket, relay, str(path))
        finally:
            events = {"pid": daemon_pid, "events": relay.since(), "errors": relay.errors}
            (args.evidence / "relay.json").write_text(
                json.dumps(events, default=bytes.hex, indent=2) + "\n", encoding="utf-8"
            )
        assert not relay.errors
        transfers = [
            e for e in relay.since() if e["opcode"] in (lmcache.RANK_GET, lmcache.RANK_PUT)
        ]
        assert {e["device"] for e in transfers} == set(devices)
        assert {e["daemon_pid"] for e in transfers} == {daemon_pid}


def tail_restores(
    args: argparse.Namespace,
    port: int,
    relay: lmcache.Relay,
    devices: list[str],
    committed: list[int],
) -> dict[str, list[int]]:
    continued = [*committed, 220, 17]
    reference = capture(args, port, "tail-cold", continued, 4, 0, "tail-reference")
    drain(port)
    offset = len(relay.since())
    tail = capture(args, port, "tail-restored", continued, 4, len(committed), "subject")
    assert tail == reference, (tail, reference)
    drain(port)
    totals = {}
    for rank, gpu in enumerate(devices):
        transfers = [e for e in relay.since(offset) if e.get("device") == gpu]
        totals[f"rank{rank}_tail_get"] = receipt(
            transfers, rank, gpu, lmcache.RANK_GET, len(committed)
        )
    return totals


def erase_rank1(daemon_socket: Path, stores: list[list[dict[str, object]]]) -> None:
    keys = [
        next(
            k
            for e in rank
            if e["opcode"] == lmcache.RANK_PUT
            for k in e["keys"]
            if k[0] == lmcache.GQA
        )
        for rank in stores
    ]
    assert keys[0] != keys[1]
    assert (
        lmcache.control(daemon_socket, "sglang", lmcache.ERASE, lmcache.key_body(keys[1]))[0] == 0
    )
    present = (1).to_bytes(4, "big") + lmcache.key_body(keys[0])
    assert lmcache.control(daemon_socket, "sglang", lmcache.LOOKUP, present) == (0, b"\x01")


def roundtrip(
    args: argparse.Namespace,
    devices: list[str],
    daemon_socket: Path,
    relay: lmcache.Relay,
    path: str,
) -> None:
    with http.serve(args, "ranked-cache", "--tp", "2", "--eager", "--mtp-steps", "4",
                    "--lmcache-socket", path) as (port, _):
        status, _, raw = http.exchange(port, "/server_info")
        (args.evidence / "server-info.json").write_bytes(raw)
        info = json.loads(raw)
        assert status == 200 and info["tp_size"] == info["tensor_parallel"] == 2
        assert info["lmcache_available"] and info["lmcache_enabled"] and info["mtp_steps"] == 4
        if not args.store_only:
            cold = capture(args, port, "cold", PROMPT, 8, 0, "reference")
            drain(port)
        offset, pre_prime, held = delayed_store(args, port, relay, devices)
        totals = {"pending_store": held}
        stores = [[e for e in relay.since(offset) if e.get("device") == gpu] for gpu in devices]
        for rank, events in enumerate(stores):
            totals[f"rank{rank}_store"] = receipt(
                events, rank, devices[rank], lmcache.RANK_PUT, len(PREFIX)
            )
        pre_restore = metrics(port)
        check_counters(pre_prime, pre_restore, totals, "store")
        assert pre_restore["stores_total"] - pre_prime["stores_total"] == 1
        token_counter = "gqa_written_tokens_total"
        assert pre_restore[token_counter] - pre_prime[token_counter] == len(PREFIX)
        if args.store_only:
            (args.evidence / "rank-receipts.json").write_text(json.dumps(totals, indent=2) + "\n")
            (args.evidence / "metrics.json").write_text(json.dumps({
                "baseline": pre_prime, "settled": pre_restore,
            }, indent=2) + "\n")
            return
        offset = len(relay.since())
        hit = capture(args, port, "restored", PROMPT, 8, len(PREFIX), "subject")
        drain(port)
        assert hit == cold, (hit, cold)
        events = relay.since(offset)
        assert sum(
            e["opcode"] == lmcache.PREFIX_DETAIL and e["status"] == 0 for e in events
        ) >= len(devices)
        for rank, gpu in enumerate(devices):
            transfers = [e for e in events if e.get("device") == gpu]
            totals[f"rank{rank}_get"] = receipt(transfers, rank, gpu, lmcache.RANK_GET, len(PREFIX))
        post_restore = metrics(port)
        check_counters(pre_restore, post_restore, totals, "get")
        (args.evidence / "metrics.json").write_text(json.dumps({
            "pre_prime": pre_prime, "pre_restore": pre_restore, "post_restore": post_restore,
        }, indent=2) + "\n")
        totals.update(tail_restores(args, port, relay, devices, [*PROMPT, *hit[0][:-1]]))
        erase_rank1(daemon_socket, stores)
        fallback = capture(args, port, "missing-rank", PROMPT, 8, 0, "subject")
        assert fallback == cold, (fallback, cold)
        drain(port)
        (args.evidence / "rank-receipts.json").write_text(json.dumps(totals, indent=2) + "\n")


def interrupted(signum, frame):
    raise SystemExit(128 + signum)


def main():
    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store-only", action="store_true")
    for key in ("server", "checkpoint", "evidence"):
        parser.add_argument(key, type=Path)
    args = parser.parse_args()
    devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    assert len(devices) == 2 and devices[0] != devices[1]
    args.evidence.mkdir(exist_ok=True, parents=True)
    uuids = [gpu.removeprefix("GPU-").replace("-", "").lower() for gpu in devices]
    with TemporaryDirectory(prefix="tp2-serving-", dir="/tmp") as directory:
        work = Path(directory)
        with daemon(args, work) as (endpoint, pid):
            exercise(args, work, uuids, endpoint, pid)


if __name__ == "__main__":
    main()
