"""Check both rank transfers and generated IDs through a single physical LMCache daemon."""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
from contextlib import closing, contextmanager
from http import HTTPStatus
from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic, sleep
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Generator
    from types import FrameType

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "tests"), str(ROOT / "tests" / "prefix"), str(ROOT / "tests" / "text")]
lmcache = import_module("lmcache_relay")
http = import_module("media.http_routes")

PREFIX = [760, 6511, 314, 9338, 369, 220, 17, 478, 220, 17, 283, 13, 198]
TP2 = 2
PAGE_CAPACITY = 16
CHECKPOINT_CAPACITY = 4


def free_port() -> str:
    with closing(socket.socket()) as sock:
        sock.bind(("127.0.0.1", 0))
        return str(sock.getsockname()[1])


@contextmanager
def daemon(
    args: argparse.Namespace, work: Path, *, memory_gib: int = 2
) -> Generator[tuple[Path, int], None, None]:
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
        str(memory_gib),
    ]
    with log_path.open("wb") as out:
        proc = subprocess.Popen(
            command, env=env, stdout=out, stderr=subprocess.STDOUT, start_new_session=True
        )
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


def erase_rank1(daemon_socket: Path, entries: list[dict[str, object]], devices: list[str]) -> None:
    rank_keys = {
        entry["device"]: key for entry in entries for key in entry["keys"] if key[0] == lmcache.GQA
    }
    assert set(rank_keys) == set(devices), rank_keys
    first, second = (rank_keys[device] for device in devices)
    assert first != second
    assert lmcache.control(daemon_socket, "sglang", lmcache.ERASE, lmcache.key_body(second))[0] == 0
    payload = (2).to_bytes(4, "big") + lmcache.key_body(first) + lmcache.key_body(second)
    assert lmcache.control(daemon_socket, "sglang", lmcache.LOOKUP, payload) == (0, b"\x01\x00")


def restart_restore(
    args: argparse.Namespace, daemon_socket: Path, relay_path: Path, relay: lmcache.Relay
) -> dict[str, object]:
    pool = import_module("pool_routes")
    options = pool.server_options(TP2, relay_path, PAGE_CAPACITY, CHECKPOINT_CAPACITY)
    prompt = pool.rows(pool.SUBJECT_ROWS)
    devices = pool.get_devices(TP2)
    report = {}
    with http.server_process(args, "restore-after-restart", *options) as (port, _, process):
        witness = pool.helpers.Witness(port, args.evidence, daemon_socket, relay)
        start = len(relay.since())
        hit = pool.generate(
            witness, "both-ranks-cpu-restart", prompt, len(prompt), cache_salt="subject"
        )
        restored = pool.transfer(relay.since(start), lmcache.RANK_GET)
        per_device = {device: [] for device in devices}
        for event in restored:
            per_device[event["device"]].append(pool.rank_receipt(event, TP2))
        actual = {
            device: [sum(values) for values in zip(*receipts, strict=True)]
            for device, receipts in per_device.items()
        }
        expected = [16896 * len(prompt), 76992512, 0, 0]
        draft = 2112 * (len(prompt) - 1)
        assert actual == {devices[0]: [*expected[:2], draft, 0], devices[1]: expected}, actual
        pool.settle(witness, "before-restart-exit")
        pool.terminate(process)
    erase_rank1(daemon_socket, restored, devices)
    with http.server_process(args, "restart-with-rank-hole", *options) as (port, _, process):
        witness = pool.helpers.Witness(port, args.evidence, daemon_socket, relay)
        start = len(relay.since())
        fallback = pool.generate(witness, "missing-rank-cold", prompt, 0, cache_salt="subject")
        assert fallback == hit, (hit, fallback)
        surviving = pool.transfer(relay.since(start), lmcache.RANK_GET)
        assert set(pool.record_families(surviving)) == {(devices[0], lmcache.DRAFT)}, surviving
        assert all(pool.rank_receipt(event, TP2)[2] > 0 for event in surviving)
        report["terminal"] = pool.generated_terminal(witness, pool.rows(257), TP2)
        pool.settle(witness, "after-rank-hole-terminal")
        pool.terminate(process)
    return report | {"rank_read_receipts": actual, "fallback_output": fallback[0]}


def exercise(args: argparse.Namespace, work: Path, daemon_socket: Path, daemon_pid: int) -> None:
    pool = import_module("pool_routes")
    relay_path = work / "relay.sock"
    devices = pool.get_devices(TP2)
    options = pool.server_options(TP2, relay_path, PAGE_CAPACITY, CHECKPOINT_CAPACITY)
    with lmcache.Relay(relay_path, daemon_socket) as relay:
        try:
            with http.server_process(args, "ranked-cache", *options) as (port, _, process):
                witness = pool.helpers.Witness(port, args.evidence, daemon_socket, relay)
                status, _, raw = http.exchange(port, "/server_info")
                (args.evidence / "server-info.json").write_bytes(raw)
                info = json.loads(raw)
                assert status == HTTPStatus.OK, raw
                assert info["tp_size"] == info["tensor_parallel"] == TP2, info
                assert info["lmcache_available"], info
                if args.store_only:
                    prompt = pool.rows(pool.SUBJECT_ROWS)
                    cold, before = pool.cold_and_hot(witness, str, prompt)
                    held, old = pool.hold_backup(witness, prompt, cold, before, devices)
                    report = {"held": held, "held_keys": old["keys"]}
                else:
                    report = pool.pressure(witness, str, TP2)
                report["settled"] = pool.settle(witness, "ranked-drained")
                pool.terminate(process)
            if not args.store_only:
                restarted = restart_restore(args, daemon_socket, relay_path, relay)
                assert restarted["fallback_output"] == report["cold"], restarted
                report["restart"] = restarted
            (args.evidence / "rank-receipts.json").write_text(
                json.dumps(report, default=bytes.hex, indent=2) + "\n", encoding="utf-8"
            )
        finally:
            events = {"pid": daemon_pid, "events": relay.since(), "errors": relay.errors}
            (args.evidence / "relay.json").write_text(
                json.dumps(events, default=bytes.hex, indent=2) + "\n", encoding="utf-8"
            )
        assert not relay.errors, relay.errors
        transfers = [entry for entry in relay.since() if entry["opcode"] in lmcache.TRANSFERS]
        assert {entry["device"] for entry in transfers} == set(devices), transfers
        assert {entry["daemon_pid"] for entry in transfers} == {daemon_pid}, transfers


def interrupted(signum: int, _frame: FrameType | None) -> None:
    raise SystemExit(128 + signum)


def main() -> None:
    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store-only", action="store_true")
    for key in ("server", "checkpoint", "evidence"):
        parser.add_argument(key, type=Path)
    args = parser.parse_args()
    devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    assert len(devices) == TP2
    assert len(set(devices)) == TP2
    args.evidence.mkdir(exist_ok=True, parents=True)
    with TemporaryDirectory(prefix="tp2-serving-", dir="/tmp") as directory:
        work = Path(directory)
        with daemon(args, work, memory_gib=4) as (endpoint, pid):
            exercise(args, work, endpoint, pid)


if __name__ == "__main__":
    main()
