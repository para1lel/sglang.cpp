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
    from collections.abc import Generator, Iterable
    from types import FrameType, ModuleType

    from lmcache_route_helpers import Witness

Event = dict[str, object]

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "tests"), str(ROOT / "tests" / "prefix"), str(ROOT / "tests" / "text")]
lmcache = import_module("lmcache_relay")
http = import_module("media.http_routes")

PREFIX = [760, 6511, 314, 9338, 369, 220, 17, 478, 220, 17, 283, 13, 198]
TP2 = 2
PAGE_CAPACITY = 16
CHECKPOINT_CAPACITY = 4
SAMPLED_TOKENS = 8
DRAFT_HOLE_ROWS = 251
DRAFT_HOLE_SALT = "rank-draft-hole"
SeededResult = tuple[list[int], str]


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


def erase_rank1(
    witness: Witness, entries: list[Event], devices: list[str], family: int
) -> list[str]:
    keys = {device: set() for device in devices}
    for entry in entries:
        keys[entry["device"]].update(key for key in entry["keys"] if key[0] == family)
    first, second = (sorted(keys[device]) for device in devices)
    assert first, keys
    assert second, keys
    assert set(first).isdisjoint(second), keys
    witness.lookup(first + second, [1] * (len(first) + len(second)))
    witness.erase(second)
    witness.lookup(first + second, [1] * len(first) + [0] * len(second))
    return [key[1].hex() for key in second]


def read_receipts(
    pool: ModuleType,
    entries: Iterable[Event],
    devices: list[str],
) -> dict[str, list[int]]:
    receipts = {device: [0] * 4 for device in devices}
    for entry in entries:
        for index, size in enumerate(pool.rank_receipt(entry, TP2)):
            receipts[entry["device"]][index] += size
    return receipts


def sampled(prompt: list[int], count: int) -> dict[str, object]:
    return http.native_text(prompt, count)["sampling_params"] | {
        "temperature": 0.8,
        "top_k": 12,
        "sampling_seed": 7103,
    }


def backup_prompt(
    witness: Witness,
    pool: ModuleType,
    label: str,
    prompt: list[int],
    devices: list[str],
) -> list[Event]:
    backed = pool.helpers.backup_under_pressure(witness, label, pool.rows(1023))
    rows = len(prompt)
    extents = {lmcache.GQA: rows, lmcache.DRAFT: rows - 1, lmcache.GDN: rows}
    records = [
        entry
        for entry in backed
        if any(
            record.get("endpoint", record.get("end")) == extents[record["family"]]
            for record in entry["records"]
        )
    ]
    families = pool.record_families(records)
    assert set(families) == {(device, family) for device in devices for family in extents}
    for entry in records:
        pool.rank_receipt(entry, TP2, exact=False)
    keys = {key for entry in records for key in entry["keys"]}
    witness.lookup(sorted(keys), [1] * len(keys))
    return records


def prepare_draft_hole(
    witness: Witness,
    pool: ModuleType,
    devices: list[str],
) -> tuple[SeededResult, list[str]]:
    prompt = pool.rows(DRAFT_HOLE_ROWS)
    oracle = pool.generate(
        witness,
        "cold-draft-hole-oracle",
        prompt,
        0,
        SAMPLED_TOKENS,
        cache_salt="draft-hole-oracle",
        sampling_params=sampled(prompt, SAMPLED_TOKENS),
    )
    pool.generate(witness, "prime-draft-hole", prompt, 0, cache_salt=DRAFT_HOLE_SALT)
    records = backup_prompt(witness, pool, "draft-hole", prompt, devices)
    return oracle, erase_rank1(witness, records, devices, lmcache.DRAFT)


def restore_draft_hole(
    args: argparse.Namespace,
    daemon_socket: Path,
    relay: lmcache.Relay,
    options: tuple[str, ...],
    prepared: tuple[SeededResult, list[str]],
) -> dict[str, object]:
    pool = import_module("pool_routes")
    devices = pool.get_devices(TP2)
    prompt = pool.rows(DRAFT_HOLE_ROWS)
    oracle, erased = prepared
    with http.server_process(args, "restart-with-draft-rank-hole", *options) as (port, _, process):
        witness = pool.helpers.Witness(port, args.evidence, daemon_socket, relay)
        start = len(relay.since())
        fallback, work = witness.changes(
            "missing-draft-rank-cold",
            prompt,
            0,
            {
                "gqa_restored_tokens": len(prompt),
                "gdn_restored_tokens": 0,
                "draft_restored_rows": 0,
                "io_errors": 0,
            },
            count=SAMPLED_TOKENS,
            cache_salt=DRAFT_HOLE_SALT,
            sampling_params=sampled(prompt, SAMPLED_TOKENS),
        )
        assert fallback[:2] == oracle, (fallback, oracle)
        surviving = pool.transfer(relay.since(start), lmcache.RANK_GET)
        actual = read_receipts(pool, surviving, devices)
        expected = {device: [16896 * len(prompt), 0, 0, 0] for device in devices}
        assert actual == expected, actual
        pool.settle(witness, "after-draft-rank-hole")
        pool.terminate(process)
    return {"erased": erased, "rank_read_receipts": actual, "work": work}


def restart_restore(
    args: argparse.Namespace,
    daemon_socket: Path,
    relay_path: Path,
    relay: lmcache.Relay,
    oracle: SeededResult,
) -> dict[str, object]:
    pool = import_module("pool_routes")
    options = pool.server_options(TP2, relay_path, PAGE_CAPACITY, CHECKPOINT_CAPACITY)
    options += ("--eager",)
    prompt = pool.rows(pool.SUBJECT_ROWS)
    devices = pool.get_devices(TP2)
    budget = SAMPLED_TOKENS
    gqa, gdn, draft = 16896 * len(prompt), 76992512, 1056 * (len(prompt) - 1)
    report = {"seeded_output": oracle[0]}
    with http.server_process(args, "restore-after-restart", *options) as (port, _, process):
        witness = pool.helpers.Witness(port, args.evidence, daemon_socket, relay)
        start = len(relay.since())
        hit, work = witness.changes(
            "both-ranks-cpu-restart",
            prompt,
            len(prompt),
            {
                "gqa_restored_tokens": len(prompt),
                "gdn_restored_tokens": len(prompt),
                "draft_restored_rows": len(prompt) - 1,
                "io_errors": 0,
            },
            count=budget,
            cache_salt="subject",
            sampling_params=sampled(prompt, budget),
        )
        assert hit[:2] == oracle, (hit, oracle)
        restored = pool.transfer(relay.since(start), lmcache.RANK_GET)
        actual = read_receipts(pool, restored, devices)
        assert actual == {device: [gqa, gdn, draft, 0] for device in devices}, actual
        report["complete"] = {"rank_read_receipts": actual, "work": work}
        erased_target = erase_rank1(witness, restored, devices, lmcache.GQA)
        pool.terminate(process)
    with http.server_process(args, "restart-with-target-rank-hole", *options) as (port, _, process):
        witness = pool.helpers.Witness(port, args.evidence, daemon_socket, relay)
        start = len(relay.since())
        fallback, work = witness.changes(
            "missing-target-rank-cold",
            prompt,
            0,
            {
                "gqa_restored_tokens": 0,
                "gdn_restored_tokens": 0,
                "draft_restored_rows": len(prompt) - 1,
                "io_errors": 0,
            },
            count=budget,
            cache_salt="subject",
            sampling_params=sampled(prompt, budget),
        )
        assert fallback[:2] == oracle, (fallback, oracle)
        surviving = pool.transfer(relay.since(start), lmcache.RANK_GET)
        actual = read_receipts(pool, surviving, devices)
        assert actual == {device: [0, 0, draft, 0] for device in devices}, actual
        report["target_hole"] = {
            "erased": erased_target,
            "rank_read_receipts": actual,
            "work": work,
        }
        prepared = prepare_draft_hole(witness, pool, devices)
        pool.settle(witness, "after-target-rank-hole")
        pool.terminate(process)
    report["draft_hole"] = restore_draft_hole(args, daemon_socket, relay, options, prepared)
    return report


def exercise(args: argparse.Namespace, work: Path, daemon_socket: Path, daemon_pid: int) -> None:
    pool = import_module("pool_routes")
    relay_path = work / "relay.sock"
    devices = pool.get_devices(TP2)
    options = pool.server_options(TP2, relay_path, PAGE_CAPACITY, CHECKPOINT_CAPACITY)
    options += ("--eager",)
    with lmcache.Relay(relay_path, daemon_socket) as relay:
        try:
            with http.server_process(args, "ranked-cache", *options) as (port, _, process):
                witness = pool.helpers.Witness(port, args.evidence, daemon_socket, relay)
                status, _, raw = http.exchange(port, "/server_info")
                (args.evidence / "server-info.json").write_bytes(raw)
                info = json.loads(raw)
                assert status == HTTPStatus.OK, raw
                assert info["batch_cuda_graphs"] is False, info
                assert info["tp_size"] == info["tensor_parallel"] == TP2, info
                assert info["lmcache_available"], info
                prompt = pool.rows(pool.SUBJECT_ROWS)
                if args.store_only or args.restart_only:
                    pool.generate(
                        witness,
                        "prime-rank-checkpoint",
                        prompt,
                        0,
                        cache_salt="subject",
                        sampling_params=sampled(prompt, 1),
                    )
                    records = backup_prompt(witness, pool, "rank-checkpoint", prompt, devices)
                    endpoints = pool.helpers.transfer_records(
                        records,
                        lmcache.RANK_PUT,
                        lmcache.GDN,
                    )
                    checkpoints = [
                        (device, key)
                        for device, key, record in endpoints
                        if record["endpoint"] == len(prompt)
                    ]
                    assert {device for device, _ in checkpoints} == set(devices), checkpoints
                    report = {
                        "stored_checkpoints": checkpoints,
                        "rank_families": sorted(pool.record_families(records)),
                    }
                else:
                    report = pool.pressure(witness, str, TP2)
                if not args.store_only:
                    oracle = pool.generate(
                        witness,
                        "cold-restart-seeded-oracle",
                        prompt,
                        0,
                        SAMPLED_TOKENS,
                        cache_salt="seeded-restart-oracle",
                        sampling_params=sampled(prompt, SAMPLED_TOKENS),
                    )
                report["settled"] = pool.settle(witness, "ranked-drained")
                pool.terminate(process)
            if not args.store_only:
                report["restart"] = restart_restore(args, daemon_socket, relay_path, relay, oracle)
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
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--store-only", action="store_true")
    scope.add_argument("--restart-only", action="store_true")
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
