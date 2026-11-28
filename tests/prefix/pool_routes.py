"""Exercise independent resident cache families, CPU pressure backup, and drain."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import signal
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from http import HTTPStatus
from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic, sleep
from typing import TYPE_CHECKING

from lmcache_relay import DRAFT, GDN, GQA, RANK_GET, RANK_PUT, Relay
from pool_api_trace import MARKER

if TYPE_CHECKING:
    from collections.abc import Callable
    from subprocess import Popen

    from lmcache_route_helpers import Witness

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]

cache_metrics = import_module("live_controls")
PENDING_STORES = cache_metrics.PENDING_STORES
PREFILL_CHECKPOINTS = cache_metrics.PREFILL_CHECKPOINTS
RUNNING = cache_metrics.RUNNING
WAITING = cache_metrics.WAITING
parse_metrics = cache_metrics.parse_metrics
serving_cache = import_module("decoder.tp2.serving_cache")
helpers = import_module("lmcache_route_helpers")
generated_terminal = import_module("lmcache_split_routes").generated_terminal
http = import_module("media.http_routes")

PREFIX = serving_cache.PREFIX
daemon = serving_cache.daemon
Stream = helpers.Stream
tokens = helpers.tokens
Metrics = dict[str, int]
Event = dict[str, object]
Output = tuple[list[int], str]
UTILIZATION = 0.88
SUBJECT_ROWS = 256
PEER_OUTPUT_ROWS = 8
TP2 = 2
MAX_ACTIVE = 2

FAMILIES = (GQA, GDN, DRAFT)
POOL_COUNTERS = (
    "prefix_gqa_resident_hits_total",
    "prefix_gqa_resident_tokens_total",
    "prefix_gdn_resident_hits_total",
    "prefix_gdn_resident_tokens_total",
    "prefix_backups_started_total",
    "prefix_backups_completed_total",
    "prefix_backups_failed_total",
    "prefix_backed_evictions_total",
    "prefix_unbacked_evictions_total",
    "prefix_gqa_page_capacity",
    "prefix_gdn_slot_capacity",
)
POOL_METRICS = tuple(f"sglang_cpp:{name}" for name in POOL_COUNTERS)
LM_METRICS = tuple(
    f"sglang_cpp:lmcache_{family}_read_bytes_total" for family in ("gqa", "gdn", "draft")
)


def observe(witness: Witness, label: str) -> Metrics:
    raw = witness.get("/metrics")
    (witness.evidence / f"{label}.prom").write_bytes(raw)
    values = parse_metrics(
        raw.decode(),
        (*POOL_METRICS, *LM_METRICS, *PREFILL_CHECKPOINTS, PENDING_STORES, RUNNING, WAITING),
    )
    return {name.removeprefix("sglang_cpp:"): value for name, value in values.items()}


def delta(before: Metrics, after: Metrics, name: str) -> int:
    return after[name] - before[name]


def settle(witness: Witness, label: str) -> Metrics:
    deadline = monotonic() + 20
    while monotonic() < deadline:
        current = observe(witness, label)
        if (
            current["lmcache_pending_stores"] == 0
            and current["prefix_backups_started_total"]
            == current["prefix_backups_completed_total"] + current["prefix_backups_failed_total"]
            and current[RUNNING.removeprefix("sglang_cpp:")] == 0
            and current[WAITING.removeprefix("sglang_cpp:")] == 0
        ):
            return current
        sleep(0.02)
    raise AssertionError(("Prefix worker did not drain", current))


def capacity(args: argparse.Namespace, witness: Witness, pages: int, slots: int) -> list[Metrics]:
    info = json.loads(witness.get("/server_info"))
    (witness.evidence / "server-info.json").write_text(json.dumps(info, indent=2) + "\n")
    assert info["gpu_memory_utilization"] == UTILIZATION, info
    assert info["tp_size"] == args.tp, info
    assert info["lmcache_available"], info
    limits = info["device_memory_pools"]
    assert sorted(value["device"] for value in limits) == list(range(args.tp)), limits
    assert len(args.device_bytes) == args.tp
    for pool in limits:
        physical = args.device_bytes[pool["device"]]
        assert 0 < pool["cache_bytes"] <= pool["capacity_bytes"] <= pool["budget_bytes"], pool
        assert pool["budget_bytes"] <= physical * 88 // 100, (pool, physical)
    before = observe(witness, "ready")
    assert (before["prefix_gqa_page_capacity"], before["prefix_gdn_slot_capacity"]) == (
        pages,
        slots,
    )
    return limits


def rows(count: int) -> list[int]:
    return (PREFIX * ((count + len(PREFIX) - 1) // len(PREFIX)))[:count]


def generate(
    witness: Witness,
    label: str,
    prompt: list[int],
    cached: int,
    count: int = 1,
    **options: object,
) -> Output:
    reply = witness.generate(label, prompt, count=count, **options)
    return tokens(http.native_result(reply, prompt, count, cached))


def transfer(entries: list[Event], opcode: int) -> list[Event]:
    return [
        e for e in entries if e["opcode"] == opcode and e.get("status") == 0 and e.get("forwarded")
    ]


def record_families(entries: list[Event]) -> Counter[tuple[str, int]]:
    return Counter(
        (entry["device"], record["family"]) for entry in entries for record in entry["records"]
    )


def rank_receipt(entry: Event, world: int, *, exact: bool = True) -> list[int]:
    expected = [0] * 4
    for record in entry["records"]:
        family = record["family"]
        if family == GDN:
            size = (76992512 - 20480) * (2 // world) + 20480
        else:
            size = record["tokens"] * (33792 // world if family == GQA else 2112)
        assert record["bytes"] == size, record
        expected[family - 1] += size
        if family == GQA and entry["opcode"] == RANK_PUT:
            expected[3] += record["tokens"]
    got = list(entry["receipt"])
    assert entry["namespace"] == "sglang", entry
    assert entry["allocations"] > 0, entry
    assert got == expected if exact else all(a <= b for a, b in zip(got, expected, strict=True)), (
        got,
        expected,
    )
    return got


def get_devices(world: int) -> list[str]:
    devices = [
        device.removeprefix("GPU-").replace("-", "").lower()
        for device in os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    ]
    assert len(devices) == world, devices
    assert len(set(devices)) == world, devices
    return devices


def pressure_metrics(witness: Witness) -> Metrics:
    text = witness.get("/metrics").decode()
    names = [
        f"sglang_cpp:lmcache_{name}"
        for name in (
            "stores_total",
            "pending_stores",
            "gdn_written_bytes_total",
            "rank0_stores_total",
            "rank1_stores_total",
            "rank0_gdn_written_bytes_total",
            "rank1_gdn_written_bytes_total",
        )
    ]
    return {
        name.removeprefix("sglang_cpp:lmcache_"): value
        for name, value in parse_metrics(text, names).items()
    }


def cold_and_hot(
    witness: Witness, mark: Callable[[str], None], prompt: list[int]
) -> tuple[Output, Metrics]:
    clean = settle(witness, "before-cold")
    offset = len(witness.relay.since())
    mark("cold")
    cold = generate(witness, "cold", prompt, 0, cache_salt="subject")
    mark("decode")
    decode_prompt = rows(127)
    decode = generate(
        witness, "decode", decode_prompt, 0, count=PEER_OUTPUT_ROWS, cache_salt="decode"
    )
    assert len(decode[0]) == PEER_OUTPUT_ROWS
    hot = generate(witness, "hot", prompt, len(prompt), cache_salt="subject")
    assert hot == cold, (hot, cold)
    before = settle(witness, "before-pressure")
    assert delta(clean, before, "prefix_gqa_resident_tokens_total") >= len(prompt), before
    assert delta(clean, before, "prefix_gdn_resident_tokens_total") >= len(prompt), before
    assert not transfer(witness.relay.since(offset), RANK_PUT), (
        "Published CPU backup without pressure"
    )
    assert not transfer(witness.relay.since(offset), RANK_GET), "Resident repeat read from CPU"
    assert delta(clean, before, "prefix_backups_started_total") == 0, before
    return cold, before


def assert_partial_rank_ack(
    witness: Witness,
    start: int,
    root_device: str,
    endpoint: int,
    old: Event,
) -> None:
    deadline = monotonic() + 5
    while monotonic() < deadline:
        completed = transfer(witness.relay.since(start), RANK_PUT)
        first_rank = [
            event
            for event in completed
            if event["device"] == root_device
            and any(record.get("endpoint") == endpoint for record in event["records"])
        ]
        midpoint = pressure_metrics(witness)
        if first_rank and midpoint["rank0_gdn_written_bytes_total"] > 0:
            break
        sleep(0.02)
    else:
        raise AssertionError("Rank 0 did not finish while rank 1's receipt was held")
    assert midpoint["pending_stores"] == 1, midpoint
    assert not old.get("forwarded"), old
    assert sum(rank_receipt(event, TP2)[1] for event in first_rank) > 0
    assert midpoint["rank0_stores_total"] == midpoint["rank1_stores_total"] + 1
    assert (
        midpoint["rank0_gdn_written_bytes_total"] - midpoint["rank1_gdn_written_bytes_total"]
        == old["receipt"][1]
    )


def held_pending(witness: Witness) -> Metrics:
    deadline = monotonic() + 5
    while monotonic() < deadline:
        held = observe(witness, "backup-reply-held")
        outstanding = held["prefix_backups_started_total"] - (
            held["prefix_backups_completed_total"] + held["prefix_backups_failed_total"]
        )
        if outstanding > 0 and held["lmcache_pending_stores"] == 1:
            return held
        sleep(0.02)
    raise AssertionError(("Held backup was not reported pending", held))


def hold_backup(
    witness: Witness,
    prompt: list[int],
    cold: Output,
    before: Metrics,
    devices: list[str],
) -> tuple[Metrics, Event]:
    world = len(devices)
    start = len(witness.relay.since())
    gate = witness.relay.arm(RANK_PUT, devices[-1], endpoint=len(prompt))

    def admissions() -> None:
        for index in range(12):
            generate(witness, f"pressure-{index}", rows(1), 0, cache_salt=f"pressure-{index}")
            if gate.entered.wait(0.05):
                break

    with ThreadPoolExecutor(max_workers=1) as executor:
        admission = executor.submit(admissions)
        try:
            old = gate.wait()
            assert old["status"] == 0, old
            assert not old.get("forwarded"), old
            assert rank_receipt(old, world)[1] > 0, old
            if world == TP2:
                assert_partial_rank_ack(witness, start, devices[0], len(prompt), old)
            held = held_pending(witness)
            assert delta(before, held, "prefix_backups_started_total") > 0, held
            follower = Stream(
                witness, "pressure-peer", prompt, PEER_OUTPUT_ROWS, cache_salt="subject"
            )
            progress_ns, progress_rows = follower.progress()
            assert progress_ns > old["reply_ns"], old
            assert progress_rows > 0
            assert not gate.release.is_set(), old
        finally:
            gate.release.set()
        admission.result()
    assert gate.finished.wait(5), old
    peer = follower.finish()
    assert http.native_result(peer, prompt, PEER_OUTPUT_ROWS, len(prompt))[0][:1] == cold[0], peer
    return held, old


def mixed_restore(
    witness: Witness,
    mark: Callable[[str], None],
    prompt: list[int],
    cold: Output,
    devices: list[str],
) -> None:
    world = len(devices)
    for index in range(4):
        generate(
            witness, f"checkpoint-reuse-{index}", rows(1), 0, cache_salt=f"checkpoint-reuse-{index}"
        )
    settle(witness, "before-mixed-restore")
    mark("mixed-restore")
    mixed_offset = len(witness.relay.since())
    mixed = generate(witness, "cpu-gdn-resident-kv", prompt, len(prompt), cache_salt="subject")
    assert mixed == cold, (mixed, cold)
    mixed_reads = transfer(witness.relay.since(mixed_offset), RANK_GET)
    assert set(record_families(mixed_reads)) == {(device, GDN) for device in devices}, mixed_reads
    assert all(rank_receipt(event, world)[1] > 0 for event in mixed_reads)


def pressure(witness: Witness, mark: Callable[[str], None], world: int) -> dict[str, object]:
    prompt = rows(SUBJECT_ROWS)
    cold, before = cold_and_hot(witness, mark, prompt)
    mark("pressure")
    start = len(witness.relay.since())
    devices = get_devices(world)
    held, old = hold_backup(witness, prompt, cold, before, devices)
    mixed_restore(witness, mark, prompt, cold, devices)
    mark("page-pressure")
    for index in range(6):
        generate(witness, f"reuse-{index}", prompt, 0, cache_salt=f"reuse-{index}")
    generate(witness, "all-pages-active", rows(2047), 0, cache_salt="all-pages-active")
    after = settle(witness, "after-pressure")
    assert delta(before, after, "prefix_backups_completed_total") > 0, after
    assert delta(before, after, "prefix_backups_failed_total") == 0, after
    assert delta(before, after, "prefix_backed_evictions_total") > 0, after
    assert delta(before, after, "prefix_unbacked_evictions_total") == 0, after
    written = transfer(witness.relay.since(start), RANK_PUT)
    families = record_families(written)
    assert set(families) == (
        {(device, family) for device in devices for family in (GQA, GDN)} | {(devices[0], DRAFT)}
    ), families
    for event in written:
        rank_receipt(event, world, exact=False)
    replacements = [
        event
        for event in written
        if any(record.get("endpoint") == len(prompt) for record in event["records"])
        and set(event["keys"]).isdisjoint(old["keys"])
        and event["device"] == old["device"]
    ]
    assert replacements
    assert any(rank_receipt(event, world, exact=False)[1] for event in replacements)
    mark("reload")
    start = len(witness.relay.since())
    restored = generate(witness, "cpu-reload", prompt, len(prompt), cache_salt="subject")
    assert restored == cold, (restored, cold)
    reloaded = settle(witness, "after-reload")
    restores = transfer(witness.relay.since(start), RANK_GET)
    inventory = record_families(restores)
    assert set(inventory) == set(families), inventory
    read_bytes = Counter()
    for event in restores:
        receipt = rank_receipt(event, world)
        for family, size in zip(FAMILIES, receipt[: len(FAMILIES)], strict=True):
            read_bytes[family] += size
    for family, counter in zip(FAMILIES, LM_METRICS, strict=True):
        assert delta(after, reloaded, counter.removeprefix("sglang_cpp:")) == read_bytes[family] > 0
    return {
        "cold": cold[0],
        "before": before,
        "held": held,
        "after": after,
        "reloaded": reloaded,
        "held_keys": old["keys"],
        "cpu_read_bytes": dict(read_bytes),
    }


def cancel_and_reset(witness: Witness, mark: Callable[[str], None]) -> dict[str, object]:
    mark("capacity")
    body = http.native_text(rows(2049), 1) | {"cache_salt": "too-large"}
    status, _, raw = http.exchange(witness.port, http.GENERATE, body)
    assert status == HTTPStatus.REQUEST_ENTITY_TOO_LARGE, (status, raw)
    assert b"capacity" in raw, (status, raw)
    settle(witness, "after-permanent-rejection")
    mark("cancel")
    prompt = rows(127)
    stream = Stream(witness, "reused-rid", prompt, 1536, cache_salt="cancel-subject")
    stream.progress()
    waiting = Stream(witness, "waiting-rid", rows(384), 384, cache_salt="waiting")
    deadline = monotonic() + 5
    while monotonic() < deadline:
        admission = observe(witness, "retryable-admission")
        if admission[RUNNING] == MAX_ACTIVE:
            break
        sleep(0.02)
    assert admission[RUNNING] == MAX_ACTIVE, admission
    assert waiting.events.empty(), admission
    assert stream.progress()[1] > 0, "Active request did not advance while pages were pinned"
    status, _, raw = http.exchange(witness.port, "/reset_prefix_cache", {})
    assert status == HTTPStatus.CONFLICT, (status, raw)
    status, _, raw = http.exchange(witness.port, "/abort_request", {"rid": waiting.label})
    assert status == HTTPStatus.OK, (status, raw)
    assert not raw, (status, raw)
    queued = waiting.finish()
    assert queued["meta_info"]["finish_reason"]["type"] == "abort", queued
    assert not queued["output_ids"], queued
    status, _, raw = http.exchange(witness.port, "/abort_request", {"rid": stream.label})
    assert status == HTTPStatus.OK, (status, raw)
    assert not raw, (status, raw)
    cancelled = stream.finish()
    assert cancelled["meta_info"]["finish_reason"]["type"] == "abort", cancelled
    settle(witness, "after-cancel")
    restarted = generate(witness, "retry-after-active-cancel", rows(384), 0, cache_salt="waiting")
    assert len(restarted[0]) == 1
    mark("reset")
    witness.reset()
    reference = generate(witness, "reset-cold", prompt, 0, cache_salt="cancel-subject")
    replacement = Stream(
        witness,
        stream.label,
        prompt,
        1,
        cache_salt="cancel-subject",
        evidence_label="after-reset-reused-rid",
    )
    assert tokens(http.native_result(replacement.finish(), prompt, 1, 127)) == reference
    return {
        "cancelled_output_ids": cancelled["output_ids"],
        "queued_output_ids": queued["output_ids"],
        "capacity_recovery": restarted[0],
    }


def anchors(witness: Witness) -> dict[str, object]:
    prompt = rows(3137)
    before = settle(witness, "before-anchor")
    expected = generate(witness, "anchor-cold", prompt, 0, cache_salt="anchor")
    after = settle(witness, "after-anchor")
    deltas = {
        name: delta(before, after, name)
        for name in (
            "lmcache_prefill_checkpoints_planned_total",
            "lmcache_prefill_checkpoints_published_total",
            "lmcache_prefill_checkpoints_declined_total",
            "prefix_backups_started_total",
        )
    }
    assert list(deltas.values()) == [3, 3, 0, 0], deltas
    assert generate(witness, "anchor-exact", prompt, len(prompt), cache_salt="anchor") == expected
    endpoints = [len(prompt) - 2048, len(prompt) - 1024, len(prompt)]
    for end in endpoints[:-1]:
        generate(witness, f"anchor-{end}", prompt[:end], end, cache_salt="anchor")
    start = len(witness.relay.since())
    before = observe(witness, "before-independent")
    split = prompt[:-512]
    actual = generate(witness, "gqa-deeper", split, len(prompt) - 1024, cache_salt="anchor")
    after = settle(witness, "after-independent")
    gqa = delta(before, after, "prefix_gqa_resident_tokens_total")
    gdn = delta(before, after, "prefix_gdn_resident_tokens_total")
    assert len(split) >= gqa > gdn == len(prompt) - 1024, (gqa, gdn)
    assert not transfer(witness.relay.since(start), RANK_GET), "Independent resident hits read CPU"
    repeat = generate(witness, "independent-repeat", split, len(split), cache_salt="anchor")
    assert repeat == actual, (repeat, actual)
    return {
        "planned": deltas,
        "exact_endpoints": endpoints,
        "matched_gqa": gqa,
        "matched_gdn": gdn,
        "output_ids": actual[0],
    }


def terminate(process: Popen[bytes]) -> None:
    process.send_signal(signal.SIGTERM)
    assert process.wait(timeout=20) == 0, process.returncode


def server_options(world: int, relay_path: Path, pages: int, slots: int) -> tuple[str, ...]:
    return (
        "--tp",
        str(world),
        "--max-active",
        str(MAX_ACTIVE),
        "--mtp-steps",
        "4",
        "--max-num-batched-tokens",
        "4096",
        "--prefix-gqa-pages",
        str(pages),
        "--prefix-gdn-slots",
        str(slots),
        "--lmcache-socket",
        str(relay_path),
    )


def exercise(
    args: argparse.Namespace,
    daemon_path: Path,
    daemon_pid: int,
    relay_path: Path,
) -> None:
    pages, slots = (64, 8) if args.anchors_only else (16, 4)
    phases = []
    marker = ctypes.CDLL(str(args.nvtx)).prefix_pool_mark if args.nvtx else None
    if marker:
        marker.argtypes = [ctypes.c_char_p]
        marker.restype = None

    def mark(label: str) -> None:
        phases.append(label)
        if marker:
            marker((MARKER + label).encode())

    with Relay(relay_path, daemon_path) as relay:
        receipt = {"tp": args.tp, "daemon_pid": daemon_pid, "phases": phases}
        try:
            with http.server_process(
                args, "pool", *server_options(args.tp, relay_path, pages, slots)
            ) as (port, _, process):
                witness = helpers.Witness(port, args.evidence, daemon_path, relay)
                mark("ready")
                receipt["server_pid"] = process.pid
                try:
                    receipt["capacity"] = capacity(args, witness, pages, slots)
                    if args.anchors_only:
                        receipt["anchors"] = anchors(witness)
                    else:
                        receipt["pressure"] = pressure(witness, mark, args.tp)
                        receipt["cancel_reset"] = cancel_and_reset(witness, mark)
                        mark("terminal-reuse")
                        receipt["terminal"] = generated_terminal(witness, rows(257), args.tp)
                    receipt["drained"] = settle(witness, "drained")
                finally:
                    mark("shutdown")
                    terminate(process)
                    receipt["shutdown_status"] = process.returncode
        finally:
            receipt["relay_errors"] = relay.errors
            (args.evidence / "run.json").write_text(
                json.dumps(receipt, default=bytes.hex, indent=2) + "\n"
            )
            (args.evidence / "relay.json").write_text(
                json.dumps(relay.since(), default=bytes.hex, indent=2) + "\n"
            )
        assert not relay.errors, relay.errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("server", "checkpoint", "evidence"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--tp", type=int, choices=(1, 2), required=True)
    parser.add_argument("--device-bytes", nargs="+", type=int, required=True)
    parser.add_argument("--nvtx", type=Path)
    parser.add_argument("--anchors-only", action="store_true")
    args = parser.parse_args()
    assert len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")) == args.tp
    assert args.nvtx is None or not args.anchors_only
    args.evidence.mkdir(parents=True)
    with TemporaryDirectory(prefix="sglang-pool-test.", dir="/tmp") as directory:
        work = Path(directory)
        with daemon(args, work, memory_gib=4) as (daemon_path, daemon_pid):
            exercise(args, daemon_path, daemon_pid, work / "relay.sock")


if __name__ == "__main__":
    main()
