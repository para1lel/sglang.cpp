"""Verify exact GDN anchors when a CPU backup pins a slot."""

from __future__ import annotations

import argparse
import json
import re
from http import HTTPStatus
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic, perf_counter_ns, sleep
from typing import TYPE_CHECKING

from lmcache_relay import RANK_PUT, Relay
from pool_routes import (
    Stream,
    daemon,
    delta,
    generate,
    get_devices,
    helpers,
    http,
    observe,
    rows,
    server_options,
    settle,
    terminate,
    tokens,
)

if TYPE_CHECKING:
    from lmcache_route_helpers import Witness
    from pool_routes import Output

ENDPOINTS = (1089, 2113, 3137)
WARM = 127
PREFILL = re.compile(rb"^batch prefill_rows=([0-9,]+) workspace_bytes=", re.MULTILINE)
PUBLISHED = "lmcache_prefill_checkpoints_published_total"
PLANNED = "lmcache_prefill_checkpoints_planned_total"
DECLINED = "lmcache_prefill_checkpoints_declined_total"
FAILED = "prefix_backups_failed_total"


def wait_for_rows(log: Path, offset: int, expected: int) -> None:
    deadline = monotonic() + 10
    while monotonic() < deadline:
        with log.open("rb") as stream:
            stream.seek(offset)
            groups = PREFILL.findall(stream.read())
        done = sum(int(count) for group in groups for count in group.split(b","))
        if done >= expected:
            assert done == expected, (done, expected)
            return
        sleep(0.02)
    raise AssertionError(("Prefill did not stop at the blocked anchor", expected, done))


def cancel_at_anchor(
    pending: Stream, log: Path, offset: int, warmed: dict[str, int]
) -> tuple[Stream, int, dict[str, int]]:
    witness = pending.witness
    blocked = ENDPOINTS[0] - 1
    wait_for_rows(log, offset, blocked)
    midpoint = observe(witness, "held-at-anchor")
    assert midpoint["lmcache_pending_stores"] == 1, midpoint
    assert pending.events.empty(), "Blocked request completed with its anchor unavailable"
    assert delta(warmed, midpoint, PLANNED) == len(ENDPOINTS), midpoint
    assert delta(warmed, midpoint, PUBLISHED) == 0, midpoint
    assert delta(warmed, midpoint, DECLINED) == 0, midpoint
    status, _, body = http.exchange(witness.port, "/abort_request", {"rid": pending.label})
    assert status == HTTPStatus.OK, (status, body)
    cancelled = pending.finish()
    assert cancelled["meta_info"]["finish_reason"]["type"] == "abort", cancelled
    assert not cancelled["output_ids"], cancelled
    offset = log.stat().st_size
    retry = Stream(witness, "exact-after-cancel", rows(ENDPOINTS[-1]), 1, cache_salt="retry")
    wait_for_rows(log, offset, blocked)
    assert retry.events.empty(), "Retry completed before the backup was released"
    return retry, offset, midpoint


def run_case(
    witness: Witness, log: Path, slots: int, world: int, expected: Output
) -> dict[str, object]:
    prompt = rows(ENDPOINTS[-1])
    drop = slots == 1
    witness.reset()
    before = settle(witness, "before-warm")
    gate = witness.relay.arm(RANK_PUT, get_devices(world)[-1], endpoint=WARM, drop=drop)
    try:
        warm = generate(witness, "held-warm", rows(WARM), 0, cache_salt="held")
        if slots > 1:
            generate(witness, "other-warm", rows(WARM + 4), 0, cache_salt="other")
        held = gate.wait()
        assert held["status"] == 0, held
        assert not held.get("forwarded"), held
        warmed = observe(witness, "after-warm")
        assert delta(before, warmed, PUBLISHED) == slots, warmed
        offset = log.stat().st_size
        pending = Stream(witness, "exact-under-pressure", prompt, 1, cache_salt="long")
        blocked = ENDPOINTS[slots - 1] - 1
        if drop:
            pending, offset, midpoint = cancel_at_anchor(pending, log, offset, warmed)
            result = None
        else:
            hit = generate(witness, "held-slot-hit", rows(WARM), WARM, cache_salt="held")
            assert hit == warm, (hit, warm)
            result = pending.finish()
            midpoint = observe(witness, "finished-with-backup-held")
            assert midpoint["lmcache_pending_stores"] == 1, midpoint
            assert delta(warmed, midpoint, PUBLISHED) == len(ENDPOINTS), midpoint
            assert not gate.finished.is_set(), held
    finally:
        released = perf_counter_ns()
        gate.release.set()
    assert gate.finished.wait(5), held
    if result is None:
        result = pending.finish(terminal_after=released)
    actual = tokens(http.native_result(result, prompt, 1, 0))
    with log.open("rb") as stream:
        stream.seek(offset)
        chunks = [sum(map(int, group.split(b","))) for group in PREFILL.findall(stream.read())]
    prefill = [chunk for chunk in chunks if chunk]
    spacing = ENDPOINTS[1] - ENDPOINTS[0]
    assert prefill == [blocked, *([spacing] * (len(ENDPOINTS) - slots)), 1], chunks
    assert actual == expected, (actual, expected)
    after = settle(witness, "after-pressure")
    assert delta(warmed, after, PUBLISHED) == len(ENDPOINTS), after
    assert delta(warmed, after, DECLINED) == 0, after
    assert delta(warmed, after, FAILED) == int(drop), after
    return {
        "before": warmed,
        "held": midpoint,
        "after": after,
        "rows_at_wait": blocked,
        "prefill_chunks": prefill,
        "generated": actual[0],
        "failed_reply": drop,
        "cancelled_during_hold": drop,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("server", "checkpoint", "evidence"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--tp", type=int, choices=(1, 2), required=True)
    args = parser.parse_args()
    evidence = args.evidence
    evidence.mkdir(parents=True)
    with TemporaryDirectory(prefix="sglang-anchors.", dir="/tmp") as directory:
        relay_path = Path(directory) / "relay.sock"
        with (
            daemon(args, Path(directory), memory_gib=4) as (socket, _),
            Relay(relay_path, socket) as relay,
        ):
            options = server_options(args.tp, relay_path, 64, 2)
            off = (*options, "--prefix-cache-entries", "0")
            with http.server_process(args, "cache-off", *off) as (port, _, process):
                witness = helpers.Witness(port, evidence, socket, relay)
                expected = generate(witness, "cache-off-exact", rows(ENDPOINTS[-1]), 0)
                terminate(process)
            report = {}
            for slots in (1, 2):
                args.evidence = evidence / f"slots-{slots}"
                args.evidence.mkdir()
                options = server_options(args.tp, relay_path, 64, slots)
                with http.server_process(args, "pressure", *options) as (port, log, process):
                    witness = helpers.Witness(port, args.evidence, socket, relay)
                    report[slots] = run_case(witness, log, slots, args.tp, expected)
                    terminate(process)
            assert not relay.errors, relay.errors
            (evidence / "run.json").write_text(json.dumps(report, indent=2) + "\n")
            (evidence / "relay.json").write_text(
                json.dumps(relay.since(), default=bytes.hex) + "\n"
            )


if __name__ == "__main__":
    main()
