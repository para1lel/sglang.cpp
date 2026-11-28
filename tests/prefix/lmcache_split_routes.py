"""Resident reuse and independently backed cache families through native serving."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, NamedTuple

from live_controls import parse_metrics
from lmcache_relay import DRAFT, GDN, GQA, PREFIX_DETAIL, RANK_GET, RANK_PUT
from lmcache_route_helpers import Stream, Witness, backup_under_pressure, tokens, transfer_records

from media import http_routes as http

if TYPE_CHECKING:
    from argparse import Namespace

    from lmcache_relay import Relay

Evidence = dict[str, object]
Output = tuple[list[int], str]
KV_ROW_BYTES = 33792
DRAFT_ROW_BYTES = 2112
GDN_STATE_BYTES_PER_HALF = 76972032
GDN_SCORE_ROW_BYTES = 20480
RANK_PAIR = 2
MTP_STEPS = 4
SHALLOW = 256
MIDDLE = 512
DEEP = 768
PROMPT = 896


class Coverage(NamedTuple):
    gqa: int = 0
    gdn: int = 0
    draft: int = 0


def cpu_read_receipts(witness: Witness, offset: int, tp: int) -> list[int]:
    restored = [
        entry
        for entry in witness.relay.since(offset)
        if entry["opcode"] == RANK_GET and entry["status"] == 0 and entry["forwarded"]
    ]
    if restored:
        assert len({entry["device"] for entry in restored}) == tp, restored
    for entry in restored:
        assert list(entry["receipt"][:3]) == [
            sum(record["bytes"] for record in entry["records"] if record["family"] == family)
            for family in (GQA, GDN, DRAFT)
        ], entry
    return [sum(entry["receipt"][i] for entry in restored) for i in range(3)]


def check_cpu_restore(
    witness: Witness,
    label: str,
    prompt: list[int],
    tp: int,
    coverage: Coverage,
    **options: object,
) -> tuple[Output, Evidence]:
    gqa, gdn, draft = coverage
    offset = len(witness.relay.since())
    result, work = witness.changes(
        label,
        prompt,
        gdn,
        {
            "gqa_restored_tokens": gqa,
            "gqa_read_only_tokens": gqa - gdn,
            "gqa_computed_tokens": len(prompt) - gqa,
            "gdn_restored_tokens": gdn,
            "gdn_computed_tokens": len(prompt) - gdn,
            "draft_restored_rows": draft,
            "io_errors": 0,
        },
        **options,
    )
    read = cpu_read_receipts(witness, offset, tp)
    gdn_bytes = 2 * GDN_STATE_BYTES_PER_HALF + tp * GDN_SCORE_ROW_BYTES
    assert read == [gqa * KV_ROW_BYTES, gdn_bytes if gdn else 0, draft * DRAFT_ROW_BYTES], (
        label,
        read,
    )
    ranks = {
        entry["coverage_query"]["rank_key"]: entry["coverage"]
        for entry in witness.relay.since(offset)
        if entry["opcode"] == PREFIX_DETAIL
        and entry["status"] == 0
        and entry["coverage_query"]["count"] == len(prompt)
    }
    assert len(ranks) == tp, (label, ranks)
    for rank in ranks.values():
        ends = [end for end, _ in rank["checkpoints"]]
        assert ends == sorted(set(ends), reverse=True), rank
    return tokens(result), {
        "work": work,
        "read_bytes": read,
        "rank_inventory": list(ranks.values()),
    }


def repeat_resident(
    witness: Witness, label: str, prompt: list[int], expected: Output, **options: object
) -> None:
    offset = len(witness.relay.since())
    result = witness.generate(label, prompt, **options)
    assert tokens(http.native_result(result, prompt, 1, len(prompt))) == expected
    assert not any(entry["opcode"] in (RANK_GET, RANK_PUT) for entry in witness.relay.since(offset))


def prime_for_pressure(
    witness: Witness, label: str, prompt: list[int], unrelated: list[int]
) -> list[Evidence]:
    offset = len(witness.relay.since())
    previous = 0
    for end in (SHALLOW, MIDDLE, DEEP):
        result = witness.generate(f"{label}-prime-{end}", prompt[:end])
        http.native_result(result, prompt[:end], 1, previous)
        previous = end
    assert not any(entry["opcode"] in (RANK_GET, RANK_PUT) for entry in witness.relay.since(offset))
    return backup_under_pressure(witness, label, unrelated)


def generated_terminal(witness: Witness, source: list[int], tp: int) -> Evidence:
    prompt = source[:257]
    prefix_generation, continuation = 8, 8
    witness.reset()
    oracle = witness.generate(
        "generated-terminal-uninterrupted", prompt, count=prefix_generation + continuation - 1
    )
    expected, _, _ = http.native_result(oracle, prompt, prefix_generation + continuation - 1, 0)
    witness.reset()
    offset = len(witness.relay.since())
    prime = witness.generate("generated-terminal-writer", prompt, count=prefix_generation)
    generated, _, _ = http.native_result(prime, prompt, prefix_generation, 0)
    assert generated == expected[:prefix_generation], (generated, expected)
    committed = [*prompt, *generated[:-1]]
    endpoint = len(committed)
    assert endpoint > len(prompt)
    resident = (
        "sglang_cpp:prefix_gqa_resident_tokens_total",
        "sglang_cpp:prefix_gdn_resident_tokens_total",
    )
    before = parse_metrics(witness.get("/metrics").decode(), resident)
    reply = witness.generate("generated-terminal-restore", committed, count=continuation)
    restored = http.native_result(reply, committed, continuation, endpoint)
    after = parse_metrics(witness.get("/metrics").decode(), resident)
    observed = {name: after[name] - before[name] for name in resident}
    assert tuple(observed.values()) == (endpoint, endpoint), observed
    assert not [
        entry for entry in witness.relay.since(offset) if entry["opcode"] in (RANK_GET, RANK_PUT)
    ], witness.relay.since(offset)
    assert restored[0] == expected[prefix_generation - 1 :], (restored, expected)
    return {
        "prompt_rows": len(prompt),
        "terminal_rows": endpoint,
        "continuation_ids": restored[0],
        "resident_tokens": observed,
        "tensor_parallel": tp,
    }


def separate_families(
    witness: Witness, source: list[int], unrelated: list[int], tp: int
) -> Evidence:
    prompt = source[:PROMPT]
    witness.reset()
    cold, control = check_cpu_restore(witness, "split-cold", prompt, tp, Coverage())
    report = {"cold": control, "output_ids": cold[0]}
    for name, gqa, matched_gdn, gdn, draft in (
        ("gqa-only", DEEP, 0, 0, 0),
        ("gdn-only", 0, DEEP, 0, 0),
        ("gqa-ahead", DEEP, MIDDLE, MIDDLE, DEEP - 1),
        ("gdn-ahead", MIDDLE, DEEP, MIDDLE, DEEP - 1),
        ("draft-hole", DEEP, DEEP, MIDDLE, MIDDLE),
    ):
        witness.reset()
        backed = prime_for_pressure(witness, name, prompt, unrelated)
        records = {
            family: list(transfer_records(backed, RANK_PUT, family)) for family in (GQA, GDN, DRAFT)
        }
        state = {
            (device, record["endpoint"]): key
            for device, key, record in records[GDN]
            if record["endpoint"] in (SHALLOW, MIDDLE, DEEP)
        }
        assert len(state) == tp * 3, state
        erase = {key for (_, endpoint), key in state.items() if endpoint > matched_gdn}
        erase.update(
            key
            for device, key, record in records[GQA]
            if record["end"] > gqa
            and (device, record["end"]) in state
            and key[1] == state[device, record["end"]][1]
        )
        if draft == 0:
            erase.update(key for _, key, _ in records[DRAFT])
        elif draft == MIDDLE:
            erase.update(
                key
                for device, key, record in records[DRAFT]
                if record["end"] == DEEP - 1 and key[1] == state[device, DEEP][1]
            )
        assert erase, name
        witness.erase(erase)
        output, observed = check_cpu_restore(witness, name, prompt, tp, Coverage(gqa, gdn, draft))
        assert output == cold, (name, output, cold)
        for rank in observed["rank_inventory"]:
            assert rank["gqa"] == gqa, (name, rank)
            assert max((end for end, _ in rank["checkpoints"]), default=0) == matched_gdn, rank
        repeat_resident(witness, f"{name}-same-history", prompt, output)
        report[name] = observed
    return report


def complementary_ranks(witness: Witness, source: list[int], unrelated: list[int]) -> Evidence:
    prompt = source[:PROMPT]
    witness.reset()
    expected, _ = check_cpu_restore(
        witness, "rank-intersection-cold", prompt, RANK_PAIR, Coverage()
    )
    witness.reset()
    backed = prime_for_pressure(witness, "rank-intersection", prompt, unrelated)
    state = list(transfer_records(backed, RANK_PUT, GDN))
    first, second = sorted({device for device, _, _ in state})
    deleted = {(first, MIDDLE), (second, DEEP)}
    erase = {key for device, key, record in state if (device, record["endpoint"]) in deleted}
    deep = next(
        key[1] for device, key, record in state if (device, record["endpoint"]) == (second, DEEP)
    )
    erase.add((GQA, deep))
    assert len(erase) == len(deleted) + 1, erase
    witness.erase(erase)
    output, observed = check_cpu_restore(
        witness, "rank-intersection", prompt, RANK_PAIR, Coverage(MIDDLE, SHALLOW, DEEP - 1)
    )
    assert output == expected, (output, expected)
    inventory = observed["rank_inventory"]
    assert sorted(rank["gqa"] for rank in inventory) == [MIDDLE, DEEP], inventory
    endpoints = [{end for end, _ in rank["checkpoints"]} for rank in inventory]
    assert sorted(map(max, endpoints)) == [MIDDLE, DEEP]
    assert max(endpoints[0] & endpoints[1]) == SHALLOW
    return observed


def failed_get(witness: Witness, source: list[int], unrelated: list[int], tp: int) -> Evidence:
    prompt = source[:PROMPT]
    witness.reset()
    expected, _ = check_cpu_restore(witness, "error-cold", prompt, tp, Coverage())
    witness.reset()
    prime_for_pressure(witness, "error", prompt, unrelated)
    before = witness.snapshot("error-before")
    gate = witness.relay.arm(RANK_GET, GQA, drop=True)
    request = Stream(witness, "dropped-daemon-reply", prompt, 1)
    try:
        gate.wait()
    finally:
        gate.release.set()
    assert gate.finished.wait(10)
    actual = http.native_result(request.finish(), prompt, 1, 0)
    after = witness.snapshot("error-after")
    assert not gate.entry["forwarded"]
    assert tokens(actual) == expected
    assert after["io_errors"] > before["io_errors"]
    for name in ("gqa_restored_tokens", "gdn_restored_tokens", "draft_restored_rows"):
        assert after[name] == before[name], (name, before, after)
    assert after["gdn_computed_tokens"] - before["gdn_computed_tokens"] == len(prompt)
    witness.get("/health")
    backed = backup_under_pressure(witness, "recovery", unrelated)
    assert list(transfer_records(backed, RANK_PUT, GDN))
    recovered, restored = check_cpu_restore(
        witness, "reconnected-get", prompt, tp, Coverage(len(prompt), len(prompt), len(prompt) - 1)
    )
    assert recovered == expected, (recovered, expected)
    return {"dropped_reply_errors": after["io_errors"] - before["io_errors"], "recovered": restored}


def worker_restart(
    args: Namespace, relay: Relay, options: tuple[str, ...], source: list[int], unrelated: list[int]
) -> Evidence:
    prompt = source[:DEEP]
    with http.serve(args, "split-publisher", *options) as (port, _):
        writer = Witness(port, args.evidence, args.daemon, relay)
        writer.reset()
        cold, _ = check_cpu_restore(writer, "restart-cold", prompt, args.tp, Coverage())
        repeat_resident(writer, "restart-resident", prompt, cold)
        backup_under_pressure(writer, "restart", unrelated)
    isolated = None
    if args.alternate_checkpoint:
        alternate = SimpleNamespace(**(vars(args) | {"checkpoint": args.alternate_checkpoint}))
        with http.serve(alternate, "split-other-weights", *options) as (port, _):
            reader = Witness(port, args.evidence, args.daemon, relay)
            _, isolated = check_cpu_restore(
                reader, "split-other-weights", prompt, args.tp, Coverage()
            )
            assert all(not inv["checkpoints"] for inv in isolated["rank_inventory"])
    with http.serve(args, "split-reader", *options) as (port, _):
        reader = Witness(port, args.evidence, args.daemon, relay)
        actual, persisted = check_cpu_restore(
            reader, "restart-persisted", prompt, args.tp, Coverage(DEEP, DEEP, DEEP - 1)
        )
        assert actual == cold, (actual, cold)
    return {"output_ids": cold[0], "persisted": persisted, "other_weights": isolated}
