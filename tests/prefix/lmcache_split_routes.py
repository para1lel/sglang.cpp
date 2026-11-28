"""Independent cache families and exact checkpoint endpoints through native serving."""

from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, NamedTuple

from cold_cache import PREFILL_CHECKPOINTS, parse_metrics
from lmcache_relay import DRAFT, GDN, GQA, PREFIX_DETAIL, RANK_GET, RANK_PUT, restore_gdn
from lmcache_route_helpers import NAMESPACE, Stream, Witness, capture_case, tokens, transfer_records

from media import http_routes as http

if TYPE_CHECKING:
    from argparse import Namespace
    from pathlib import Path

    from lmcache_relay import Relay

Evidence = dict[str, Any]
Record = tuple[str, tuple[int, bytes], Evidence]
Output = tuple[list[int], str]
CACHE_CHUNK_ROWS = 256
RANK_PAIR = 2
MTP_STEPS = 4
SPLIT_PROMPT_ROWS = 896
OFFSET_START = 768
OFFSET_END = 2390
OFFSET_ANCHORS = [1366, 1878, 2134, 2262]


class ExpectedWork(NamedTuple):
    gqa: int
    gdn: int
    count: int = 8


SPLIT_COLD_IDS = [13332, 220, 16, 17, 23, 11268, 8928, 13]
SPLIT_REPLAY_IDS = [13332, 220, 16, 17, 23, 15746, 14103, 13]
TP2_ATTENTION_ONLY_IDS = [13332, 220, 21, 24, 11268, 8928, 13, 40149]


def published_since(witness: Witness, offset: int, family: int) -> list[Record]:
    witness.drain()
    return list(transfer_records(witness.relay.since(offset), RANK_PUT, family))


def phase_coverage(witness: Witness, offset: int, tp: int, tokens: int) -> list[Evidence]:
    observed = {
        entry["coverage_query"]["rank_key"]: entry["coverage"]
        for entry in witness.relay.since(offset)
        if entry["opcode"] == PREFIX_DETAIL
        and entry["status"] == 0
        and entry["coverage_query"]["count"] == tokens
    }
    assert len(observed) == tp, (tokens, observed)
    for inventory in observed.values():
        ends = [end for end, _ in inventory["checkpoints"]]
        assert ends == sorted(set(ends), reverse=True), inventory
    return list(observed.values())


def work(
    witness: Witness, label: str, prompt: list[int], tp: int, expected: ExpectedWork
) -> tuple[Output, Evidence]:
    gqa, gdn, count = expected
    offset = len(witness.relay.since())
    result, delta = witness.changes(
        label,
        prompt,
        gdn,
        {
            "gqa_restored_tokens": gqa,
            "gqa_read_only_tokens": gqa - gdn,
            "gqa_computed_tokens": len(prompt) - gqa,
            "gdn_restored_tokens": gdn,
            "gdn_computed_tokens": len(prompt) - gdn,
            "draft_restored_rows": max(gdn - 1, 0),
            "io_errors": 0,
        },
        count=count,
    )
    entries = witness.relay.since(offset)
    restores = [
        entry
        for entry in entries
        if entry["opcode"] == RANK_GET and entry["status"] == 0 and entry.get("forwarded")
    ]
    bytes_read = [sum(entry["receipt"][i] for entry in restores) for i in range(3)]
    for entry in restores:
        assert list(entry["receipt"][:3]) == [
            sum(record["bytes"] for record in entry["records"] if record["family"] == family)
            for family in (GQA, GDN, DRAFT)
        ], entry
    assert bytes_read[0] == gqa * 33792, bytes_read
    assert bool(bytes_read[1]) == bool(gdn), bytes_read
    assert bytes_read[2] == max(gdn - 1, 0) * 2112, bytes_read
    if gqa:
        assert len({entry["device"] for entry in restores}) == tp
    return tokens(result), {
        "work": delta,
        "read_bytes": bytes_read,
        "rank_inventory": phase_coverage(witness, offset, tp, len(prompt)),
    }


def exact_gqa_history(witness: Witness, source: list[int], tp: int) -> Evidence:
    prompt = source[:SPLIT_PROMPT_ROWS]
    witness.reset()
    offset = len(witness.relay.since())
    cold, original = work(witness, "exact-history-cold", prompt, tp, ExpectedWork(0, 0))
    states = published_since(witness, offset, GDN)
    assert sum(record["endpoint"] == len(prompt) for _, _, record in states) == tp
    witness.erase({key for _, key, _ in states})
    restored, observed = work(
        witness, "exact-history-gqa-only", prompt, tp, ExpectedWork(len(prompt), 0)
    )
    assert restored == cold, (restored, cold)
    return {"ids": cold[0], "cold": original, "restored": observed}


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
    states = published_since(witness, offset, GDN)
    terminal = {key for _, key, record in states if record["endpoint"] == endpoint}
    assert len(terminal) == tp, states
    witness.erase({key for _, key, record in states if record["endpoint"] != endpoint})
    restored, observed = work(
        witness,
        "generated-terminal-restore",
        committed,
        tp,
        ExpectedWork(endpoint, endpoint, continuation),
    )
    inventories = [inv["checkpoints"] for inv in observed["rank_inventory"]]
    assert all(len(items) == 1 and items[0][0] == endpoint for items in inventories), observed
    assert sum(items[0][1] for items in inventories) == 1, observed
    assert restored[0] == expected[prefix_generation - 1 :], (restored, expected)
    return {
        "prompt_rows": len(prompt),
        "terminal_rows": endpoint,
        "continuation_ids": restored[0],
        "restore": observed,
    }


def separate_families(witness: Witness, source: list[int], tp: int) -> Evidence:
    prompt = source[:SPLIT_PROMPT_ROWS]
    assert len(prompt) == SPLIT_PROMPT_ROWS
    witness.reset()
    cold, control = work(witness, "split-cold", prompt, tp, ExpectedWork(0, 0))
    report = {"cold": control}
    mismatches = {}
    for name, expected_gqa, matched_gdn, usable in (
        ("gqa-only", 768, 0, 0),
        ("gdn-only", 0, 768, 0),
        ("gqa-ahead", 768, 512, 512),
        ("gdn-ahead", 512, 768, 512),
    ):
        witness.reset()
        offset = len(witness.relay.since())
        restored = 0
        for end in (512, 768):
            generated = witness.generate(f"{name}-prime-{end}", prompt[:end])
            http.native_result(generated, prompt[:end], 1, restored)
            restored = end
            witness.drain()
        kv = published_since(witness, offset, GQA)
        state = published_since(witness, offset, GDN)
        draft = published_since(witness, offset, DRAFT)
        assert {record["endpoint"] for _, _, record in state} >= {512, 768}
        erase = {key for _, key, record in kv if record["end"] > expected_gqa}
        erase.update(key for _, key, record in state if record["endpoint"] > matched_gdn)
        if not expected_gqa:
            erase.update(key for _, key, _ in draft)
        witness.erase(erase)
        if name == "gdn-only":
            devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
            ordinals = {
                uuid.removeprefix("GPU-").replace("-", "").lower(): ordinal
                for ordinal, uuid in enumerate(devices)
            }
            standalone = {
                device: restore_gdn(
                    witness.relay.server_address, NAMESPACE, (ordinals[device], device), key, record
                )
                for device, key, record in state
                if record["endpoint"] == matched_gdn
            }
            assert len(standalone) == tp
            report["standalone_gdn"] = standalone
        output, observed = work(witness, name, prompt, tp, ExpectedWork(expected_gqa, usable))
        if output != cold:
            mismatches[name] = {"restored": output[0], "cold": cold[0]}
        for rank in observed["rank_inventory"]:
            assert rank["gqa"] == expected_gqa, (name, rank)
            assert max((end for end, _ in rank["checkpoints"]), default=0) == matched_gdn, rank
        repeat, restored = work(
            witness, f"{name}-same-history", prompt, tp, ExpectedWork(len(prompt), len(prompt))
        )
        assert repeat == output, (name, repeat, output)
        report[name] = observed | {"same_history": restored, "output_ids": output[0]}
    report["cold_ids"] = cold[0]
    report["different_cold_history"] = mismatches
    (witness.evidence / "separate-families.json").write_text(json.dumps(report, indent=2) + "\n")
    for name, pair in mismatches.items():
        expected = (
            TP2_ATTENTION_ONLY_IDS if name == "gqa-only" and tp == RANK_PAIR else SPLIT_REPLAY_IDS
        )
        assert name != "gdn-only", pair
        assert pair == {"cold": SPLIT_COLD_IDS, "restored": expected}, pair
    return report


def complementary_ranks(witness: Witness, source: list[int]) -> Evidence:
    prompt = source[:672]
    witness.reset()
    expected, _ = work(witness, "rank-intersection-cold", prompt, 2, ExpectedWork(0, 0))
    witness.reset()
    offset = len(witness.relay.since())
    for end in (256, 384, 512):
        witness.generate(f"rank-intersection-prime-{end}", prompt[:end])
        witness.drain()
    states = published_since(witness, offset, GDN)
    kv = published_since(witness, offset, GQA)
    first, second = sorted({device for device, _, _ in states})
    deleted = {(first, 384), (second, 512)}
    witness.erase(
        {key for device, key, record in states if (device, record["endpoint"]) in deleted}
        | {key for device, key, record in kv if (device, record["end"]) in deleted}
    )
    output, observed = work(witness, "rank-intersection", prompt, 2, ExpectedWork(256, 256))
    assert output == expected, (output, expected)
    inventory = observed["rank_inventory"]
    assert sorted(rank["gqa"] for rank in inventory) == [384, 512], inventory
    endpoints = [{end for end, _ in rank["checkpoints"]} for rank in inventory]
    assert sorted(map(max, endpoints)) == [384, 512]
    assert max(endpoints[0] & endpoints[1]) == CACHE_CHUNK_ROWS
    return observed


def offset_retention(
    witness: Witness, offset: int, states: list[Record], end: int, positions: list[int]
) -> dict[str, Evidence]:
    earlier = {
        (device, record["endpoint"]): key[1]
        for device, key, record in states
        if record["endpoint"] in positions
    }
    slots = {
        gpu.removeprefix("GPU-").replace("-", "").lower(): slot
        for slot, gpu in enumerate(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
    }
    recency = {}
    events = witness.relay.since(offset)
    for entry in events:
        if entry["opcode"] != RANK_PUT or not any(
            record.get("endpoint") == end for record in entry["records"]
        ):
            continue
        device = entry["device"]
        hint = entry["prior_gdn"]
        assert hint["anchors"] == [(anchor, earlier[device, anchor]) for anchor in positions]
        assert hint["draft_required"] == (slots[device] == 0), entry
        prerequisites = {
            (family, position): digest for family, position, digest in entry["prior_prerequisites"]
        }
        current = complete_chunks(entry)
        required = {(GQA, position) for position in range(256, positions[-1], 256)}
        if hint["draft_required"]:
            required |= {(DRAFT, position) for position in range(257, positions[-1], 256)}
        assert prerequisites.keys() == required - current.keys(), entry
        referenced = current | prerequisites
        known = {}
        for event in events:
            if event["opcode"] == RANK_PUT and event.get("device") == device:
                known.update(complete_chunks(event))
        assert all(referenced[key] == known[key] for key in required), entry
        recency[device] = {
            "anchors": positions,
            "prerequisite_count": len(prerequisites),
        }
    return recency


def complete_chunks(entry: Evidence) -> dict[tuple[int, int], bytes]:
    return {
        (record["family"], record["end"] + (record["family"] == DRAFT)): key[1]
        for key, record in zip(entry["keys"], entry["records"], strict=True)
        if record["tokens"] == CACHE_CHUNK_ROWS
    }


def offset_publish(witness: Witness, source: list[int], unrelated: list[int], tp: int) -> Evidence:
    prompt = source[:OFFSET_END]
    assert len(prompt) == OFFSET_END
    budget = 24
    witness.reset()
    cold = http.native_result(
        witness.generate("offset-cold", prompt, count=budget), prompt, budget, 0
    )
    witness.reset()
    witness.generate("offset-prime-768", prompt[:768])
    witness.drain()
    offset = len(witness.relay.since())
    gate = witness.relay.arm(RANK_PUT, GDN, endpoint=1366)
    producer = Stream(witness, "offset-producer", prompt, budget)
    try:
        held = gate.wait()
        peer = Stream(witness, "offset-peer", unrelated[:128], 8)
        progressed, _ = peer.progress(held["reply_ns"])
        result = http.native_result(producer.finish(), prompt, budget, 768)
        assert not gate.release.is_set()
        assert witness.pending_stores() > 0
    finally:
        gate.release.set()
    assert gate.finished.wait(10)
    http.native_result(peer.finish(), unrelated[:128], 8, 0)
    states = published_since(witness, offset, GDN)
    precursors = {
        (device, record["endpoint"])
        for device, _, record in states
        if OFFSET_START < record["endpoint"] <= len(prompt)
    }
    assert len(precursors) == tp * (len(OFFSET_ANCHORS) + 1), precursors
    assert {end for _, end in precursors} == {*OFFSET_ANCHORS, OFFSET_END}, precursors
    recency = offset_retention(witness, offset, states, len(prompt), OFFSET_ANCHORS)
    assert len(recency) == tp, recency
    exact, full = work(witness, "offset-exact", prompt, tp, ExpectedWork(2390, 2390))
    other = next(token for token in unrelated if token != prompt[-1])
    fork = [*prompt[:-1], other, *unrelated[1:4]]
    output, observed = work(witness, "offset-last-row-fork", fork, tp, ExpectedWork(2304, 2262))
    witness.reset()
    fork_cold = http.native_result(witness.generate("offset-fork-cold", fork, count=8), fork, 8, 0)
    report = {
        "endpoints": sorted(precursors),
        "peer_progress_ns": progressed,
        "held_ack_ns": held["reply_ns"],
        "full": full,
        "fork": observed,
        "recency": recency,
        "cold_producer_ids": cold[0],
        "segmented_producer_ids": result[0],
        "exact_restore_ids": exact[0],
    }
    (witness.evidence / "offset-publish.json").write_text(json.dumps(report, indent=2) + "\n")
    assert exact[0] == result[0][: len(exact[0])], (exact, result[0])
    assert output == tokens(fork_cold), (output, fork_cold[0])
    return report


def offset_boundaries(
    witness: Witness, source: list[int], unrelated: list[int], tp: int
) -> Evidence:
    limit = 32768
    prompt = (source * ((limit + len(source) - 1) // len(source)))[:limit]
    report = {}
    mismatches = {}
    for start, end, planned in (
        (1000, 1256, [1128, 1256]),
        (30000, 32768, [30720, 31744, 32256, 32512, 32640, 32768]),
    ):
        subject = prompt[:end]
        witness.reset()
        witness.generate(f"offset-prime-{start}", prompt[:start])
        witness.drain()
        offset = len(witness.relay.since())
        witness.generate(f"offset-extend-{end}", subject)
        endpoints = {
            (device, record["endpoint"])
            for device, _, record in published_since(witness, offset, GDN)
        }
        assert len(endpoints) == len(planned) * tp, (start, end, endpoints)
        assert {pos for _, pos in endpoints} == set(planned), (start, end, endpoints, planned)
        replacement = next(token for token in unrelated if token != subject[-1])
        fork = [*subject[:-1], replacement, *unrelated[1:4]]
        output, observed = work(
            witness, f"offset-boundary-fork-{end}", fork, tp, ExpectedWork(planned[-2], planned[-2])
        )
        repeat, exact = work(
            witness, f"offset-boundary-repeat-{end}", fork, tp, ExpectedWork(len(fork), len(fork))
        )
        assert repeat == output, (end, repeat, output)
        witness.reset()
        cold, _ = work(witness, f"offset-boundary-cold-{end}", fork, tp, ExpectedWork(0, 0))
        if output != cold:
            mismatches[end] = {"restored": output[0], "cold": cold[0]}
        report[end] = {
            "restored": start,
            "endpoints": sorted(endpoints),
            "fork": observed,
            "same_history": exact,
            "cold_ids": cold[0],
            "restored_ids": output[0],
        }
    (witness.evidence / "offset-boundaries.json").write_text(json.dumps(report, indent=2) + "\n")
    known = {
        1256: {
            "restored": [198, 16, 19, 25, 15, 15, 11, 279],
            "cold": [198, 16, 17, 25, 15, 15, 11, 279],
        }
    }
    if mismatches:
        assert tp == RANK_PAIR, mismatches
        assert mismatches == known, mismatches
    return report


def generate_burst(
    witness: Witness, prompts: list[list[int]], salts: list[str], count: int, label: str
) -> list[list[int]]:
    barrier = Barrier(len(prompts))

    def issue(slot: int) -> list[int]:
        barrier.wait(timeout=5)
        prompt, salt = prompts[slot], salts[slot]
        reply = witness.generate(f"{label}-{slot}", prompt, count=count, cache_salt=salt)
        return http.native_result(reply, prompt, count, 0)[0]

    with ThreadPoolExecutor(max_workers=len(prompts)) as executor:
        return list(executor.map(issue, range(len(prompts))))


def offset_burst(witness: Witness, source: list[int], tp: int) -> Evidence:
    width = 8 * tp
    prompts = [source[slot * 256 : slot * 256 + 255] for slot in range(width)]
    assert len({tuple(prompt) for prompt in prompts}) == width
    salts = [f"burst-{slot}" for slot in range(width)]
    witness.reset()
    expected = [
        http.native_result(
            witness.generate(f"burst-cold-{slot}", prompt, cache_salt=salt), prompt, 1, 0
        )[0]
        for slot, (prompt, salt) in enumerate(zip(prompts, salts, strict=True))
    ]
    witness.reset()
    offset = len(witness.relay.since())
    actual = generate_burst(witness, prompts, salts, 1, "burst-prime")
    records = published_since(witness, offset, GDN)
    counts = {}
    for end in (127, 255):
        keys = {key for _, key, record in records if record["endpoint"] == end}
        assert len(keys) == width * tp, (end, len(keys), width, tp)
        witness.lookup(sorted(keys), (1,) * len(keys))
        counts[end] = len(keys)
    for slot, (prompt, salt) in enumerate(zip(prompts, salts, strict=True)):
        reply = witness.generate(f"burst-repeat-{slot}", prompt, cache_salt=salt)
        repeated = http.native_result(reply, prompt, 1, len(prompt))[0]
        assert repeated == actual[slot], (slot, repeated, actual[slot])
    differences = {
        slot: {"concurrent": got, "serial": cold}
        for slot, (got, cold) in enumerate(zip(actual, expected, strict=True))
        if got != cold
    }
    report = {"width": width, "endpoint_counts": counts, "cold_partition_differences": differences}
    (witness.evidence / "offset-burst.json").write_text(json.dumps(report, indent=2) + "\n")
    known = {8: {"concurrent": [17], "serial": [16]}}
    if differences:
        assert tp == RANK_PAIR, differences
        assert differences == known, differences
    return report


def wide_burst(witness: Witness, source: list[int], tp: int, server_log: Path) -> Evidence:
    width, rows, budget = 8 * tp, 1535, 64
    endpoints = (511, 1023, 1279, 1407, 1535)
    prompts = [source[slot * 97 : slot * 97 + rows] for slot in range(width)]
    assert all(len(prompt) == rows for prompt in prompts)
    assert len({tuple(prompt) for prompt in prompts}) == width
    salts = [f"wide-burst-{slot}" for slot in range(width)]
    witness.reset()
    offset = len(witness.relay.since())
    log_offset = server_log.stat().st_size
    before = parse_metrics(witness.get("/metrics").decode(), PREFILL_CHECKPOINTS)
    generate_burst(witness, prompts, salts, budget, "wide-burst")
    records = published_since(witness, offset, GDN)
    after = parse_metrics(witness.get("/metrics").decode(), PREFILL_CHECKPOINTS)
    counters = {
        key.removeprefix("sglang_cpp:lmcache_").removesuffix("_total"): after[key] - before[key]
        for key in before
    }
    counts = {}
    for end in endpoints:
        keys = {key for _, key, record in records if record["endpoint"] == end}
        assert len(keys) == width * tp, (end, len(keys), width, tp)
        witness.lookup(sorted(keys), (1,) * len(keys))
        counts[end] = len(keys)
    with server_log.open("rb") as log:
        log.seek(log_offset)
        widths = [int(value) for value in re.findall(rb"\bbatch width=(\d+)", log.read())]
    report = {
        "width": width,
        "observed_peak": max(widths, default=0),
        "prompt_rows": rows,
        "output_rows": budget,
        "endpoint_counts": counts,
        "counters": counters,
    }
    (witness.evidence / "wide-burst.json").write_text(json.dumps(report, indent=2) + "\n")
    assert report["observed_peak"] == width, report
    expected = width * len(endpoints)
    assert counters == {
        "prefill_checkpoints_planned": expected,
        "prefill_checkpoints_resident": 0,
        "prefill_checkpoints_published": expected,
        "prefill_checkpoints_declined": 0,
        "prefill_checkpoint_stores": expected,
        "prefill_checkpoint_failures": 0,
    }, report
    return report


def split_persistence(
    args: Namespace, relay: Relay, options: tuple[str, ...], source: list[int]
) -> Evidence:
    prompt = source[:1256]
    with http.serve(args, "split-publisher", *options) as (port, _):
        writer = Witness(port, args.evidence, args.daemon, relay)
        writer.reset()
        cold, _ = work(writer, "split-full-cold", prompt, args.tp, ExpectedWork(0, 0, 1))
        offset = len(relay.since())
        warm, in_process = work(
            writer, "split-full-repeat", prompt, args.tp, ExpectedWork(1256, 1256, 1)
        )
        assert not published_since(writer, offset, GDN), (
            "Repeat republished resident exact GDN state"
        )
        assert warm == cold, (warm, cold)
    isolated = None
    if args.alternate_checkpoint:
        alternate = SimpleNamespace(**(vars(args) | {"checkpoint": args.alternate_checkpoint}))
        with http.serve(alternate, "split-other-weights", *options) as (port, _):
            reader = Witness(port, args.evidence, args.daemon, relay)
            _, isolated = work(
                reader, "split-other-weights", prompt, args.tp, ExpectedWork(0, 0, 1)
            )
            assert all(not inv["checkpoints"] for inv in isolated["rank_inventory"])
    with http.serve(args, "split-reader", *options) as (port, _):
        reader = Witness(port, args.evidence, args.daemon, relay)
        actual, persisted = work(
            reader, "split-full-persisted", prompt, args.tp, ExpectedWork(1256, 1256, 1)
        )
        assert actual == cold, (actual, cold)
    return {"in_process": in_process, "persisted": persisted, "other_weights": isolated}


def split_only(
    args: Namespace, relay: Relay, path: Path, source: list[int], unrelated: list[int]
) -> Evidence:
    report, failures = {}, {}
    options = (
        "--tp",
        str(args.tp),
        "--mtp-steps",
        "4",
        "--max-num-batched-tokens",
        "32768",
        "--max-active",
        str(8 * args.tp),
        "--lmcache-socket",
        str(path),
    )
    with http.serve(args, "split-cache", *options) as (port, server_log):
        witness = Witness(port, args.evidence, args.daemon, relay)
        info = json.loads(witness.get("/server_info"))
        assert info["lmcache_available"]
        assert info["tp_size"] == args.tp
        assert info["mtp_steps"] == MTP_STEPS
        capture_case(
            report, failures, "exact_gqa_history", exact_gqa_history, witness, source, args.tp
        )
        capture_case(
            report, failures, "generated_terminal", generated_terminal, witness, source, args.tp
        )
        capture_case(
            report, failures, "separate_families", separate_families, witness, source, args.tp
        )
        capture_case(
            report, failures, "offset_publish", offset_publish, witness, source, unrelated, args.tp
        )
        capture_case(
            report,
            failures,
            "offset_boundaries",
            offset_boundaries,
            witness,
            source,
            unrelated,
            args.tp,
        )
        capture_case(report, failures, "offset_burst", offset_burst, witness, source, args.tp)
        capture_case(
            report, failures, "wide_burst", wide_burst, witness, source, args.tp, server_log
        )
        if args.tp == RANK_PAIR:
            capture_case(
                report, failures, "complementary_ranks", complementary_ranks, witness, source
            )
    capture_case(
        report, failures, "split_persistence", split_persistence, args, relay, options, source
    )
    return report | {"failures": failures}
