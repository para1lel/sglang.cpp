"""Untimed wire and CUDA-work assertions against real LMCache storage."""

import argparse
import base64
import hashlib
import json
import sys
from copy import deepcopy
from http import HTTPStatus
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic, perf_counter_ns, sleep
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks" / "paired"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cold_cache import RUNNING, WAITING, metrics
from lmcache_relay import (
    DRAFT,
    ERASE,
    GDN,
    GQA,
    LEASE,
    PREFIX_DETAIL,
    RANK_GET,
    RANK_PUT,
    READY,
    WIRE_VERSION,
    Relay,
    arm_device_fence,
    control,
    key_body,
)
from lmcache_route_helpers import (
    NAMESPACE,
    Stream,
    Witness,
    capture_case,
    tokens,
    transfer_records,
    wait_until,
)
from lmcache_send_trace import require_deferred_send, terminal_times, traced_worker
from lmcache_split_routes import split_only

from benchmarks.concurrent.reference import load_lanes
from media import http_routes as http

LOW, HIGH, PROMPT = 1279, 1391, 1536
TERMINAL_OWNER_STORES = 3


def fixture(fixtures: Path) -> list[list[int]]:
    lanes = load_lanes(fixtures, "W3")
    return [case["prompt_token_ids"] for _, _, case in lanes[:2]]


def checkpoint_keys(witness: Witness, begin: int, endpoint: int) -> set[tuple[int, bytes]]:
    return {
        key
        for _, key, record in transfer_records(witness.relay.since(begin), RANK_PUT, GDN)
        if record["endpoint"] == endpoint
    }


def terminal_checkpoint(witness: Witness, begin: int, endpoint: int) -> tuple[int, bytes]:
    (key,) = checkpoint_keys(witness, begin, endpoint)
    return key


def selective(witness, source):
    prefix = source[:PROMPT]
    witness.reset()
    cold, cold_delta = witness.changes(
        "selective-cold",
        prefix,
        0,
        {
            "gqa_restored_tokens": 0,
            "gqa_read_only_tokens": 0,
            "gdn_restored_tokens": 0,
            "gqa_computed_tokens": PROMPT,
            "gdn_computed_tokens": PROMPT,
            "draft_restored_rows": 0,
            "io_errors": 0,
        },
    )
    assert cold_delta["gqa_written_tokens"] >= PROMPT, cold_delta
    witness.reset()
    offset = len(witness.relay.since())
    _, low_keys = witness.stores("prime-1279", prefix[:LOW])
    low_gdn = terminal_checkpoint(witness, offset, LOW)
    offset = len(witness.relay.since())
    _, high_keys = witness.stores("prime-1391", prefix[:HIGH])
    high_gdn = terminal_checkpoint(witness, offset, HIGH)
    assert low_gdn != high_gdn
    records = transfer_records(witness.relay.since(offset), RANK_PUT, GDN)
    witness.erase({key for _, key, record in records if record["endpoint"] > LOW})
    witness.lookup([low_gdn, high_gdn], (1, 0))
    offset = len(witness.relay.since())
    ahead, ahead_delta = witness.changes(
        "gqa-ahead",
        prefix,
        LOW,
        {
            "gqa_restored_tokens": HIGH,
            "gqa_read_only_tokens": HIGH - LOW,
            "gqa_computed_tokens": PROMPT - HIGH,
            "gdn_restored_tokens": LOW,
            "gdn_computed_tokens": PROMPT - LOW,
            "draft_restored_rows": LOW - 1,
            "io_errors": 0,
        },
    )
    assert tokens(ahead) == tokens(cold), (ahead, cold)
    restored = set().union(*(read_keys(witness, offset, family) for family in (GQA, GDN, DRAFT)))
    assert low_gdn in restored and high_gdn not in restored, restored
    assert any(key in restored for key in high_keys - low_keys if key[0] == GQA), restored
    witness.reset()
    offset = len(witness.relay.since())
    witness.stores("prime-inverse", prefix[:HIGH])
    gqa_root = next(
        key
        for entry in witness.relay.since(offset)
        if entry["opcode"] == RANK_PUT
        for key in entry["keys"]
        if key[0] == GQA
    )
    gdn = terminal_checkpoint(witness, offset, HIGH)
    assert witness.action(ERASE, key_body(gqa_root))[0] == 0
    witness.lookup([gqa_root, gdn], (0, 1))
    offset = len(witness.relay.since())
    inverse, inverse_delta = witness.changes(
        "gdn-ahead",
        prefix,
        0,
        {
            key: cold_delta[key]
            for key in (
                "gqa_restored_tokens",
                "gqa_read_only_tokens",
                "gdn_restored_tokens",
                "gqa_computed_tokens",
                "gdn_computed_tokens",
                "draft_restored_rows",
                "io_errors",
            )
        },
    )
    writes = [
        entry["receipt"][3]
        for entry in witness.relay.since(offset)
        if entry["opcode"] == RANK_PUT and entry["status"] == 0 and entry.get("forwarded")
    ]
    assert inverse_delta["gqa_written_tokens"] == sum(writes) > 0, (inverse_delta, writes)
    witness.lookup([gqa_root], (1,))
    assert tokens(inverse) == tokens(cold), (inverse, cold)
    assert gdn not in read_keys(witness, offset, GDN)
    return {"cold": cold_delta, "gqa_ahead": ahead_delta, "gdn_ahead": inverse_delta}


def natural_prefix(witness, source, unrelated):
    stock = source[:1416]
    changed = next(token for token in unrelated[:64] if token != stock[HIGH])
    prompt = source[:HIGH] + [changed] + source[HIGH + 1:PROMPT]
    assert prompt[:HIGH] == stock[:HIGH] and prompt[HIGH] != stock[HIGH]
    witness.reset()
    cold = http.native_result(witness.generate("natural-prefix-cold", prompt), prompt, 1, 0)
    witness.reset()
    offset = len(witness.relay.since())
    witness.stores("natural-prime-1279", stock[:LOW])
    coherent = terminal_checkpoint(witness, offset, LOW)
    offset = len(witness.relay.since())
    witness.stores("natural-prime-1416", stock)
    divergent = terminal_checkpoint(witness, offset, len(stock))
    records = transfer_records(witness.relay.since(offset), RANK_PUT, GDN)
    witness.erase({key for _, key, record in records if LOW < record["endpoint"] <= HIGH})
    witness.lookup([coherent, divergent], (1, 1))
    offset = len(witness.relay.since())
    attention_rows = len(stock) - 128
    warm, delta = witness.changes(
        "natural-lcp-1391",
        prompt,
        LOW,
        {
            "gqa_restored_tokens": attention_rows,
            "gqa_read_only_tokens": attention_rows - LOW,
            "gqa_computed_tokens": PROMPT - attention_rows,
            "gdn_restored_tokens": LOW,
            "gdn_computed_tokens": PROMPT - LOW,
            "draft_restored_rows": LOW - 1,
            "io_errors": 0,
        },
    )
    assert tokens(warm) == tokens(cold)
    assert read_keys(witness, offset, GDN) == {coherent}
    return {"stock_prompt_tokens": len(stock), "common_prompt_tokens": HIGH, "work": delta}


def public_salt(witness, source):
    prompt = source[:PROMPT]
    witness.reset()
    salts = ("tenant\0a", "tenant\0b")
    controls = {}
    for name, salt in enumerate(salts):
        offset = len(witness.relay.since())
        cold, _ = witness.changes(f"salt-{name}-cold", prompt, 0, {
            "gqa_restored_tokens": 0, "gdn_restored_tokens": 0, "draft_restored_rows": 0,
            "gqa_computed_tokens": PROMPT, "gdn_computed_tokens": PROMPT, "io_errors": 0,
        }, cache_salt=salt)
        keys = witness.published_keys(offset)
        assert {family for family, _ in keys} == {GQA, GDN, DRAFT}
        controls[salt] = cold, keys
    first, second = (controls[salt][1] for salt in salts)
    assert first.isdisjoint(second)
    witness.lookup(sorted(first | second), [1] * (len(first) + len(second)))
    report = []
    for name, salt in enumerate(salts):
        offset = len(witness.relay.since())
        warm, delta = witness.changes(f"salt-{name}-warm", prompt, PROMPT, {
            "gqa_restored_tokens": PROMPT, "gdn_restored_tokens": PROMPT,
            "draft_restored_rows": PROMPT - 1, "io_errors": 0,
        }, cache_salt=salt)
        cold, keys = controls[salt]
        assert tokens(warm) == tokens(cold)
        reads = set().union(*(read_keys(witness, offset, family) for family in (GQA, GDN, DRAFT)))
        assert reads <= keys, (name, keys, reads)
        assert {family for family, _ in reads} == {GQA, GDN, DRAFT}, (name, reads)
        report.append({"restored": delta, "keys_read": len(reads)})
    return report


def media_identity(witness):
    image = http.visual("image_url", 32, 64)
    different = deepcopy(image)
    different["image_url"]["url"] = http.RGB8 + base64.b64encode(
        bytes((195, 30, 80)) * (32 * 64)).decode()
    message = {"type": "text", "text": "Describe what is in this image."}
    budget = 1
    bodies = [http.chat([part, message], budget) for part in (image, different)]

    def generate(label, body):
        return http.capture(SimpleNamespace(evidence=witness.evidence), label, witness.port,
                            http.CHAT, body)

    witness.reset()
    changed_cold = generate("media-changed-cold", bodies[1])
    changed_ids, changed_usage = http.received(changed_cold, budget)
    witness.reset()
    offset = len(witness.relay.since())
    primed = generate("media-original-prime", bodies[0])
    original_ids, original_usage = http.received(primed, budget)
    witness.drain()
    state = terminal_checkpoint(witness, offset, original_usage["prompt_tokens"])
    prompt_tokens = original_usage["prompt_tokens"]
    assert changed_usage["prompt_tokens"] == prompt_tokens
    assert primed["prompt_token_ids"] == changed_cold["prompt_token_ids"]
    offset = len(witness.relay.since())
    same = generate("media-original-warm", bodies[0])
    same_ids, _ = http.received(same, budget, prompt_tokens)
    assert same_ids == original_ids and read_keys(witness, offset, GDN) == {state}
    offset = len(witness.relay.since())
    changed = generate("media-changed-probe", bodies[1])
    cached = changed["usage"]["prompt_tokens_details"]["cached_tokens"]
    assert 0 <= cached < prompt_tokens and state not in read_keys(witness, offset, GDN)
    assert http.received(changed, budget, cached)[0] == changed_ids
    witness.drain()
    changed_state = terminal_checkpoint(witness, offset, prompt_tokens)
    assert changed_state != state
    witness.lookup([state, changed_state], (1, 1))
    offset = len(witness.relay.since())
    same_changed = generate("media-changed-warm", bodies[1])
    assert http.received(same_changed, budget, prompt_tokens)[0] == changed_ids
    assert read_keys(witness, offset, GDN) == {changed_state}
    return {"prompt_tokens": prompt_tokens, "same_media_cached_tokens": prompt_tokens,
            "changed_media_cached_tokens": cached, "same_wire_prompt_tokens": True}


def scored_reference(args, witness, source):
    prompt = source[:132]
    options = {"return_logprob": True, "logprob_start_len": 128, "top_logprobs_num": 5}
    with http.serve(args, "scored-reference", "--prefix-cache-entries", "0",
                    "--max-num-batched-tokens", "128") as (port, _):
        disabled = Witness(port, args.evidence, None, None)
        reference = disabled.generate("scored-split-reference", prompt, count=4, **options)
    cold = http.native_result(reference, prompt, 4, 0, start=128)
    witness.reset()
    witness.stores("prime-scored-128", prompt[:128])
    before = witness.snapshot("scored-warm-before")
    generated = witness.generate("scored-warm", prompt, count=4, **options)
    warm = http.native_result(generated, prompt, 4, 128, start=128)
    after = witness.snapshot("scored-warm-after")
    assert cold == warm, (cold, warm)
    assert after["gdn_restored_tokens"] - before["gdn_restored_tokens"] == 128
    assert after["gdn_computed_tokens"] - before["gdn_computed_tokens"] == 4
    assert after["draft_restored_rows"] - before["draft_restored_rows"] == 127
    return {"cached_tokens": 128, "prompt_score_rows": len(warm[2]["input_token_logprobs"]),
            "generated_score_rows": len(warm[2]["output_token_logprobs"])}


def read_keys(witness, offset, family):
    return {key for _, key, _ in transfer_records(witness.relay.since(offset), RANK_GET, family)}


def shifted_draft(witness, source, unrelated, position):
    prefix = source[:position]
    other = next(token for token in unrelated[:64] if token != prefix[-1])
    changed = prefix[:-1] + [other]
    suffix = [unrelated[64]]
    original, neighbor = prefix + suffix, changed + suffix
    expected = []
    for name, prompt in (("original", original), ("neighbor", neighbor)):
        witness.reset()
        cold = witness.generate(f"draft-{position}-{name}-cold", prompt, count=8)
        expected.append(tokens(http.native_result(cold, prompt, 8, 0)))
    witness.reset()
    _, shared = witness.stores(f"draft-{position}-shared", prefix[:-1])
    _, exact = witness.stores(f"draft-{position}-exact", prefix)
    shifted = {key for key in exact - shared if key[0] == DRAFT}
    terminal, = shifted
    witness.lookup([terminal], (1,))
    recorded = {}
    for name, prompt, cache, output in (
            ("original", original, position, expected[0]),
            ("neighbor", neighbor, position - 1, expected[1])):
        offset = len(witness.relay.since())
        result, delta = witness.changes(f"draft-{position}-{name}-warm", prompt, cache, {
            "gdn_restored_tokens": cache, "draft_restored_rows": cache - 1, "io_errors": 0,
        }, count=8)
        selected = read_keys(witness, offset, DRAFT)
        assert bool(selected & shifted) == (name == "original"), (name, selected, shifted)
        assert selected, (name, delta)
        recorded[name] = delta | {"cold_ids": output[0], "restored_ids": tokens(result)[0],
                                  "parity": tokens(result) == output}
    witness.reset()
    witness.stores(f"draft-{position}-hole-shared", prefix[:-1])
    offset = len(witness.relay.since())
    _, primed = witness.stores(f"draft-{position}-hole-exact", prefix)
    gdn = terminal_checkpoint(witness, offset, position)
    assert terminal in primed and witness.action(ERASE, key_body(terminal))[0] == 0
    witness.lookup([terminal, gdn], (0, 1))
    offset = len(witness.relay.since())
    result, delta = witness.changes(f"draft-{position}-hole-warm", original, position - 1, {
        "gqa_restored_tokens": position, "gqa_read_only_tokens": 1,
        "gdn_restored_tokens": position - 1, "draft_restored_rows": position - 2, "io_errors": 0,
    }, count=8)
    assert terminal not in read_keys(witness, offset, DRAFT)
    recorded["missing_terminal_draft"] = delta | {"cold_ids": expected[0][0],
        "restored_ids": tokens(result)[0], "parity": tokens(result) == expected[0]}
    path = witness.evidence / f"draft-{position}.json"
    path.write_text(json.dumps(recorded, indent=2) + "\n")
    assert recorded["neighbor"]["parity"], (position, recorded)
    if position == LOW:
        assert (recorded["original"]["restored_ids"] ==
                recorded["missing_terminal_draft"]["restored_ids"]), recorded
    else:
        assert all(case["parity"] for case in recorded.values()), (position, recorded)
    return recorded


def event_receipt(relay):
    events = relay.since()
    return [entry | {"keys": [[family, hashlib.sha256(identity).hexdigest()]
                               for family, identity in entry["keys"]]} for entry in events]




def held_get(witness, source, unrelated):
    prime, prompt = source[:HIGH], source[:PROMPT]
    witness.reset()
    witness.stores("prime-held", prime)
    peer = Stream(witness, "unrelated-in-progress", unrelated[:128], 512)
    first_ns, first_count = peer.progress()
    gate = witness.relay.arm(RANK_GET, GQA)
    subject = Stream(witness, "held-cache-read", prompt, 8)
    try:
        upstream = gate.wait()
        assert not gate.release.is_set()
        peer_ns, peer_count = peer.progress(upstream["reply_ns"])
        released_ns = perf_counter_ns()
        assert first_ns < upstream["reply_ns"] < peer_ns <= released_ns
        assert first_count < peer_count
    finally:
        gate.release.set()
    assert gate.finished.wait(10)
    http.native_result(subject.finish(), prompt, 8, HIGH)
    http.native_result(peer.finish(), unrelated[:128], 512, 0)
    return {"peer_started_ns": first_ns, "daemon_reply_ns": upstream["reply_ns"],
            "peer_progress_ns": peer_ns, "released_ns": released_ns,
            "peer_ids_before": first_count, "peer_ids_during": peer_count}


def expect_load(witness, running, waiting):
    deadline = monotonic() + 5
    while monotonic() < deadline:
        load = metrics(witness.get("/metrics").decode(), "lmcache")
        if (load[RUNNING], load[WAITING]) == (running, waiting):
            return perf_counter_ns()
        sleep(0.01)
    raise AssertionError(("unexpected load", load[RUNNING], load[WAITING], running, waiting))


def abort_pending(witness, source):
    prompt = source[:PROMPT]
    witness.reset()
    witness.stores("prime-abort", source[:HIGH])
    offset = len(witness.relay.since())
    gate = witness.relay.arm(PREFIX_DETAIL)
    waiting = Stream(witness, "cancel-held-lookup", prompt, 8)
    try:
        upstream = gate.wait()
        status, _, raw = http.exchange(witness.port, "/abort_request",
                                       {"rid": waiting.label}, timeout=5)
        assert status == 200 and not raw, (status, raw)
        terminal = waiting.finish()
        assert terminal["meta_info"]["finish_reason"]["type"] == "abort"
        assert terminal["output_ids"] == []
        expect_load(witness, 0, 0)
        replacement = Stream(witness, waiting.label, prompt, 8, evidence_label="reused-abort-id")
        admitted_ns = expect_load(witness, 1, 0)
        assert upstream["reply_ns"] < admitted_ns and not gate.release.is_set()
        entries = witness.relay.since(offset)
        assert not any(entry["opcode"] in (RANK_GET, RANK_PUT) for entry in entries)
    finally:
        gate.release.set()
    assert gate.finished.wait(10)
    http.native_result(replacement.finish(), prompt, 8, HIGH)
    return {"lookup_held_ns": upstream["reply_ns"], "replacement_admitted_ns": admitted_ns,
            "replacement_cached_tokens": HIGH}


def held_put(witness, source, unrelated):
    position = 129
    prompt, peer_prompt = source[: position + 128], unrelated[:128]
    budget = 8
    assert len(prompt) == position + 128
    assert prompt[0] != peer_prompt[0]
    witness.reset()
    cold = witness.generate("held-put-cold", prompt, count=budget)
    expected = http.native_result(cold, prompt, budget, 0)
    witness.reset()
    offset = len(witness.relay.since())
    gate = witness.relay.arm(RANK_PUT, GDN, endpoint=position)
    writer = Stream(witness, "held-prefill-put", prompt, budget)
    try:
        upstream = gate.wait()
        (anchor,) = (key for key in upstream["keys"] if key[0] == GDN)
        anchor_bytes = upstream["receipt"][GDN - 1]
        assert anchor_bytes > 0
        peer = Stream(witness, "peer-before-put-ack", peer_prompt, budget)
        peer_ns, _ = peer.progress(upstream["reply_ns"])
        witness.lookup([anchor], (1,))
        written = http.native_result(writer.finish(), prompt, budget, 0)
        finished_ns = perf_counter_ns()
        assert not gate.release.is_set() and witness.pending_stores() > 0
        status, _, body = http.exchange(witness.port, "/reset_prefix_cache", {})
        assert status == 409, (status, body)
        released_ns = perf_counter_ns()
    finally:
        gate.release.set()
    assert gate.finished.wait(10) and upstream.get("forwarded") is True
    assert upstream["reply_ns"] < finished_ns <= released_ns
    http.native_result(peer.finish(), peer_prompt, budget, 0)
    assert tokens(written) == tokens(expected)
    witness.drain()
    states = {
        key: record["endpoint"]
        for _, key, record in transfer_records(witness.relay.since(offset), RANK_PUT, GDN)
    }
    expected_states = {
        1,
        position,
        len(prompt),
        len(prompt) + budget - 1,
        len(peer_prompt),
        len(peer_prompt) + budget - 1,
    }
    assert anchor in states, states
    assert set(states.values()) == expected_states, states
    assert len(states) == len(expected_states), states
    witness.lookup(list(states), (1,) * len(states))
    terminal_prompt = prompt + written[0][:-1]
    read_offset = len(witness.relay.since())
    restored, delta = witness.changes(
        "held-put-terminal-restore",
        terminal_prompt,
        len(terminal_prompt),
        {
            "gdn_restored_tokens": len(terminal_prompt),
            "draft_restored_rows": len(terminal_prompt) - 1,
            "io_errors": 0,
        },
    )
    (terminal,) = read_keys(witness, read_offset, GDN)
    assert terminal != anchor
    assert terminal in states
    witness.reset()
    replay = witness.generate("held-put-terminal-cold", terminal_prompt)
    assert tokens(restored) == tokens(http.native_result(replay, terminal_prompt, 1, 0))
    return {
        "anchor_rows": position,
        "anchor_gdn_bytes": anchor_bytes,
        "terminal_rows": len(terminal_prompt),
        "terminal_restore": delta,
        "anchor_ack_held_ns": upstream["reply_ns"],
        "peer_progress_ns": peer_ns,
        "writer_finished_ns": finished_ns,
        "anchor_ack_released_ns": released_ns,
    }


def failed_get(witness, source):
    prime, prompt = source[:HIGH], source[:PROMPT]
    witness.reset()
    expected = witness.generate("error-cold-reference", prompt, count=8)
    cold = http.native_result(expected, prompt, 8, 0)
    witness.reset()
    witness.stores("prime-error", prime)
    before = witness.snapshot("error-before")
    gate = witness.relay.arm(RANK_GET, GQA, drop=True)
    request = Stream(witness, "dropped-daemon-reply", prompt, 8)
    try:
        gate.wait()
    finally:
        gate.release.set()
    assert gate.finished.wait(10)
    got = http.native_result(request.finish(), prompt, 8, 0)
    after = witness.snapshot("error-after")
    assert not gate.entry["forwarded"] and tokens(got) == tokens(cold)
    assert after["io_errors"] > before["io_errors"]
    assert after["gqa_restored_tokens"] == before["gqa_restored_tokens"]
    witness.get("/health")
    _, reconnect = witness.changes("reconnected-get", prime, HIGH, {
        "gqa_restored_tokens": HIGH, "gdn_restored_tokens": HIGH, "io_errors": 0,
    })
    return {"dropped_reply_errors": after["io_errors"] - before["io_errors"],
            "reconnected": reconnect}




def matching_lease(witness, admission, status):
    return next((entry for entry in witness.relay.since()
                 if entry["opcode"] == LEASE and entry["status"] == status and
                 entry["incarnation"] == admission["incarnation"] and
                 entry["lease"] == admission["lease"]), None)


def abort_armed_get(witness, fence, source, unrelated):
    prime, prompt = source[:HIGH], source[:PROMPT]
    peer_prompt = unrelated[:128]
    witness.reset()
    witness.stores("prime-armed-abort", prime)
    peer = Stream(witness, "peer-during-armed-abort", peer_prompt, 512)
    _, peer_count = peer.progress()
    fence = arm_device_fence(fence)
    gate = witness.relay.arm(RANK_GET, GQA, phase="arm", drop=True)
    subject = Stream(witness, "abort-armed-get", prompt, 8)
    try:
        admission = gate.wait()
        wait_until((fence / "entered").exists, "GPU transfer did not reach the held stream fence")
        gate.release.set()
        assert gate.finished.wait(5)
        active = wait_until(lambda: matching_lease(witness, admission, 4),
                            "Native did not wait on the exact armed lease")
        status, _, raw = http.exchange(witness.port, "/abort_request",
                                       {"rid": subject.label}, timeout=5)
        assert status == 200 and not raw, (status, raw)
        abort_ns = perf_counter_ns()
        peer_ns, after_count = peer.progress(abort_ns)
        assert peer_ns > abort_ns > active["reply_ns"] and after_count > peer_count
        assert not (fence / "release").exists() and subject.thread.is_alive()
        assert matching_lease(witness, admission, 0) is None
    finally:
        gate.release.set()
        release_ns = perf_counter_ns()
        (fence / "release").touch()
    released = wait_until(lambda: matching_lease(witness, admission, 0),
                          "Native did not observe definitive lease release")
    terminal = subject.finish(release_ns)
    assert terminal["meta_info"]["finish_reason"]["type"] == "abort"
    assert terminal["output_ids"] == []
    replacement = Stream(witness, subject.label, prompt, 8, evidence_label="reused-armed-id")
    http.native_result(replacement.finish(), prompt, 8, HIGH)
    replacement_ns = perf_counter_ns()
    reused_peer_ns, _ = peer.progress(replacement_ns)
    assert reused_peer_ns > replacement_ns
    http.native_result(peer.finish(), peer_prompt, 512, 0)
    return {"armed_ns": admission["armed_ns"], "active_ns": active["reply_ns"],
            "abort_ns": abort_ns, "peer_progress_ns": peer_ns,
            "fence_released_ns": release_ns, "lease_released_ns": released["reply_ns"],
            "daemon_pid": admission["daemon_pid"], "lease": admission["lease"],
            "reuse_ns": replacement_ns, "peer_progress_after_reuse_ns": reused_peer_ns,
            "replacement_cached_tokens": HIGH}


def shutdown_armed_get(args, relay, relay_path, source):
    prefix, prompt = source[:HIGH], source[:PROMPT]
    with http.server_process(args, "armed-shutdown", "--lmcache-socket", str(relay_path),
                             "--max-active", "1") as (port, _, process):
        witness = Witness(port, args.evidence, args.daemon, relay)
        witness.reset()
        witness.stores("prime-armed-shutdown", prefix)
        begin = len(relay.since())
        gate = relay.arm(RANK_GET, GQA, drop=True)
        fence = arm_device_fence(args.device_fence_dir)
        subject = Stream(witness, "shutdown-armed-get", prompt, 8)
        try:
            wait_until((fence / "entered").exists, "Shutdown GET did not reach its GPU fence")
            process.terminate()
            active = wait_until(lambda: next((entry for entry in relay.since(begin)
                                if entry["opcode"] == LEASE and entry["status"] == 4), None),
                                "Stopping native server did not query the active lease")
            assert process.poll() is None and not gate.entered.is_set()
            (fence / "release").touch()
            admission = gate.wait()
            assert (active["incarnation"], active["lease"]) == (
                admission["incarnation"], admission["lease"])
            assert process.wait(timeout=10) == 0
            released = matching_lease(witness, admission, 0)
            assert released
            assert witness.action(READY) == (0, WIRE_VERSION)
            subject.thread.join(5)
            assert not subject.thread.is_alive()
            return {"active_ns": active["reply_ns"], "released_ns": released["reply_ns"],
                    "held_reply_ns": admission["reply_ns"], "exit_code": process.returncode}
        finally:
            (fence / "release").touch()
            gate.release.set()


def fresh_worker(args, relay, relay_path, source):
    prefix = source[:HIGH]
    with http.serve(args, "publisher", "--lmcache-socket", str(relay_path)) as (port, _):
        witness = Witness(port, args.evidence, args.daemon, relay)
        witness.reset()
        published, _ = witness.stores("cross-process-prime", prefix)
        http.native_result(published, prefix, 1, 0)
    with http.serve(args, "reader", "--lmcache-socket", str(relay_path)) as (port, _):
        witness = Witness(port, args.evidence, args.daemon, relay)
        hit, delta = witness.changes("fresh-worker-hit", source[:PROMPT], HIGH, {
            "gqa_restored_tokens": HIGH, "gdn_restored_tokens": HIGH,
            "draft_restored_rows": HIGH - 1, "io_errors": 0,
        }, count=8)
        witness.reset()
        cold = witness.generate("fresh-worker-cold", source[:PROMPT], count=8)
        assert tokens(hit) == tokens(http.native_result(cold, source[:PROMPT], 8, 0))
    return delta


def exercise(args, relay, path, source, unrelated):
    report, failures = {}, {}
    with http.serve(args, "functional", "--lmcache-socket", str(path)) as (port, _):
        witness = Witness(port, args.evidence, args.daemon, relay)
        info = json.loads(witness.get("/server_info"))
        assert info["lmcache_enabled"] is info["lmcache_available"] is True
        for label, check, inputs in (
            ("selective", selective, (source,)),
            ("natural_prefix", natural_prefix, (source, unrelated)),
            ("cache_salt", public_salt, (source,)),
            ("media_identity", media_identity, ()),
            ("unrelated_progress", held_get, (source, unrelated)),
            ("shifted_draft_1279", shifted_draft, (source, unrelated, LOW)),
            ("shifted_draft_8192", shifted_draft, (source, unrelated, 8192)),
            ("injected_get_errors", failed_get, (source,)),
            ("immutable_prefill_put", held_put, (source, unrelated)),
        ):
            capture_case(report, failures, label, check, witness, *inputs)
        capture_case(report, failures, "scored_same_history", scored_reference,
                     args, witness, source)
    with traced_worker(args, path) as (port, trace):
        one = Witness(port, args.evidence, args.daemon, relay)
        assert json.loads(one.get("/server_info"))["max_running_requests"] == 1
        capture_case(report, failures, "cancelled_lookup", abort_pending, one, source)
        capture_case(report, failures, "exact_terminal", exact_terminal,
                     one, source, unrelated, trace)
    if args.device_fence_dir:
        with http.serve(args, "armed-abort", "--lmcache-socket", str(path),
                        "--max-active", "2") as (port, _):
            two = Witness(port, args.evidence, args.daemon, relay)
            capture_case(report, failures, "cancelled_armed_get", abort_armed_get, two,
                         args.device_fence_dir, source, unrelated)
        capture_case(report, failures, "shutdown_armed_get", shutdown_armed_get,
                     args, relay, path, source)
    capture_case(report, failures, "fresh_worker", fresh_worker, args, relay, path, source)
    return report | {"failures": failures}


def terminal_send_order(trace: Path, owner: str, claimant: str, opened: int) -> dict[str, int]:
    _, owner_end = terminal_times(trace, owner)
    claimant_start, _ = terminal_times(trace, claimant)
    assert owner_end < opened, (owner, owner_end, opened)
    require_deferred_send(claimant, claimant_start, opened)
    return {
        "owner_send_end_unix_ns": owner_end,
        "barrier_open_unix_ns": opened,
        "claimant_send_start_unix_ns": claimant_start,
    }


def exact_terminal(
    witness: Witness, source: list[int], unrelated: list[int], trace: Path
) -> dict[str, object]:
    prompt_rows = 96
    prompt = source[:prompt_rows]
    endpoint = prompt_rows + 1
    owner_prompt = unrelated[:endpoint]
    assert len(prompt) == prompt_rows
    assert len(owner_prompt) == endpoint
    witness.reset()
    assert json.loads(witness.get("/server_info"))["max_running_requests"] == 1
    reference = witness.generate("boundary-length", prompt, count=2)
    ids, text, _ = http.native_result(reference, prompt, 2, 0)
    assert text, reference

    def prove_release_order(label: str) -> dict[str, int]:
        witness.reset()
        gate = witness.relay.arm(RANK_PUT, GDN, endpoint=len(owner_prompt) + 1)
        try:
            first = Stream(witness, f"{label}-held-owner", owner_prompt, 2, cache_salt="first")
            owner = gate.wait()
            http.native_result(first.finish(), owner_prompt, 2, 0)
            queued_prompt = unrelated[:98]
            queued = witness.generate(
                f"{label}-queued-owner", queued_prompt, count=2, cache_salt="second"
            )
            http.native_result(queued, queued_prompt, 2, 0)
            held_stores = witness.pending_stores()
            assert held_stores == TERMINAL_OWNER_STORES, held_stores
            begin = len(witness.relay.since())
            count, sampling = (2, {}) if label == "length" else (4, {"stop": [text]})
            request = Stream(witness, f"boundary-{label}-claim", prompt, count, **sampling)
            wait_until(
                lambda held_stores=held_stores: witness.pending_stores() == held_stores + 2,
                "Terminal checkpoint never deferred behind both owners",
            )
            status, _, raw = http.exchange(witness.port, "/abort_request", {"rid": request.label})
            assert status == HTTPStatus.OK
            assert not raw
            assert witness.pending_stores() == held_stores + 2
        finally:
            gate.release.set()
        assert gate.finished.wait(10)
        assert owner.get("forwarded") is True
        final = request.finish()
        timing = terminal_send_order(
            trace, first.label, request.label, owner["barrier_open_unix_ns"]
        )
        if label == "length":
            assert http.native_result(final, prompt, 2, 0)[0] == ids
        else:
            assert final["meta_info"]["finish_reason"] == {"type": "stop", "matched": text}
            assert final["output_ids"] == ids, final
            assert final["text"] == "", final
        witness.drain()
        endpoints = [
            record["endpoint"]
            for entry in witness.relay.since(begin)
            if entry["opcode"] == RANK_PUT and entry["status"] == 0
            for record in entry["records"]
            if record.get("endpoint") == endpoint
        ]
        assert endpoints == [endpoint], (label, endpoints)
        warm = witness.generate(f"boundary-{label}-warm", prompt + ids)
        http.native_result(warm, prompt + ids, 1, endpoint)
        return timing

    ordering = {label: prove_release_order(label) for label in ("length", "stop")}
    return {"endpoint": endpoint, "output_ids": ids, "matched_stop": text, "ordering": ordering}


def checkpoint_only(
    args: argparse.Namespace, relay: Relay, path: Path, source: list[int], unrelated: list[int]
) -> dict:
    report, failures = {}, {}
    with http.serve(args, "checkpoints", "--lmcache-socket", str(path)) as (port, _):
        witness = Witness(port, args.evidence, args.daemon, relay)
        capture_case(report, failures, "natural_prefix", natural_prefix, witness, source, unrelated)
        capture_case(
            report, failures, "immutable_prefill_put", held_put, witness, source, unrelated
        )
    return report | {"failures": failures}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("server", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("fixtures", type=Path)
    parser.add_argument("daemon", type=Path, help="Private LMCache 0.5.8 Unix socket")
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--checkpoint-only", action="store_true")
    parser.add_argument("--split-only", action="store_true")
    parser.add_argument(
        "--tp",
        type=int,
        choices=(1, 2),
        default=1,
        help="Tensor parallel width for the focused split cache cases",
    )
    parser.add_argument(
        "--alternate-checkpoint", type=Path, help="Compatible checkpoint with distinct weights"
    )
    parser.add_argument("--device-fence-dir", type=Path,
                        help="Private directory passed to install_device_fence in the daemon")
    args = parser.parse_args()
    assert not (args.split_only and args.checkpoint_only)
    assert args.tp == 1 or args.split_only
    assert args.split_only or not args.alternate_checkpoint
    args.evidence.mkdir(parents=True)
    assert control(args.daemon, NAMESPACE, READY) == (0, WIRE_VERSION)
    source, unrelated = fixture(args.fixtures)
    with TemporaryDirectory(prefix="sglang-cache-test.", dir="/tmp") as work:
        relay_path = Path(work) / "cache.sock"
        relay = Relay(relay_path, args.daemon)
        try:
            with relay:
                run = exercise
                if args.split_only:
                    run = split_only
                elif args.checkpoint_only:
                    run = checkpoint_only
                report = run(args, relay, relay_path, source, unrelated)
        finally:
            tape = {"events": event_receipt(relay), "errors": relay.errors}
            (args.evidence / "relay.json").write_text(
                json.dumps(tape, default=bytes.hex, indent=2) + "\n"
            )
        assert not relay.errors, relay.errors
    (args.evidence / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    assert not report["failures"], sorted(report["failures"])


if __name__ == "__main__":
    main()
