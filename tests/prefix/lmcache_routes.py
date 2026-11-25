"""Untimed wire and CUDA-work assertions against real LMCache storage."""

import argparse
import base64
from contextlib import closing
from copy import deepcopy
import hashlib
from http.client import HTTPConnection
import json
from pathlib import Path
import queue
import struct
import sys
import traceback
from tempfile import TemporaryDirectory
from threading import Thread
from time import monotonic, perf_counter_ns, sleep
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "tests"), str(ROOT / "benchmarks" / "paired")]

from cold_cache import LMCACHE_DRAIN, LMCACHE_METRICS, PENDING_STORES, RUNNING, WAITING, metrics
from media import http_routes as http
from lmcache_relay import DRAFT, ERASE, GET, GDN, GQA, LEASE, LOOKUP, PREFIX, PUT, READY, RESET
from lmcache_relay import Relay, arm_device_fence, control, key_body


LOW, HIGH, PROMPT = 1279, 1391, 1536
NAMESPACE = "sglang"


def fixture(fixtures):
    from benchmarks.concurrent.reference import load_lanes

    lanes = load_lanes(fixtures, "W3")
    return [case["prompt_token_ids"] for _, _, case in lanes[:2]]


class Witness:
    def __init__(self, port, evidence, daemon, relay):
        self.port = port
        self.evidence = evidence
        self.daemon = daemon
        self.relay = relay

    def action(self, opcode, payload=b""):
        return control(self.daemon, NAMESPACE, opcode, payload)

    def get(self, route):
        status, _, raw = http.exchange(self.port, route)
        assert status == 200, (route, status, raw)
        return raw

    def pending_stores(self):
        return metrics(self.get("/metrics").decode(), LMCACHE_DRAIN)[PENDING_STORES]

    def drain(self):
        wait_until(lambda: self.pending_stores() == 0, "LMCache stores did not settle")

    def snapshot(self, label):
        self.drain()
        raw = self.get("/metrics")
        (self.evidence / f"{label}.prom").write_bytes(raw)
        summary = metrics(raw.decode(), "lmcache")
        return {name.removeprefix("sglang_cpp:lmcache_").removesuffix("_total"): summary[name]
                for name in LMCACHE_METRICS}

    def reset(self):
        self.drain()
        count = len(self.relay.since())
        status, _, body = http.exchange(self.port, "/reset_prefix_cache", {})
        assert status == 200 and json.loads(body) == {"success": True}, (status, body)
        resets = [entry for entry in self.relay.since(count) if entry["opcode"] == RESET]
        assert len(resets) == 1 and resets[0]["status"] == 0, resets

    def generate(self, label, prompt, *, count=1, **options):
        body = http.native_text(prompt, count) | options
        return http.capture(SimpleNamespace(evidence=self.evidence), label, self.port,
                            http.GENERATE, body)

    def stores(self, label, prompt):
        begin = len(self.relay.since())
        response = self.generate(label, prompt)
        assert len(response["output_ids"]) == 1
        keys = self.published_keys(begin)
        assert {family for family, _ in keys} == {GQA, GDN, DRAFT}, (label, keys)
        return response, keys

    def published_keys(self, begin):
        self.drain()
        entries = [entry for entry in self.relay.since(begin) if entry["opcode"] == PUT]
        assert all(entry["status"] == 0 for entry in entries), entries
        return {key for entry in entries for key in entry["keys"]}

    def changes(self, label, prompt, cached, expected, *, count=1, **options):
        before = self.snapshot(f"{label}-before")
        result = self.generate(label, prompt, count=count, **options)
        outputs = http.native_result(result, prompt, count, cached)
        after = self.snapshot(f"{label}-after")
        delta = {name: after[name] - before[name] for name in before}
        assert all(amount >= 0 for amount in delta.values()), delta
        for name, amount in expected.items():
            assert delta[name] == amount, (label, name, amount, delta)
        return outputs, delta

    def lookup(self, keys, expected):
        payload = struct.pack("!I", len(keys)) + b"".join(map(key_body, keys))
        assert self.action(LOOKUP, payload) == (0, bytes(expected)), (keys, expected)


def unique(keys, family):
    key, = (key for key in keys if key[0] == family)
    return key


def tokens(result):
    ids, text, _ = result
    return ids, text


def selective(witness, source):
    prefix = source[:PROMPT]
    witness.reset()
    cold, cold_delta = witness.changes("selective-cold", prefix, 0, {
        "gqa_restored_tokens": 0, "gqa_read_only_tokens": 0, "gdn_restored_tokens": 0,
        "gqa_computed_tokens": PROMPT, "gqa_written_tokens": PROMPT,
        "gdn_computed_tokens": PROMPT, "draft_restored_rows": 0, "io_errors": 0,
    })
    witness.reset()
    _, low_keys = witness.stores("prime-1279", prefix[:LOW])
    _, high_keys = witness.stores("prime-1391", prefix[:HIGH])
    low_gdn, high_gdn = unique(low_keys, GDN), unique(high_keys, GDN)
    assert low_gdn != high_gdn
    assert witness.action(ERASE, key_body(high_gdn))[0] == 0
    witness.lookup([low_gdn, high_gdn], (1, 0))
    offset = len(witness.relay.since())
    ahead, ahead_delta = witness.changes("gqa-ahead", prefix, LOW, {
        "gqa_restored_tokens": HIGH, "gqa_read_only_tokens": HIGH - LOW,
        "gqa_computed_tokens": PROMPT - HIGH, "gdn_restored_tokens": LOW,
        "gdn_computed_tokens": PROMPT - LOW, "draft_restored_rows": LOW - 1, "io_errors": 0,
    })
    assert tokens(ahead) == tokens(cold), (ahead, cold)
    restored = {key for entry in witness.relay.since(offset)
                if entry["opcode"] == GET and entry["status"] == 0 for key in entry["keys"]}
    assert low_gdn in restored and high_gdn not in restored, restored
    assert any(key in restored for key in high_keys - low_keys if key[0] == GQA), restored
    witness.reset()
    offset = len(witness.relay.since())
    _, common = witness.stores("prime-inverse", prefix[:HIGH])
    gqa_root = next(key for entry in witness.relay.since(offset)
                    if entry["opcode"] == PUT for key in entry["keys"] if key[0] == GQA)
    gdn = unique(common, GDN)
    assert witness.action(ERASE, key_body(gqa_root))[0] == 0
    witness.lookup([gqa_root, gdn], (0, 1))
    offset = len(witness.relay.since())
    # Four full 256-row records remain after erasing the first one. The old final partial
    # record has a distinct endpoint, so the recomputed turn publishes two full records.
    inverse, inverse_delta = witness.changes("gdn-ahead", prefix, 0, cold_delta | {
        "gqa_written_tokens": 2 * 256,
    })
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
    _, low_keys = witness.stores("natural-prime-1279", stock[:LOW])
    _, stock_keys = witness.stores("natural-prime-1416", stock)
    coherent, divergent = unique(low_keys, GDN), unique(stock_keys, GDN)
    witness.lookup([coherent, divergent], (1, 1))
    offset = len(witness.relay.since())
    full_rows = 1280
    warm, delta = witness.changes("natural-lcp-1391", prompt, LOW, {
        "gqa_restored_tokens": full_rows, "gqa_read_only_tokens": full_rows - LOW,
        "gqa_computed_tokens": PROMPT - full_rows, "gdn_restored_tokens": LOW,
        "gdn_computed_tokens": PROMPT - LOW, "draft_restored_rows": LOW - 1, "io_errors": 0,
    })
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
        assert keys == reads, (name, keys, reads)
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
    state = unique(witness.published_keys(offset), GDN)
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
    changed_state = unique(witness.published_keys(offset), GDN)
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
    return {key for entry in witness.relay.since(offset)
            if entry["opcode"] == GET and entry["status"] == 0
            for key in entry["keys"] if key[0] == family}


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
    _, primed = witness.stores(f"draft-{position}-hole-exact", prefix)
    gdn = unique(primed, GDN)
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
    # The 1278+2 history can change a near tie relative to ordinary cold prefill.
    # The separate matched-history probe compares all scores on this same fixture.
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


class Stream:
    def __init__(self, witness, label, prompt, count, *, evidence_label=None, **sampling):
        self.witness = witness
        self.label = label
        self.evidence_label = evidence_label or label
        self.body = http.native_text(prompt, count) | {"rid": label, "stream": True}
        self.body["sampling_params"].update(sampling)
        self.events = queue.Queue()
        self.thread = Thread(target=self.read)
        self.thread.start()

    def read(self):
        try:
            self.capture()
        except Exception as error:
            self.events.put(error)

    def capture(self):
        path = self.witness.evidence / f"{self.evidence_label}.sse"
        with path.open("wb") as raw, closing(HTTPConnection(
                "127.0.0.1", self.witness.port, timeout=90)) as session:
            session.request("POST", http.GENERATE, json.dumps(self.body), http.HEADERS)
            with session.getresponse() as response:
                assert response.status == 200, (response.status, response.read())
                assert response.headers.get_content_type() == "text/event-stream"
                for line in iter(response.readline, b""):
                    raw.write(line)
                    if line.startswith(b"data: "):
                        body = line[6:].strip()
                        value = "[DONE]" if body == b"[DONE]" else json.loads(body)
                        self.events.put((perf_counter_ns(), value))
        self.events.put(None)

    def event(self, timeout=10):
        record = self.events.get(timeout=timeout)
        if isinstance(record, Exception):
            raise record
        assert record is not None, (self.label, "unexpected EOF")
        return record

    def progress(self, after=0):
        deadline = monotonic() + 5
        while (remaining := deadline - monotonic()) > 0:
            instant, frame = self.event(remaining)
            if frame == "[DONE]":
                raise AssertionError((self.label, "completed before demonstrating progress"))
            assert "meta_info" in frame, frame
            if instant > after and frame["output_ids"] and not frame["meta_info"]["finish_reason"]:
                return instant, len(frame["output_ids"])
        raise TimeoutError(f"No stream progress from {self.label}")

    def finish(self, terminal_after=0):
        deadline = monotonic() + 90
        while True:
            instant, frame = self.event(max(0.001, deadline - monotonic()))
            if frame == "[DONE]" or frame["meta_info"]["finish_reason"]:
                assert instant > terminal_after, (
                    self.label, "terminal before the held lease was released")
            if frame == "[DONE]":
                break
        self.thread.join(5)
        assert not self.thread.is_alive() and self.events.get(timeout=1) is None
        raw = (self.witness.evidence / f"{self.evidence_label}.sse").read_bytes()
        frames = http.stream_frames(raw)
        last = frames[-1]
        assert all(last["output_ids"][:len(frame["output_ids"])] == frame["output_ids"]
                   and frame["meta_info"]["finish_reason"] is None for frame in frames[:-1])
        return last


def held_get(witness, source, unrelated):
    prime, prompt = source[:HIGH], source[:PROMPT]
    witness.reset()
    witness.stores("prime-held", prime)
    peer = Stream(witness, "unrelated-in-progress", unrelated[:128], 512)
    first_ns, first_count = peer.progress()
    gate = witness.relay.arm(GET, GQA)
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
    gate = witness.relay.arm(PREFIX)
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
        assert not any(entry["opcode"] in (GET, PUT) for entry in witness.relay.since(offset))
    finally:
        gate.release.set()
    assert gate.finished.wait(10)
    http.native_result(replacement.finish(), prompt, 8, HIGH)
    return {"lookup_held_ns": upstream["reply_ns"], "replacement_admitted_ns": admitted_ns,
            "replacement_cached_tokens": HIGH}


def held_put(witness, source, unrelated):
    position = http.CHECKPOINT_INTERVAL
    prompt, peer_prompt = source[:position] + [unrelated[128]], unrelated[:128]
    budget = 8
    assert len(prompt) == position + 1 and prompt[0] != peer_prompt[0]
    witness.reset()
    cold = witness.generate("held-put-cold", prompt, count=budget)
    expected = http.native_result(cold, prompt, budget, 0)
    witness.reset()
    offset = len(witness.relay.since())
    gate = witness.relay.arm(PUT, GDN)
    writer = Stream(witness, "held-periodic-put", prompt, budget)
    try:
        upstream = gate.wait()
        periodic, = (key for key in upstream["keys"] if key[0] == GDN)
        periodic_bytes = upstream["receipt"][GDN - 1]
        assert periodic_bytes > 0
        peer = Stream(witness, "peer-before-put-ack", peer_prompt, budget)
        peer_ns, _ = peer.progress(upstream["reply_ns"])
        witness.lookup([periodic], (1,))
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
    states = [key for entry in witness.relay.since(offset) if entry["opcode"] == PUT
              for key in entry["keys"] if key[0] == GDN]
    assert len(states) == len(set(states)) == 3 and states[0] == periodic, states
    witness.lookup(states, (1, 1, 1))
    terminal_prompt = prompt + written[0][:-1]
    read_offset = len(witness.relay.since())
    restored, delta = witness.changes("held-put-terminal-restore", terminal_prompt,
                                     len(terminal_prompt), {
        "gdn_restored_tokens": len(terminal_prompt),
        "draft_restored_rows": len(terminal_prompt) - 1, "io_errors": 0,
    })
    terminal, = read_keys(witness, read_offset, GDN)
    assert terminal != periodic and terminal in states
    witness.reset()
    replay = witness.generate("held-put-terminal-cold", terminal_prompt)
    assert tokens(restored) == tokens(http.native_result(replay, terminal_prompt, 1, 0))
    return {"periodic_rows": position, "periodic_gdn_bytes": periodic_bytes,
            "terminal_rows": len(terminal_prompt), "terminal_restore": delta,
            "periodic_ack_held_ns": upstream["reply_ns"], "peer_progress_ns": peer_ns,
            "writer_finished_ns": finished_ns,
            "periodic_ack_released_ns": released_ns}


def failed_get(witness, source):
    prime, prompt = source[:HIGH], source[:PROMPT]
    witness.reset()
    expected = witness.generate("error-cold-reference", prompt, count=8)
    cold = http.native_result(expected, prompt, 8, 0)
    witness.reset()
    witness.stores("prime-error", prime)
    before = witness.snapshot("error-before")
    gate = witness.relay.arm(GET, GQA, drop=True)
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


def wait_until(predicate, message):
    deadline = monotonic() + 5
    while monotonic() < deadline:
        result = predicate()
        if result:
            return result
        sleep(0.01)
    raise AssertionError(message)


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
    gate = witness.relay.arm(GET, GQA, phase="arm", drop=True)
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
        gate = relay.arm(GET, GQA, drop=True)
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
            assert released and witness.action(READY) == (0, b"0.5.8+native-v3")
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


def capture_case(report, failures, name, check, *args):
    try:
        report[name] = check(*args)
    except Exception:
        failures[name] = traceback.format_exc()
        print(failures[name], file=sys.stderr, flush=True)


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
                ("immutable_periodic_put", held_put, (source, unrelated))):
            capture_case(report, failures, label, check, witness, *inputs)
        capture_case(report, failures, "scored_same_history", scored_reference,
                     args, witness, source)
    with http.serve(args, "single-slot", "--lmcache-socket", str(path),
                    "--max-active", "1") as (port, _):
        one = Witness(port, args.evidence, args.daemon, relay)
        assert json.loads(one.get("/server_info"))["max_running_requests"] == 1
        capture_case(report, failures, "cancelled_lookup", abort_pending, one, source)
        capture_case(report, failures, "exact_terminal", exact_terminal, one, source, unrelated)
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


def exact_terminal(witness, source, unrelated):
    endpoint = http.CHECKPOINT_INTERVAL
    prompt = source[:endpoint]
    assert len(prompt) == endpoint
    witness.reset()
    assert json.loads(witness.get("/server_info"))["max_running_requests"] == 1
    reference = witness.generate("boundary-length", prompt)
    ids, text, _ = http.native_result(reference, prompt, 1, 0)
    assert text, reference
    for label in ("length", "stop"):
        witness.reset()
        gate = witness.relay.arm(PUT, GDN)
        try:
            first = Stream(witness, f"{label}-held-owner", unrelated[:97], 1)
            gate.wait()
            http.native_result(first.finish(), unrelated[:97], 1, 0)
            witness.generate(f"{label}-queued-owner", unrelated[:98])
            assert witness.pending_stores() == 2
            begin = len(witness.relay.since())
            count, sampling = (1, {}) if label == "length" else (4, {"stop": [text]})
            request = Stream(witness, f"boundary-{label}-claim", prompt, count, **sampling)
            wait_until(lambda: witness.pending_stores() == 3, "Terminal checkpoint never deferred")
            status, _, raw = http.exchange(witness.port, "/abort_request", {"rid": request.label})
            assert status == 200 and not raw
            assert request.thread.is_alive() and witness.pending_stores() == 3
        finally:
            release_ns = perf_counter_ns()
            gate.release.set()
        assert gate.finished.wait(10)
        final = request.finish(release_ns)
        if label == "length":
            assert http.native_result(final, prompt, 1, 0)[0] == ids
        else:
            assert final["meta_info"]["finish_reason"] == {"type": "stop", "matched": text}
            assert final["output_ids"] == ids and final["text"] == "", final
        witness.drain()
        endpoints = [record["endpoint"] for entry in witness.relay.since(begin)
                     if entry["opcode"] == PUT and entry["status"] == 0
                     for record in entry["records"] if record.get("endpoint") == endpoint]
        assert endpoints == [endpoint], (label, endpoints)
        warm = witness.generate(f"boundary-{label}-warm", prompt + ids)
        http.native_result(warm, prompt + ids, 1, endpoint)
    return {"endpoint": endpoint, "first_id": ids[0], "matched_stop": text}


def periodic_only(args, relay, path, source, unrelated):
    with http.serve(args, "held-put", "--lmcache-socket", str(path)) as (port, _):
        witness = Witness(port, args.evidence, args.daemon, relay)
        return {"immutable_periodic_put": held_put(witness, source, unrelated), "failures": {}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("server", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("fixtures", type=Path)
    parser.add_argument("daemon", type=Path, help="Private LMCache 0.5.8 Unix socket")
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--periodic-only", action="store_true")
    parser.add_argument("--device-fence-dir", type=Path,
                        help="Private directory passed to install_device_fence in the daemon")
    args = parser.parse_args()
    args.evidence.mkdir(parents=True)
    assert control(args.daemon, NAMESPACE, READY) == (0, b"0.5.8+native-v3")
    source, unrelated = fixture(args.fixtures)
    with TemporaryDirectory(prefix="sglang-cache-test.", dir="/tmp") as work:
        relay_path = Path(work) / "cache.sock"
        relay = Relay(relay_path, args.daemon)
        try:
            with relay:
                run = periodic_only if args.periodic_only else exercise
                report = run(args, relay, relay_path, source, unrelated)
        finally:
            tape = {"events": event_receipt(relay), "errors": relay.errors}
            (args.evidence / "relay.json").write_text(json.dumps(tape, indent=2) + "\n")
        assert not relay.errors, relay.errors
    (args.evidence / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    assert not report["failures"], sorted(report["failures"])


if __name__ == "__main__":
    main()
