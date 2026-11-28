"""Check W3 cache reuse, selective recovery, and isolation using a private real daemon."""

import argparse
import base64
import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks" / "paired"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lmcache_relay import GDN, RANK_GET, RANK_PUT, READY, WIRE_VERSION, Relay, control
from lmcache_route_helpers import (
    CACHE_PAGES,
    CHECKPOINT_SLOTS,
    NAMESPACE,
    Witness,
    backup_under_pressure,
    capture_case,
    transfer_records,
)
from lmcache_split_routes import (
    DEEP,
    MTP_STEPS,
    PROMPT,
    RANK_PAIR,
    SHALLOW,
    Coverage,
    Evidence,
    check_cpu_restore,
    complementary_ranks,
    failed_get,
    generated_terminal,
    repeat_resident,
    separate_families,
    worker_restart,
)

from benchmarks.concurrent.reference import load_lanes
from media import http_routes as http


def fixture(fixtures: Path) -> list[list[int]]:
    lanes = load_lanes(fixtures, "W3")
    return [case["prompt_token_ids"] for _, _, case in lanes[:2]]


def natural_prefix(witness: Witness, source: list[int], unrelated: list[int], tp: int) -> Evidence:
    changed = next(token for token in unrelated if token != source[DEEP])
    prompt = [*source[:DEEP], changed, *source[DEEP + 1 : PROMPT]]
    witness.reset()
    cold, _ = check_cpu_restore(witness, "fork-cold", prompt, tp, Coverage())
    witness.reset()
    for end, cached in ((DEEP, 0), (DEEP + 1, DEEP)):
        result = witness.generate(f"fork-prime-{end}", source[:end])
        http.native_result(result, source[:end], 1, cached)
    backed = backup_under_pressure(witness, "fork", unrelated)
    siblings = {
        key
        for _, key, record in transfer_records(backed, RANK_PUT, GDN)
        if record["endpoint"] == DEEP + 1
    }
    assert len(siblings) == tp, siblings
    witness.lookup(sorted(siblings), [1] * tp)
    offset = len(witness.relay.since())
    warm, restored = check_cpu_restore(
        witness, "fork-warm", prompt, tp, Coverage(DEEP, DEEP, DEEP - 1)
    )
    reads = {key for _, key, _ in transfer_records(witness.relay.since(offset), RANK_GET, GDN)}
    assert len(reads) == tp, reads
    assert reads.isdisjoint(siblings), reads
    assert warm == cold, (warm, cold)
    repeat_resident(witness, "fork-repeat", prompt, warm)
    return {"common_prompt_tokens": DEEP, "output_ids": warm[0], "restored": restored}


def public_salt(
    witness: Witness, source: list[int], unrelated: list[int], tp: int
) -> list[Evidence]:
    prompt = source[:SHALLOW]
    witness.reset()
    salts = ("tenant\0a", "tenant\0b")
    offset = len(witness.relay.since())
    cold = [
        check_cpu_restore(witness, f"tenant-{i}-cold", prompt, tp, Coverage(), cache_salt=salt)
        for i, salt in enumerate(salts)
    ]
    for i, (salt, (expected, _)) in enumerate(zip(salts, cold, strict=True)):
        repeat_resident(witness, f"tenant-{i}-resident", prompt, expected, cache_salt=salt)
    assert not any(entry["opcode"] == RANK_PUT for entry in witness.relay.since(offset))
    backup_under_pressure(witness, "tenant", unrelated)
    reports, states = [], []
    for i, (salt, (expected, _)) in enumerate(zip(salts, cold, strict=True)):
        offset = len(witness.relay.since())
        warm, restored = check_cpu_restore(
            witness,
            f"tenant-{i}-cpu",
            prompt,
            tp,
            Coverage(SHALLOW, SHALLOW, SHALLOW - 1),
            cache_salt=salt,
        )
        assert warm == expected, (i, warm, expected)
        reads = {key for _, key, _ in transfer_records(witness.relay.since(offset), RANK_GET, GDN)}
        assert len(reads) == tp, reads
        states.append(reads)
        reports.append(restored)
    assert states[0].isdisjoint(states[1]), states
    return reports


def media_identity(witness: Witness) -> Evidence:
    image = http.visual("image_url", 32, 64)
    different = deepcopy(image)
    different["image_url"]["url"] = (
        http.RGB8 + base64.b64encode(bytes((195, 30, 80)) * (32 * 64)).decode()
    )
    message = {"type": "text", "text": "Describe what is in this image."}
    bodies = [http.chat([part, message], 1) for part in (image, different)]

    def generate(label: str, body: dict[str, object]) -> dict[str, object]:
        return http.capture(
            SimpleNamespace(evidence=witness.evidence), label, witness.port, http.CHAT, body
        )

    witness.reset()
    changed_cold = generate("media-changed-cold", bodies[1])
    changed_ids, changed_usage = http.received(changed_cold, 1, 0)
    witness.reset()
    offset = len(witness.relay.since())
    original = generate("media-original-cold", bodies[0])
    original_ids, original_usage = http.received(original, 1, 0)
    prompt_tokens = original_usage["prompt_tokens"]
    assert changed_usage["prompt_tokens"] == prompt_tokens
    assert original["prompt_token_ids"] == changed_cold["prompt_token_ids"]
    same = generate("media-original-repeat", bodies[0])
    assert http.received(same, 1, prompt_tokens)[0] == original_ids
    changed = generate("media-changed-probe", bodies[1])
    cached = changed["usage"]["prompt_tokens_details"]["cached_tokens"]
    assert 0 <= cached < prompt_tokens, changed
    assert http.received(changed, 1, cached)[0] == changed_ids
    repeated = generate("media-changed-repeat", bodies[1])
    assert http.received(repeated, 1, prompt_tokens)[0] == changed_ids
    assert not any(entry["opcode"] in (RANK_GET, RANK_PUT) for entry in witness.relay.since(offset))
    return {
        "prompt_tokens": prompt_tokens,
        "same_media_cached_tokens": prompt_tokens,
        "changed_media_cached_tokens": cached,
        "same_wire_prompt_tokens": True,
    }


def event_receipt(relay: Relay) -> list[Evidence]:
    return [
        entry
        | {
            "keys": [
                [family, hashlib.sha256(identity).hexdigest()] for family, identity in entry["keys"]
            ]
        }
        for entry in relay.since()
    ]


def exercise(
    args: argparse.Namespace, relay: Relay, path: Path, source: list[int], unrelated: list[int]
) -> Evidence:
    report, failures = {}, {}
    options = (
        "--tp",
        str(args.tp),
        "--mtp-steps",
        str(MTP_STEPS),
        "--max-active",
        "2",
        "--max-num-batched-tokens",
        "4096",
        "--prefix-gqa-pages",
        str(CACHE_PAGES),
        "--prefix-gdn-slots",
        str(CHECKPOINT_SLOTS),
        "--lmcache-socket",
        str(path),
    )
    with http.serve(args, "functional", *options) as (port, _):
        witness = Witness(port, args.evidence, args.daemon, relay)
        info = json.loads(witness.get("/server_info"))
        assert info["lmcache_enabled"] is info["lmcache_available"] is True
        assert (info["tp_size"], info["mtp_steps"]) == (args.tp, MTP_STEPS), info
        cases = [
            ("natural_prefix", natural_prefix, (source, unrelated, args.tp)),
            ("cache_salt", public_salt, (source, unrelated, args.tp)),
            ("generated_terminal", generated_terminal, (source, args.tp)),
            ("separate_families", separate_families, (source, unrelated, args.tp)),
            ("injected_get_error", failed_get, (source, unrelated, args.tp)),
        ]
        if args.tp == RANK_PAIR:
            cases.append(("complementary_ranks", complementary_ranks, (source, unrelated)))
        else:
            cases.append(("media_identity", media_identity, ()))
        for label, check, inputs in cases:
            capture_case(report, failures, label, check, witness, *inputs)
    capture_case(
        report, failures, "worker_restart", worker_restart, args, relay, options, source, unrelated
    )
    return report | {"failures": failures}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("server", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("fixtures", type=Path)
    parser.add_argument("daemon", type=Path, help="Private LMCache 0.5.8 Unix socket")
    parser.add_argument("evidence", type=Path)
    parser.add_argument(
        "--tp",
        type=int,
        choices=(1, 2),
        default=1,
        help="Tensor-parallel width for live cache checks",
    )
    parser.add_argument(
        "--alternate-checkpoint", type=Path, help="Compatible checkpoint with distinct weights"
    )
    args = parser.parse_args()
    args.evidence.mkdir(parents=True)
    assert control(args.daemon, NAMESPACE, READY) == (0, WIRE_VERSION)
    source, unrelated = fixture(args.fixtures)
    with TemporaryDirectory(prefix="sglang-cache-test.", dir="/tmp") as work:
        relay_path = Path(work) / "cache.sock"
        relay = Relay(relay_path, args.daemon)
        try:
            with relay:
                report = exercise(args, relay, relay_path, source, unrelated)
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
