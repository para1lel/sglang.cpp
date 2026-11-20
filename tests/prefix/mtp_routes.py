"""Check real draft restoration and verification through the completion routes."""

import argparse
import json
from pathlib import Path
import re
import sys

from grammar_routes import CHAT, TEXT, check, fixtures, grammar
from live_controls import CACHED, COMPUTED, DEVICE_BYTES, ENTRIES, PAGES, STORES, Controls

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mtp"))
from sampled_http import CHECKPOINT, server

DRAFT_BYTES = "sglang_cpp:prefix_cache_draft_device_bytes"
DRAFT_CACHED = "sglang_cpp:prefix_cache_draft_cached_rows_total"
DRAFT_COMPUTED = "sglang_cpp:prefix_cache_draft_prefill_rows_total"
DRAFT_MISSES = "sglang_cpp:prefix_cache_draft_misses_total"
ROUNDS = re.compile(rb"^batch mtp slot=\d+ drafted=(\d+) accepted=(\d+) verified=(\d+) "
                    rb"replayed=(\d+) boundary=(\d+)$", re.M)


def cases():
    original = fixtures()
    return {
        "text": (TEXT, original["exact"][1], (
            "The colors are red, orange, yellow, green, blue, and violet.",
            "Please write yes, no, and maybe on the next line.",
        )),
        "partial": (TEXT, original["partial"][1], (
            "Zebras and antelopes graze at different times every day.",
        )),
        "chat": (CHAT, original["chat"][1], (
            "Café serves fresh coffee and blackberry muffins every morning.",
        )),
    }


def traced(routes, log, name, case, hit, depth, *, stream):
    offset = log.stat().st_size
    route, body, choices = case
    request = route, body | {"ebnf": grammar(choices)}, choices
    salt = "text" if route == TEXT else "chat"
    transcript = check(routes, name, request, hit, salt, stream)
    with log.open("rb") as source:
        source.seek(offset)
        rounds = [tuple(map(int, values)) for values in ROUNDS.findall(source.read())]
    if depth:
        assert any(drafted and verified == drafted + 1 for drafted, _, verified, _, _ in rounds), (
            name, rounds)
        assert all(0 <= accepted <= drafted <= depth for drafted, accepted, _, _, _ in rounds)
    else:
        assert not rounds, (name, rounds)
    return transcript


def check_delta(before, after, hit, computed, depth, name):
    expected = {CACHED: hit, COMPUTED: computed, DRAFT_MISSES: 0,
                DRAFT_CACHED: hit - 1 if depth and hit else 0,
                DRAFT_COMPUTED: computed if depth and hit else max(0, computed - 1) * bool(depth)}
    actual = {key: int(after[key] - before[key]) for key in expected}
    assert actual == expected, (name, actual, expected)


def control(args, inputs):
    with server(args, args.output / "k0-cold", 2, 0, token_budget=128) as (port, log):
        routes = Controls(port, log.parent)
        return {name: traced(routes, log, name, case, 0, 0, stream=False)
                for name, case in inputs.items()}


def composed(args, depth, inputs, expected):
    flags = ("--prefix-cache-entries", "2", "--batch-cuda-graphs")
    with server(args, args.output / f"k{depth}-warm", 2, depth, flags=flags,
                token_budget=128) as (port, log):
        routes = Controls(port, log.parent)
        initial = routes.wait_for("initial", 0, 0)
        assert initial[ENTRIES] == initial[PAGES] == 0, initial
        assert initial[DRAFT_BYTES] > 0 and initial[DEVICE_BYTES] > initial[DRAFT_BYTES], initial
        for name, case in inputs.items():
            reference = expected[name]
            before = routes.wait_for(f"{name}-before", 0, 0)
            hit = 128 if name == "partial" else 0
            cold = traced(routes, log, f"{name}-first", case, hit, depth, stream=False)
            assert cold == reference, (name, cold, reference)
            after = routes.wait_for(f"{name}-after", 0, 0)
            check_delta(before, after, hit, reference["prompt"] - hit, depth, name)
            full = reference["prompt"] // 128 * 128
            warm = traced(routes, log, f"{name}-repeat", case, full, depth, stream=True)
            assert warm == reference, (name, warm, reference)
            repeated = routes.wait_for(f"{name}-repeated", 0, 0)
            check_delta(after, repeated, full, reference["prompt"] - full, depth, name)
            assert repeated[STORES] == after[STORES], (name, after, repeated)
        routes.reset("idle-reset")
        reset = routes.wait_for("reset", 0, 0)
        assert reset[ENTRIES] == reset[PAGES] == 0 and reset[DEVICE_BYTES] == initial[DEVICE_BYTES]
        cold = traced(routes, log, "after-reset", inputs["text"], 0, depth, stream=True)
        assert cold == expected["text"], cold


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    args = parser.parse_args()
    args.output.mkdir()
    inputs = cases()
    expected = control(args, inputs)
    assert expected["text"]["prompt"] == 128 and expected["partial"]["prompt"] == 129
    assert expected["text"]["ids"][0] != expected["partial"]["ids"][0]
    for depth in (1, 4):
        composed(args, depth, inputs, expected)
    (args.output / "reference.json").write_text(json.dumps(expected, indent=2) + "\n")
    print("Prefix MTP seeded text and chat, full and partial restores, and verification passed")


if __name__ == "__main__":
    main()
