import argparse
import itertools
import json
import math
import random
import re
import subprocess
from pathlib import Path


def budget_oracle(remaining, selected, budget):
    live = [slot for slot, count in enumerate(remaining) if selected[slot] and count >= 0]
    pending = [slot for slot in live if remaining[slot]]
    rows = [int(slot in pending) for slot in range(4)]
    ranked = sorted(
        (depth, slot)
        for slot in pending
        for depth in range(2, min(128, remaining[slot]) + 1)
    )
    for _, slot in ranked[:max(0, budget - len(live))]:
        rows[slot] += 1
    return rows


def host_cases():
    directed = [
        ([0, -1, 257, 19], [1, 1, 1, 1], 5),
        ([0, -1, 257, 19], [1, 0, 1, 1], 17),
        ([1, 127, 128, 129], [1, 1, 1, 1], 512),
        ([2, 2, 2, 2], [1, 1, 1, 1], 7),
        ([5, 6, 7, 8], [0, 0, 0, 0], 7),
    ]
    rng = random.Random(70359)
    for width in range(2, 5):
        for budget in (0, 4, 5, 7, 17, 31, 127, 128, 129, 255, 511, 512):
            for bits in itertools.product((0, 1), repeat=4):
                for _ in range(5):
                    counts = rng.choices([-2, -1, 0, 1, 2, 3, 63, 127, 128, 129, 257], k=width)
                    directed.append((counts + [-1] * (4 - width), list(bits), budget))
    return directed


def check_planner(source, work):
    text = source.read_text()
    planner = re.search(r"^PrefillRows plan_prefill\([^\n]+\n.*?^\}", text, re.M | re.S)
    if planner is None:
        raise AssertionError("Cannot locate the production planner")
    work.mkdir(parents=True, exist_ok=True)
    probe = work / "plan.cpp"
    driver = r"""
#include <algorithm>
#include <array>
#include <iostream>
#include <optional>
constexpr int decoder_batch_max_lanes = 4;
constexpr int prefill_chunk_rows = 128;
using Selection = std::array<bool, decoder_batch_max_lanes>;
using PrefillRows = std::array<int, decoder_batch_max_lanes>;
struct Lane {
    bool request;
    int remaining;
    int remaining_prompt() const { return remaining; }
};
struct BatchStorage { std::array<std::optional<Lane>, decoder_batch_max_lanes> slots; };
PLANNER
int main() {
    for (int budget; std::cin >> budget;) {
        BatchStorage storage{};
        Selection selection{};
        for (auto& lane : storage.slots) {
            int remaining;
            std::cin >> remaining;
            if (remaining != -1) lane = Lane{remaining >= 0, std::max(0, remaining)};
        }
        for (auto& selected : selection) std::cin >> selected;
        for (auto rows : plan_prefill(storage, selection, budget)) std::cout << rows << ' ';
        std::cout << '\n';
    }
}
"""
    corpus = host_cases()
    data = "".join(
        " ".join(map(str, [budget, *counts, *selected])) + "\n"
        for counts, selected, budget in corpus
    )
    expected = [budget_oracle(*case) for case in corpus]
    (work / "cases.json").write_text(json.dumps(
        [{"remaining": c, "selected": s, "budget": b, "expected_rows": e}
         for (c, s, b), e in zip(corpus, expected, strict=True)]
    ) + "\n")
    results = {}
    for name, body in (
        ("real", planner.group()),
        ("skip_decode_budget", planner.group().replace(
            "--unused;", "if (lane->remaining_prompt()) --unused;", 1)),
    ):
        probe.write_text(driver.replace("PLANNER", body))
        binary = work / name
        subprocess.run(["g++", "-std=c++17", "-O2", str(probe), "-o", str(binary)], check=True)
        output = subprocess.run([str(binary)], input=data, capture_output=True, text=True,
                                check=True)
        actual = [list(map(int, line.split())) for line in output.stdout.splitlines()]
        pairs = zip(actual, expected, strict=True)
        mismatches = [index for index, (left, right) in enumerate(pairs)
                      if left != right]
        results[name] = {"mismatches": len(mismatches), "first": None}
        if mismatches:
            index = mismatches[0]
            results[name]["first"] = {"case": corpus[index], "actual": actual[index],
                                      "expected": expected[index]}
        if bool(mismatches) != (name != "real"):
            raise AssertionError(results)
    probe.write_text(driver.replace("PLANNER", planner.group()))
    return {"cases": len(corpus), "variants": results}


def isolated(sequence):
    result = []
    memory = 0.0
    for index, query in enumerate(sequence):
        keys = sequence[:index + 1]
        weights = [math.exp(query * key) for key in keys]
        attention = sum(key * weight for key, weight in zip(keys, weights)) / sum(weights)
        memory = 0.625 * memory + query * (1.0 - 0.125 * memory)
        result.append((attention, memory))
    return result


def lifecycle(*, global_mask=False, global_state=False, stale_reuse=False, advance_empty=False,
              include_future=False):
    streams = {
        "fast": [0.5, -0.75, 0.125, -0.375],
        "slow": [-0.5, 0.25, 0.75, -0.25, 0.375, 0.125, -0.625, 0.25],
        "old": [0.875, 0.5, -0.625],
        "replacement": [-0.75, 0.625, -0.25],
    }
    slow = streams["slow"]
    streams["slow_other_suffix"] = [*slow[:4], -0.875, 0.75, 0.875, -0.75]
    streams["alt_neighbor"] = [-0.875, 0.25, -0.625, 0.5]

    def replay(fast_name, slow_name):
        scripts = [
            [(0, fast_name, 0, 2), (2, "old", 0, 2), (3, slow_name, 0, 3)],
            [],
            [(3, slow_name, 3, 1), (0, fast_name, 2, 1), (2, "replacement", 0, 2)],
            [(0, fast_name, 3, 1), (3, slow_name, 4, 4), (2, "replacement", 2, 1)],
        ]
        cached = {}
        carried = {}
        visible = []
        outputs = {}
        for step, segments in enumerate(scripts):
            if step == 1 and advance_empty:
                carried[0] += 1.0
            for slot, name, start, count in segments:
                if start == 0 and not stale_reuse:
                    cached.pop(slot, None)
                    carried.pop(slot, None)
                prefix = cached.setdefault(slot, [])
                for offset in range(start, start + count):
                    token = streams[name][offset]
                    prefix.append(token)
                    visible.append(token)
                    scope = visible if global_mask else prefix
                    if include_future:
                        scope = [*scope, *streams[name][offset + 1:]]
                    weights = [math.exp(token * key) for key in scope]
                    attention = sum(x * y for x, y in zip(scope, weights)) / sum(weights)
                    owner = -1 if global_state else slot
                    previous = carried.get(owner, 0.0)
                    carried[owner] = 0.625 * previous + token * (1.0 - 0.125 * previous)
                    outputs[name, offset] = (attention, carried[owner])
        return outputs

    original = replay("fast", "slow")
    perturbed = replay("alt_neighbor", "slow_other_suffix")
    for actual in (original, perturbed):
        for (name, offset), result in actual.items():
            if result != isolated(streams[name])[offset]:
                raise AssertionError((name, offset, result, isolated(streams[name])[offset]))
    for offset in range(4):
        assert original["slow", offset] == perturbed["slow_other_suffix", offset]
    return len(original) + len(perturbed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("evidence", type=Path)
    args = parser.parse_args()
    result = {"planner": check_planner(args.source, args.evidence), "reference_rows": lifecycle()}
    for fault in ("global_mask", "global_state", "stale_reuse", "advance_empty", "include_future"):
        try:
            lifecycle(**{fault: True})
        except AssertionError as error:
            result[fault] = str(error)
        else:
            raise AssertionError(f"Mutation survived: {fault}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
