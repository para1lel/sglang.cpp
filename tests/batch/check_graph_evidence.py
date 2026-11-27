"""Check real graph activity, greedy scores, and release HTTP evidence."""

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import struct
import tempfile


LIFECYCLE = (
    "cudaStreamBeginCapture", "cudaStreamEndCapture", "cudaGraphInstantiate",
    "cudaGraphExecDestroy", "cudaGraphDestroy",
)
RECOVERABLE = ("begin", "instantiate", "launch", "scratch-oom", "instantiate-oom")
BATCH_GRAPH_WIDTHS = (1,)
BATCH_MAX_SLOTS = 32
VOCAB_SIZE = 248320


def trace(path):
    device, runtime, launches = [], [], set()
    for line in path.read_text().splitlines():
        kind, *fields = line.split("\t")
        if kind == "gpu":
            corr, graph, gpu, context, stream, start, end = map(int, fields)
            assert corr > 0 and graph > 0 and 0 < start < end, line
            device.append((start, corr, graph, gpu, context, stream))
            continue
        assert kind in ("runtime", "driver"), line
        name, corr, status, start, end = fields
        assert int(status) == 0 and 0 < int(start) <= int(end), line
        if name.startswith(("cudaGraphLaunch", "cuGraphLaunch")):
            launches.add(int(corr))
        if kind == "runtime":
            runtime.append(name)
    completed = sorted(device)
    assert len(completed) == count(runtime, "cudaGraphLaunch"), path
    assert len({row[1] for row in completed} & launches) == len(completed), path
    assert not completed or len({row[3:] for row in completed}) == 1, path
    return [row[2] for row in completed], runtime


def count(api, prefix):
    return sum(name.startswith(prefix) for name in api)


def lane_trace(
    path: Path,
    *,
    require_replays: bool = False,
    mixed_replay_step: int | None = None,
    expected_replay_steps: list[int] | None = None,
) -> tuple[list[tuple[str, ...]], list[int], list[int]]:
    normalized, masks, tokens, replay_steps = [], [], [], []
    step = None
    selected, graph, lanes, decoded, prefill = 0, 0, 0, 0, 0
    replayed_widths = set()
    metadata_fields = {"admit": 5, "cancel": 4, "failed-closed": 0}

    def finish_step():
        nonlocal step
        if step is None:
            return
        width = decoded.bit_count()
        if graph or (require_replays and not prefill and width in replayed_widths):
            assert (graph, prefill) == (decoded, int(step == mixed_replay_step)), (
                path,
                step,
                f"replayed mask {graph}, expected decode mask {decoded}",
            )
            masks.append(graph)
            replay_steps.append(step)
            replayed_widths.add(width)
        step = None

    for row in path.read_text(encoding="utf-8").splitlines():
        kind, *values = row.split()
        if kind == "step":
            finish_step()
            assert len(values) == 5, (path, row)
            step, selected, graph = map(int, values[:3])
            values.pop(2)
            lanes, decoded, prefill = 0, 0, 0
        elif kind == "lane":
            slot, _, rows, decode, token, _ = map(int, values)
            assert step is not None and 0 <= slot < BATCH_MAX_SLOTS and decode in (0, 1), (
                path, row)
            slot_bit = 1 << slot
            assert selected & slot_bit and not lanes & slot_bit, (path, row)
            lanes |= slot_bit
            prefill += rows
            if decode:
                assert rows == 0 and token >= 0, (path, step, row)
                decoded |= slot_bit
            if token >= 0:
                tokens.append(token)
        else:
            finish_step()
            assert kind in metadata_fields and len(values) == metadata_fields[kind], (path, row)
        normalized.append((kind, *values))
    finish_step()
    if expected_replay_steps is not None:
        assert replay_steps == expected_replay_steps, (path, replay_steps)
    return normalized, masks, tokens


def verify_scores(path, expected):
    with path.open("rb") as stream:
        for index, token in enumerate(expected):
            header = stream.read(8)
            assert len(header) == 8, (path, f"missing score row {index} for token {token}")
            actual, size = struct.unpack("<ii", header)
            assert (actual, size) == (token, VOCAB_SIZE), (path, index, token, actual, size)
            row = stream.read(size * 2)
            assert len(row) == size * 2, (path, f"truncated score row {index}", len(row))
            values = (bits[0] for bits in struct.iter_unpack("<H", row))
            winner, score = -1, -32768
            for vocab_index, bits in enumerate(values):
                magnitude = bits & 0x7fff
                assert magnitude < 0x7f80, (path, index, token, vocab_index)
                rank = -magnitude if bits & 0x8000 else magnitude
                if rank > score:
                    winner, score = vocab_index, rank
            assert winner == token, (path, index, token, winner)
        assert stream.read(1) == b"", path


def compare_scores(actual, expected, steps):
    labels = []
    for row in steps:
        if row[0] == "step":
            step = int(row[1])
        elif row[0] == "lane" and int(row[-2]) >= 0:
            labels.append((step, int(row[1])))
    mismatched = 0
    first = None
    row_size = 8 + VOCAB_SIZE * 2
    with actual.open("rb") as graph, expected.open("rb") as eager:
        for step, slot in labels:
            graph_row, eager_row = graph.read(row_size), eager.read(row_size)
            assert len(graph_row) == len(eager_row) == row_size, (actual, step, slot)
            if graph_row == eager_row:
                continue
            mismatched += 1
            if first is None:
                if graph_row[:8] != eager_row[:8]:
                    first = (f"step {step} slot {slot}: graph header "
                             f"{struct.unpack('<ii', graph_row[:8])}, eager header "
                             f"{struct.unpack('<ii', eager_row[:8])}")
                else:
                    graph_bits = struct.iter_unpack("<H", memoryview(graph_row)[8:])
                    eager_bits = struct.iter_unpack("<H", memoryview(eager_row)[8:])
                    index, observed, reference = next(
                        (index, observed[0], reference[0])
                        for index, (observed, reference) in enumerate(zip(graph_bits, eager_bits))
                        if observed != reference
                    )
                    first = (f"step {step} slot {slot} vocab index {index}: "
                             f"graph 0x{observed:04x}, eager 0x{reference:04x}")
        assert graph.read(1) == eager.read(1) == b"", (actual, expected)
    assert not mismatched, (actual, f"{mismatched}/{len(labels)} BF16 rows differ", first)


def verify_singleton_binding(device, runtime, masks):
    widths = [mask.bit_count() for _, mask in zip(device, masks, strict=True)]
    assert widths and set(widths) == {1} and len(set(device)) == 1, (device, masks)
    assert all(count(runtime, api) == 1 for api in LIFECYCLE), runtime


def verify_lanes(evidence, policy):
    eager = evidence / f"{policy}-eager"
    graph = evidence / f"{policy}-graph"
    baseline, masks, tokens = lane_trace(eager.with_suffix(".lanes"))
    assert not masks and trace(eager.with_suffix(".cupti.tsv")) == ([], [])
    steps, masks, observed = lane_trace(graph.with_suffix(".lanes"), require_replays=True)
    assert tokens and observed
    assert [row for row in steps if row[0] == "step"][-1][-2] == "0"
    device, runtime = trace(graph.with_suffix(".cupti.tsv"))
    verify_singleton_binding(device, runtime, masks)
    verify_scores(graph.with_suffix(".scores"), observed)
    compare_scores(graph.with_suffix(".scores"), eager.with_suffix(".scores"), steps)
    assert baseline == steps and tokens == observed


def verify_faults(evidence, baseline, expected_tokens, expected_scores):
    for role in (*RECOVERABLE, "hard"):
        stem = evidence / f"fault-{role}"
        output = stem.with_suffix(".log").read_text()
        assert output.count(f"inject: {role}") == 1, output
        steps, masks, tokens = lane_trace(stem.with_suffix(".lanes"))
        device, runtime = trace(stem.with_suffix(".cupti.tsv"))
        releases = count(runtime, "cudaGraphExecDestroy"), count(runtime, "cudaGraphDestroy")
        if role == "hard":
            assert steps[-1] == ("failed-closed",) and len(device) == 1
            assert releases == (1, 1)
            continue
        assert steps == baseline and tokens == expected_tokens
        assert expected_scores.read_bytes() == stem.with_suffix(".scores").read_bytes()
        assert all(mask.bit_count() == 1 for mask in masks)
        assert bool(masks) == (role == "launch")
        expected = {"begin": (0, 0), "scratch-oom": (0, 0), "launch": (1, 1)}
        assert releases == expected.get(role, (0, 1))
        assert len(device) == len(masks)
        if role == "scratch-oom":
            assert "released partial graph scratch bindings" in output


def verify_singleton(evidence):
    eager = evidence / "singleton-eager"
    baseline, masks, tokens = lane_trace(eager.with_suffix(".lanes"))
    assert not masks and trace(eager.with_suffix(".cupti.tsv")) == ([], [])
    scores = eager.with_suffix(".scores").read_bytes()
    far_mask = 1 << (BATCH_MAX_SLOTS - 1)
    expected = {"graph": [3, 4, 5, 7, 8, 9, 13], "launch": [3]}
    for role, replay_steps in expected.items():
        stem = evidence / f"singleton-{role}"
        steps, masks, observed = lane_trace(
            stem.with_suffix(".lanes"),
            mixed_replay_step=4 if role == "graph" else None,
            expected_replay_steps=replay_steps,
        )
        assert steps == baseline and observed == tokens and tokens
        assert masks == [far_mask] * len(replay_steps), (stem, masks)
        verify_scores(stem.with_suffix(".scores"), tokens)
        assert stem.with_suffix(".scores").read_bytes() == scores
        device, runtime = trace(stem.with_suffix(".cupti.tsv"))
        singleton_ids = {graph_id for graph_id, mask in zip(device, masks, strict=True)
                         if mask == far_mask}
        assert len(singleton_ids) == 1
        assert all(count(runtime, api) == 1 for api in LIFECYCLE)
        output = stem.with_suffix(".log").read_text()
        assert output.count("inject: launch") == (role == "launch")
        assert ("at width 1:" in output) == (role == "launch"), output
    print(f"Singleton warmup, sparse replay, recycling and fallback passed ({len(tokens)} rows)")


def verify_k1(evidence):
    def records(role):
        stem = evidence / f"k1-{role}"
        lines = stem.with_suffix(".lanes").read_text().splitlines()
        masks = [tuple(map(int, line.split()[3:])) for line in lines
                 if line.startswith("verify graphs ")]
        device, _ = trace(stem.with_suffix(".cupti.tsv"))
        log = stem.with_suffix(".log").read_text()
        return [line for line in lines if not line.startswith("verify graphs ")], masks, device, log

    eager, masks, device, _ = records("eager")
    assert masks and not any(k0 or k1 for k0, k1 in masks) and not device
    closed = {"launch": 1, "submitted": 2, "hard": 1, "hard-drain": 1,
              "cleanup": 0, "exec-cleanup": 1}
    recovered = {"upload": 2}
    fallbacks = ("prepare", "begin", "instantiate")
    for role in ("graph", *fallbacks, *closed, *recovered):
        steps, masks, device, log = records(role)
        if role in closed or role in recovered:
            terminal = "verify failed-closed" if role in closed else "verify recovered"
            assert steps[-1] == terminal
            assert steps[:-1] == eager[:len(steps) - 1]
            assert len(device) == (closed | recovered)[role] and len(set(device)) <= 1
            assert "using eager K1 target verification" not in log
        else:
            assert steps == eager
            submitted = [(k0, k1) for k0, k1 in masks if k0 or k1]
            identities = {depth: {graph_id for (k0, k1), graph_id
                                  in zip(submitted, device, strict=True)
                                  if (k0, k1)[depth]} for depth in (0, 1)}
            assert len(identities[0]) == 1 and identities[0].isdisjoint(identities[1])
            assert len(identities[1]) == (role == "graph")
            assert all(not (k0 and k1) for k0, k1 in submitted)
            if role == "graph":
                assert len(submitted) == 4
            if role in fallbacks:
                assert "using eager K1 target verification" in log
        assert log.count(f"inject: {role}\n") == (role != "graph")
        assert log.count("inject: terminal drain failure\n") == (role == "hard-drain")
    print("Sparse K1 replay, independent K0, and graph recovery checks passed")


def payloads(data):
    return [
        [stream["events"], stream["tokens"], stream["finish"], stream["usage"]]
        for cohort in data.values() if "streams" in cohort
        for stream in cohort["streams"]
    ]


def verify_http(evidence, policy):
    eager, graph = (evidence / f"http-{policy}-{route}" for route in ("eager", "graph"))
    cohorts = [json.loads((path / "http.json").read_text()) for path in (eager, graph)]
    assert payloads(cohorts[0]) == payloads(cohorts[1])
    assert churn_responses(cohorts[0]) == churn_responses(cohorts[1])
    assert cohorts[0]["recovery"]["events"] == cohorts[1]["recovery"]["events"]
    assert trace(eager / "cupti.tsv") == ([], [])
    device, runtime = trace(graph / "cupti.tsv")
    log = (graph / "server.log").read_text()
    masks = [int(mask) for mask in re.findall(r"^batch graph_slots=(\d+)$", log, re.M)]
    verify_singleton_binding(device, runtime, masks)


def churn_responses(data):
    churn = data["cancel-and-replace"]
    assert churn["cancel"]["reset_sent"] and len(churn["cancel"]["events"]) == 4
    return [row["events"] for row in (churn["cancel"], churn["replacement"], *churn["peers"])]


def verify_operators(evidence):
    failure = evidence / "operators.failure"
    assert not failure.exists(), failure.read_text()
    output = (evidence / "operators.log").read_text()
    assert "Rebound cache, page table and position on four device graph replays" in output
    assert "Rebound GDN history and recurrent banks on six device graph replays" in output
    overflow_cases = ((65, 33), (32768, 17), (32769, 2), (131072, 2), (131073, 2))
    for visible, width in overflow_cases:
        receipt = f"Prefill overflow visible={visible} grid={width}"
        assert f"{receipt} matched unaligned and FP64 reference" in output
    assert "Paged decode matched FP64 and eager across 24 rebound layer-lanes" in output
    assert "Paged decode recovered from finite-input overflow and value cancellation" in output
    assert "MXFP8 GQA check passed" in output
    device, runtime = trace(evidence / "operators.cupti.tsv")
    runs = (
        ("exact decode", 8),
        ("cache rebinding", 4),
        ("GDN rebinding", 6),
        ("prefill causal boundary", 1),
        *((f"prefill overflow at {visible} visible tokens", 1) for visible, _ in overflow_cases),
        ("paged decode rebinding", 3),
        ("paged decode overflow and cancellation", 3),
        ("ordered finite-range decode", 1),
        ("split finite-range decode", 1),
        ("ordered finite-range decode", 1),
        ("split finite-range decode", 1),
    )
    expected = [name for name, launches in runs for _ in range(launches)]
    previous, flash = device[:len(expected)], device[len(expected):]
    names = list(dict.fromkeys(expected))
    graph_ids = list(dict.fromkeys(previous))
    assert len(graph_ids) == len(names), (names, Counter(device))
    labels = dict(zip(graph_ids, names, strict=True))
    assert [labels[graph] for graph in previous] == expected, (runs, labels, device)
    flash_labels = re.findall(r"^MXFP8 prefill (.+) launch \S+ grid=", output, re.M)
    flash_repeat = re.findall(r"^MXFP8 prefill (.+) repeated graph matched ", output, re.M)
    required = {
        "TMEM first aligned start", "early tile capture with scratch",
        "staged tile capture with scratch", "paired scratch raw reference",
        "aligned causal tail", "aligned exact", "131072 visible", "131073 visible",
    }
    assert (
        len(flash_labels) == len(set(flash_labels)) and required <= set(flash_labels)
    ), flash_labels
    assert set(flash_repeat) <= set(flash_labels), flash_repeat
    flash_ids = list(dict.fromkeys(flash))
    assert len(flash_ids) == len(flash_labels) and set(flash_ids).isdisjoint(graph_ids)
    expected_flash = [graph for graph, name in zip(flash_ids, flash_labels, strict=True)
                      for _ in range(1 + flash_repeat.count(name))]
    assert flash == expected_flash, (flash, expected_flash)
    executable_count = len(names) + len(flash_labels)
    for api in LIFECYCLE:
        expected_count = executable_count + int(api not in
                                               ("cudaGraphInstantiate", "cudaGraphExecDestroy"))
        assert count(runtime, api) == expected_count, (api, expected_count)


def check_slot_mask():
    with tempfile.TemporaryDirectory() as directory:
        lanes = Path(directory) / "lanes"
        target = ("admit 0 1 5 8 100\nadmit 2 3 5 8 200\n"
                  "step 0 5 5 200 0\nlane 0 1 0 1 21 -1\nlane 2 3 0 1 22 -1\n"
                  "cancel 2 3 200 100\nadmit 3 4 5 8 200\nadmit 1 5 5 0 300\n"
                  "step 2 11 {mask} 0 0\nlane 0 1 0 1 23 -1\n"
                  "lane 1 5 0 0 -1 0\nlane 3 4 0 1 25 -1\n"
                  "admit 1 6 5 8 300\n"
                  "step 3 9 9 0 0\nlane 0 1 0 1 27 -1\nlane 3 4 0 1 29 -1\n"
                  "cancel 1 6 300 200\n")
        lanes.write_text(target.format(mask=9))
        _, masks, tokens = lane_trace(lanes, require_replays=True)
        assert masks == [5, 9, 9] and tokens == [21, 22, 23, 25, 27, 29]
        for bad_mask in (3, 0):
            lanes.write_text(target.format(mask=bad_mask))
            try:
                lane_trace(lanes, require_replays=True)
            except AssertionError as error:
                assert f"replayed mask {bad_mask}, expected decode mask 9" in str(error)
            else:
                raise AssertionError(f"Invalid replay mask accepted: {bad_mask}")
    print("Graph slot mask check accepted interleaved metadata and mask 9, rejected 3 and 0")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path, nargs="?")
    parser.add_argument("--check-slot-mask", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--singleton-only", action="store_true")
    mode.add_argument("--k1-only", action="store_true")
    args = parser.parse_args()
    if args.check_slot_mask:
        check_slot_mask()
    output = args.evidence
    if output is None:
        if not args.check_slot_mask or args.singleton_only or args.k1_only:
            parser.error("Pass an evidence directory or --check-slot-mask")
        return
    if args.singleton_only:
        verify_singleton(output)
        return
    if args.k1_only:
        verify_k1(output)
        return
    failures = []

    def check(label, verify, *params):
        try:
            verify(output, *params)
        except (AssertionError, OSError, ValueError) as error:
            failures.append(label)
            print(f"FAIL {label}: {error!r}")
        else:
            print(f"PASS {label}")

    def default_faults(evidence):
        eager = evidence / "default-eager"
        baseline, _, tokens = lane_trace(eager.with_suffix(".lanes"))
        verify_faults(evidence, baseline, tokens, eager.with_suffix(".scores"))

    check("operators", verify_operators)
    check("default lanes", verify_lanes, "default")
    check("default faults", default_faults)
    check("default HTTP", verify_http, "default")
    check("singleton", verify_singleton)
    check("K1", verify_k1)
    if failures:
        raise SystemExit(f"Batch CUDA Graph evidence failed in {', '.join(failures)}")
    print("Batch CUDA Graph operator, lane, fault and HTTP checks passed")


if __name__ == "__main__":
    main()
