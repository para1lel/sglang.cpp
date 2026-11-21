"""Check real graph activity, greedy scores, and release HTTP evidence."""

import argparse
from collections import Counter, defaultdict
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
BATCH_GRAPH_WIDTHS = (1, 2, 3, 4)
BATCH_MAX_SLOTS = 32
PACKED_BUCKETS = (2, 4, 8, 12, 16, 24, 32)
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


def lane_trace(path, require_replays=False, compare_request_bytes=True, packed=False):
    request_bytes = {"step": slice(2, 3), "admit": slice(-1, None), "cancel": slice(2, None)}
    normalized, masks, tokens = [], [], []
    step = None
    selected, graph, lanes, decoded, prefill = 0, 0, 0, 0, 0
    replayed_widths, warmed_buckets = set(), set()
    metadata_fields = {"admit": 5, "cancel": 4, "failed-closed": 0}

    def finish_step():
        nonlocal step
        if step is None:
            return
        width = decoded.bit_count()
        if packed and require_replays and width > 1 and not prefill:
            bucket = bucket_for(width, packed=True)
            expected = decoded if bucket in warmed_buckets else 0
            assert graph == expected, (
                path, step, f"replayed mask {graph}, expected {expected} for bucket {bucket}"
            )
            warmed_buckets.add(bucket)
        if graph or (require_replays and not packed and not prefill and width in replayed_widths):
            assert graph == decoded and not prefill, (
                path, step, f"replayed mask {graph}, expected decode mask {decoded}"
            )
            masks.append(graph)
            replayed_widths.add(width)
        step = None

    for row in path.read_text().splitlines():
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
        if not compare_request_bytes and kind in request_bytes:
            del values[request_bytes[kind]]
        normalized.append((kind, *values))
    finish_step()
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


def bucket_for(width, packed):
    if packed and width > 1:
        return next(bucket for bucket in PACKED_BUCKETS if bucket >= width)
    return width


def verify_buckets(device, runtime, masks, widths, packed=False):
    groups = defaultdict(set)
    bindings = defaultdict(set)
    for graph_id, mask in zip(device, masks, strict=True):
        width = mask.bit_count()
        groups[bucket_for(width, packed)].add(graph_id)
        bindings[width].add(mask)
    widths = set(widths)
    assert set(bindings) == widths, (
        "unexpected replay widths", sorted(set(bindings) - widths),
        "missing replay widths", sorted(widths - set(bindings))
    )
    expected = {bucket_for(width, packed) for width in widths}
    assert set(groups) == expected and all(len(ids) == 1 for ids in groups.values()), groups
    assert len(set.union(*groups.values())) == len(expected), groups
    lifecycle = {api: count(runtime, api) for api in LIFECYCLE}
    assert set(lifecycle.values()) == {len(expected)}, (groups, lifecycle)
    return bindings


def verify_packed_rows(stem, steps, enabled, bulk=True):
    trace_path = stem.with_suffix(".packed")
    if not enabled:
        assert not trace_path.exists(), trace_path
        return
    observed = iter(trace_path.read_text().splitlines())
    graph_masks = {}
    for line in stem.with_suffix(".lanes").read_text().splitlines():
        kind, *values = line.split()
        if kind == "step":
            graph_masks[int(values[0])] = int(values[2])
    positions, previous_tokens = {}, {}
    by_step = {}
    for row in steps:
        kind, *values = row
        if kind == "admit":
            positions[int(values[1])] = 0
        elif kind == "step":
            step = int(values[0])
            by_step[step] = []
        elif kind == "lane":
            slot, job, prefill, decode, token, _ = map(int, values)
            by_step[step].append((slot, positions[job], prefill, decode, token,
                                  previous_tokens.get(job)))
            positions[job] += prefill + decode
            if token >= 0:
                previous_tokens[job] = token
    coverage, mixed_scatter, bulk_calls = Counter(), 0, 0
    for step, lanes in by_step.items():
        decode_lanes = [(slot, position, previous) for slot, position, _, decode, _, previous
                        in lanes if decode]
        if len(decode_lanes) < 2:
            continue
        width = len(decode_lanes)
        bucket = bucket_for(width, True)
        row = next(observed, None)
        assert row is not None, (stem, step, f"missing packed evidence for width {width}")
        replay, rows, active, readback, *binding = map(int, row.split())
        assert (rows, active) == (bucket, width), (stem, step, row, bucket, width)
        expected_bulk = width if bulk and not any(lane[2] for lane in lanes) else 0
        assert readback == expected_bulk, (stem, step, "bulk winner rows", readback, expected_bulk)
        decode_mask = sum(1 << slot for slot, _, _ in decode_lanes)
        expected_replay = int(graph_masks[step] == decode_mask)
        assert replay == expected_replay, (stem, step, row, decode_mask)
        assert binding[:width] == [position for _, position, _ in decode_lanes], (stem, step, row)
        assert all(position < 0 for position in binding[width:rows]), (stem, step, row)
        assert binding[rows:rows + active] == [previous for _, _, previous in decode_lanes], (
            stem, step, "unexpected active tokens", row
        )
        row_for_slot = {slot: index for index, (slot, _, _) in enumerate(decode_lanes)}
        expected_rows = [row_for_slot.get(slot, -1) for slot, _, _, _, token, _ in lanes
                         if token >= 0]
        assert binding[rows + active:] == expected_rows, (
            stem, step, "unexpected score scatter", binding[rows + active:], expected_rows
        )
        mixed_scatter += -1 in expected_rows
        bulk_calls += bool(readback)
        coverage[(width, bucket, bool(replay))] += 1
    assert next(observed, None) is None, (stem, "unexpected packed submission")
    assert coverage, stem
    print(f"{stem.name}: checked {sum(coverage.values())} packed submissions, "
          f"{sum(width * calls for (width, _, _), calls in coverage.items())} active rows, "
          f"{sum((bucket - width) * calls for (width, bucket, _), calls in coverage.items())} "
          f"negative padding rows, {bulk_calls} bulk winner copies, "
          f"{mixed_scatter} mixed score submissions")
    return coverage, mixed_scatter


def verify_lanes(evidence, policy, packed=False):
    eager = evidence / f"{policy}-eager"
    graph = evidence / f"{policy}-graph"
    baseline, masks, tokens = lane_trace(eager.with_suffix(".lanes"))
    assert not masks and trace(eager.with_suffix(".cupti.tsv")) == ([], [])
    steps, masks, observed = lane_trace(graph.with_suffix(".lanes"), require_replays=True,
                                        packed=packed)
    assert tokens and observed
    assert [row for row in steps if row[0] == "step"][-1][-2] == "0"
    device, runtime = trace(graph.with_suffix(".cupti.tsv"))
    bindings = verify_buckets(device, runtime, masks, (2, 3, 4), packed)
    assert min(len(bindings[2]), len(bindings[3])) > 1
    if packed:
        _, eager_mixed = verify_packed_rows(eager, baseline, True)
        _, graph_mixed = verify_packed_rows(graph, steps, True)
        assert eager_mixed and graph_mixed, (policy, eager_mixed, graph_mixed)
    else:
        verify_packed_rows(eager, baseline, False)
        verify_packed_rows(graph, steps, False)
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
        assert Counter(mask.bit_count() for mask in masks)[4] == (role == "launch")
        assert set(mask.bit_count() for mask in masks) >= {2, 3}
        expected = {"begin": (2, 2), "scratch-oom": (2, 2), "launch": (3, 3)}
        assert releases == expected.get(role, (2, 3))
        assert len(device) == len(masks)
        if role == "scratch-oom":
            assert "released partial graph scratch bindings" in output


def verify_boundary(evidence):
    baseline, masks, tokens = lane_trace(evidence / "boundary-eager.lanes",
                                         compare_request_bytes=False)
    assert not masks and trace(evidence / "boundary-eager.cupti.tsv") == ([], [])
    scores = (evidence / "boundary-eager.scores").read_bytes()
    for role in ("graph", "score-oom"):
        stem = evidence / f"boundary-{role}"
        steps, masks, observed = lane_trace(stem.with_suffix(".lanes"),
                                           compare_request_bytes=False)
        assert steps == baseline and observed == tokens and tokens
        assert steps[-1] == ("step", steps[-1][1], "15", "0")
        verify_scores(stem.with_suffix(".scores"), tokens)
        assert stem.with_suffix(".scores").read_bytes() == scores
        device, runtime = trace(stem.with_suffix(".cupti.tsv"))
        buckets = defaultdict(set)
        for graph_id, mask in zip(device, masks, strict=True):
            buckets[mask.bit_count()].add(graph_id)
        assert set(buckets) == {2, 3} and all(len(ids) == 1 for ids in buckets.values())
        assert len(set.union(*buckets.values())) == 2
        assert all(count(runtime, api) == 2 for api in LIFECYCLE)
        output = stem.with_suffix(".log").read_text()
        assert output.count("inject: score-oom") == (role == "score-oom")
        assert output.count("Allocate batch scores for 5248 positions") == (role == "score-oom")
    print(f"Long batch graph boundary, score OOM and recycling checks passed ({len(tokens)} rows)")


def verify_packed_boundary(evidence, roles=("packed-boundary", "packed-oom")):
    expected = {"packed-boundary": [5] * 6 + [7] + [3] * 3 + [1] * 4,
                "packed-oom": [5] + [6] * 4}
    for role in roles:
        expected_masks = expected[role]
        stem = evidence / role
        eager = evidence / f"{role}-eager"
        baseline, eager_masks, eager_tokens = lane_trace(eager.with_suffix(".lanes"))
        assert not eager_masks and trace(eager.with_suffix(".cupti.tsv")) == ([], [])
        steps, masks, tokens = lane_trace(stem.with_suffix(".lanes"))
        assert tokens and eager_tokens, role
        assert masks == expected_masks, (role, masks)
        assert steps[-1] == ("step", steps[-1][1], "15", "0", "0")
        device, runtime = trace(stem.with_suffix(".cupti.tsv"))
        widths = {mask.bit_count() for mask in expected_masks}
        verify_buckets(device, runtime, masks, widths, role == "packed-boundary")
        log = stem.with_suffix(".log").read_text()
        if role == "packed-boundary":
            before, replay = log.split("packed steady replay begin\n")
            during, _ = replay.split("packed steady replay end\n")
            assert "cudaMalloc " in before and "cudaMalloc " not in during
        else:
            assert log.count("inject: packed-oom after allocating packed hidden rows") == 1
            assert log.count("released partial packed hidden rows") == 1
            assert log.count("packed decode unavailable") == 1
            assert (eager.with_suffix(".log").read_text()).count("inject: packed-oom") == 1
        verify_packed_rows(eager, baseline, role == "packed-boundary")
        verify_packed_rows(stem, steps, role == "packed-boundary")
        verify_scores(stem.with_suffix(".scores"), tokens)
        compare_scores(stem.with_suffix(".scores"), eager.with_suffix(".scores"), steps)
        assert (steps, tokens) == (baseline, eager_tokens), role
        print(f"{role} replay, scratch allocation, and score checks passed")


def verify_packed_buckets(evidence):
    eager = evidence / "packed-buckets-eager"
    graph = evidence / "packed-buckets-graph"
    baseline, masks, tokens = lane_trace(eager.with_suffix(".lanes"))
    assert not masks and trace(eager.with_suffix(".cupti.tsv")) == ([], [])
    steps, masks, observed = lane_trace(graph.with_suffix(".lanes"), require_replays=True,
                                        packed=True)
    assert tokens and observed
    device, runtime = trace(graph.with_suffix(".cupti.tsv"))
    widths = range(2, BATCH_MAX_SLOTS + 1)
    bindings = verify_buckets(device, runtime, masks, widths, packed=True)
    full_mask = (1 << BATCH_MAX_SLOTS) - 1
    for width in widths:
        low = (1 << width) - 1
        high = low << (BATCH_MAX_SLOTS - width)
        assert bindings[width] == {low, high}, (width, bindings[width], low, high)
    assert steps[-1] == ("step", steps[-1][1], str(full_mask), "0", "0")
    eager_coverage, _ = verify_packed_rows(eager, baseline, True)
    graph_coverage, _ = verify_packed_rows(graph, steps, True)
    assert {width for width, _, _ in eager_coverage} == set(widths), eager_coverage
    assert {width for width, _, replay in graph_coverage if replay} == set(widths), graph_coverage
    verify_scores(graph.with_suffix(".scores"), observed)
    compare_scores(graph.with_suffix(".scores"), eager.with_suffix(".scores"), steps)
    assert (baseline, tokens) == (steps, observed)
    print(f"All packed bucket widths passed complete BF16 score comparison ({len(tokens)} rows)")


def verify_packed_pin_release(evidence):
    stem = evidence / "packed-pin-release"
    log = stem.with_suffix(".log").read_text().splitlines()
    expected = ["inject: packed-pin-release", "released packed winner pin after rejected free"]
    assert [line for line in log if line in expected] == expected, (stem, log)
    steps, masks, tokens = lane_trace(stem.with_suffix(".lanes"))
    assert masks == [5] and len(tokens) == 6, (stem, masks, tokens)
    verify_packed_rows(stem, steps, True)
    verify_scores(stem.with_suffix(".scores"), tokens)
    print("Packed winner pin recovered after rejected release")


def verify_singleton(evidence):
    eager = evidence / "singleton-eager"
    baseline, masks, tokens = lane_trace(eager.with_suffix(".lanes"))
    assert not masks and trace(eager.with_suffix(".cupti.tsv")) == ([], [])
    scores = eager.with_suffix(".scores").read_bytes()
    far_mask = 1 << (BATCH_MAX_SLOTS - 1)
    pair = far_mask | 1
    expected = {"graph": [far_mask] * 5 + [pair, far_mask], "launch": [far_mask, pair]}
    for role, expected_masks in expected.items():
        stem = evidence / f"singleton-{role}"
        steps, masks, observed = lane_trace(stem.with_suffix(".lanes"))
        assert steps == baseline and observed == tokens and tokens
        assert masks == expected_masks, (stem, masks)
        verify_scores(stem.with_suffix(".scores"), tokens)
        assert stem.with_suffix(".scores").read_bytes() == scores
        device, runtime = trace(stem.with_suffix(".cupti.tsv"))
        singleton_ids = {graph_id for graph_id, mask in zip(device, masks, strict=True)
                         if mask == far_mask}
        assert len(singleton_ids) == 1 and device[masks.index(pair)] not in singleton_ids
        assert all(count(runtime, api) == 2 for api in LIFECYCLE)
        output = stem.with_suffix(".log").read_text()
        assert output.count("inject: launch") == (role == "launch")
        assert ("at width 1:" in output) == (role == "launch"), output
    print(f"Singleton warmup, sparse replay, recycling and fallback passed ({len(tokens)} rows)")


def verify_packed_setup_terminal(evidence):
    stem = evidence / "packed-setup-terminal"
    steps, masks, tokens = lane_trace(stem.with_suffix(".lanes"))
    assert not masks and len(tokens) == 2 and steps[-1] == ("failed-closed",), steps
    assert [row[1:3] for row in steps if row[0] == "step"] == [("1", "5")], steps
    verify_scores(stem.with_suffix(".scores"), tokens)
    verify_packed_rows(stem, steps, False)
    log = stem.with_suffix(".log").read_text()
    faults = re.findall(r"^inject: packed setup illegal address at (\d+)$", log, re.MULTILINE)
    assert len(faults) == 1 and "packed decode unavailable" not in log, log
    fault = int(faults[0])
    activity = stem.with_suffix(".cupti.tsv")
    device, runtime = trace(activity)
    launches = [line.split("\t") for line in activity.read_text().splitlines()
                if "LaunchKernel" in line]
    assert not device and not any(count(runtime, api) for api in LIFECYCLE), activity
    assert any(kind == "runtime" for kind, *_ in launches), activity
    assert all(int(end) <= fault for *_, end in launches), (activity, fault)
    print(f"Terminal packed setup failed closed after {len(launches)} prefill launch records")


def verify_packed_setup_ineligible(evidence):
    expected = {
        "prefill": [("0", "1", "0"), ("2", "1", "0")],
        "mixed": [("0", "1", "0"), ("0", "0", "1"), ("2", "1", "0")],
    }
    for role, expected_lanes in expected.items():
        control = evidence / f"packed-setup-{role}-control"
        armed = evidence / f"packed-setup-{role}-armed"
        baseline, baseline_masks, baseline_tokens = lane_trace(control.with_suffix(".lanes"))
        steps, masks, tokens = lane_trace(armed.with_suffix(".lanes"))
        assert (steps, masks, tokens) == (baseline, baseline_masks, baseline_tokens), armed
        lanes = [(row[1], row[3], row[4]) for row in steps if row[0] == "lane"]
        assert lanes == expected_lanes and len(tokens) == len(lanes) and not masks, armed
        for stem in (control, armed):
            assert trace(stem.with_suffix(".cupti.tsv")) == ([], []), stem
            assert "inject:" not in stem.with_suffix(".log").read_text(), stem
            verify_packed_rows(stem, steps, False)
        verify_scores(armed.with_suffix(".scores"), tokens)
        compare_scores(armed.with_suffix(".scores"), control.with_suffix(".scores"), steps)
        print(f"Ineligible packed setup {role} matched its healthy control ({len(tokens)} rows)")


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


def verify_http(evidence, policy, packed=False):
    eager, graph = (evidence / f"http-{policy}-{route}" for route in ("eager", "graph"))
    cohorts = [json.loads((path / "http.json").read_text()) for path in (eager, graph)]
    assert payloads(cohorts[0]) == payloads(cohorts[1])
    assert churn_responses(cohorts[0]) == churn_responses(cohorts[1])
    assert cohorts[0]["recovery"]["events"] == cohorts[1]["recovery"]["events"]
    assert trace(eager / "cupti.tsv") == ([], [])
    device, runtime = trace(graph / "cupti.tsv")
    log = (graph / "server.log").read_text()
    masks = [int(mask) for mask in re.findall(r"^batch graph_slots=(\d+)$", log, re.M)]
    verify_buckets(device, runtime, masks, BATCH_GRAPH_WIDTHS, packed)


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
    assert "Score overflow matches protected tiles at 32768 and 32769 visible tokens" in output
    assert "Paged decode matched FP64 and eager across 24 rebound layer-lanes" in output
    assert "Paged decode recovered from finite-input overflow and value cancellation" in output
    assert "MXFP8 GQA check passed" in output
    device, runtime = trace(evidence / "operators.cupti.tsv")
    runs = (
        ("exact decode", 8),
        ("cache rebinding", 4),
        ("GDN rebinding", 6),
        ("prefill causal boundary", 1),
        ("prefill overflow at 65 visible tokens", 1),
        ("prefill overflow at 32768 visible tokens", 1),
        ("prefill overflow at 32769 visible tokens", 1),
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
    assert len(flash_labels) == len(set(flash_labels)) == 36, flash_labels
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
    parser.add_argument("--layout", choices=("legacy", "packed"), default="packed")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--boundary-only", action="store_true")
    mode.add_argument("--packed-setup-only", action="store_true")
    mode.add_argument("--singleton-only", action="store_true")
    mode.add_argument("--k1-only", action="store_true")
    args = parser.parse_args()
    packed = args.layout == "packed"
    if args.check_slot_mask:
        check_slot_mask()
    output = args.evidence
    if output is None:
        if (not args.check_slot_mask or args.boundary_only or args.packed_setup_only or
                args.singleton_only or args.k1_only):
            parser.error("Pass an evidence directory or --check-slot-mask")
        return
    if args.boundary_only:
        verify_boundary(output)
        if packed:
            verify_packed_boundary(output)
        return
    if args.packed_setup_only:
        verify_packed_boundary(output, ("packed-oom",))
        verify_packed_pin_release(output)
        verify_packed_setup_terminal(output)
        verify_packed_setup_ineligible(output)
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

    def exact_faults(evidence):
        eager = evidence / "exact-eager"
        baseline, _, tokens = lane_trace(eager.with_suffix(".lanes"))
        verify_faults(evidence, baseline, tokens, eager.with_suffix(".scores"))

    check("operators", verify_operators)
    check("exact lanes", verify_lanes, "exact")
    check("exact faults", exact_faults)
    check("exact HTTP", verify_http, "exact")
    check("tensor-core lanes", verify_lanes, "tensor-core", packed)
    check("tensor-core HTTP", verify_http, "tensor-core", packed)
    check("legacy boundary", verify_boundary)
    if packed:
        check("packed boundary", verify_packed_boundary)
        check("packed pin release", verify_packed_pin_release)
        check("packed setup terminal", verify_packed_setup_terminal)
        check("packed setup eligibility", verify_packed_setup_ineligible)
        check("packed buckets", verify_packed_buckets)
    check("singleton", verify_singleton)
    check("K1", verify_k1)
    if failures:
        raise SystemExit(f"Batch CUDA Graph evidence failed in {', '.join(failures)}")
    print("Batch CUDA Graph operator, lane, fault and HTTP checks passed")


if __name__ == "__main__":
    main()
