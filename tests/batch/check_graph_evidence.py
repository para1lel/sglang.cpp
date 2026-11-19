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


def lane_trace(path, require_replays=False, compare_request_bytes=True):
    request_bytes = {"step": slice(2, 3), "admit": slice(-1, None), "cancel": slice(2, None)}
    normalized, masks, tokens = [], [], []
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
        for token in expected:
            actual, size = struct.unpack("<ii", stream.read(8))
            assert actual == token and size == 248320, path
            row = stream.read(size * 2)
            values = (bits[0] for bits in struct.iter_unpack("<H", row))
            winner, score = -1, -32768
            for index, bits in enumerate(values):
                magnitude = bits & 0x7fff
                assert magnitude < 0x7f80, (path, token)
                rank = -magnitude if bits & 0x8000 else magnitude
                if rank > score:
                    winner, score = index, rank
            assert len(row) == size * 2 and winner == token, (path, token, winner)
        assert stream.read(1) == b"", path


def verify_lanes(evidence, policy):
    eager = evidence / f"{policy}-eager"
    graph = evidence / f"{policy}-graph"
    baseline, masks, tokens = lane_trace(eager.with_suffix(".lanes"))
    assert not masks and trace(eager.with_suffix(".cupti.tsv")) == ([], [])
    steps, masks, observed = lane_trace(graph.with_suffix(".lanes"), require_replays=True)
    assert baseline == steps and tokens == observed and tokens
    assert [row for row in steps if row[0] == "step"][-1][-2] == "0"
    verify_scores(graph.with_suffix(".scores"), tokens)
    assert graph.with_suffix(".scores").read_bytes() == eager.with_suffix(".scores").read_bytes()
    device, runtime = trace(graph.with_suffix(".cupti.tsv"))
    buckets, bindings = defaultdict(set), defaultdict(set)
    for graph_id, mask in zip(device, masks, strict=True):
        width = mask.bit_count()
        buckets[width].add(graph_id)
        bindings[width].add(mask)
    assert set(buckets) == {2, 3, 4}
    assert all(len(ids) == 1 for ids in buckets.values())
    assert len(set.union(*buckets.values())) == 3
    assert min(len(bindings[2]), len(bindings[3])) > 1
    assert all(count(runtime, api) == 3 for api in LIFECYCLE)
    return baseline, tokens, eager.with_suffix(".scores")


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
    grouped = defaultdict(set)
    for graph_id, mask in zip(device, masks, strict=True):
        grouped[mask.bit_count()].add(graph_id)
    assert set(grouped) == set(BATCH_GRAPH_WIDTHS)
    assert all(len(ids) == 1 for ids in grouped.values())
    assert len(set.union(*grouped.values())) == len(BATCH_GRAPH_WIDTHS)
    assert all(count(runtime, api) == len(BATCH_GRAPH_WIDTHS) for api in LIFECYCLE)


def churn_responses(data):
    churn = data["cancel-and-replace"]
    assert churn["cancel"]["reset_sent"] and len(churn["cancel"]["events"]) == 4
    return [row["events"] for row in (churn["cancel"], churn["replacement"], *churn["peers"])]


def verify_operators(evidence):
    output = (evidence / "operators.log").read_text()
    assert "Rebound cache, page table and position on four device graph replays" in output
    assert "Rebound GDN history and recurrent banks on six device graph replays" in output
    assert "Paged decode matched FP64 and eager across 24 rebound layer-lanes" in output
    assert "Paged decode recovered from finite-input overflow and value cancellation" in output
    assert "MXFP8 GQA check passed" in output
    device, runtime = trace(evidence / "operators.cupti.tsv")
    replays = [8, 4, 6, 1, 3, 3, 2, 2]
    assert list(Counter(device).values()) == replays
    assert all(count(runtime, api) == len(replays) for api in LIFECYCLE)


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
    mode.add_argument("--boundary-only", action="store_true")
    mode.add_argument("--singleton-only", action="store_true")
    args = parser.parse_args()
    if args.check_slot_mask:
        check_slot_mask()
    output = args.evidence
    if output is None:
        if not args.check_slot_mask or args.boundary_only or args.singleton_only:
            parser.error("Pass an evidence directory or --check-slot-mask")
        return
    if args.boundary_only:
        verify_boundary(output)
        return
    if args.singleton_only:
        verify_singleton(output)
        return
    verify_operators(output)
    for policy in ("exact", "tensor-core"):
        evidence = verify_lanes(output, policy)
        if policy == "exact":
            verify_faults(output, *evidence)
        verify_http(output, policy)
    verify_boundary(output)
    verify_singleton(output)
    print("Batch CUDA Graph operator, lane, fault and HTTP checks passed")


if __name__ == "__main__":
    main()
