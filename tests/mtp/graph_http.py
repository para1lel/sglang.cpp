"""Exercise two-row target graph replay through the native HTTP batch scheduler."""

import argparse
import ctypes
import json
import os
from pathlib import Path
import re
import sys
import time
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sampled_http as http
from batch.run_native import Stream, request_body
from batch.check_graph_evidence import trace as graph_trace


POOL = (13, 198, 279, 826, 760, 6511, 314, 9338, 369, 1570, 315, 4950)
ROUND = re.compile(rb"^batch mtp slot=(\d+) drafted=(\d+) accepted=(\d+) "
                   rb"verified=(\d+) replayed=(\d+) boundary=(\d+)$", re.M)
RETIRED = b"batch width=0 decode=0 request_bytes=0\n"


def activity_clock(activity):
    get_timestamp = ctypes.CDLL(str(activity)).cuptiGetTimestamp
    get_timestamp.argtypes = [ctypes.POINTER(ctypes.c_uint64)]

    def timestamp():
        result = ctypes.c_uint64()
        http.require(get_timestamp(ctypes.byref(result)) == 0, "Cannot read CUPTI timestamp")
        return result.value

    return timestamp


def benchmark(length, output, offset=0):
    prompt = [POOL[(index + offset) % len(POOL)] for index in range(length)]
    return request_body({"prompt": prompt, "max_tokens": output})


def trace_slice(log, offset):
    with log.open("rb") as source:
        source.seek(offset)
        return source.read()


def wait_retired(log, offset):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if RETIRED in trace_slice(log, offset):
            return
        time.sleep(0.01)
    raise AssertionError("Disconnected K1 request did not retire")


def request(port, log, output, name, route, body):
    offset = log.stat().st_size
    frames = http.post(port, route, body)
    trace = trace_slice(log, offset)
    (output / f"{name}.trace").write_bytes(trace)
    http.write_json(output / f"{name}.json", {"request": body, "frames": frames})
    return http.transcript(frames), trace


def exercise(args, arm, graphs, clock=lambda: 0):
    output = args.evidence / arm
    flags = ("--batch-cuda-graphs",) if graphs else ("--eager",)
    cases = {
        "warm": (http.TEXT, benchmark(4, 4)),
        "cross-page": (http.TEXT, benchmark(127, 32)),
        "last-bound": (http.TEXT, benchmark(5118, 7, 1)),
        "straddles": (http.TEXT, benchmark(5119, 7, 2)),
        "outside": (http.TEXT, benchmark(5120, 7, 3)),
        "single-output": (http.TEXT, benchmark(127, 2, 2)),
        "sampled": (http.TEXT, http.MAIN),
        "eos": (http.CHAT, http.EOS),
        "selector": (http.TEXT, http.MASKED),
    }
    wire, traces, intervals = {}, {}, {}
    with http.server(args, output, 2, 1, flags) as (port, log):
        for name, (route, body) in cases.items():
            start = clock()
            wire[name], traces[name] = request(port, log, output, name, route, body)
            intervals[name] = (start, clock())
        stop = http.stop_case(wire["sampled"])
        start = clock()
        wire["stop"], traces["stop"] = request(port, log, output, "stop", http.TEXT, stop)
        intervals["stop"] = (start, clock())
        case = {"name": "cancel-and-replace", "prompt": benchmark(127, 1, 4)["prompt"],
                "max_tokens": 1024}
        start = clock()
        started = log.stat().st_size
        with Stream(port, case) as stream:
            stream.post()
            stream.open()
            for _ in range(5):
                stream.event()
            offset = log.stat().st_size
            stream.reset()
            wait_retired(log, offset)
            http.write_json(output / "cancel.json", stream.evidence())
        traces["cancel"] = trace_slice(log, started)
        intervals["cancel"] = (start, clock())
        (output / "cancel.trace").write_bytes(traces["cancel"])
        start = clock()
        wire["replacement"], traces["replacement"] = request(
            port, log, output, "replacement", http.TEXT, benchmark(128, 16, 4))
        intervals["replacement"] = (start, clock())
    return wire, traces, intervals


def check_rounds(wire, traces):
    rows = {name: [tuple(map(int, row[1:])) for row in ROUND.findall(traces[name])]
            for name in wire}
    greedy = rows["cross-page"]
    http.require(any(drafted == 1 and accepted == 1 and verified == 2 and not replayed
                     for drafted, accepted, verified, replayed, _ in greedy), greedy)
    http.require(any(drafted == 1 and accepted == 0 and verified == 2 and replayed == 1
                     for drafted, accepted, verified, replayed, _ in rows["sampled"]),
                 rows["sampled"])
    http.require(all(wire[name]["finish"] == "length" for name in
                     ("cross-page", "last-bound", "straddles", "outside", "single-output")),
                 "Expected output-cap exits")
    http.require(wire["stop"]["finish"] == wire["eos"]["finish"] == "stop",
                 (wire["stop"], wire["eos"]))
    http.require(wire["stop"]["tokens"] == wire["sampled"]["tokens"][:len(wire["stop"]["tokens"])]
                 and wire["eos"]["tokens"] == [9405, 248046], "Unexpected stopped tokens")
    http.require(rows["single-output"] == [(0, 0, 0, 0, 1)], rows["single-output"])
    http.require(all(any(row[0] == 1 and row[2] == 2 for row in rows[name])
                     for name in ("last-bound", "straddles", "outside", "sampled",
                                  "replacement")), rows)
    http.require(any(drafted == 1 and verified > 0
                     for drafted, _, verified, _, _ in rows["selector"]), rows["selector"])
    return rows


def check_replays(wire, traces, intervals, activity, graphs):
    completions, _ = graph_trace(activity)
    by_case = {name: [] for name in intervals}
    for line in activity.read_text().splitlines():
        if line.startswith("gpu\t"):
            _, _, graph_id, _, _, _, begin, end = line.split("\t")
            name, = (name for name, (start, stop) in intervals.items()
                     if start <= int(begin) <= int(end) <= stop)
            by_case[name].append(int(graph_id))
    http.require(sum(map(len, by_case.values())) == len(completions), by_case)
    positions, target_ids = {}, set()
    warmed = False
    for name, case_trace in traces.items():
        position = len(wire[name]["prompt"]) if name in wire else 127
        positions[name] = []
        graph_ids = by_case[name]
        if name == "selector":
            http.require(graphs or not graph_ids, (name, graph_ids))
            continue
        for _, drafted, _, verified, replayed, _ in ROUND.findall(case_trace):
            if int(drafted) == 1 and int(verified) == 2 and position + 2 <= 5120:
                if graphs and warmed:
                    positions[name].append(position)
                warmed = True
            position += int(replayed) or int(verified)
        http.require(len(graph_ids) == len(positions[name]),
                     f"{name}: expected K1 graph positions {positions[name]}, got {graph_ids}")
        target_ids.update(graph_ids)
    if graphs:
        http.require(len(target_ids) == 1 and target_ids.isdisjoint(by_case["selector"]), by_case)
        http.require(positions["cross-page"] and positions["cross-page"][0] == 127,
                     "Missing K1 target graph at the 127/128 page crossing")
        http.require(positions["last-bound"] == [5118], positions)
        for name in ("sampled", "stop", "eos", "cancel", "replacement"):
            http.require(positions[name], f"Missing K1 target graph in {name}")
        for name in ("straddles", "outside", "single-output", "selector"):
            http.require(not positions[name], (name, positions[name]))
    return positions


def observed_exercise(args, arm, graphs, clock):
    activity = args.evidence / f"{arm}.cupti.tsv"
    with patch.dict(os.environ, LD_PRELOAD=str(args.activity),
                    BATCH_GRAPH_ACTIVITY=str(activity)):
        wire, traces, intervals = exercise(args, arm, graphs, clock)
    http.write_json(args.evidence / f"{arm}.intervals.json", intervals)
    return wire, traces, intervals, activity


def check_capped_depth(args):
    records = {}
    graph_launches = {}
    for arm in ("eager", "graph"):
        output = args.evidence / f"k4-two-rows-{arm}"
        activity = output.with_suffix(".cupti.tsv")
        flags = ("--batch-cuda-graphs",) if arm == "graph" else ("--eager",)
        with patch.dict(os.environ, LD_PRELOAD=str(args.activity),
                        BATCH_GRAPH_ACTIVITY=str(activity)):
            with http.server(args, output, 1, 4, flags=flags, token_budget=2) as (port, log):
                request(port, log, output, "warm", http.TEXT, benchmark(4, 8))
                row, trace = request(port, log, output, "sampled", http.TEXT, http.MAIN)
        rounds = [tuple(map(int, value[1:])) for value in ROUND.findall(trace)]
        http.require(any(drafted == 1 and verified == 2
                         for drafted, _, verified, _, _ in rounds), (arm, rounds))
        http.require(all(verified <= 2 for _, _, verified, _, _ in rounds), (arm, rounds))
        completions, _ = graph_trace(activity)
        graph_launches[arm] = len(completions)
        records[arm] = row, rounds
    http.require(records["graph"] == records["eager"], records)
    http.require(graph_launches["graph"] > 0 and graph_launches["eager"] == 0, graph_launches)
    http.write_json(args.evidence / "k4-two-rows.json", {
        "completion_counts": graph_launches, "transcript_and_rounds": records,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=http.CHECKPOINT)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--activity", type=Path, required=True)
    args = parser.parse_args()
    args.evidence = args.evidence.resolve()
    args.activity = args.activity.resolve()
    args.server = args.server.resolve()
    args.evidence.mkdir()
    clock = activity_clock(args.activity)
    eager = observed_exercise(args, "eager", False, clock)
    graph = observed_exercise(args, "graph", True, clock)
    http.require(graph[0] == eager[0], "K1 graph altered HTTP output or usage")
    resets = [json.loads((args.evidence / arm / "cancel.json").read_text())
              for arm in ("eager", "graph")]
    http.require(all(row["reset_sent"] and len(row["events"]) == 5 for row in resets)
                 and resets[0]["events"] == resets[1]["events"],
                 "K1 graph changed the output before cancellation")
    http.require(check_rounds(*graph[:2]) == check_rounds(*eager[:2]),
                 "K1 graph changed speculative round accounting")
    check_replays(*eager, False)
    http.write_json(args.evidence / "replays.json", check_replays(*graph, True))
    check_capped_depth(args)
    print("K1 and capped K4 HTTP two-row graph checks passed", flush=True)


if __name__ == "__main__":
    main()
