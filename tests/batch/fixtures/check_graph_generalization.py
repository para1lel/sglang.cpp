import argparse
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import TypedDict

REPLAY_ATTENTION = re.compile(
    r"attention|gqa|gdn|recurrent|exact_replay|vector_replay", re.IGNORECASE
)
WARM_TARGET = re.compile(
    r"exact_scores_kernel|vector_replay_kernel|target_split|"
    r"gdn_replay_ssm.*(?:convolve|recurrent|finish)",
    re.IGNORECASE,
)
LONG_ATTENTION = re.compile(
    r"target_split|split_decode_attention|tmem_staged.*attention", re.IGNORECASE
)
CAPTURE = re.compile(r"^(?:cuda|cu)StreamBeginCapture")
LAUNCH = re.compile(r"^(?:cuda|cu)GraphLaunch")
SPLIT = 2047
BOUNDARY = 32768
MAX_CONTEXT = 262144
MIXED_CROSSOVER_BEGIN = 32000
MIXED_CROSSOVER_END = 33000
CASE_FIELD_COUNT = 7
MAX_DRAFT_DEPTH = 4
BASELINE_GROUP_CAP = 4
TWO_RANKS = 2
SHARDED_DRAFT_HEADS = 2
MIN_REPLAYS = 2
MIN_VERIFY_ROWS = 2
TIMING_REPETITIONS = 3

Case = tuple[int, int, int, str, str, str]
Step = tuple[int, int, int, int, int, int, int, int, float, int, int, int, int, int, int]
Lane = tuple[int, int, int, int, int, int, int, list[int]]
Steps = dict[int, Step]
Lanes = dict[int, dict[int, Lane]]
Restores = dict[tuple[int, int], tuple[int, ...]]
Memory = dict[tuple[str, int, int], tuple[int, int, int]]
Budgets = dict[int, tuple[int, int]]
Planes = dict[tuple[int, str, int], tuple[int, int, int]]
Run = tuple[Case, Steps, Lanes, Restores, Memory, dict[int, list[int]], dict[int, int]]
Api = tuple[str, int, int, int, int]
Grid = tuple[int, int, int]
Node = tuple[int, int, int, int, int, int, str, Grid]
Trace = tuple[list[Api], list[Node]]


class Execution(TypedDict):
    active_mask: int
    replay_mask: int
    regular_graph_mask: int
    verification_graph_mask: int
    fused_submissions: int
    fused_rows: int
    fused_collectives: int
    fused_two_shot_collectives: int
    completed_kernel_nodes: dict[int, int]
    graph_ids: list[int]
    attention: list[tuple[int, str]]
    attention_grids: list[tuple[int, str, Grid]]
    input_positions: list[int]
    verified_rows: dict[int, int]


class Replay(TypedDict):
    warm_steps: list[int]
    capture_steps: list[int]
    replay_steps: list[int]
    full_steps: list[int]
    execution: dict[int, Execution]


def need(condition: object, message: str, *details: object) -> None:
    if not condition:
        raise ValueError((message, *details))


def read_run(path: Path) -> Run:
    case = None
    steps, lanes, restores, memory, budgets = {}, defaultdict(dict), {}, {}, {}
    devices = {}
    planes = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split("\t")
        kind = fields[0]
        if kind == "case":
            need(case is None and len(fields) == CASE_FIELD_COUNT, "Invalid run identity", line)
            case = (*map(int, fields[1:4]), *fields[4:])
        elif kind == "step":
            step, selected, k0, verify, rows, vl, vr, begin, end = map(int, fields[1:10])
            usec = float(fields[10])
            request, workspace, fused, fused_rows, fused_collectives, two_shot = map(
                int, fields[11:17]
            )
            need(
                step == len(steps) and selected and not (k0 & verify or (k0 | verify) & ~selected),
                "Invalid replay mask or step sequence",
                line,
            )
            need(
                math.isfinite(usec) and usec > 0 and (end > begin or end == begin == 0),
                "Invalid step timestamps",
                line,
            )
            steps[step] = (
                selected,
                k0,
                verify,
                rows,
                vl,
                vr,
                begin,
                end,
                usec,
                request,
                workspace,
                fused,
                fused_rows,
                fused_collectives,
                two_shot,
            )
        elif kind == "lane":
            step, slot, position, drafted, accepted, verified, repair, boundary, finished, *ids = (
                map(int, fields[1:])
            )
            need(
                ids and slot not in lanes[step] and 0 <= accepted <= drafted <= MAX_DRAFT_DEPTH,
                "Invalid lane row",
                line,
            )
            lanes[step][slot] = (
                position,
                drafted,
                accepted,
                verified,
                repair,
                boundary,
                finished,
                ids,
            )
        elif kind == "restore":
            slot, rank, *layout = map(int, fields[1:])
            need((slot, rank) not in restores, "Duplicate synthetic restore", line)
            restores[slot, rank] = tuple(layout)
        elif kind == "plane":
            _, slot, family, rank, pages, payload, scale_bytes = fields
            key = (int(slot), family, int(rank))
            need(key not in planes, "Duplicate physical restore plane", line)
            planes[key] = (int(pages), int(payload), int(scale_bytes))
        elif kind == "admit":
            slot, length, budget = map(int, fields[1:])
            need(slot not in budgets, "Duplicate synthetic admission", line)
            budgets[slot] = (length, budget)
        elif kind == "device":
            rank, device = map(int, fields[1:])
            need(
                rank not in devices and device not in devices.values(),
                "Duplicate trace device mapping",
                line,
            )
            devices[rank] = device
        elif kind == "memory":
            label, step, rank, request, available, total = fields[1:]
            key = (label, int(step), int(rank))
            need(
                key not in memory and 0 <= int(available) <= int(total),
                "Invalid memory sample",
                line,
            )
            memory[key] = (int(request), int(available), int(total))
    need(case and steps and len(budgets) == case[1], "Incomplete inference process", str(path))
    check_restoration(path, case, budgets, restores, planes)
    check_memory(path, case[0], memory, steps)
    output = check_output(path, case, steps, lanes, budgets)
    return case, steps, lanes, restores, memory, output, devices


def check_restoration(
    path: Path,
    case: Case,
    budgets: Budgets,
    restores: Restores,
    planes: Planes,
) -> None:
    world, _, depth, _, _, _ = case
    expected_ranks = {(slot, rank) for slot in budgets for rank in range(world)}
    need(set(restores) == expected_ranks, "Restore omitted a rank or lane", str(path))
    expected_planes = set()
    for (slot, rank), (target, heads, draft, draft_heads) in restores.items():
        need(
            target == budgets[slot][0] - 2 and heads == 4 // world,
            "Unexpected target restore coverage",
            restores,
        )
        if depth:
            need(
                (draft == target - 1 and draft_heads in (SHARDED_DRAFT_HEADS, 4))
                or (world == TWO_RANKS and rank == 1 and draft == 0),
                "Invalid draft coverage",
                restores,
            )
        else:
            need(draft == 0, "Target-only request received draft state", restores)
        for family, rows, kv_heads in (("target", target, heads), ("draft", draft, draft_heads)):
            if rows:
                key = (slot, family, rank)
                expected_planes.add(key)
                pages, payload, scale_bytes = planes.get(key, (0, 0, 0))
                need(
                    pages * 128 >= rows
                    and payload == kv_heads * 128 * 256
                    and scale_bytes == kv_heads * 128 * 8,
                    "Invalid physical restore geometry",
                    str(path),
                    key,
                    rows,
                    planes.get(key),
                )
    need(set(planes) == expected_planes, "Unexpected physical restore plane", str(path))


def check_memory(path: Path, world: int, memory: Memory, steps: Steps) -> None:
    for rank in range(world):
        keys = {(stage, -1, rank) for stage in ("model", "admitted", "retired", "released")}
        keys |= {("step", step, rank) for step in steps}
        need(
            keys <= memory.keys() and memory["retired", -1, rank][0] == 0,
            "Missing live/retired rank memory",
            str(path),
            rank,
        )


def check_output(
    path: Path,
    case: Case,
    steps: Steps,
    lanes: Lanes,
    budgets: Budgets,
) -> dict[int, list[int]]:
    _, _, depth, _, _, shape = case
    output = defaultdict(list)
    finished = set()
    for step, stats in steps.items():
        need(
            sum(1 << slot for slot in lanes[step]) == stats[0],
            "Selected slots and output differ",
            str(path),
            step,
        )
        for slot, event in lanes[step].items():
            position, _, _, _, _, _, ended, ids = event
            length, budget = budgets[slot]
            need(
                slot not in finished and position == length + len(output[slot]) - 1,
                "Position advanced incorrectly",
                str(path),
                step,
                slot,
                event,
            )
            output[slot].extend(ids)
            if ended:
                need(len(output[slot]) == budget, "Terminal output lost tokens", str(path), slot)
                finished.add(slot)
    need(finished == set(budgets), "Missing terminal requests", str(path), finished)
    if shape == "max":
        need(
            all(length + budget == MAX_CONTEXT for length, budget in budgets.values()),
            "Case did not reach maximum context",
            str(path),
        )
    elif shape == "mixed-max":
        ends = [length + budget for length, budget in budgets.values()]
        need(
            ends.count(MAX_CONTEXT) == 1 and min(ends) < SPLIT,
            "Mixed case did not retain short and maximum contexts",
            str(path),
        )
    if depth:
        need(
            any(
                event[1] == depth and event[3] == depth + 1
                for lane in lanes.values()
                for event in lane.values()
            ),
            "Case did not run its requested speculation depth",
            str(path),
        )
    return output


def read_trace(path: Path) -> Trace:
    api, nodes = [], []
    with path.open(encoding="utf-8") as source:
        for line in source:
            fields = line.rstrip("\n").split("\t")
            if fields[0] in ("runtime", "driver"):
                _, name, correlation, status, begin, end = fields
                api.append((name, *map(int, (correlation, status, begin, end))))
            elif fields[0] == "node":
                _, graph, node, correlation, device, begin, end, name, x, y, z = fields
                ids = tuple(map(int, (graph, node, correlation, device, begin, end)))
                nodes.append((*ids, name, tuple(map(int, (x, y, z)))))
            elif fields[0] == "dropped":
                need(int(fields[1]) == 0, "CUPTI dropped records", str(path), line)
    need(nodes, "Missing GPU execution records", str(path))
    return api, nodes


def execution_receipt(
    stats: Step,
    lane: dict[int, Lane],
    mask: int,
    nodes: list[Node],
    ranks: dict[int, int],
) -> Execution:
    selected, k0, verify, _, _, _, _, _, _, _, _, fused, rows, collectives, two_shot = stats
    attention = [
        (ranks[node[3]], node[6], node[7]) for node in nodes if REPLAY_ATTENTION.search(node[6])
    ]
    return {
        "active_mask": selected,
        "replay_mask": mask,
        "regular_graph_mask": k0,
        "verification_graph_mask": verify,
        "fused_submissions": fused,
        "fused_rows": rows,
        "fused_collectives": collectives,
        "fused_two_shot_collectives": two_shot,
        "completed_kernel_nodes": dict(Counter(ranks[node[3]] for node in nodes)),
        "graph_ids": sorted({node[0] for node in nodes}),
        "attention": sorted({(rank, name) for rank, name, _ in attention}),
        "attention_grids": sorted(set(attention)),
        "input_positions": [row[0] for row in lane.values()],
        "verified_rows": {slot: row[3] for slot, row in lane.items()},
    }


def replay_receipt(run: Run, trace: Trace, *, candidate: bool = False) -> Replay:
    case, steps, lanes, _, _, _, devices = run
    world, _, depth, route, _, _ = case
    apis, nodes = trace
    need(set(devices) == set(range(world)), "Missing CUPTI rank mapping", case, devices)
    ranks_by_device = {device: rank for rank, device in devices.items()}
    all_slots = steps[0][0]
    warm_steps, capture_steps, replay_steps, full_steps = [], [], [], []
    full_graph_ids: Counter[int] = Counter()
    routes: dict[int, Execution] = {}
    for index, stats in steps.items():
        selected, k0, verify, _, _, vr, start, end, _, _, _, fused, fr, fc, two_shot = stats
        need(start and end > start, "Trace clocks were absent", case, index)
        api = [call for call in apis if start <= call[3] <= end]
        launch = {corr for name, corr, code, _, _ in api if code == 0 and LAUNCH.match(name)}
        capture = [name for name, _, code, _, _ in api if code == 0 and CAPTURE.match(name)]
        current = [
            node for node in nodes if node[4] >= start and node[5] <= end and node[5] > node[4]
        ]
        completed = [node for node in current if node[0] and node[1] and node[2] in launch]
        ranks = {
            ranks_by_device.get(node[3]) for node in completed if REPLAY_ATTENTION.search(node[6])
        }
        replay_mask = k0 if depth == 0 else verify
        reported = bool(k0 or verify or fused)
        if route == "eager":
            need(
                not capture and not launch and not reported and not completed,
                "Eager control launched a decode graph",
                case,
                index,
                launch,
            )
        if capture:
            capture_steps.append(index)
        if (
            index > 0
            and replay_mask != selected
            and any(not node[0] and WARM_TARGET.search(node[6]) for node in current)
        ):
            warm_steps.append(index)
        if reported:
            need(
                completed and ranks == set(range(world)),
                "Graph receipt lacks completed attention kernels on required ranks",
                case,
                index,
                dict(Counter(node[3] for node in completed)),
            )
            need(
                index > 0 and (world == 1 or not depth or fused > 0),
                "Replay receipt does not identify the verifier submission",
                case,
                index,
            )
            if candidate and world == TWO_RANKS and depth:
                verified_rows = sum(
                    event[3] for slot, event in lanes[index].items() if verify & (1 << slot)
                )
                need(
                    0 < fr == k0.bit_count() + verified_rows and fc > 0 and 0 <= two_shot <= fc,
                    "TP2 graph row count differs from regular and verification masks",
                    case,
                    index,
                    stats,
                )
            replay_steps.append(index)
            routes[index] = execution_receipt(
                stats, lanes[index], replay_mask, completed, ranks_by_device
            )
            if selected == all_slots and (
                replay_mask == selected
                or (
                    not candidate
                    and fused
                    and all(event[3] >= MIN_VERIFY_ROWS for event in lanes[index].values())
                )
            ):
                if candidate and world == TWO_RANKS and depth:
                    need(
                        fr == vr,
                        "Full-group mask overstated verified graph rows",
                        case,
                        index,
                        fr,
                        vr,
                    )
                full_graph_ids.update(routes[index]["graph_ids"])
                full_steps.append(index)
    receipt: Replay = {
        "warm_steps": warm_steps,
        "capture_steps": capture_steps,
        "replay_steps": replay_steps,
        "full_steps": full_steps,
        "execution": routes,
    }
    check_reuse(run, receipt, full_graph_ids, candidate=candidate)
    return receipt


def check_reuse(run: Run, receipt: Replay, graph_ids: Counter[int], *, candidate: bool) -> None:
    case, _, _, restores, _, _, _ = run
    world, width, depth, route, _, _ = case
    full_steps = receipt["full_steps"]
    if candidate and depth and world == TWO_RANKS:
        need(
            all(
                draft == target - 1 and draft_heads == SHARDED_DRAFT_HEADS
                for target, _, draft, draft_heads in restores.values()
            ),
            "TP2 synthetic restore did not bind both physical draft shards",
            case,
            restores,
        )
    if candidate and route == "graph":
        capture = receipt["capture_steps"]
        need(
            capture
            and len(full_steps) >= MIN_REPLAYS
            and max(graph_ids.values(), default=0) >= MIN_REPLAYS,
            "Capture, full-group execution, and graph reuse were not all observed",
            case,
            capture,
            full_steps,
            graph_ids,
        )
        warm = receipt["warm_steps"]
        if world == 1:
            need(
                any(i < full_steps[0] for i in warm),
                "No eager warm decode preceded TP1 replay",
                case,
                warm,
                full_steps,
            )
        check_context(run, full_steps, receipt["execution"])
    if not candidate and route == "graph" and world == 1 and width > BASELINE_GROUP_CAP:
        need(not full_steps, "Baseline control unexpectedly replayed above its group cap", case)


def check_context(run: Run, full_steps: list[int], routes: dict[int, Execution]) -> None:
    case, _, lanes, _, memory, _, _ = run
    world, _, _, _, _, shape = case
    positions = [routes[i]["input_positions"] for i in full_steps]
    if shape == "split":
        need(
            any(all(p < SPLIT for p in pos) for pos in positions)
            and any(
                all(p >= SPLIT for p in routes[i]["input_positions"])
                and any(LONG_ATTENTION.search(name) for _, name in routes[i]["attention"])
                for i in full_steps
            ),
            "Replay did not span the split boundary",
            case,
        )
    if shape == "edge":
        need(
            any(all(p < BOUNDARY for p in pos) for pos in positions)
            and any(all(p >= BOUNDARY for p in pos) for pos in positions),
            "Replay didn't cross the 32k boundary",
            case,
        )
    if shape == "mixed-max":
        need(
            all(min(pos) < SPLIT for pos in positions)
            and any(any(MIXED_CROSSOVER_BEGIN <= p < BOUNDARY for p in pos) for pos in positions)
            and any(any(BOUNDARY <= p < MIXED_CROSSOVER_END for p in pos) for pos in positions),
            "Full-group replay omitted the mixed 32k crossover",
            case,
        )
    if shape in ("max", "mixed-max"):
        need(
            any(max(pos) >= MAX_CONTEXT - 16 for pos in positions),
            "Replay wasn't live near 256k",
            case,
        )
    if shape in ("edge", "max", "mixed-max"):
        long_ranks = {
            rank
            for i in full_steps
            for rank, name in routes[i]["attention"]
            if LONG_ATTENTION.search(name)
        }
        need(
            long_ranks == set(range(world)),
            "Long replay omitted a bounded split or staged attention operator",
            case,
        )
        live = [i for i in full_steps if not any(r[6] for r in lanes[i].values())]
        need(len(live) >= MIN_REPLAYS, "No stable long replay memory interval", case, live)
        sizes = {memory["step", i, rank][0] for i in live for rank in range(world)}
        need(
            len(sizes) == 1 and next(iter(sizes)) > 0,
            "Long replay grew request storage with pages already provisioned",
            case,
            sizes,
        )


def p95(values: tuple[float, ...]) -> float:
    return sorted(values)[math.ceil(0.95 * len(values)) - 1]


def measure(run: Run, *, skip: int, capture_steps: set[int]) -> dict[str, object]:
    (_, _, _, route, _, _), steps, lanes, _, memory, _, _ = run
    need(all(step[6] == step[7] == 0 for step in steps.values()), "Timing arm ran under CUPTI")
    steady = [
        (steps[i][8], sum(len(event[7]) for event in lanes[i].values()))
        for i in steps
        if i >= skip and i not in capture_steps
    ]
    need(steady and all(tokens for _, tokens in steady), "No steady-state timing observations")
    durations, tokens = zip(*steady, strict=True)
    by_rank = defaultdict(list)
    for (stage, step, rank), (_, free, total) in memory.items():
        by_rank[rank].append((stage, step, total - free))
    footprint = {}
    for rank, records in by_rank.items():
        by_stage = {stage: used for stage, _, used in records if stage != "step"}
        live = max(used for stage, _, used in records if stage in ("admitted", "step"))
        footprint[rank] = {
            "model_used_bytes": by_stage["model"],
            "peak_used_bytes": live,
            "incremental_used_bytes": live - by_stage["model"],
            "released_used_bytes": by_stage["released"],
        }
    return {
        "route": route,
        "median_step_us": statistics.median(durations),
        "p95_step_us": p95(durations),
        "us_per_emitted_token": sum(durations) / sum(tokens),
        "first_decode_us": steps[1][8],
        "steady_steps": len(steady),
        "steady_tokens": sum(tokens),
        "tracked_request_peak_bytes": max(step[9] for step in steps.values()),
        "prefill_workspace_peak_bytes": max(step[10] for step in steps.values()),
        "rank_memory": footprint,
    }


def validate(directory: Path, *, trace_only: bool = False) -> dict[str, object]:
    evidence = {}
    for version in ("baseline", "candidate"):
        evidence[version] = {}
        for route in ("eager", "graph"):
            stem = f"{version}-{route}"
            observed = read_run(directory / f"{stem}-trace.tsv")
            receipt = replay_receipt(
                observed,
                read_trace(directory / f"{stem}-cupti.tsv"),
                candidate=version == "candidate",
            )
            quiet = []
            if not trace_only:
                quiet = [read_run(path) for path in sorted(directory.glob(f"{stem}-timing-*.tsv"))]
                need(
                    len(quiet) == TIMING_REPETITIONS,
                    "Expected three unprofiled repetitions",
                    stem,
                    len(quiet),
                )
                for run in quiet:
                    need(
                        run[0] == observed[0]
                        and run[2] == observed[2]
                        and run[5] == observed[5]
                        and list(run[1]) == list(observed[1])
                        and all(
                            run[1][i][:6] == step[:6] and run[1][i][11:] == step[11:]
                            for i, step in observed[1].items()
                        ),
                        "Unprofiled run changed route, replay, or output",
                        stem,
                    )
            evidence[version][route] = (observed, receipt, quiet)
        eager, graph = (evidence[version][route][0] for route in ("eager", "graph"))
        need(
            eager[5] == graph[5],
            "Graph and eager output token IDs differ",
            version,
            dict(eager[5]),
            dict(graph[5]),
        )
        need(
            dict(eager[2]) == dict(graph[2]),
            "Replay changed lane transitions",
            version,
            dict(eager[2]),
            dict(graph[2]),
        )
    baseline = evidence["baseline"]["eager"][0]
    candidate = evidence["candidate"]["eager"][0]
    need(
        baseline[0] == candidate[0] and baseline[5] == candidate[5],
        "Candidate changed baseline output or case identity",
        dict(baseline[5]),
        dict(candidate[5]),
    )
    if trace_only:
        return {
            "case": candidate[0],
            "exact_output_ids": dict(candidate[5]),
            "arms": {
                version: {route: {"trace": arm[1]} for route, arm in by_route.items()}
                for version, by_route in evidence.items()
            },
        }
    receipts = [arm[1] for by_route in evidence.values() for arm in by_route.values()]
    skip = max([3] + [r["full_steps"][0] + 1 for r in receipts if r["full_steps"]])
    capture_steps = {step for r in receipts for step in r["capture_steps"]}
    for by_route in evidence.values():
        for route, (observed, receipt, quiet) in by_route.items():
            timings = [measure(run, skip=skip, capture_steps=capture_steps) for run in quiet]
            by_route[route] = (observed, receipt, timings)
    table = {
        version: {route: {"trace": arm[1], "timings": arm[2]} for route, arm in by_route.items()}
        for version, by_route in evidence.items()
    }
    cost = {
        version: {
            route: statistics.median(t["us_per_emitted_token"] for t in arm[2])
            for route, arm in by_route.items()
        }
        for version, by_route in evidence.items()
    }
    candidate_cost, baseline_cost = cost["candidate"], cost["baseline"]
    return {
        "case": candidate[0],
        "exact_output_ids": dict(candidate[5]),
        "arms": table,
        "candidate_graph_to_candidate_eager": (candidate_cost["graph"] / candidate_cost["eager"]),
        "candidate_graph_to_baseline_graph": candidate_cost["graph"] / baseline_cost["graph"],
        "candidate_eager_to_baseline_eager": candidate_cost["eager"] / baseline_cost["eager"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check frozen synthetic graph evidence")
    parser.add_argument("directory", type=Path)
    parser.add_argument("--trace-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(validate(args.directory, trace_only=args.trace_only), indent=2))


if __name__ == "__main__":
    main()
