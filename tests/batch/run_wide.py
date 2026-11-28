"""Check wide K0 admission on the CPU and run opt-in B200 serving regressions."""

import argparse
import os
from pathlib import Path
import re
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_graph_evidence
import test_batch_graph as http


WIDTHS = (16, 32)
PREFILL_ROW_BUDGET = 128
BATCH_GRAPH_WIDTHS = check_graph_evidence.BATCH_GRAPH_WIDTHS
NO_DEVICE = "Select device for CUDA arena startup: no CUDA-capable device is detected"
NATIVE_STEP = re.compile(r"^step decode=(\d+) graph=(\d+) bytes=(\d+)$", re.M)


def check_cli(args):
    valid = [([], NO_DEVICE)]
    valid.extend((["--max-active", str(width)], NO_DEVICE) for width in (1, 2, 3, 4, *WIDTHS))
    valid.extend((["--max-active", str(width), "--mtp-steps", str(depth)], NO_DEVICE)
                 for width in (1, 2, 3, 4, 5, *WIDTHS) for depth in (1, 4))
    valid.extend((["--max-active", str(width), "--batch-cuda-graphs"], NO_DEVICE)
                 for width in (1, *WIDTHS))
    invalid = [(["--max-active", value], "--max-active in [1,")
               for value in ("0", "-1", "33", "wide", "2147483648")]
    valid.append((["--max-active", "16", "--eager"], NO_DEVICE))
    invalid.append((["--max-active", "4", "--mtp-steps", "5"], "--mtp-steps in [0, 4]"))
    missing = args.evidence / "no-checkpoint"
    records = []
    for flags, diagnostic in valid + invalid:
        command = [str(args.binary.resolve()), "--checkpoint", str(missing.resolve()), *flags]
        result = subprocess.run(command, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
                                capture_output=True, text=True, timeout=15)
        passed = (result.returncode == 1 and diagnostic in result.stderr and
                  (NO_DEVICE not in result.stderr or diagnostic == NO_DEVICE))
        records.append({"command": command, "expected": diagnostic, "stderr": result.stderr,
                        "exit": result.returncode, "passed": passed})
    path = args.evidence / "cli.json"
    http.save(path, records)
    failures = [record for record in records if not record["passed"]]
    assert not failures, (path, [(row["command"][3:], row["stderr"]) for row in failures])
    print(f"CPU CLI passed {len(records)} policies and controls", flush=True)


def check_native(args, width):
    output = args.evidence / f"native-{width}"
    output.mkdir()
    command = [str(args.lanes.resolve()), str(args.checkpoint.resolve()), str(width)]
    activity = output / "cupti.tsv"
    env = dict(os.environ, LD_PRELOAD=str(args.activity.resolve()),
               BATCH_GRAPH_ACTIVITY=str(activity.resolve()))
    http.save(output / "command.json", command)
    log = output / "lanes.log"
    with log.open("wb") as stream:
        result = subprocess.run(command, env=env, stdout=stream, stderr=stream, timeout=180)
    assert result.returncode == 0, log
    steps = [tuple(map(int, step)) for step in NATIVE_STEP.findall(log.read_text())]
    assert steps and steps[-1][2] == 0, log
    masks = [mask for _, mask, _ in steps if mask]
    assert (1 << (width - 1)) in masks, log
    assert all(not mask or decode.bit_count() in BATCH_GRAPH_WIDTHS
               for decode, mask, _ in steps), log
    device, _ = check_graph_evidence.trace(activity)
    assert len(device) == len(masks), (log, activity)
    print(f"Native width {width} passed eager decode, high-slot refill, and CUPTI", flush=True)


def check_http(args, width):
    out = args.evidence / f"http-{width}"
    spec = argparse.Namespace(output=out, binary=args.binary, checkpoint=args.checkpoint,
                              width=width, route="graph", activity=args.activity,
                              max_num_batched_tokens=PREFILL_ROW_BUDGET)
    with http.server(spec) as port:
        log = out / "server.log"
        case = http.FIXTURES["held-out"]
        reference, = http.parallel(port, [case], exact=False)
        cases = http.native.serial_cases(port, http.FIXTURES, out / "serial-reference.json")
        offset = log.stat().st_size
        cohort = [{**case, "name": f"lane-{lane}"} for lane in range(width)]
        responses = http.parallel(port, cohort, exact=False)
        http.native.await_retirement(log, offset, width)
        rows = http.scheduler(log, offset)
        http.save(out / "cohort.json", {"reference": reference, "streams": responses, "rows": rows})
        assert all(item["tokens"] == reference["tokens"] and item["finish"] == reference["finish"]
                   for item in responses), out
        assert max(item["tokens_ns"][0] for item in responses) < min(
            item["done_ns"] for item in responses), "A stream finished before every lane emitted"
        assert any(row["width"] == width for row in rows), out
        assert any(row["decode"] > 4 for row in rows), out
        http.native.check_admission(port, log, out / "cancel-and-reuse.json",
                                    cases, width)
        http.native.await_retirement(log, offset, width)
        http_rows = http.scheduler(log, 0)
    assert all(0 <= row["decode"] <= row["width"] <= width for row in http_rows), out
    for row in http_rows:
        if row["width"]:
            prefill = row["prefill"]
            assert len(prefill) == width and sum(prefill) + row["decode"] <= PREFILL_ROW_BUDGET, row
        if mask := row["graph_slots"]:
            assert (mask < 1 << width and mask.bit_count() == row["decode"]
                    and row["decode"] in BATCH_GRAPH_WIDTHS), row
    device, _ = check_graph_evidence.trace(out / "cupti.tsv")
    assert len(device) == sum(bool(row["graph_slots"]) for row in http_rows), out
    print(f"HTTP width {width} passed exact overlapping SSE, queue, cancel, and refill", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True, help="Fresh evidence directory")
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--activity", type=Path,
                        help="CUPTI observer built from graph_activity.cpp")
    parser.add_argument("--lanes", type=Path,
                        help="Native executable built from check_wide_lanes.cpp")
    parser.add_argument("--width", type=int, action="append", choices=WIDTHS)
    args = parser.parse_args()
    if not args.cpu_only and not all((args.checkpoint, args.activity, args.lanes)):
        parser.error("GPU checks require --checkpoint, --activity, and --lanes")
    args.evidence.mkdir()
    check_cli(args)
    if not args.cpu_only:
        for width in args.width or WIDTHS:
            check_native(args, width)
            check_http(args, width)


if __name__ == "__main__":
    main()
