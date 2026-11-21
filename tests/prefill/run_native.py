"""Opt-in native CLI and HTTP acceptance for exact prefill on a reserved B200."""

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import struct
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from batch.run_native import (
    TRACE, Stream, await_retirement, burst, request, require, server, write_json,
)

BOUNDARIES = (1, 2, 127, 128, 129, 257, 511)
PROMPT_SHA256 = "314fb913d558ac717a0a0a2ac86c7d310ea06dd0b4558ce160a141dd18564c43"
BATCH_TOKEN_BUDGET = 128
PREFILL = re.compile(
    rb"^batch width=(\d+) decode=(\d+) request_bytes=\d+\n"
    rb"batch prefill_rows=(\d+(?:,\d+){3}) workspace_bytes=(\d+)$", re.M
)


def load_cases():
    prompt_file = ROOT.parent.parent / "benchmarks" / "streaming" / "W2.json"
    prompt_bytes = prompt_file.read_bytes()
    require(hashlib.sha256(prompt_bytes).hexdigest() == PROMPT_SHA256, "W2 prompt changed")
    prompt = json.loads(prompt_bytes)["prompt_token_ids"]
    oracle = json.loads((ROOT / "serial-oracle.json.fixture").read_text())
    require(tuple(map(int, oracle)) == BOUNDARIES, "Invalid prefill boundary fixture")
    cases = {}
    for length in BOUNDARIES:
        entry = oracle[str(length)]
        tokens = entry["output"]
        require(len(tokens) == 5 and all(0 <= token < 248320 for token in tokens) and
                re.fullmatch(r"[0-9a-f]{64}", entry["top64_sha256"]),
                f"Invalid frozen scores at {length} rows")
        cases[length] = {
            "name": f"prefill-{length}", "prompt": prompt[:length], "max_tokens": len(tokens),
            "finish_reason": "length", **entry,
        }
    return cases


def score_digest(rows):
    digest = hashlib.sha256()
    for row in rows:
        for score in row:
            digest.update(struct.pack("<If", score["token_id"], score["logit"]))
    return digest.hexdigest()


def check_policy(args):
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    for executable, serving in ((args.generate, False), (args.server, True)):
        if executable is None:
            continue
        executable = executable.resolve(strict=True)
        cases = [
            (["--help"], 0, "--fp8-prefill exact|tensor-core"),
            (["--fp8-prefill", "auto"], 1, "Expected --fp8-prefill exact or tensor-core"),
            (["--fp8-prefill", ""], 1, "Expected --fp8-prefill exact or tensor-core"),
        ]
        missing = "value" if serving else "argument"
        cases.append((["--fp8-prefill"], 1, f"Missing {missing} for --fp8-prefill"))
        if serving:
            cases.append((
                ["--fp8-prefill", "tensor-core", "--max-active", "2", "--mtp-steps", "5"],
                1, "Expected --mtp-steps in [0, 4]",
            ))
        for flags, status, message in cases:
            result = subprocess.run([str(executable), *flags], env=environment, text=True,
                                    capture_output=True, timeout=10)
            output = result.stdout if status == 0 else result.stderr
            require(result.returncode == status and message in output,
                    f"{executable.name} {flags}: {result.returncode}, {output}")
    print("FP8 prefill policy option checks passed with CUDA devices hidden")


def check_cli(args, cases):
    for case in cases.values():
        evidence = args.evidence / case["name"]
        evidence.mkdir()
        prompt = case["prompt"]
        command = [
            str(args.generate), "--checkpoint", str(args.checkpoint), "--device", str(args.device),
            "--prompt-ids", ",".join(map(str, prompt)), "--max-new-tokens", str(case["max_tokens"]),
            "--top-logits", "64",
        ]
        write_json(evidence / "command.json", command)
        output_path = evidence / "stdout.json"
        with output_path.open("wb") as output, (evidence / "stderr.log").open("wb") as errors:
            subprocess.run(command, stdout=output, stderr=errors, check=True, timeout=120)
        result = json.loads(output_path.read_text())
        rows = result["step_top_logits"]
        require(result["prompt_token_ids"] == prompt
                and result["output_token_ids"] == case["output"]
                and result["finish_reason"] == "length" and len(rows) == case["max_tokens"]
                and all(len(row) == 64 for row in rows)
                and score_digest(rows) == case["top64_sha256"], case["name"])


def prefill_rows(log_path, offset):
    with log_path.open("rb") as log:
        log.seek(offset)
        data = log.read()
    return [
        (int(width), int(decode), tuple(map(int, rows.split(b","))), int(scratch))
        for width, decode, rows, scratch in PREFILL.findall(data)
    ]


def await_request_retirement(log_path, offset, capacity, name):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        with log_path.open("rb") as log:
            log.seek(offset)
            data = log.read()
        rows = [(offset + match.start(), *map(int, match.groups()))
                for match in TRACE.finditer(data) if match.end() < len(data)]
        allocation = next((row for row in rows if row[3] > 0), None)
        if allocation is not None and rows[-1][3] == 0:
            require(all(0 <= decode <= width <= capacity for _, width, decode, _ in rows),
                    f"Scheduler exceeded width {capacity}: {rows!r}")
            return {
                "trace_offset": offset, "allocation_offset": allocation[0],
                "allocation_bytes": allocation[3], "retirement_offset": rows[-1][0],
            }
        time.sleep(0.01)
    raise AssertionError(f"No fresh allocation-to-zero transition for {name}. See {log_path}")


def check_reset(port, log_path, evidence, width, cases):
    offset = log_path.stat().st_size
    with Stream(port, cases[257]) as cancelled:
        cancelled.post()
        cancelled.open()
        admitted = time.monotonic()
        processed = sum(sum(rows) for _, _, rows, _ in prefill_rows(log_path, offset))
        require(processed < len(cancelled.request["prompt"]),
                "Streaming headers must arrive before prefill completes")
        deadline = admitted + 15
        while not any(max(rows) > 1 for _, _, rows, _ in prefill_rows(log_path, offset)):
            require(time.monotonic() < deadline, "No submitted chunk before reset")
            time.sleep(0.005)
        cancelled.reset()
    cancelled_retirement = await_request_retirement(log_path, offset, width, "cancelled request")
    offset = log_path.stat().st_size
    replacement = request(port, cases[129])
    replacement_retirement = await_request_retirement(log_path, offset, width, "replacement")
    write_json(evidence, {
        "cancelled": cancelled.evidence(), "replacement": replacement.evidence(),
        "cancelled_retirement": cancelled_retirement,
        "replacement_retirement": replacement_retirement,
    })


def check_http_baseline(args, cases, name, width, flags, limits):
    evidence = args.evidence / name
    with server(args, evidence, width, flags=flags, expected_health=limits) as (port, _):
        streams = [request(port, case).evidence() for case in cases.values()]
        write_json(evidence / "streams.json", streams)


def check_http(args, cases):
    check_http_baseline(args, cases, "http-default", None, (),
                        {"max_active": 32, "max_num_batched_tokens": 32768})
    check_http_baseline(args, cases, "http-1", "1", ("--eager", "--fp8-prefill", "exact"),
                        {"max_active": 1, "max_num_batched_tokens": 32768})
    for width, lengths in ((2, (129, 257)), (4, (2, 127, 129, 257))):
        evidence = args.evidence / f"http-{width}"
        flags = ("--max-num-batched-tokens", str(BATCH_TOKEN_BUDGET))
        limits = {"max_active": width, "max_num_batched_tokens": BATCH_TOKEN_BUDGET}
        with server(args, evidence, str(width), flags=flags,
                    expected_health=limits) as (port, log_path):
            offset = log_path.stat().st_size
            burst(port, log_path, evidence / "cohort.json", [cases[size] for size in lengths],
                  full_decode=False)
            await_retirement(log_path, offset, width)
            rows = prefill_rows(log_path, offset)
            write_json(evidence / "prefill.json", rows)
            require(rows and all(
                sum(positions) + decodes <= BATCH_TOKEN_BUDGET
                and (max(positions) > 1) == (scratch > 0)
                for _, decodes, positions, scratch in rows
            ), "Invalid per-step position or scratch accounting")
            require(any(sum(count > 1 for count in positions) >= 2
                        for _, _, positions, _ in rows), "No concurrent chunked prefill lanes")
            require(any(decodes and max(positions) > 1 for _, decodes, positions, _ in rows),
                    "No decode progress alongside chunked prefill")
            if width == 4:
                check_reset(port, log_path, evidence / "reset.json", width, cases)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate", type=Path, help="Release sglang-generate")
    parser.add_argument("--server", type=Path, help="Release sglang-server")
    parser.add_argument("--checkpoint", type=Path, default=Path("/data0/shared/Qwen3.8-27B-FP8"))
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--check-fixtures", action="store_true")
    parser.add_argument("--check-policy", action="store_true",
                        help="Check CLI options without CUDA")
    args = parser.parse_args()
    if args.check_policy:
        if not (args.generate or args.server):
            parser.error("Provide at least one release executable for --check-policy")
        check_policy(args)
        return
    cases = load_cases()
    if args.check_fixtures and not (args.generate or args.server):
        print(f"{len(cases)} prefill fixtures passed")
        return
    if not args.evidence or not (args.generate or args.server):
        parser.error("Provide --evidence and at least one release executable")
    args.evidence = args.evidence.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.evidence.mkdir(parents=True)
    if args.generate:
        args.generate = args.generate.resolve(strict=True)
        check_cli(args, cases)
    if args.server:
        args.binary = args.server.resolve(strict=True)
        check_http(args, cases)
    write_json(args.evidence / "status.json", {"status": "passed"})


if __name__ == "__main__":
    main()
