"""Opt-in native CLI and HTTP acceptance for exact prefill on a reserved B200."""

import argparse
import hashlib
import json
import re
import select
from pathlib import Path
import struct
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from batch.run_native import Stream, await_retirement, burst, request, require, server, write_json

BOUNDARIES = (1, 2, 127, 128, 129, 257)
PROMPT_SHA256 = "314fb913d558ac717a0a0a2ac86c7d310ea06dd0b4558ce160a141dd18564c43"
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


def check_reset(port, log_path, evidence, width, cases):
    offset = log_path.stat().st_size
    with Stream(port, cases[257]) as cancelled:
        cancelled.post()
        deadline = time.monotonic() + 15
        while not any(max(rows) > 1 for _, _, rows, _ in prefill_rows(log_path, offset)):
            require(time.monotonic() < deadline, "No submitted chunk before reset")
            time.sleep(0.005)
        require(not select.select([cancelled.sock], [], [], 0)[0], "Reset must precede output")
        cancelled.reset()
    await_retirement(log_path, offset, width)
    replacement = request(port, cases[129])
    await_retirement(log_path, offset, width)
    write_json(evidence, {"cancelled": cancelled.evidence(), "replacement": replacement.evidence()})


def check_http(args, cases):
    with server(args, args.evidence / "http-1", "default") as (port, _):
        streams = [request(port, case).evidence() for case in cases.values()]
        write_json(args.evidence / "serial.json", streams)
    for width, lengths in ((2, (129, 257)), (4, (2, 127, 129, 257))):
        evidence = args.evidence / f"http-{width}"
        with server(args, evidence, str(width)) as (port, log_path):
            offset = log_path.stat().st_size
            burst(port, log_path, evidence / "cohort.json", [cases[size] for size in lengths],
                  full_decode=False)
            await_retirement(log_path, offset, width)
            rows = prefill_rows(log_path, offset)
            write_json(evidence / "prefill.json", rows)
            require(rows and all(
                sum(positions) + decodes <= 128 and (max(positions) > 1) == (scratch > 0)
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
    args = parser.parse_args()
    cases = load_cases()
    if args.check_fixtures and not (args.generate or args.server):
        print("Six prefill boundary fixtures passed")
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
