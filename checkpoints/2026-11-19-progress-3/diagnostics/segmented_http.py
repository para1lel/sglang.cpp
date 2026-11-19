import gzip
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import urlopen

from benchmarks.concurrent import reference
from benchmarks.concurrent.client import capture_cohort, replay_lane, validate_cohort
from benchmarks.streaming.client import json_bytes, request_body


BATCH = re.compile(r"batch width=(\d+) decode=(\d+) request_bytes=(\d+)")
PREFILL = re.compile(r"batch prefill_rows=([\d,]+) workspace_bytes=(\d+) packed_rows=(\d+)")


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def diagnostic_lanes(fixtures, row, width):
    lanes = []
    for index, (original, _, source) in enumerate(reference.load_lanes(fixtures, row)[:width]):
        case = {"prompt_token_ids": source["prompt_token_ids"][:256], "output_tokens": 8}
        body = json_bytes(request_body(case))
        metadata = {"lane": index, "source_lane": original["lane"],
                    "source_request_sha256": original["request_sha256"],
                    "request_sha256": hashlib.sha256(body).hexdigest()}
        lanes.append((metadata, body, case))
    return lanes


def wait_ready(server, url):
    deadline = time.monotonic() + 90
    while server.poll() is None and time.monotonic() < deadline:
        try:
            with urlopen(f"{url}/health", timeout=1) as response:
                if response.status == 200:
                    return
        except (OSError, URLError):
            time.sleep(0.1)
    raise RuntimeError(f"Readiness failed (server status {server.poll()})")


def summarize_trace(text, width, packed):
    steps = [tuple(map(int, match.groups())) for match in BATCH.finditer(text)]
    details = [(tuple(map(int, match[1].split(","))), int(match[2]), int(match[3]))
               for match in PREFILL.finditer(text)]
    observed_width = max((step[0] for step in steps), default=0)
    prompt_rows = sum(sum(detail[0]) for detail in details)
    packed_rows = sum(detail[2] for detail in details)
    decode_rows = sum(step[1] for step in steps)
    if (observed_width != width or prompt_rows != 256 * width or decode_rows != 7 * width
            or bool(packed_rows) != packed):
        raise ValueError(f"Unexpected trace at width {width}/{packed}: "
                         f"{observed_width=}, {prompt_rows=}, {decode_rows=}, {packed_rows=}")
    return {"max_width": observed_width, "prompt_rows": prompt_rows, "decode_rows": decode_rows,
            "packed_rows": packed_rows,
            "peak_workspace_bytes": max(detail[1] for detail in details),
            "peak_request_bytes": max(step[2] for step in steps)}


def run_arm(server_path, checkpoint, target, row, width, arm, policy, lanes):
    packed = arm == "packed"
    with socket.socket() as reserved_port:
        reserved_port.bind(("127.0.0.1", 0))
        port = reserved_port.getsockname()[1]
    args = [server_path, "--checkpoint", checkpoint, "--device", "0", "--port", str(port),
            "--max-active", str(width), "--prefill-rows", "128", "--fp8-prefill", "exact",
            "--trace-batches"]
    if not packed:
        args.append("--per-request-prefill")
    output = target / f"{row}-width{width}-{arm}"
    output.mkdir()
    write_json(output / "args.json", args)
    url = f"http://127.0.0.1:{port}"
    endpoint = urlsplit(url)
    results = []
    stderr = output / "server.stderr"
    with (output / "server.stdout").open("wb") as server_out, stderr.open("wb") as server_err:
        server = subprocess.Popen(args, stdout=server_out, stderr=server_err)
        try:
            wait_ready(server, url)
            counts = (("warmup", policy["warmup_cohorts"]),
                      ("measured", policy["measured_cohorts"]))
            for phase, count in counts:
                for ordinal in range(1, count + 1):
                    label = f"{phase}-{ordinal:02}"
                    offset = stderr.stat().st_size
                    raw = capture_cohort(endpoint, row, lanes, label, policy)
                    packet = gzip.compress(json.dumps(raw).encode())
                    (output / f"{label}.json.gz").write_bytes(packet)
                    records = [replay_lane(source, lane)
                               for source, lane in zip(raw["requests"], lanes, strict=True)]
                    sent = [item["submit_ns"] for item in records]
                    span = (max(sent) - min(sent)) / 1e6
                    if raw["instrumentation"] or span > policy["max_submission_span_ms"]:
                        raise ValueError(f"Invalid capture instrumentation or submissions: {label}")
                    try:
                        timing = validate_cohort(raw, lanes, policy)
                        overlap_error = None
                    except ValueError as error:
                        timing, overlap_error = None, str(error)
                    trace = summarize_trace(stderr.read_bytes()[offset:].decode(), width, packed)
                    results.append({"label": label, "submission_span_ms": span, "trace": trace,
                                    "overlap_error": overlap_error, "cohort_timing": timing,
                                    "output_ids_by_lane": [item["output_ids"] for item in records]})
        finally:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
    write_json(output / "summary.json", results)
    return results


def main(server_path, checkpoint, fixtures, mode="default"):
    target = Path(os.environ["SEGMENTED_RUN_DIR"])
    policy = reference.reference()["policy"]
    if mode == "default":
        matrix = (("W4", 2, ("separate", "packed")), ("W5", 4, ("packed", "separate")))
    elif mode == "reverse-width2":
        matrix = (("W4", 2, ("packed", "separate")),)
    else:
        raise ValueError(f"Unknown diagnostic matrix {mode}")
    for row, width, order in matrix:
        lanes = diagnostic_lanes(Path(fixtures), row, width)
        arms = {arm: run_arm(server_path, checkpoint, target, row, width, arm, policy, lanes)
                for arm in order}
        cohorts = [cohort for cohorts in arms.values() for cohort in cohorts]
        baseline = cohorts[0]["output_ids_by_lane"]
        if any(cohort["output_ids_by_lane"] != baseline for cohort in cohorts):
            raise ValueError(f"Diagnostic HTTP greedy IDs changed at width {width}")
        print(f"{row} width={width} cohorts={len(cohorts)} ids={len(cohorts) * width * 8}",
              flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
