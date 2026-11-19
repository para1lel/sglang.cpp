"""Check matched native score trajectories without historic score vectors."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import statistics
import struct


VOCAB = 248320
WORD_BYTES = 2


def require(condition, message):
    if not condition:
        raise ValueError(message)


def compare(left, right, size=None):
    data = left.read_bytes()
    require(size is None or len(data) == size, f"Unexpected length: {left}")
    require(bool(data) and data == right.read_bytes(), f"Byte difference: {left} and {right}")
    return hashlib.sha256(data).digest()


def w3(args):
    hashes = []
    for lane in range(4):
        name = f"lane-{lane}.bf16"
        hashes.append(compare(args.left / name, args.right / name, 128 * VOCAB * WORD_BYTES))
    name = "winners.tsv"
    hashes.append(compare(args.left / name, args.right / name))
    lines = (args.left / name).read_text().splitlines()
    require(len(lines) == 512, "Expected 512 W3 winners")
    rows = [tuple(map(int, line.split("\t"))) for line in lines]
    require({(lane, step) for lane, step, _, _ in rows} ==
            {(lane, step) for lane in range(4) for step in range(128)}, "Missing W3 positions")
    return {"bf16_words": 512 * VOCAB, "winners": len(lines), "sha256": fold(hashes)}


def w3_http(args):
    records = []
    for phase, count in (("warmup", 2), ("measured", 5)):
        for trial in range(1, count + 1):
            tag = f"{phase}-{trial:02d}"
            cohort = []
            for arm in (args.left, args.right):
                with gzip.open(arm / f"{tag}.json.gz", "rt") as stream:
                    raw = json.load(stream)
                require(raw["row"] == "W3" and raw["tag"] == tag and not raw["errors"],
                        f"Invalid W3 cohort: {arm}/{tag}")
                requests = sorted(raw["requests"], key=lambda request: request["lane"])
                require([request["lane"] for request in requests] == list(range(4)),
                        f"Missing lanes: {arm}/{tag}")
                require(all(request["finish_reason"] == "length" and
                            len(request["output_ids"]) == 1024 and request["http_status"] == 200
                            for request in requests), f"Incomplete request: {arm}/{tag}")
                cohort.append([(request["request_sha256"], request["output_ids"])
                               for request in requests])
            require(cohort[0] == cohort[1], f"W3 fixtures or output IDs differ: {tag}")
            records.extend(cohort[0])
    payload = json.dumps(records, separators=(",", ":")).encode()
    return {"output_ids": sum(len(ids) for _, ids in records),
            "sha256": hashlib.sha256(payload).hexdigest()}


def w2(args):
    hashes = []
    count = 0
    for workload in ("W1", "W2"):
        name = f"{workload}.winners"
        history = (args.left / f"{workload}.teacher").read_bytes()
        require(history == (args.right / f"{workload}.teacher").read_bytes(),
                f"Different teacher histories: {workload}")
        compare(args.left / f"{workload}.prompt", args.right / f"{workload}.prompt")
        hashes.append(compare(args.left / name, args.right / name))
        expected_count = len(history.split())
        require(len((args.left / name).read_bytes().split()) == expected_count,
                f"Missing winners: {workload}")
        name = f"{workload}.bf16"
        size = expected_count * VOCAB * WORD_BYTES
        hashes.append(compare(args.left / name, args.right / name, size))
        count += expected_count
    return {"bf16_words": count * VOCAB, "winners": count, "sha256": fold(hashes)}


def fold(hashes):
    return hashlib.sha256(b"".join(hashes)).hexdigest()


def winner(path):
    values = struct.unpack(f"<{VOCAB}H", path.read_bytes())
    require(all(value & 0x7f80 != 0x7f80 for value in values), f"Nonfinite scores: {path}")
    return max(range(VOCAB), key=lambda index: bf16_order(values[index]))


def bf16_order(word):
    return -(word & 0x7fff) if word & 0x8000 else word


def w6_timing(log):
    samples = {0: [], 1: []}
    pattern = re.compile(r"projection_timing ([01]) \d+ (\d+) rows=131072 ms=(\S+)")
    for line in log.read_text().splitlines():
        match = pattern.fullmatch(line)
        if match and int(match[2]) > 0:
            samples[int(match[1])].append(float(match[3]))
    require(all(len(values) == 42 for values in samples.values()),
            "Expected 42 warm samples per W6 arm")
    return {label: statistics.median(samples[route])
            for route, label in enumerate(("mma_graph4", "packed_graph4"))}


def w6(args):
    hashes = []
    root = args.output
    size = VOCAB * WORD_BYTES
    for step in range(15):
        for lane in range(4):
            pair = "a" if lane % 2 == 0 else "b"
            suffix = f"{step}.{lane}.bf16"
            hashes.append(compare(root / f"policy_packed_graph4.{suffix}",
                                  root / f"policy_pair_{pair}.{suffix}", size))
    hashes.append(compare(root / "policy_packed_graph4.trace", root / "policy_pair_a.trace",
                          64 * 18 * 17408 * 2))
    for phase, lanes, expected in (("before", range(4), 45366),
                                   ("survivor", range(1, 4), 369),
                                   ("recycled", range(1, 4), 7629)):
        for lane in lanes:
            actual = root / f"cancel.{phase}.{lane}.bf16"
            hashes.append(compare(actual, root / f"cancel.expected_{phase}.{lane}.bf16", size))
            require(winner(actual) == expected, f"Unexpected cancellation winner: {actual}")
            if args.original and phase == "before":
                compare(actual, args.original / f"graph4.0.{lane}.bf16", size)
                compare(root / f"prefill.{lane}.bf16", args.original / f"prefill.{lane}.bf16", size)
    retired = re.search(r"^cancel before=(\d+) after=(\d+) recycled=(\d+) ",
                        args.log.read_text(), re.M)
    require(retired is not None, "Missing cancellation allocation trace")
    before, after, recycled = map(int, retired.groups())
    require(after < recycled < before and 4 * after == 3 * before,
            "Invalid cancellation accounting")
    return {"bf16_words": 60 * VOCAB, "winners": 60, "cancellation_rows": 10,
            "projection_p50_ms": w6_timing(args.log), "sha256": fold(hashes)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(required=True)
    for name, function in (("w2", w2), ("w3", w3), ("w3-http", w3_http)):
        command = commands.add_parser(name)
        command.add_argument("left", type=Path)
        command.add_argument("right", type=Path)
        command.set_defaults(run=function)
    command = commands.add_parser("w6")
    command.add_argument("output", type=Path)
    command.add_argument("log", type=Path)
    command.add_argument("--original", type=Path, help="Optional previous full-mode output")
    command.set_defaults(run=w6)
    args = parser.parse_args()
    try:
        print(json.dumps(args.run(args), sort_keys=True))
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
