"""Measure concurrent cohorts and require repeatable greedy IDs within the arm."""

import argparse
import json
import sys
from pathlib import Path
from unittest.mock import patch

from benchmarks.concurrent import client


def measure(args):
    reference = client.reference()
    reference["policy"].update(warmup_cohorts=args.warmups, measured_cohorts=args.measured)
    validate_cohort = client.validate_cohort
    first_outputs = None
    mismatch = None

    def validate_repeats(raw, lanes, policy):
        nonlocal first_outputs, mismatch
        result = validate_cohort(raw, lanes, policy)
        outputs = [request["output_ids"] for request in raw["requests"]]
        if first_outputs is None:
            first_outputs = outputs
            return result
        for request, expected in zip(raw["requests"], first_outputs, strict=True):
            if request["output_ids"] == expected:
                continue
            for offset, (before, after) in enumerate(
                zip(expected, request["output_ids"], strict=True)
            ):
                if before != after:
                    mismatch = {"tag": raw["tag"], "reference_tag": "warmup-01",
                                "lane": request["lane"], "offset": offset,
                                "expected_id": before, "observed_id": after}
                    raise ValueError(f"Greedy output ID differs from first warmup: {mismatch}")
        return result

    with patch.object(client, "reference", return_value=reference), \
            patch.object(client, "validate_cohort", side_effect=validate_repeats):
        exit_code = client.run(args.row, args.endpoint, args.fixtures, args.output)
    path = args.output / "summary.json"
    report = json.loads(path.read_bytes())
    if mismatch is not None:
        report.update(status="repeat_id_mismatch", repeat_id_mismatch=mismatch)
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(f"{args.row}: repeat_id_mismatch {mismatch}", file=sys.stderr)
        return 3
    return 0 if report["status"] == "complete" else exit_code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("row", choices=client.ROWS)
    parser.add_argument("endpoint")
    parser.add_argument("fixtures", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--measured", type=int, default=5)
    args = parser.parse_args()
    if args.warmups < 1 or args.measured < 2:
        parser.error("Use at least one warmup and two measured cohorts")
    return measure(args)


if __name__ == "__main__":
    raise SystemExit(main())
