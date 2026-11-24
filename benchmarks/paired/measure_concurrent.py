"""Measure concurrent cohorts and require repeatable greedy IDs within the arm."""

import argparse
from pathlib import Path

from benchmarks.concurrent import client


def main(*, validate=client.validate_cohort):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("row", choices=client.ROWS)
    parser.add_argument("endpoint")
    parser.add_argument("fixtures", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--warmups", type=int, default=1, choices=(1,))
    parser.add_argument("--measured", type=int, default=1, choices=(1,))
    args = parser.parse_args()
    return client.run(args.row, args.endpoint, args.fixtures, args.output,
                      validate=validate, enforce_gate=False)


if __name__ == "__main__":
    raise SystemExit(main())
