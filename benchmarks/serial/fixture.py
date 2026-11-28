"""Prepare or verify the shared token inputs for the paired serial W6 row."""

import argparse
import gzip
import json
from importlib import import_module
from pathlib import Path
from typing import TypedDict

from benchmarks.concurrent.freeze_prompts import exact_prompt
from benchmarks.concurrent.reference import digest, lane_manifest
from benchmarks.streaming.client import json_bytes, token_ids

NAME = "w6-conversation-v1"
ROW = "W6"
K = 1024
TURNS = 20
FRESH_TOKENS = 9 * K
OUTPUT_TOKENS = K
PROMPT_TOKENS = tuple((10 * turn - 1) * K for turn in range(1, TURNS + 1))
MAX_CONTEXT = PROMPT_TOKENS[-1] + OUTPUT_TOKENS
INPUTS_SHA256 = "524fd76fc74cb86d92c9b173b818e72e82e7caf9886b842affcb71068a4f6ab1"
FILE = Path(ROW) / "inputs.json.gz"


class Inputs(TypedDict):
    warmup: list[int]
    fresh: list[list[int]]


def metadata() -> dict[str, str | int | list[int]]:
    return {
        "protocol": NAME,
        "k_tokens": K,
        "warmup_prompt_tokens": PROMPT_TOKENS[-1],
        "measured_prompt_tokens": list(PROMPT_TOKENS),
        "fresh_tokens_per_turn": FRESH_TOKENS,
        "output_tokens_per_request": OUTPUT_TOKENS,
        "measured_requests": TURNS,
        "total_requests": TURNS + 1,
        "inputs_sha256": INPUTS_SHA256,
    }


def load_inputs(root: Path) -> Inputs:
    raw = gzip.decompress((root / FILE).read_bytes())
    if digest(raw) != INPUTS_SHA256:
        raise ValueError("W6 input digest differs from the frozen fixture")
    inputs: Inputs = json.loads(raw)
    warmup, fresh = inputs["warmup"], inputs["fresh"]
    if (
        not token_ids(warmup)
        or len(warmup) != PROMPT_TOKENS[-1]
        or len(fresh) != TURNS
        or any(not token_ids(ids) or len(ids) != FRESH_TOKENS for ids in fresh)
        or warmup[0] == fresh[0][0]
    ):
        raise ValueError("Invalid or overlapping W6 fixture inputs")
    return inputs


def prepare(tokenizer_path: Path, root: Path) -> None:
    if digest(tokenizer_path.read_bytes()) != lane_manifest()["tokenizer_sha256"]:
        raise ValueError("Checkpoint tokenizer digest mismatch")
    tokenizer = import_module("tokenizers").Tokenizer.from_file(str(tokenizer_path))

    def ids(title: str, segment: int, count: int) -> list[int]:
        seed = f"sglang.cpp-paired-w6-v1/{segment}"
        return exact_prompt(tokenizer, title, ROW, segment, count, seed)[1]

    inputs = {
        "warmup": ids("Amber", 0, PROMPT_TOKENS[-1]),
        "fresh": [
            ids("Zephyr" if turn == 1 else "Delta", turn, FRESH_TOKENS)
            for turn in range(1, TURNS + 1)
        ],
    }
    raw = json_bytes(inputs)
    if digest(raw) != INPUTS_SHA256:
        raise ValueError("Generated W6 inputs differ from the frozen fixture")
    destination = root / FILE
    destination.parent.mkdir(parents=True, exist_ok=True)
    with (
        destination.open("xb") as stream,
        gzip.GzipFile(filename="", fileobj=stream, mode="wb", mtime=0) as compressed,
    ):
        compressed.write(raw)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("prepare")
    freeze.add_argument("tokenizer", type=Path)
    freeze.add_argument("root", type=Path)
    commands.add_parser("verify").add_argument("root", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.tokenizer, args.root)
    load_inputs(args.root)
    print(json.dumps(metadata(), indent=2))


if __name__ == "__main__":
    main()
