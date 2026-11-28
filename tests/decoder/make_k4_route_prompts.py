"""Generate fixed 32K token histories for the native K4 route probes."""

from __future__ import annotations

import argparse
import hashlib
import random
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from tokenizers import Tokenizer

if TYPE_CHECKING:
    from collections.abc import Iterator


SITES = (
    "the western depot",
    "the north pier",
    "the public reading room",
    "the upper greenhouse",
    "the inspection shed",
    "the old water tower",
    "the maintenance yard",
    "the gravel road",
    "the freight office",
)
OBJECTS = (
    "shipping labels",
    "pressure gauges",
    "replacement hinges",
    "transit maps",
    "sample jars",
    "weather seals",
    "filter housings",
    "inspection tags",
    "circuit covers",
    "crate dividers",
)
OBSERVATIONS = (
    "The indicator returned to its resting position before the shift ended.",
    "The site supervisor checked the count against the previous entry.",
    "The incoming team recorded the change on the public notice board.",
    "The morning crew moved the equipment away from the service door.",
    "The technician compared the replacement parts against the drawing.",
    "The clerk copied the inventory number into the paper register.",
    "The second inspection confirmed the markings on both containers.",
    "The afternoon crew left the aisle clear for a delivery.",
)
TAIL = (
    "\n\nThe report above is complete. The next file is a plain text test fixture. "
    "Continue the fixture by emitting the line `The ledger is closed.` repeatedly, "
    "one copy per line, for the entire available response. Do not number the lines "
    "or add any explanation.<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    "The ledger is closed.\nThe ledger is closed.\nThe ledger is closed.\n"
    "The ledger is closed.\nThe ledger is closed.\nThe ledger is closed.\n"
    "The ledger is closed.\nThe ledger is closed.\n"
)

COHORTS = {
    16: (
        "W4",
        (
            "Atlas",
            "Autumn",
            "Bamboo",
            "Barrel",
            "Basin",
            "Beach",
            "Birch",
            "Blue",
            "Brass",
            "Bridge",
            "Bronze",
            "Cedar",
            "Chalk",
            "Clay",
            "Cloud",
            "Crystal",
        ),
        "e98ece04e9b2834f31538edf2d5b31db14fb95cddf55ad52667cfa21a4078766",
    ),
    32: (
        "W5",
        (
            "Dawn",
            "Delta",
            "Elm",
            "Falcon",
            "Fern",
            "Field",
            "Flame",
            "Forest",
            "Fox",
            "Garden",
            "Glass",
            "Gold",
            "Grain",
            "Granite",
            "Harbor",
            "Hazel",
            "Iron",
            "Ivy",
            "Jade",
            "Lake",
            "Lark",
            "Laurel",
            "Leaf",
            "Light",
            "Maple",
            "Marsh",
            "Moss",
            "Oak",
            "Ocean",
            "Olive",
            "Onyx",
            "Orchid",
        ),
        "eeca86a6bd804048dc264f17c97828ff19b0589299ed7e84b7905d7ab72a5cfb",
    ),
}
TOKENIZER_SHA256 = "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3"
SOURCE_SEED = "sglang.cpp-concurrent-baseline-20261118-v2"


def entries(title: str, name: str, lane: int, seed: str, count: int) -> Iterator[str]:
    rng = random.Random(int(hashlib.sha256(seed.encode()).hexdigest(), 16))
    yield f"{title} archive marker.\n<|im_start|>user\nInventory records for {name}-{lane:02d}.\n\n"
    for index in range(1, count + 1):
        day, hour = rng.randrange(1, 29), rng.randrange(8, 21)
        site, obj = rng.choice(SITES), rng.choice(OBJECTS)
        number = rng.randrange(3, 190)
        yield (
            f"Entry {index}: On October {day} at {hour}:00, the team at {site} "
            f"logged {number} {obj}. {rng.choice(OBSERVATIONS)} "
            f"File reference {rng.randrange(100000, 999999)}.\n"
        )


def exact_prompt(tokenizer: Tokenizer, title: str, name: str, lane: int) -> list[int]:
    count = 32768
    seed = f"{SOURCE_SEED}/{name}/{lane}"
    body = "".join(entries(title, name, lane, seed, count // 25))
    body_ids = tokenizer.encode(body, add_special_tokens=False).ids
    tail_ids = tokenizer.encode(TAIL, add_special_tokens=False).ids
    offset = count - len(tail_ids) - 16
    body_fragment = tokenizer.decode(body_ids[:offset], skip_special_tokens=False).rstrip()
    pad = 8
    for _ in range(4):
        text = body_fragment + "\nArchive tags:" + " inventory" * pad + TAIL
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        if len(ids) == count and "\ufffd" not in text:
            return ids
        pad += count - len(ids)
    raise ValueError(f"Lane {lane} could not reach {count} tokens")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tokenizer", type=Path)
    parser.add_argument("lanes", type=int, choices=COHORTS)
    args = parser.parse_args()
    if hashlib.sha256(args.tokenizer.read_bytes()).hexdigest() != TOKENIZER_SHA256:
        raise ValueError("Checkpoint tokenizer digest mismatch")
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    name, titles, expected = COHORTS[args.lanes]
    lines = []
    for lane, title in enumerate(titles):
        ids = exact_prompt(tokenizer, title, name, lane)
        lines.append(" ".join(map(str, ids)))
    output = ("\n".join(lines) + "\n").encode("ascii")
    if hashlib.sha256(output).hexdigest() != expected:
        raise ValueError("Generated K4 route fixture differs from the fixed token histories")
    sys.stdout.buffer.write(output)


if __name__ == "__main__":
    main()
