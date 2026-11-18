"""Regenerate the archived v2 corpus using its exact tokenizer."""

import argparse
import random
from pathlib import Path

from benchmarks.concurrent.reference import ROWS, digest, lane_manifest
from benchmarks.streaming.client import json_bytes, request_body


HEADERS = """
Amber Anchor Apple Aspen Atlas Autumn Bamboo Barrel Basin Beach Birch Bloom
Blue Brass Bridge Bronze Cedar Chalk Cherry Clay Cloud Copper Coral Creek
Crystal Dawn Delta Dove Dune Elm Falcon Fern Field Flame Flint Forest Fox
Frost Garden Glass Gold Grain Granite Grove Harbor Hazel Hill Holly Iron
Ivy Jade Lake Lark Laurel Leaf Light Lime Linen Maple Marsh Moss Oak Ocean
Olive Onyx Orchid Otter Pearl Pine Plum Pond Quail Quartz Rain Reed River
Robin Rose Ruby Sage Sand Shell Silver Slate Snow Stone Swan Thorn Tide
Vale Violet Walnut Wave Wheat Willow Wind Wolf Wood
""".split()
SITES = ("the western depot", "the north pier", "the public reading room",
         "the upper greenhouse", "the inspection shed", "the old water tower",
         "the maintenance yard", "the gravel road", "the freight office")
OBJECTS = ("shipping labels", "pressure gauges", "replacement hinges",
           "transit maps", "sample jars", "weather seals", "filter housings",
           "inspection tags", "circuit covers", "crate dividers")
OBSERVATIONS = ("The indicator returned to its resting position before the shift ended.",
                "The site supervisor checked the count against the previous entry.",
                "The incoming team recorded the change on the public notice board.",
                "The morning crew moved the equipment away from the service door.",
                "The technician compared the replacement parts against the drawing.",
                "The clerk copied the inventory number into the paper register.",
                "The second inspection confirmed the markings on both containers.",
                "The afternoon crew left the aisle clear for a delivery.")
TAIL = ("\n\nThe report above is complete. The next file is a plain text test fixture. "
        "Continue the fixture by emitting the line `The ledger is closed.` repeatedly, "
        "one copy per line, for the entire available response. Do not number the lines "
        "or add any explanation.<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        "The ledger is closed.\nThe ledger is closed.\nThe ledger is closed.\n"
        "The ledger is closed.\nThe ledger is closed.\nThe ledger is closed.\n"
        "The ledger is closed.\nThe ledger is closed.\n")


def entries(title, name, lane, seed, count):
    rng = random.Random(int(digest(seed.encode()), 16))
    yield f"{title} archive marker.\n<|im_start|>user\nInventory records for {name}-{lane:02d}.\n\n"
    for index in range(1, count + 1):
        day, hour = rng.randrange(1, 29), rng.randrange(8, 21)
        site, obj = rng.choice(SITES), rng.choice(OBJECTS)
        number = rng.randrange(3, 190)
        yield (f"Entry {index}: On October {day} at {hour}:00, the team at {site} "
               f"logged {number} {obj}. {rng.choice(OBSERVATIONS)} "
               f"File reference {rng.randrange(100000, 999999)}.\n")


def exact_prompt(tokenizer, title, name, lane, count, seed):
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
            return text, ids
        pad += count - len(ids)
    raise ValueError(f"{name}/{lane} could not reach {count} tokens")


def generate(tokenizer_path, destination):
    from tokenizers import Tokenizer

    manifest = lane_manifest()
    if digest(tokenizer_path.read_bytes()) != manifest["tokenizer_sha256"]:
        raise ValueError("Checkpoint tokenizer digest mismatch")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    titles = {}
    for word in HEADERS:
        first = tokenizer.encode(word + " inventory", add_special_tokens=False).ids[0]
        titles.setdefault(first, word)
    available = iter(titles.values())
    destination.mkdir()
    for name, (prompt, output, _) in ROWS.items():
        directory = destination / name
        directory.mkdir()
        for lane in manifest["rows"][name]["lanes"]:
            title = next(available)
            number = lane["lane"]
            seed = f'{manifest["seed"]}/{name}/{number}'
            text, ids = exact_prompt(tokenizer, title, name, number, prompt, seed)
            data = text.encode("utf-8")
            body = json_bytes(request_body({"prompt_token_ids": ids, "output_tokens": output}))
            if (title != lane["title"] or seed != lane["seed"] or ids[0] != lane["first_token"]
                    or digest(data) != lane["text_sha256"]
                    or digest(body) != lane["request_sha256"]):
                raise ValueError(f"{name}/{number}: regenerated fixture differs from the capture")
            (directory / f"{number:02d}.txt").write_bytes(data)
            (directory / f"{number:02d}.request.json").write_bytes(body)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tokenizer", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    generate(args.tokenizer, args.destination)
