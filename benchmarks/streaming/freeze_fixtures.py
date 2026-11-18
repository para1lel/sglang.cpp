import hashlib
import json
import sys
from pathlib import Path

from tokenizers import Tokenizer


TOKENIZER_SHA256 = "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3"

W1 = (
    "A small library lends hand tools as well as books. On Saturday, Mara opened "
    "the side door and found a line of people waiting with broken lamps, stiff "
    "umbrellas, and bicycles that had spent the winter outside. She put a blue "
    "cloth on each workbench and set out trays for screws. One volunteer knew "
    "how to solder; another could true a wheel and explain the adjustment in "
    "plain language. They agreed to keep a notebook listing each repair and "
    "the parts it needed. By midday the notebook had become a conversation "
    "between strangers, with sketches in the margins and suggestions for the "
    "next person who attempted the same repair. On the following Saturday, "
    "Mara asked the volunteers to read one entry aloud before opening the door."
)

SITES = (
    ("the old ferry steps", "river silt", "a brass ruler", "water level"),
    ("the market arch", "fallen leaves", "a linen cord", "stone movement"),
    ("the north orchard", "green apples", "a wire basket", "fruit damage"),
    ("the clay footbridge", "a cracked tile", "a paper gauge", "surface wear"),
    ("the hill reservoir", "a bent reed", "a compass", "wind direction"),
    ("the school courtyard", "a loose brick", "a tape measure", "shifting mortar"),
    ("the railway embankment", "white stones", "a wooden frame", "plant cover"),
    ("the mill wall", "damp plaster", "a chalk line", "a growing stain"),
    ("the churchyard gate", "a bent latch", "a spare hinge", "gate alignment"),
    ("the abandoned kiln", "red dust", "a hand lens", "the texture of clay"),
    ("the meadow drain", "broken rushes", "a marked stick", "standing water"),
    ("the gravel causeway", "flat pebbles", "a short chain", "the road edge"),
)


def long_prompt():
    entries = [
        "Field notebook of a walking survey. The survey follows public paths around "
        "a small town. Its purpose is to record ordinary changes over several visits "
        "in a form that a second observer can check. Each entry describes a place, "
        "an object, the method used, and a reason to come back."
    ]
    for number, (place, object_, tool, topic) in enumerate(SITES * 4, 1):
        entries.append(
            f"\n\nEntry {number}. At {place}, I noticed {object_} near the path. "
            f"I used {tool} as a reference and made a pencil sketch to document "
            f"{topic}. A second visitor should begin from the same side of the path "
            "and note the weather before comparing the sketch with the current scene. "
            "I left the site undisturbed and marked the visit on the printed map."
        )
    return "".join(entries)


def main():
    tokenizer_path, output_dir = map(Path, sys.argv[1:])
    if hashlib.sha256(tokenizer_path.read_bytes()).hexdigest() != TOKENIZER_SHA256:
        raise SystemExit("tokenizer.json differs from the fixture source")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    for name, text, count, output in (
        ("W1", W1, 128, 128),
        ("W2", long_prompt(), 2048, 512),
    ):
        ids = tokenizer.encode(text, add_special_tokens=False).ids[:count]
        if len(ids) < count:
            raise ValueError(f"{name} has only {len(ids)} tokens")
        rows = [", ".join(map(str, ids[start:start + 12])) for start in range(0, count, 12)]
        (output_dir / f"{name}.json").write_text(
            f'{{\n  "workload": {json.dumps(name)},\n  "prompt_token_ids": [\n    '
            + ",\n    ".join(rows)
            + f'\n  ],\n  "output_tokens": {output}\n}}\n',
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
