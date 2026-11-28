import json
from pathlib import Path
from typing import cast

HERE = Path(__file__).resolve().with_name("fixtures")


def load_prompt(name: str) -> list[int]:
    return cast("list[int]", json.loads((HERE / f"{name}_prompt.json").read_bytes()))
