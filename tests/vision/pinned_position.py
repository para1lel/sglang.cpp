import argparse
import ast
import hashlib
import json
from pathlib import Path
import struct
from types import SimpleNamespace

import numpy as np
from safetensors import safe_open
import torch


PINNED_SOURCE_SHA256 = "a7979e1fe7606d0acede19e41653adc5ce480cc1a6c7df253f3196f3ac7b8741"


def upstream_position(source):
    content = source.read_bytes()
    if hashlib.sha256(content).hexdigest() != PINNED_SOURCE_SHA256:
        raise ValueError("SGLang source does not match the pinned Qwen3-VL digest")
    model = next(node for node in ast.parse(content).body
                 if isinstance(node, ast.ClassDef) and node.name == "Qwen3VLMoeVisionModel")
    method = next(node for node in model.body
                  if isinstance(node, ast.FunctionDef)
                  and node.name == "fast_pos_embed_interpolate_from_list")
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    return namespace[method.name]


def main():
    parser = argparse.ArgumentParser(description="Compare native positions with pinned SGLang")
    parser.add_argument("source", type=Path)
    parser.add_argument("shard", type=Path)
    parser.add_argument("cases", type=Path)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()

    interpolate = upstream_position(args.source)
    with safe_open(args.shard, framework="pt") as shard:
        weights = shard.get_tensor("model.visual.pos_embed.weight")
    assert weights.shape == (2304, 1152) and weights.dtype == torch.bfloat16
    state = SimpleNamespace(
        num_grid_per_side=48, spatial_merge_size=2, dtype=weights.dtype, device=args.device,
        pos_embed=torch.nn.Embedding.from_pretrained(weights.to(args.device)),
    )
    results = {}
    for name, case in json.loads(args.cases.read_text()).items():
        grid = struct.unpack("<3q", (Path(case["fixture"]) / "grid.i64").read_bytes())
        expected = interpolate(state, [grid]).view(torch.int16).cpu().numpy()
        native = np.fromfile(args.trace / name / "position.bf16", dtype="<i2")
        if native.size != expected.size:
            raise ValueError(f"{name} has {native.size} values, expected {expected.size}")
        mismatches = int(np.count_nonzero(native != expected.reshape(-1)))
        results[name] = {"grid": grid, "bf16_values": native.size, "mismatches": mismatches}
    print(json.dumps({"torch": torch.__version__, "device": args.device, "results": results},
                     indent=2))
    return int(any(case["mismatches"] for case in results.values()))


if __name__ == "__main__":
    raise SystemExit(main())
