#!/usr/bin/env python3
"""Recreate token histories and check fresh model-reference captures."""

import argparse
import hashlib
import json
from pathlib import Path
import struct
import tempfile


HERE = Path(__file__).resolve().parent
VOCAB = 248320
ROW_BYTES = VOCAB * 4
MODEL_MANIFEST = "4d5c57825022fa8c1177b7c6b125b8ec704332b1f57cfd2a2306dae856f418fb"
W3_HISTORY = "1f626922f2149dbd8e7c61dcd09c45eb4b5edee77117533f2912aef3b828150b"
W3_CYCLE = [760, 45366, 369, 7629, 13, 198]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def load(path):
    return json.loads(path.read_bytes())


def recipe(name):
    return load(HERE / "recipes" / f"{name}-inputs.json")


def checked(data, expected, label):
    if digest(data) != expected:
        raise ValueError(f"Digest mismatch: {label}")
    return data


def emit(path, data):
    with path.open("xb") as stream:
        stream.write(data)
    print(f"{digest(data)}  {path}")


def token_line(tokens):
    if not tokens or any(type(token) is not int or not 0 <= token < VOCAB for token in tokens):
        raise ValueError("Invalid token list")
    return (" ".join(map(str, tokens)) + "\n").encode()


def model(args):
    source = recipe("model")
    cases = source["cases"]
    for case in cases.values():
        if "prefix" in case:
            parent, size = case.pop("prefix")
            case["prompt"] = cases[parent]["prompt"][:size]
    data = (json.dumps(cases, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    emit(args.output, checked(data, source["inputs_sha256"], "model inputs"))


def concurrent(args):
    if args.kind == "w3":
        lines = []
        for lane in range(4):
            path = args.fixtures / "W3" / f"{lane:02}.request.json"
            prompt = load(path)["prompt"]
            if len(prompt) != 8192:
                raise ValueError(f"Expected 8192 prompt tokens in {path}")
            lines.append(token_line(prompt + (W3_CYCLE * 22)[:128]))
        emit(args.output, checked(b"".join(lines), W3_HISTORY, "W3 teacher history"))
        return

    source = recipe("w6")
    chunks = []
    for index, lane in enumerate(source["lanes"]):
        path = args.fixtures / "W6" / f"{index:02}.request.json"
        request = checked(path.read_bytes(), lane["request_sha256"], path)
        prompt = json.loads(request)["prompt"]
        teacher = lane["teacher"]
        if (lane["lane"] != index or len(prompt) != 131072 or
                prompt[0] != lane["first"] or len(teacher) != 16):
            raise ValueError(f"Unexpected W6 lane or token count in {path}")
        chunks.append(struct.pack("<131088i", *prompt, *teacher))
    emit(args.output, checked(b"".join(chunks), source["fixture_sha256"], "W6 fixture"))


def w2(args):
    source = recipe("w2")
    reference = load(args.streaming / "gpu3-reference.json")["workloads"]
    files = {}
    for name, case in source.items():
        request = load(args.streaming / f"{name}.json")
        prompt = checked(token_line(request["prompt_token_ids"]), case["prompt_sha256"], name)
        native = checked(token_line(case["native_teacher"]),
                         case["native_sha256"], f"{name} native")
        pinned = checked(token_line(reference[name]["greedy_output_ids"]),
                         case["pinned_sha256"], f"{name} vLLM")
        for arm, teacher in (("native", native), ("vllm", pinned)):
            files[arm, name] = prompt, teacher
    args.output.mkdir()
    for arm in ("native", "vllm", "reset"):
        (args.output / arm).mkdir()
    for (arm, name), (prompt, teacher) in files.items():
        path = args.output / arm
        emit(path / f"{name}.prompt", prompt)
        emit(path / f"{name}.teacher", teacher)
        if arm == "native":
            if name == "W2":
                prompt = prompt.rstrip() + b" " + prompt.split()[0] + b"\n"
            emit(args.output / "reset" / f"{name}.prompt", prompt)


def capture(args):
    frozen = args.suite / "manifest.json"
    manifest = json.loads(checked(frozen.read_bytes(), MODEL_MANIFEST, frozen))
    mismatches = []
    with tempfile.TemporaryDirectory(prefix=".model-capture-", dir=args.report.parent) as directory:
        staged = Path(directory) / "bundle"
        staged.mkdir()
        for name, expected in sorted(manifest["files"].items()):
            if name == "inputs.json":
                data = args.inputs.read_bytes()
            elif name == "capture.json":
                data = (args.capture / name).read_bytes()
            else:
                case, repeat, extension = name.split(".")
                if extension != "f32":
                    raise ValueError(f"Unexpected frozen capture filename: {name}")
                data = (args.capture / name).read_bytes()
                if manifest["cases"][case]["repeats"] == 1:
                    steps = manifest["cases"][case]["steps"]
                    if repeat != "0" or len(data) < (max(steps) + 1) * ROW_BYTES:
                        raise ValueError(f"Insufficient archived-case rows: {name}")
                    data = b"".join(data[step * ROW_BYTES:(step + 1) * ROW_BYTES] for step in steps)
            observed = digest(data)
            if observed == expected:
                if args.output:
                    (staged / name).write_bytes(data)
            else:
                mismatches.append({"file": name, "expected": expected, "observed": observed})
        report = {"manifest_sha256": MODEL_MANIFEST, "files": len(manifest["files"]),
                  "mismatches": mismatches}
        emit(args.report, (json.dumps(report, indent=2) + "\n").encode())
        if mismatches:
            raise ValueError("Fresh capture drifts from the frozen fixture. See the digest report.")
        if args.output:
            if args.output.exists() or args.output.is_symlink():
                raise ValueError(f"Output already exists: {args.output}")
            (staged / "manifest.json").write_bytes(frozen.read_bytes())
            staged.rename(args.output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="kind", required=True)
    model_parser = commands.add_parser("model")
    model_parser.add_argument("output", type=Path)
    model_parser.set_defaults(run=model)
    for name in ("w3", "w6"):
        command = commands.add_parser(name)
        command.add_argument("fixtures", type=Path)
        command.add_argument("output", type=Path)
        command.set_defaults(run=concurrent)
    w2_parser = commands.add_parser("w2")
    w2_parser.add_argument("streaming", type=Path)
    w2_parser.add_argument("output", type=Path)
    w2_parser.set_defaults(run=w2)
    compare = commands.add_parser("capture")
    compare.add_argument("suite", type=Path)
    compare.add_argument("inputs", type=Path)
    compare.add_argument("capture", type=Path)
    compare.add_argument("report", type=Path)
    compare.add_argument("--output", type=Path)
    compare.set_defaults(run=capture)
    args = parser.parse_args()
    try:
        args.run(args)
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
