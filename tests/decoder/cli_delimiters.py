"""Check the native CLI prompt delimiters without opening the checkpoint."""

import argparse
import os
import subprocess
from pathlib import Path


def invoke(binary, prompt_option, value):
    command = [str(binary), prompt_option, str(value), "--help"]
    return subprocess.run(
        command, capture_output=True, text=True, timeout=20,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES=""), check=False,
    )


def check(binary, destination):
    destination.mkdir()
    cases = (
        ("spaces", " 16 11\t220\n17\r11 ", None),
        ("form-feed", "\f16\f11\f", None),
        ("vertical-tab", "\v16\v11\v", None),
        ("mixed", "\v16 \f, \t11\r\n220\v,\f17 ", None),
        ("leading-comma", ",16", "Expected a nonnegative integer for prompt token"),
        ("empty-field", "16, ,11", "Expected a nonnegative integer for prompt token"),
        ("trailing-comma", "16, \t", "Missing prompt token after comma"),
    )
    failures = []
    for name, prompt, diagnostic in cases:
        prompt_file = destination / f"{name}.prompt"
        prompt_file.write_bytes(prompt.encode("ascii"))
        for option, value in (("--prompt-ids", prompt), ("--prompt-file", prompt_file)):
            result = invoke(binary, option, value)
            accepted = (
                result.returncode == 0 and result.stdout.startswith("Usage: sglang-generate")
                and not result.stderr
            )
            rejected = result.returncode == 1 and diagnostic and diagnostic in result.stderr
            valid = accepted if diagnostic is None else rejected
            print(f"{name} {option} {'PASS' if valid else 'FAIL'}")
            if not valid:
                failures.append(f"{name} {option}: status {result.returncode}, {result.stderr!r}")
    if failures:
        raise SystemExit("\n".join(failures))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    check(args.binary.resolve(strict=True), args.destination)


if __name__ == "__main__":
    main()
