#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


BUNDLE = Path(__file__).resolve().parent


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def checked(path, expected):
    require(digest(path) == expected, f"SHA-256 mismatch for {path}")


def git_environment():
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith("GIT_")
    }
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    return environment


def run(command, cwd, environment):
    return subprocess.check_output(command, cwd=cwd, env=environment, text=True).strip()


def revision_tree(repo, revision, environment):
    return run(["git", "rev-parse", "--verify", f"{revision}^{{tree}}"], repo, environment)


def export(repo, revision, target, environment):
    target.mkdir()
    with subprocess.Popen(
        ["git", "archive", revision], cwd=repo, env=environment, stdout=subprocess.PIPE
    ) as archive:
        subprocess.run(["tar", "-xf", "-", "-C", target], stdin=archive.stdout, check=True)
        archive.stdout.close()
        require(archive.wait() == 0, f"git archive failed for {revision}")


def apply_patch(name, patch, tree, environment):
    path = BUNDLE / patch["file"]
    checked(path, patch["sha256"])
    command = ["git", "-c", "apply.ignoreWhitespace=no", "apply", "--whitespace=nowarn"]
    rows = run([*command, "--numstat", path], tree, environment).splitlines()
    fields = [row.split("\t") for row in rows]
    require(all(len(row) == 3 and all(count.isdigit() for count in row[:2])
                for row in fields), f"Expected a text-only patch for {name}")
    paths = sorted(row[2] for row in fields)
    require(paths == patch["paths"], f"Patch path mismatch for {name}")
    require(all(not Path(path).is_absolute() and ".." not in Path(path).parts
                for path in paths), f"Unsafe path in {name}")
    run([*command, "--check", path], tree, environment)
    run([*command, path], tree, environment)
    return set(paths)


def verify_sources(series, tree, touched):
    receipt = BUNDLE / series["sources"]
    checked(receipt, series["sources_sha256"])
    sources = {}
    for line in receipt.read_text().splitlines():
        expected, name = line.split("  ", 1)
        sources[name] = expected

    inventory = {"xmake.lua"} | {name for name in touched if (tree / name).is_file()}
    for directory in ("apps", "src", "include"):
        inventory.update(path.relative_to(tree).as_posix()
                         for path in (tree / directory).rglob("*") if path.is_file())
    require(inventory == sources.keys(), f"Source inventory mismatch for {tree.name}")
    for name, expected in sources.items():
        checked(tree / name, expected)
    return len(sources)


def verify(repo, selected, output, manifest):
    published = manifest["publication_anchor"]
    environment = git_environment()
    require(revision_tree(repo, published, environment) == manifest["anchor_tree"],
            "Publication anchor tree mismatch")
    with tempfile.TemporaryDirectory(prefix="sglang-source-replay-") as scratch:
        scratch = Path(scratch)
        apply_environment = {**environment, "GIT_CEILING_DIRECTORIES": str(scratch)}
        for name in selected:
            series = manifest["series"][name]
            base = series["base"]
            require(revision_tree(repo, base, environment) == series["base_tree"],
                    f"Base tree mismatch for {name}")
            run(["git", "merge-base", "--is-ancestor", base, published], repo, environment)
            tree = scratch / name
            export(repo, base, tree, environment)
            touched = set()
            for step in series["apply"]:
                patch = manifest["patches"][step]
                touched.update(apply_patch(step, patch, tree, apply_environment))
            count = verify_sources(series, tree, touched)
            print(f"{name}: {len(series['apply'])} patch(es), {count} source hashes verified")
        if output is not None:
            output.mkdir()
            for name in selected:
                shutil.copytree(scratch / name, output / name)
    print(f"{len(selected)} series verified")


def main():
    manifest = json.loads((BUNDLE / "manifest.json").read_text())
    parser = argparse.ArgumentParser(description="Reconstruct and verify the checkpoint source.")
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, help="Keep verified exports in this new directory")
    parser.add_argument("series", nargs="*", metavar="SERIES",
                        help=f"Defaults to all. Choices: {', '.join(manifest['series'])}")
    args = parser.parse_args()
    selected = args.series or manifest["series"]
    unknown = set(selected) - manifest["series"].keys()
    if unknown:
        parser.error(f"unknown series: {', '.join(sorted(unknown))}")
    if len(set(selected)) != len(selected):
        parser.error("select each series once")
    try:
        verify(args.repo.resolve(), selected, args.output, manifest)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
