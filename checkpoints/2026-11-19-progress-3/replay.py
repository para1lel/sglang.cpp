#!/usr/bin/env python3
"""Restore the sealed source series using only Git objects and local tools."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import tempfile


BUNDLE = Path(__file__).resolve().parent
ANCHOR = "c2045c1cd769209b3476d799237304723e79c680"
ANCHOR_TREE = "4d41bd5baab2d45d085c50ccb7b7a17d98a26385"
PRIOR = "checkpoints/2026-11-19-performance"
MANIFEST_SHA256 = "988500765c79acb14a156e041dd20dd63897d93a70654be376940f3f939a4f79"
MARKER = ".verified-replay.json"
GIT_ENV = {
    **{key: value for key, value in os.environ.items() if not key.startswith("GIT_")},
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_NO_REPLACE_OBJECTS": "1",
    "GIT_NO_LAZY_FETCH": "1",
    "GIT_OPTIONAL_LOCKS": "0",
    "GIT_TERMINAL_PROMPT": "0",
    "GIT_ALLOW_PROTOCOL": "",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def safe_path(name):
    return (re.fullmatch(r"[a-zA-Z0-9._/-]+", name) and
            all(part not in {"", ".", "..", ".git"} for part in name.split("/")))


def inventory(root, omit=()):
    require(stat.S_ISDIR(root.lstat().st_mode), f"Expected source directory: {root}")
    entries = {}
    for path in root.rglob("*"):
        name = path.relative_to(root).as_posix()
        require(safe_path(name), f"Unsafe path: {name}")
        mode = path.lstat().st_mode
        if stat.S_ISDIR(mode):
            require(any(path.iterdir()), f"Empty directory: {name}")
            continue
        require(stat.S_ISREG(mode) and not mode & (stat.S_ISUID | stat.S_ISGID | stat.S_ISVTX),
                f"Expected an ordinary regular file: {name}")
        require(mode & stat.S_IXUSR or not mode & (stat.S_IXGRP | stat.S_IXOTH),
                f"Invalid executable mode: {name}")
        if name not in omit:
            entries[name] = (f"100755" if mode & stat.S_IXUSR else "100644",
                             digest(path.read_bytes()))
    return entries


def tree_identity(root, omit=()):
    entries = inventory(root, omit)
    records = "".join(f"{mode} {hash_value}  {name}\n"
                      for name, (mode, hash_value) in sorted(entries.items()))
    return {"files": len(entries), "sha256": digest(records.encode())}


def load_bundle(seal):
    actual = inventory(BUNDLE)
    require("SHA256SUMS" in actual and actual["SHA256SUMS"] == ("100644", seal),
            "Bundle seal or checksum file mode mismatch")
    expected = {}
    for line in (BUNDLE / "SHA256SUMS").read_text().splitlines():
        match = re.fullmatch(r"(100644|100755) ([0-9a-f]{64})  (.+)", line)
        require(match is not None, "Invalid checksum record")
        mode, hash_value, name = match.groups()
        require(safe_path(name) and name not in expected and name != "SHA256SUMS",
                f"Invalid bundle path: {name}")
        expected[name] = mode, hash_value
    require(set(actual) == set(expected) | {"SHA256SUMS"}, "Bundle path set mismatch")
    for name, record in expected.items():
        require(actual[name] == record, f"Bundle bytes or mode mismatch: {name}")
    require(actual["manifest.json"][1] == MANIFEST_SHA256, "Frozen manifest mismatch")
    return json.loads((BUNDLE / "manifest.json").read_bytes())


def git(repo, *arguments, **kwargs):
    environment = kwargs.pop("env", GIT_ENV)
    return subprocess.check_output(["git", "-C", repo, *arguments], env=environment, **kwargs)


def archive(repo, revision, destination, *paths):
    destination.mkdir()
    data = git(repo, "archive", "--format=tar", revision, *paths)
    subprocess.run(["tar", "-xf", "-", "-C", destination], input=data, check=True)


def published_tree(repo, revision):
    git(repo, "merge-base", "--is-ancestor", revision, ANCHOR)
    return git(repo, "rev-parse", "--verify", f"{revision}^{{tree}}").decode().strip()


def published_donors(repo, manifest, work):
    export = work / "published"
    archive(repo, ANCHOR, export, PRIOR)
    bundle = export / PRIOR
    donors = work / "donors"
    subprocess.run(
        ["python3", "-IB", bundle / "verify.py", "--repo", repo,
         "--seal", manifest["prior_seal"], "--output", donors, "serial-cta", "w3-fused"],
        env=GIT_ENV, check=True, stdout=subprocess.PIPE,
    )
    for name in ("serial-cta", "w3-fused"):
        (donors / name / MARKER).unlink()
        expect_tree(donors / name, name, manifest["donors"][name])
    return donors


def apply_patch(tree, patch):
    content = (BUNDLE / patch["file"]).read_bytes()
    require(digest(content) == patch["sha256"], f"Patch digest mismatch: {patch['file']}")
    environment = {**GIT_ENV, "GIT_CEILING_DIRECTORIES": str(tree.parent)}
    options = ("-c", "apply.ignoreWhitespace=no", "apply", "-p1")
    arguments = {"input": content, "env": environment}
    rows = git(tree, *options, "--numstat", "-", **arguments).decode().splitlines()
    fields = [row.split("\t") for row in rows]
    require(fields and all(len(row) == 3 and row[0].isdigit() and row[1].isdigit() and
                           safe_path(row[2]) for row in fields), "Expected text-only safe paths")
    require(sorted(row[2] for row in fields) == patch["paths"],
            f"Patch path mismatch: {patch['file']}")
    git(tree, *options, "--check", "-", **arguments)
    git(tree, *options, "--whitespace=nowarn", "-", **arguments)


def restore(repo, manifest, name, destination, donors):
    series = manifest["series"][name]
    if "donor" in series:
        shutil.copytree(donors / series["donor"], destination)
    else:
        base = series["base"]
        require(published_tree(repo, base) == series["base_tree"], f"Base tree mismatch: {name}")
        archive(repo, base, destination)
    for path, expected in series.get("serial_cta_files", {}).items():
        source = donors / "serial-cta" / path
        require(digest(source.read_bytes()) == expected, f"Donor blob mismatch: {path}")
        target = destination / path
        require(not os.path.lexists(target), f"Donor blob would overwrite: {path}")
        shutil.copy2(source, target)
    for step in series["apply"]:
        apply_patch(destination, manifest["patches"][step])


def expect_tree(tree, name, expected, omit=()):
    observed = tree_identity(tree, omit)
    require(observed == expected, f"Source bytes, modes or paths differ: {name} ({observed})")


def marker(seal, name, series):
    return (json.dumps({"bundle_seal": seal, "series": name, **series["source"]},
                       sort_keys=True) + "\n").encode()


def check_tree(tree, seal, name, series):
    require(stat.S_ISDIR(tree.lstat().st_mode), f"Expected source directory: {tree}")
    record = tree / MARKER
    mode = record.lstat().st_mode
    require(stat.S_ISREG(mode) and not mode & 0o7111 and
            record.read_bytes() == marker(seal, name, series), "Verification record mismatch")
    expect_tree(tree, name, series["source"], {MARKER})


def verify(repo, manifest, seal, names, work, output):
    require(published_tree(repo, ANCHOR) == ANCHOR_TREE, "Anchor tree mismatch")
    donors = published_donors(repo, manifest, work)
    staged = work / "trees"
    staged.mkdir()
    for name in names:
        tree = staged / name
        restore(repo, manifest, name, tree, donors)
        series = manifest["series"][name]
        expect_tree(tree, name, series["source"])
        with (tree / MARKER).open("xb") as stream:
            stream.write(marker(seal, name, series))
        print(f"{name} {series['source']['files']} {series['source']['sha256']}")
    if output:
        subprocess.run(["mv", "-T", "-n", "--", staged, output], check=True)
        require(not staged.exists(), f"Output already exists: {output}")


def main():
    os.umask(0o022)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seal", required=True, help="Published SHA-256 of SHA256SUMS")
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--output", type=Path, help="New directory outside the repository")
    mode.add_argument("--check-tree", type=Path, help="Check one restored tree by series name")
    parser.add_argument("series", nargs="*")
    args = parser.parse_args()
    try:
        manifest = load_bundle(args.seal)
        names = args.series or list(manifest["series"])
        require(len(set(names)) == len(names) and set(names) <= manifest["series"].keys(),
                "Select unique series from manifest.json")
        if args.check_tree is not None:
            require(len(args.series) == 1, "Tree checking requires exactly one series")
            name = names[0]
            check_tree(args.check_tree.absolute(), args.seal, name, manifest["series"][name])
            return
        repo = args.repo.resolve()
        output = args.output.parent.resolve() / args.output.name if args.output else None
        if output is not None:
            require(not output.is_relative_to(repo) and not output.is_relative_to(BUNDLE),
                    "Output must be outside the repository and bundle")
            require(not os.path.lexists(output), f"Output already exists: {output}")
        with tempfile.TemporaryDirectory(prefix=".sglang-progress-replay-",
                                         dir=output.parent if output else None) as temporary:
            verify(repo, manifest, args.seal, names, Path(temporary), output)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
