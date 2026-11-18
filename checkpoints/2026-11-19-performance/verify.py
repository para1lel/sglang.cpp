#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import tempfile


BUNDLE = Path(__file__).resolve().parent
MANIFEST_SHA256 = "d1893b396b4049919ee741261b00a807d132ab44fdbde89e5854eeb7b2533268"
VERIFIED = ".verified-replay.json"
GIT_ENV = {
    **{key: value for key, value in os.environ.items() if not key.startswith("GIT_")},
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_OPTIONAL_LOCKS": "0",
    "GIT_NO_REPLACE_OBJECTS": "1",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def load_manifest(seal):
    checksums = (BUNDLE / "SHA256SUMS").read_bytes()
    require(sha256(checksums) == seal, "Bundle seal mismatch")
    expected = {}
    for line in checksums.decode().splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  ([a-zA-Z0-9._/-]+)", line)
        require(match is not None, "Invalid SHA256SUMS record")
        digest, name = match.groups()
        require(name not in expected and name != "SHA256SUMS" and
                not Path(name).is_absolute() and ".." not in Path(name).parts and
                Path(name).as_posix() == name, f"Invalid bundle path: {name}")
        expected[name] = digest
    observed = set()
    for path in BUNDLE.rglob("*"):
        mode = path.lstat().st_mode
        if stat.S_ISDIR(mode):
            continue
        name = path.relative_to(BUNDLE).as_posix()
        require(stat.S_ISREG(mode), f"Expected a regular bundle file: {name}")
        observed.add(name)
    require(observed == expected.keys() | {"SHA256SUMS"}, "Bundle file set mismatch")
    for name, digest in expected.items():
        require(sha256((BUNDLE / name).read_bytes()) == digest, f"Bundle checksum mismatch: {name}")
    manifest = (BUNDLE / "manifest.json").read_bytes()
    require(sha256(manifest) == MANIFEST_SHA256, "Frozen ordered manifest mismatch")
    return json.loads(manifest)


def git(directory, *arguments, environment=GIT_ENV, **kwargs):
    return subprocess.check_output(["git", "-C", directory, *arguments], env=environment, **kwargs)


def revision_tree(repo, revision):
    return git(repo, "rev-parse", "--verify", f"{revision}^{{tree}}").decode().strip()


def archive(repo, revision, tree):
    tree.mkdir()
    with subprocess.Popen(
        ["git", "-C", repo, "archive", revision], env=GIT_ENV, stdout=subprocess.PIPE
    ) as process:
        subprocess.run(["tar", "-xf", "-", "-C", tree], stdin=process.stdout, check=True)
        process.stdout.close()
        require(process.wait() == 0, f"git archive failed for {revision}")


def patch_bytes(repo, patch):
    path = patch["file"]
    if "commit" in patch:
        content = git(repo, "show", f"{patch['commit']}:{path}")
    else:
        content = (BUNDLE / path).read_bytes()
    require(sha256(content) == patch["sha256"], f"Patch hash mismatch for {path}")
    return content


def apply(tree, patch, content):
    options = ["-c", "apply.ignoreWhitespace=no", "apply", f"-p{patch.get('strip', 1)}"]
    environment = {**GIT_ENV, "GIT_CEILING_DIRECTORIES": str(tree.parent)}
    arguments = {"input": content, "environment": environment}
    rows = git(tree, *options, "--numstat", "-", **arguments).decode().splitlines()
    fields = [row.split("\t") for row in rows]
    require(all(len(row) == 3 and all(count.isdigit() for count in row[:2])
                for row in fields), "Expected a text-only patch")
    paths = sorted(row[2] for row in fields)
    require(paths == patch["paths"], f"Patch path mismatch for {patch['file']}")
    require(all(not Path(path).is_absolute() and ".." not in Path(path).parts
                for path in paths), "Unsafe patch path")
    git(tree, *options, "--check", "-", **arguments)
    git(tree, *options, "--whitespace=nowarn", "-", **arguments)
    return paths


def inventory(tree, roots, omit):
    entries = {}
    for root in roots:
        path = tree / root
        require(path.exists(), f"Source input missing: {root}")
        for source in [path, *path.rglob("*")] if path.is_dir() else [path]:
            name = source.relative_to(tree).as_posix()
            if name in omit:
                continue
            mode = source.lstat().st_mode
            if stat.S_ISDIR(mode):
                continue
            require(stat.S_ISREG(mode), f"Expected a regular file: {name}")
            git_mode = "100755" if mode & stat.S_IXUSR else "100644"
            entries[name] = f"{git_mode} {sha256(source.read_bytes())}  {name}\n"
    return "".join(entries[name] for name in sorted(entries)).encode()


def check_sources(tree, name, series):
    receipt = (BUNDLE / series["sources"]).read_bytes()
    require(sha256(receipt) == series["sources_sha256"], f"Receipt hash mismatch for {name}")
    observed = inventory(tree, series["roots"], series.get("omit", []))
    require(observed == receipt, f"Source bytes, modes or paths differ for {name}")
    return len(observed.splitlines())


def verification_record(tree, name, series, seal):
    return {
        "series": name,
        "bundle_seal": seal,
        "sources_sha256": series["sources_sha256"],
        "tree_sha256": sha256(inventory(tree, ["."], {VERIFIED})),
    }


def check_tree(tree, name, seal):
    manifest = load_manifest(seal)
    require(name in manifest["series"], f"Unknown replay series: {name}")
    series = manifest["series"][name]
    marker = tree / VERIFIED
    require(marker.is_file() and not marker.is_symlink(), f"Missing verification record: {tree}")
    require(json.loads(marker.read_bytes()) == verification_record(tree, name, series, seal),
            f"Verified tree changed: {tree}")
    check_sources(tree, name, series)


def verify(repo, manifest, seal, selected, output):
    anchor = manifest["publication_anchor"]
    require(revision_tree(repo, anchor) == manifest["anchor_tree"], "Anchor tree mismatch")
    old = manifest["historical_checkpoint"]
    git(repo, "merge-base", "--is-ancestor", old["commit"], anchor)
    old_tree = git(repo, "rev-parse", "--verify", f"{old['commit']}:{old['directory']}")
    require(old_tree.decode().strip() == old["tree"], "Historical checkpoint tree mismatch")
    for name in selected:
        series = manifest["series"][name]
        base = series["base"]
        git(repo, "merge-base", "--is-ancestor", base, anchor)
        require(revision_tree(repo, base) == series["base_tree"], f"Base tree mismatch for {name}")
        tree = output / name
        archive(repo, base, tree)
        for step in series["apply"]:
            patch = manifest["patches"][step]
            apply(tree, patch, patch_bytes(repo, patch))
        count = check_sources(tree, name, series)
        marker = tree / VERIFIED
        with marker.open("x") as stream:
            json.dump(verification_record(tree, name, series, seal), stream, sort_keys=True)
            stream.write("\n")
        yield f"{name}: {count} source modes and SHA-256 hashes verified"


def main():
    parser = argparse.ArgumentParser(description="Replay the experimental source checkpoint.")
    parser.add_argument("--seal", required=True, help="Published SHA-256 of SHA256SUMS")
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path,
                        help="Keep verified trees in a new external directory")
    parser.add_argument("--check-tree", type=Path, help="Validate a previously replayed tree")
    parser.add_argument("series", nargs="*", metavar="SERIES",
                        help="Defaults to all series in manifest.json")
    args = parser.parse_args()
    repo = args.repo.resolve()
    try:
        if args.check_tree:
            require(args.output is None and len(args.series) == 1,
                    "Tree validation requires exactly one series and no --output")
            check_tree(args.check_tree.resolve(), args.series[0], args.seal)
            return
        manifest = load_manifest(args.seal)
        selected = args.series or list(manifest["series"])
        unique = set(selected)
        require(len(selected) == len(unique) and unique <= manifest["series"].keys(),
                "Select unique series from manifest.json")
        destination = args.output.parent.resolve() / args.output.name if args.output else None
        if destination:
            require(not destination.is_relative_to(repo) and not destination.is_relative_to(BUNDLE),
                    "Output must be outside the clone and bundle")
            require(not os.path.lexists(destination), f"Output already exists: {destination}")
        parent = destination.parent if destination else None
        with tempfile.TemporaryDirectory(prefix=".sglang-performance-replay-",
                                         dir=parent) as temporary:
            staged = Path(temporary) / "trees"
            staged.mkdir()
            verified = list(verify(repo, manifest, args.seal, selected, staged))
            if destination:
                subprocess.run(["mv", "-T", "-n", "--", staged, destination], check=True)
                require(not staged.exists(), f"Output already exists: {destination}")
        print("\n".join(verified))
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
