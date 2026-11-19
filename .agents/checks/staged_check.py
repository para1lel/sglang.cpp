import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import shutil
import subprocess
import sys
import tempfile

def git(root, arguments, environment, required=True, input_data=None):
    result = subprocess.run(
        ["git", "--no-replace-objects", "--no-optional-locks", "-C", str(root), *arguments],
        env=environment, capture_output=True, check=False, input=input_data,
    )
    if required and result.returncode:
        raise ValueError(result.stderr.decode("utf-8", errors="replace").strip())
    return result.stdout


def freeze(root, temporary):
    environment = os.environ.copy()
    if environment.get("GIT_INDEX_FILE"):
        selected = Path(environment["GIT_INDEX_FILE"])
        environment["GIT_INDEX_FILE"] = str(selected if selected.is_absolute() else root / selected)
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    environment["GIT_NO_REPLACE_OBJECTS"] = "1"
    value = git(root, ["rev-parse", "--git-path", "index"], environment)
    path = Path(os.fsdecode(value).rstrip("\n"))
    index = path if path.is_absolute() else root / path
    original = index.read_bytes() if index.exists() else None
    frozen = temporary / "index"
    if original is not None:
        frozen.write_bytes(original)
    environment["GIT_INDEX_FILE"] = str(frozen)
    value = git(root, ["rev-parse", "--git-path", "objects"], environment)
    object_path = Path(os.fsdecode(value).rstrip("\r\n"))
    object_path = object_path if object_path.is_absolute() else root / object_path
    objects = temporary / "objects"
    (objects / "info").mkdir(parents=True)
    (objects / "info/alternates").write_bytes(
        (object_path.resolve().as_posix() + "\n").encode("utf-8")
    )
    environment["GIT_OBJECT_DIRECTORY"] = str(objects)
    tree = git(root, ["write-tree"], environment).strip()
    listing = git(root, ["ls-tree", "-r", "-z", tree.decode("ascii")], environment)
    entries = {}
    for record in listing.split(b"\0"):
        if not record:
            continue
        metadata, raw_path = record.split(b"\t", 1)
        mode, kind, object_id = metadata.split()
        name = os.fsdecode(raw_path)
        parts = PurePosixPath(name).parts
        if PurePosixPath(name).is_absolute() or PureWindowsPath(name).drive:
            raise ValueError("An indexed path is absolute.")
        if ".." in parts or ".git" in parts or "\\" in name:
            raise ValueError(f"An indexed path is unsafe: {name!r}")
        entries[name] = (mode, kind, object_id)
    changed = git(
        root, ["diff", "--cached", "--name-only", "--no-renames", "-z"], environment,
    )
    changed = [os.fsdecode(name) for name in changed.split(b"\0") if name]
    return entries, changed, environment, index, original


def read_blob(reader, destination, size, expected):
    algorithm = "sha1" if len(expected) == 40 else "sha256"
    identity = hashlib.new(algorithm, f"blob {size}\0".encode("ascii"))
    with destination.open("wb") as output:
        remaining = size
        while remaining:
            chunk = reader.read(min(remaining, 1024 * 1024))
            if not chunk:
                raise ValueError("Git returned an incomplete staged blob.")
            identity.update(chunk)
            output.write(chunk)
            remaining -= len(chunk)
    if reader.read(1) != b"\n" or identity.hexdigest().encode("ascii") != expected:
        raise ValueError("The staged blob does not match its recorded object identity.")


def materialize(root, snapshot, entries, environment):
    with subprocess.Popen(
        ["git", "--no-replace-objects", "-C", str(root), "cat-file", "--batch"],
        env=environment, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
    ) as process:
        try:
            for name, (mode, kind, object_id) in entries.items():
                destination = snapshot / name
                if not destination.resolve().is_relative_to(snapshot):
                    raise ValueError("An indexed resource leaves the snapshot.")
                if mode == b"160000":
                    destination.mkdir(parents=True, exist_ok=True)
                    continue
                if mode not in {b"100644", b"100755", b"120000"} or kind != b"blob":
                    raise ValueError("The selected index has an unsupported entry.")
                destination.parent.mkdir(parents=True, exist_ok=True)
                process.stdin.write(object_id + b"\n")
                process.stdin.flush()
                header = process.stdout.readline().split()
                if len(header) != 3 or header[:2] != [object_id, b"blob"]:
                    raise ValueError(f"Git could not read a staged blob: {name!r}")
                read_blob(process.stdout, destination, int(header[2]), object_id)
                if mode == b"120000":
                    target = destination.read_text(encoding="utf-8")
                    if not (destination.parent / target).resolve().is_relative_to(snapshot):
                        raise ValueError("An indexed symlink leaves the snapshot.")
                    destination.unlink()
                    destination.symlink_to(target)
                elif mode == b"100755":
                    destination.chmod(0o755)
        finally:
            process.stdin.close()
            process.stdout.close()
        if process.wait():
            raise ValueError("Git could not export the selected index.")


def snapshot_environment():
    return {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}


def shared_change(name):
    path = PurePosixPath(name)
    rules = {
        "AGENTS.md", "docs/code-standards.md", "docs/requirements.md",
        ".agents/gates.json",
    }
    configuration = {
        ".editorconfig", ".pre-commit-config.yaml", "pyproject.toml", "setup.cfg", "tox.ini",
    }
    requirements = path.name.startswith("requirements") and path.suffix in {".txt", ".in"}
    return (
        name in rules or path.name in configuration or requirements
        or name.startswith(".githooks/")
        or name.startswith("skills/inject-mode/scripts/")
        or name.startswith(".agents/checks/")
    )


def selected_helpers(root, snapshot):
    source = Path(__file__).resolve().with_name("gate_config.py")
    if source.is_relative_to(root):
        source = snapshot / source.relative_to(root)
    if not source.is_file():
        raise ValueError("Stage the gate's helper before using the repository gate.")
    specification = importlib.util.spec_from_file_location("snapshot_gate_config", source)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def snapshot_repository(root, snapshot, entries, environment, head):
    clean = snapshot_environment()
    object_format = git(root, ["rev-parse", "--show-object-format"], environment).strip()
    git(snapshot, ["init", "--quiet", "--object-format=" + object_format.decode("ascii")], clean)
    object_path = Path(os.fsdecode(git(root, ["rev-parse", "--git-path", "objects"], environment)))
    object_path = Path(str(object_path).rstrip("\r\n"))
    object_path = object_path if object_path.is_absolute() else root / object_path
    alternates = snapshot / ".git/objects/info/alternates"
    alternates.write_bytes((object_path.resolve().as_posix() + "\n").encode("utf-8"))
    listing = b"".join(
        mode + b" " + identity + b"\t" + os.fsencode(name) + b"\0"
        for name, (mode, _, identity) in entries.items()
    )
    git(snapshot, ["update-index", "-z", "--index-info"], clean, input_data=listing)
    if head.strip():
        git(snapshot, ["update-ref", "HEAD", head.strip().decode("ascii")], clean)


def check_header_layout(entries):
    suffixes = {".h", ".hh", ".hpp", ".hxx", ".h++", ".cuh"}
    for name in entries:
        if (
            PurePosixPath(name).suffix.lower() in suffixes
            and not name.startswith("include/sglang/")
        ):
            raise ValueError(f"Move project header {name!r} under include/sglang/.")


def run(root):
    with tempfile.TemporaryDirectory(prefix="meta-harness-staged-") as directory:
        temporary = Path(directory).resolve()
        if not temporary.is_relative_to(Path(tempfile.gettempdir()).resolve()):
            raise ValueError("The temporary snapshot leaves its selected directory.")
        entries, changed, environment, index, original = freeze(root, temporary)
        check_header_layout(entries)
        if not changed:
            print("Passed: the selected index has no changed files.")
            return 0
        head = git(root, ["rev-parse", "--verify", "HEAD"], environment, required=False)
        snapshot = temporary / "source"
        snapshot.mkdir()
        materialize(root, snapshot, entries, environment)
        gate_config = selected_helpers(root, snapshot)
        configuration = gate_config.load(snapshot)
        gate_config.baseline_unchanged(root, configuration, environment)
        clean_environment = snapshot_environment()
        snapshot_repository(root, snapshot, entries, environment, head)
        checker = Path(__file__).resolve().with_name("check_rules.py")
        if checker.is_relative_to(root):
            selected_checker = snapshot / checker.relative_to(root)
            if not selected_checker.is_file():
                raise ValueError("Stage the rule checker before using the repository gate.")
        else:
            selected_checker = temporary / "check_rules.py"
            selected_checker.write_bytes(checker.read_bytes())
            selected_checker.with_name("gate_config.py").write_bytes(
                checker.with_name("gate_config.py").read_bytes()
            )
        full = any(
            shared_change(name) or gate_config.matches(
                name, configuration.get("shared_paths", [])
            )
            for name in changed
        )
        selection = temporary / "selection.json"
        selection.write_text(json.dumps({
            "files": sorted(entries), "changed": changed, "full": full,
        }), encoding="utf-8")
        result = subprocess.run(
            [sys.executable, "-I", str(selected_checker), "check", str(snapshot),
             "--selection", str(selection)],
            cwd=snapshot, env=clean_environment, check=False,
        )
        package = snapshot / ("skills" if (snapshot / "skills").is_dir() else ".agents/skills")
        resources_changed = full or any(
            name.startswith(("skills/", ".agents/skills/")) for name in changed
        )
        if result.returncode == 0 and resources_changed and any(package.glob("*/SKILL.md")):
            result = subprocess.run(
                [sys.executable, "-I", str(selected_checker), "package", str(package),
                 "--policy-root", str(snapshot)],
                cwd=snapshot, env=clean_environment, check=False,
            )
        if result.returncode == 0:
            gate_config.run_checks(snapshot, configuration, changed, full)
        current = index.read_bytes() if index.exists() else None
        if current != original:
            raise ValueError("The caller's index changed during verification.")
        if git(root, ["rev-parse", "--verify", "HEAD"], environment, required=False) != head:
            raise ValueError("HEAD changed during verification.")
        return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Examine a fixed, selected Git index.")
    parser.add_argument("root", nargs="?", default=".")
    arguments = parser.parse_args()
    try:
        return run(Path(arguments.root).resolve())
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
