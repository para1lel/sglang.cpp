import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import NamedTuple, TypedDict

from function_spans import HEADER_SUFFIXES
from gate_config import PYTHON_SUFFIXES
from native_lint import run as native_lint
from staged_check import materialize, read_tree, snapshot_environment
from tool_process import invoke

NATIVE_SUFFIXES = {".c", ".cc", ".cpp", ".cxx", ".cu"} | HEADER_SUFFIXES
GIT = ["git", "--no-replace-objects", "--no-optional-locks", "--literal-pathspecs"]
HUNK_HEADER = re.compile(rb"^@@ -\d+(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", re.MULTILINE)
CLANG_STYLE = "--style=file:.clang-format"
RUFF_STYLE = "--config=ruff.toml"


class Hunk(NamedTuple):
    old_count: int
    new_start: int
    new_count: int


class Position(TypedDict):
    row: int
    column: int


class Diagnostic(TypedDict):
    filename: str
    code: str | None
    name: str
    message: str
    location: Position
    end_location: Position


FileHunks = dict[str, list[Hunk] | None]
LineRange = tuple[int, int]
DiagnosticKey = tuple[str, int, int, int, int]


def git(root: Path, *arguments: str) -> bytes:
    result = invoke([*GIT, *arguments], root)
    result.check_returncode()
    return result.stdout


def staged_hunks(root: Path, name: str) -> list[Hunk]:
    patch = git(
        root,
        "diff",
        "--cached",
        "--no-renames",
        "--no-ext-diff",
        "--no-textconv",
        "--no-color",
        "--text",
        "--diff-algorithm=myers",
        "--no-indent-heuristic",
        "--inter-hunk-context=0",
        "--unified=0",
        "--",
        name,
    )
    return [
        Hunk(int(old_count or 1), int(new), int(new_count or 1))
        for old_count, new, new_count in HUNK_HEADER.findall(patch)
    ]


def staged_sources(root: Path, changed: dict[str, str], suffixes: set[str]) -> FileHunks:
    files = {}
    for name, status in changed.items():
        path = root / name
        if path.suffix in suffixes and path.is_file() and not path.is_symlink():
            files[name] = None if status in {"A", "T"} else staged_hunks(root, name)
    return files


def postimage(hunks: list[Hunk]) -> list[LineRange]:
    return [(start, start + count - 1) for _, start, count in hunks if count]


def previous_row(row: int, hunks: list[Hunk]) -> int:
    offset = 0
    for hunk in hunks:
        first_unchanged = hunk.new_start + (hunk.new_count or 1)
        if row < first_unchanged:
            break
        offset += hunk.old_count - hunk.new_count
    return row + offset


def forward_output(output: bytes, stream: str) -> None:
    destination = sys.stderr if stream == "stderr" else sys.stdout
    destination.buffer.write(output)
    destination.flush()


def check_format(root: Path, name: str, check: list[str], fix: list[str]) -> bool:
    result = invoke([*check, f"./{name}"], root)
    if result.returncode:
        forward_output(result.stdout, "stdout")
        forward_output(result.stderr, "stderr")
        print(f"Format {name} from the repository root with:", file=sys.stderr)
        print(f"  {shlex.join([*fix, f'./{name}'])}", file=sys.stderr)
    return result.returncode == 0


def validate(root: Path, command: list[str]) -> bool:
    result = invoke(command, root)
    if result.returncode:
        forward_output(result.stdout, "stdout")
        forward_output(result.stderr, "stderr")
    return result.returncode == 0


def clang_format(root: Path, files: FileHunks, *, config_changed: bool) -> bool:
    if config_changed and not validate(
        root,
        ["clang-format", CLANG_STYLE, "--assume-filename=input.cpp", "--dump-config"],
    ):
        return False
    passed = True
    for name, hunks in files.items():
        ranges = [] if hunks is None else [f"--lines={a}:{b}" for a, b in postimage(hunks)]
        if hunks is not None and not ranges:
            continue
        arguments = ["clang-format", CLANG_STYLE, *ranges]
        check = [*arguments, "--dry-run", "--Werror"]
        passed = check_format(root, name, check, [*arguments, "-i"]) and passed
    return passed


def ruff_diagnostics(
    root: Path,
    paths: list[str],
    source: bytes | None = None,
) -> list[Diagnostic]:
    arguments = [*paths] if source is None else [f"--stdin-filename={paths[0]}", "-"]
    result = invoke(
        [
            "ruff",
            "check",
            RUFF_STYLE,
            "--output-format=json",
            "--no-cache",
            "--no-fix",
            "--no-fix-only",
            "--ignore-noqa",
            *arguments,
        ],
        root,
        source,
    )
    forward_output(result.stderr, "stderr")
    if result.returncode > 1 or result.returncode < 0:
        result.check_returncode()
    return json.loads(result.stdout)


def original_diagnostics(root: Path, name: str) -> set[DiagnosticKey]:
    source = git(root, "show", f"HEAD:{name}")
    return {diagnostic_key(item, []) for item in ruff_diagnostics(root, [f"./{name}"], source)}


def diagnostic_key(item: Diagnostic, hunks: list[Hunk]) -> DiagnosticKey:
    start = item["location"]
    end = item["end_location"]
    end_row = previous_row(end["row"], hunks)
    if end["column"] == 1 and end["row"] > start["row"]:
        end_row = previous_row(end["row"] - 1, hunks) + 1
    return (
        item["code"] or item["name"],
        previous_row(start["row"], hunks),
        start["column"],
        end_row,
        end["column"],
    )


def overlaps(item: Diagnostic, hunks: list[Hunk]) -> bool:
    start = item["location"]["row"]
    end = item["end_location"]["row"]
    if end > start and item["end_location"]["column"] == 1:
        end -= 1
    return any(lower <= end and start <= upper for lower, upper in postimage(hunks))


def ruff_check(root: Path, files: FileHunks, mode_changes: set[str]) -> bool:
    if not files:
        return True
    originals: dict[str, set[DiagnosticKey]] = {}
    passed = True
    for item in ruff_diagnostics(root, [f"./{name}" for name in files]):
        name = (root / item["filename"]).relative_to(root).as_posix()
        hunks = files[name]
        whole_file = item["code"] is None or (
            name in mode_changes and item["code"] in {"EXE001", "EXE002"}
        )
        if hunks is not None and not whole_file and not overlaps(item, hunks):
            if name not in originals:
                originals[name] = original_diagnostics(root, name)
            if diagnostic_key(item, hunks) in originals[name]:
                continue
        row, column = item["location"]["row"], item["location"]["column"]
        code = item["code"] or item["name"]
        print(f"{name}:{row}:{column}: {code} {item['message']}", file=sys.stderr)
        passed = False
    return passed


def ruff_format(root: Path, files: FileHunks) -> bool:
    passed = True
    for name, hunks in files.items():
        ranges = [""] if hunks is None else [f"--range={a}-{b + 1}" for a, b in postimage(hunks)]
        for selected in ranges:
            arguments = ["ruff", "format", RUFF_STYLE, "--no-cache"]
            if selected:
                arguments.append(selected)
            passed = check_format(root, name, [*arguments, "--check"], arguments) and passed
    return passed


def staged_changes(root: Path) -> tuple[dict[str, str], set[str]]:
    raw = git(root, "diff", "--cached", "--raw", "--no-renames", "-z")
    fields = os.fsdecode(raw).split("\0")[:-1]
    changed = {}
    mode_changes = set()
    for metadata, name in zip(fields[::2], fields[1::2], strict=True):
        old_mode, new_mode, _, _, status = metadata[1:].split()
        changed[name] = status
        if old_mode != new_mode:
            mode_changes.add(name)
    return changed, mode_changes


def check_native(root: Path, files: dict[str, list[LineRange] | None]) -> bool:
    if not files:
        return native_lint(root, files)
    head = invoke([*GIT, "rev-parse", "--verify", "-q", "HEAD"], root)
    if head.returncode == 1:
        return native_lint(root, files)
    head.check_returncode()
    environment = snapshot_environment()
    entries = read_tree(root, head.stdout.decode("ascii").strip(), environment)
    with tempfile.TemporaryDirectory(prefix="meta-harness-head-") as directory:
        baseline = Path(directory).resolve()
        materialize(root, baseline, entries, environment)
        return native_lint(root, files, baseline=baseline)


def run(root: Path) -> bool:
    changed, mode_changes = staged_changes(root)
    native = staged_sources(root, changed, NATIVE_SUFFIXES)
    python = staged_sources(root, changed, PYTHON_SUFFIXES)
    native_format_ok = clang_format(root, native, config_changed=".clang-format" in changed)
    staged = {name: None if hunks is None else postimage(hunks) for name, hunks in native.items()}
    for name, status in changed.items():
        if status == "D" and Path(name).suffix in NATIVE_SUFFIXES:
            staged[name] = []
    native_lint_ok = True
    if staged or ".clang-tidy" in changed or ".agents/checks/native_lint.py" in changed:
        native_lint_ok = check_native(root, staged)
    python_settings_ok = True
    if "ruff.toml" in changed:
        python_settings_ok = validate(
            root,
            ["ruff", "check", RUFF_STYLE, "--show-settings", ".agents/checks/source_style.py"],
        )
    python_lint_ok = python_settings_ok and ruff_check(root, python, mode_changes)
    python_format_ok = python_settings_ok and ruff_format(root, python)
    return all((native_format_ok, native_lint_ok, python_lint_ok, python_format_ok))


def main() -> int:
    try:
        return 0 if run(Path.cwd()) else 1
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
