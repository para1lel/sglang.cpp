"""Fast host checks for the rule checker and the isolated selected-index hook."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


CHECKER = Path(__file__).resolve().parents[1] / ".agents/checks/check_rules.py"
ROOT = CHECKER.parents[2]


def clean_environment():
    return {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}


def checker(*arguments):
    return subprocess.run(
        [sys.executable, "-I", "-B", CHECKER, *arguments], env=clean_environment(),
        capture_output=True, text=True, timeout=15,
    )


@pytest.mark.parametrize(
    ("message", "diagnostic"),
    [
        ("feat(api)!: change reply\n\nBody.\n\nBREAKING CHANGE: use v2.\n", None),
        ("fix: repair reply\nBody without separator.\n", "blank line"),
        ("fix: repair reply\n\nbreaking change: use v2.\n", "uppercase"),
    ],
)
def test_message_body_and_footer(tmp_path, message, diagnostic):
    source = tmp_path / "message"
    source.write_text(message, encoding="utf-8")
    result = checker("commit-message", source)
    if diagnostic is None:
        assert result.returncode == 0, result.stderr
    else:
        assert result.returncode != 0
        assert diagnostic in result.stderr


def test_code_link_detects_drift_at_the_same_line(tmp_path):
    root = tmp_path / "skills"
    skill = root / "example"
    policy = skill / "agents/openai.yaml"
    policy.parent.mkdir(parents=True)
    policy.write_text("policy:\n  allow_implicit_invocation: false\n", encoding="utf-8")
    guide = skill / "SKILL.md"
    source = skill / "example.py"
    source.write_text("def first():\n    return 1\n", encoding="utf-8")
    link = checker("code-link", "example/example.py", "1", "--document", guide, "--root", root)
    assert link.returncode == 0, link.stderr
    guide.write_text(
        "---\nname: example\ndescription: Example.\ndisable-model-invocation: true\n---\n"
        + link.stdout, encoding="utf-8",
    )
    result = checker("package", root)
    assert result.returncode == 0, result.stderr
    source.write_text("def other():\n    return 1\n", encoding="utf-8")
    result = checker("package", root)
    assert result.returncode != 0
    assert "example.py#L1" in result.stderr and "prefix is stale" in result.stderr


def git_fixture(root, *arguments, index=None):
    environment = clean_environment()
    if index is not None:
        environment["GIT_INDEX_FILE"] = str(index)
    result = subprocess.run(
        ["git", "-C", root, *arguments], env=environment,
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stderr


def write_fixture(root, name, contents):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")


@pytest.fixture
def header_repository(tmp_path):
    git_fixture(tmp_path, "init", "--quiet")
    files = (
        ".githooks/pre-commit", ".githooks/run", ".agents/checks/check_rules.py",
        ".agents/checks/staged_check.py", ".agents/checks/gate_config.py",
        ".agents/checks/function_spans.py",
    )
    for name in files:
        write_fixture(tmp_path, name, (ROOT / name).read_text(encoding="utf-8"))
    configuration = json.loads((ROOT / ".agents/gates.json").read_text(encoding="utf-8"))
    configuration["checks"] = []
    write_fixture(tmp_path, ".agents/gates.json", json.dumps(configuration))
    write_fixture(tmp_path, "src/legacy.h", "#pragma once\n")
    git_fixture(tmp_path, "add", "--all")
    git_fixture(
        tmp_path, "-c", "core.hooksPath=/dev/null", "-c", "user.name=Gate fixture",
        "-c", "user.email=fixture@example.invalid", "commit", "-qm", "Seed header fixture",
    )
    return tmp_path


def selected_hook(root, index):
    environment = clean_environment()
    environment["GIT_INDEX_FILE"] = str(index)
    original = index.read_bytes()
    default = root / ".git/index"
    original_default = default.read_bytes()
    result = subprocess.run(
        ["/bin/sh", ".githooks/pre-commit"], cwd=root, env=environment,
        capture_output=True, text=True, timeout=15,
    )
    assert index.read_bytes() == original
    assert default.read_bytes() == original_default
    return result


def assert_header_rejected(result, path):
    assert result.returncode != 0, result.stdout + result.stderr
    assert path in result.stderr and "include/" in result.stderr


@pytest.fixture
def alternate_index(header_repository):
    index = header_repository / ".git/selected-index"
    shutil.copyfile(header_repository / ".git/index", index)
    write_fixture(header_repository, "README.md", "# Updated fixture\n")
    git_fixture(header_repository, "add", "README.md", index=index)
    return index


def test_header_location_rejects_retained_header(header_repository, alternate_index):
    result = selected_hook(header_repository, alternate_index)
    assert_header_rejected(result, "src/legacy.h")


@pytest.mark.parametrize(
    "misplaced", ["src/new.cuh", "include/new.hpp", "new.h", "extensions/future/new.HH"],
)
def test_header_location_rejects_staged_header(header_repository, alternate_index, misplaced):
    root = header_repository
    git_fixture(root, "rm", "--cached", "src/legacy.h", index=alternate_index)
    write_fixture(root, misplaced, "#pragma once\n")
    git_fixture(root, "add", misplaced, index=alternate_index)
    (root / misplaced).unlink()
    assert_header_rejected(selected_hook(root, alternate_index), misplaced)


@pytest.mark.parametrize("rename", [False, True], ids=["deletion", "rename"])
def test_header_location_accepts_relocation(header_repository, alternate_index, rename):
    root = header_repository
    git_fixture(root, "rm", "--cached", "src/legacy.h", index=alternate_index)
    if rename:
        allowed = "include/sglang/server/legacy.h"
        write_fixture(root, allowed, "#pragma once\n")
        git_fixture(root, "add", allowed, index=alternate_index)
    result = selected_hook(root, alternate_index)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (root / "src/legacy.h").is_file()
