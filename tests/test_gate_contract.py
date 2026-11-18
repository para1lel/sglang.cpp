"""Fast host checks for the copied rule checker. No hook runs inside these tests."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


CHECKER = Path(__file__).resolve().parents[1] / ".agents/checks/check_rules.py"


def checker(*arguments):
    environment = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    return subprocess.run(
        [sys.executable, "-I", "-B", CHECKER, *arguments], env=environment,
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
