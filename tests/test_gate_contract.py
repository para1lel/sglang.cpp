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


def seed_repository(root, files):
    git_fixture(root, "init", "--quiet")
    for name in (".githooks/pre-commit", ".githooks/run"):
        write_fixture(root, name, (ROOT / name).read_text(encoding="utf-8"))
    for helper in CHECKER.parent.glob("*.py"):
        write_fixture(root, helper.relative_to(ROOT), helper.read_text(encoding="utf-8"))
    configuration = json.loads((ROOT / ".agents/gates.json").read_text(encoding="utf-8"))
    configuration["checks"] = []
    write_fixture(root, ".agents/gates.json", json.dumps(configuration, indent=2) + "\n")
    for name, contents in files.items():
        write_fixture(root, name, contents)
    git_fixture(root, "add", "--all")
    git_fixture(
        root, "-c", "core.hooksPath=/dev/null", "-c", "user.name=Gate fixture",
        "-c", "user.email=fixture@example.invalid", "commit", "-qm", "Seed gate fixture",
    )
    return root


@pytest.fixture
def header_repository(tmp_path):
    return seed_repository(tmp_path, {"src/legacy.h": "#pragma once\n"})


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


@pytest.fixture
def comment_repository(tmp_path):
    root = seed_repository(tmp_path, {
        "legacy.py": "# inherited\nvalue = 1\n",
        "removed.cpp": "int value; // inherited\n",
    })
    index = root / ".git/selected-index"
    shutil.copyfile(root / ".git/index", index)
    return root, index


def assert_comment_rejected(result, name, line=1):
    assert result.returncode != 0, result.stdout + result.stderr
    assert f"{name}: Line {line} contains a comment" in result.stderr, result.stderr


def test_whole_staged_file_and_worktree_disagreement(comment_repository):
    root, index = comment_repository
    name = "legacy.py"
    staged = "# inherited\nvalue = 2\n"
    clean = "value = 2\n"
    write_fixture(root, name, staged)
    git_fixture(root, "add", name, index=index)
    write_fixture(root, name, clean)
    assert_comment_rejected(selected_hook(root, index), name)
    git_fixture(root, "add", name, index=index)
    write_fixture(root, name, staged)
    result = selected_hook(root, index)
    assert result.returncode == 0, result.stdout + result.stderr


def test_shared_rule_change_ignores_untouched_and_deleted_comments(comment_repository):
    root, index = comment_repository
    git_fixture(root, "rm", "--cached", "removed.cpp", index=index)
    write_fixture(root, "AGENTS.md", "# Shared rules\n")
    git_fixture(root, "add", "AGENTS.md", index=index)
    result = selected_hook(root, index)
    assert result.returncode == 0, result.stdout + result.stderr


def test_rename_checks_destination_with_existing_comment(comment_repository):
    root, index = comment_repository
    target = "renamed.cpp"
    git_fixture(root, "mv", "removed.cpp", target, index=index)
    assert_comment_rejected(selected_hook(root, index), target)


COMMENT_CASES = [
    ("code.py", '#!/usr/bin/env python3\n"""# first\n# second"""\n', "value = 0  # inline\n"),
    ("code.cpp", 'auto text = R"tag(// literal\n/* text */)tag";\n', "int x; // inline\n"),
    ("include/sglang/code.h", '#define TEXT "//"\n#pragma once\n', "#define X 1 /* text */\n"),
    ("include/sglang/code.HH", "#pragma once\n", "int x; // inline\n"),
    ("include/sglang/code.cuh", '#pragma once\n', "/* first\nsecond */\n"),
    ("code.cu", '__global__ void work() {}\n', "int x; /* inline */\n"),
    ("code.sh", "cat <<'END'\n# body\nEND\n", "echo ok # inline\n"),
    (".githooks/custom", "#!/bin/sh\nprintf '%s' '# quoted'\n", "# real comment\n"),
    ("tools/run", '#!/usr/bin/env bash\necho "${value##*/}" "$#"\n', "# real comment\n"),
    ("xmake.lua", "local s = [=[--[[literal]]]=]\n", "--[==[ first\nsecond ]==]\n"),
    ("config.yaml", "name: 'value # text'\nbody: |\n  # literal\n", "a: 1 # inline\n"),
    ("config.toml", 'name = """first\n# literal"""\n', "a = 1 # inline\n"),
    ("tools/split-env", "#!/usr/bin/env -S bash -eu\nvalue=$((16#ff))\n", "# real comment\n"),
]


@pytest.mark.parametrize(
    "name,clean,comment", COMMENT_CASES, ids=[case[0] for case in COMMENT_CASES],
)
def test_staged_comment_syntax(comment_repository, name, clean, comment):
    root, index = comment_repository
    write_fixture(root, name, clean)
    git_fixture(root, "add", name, index=index)
    valid = selected_hook(root, index)
    assert valid.returncode == 0, valid.stdout + valid.stderr
    write_fixture(root, name, clean + comment)
    git_fixture(root, "add", name, index=index)
    assert_comment_rejected(selected_hook(root, index), name, clean.count("\n") + 1)


@pytest.fixture
def comment_errors(monkeypatch):
    monkeypatch.syspath_prepend(CHECKER.parent)
    import comment_rules

    return comment_rules.comment_errors


def test_other_recognized_languages_use_comment_parsers(comment_repository, comment_errors):
    root, index = comment_repository
    sources = {
        "example.go": ('package main\nvar s = `// text`\n', "// comment\n"),
        "example.java": ('class C { String s = "// text"; }\n', "/* comment */\n"),
        "example.js": ('const s = `// text`;\n', "// comment\n"),
        "example.jsx": ('const e = <span>/* text */</span>;\n', "/* comment */\n"),
        "example.mjs": ('export const s = "// text";\n', "// comment\n"),
        "example.ts": ('const s: string = "// text";\n', "/* comment */\n"),
        "example.tsx": ('const e = <span>/* text */</span>;\n', "// comment\n"),
        "example.rs": ('const S: &str = r#"/* text */"#;\n', "/* outer /* inner */ */\n"),
        "example.ps1": ("$s = 'literal # text'\n", "<# first\nsecond #>\n"),
    }
    for name in sources:
        write_fixture(root, name, "")
    git_fixture(root, "add", "--", *sources, index=index)
    valid = selected_hook(root, index)
    assert valid.returncode == 0, valid.stdout + valid.stderr
    for name, (source, comment) in sources.items():
        write_fixture(root, name, source)
        assert comment_errors(root / name, True) == []
        write_fixture(root, name, source + comment)
    git_fixture(root, "add", "--", *sources, index=index)
    result = selected_hook(root, index)
    for name, (source, _) in sources.items():
        line = source.count("\n") + 1
        assert_comment_rejected(result, name, line)
        for earlier_line in range(1, line):
            assert f"{name}: Line {earlier_line} contains a comment" not in result.stderr


def test_cpp_macro_literals_and_line_splicing(tmp_path, comment_errors):
    path = tmp_path / "macro.cpp"
    directive = '#define TEXT R"tag(embedded " // literal /* text */)tag"'
    path.write_text(directive + "\n", encoding="utf-8")
    assert comment_errors(path, True) == []
    path.write_text(directive + " // comment\nint x; /\\\n/ spliced comment\n", encoding="utf-8")
    diagnostics = comment_errors(path, True)
    assert len(diagnostics) == 2
    assert diagnostics[0].startswith("Line 1 contains a comment")
    assert diagnostics[1].startswith("Line 2 contains a comment")
