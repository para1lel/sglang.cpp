"""Fast host checks for the rule checker and the isolated selected-index hook."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from types import ModuleType


CHECKER = Path(__file__).resolve().parents[1] / ".agents/checks/check_rules.py"
ROOT = CHECKER.parents[2]


def clean_environment() -> dict[str, str]:
    environment = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    environment.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull)
    return environment


def run_fixture(
    command: list[str | Path],
    *,
    cwd: Path | None = None,
    environment: dict[str, str] | None = None,
    source: str | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        env=clean_environment() if environment is None else environment,
        input=source,
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )


def checker(*arguments: str | Path) -> subprocess.CompletedProcess[str]:
    return run_fixture([sys.executable, "-I", "-B", CHECKER, *arguments])


@pytest.mark.parametrize(
    ("message", "diagnostic"),
    [
        ("feat(api)!: change reply\n\nBody.\n\nBREAKING CHANGE: use v2.\n", None),
        ("fix: repair reply\nBody without separator.\n", "blank line"),
        ("fix: repair reply\n\nbreaking change: use v2.\n", "uppercase"),
    ],
)
def test_message_body_and_footer(tmp_path: Path, message: str, diagnostic: str | None) -> None:
    source = tmp_path / "message"
    source.write_text(message, encoding="utf-8")
    result = checker("commit-message", source)
    if diagnostic is None:
        assert result.returncode == 0, result.stderr
    else:
        assert result.returncode != 0
        assert diagnostic in result.stderr


def test_code_link_detects_drift_at_the_same_line(tmp_path: Path) -> None:
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
        + link.stdout,
        encoding="utf-8",
    )
    result = checker("package", root)
    assert result.returncode == 0, result.stderr
    source.write_text("def other():\n    return 1\n", encoding="utf-8")
    result = checker("package", root)
    assert result.returncode != 0
    assert "example.py#L1" in result.stderr
    assert "prefix is stale" in result.stderr


def git_fixture(root: Path, *arguments: str, index: Path | None = None) -> str:
    environment = clean_environment()
    if index is not None:
        environment["GIT_INDEX_FILE"] = str(index)
    result = run_fixture(["git", "-C", root, *arguments], environment=environment)
    assert result.returncode == 0, result.stderr
    return result.stdout


def write_fixture(root: Path, name: str | Path, contents: str) -> None:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")


def stage_files(root: Path, files: dict[str, str], *, index: Path | None = None) -> None:
    for name, contents in files.items():
        write_fixture(root, name, contents)
    git_fixture(root, "add", "--", *files, index=index)


def copy_index(root: Path) -> Path:
    index = root / ".git/selected-index"
    shutil.copyfile(root / ".git/index", index)
    return index


def seed_repository(
    root: Path, files: dict[str, str], *, style: bool = False, commit: bool = True
) -> Path:
    root.mkdir(exist_ok=True)
    git_fixture(root, "init", "--quiet", "--template=")
    git_fixture(root, "config", "core.hooksPath", ".githooks")
    names = [".githooks/pre-commit", ".githooks/run"]
    if style:
        names.extend([".clang-format", ".clang-tidy", "ruff.toml"])
    for name in names:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    for helper in CHECKER.parent.glob("*.py"):
        write_fixture(root, helper.relative_to(ROOT), helper.read_text(encoding="utf-8"))
    configuration = json.loads((ROOT / ".agents/gates.json").read_text(encoding="utf-8"))
    configuration["checks"] = [
        check
        for check in configuration["checks"]
        if style and check["name"] == "Staged source lint and formatting"
    ]
    if style:
        assert len(configuration["checks"]) == 1
    write_fixture(root, ".agents/gates.json", json.dumps(configuration, indent=2) + "\n")
    for name, contents in files.items():
        write_fixture(root, name, contents)
    git_fixture(root, "add", "--all")
    if commit:
        git_fixture(
            root,
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "user.name=Gate fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            "Seed gate fixture",
        )
    return root


@pytest.fixture
def header_repository(tmp_path: Path) -> Path:
    return seed_repository(tmp_path, {"src/legacy.h": "inline int value() { return 7; }\n"})


def selected_hook(
    root: Path,
    index: Path | None = None,
    *,
    path: str | Path | None = None,
    host_root: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    environment = clean_environment()
    if index is not None:
        environment["GIT_INDEX_FILE"] = str(index)
    if path is not None:
        environment["PATH"] = str(path)
        environment.pop("CONDA_PREFIX", None)
        environment.pop("SGLANG_HOST_ROOT", None)
    if host_root is not None:
        environment["SGLANG_HOST_ROOT"] = str(host_root)
    indexes = {root / ".git/index"}
    if index is not None:
        indexes.add(index)
    original = {name: name.read_bytes() for name in indexes}
    worktree = worktree_files(root)
    result = run_fixture(
        ["git", "-C", root, "hook", "run", "pre-commit"], cwd=root.parent, environment=environment
    )
    assert {name: name.read_bytes() for name in indexes} == original
    assert worktree_files(root) == worktree
    return result


def worktree_files(root: Path) -> dict[Path, tuple[int, bytes | None]]:
    paths = {}
    for child in root.iterdir():
        if child.name == ".git":
            continue
        for path in (child, *child.rglob("*")):
            contents = path.read_bytes() if path.is_file() else None
            paths[path.relative_to(root)] = path.stat().st_mode, contents
    return paths


def assert_passed(result: subprocess.CompletedProcess[str]) -> None:
    assert result.returncode == 0, result.stdout + result.stderr


def assert_rejected(result: subprocess.CompletedProcess[str], *diagnostics: str) -> str:
    output = result.stdout + result.stderr
    assert result.returncode != 0, output
    for diagnostic in diagnostics:
        assert diagnostic in output, output
    return output


def assert_header_rejected(
    result: subprocess.CompletedProcess[str], path: str, line: int = 1
) -> None:
    assert result.returncode != 0, result.stdout + result.stderr
    assert f"{path}: Line {line} contains a function definition." in result.stderr


@pytest.fixture
def alternate_index(header_repository: Path) -> Path:
    index = copy_index(header_repository)
    stage_files(header_repository, {"README.md": "# Updated fixture\n"}, index=index)
    return index


def test_header_rule_rejects_retained_internal_definition(
    header_repository: Path, alternate_index: Path
) -> None:
    assert_header_rejected(selected_hook(header_repository, alternate_index), "src/legacy.h")


def test_header_rule_ignores_generated_and_unselected_headers(
    header_repository: Path, alternate_index: Path
) -> None:
    root = header_repository
    git_fixture(root, "rm", "--cached", "src/legacy.h", index=alternate_index)
    write_fixture(root, "build/generated.hpp", "int generated() { return 7; }\n")
    assert_passed(selected_hook(root, alternate_index))
    assert (root / "src/legacy.h").is_file()


@pytest.mark.parametrize("path", ["src/new.cuh", "tests/probe.ipp", "tools/probe.tpp"])
def test_header_rule_accepts_internal_declarations(
    header_repository: Path, alternate_index: Path, path: str
) -> None:
    root = header_repository
    git_fixture(root, "rm", "--cached", "src/legacy.h", index=alternate_index)
    stage_files(root, {path: "#pragma once\nint declared();\n"}, index=alternate_index)
    assert_passed(selected_hook(root, alternate_index))
    assert (root / "src/legacy.h").is_file()


@pytest.fixture
def comment_repository(tmp_path: Path) -> tuple[Path, Path]:
    root = seed_repository(
        tmp_path,
        {"legacy.py": "# inherited\nvalue = 1\n", "removed.cpp": "int value; // inherited\n"},
    )
    return root, copy_index(root)


def assert_comment_rejected(
    result: subprocess.CompletedProcess[str], name: str, line: int = 1
) -> None:
    assert result.returncode != 0, result.stdout + result.stderr
    assert f"{name}: Line {line} contains a comment" in result.stderr, result.stderr


def test_whole_staged_file_and_worktree_disagreement(comment_repository: tuple[Path, Path]) -> None:
    root, index = comment_repository
    name = "legacy.py"
    staged = "# inherited\nvalue = 2\n"
    clean = "value = 2\n"
    stage_files(root, {name: staged}, index=index)
    write_fixture(root, name, clean)
    assert_comment_rejected(selected_hook(root, index), name)
    git_fixture(root, "add", name, index=index)
    write_fixture(root, name, staged)
    assert_passed(selected_hook(root, index))


def test_shared_rule_change_ignores_untouched_and_deleted_comments(
    comment_repository: tuple[Path, Path],
) -> None:
    root, index = comment_repository
    git_fixture(root, "rm", "--cached", "removed.cpp", index=index)
    stage_files(root, {"AGENTS.md": "# Shared rules\n"}, index=index)
    assert_passed(selected_hook(root, index))


def test_rename_checks_destination_with_existing_comment(
    comment_repository: tuple[Path, Path],
) -> None:
    root, index = comment_repository
    target = "renamed.cpp"
    git_fixture(root, "mv", "removed.cpp", target, index=index)
    assert_comment_rejected(selected_hook(root, index), target)


COMMENT_CASES = [
    ("code.py", '#!/usr/bin/env python3\n"""# first\n# second"""\n', "value = 0  # inline\n"),
    ("code.cpp", 'auto text = R"tag(// literal\n/* text */)tag";\n', "int x; // inline\n"),
    ("include/sglang/code.h", '#define TEXT "//"\n#pragma once\n', "#define X 1 /* text */\n"),
    ("include/sglang/code.HH", "#pragma once\n", "int x; // inline\n"),
    ("include/sglang/code.cuh", "#pragma once\n", "/* first\nsecond */\n"),
    ("code.cu", "__global__ void work() {}\n", "int x; /* inline */\n"),
    ("code.sh", "cat <<'END'\n# body\nEND\n", "echo ok # inline\n"),
    (".githooks/custom", "#!/bin/sh\nprintf '%s' '# quoted'\n", "# real comment\n"),
    ("tools/run", '#!/usr/bin/env bash\necho "${value##*/}" "$#"\n', "# real comment\n"),
    ("xmake.lua", "local s = [=[--[[literal]]]=]\n", "--[==[ first\nsecond ]==]\n"),
    ("config.yaml", "name: 'value # text'\nbody: |\n  # literal\n", "a: 1 # inline\n"),
    ("config.toml", 'name = """first\n# literal"""\n', "a = 1 # inline\n"),
    ("tools/split-env", "#!/usr/bin/env -S bash -eu\nvalue=$((16#ff))\n", "# real comment\n"),
]


@pytest.mark.parametrize(
    ("name", "clean", "comment"), COMMENT_CASES, ids=[case[0] for case in COMMENT_CASES]
)
def test_staged_comment_syntax(
    comment_repository: tuple[Path, Path], name: str, clean: str, comment: str
) -> None:
    root, index = comment_repository
    stage_files(root, {name: clean}, index=index)
    assert_passed(selected_hook(root, index))
    stage_files(root, {name: clean + comment}, index=index)
    assert_comment_rejected(selected_hook(root, index), name, clean.count("\n") + 1)


@pytest.fixture
def comment_parser(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(CHECKER.parent)
    return import_module("comment_rules")


def test_other_recognized_languages_use_comment_parsers(
    comment_repository: tuple[Path, Path], comment_parser: ModuleType
) -> None:
    root, index = comment_repository
    sources = {
        "example.go": ("package main\nvar s = `// text`\n", "// comment\n"),
        "example.java": ('class C { String s = "// text"; }\n', "/* comment */\n"),
        "example.js": ("const s = `// text`;\n", "// comment\n"),
        "example.jsx": ("const e = <span>/* text */</span>;\n", "/* comment */\n"),
        "example.mjs": ('export const s = "// text";\n', "// comment\n"),
        "example.ts": ('const s: string = "// text";\n', "/* comment */\n"),
        "example.tsx": ("const e = <span>/* text */</span>;\n", "// comment\n"),
        "example.rs": ('const S: &str = r#"/* text */"#;\n', "/* outer /* inner */ */\n"),
        "example.ps1": ("$s = 'literal # text'\n", "<# first\nsecond #>\n"),
    }
    stage_files(root, dict.fromkeys(sources, ""), index=index)
    assert_passed(selected_hook(root, index))
    for name, (source, _) in sources.items():
        write_fixture(root, name, source)
        assert comment_parser.comment_errors(root / name, code=True) == []
    dirty_sources = {name: source + comment for name, (source, comment) in sources.items()}
    stage_files(root, dirty_sources, index=index)
    result = selected_hook(root, index)
    for name, (source, _) in sources.items():
        line = source.count("\n") + 1
        assert_comment_rejected(result, name, line)
        for earlier_line in range(1, line):
            assert f"{name}: Line {earlier_line} contains a comment" not in result.stderr


def test_cpp_macro_literals_and_line_splicing(tmp_path: Path, comment_parser: ModuleType) -> None:
    path = tmp_path / "macro.cpp"
    directive = '#define TEXT R"tag(embedded " // literal /* text */)tag"'
    path.write_text(directive + "\n", encoding="utf-8")
    assert comment_parser.comment_errors(path, code=True) == []
    path.write_text(directive + " // comment\nint x; /\\\n/ spliced comment\n", encoding="utf-8")
    diagnostics = comment_parser.comment_errors(path, code=True)
    expected = ("Line 1 contains a comment", "Line 2 contains a comment")
    for diagnostic, prefix in zip(diagnostics, expected, strict=True):
        assert diagnostic.startswith(prefix)


@pytest.mark.parametrize(
    ("extension", "line", "source"),
    [
        ("h", 1, "inline int value() { return 7; }\n"),
        ("hh", 1, "template<class T> T value(T x) { return x; }\n"),
        ("hpp", 1, "struct Probe { Probe() : value(7) {} int value; };\n"),
        ("hxx", 1, "struct Probe { Probe() try {} catch (...) {} };\n"),
        ("h++", 1, "struct Probe { ~Probe() {} };\n"),
        ("cuh", 1, "__device__ int value() { return 7; }\n"),
        ("inl", 1, "struct Probe { explicit operator bool() const { return true; } };\n"),
        ("ipp", 1, "inline int value() { return 7; }\n"),
        ("tpp", 1, "template<class T> T value(T x) { return x; }\n"),
        ("hpp", 1, "inline constexpr auto value = [] { return 7; };\n"),
        ("h", 1, "struct Probe { Probe() = default; };\n"),
        ("cuh", 2, "#define DEVICE __device__\nDEVICE int value() { return 7; }\n"),
        ("h", 2, "#define BODY() \\\n    inline int value() { return 7; }\n"),
        ("h", 1, "#define CALLBACK [] { return 7; }\n"),
        (
            "hpp",
            2,
            (
                "struct Probe { Probe& operator=(const Probe&); };\n"
                "Probe& Probe::operator=(const Probe&) = default;\n"
            ),
        ),
        ("h", 1, 'void unavailable() = delete("unsupported");\n'),
        ("hpp", 2, "#if 0\nint inactive() { return 7; }\n#endif\n"),
        ("hpp", 2, "#if 0\nauto inactive = [] { return 7; };\n#endif\n"),
        ("hpp", 2, "#define BODY { return 7; }\nint active() BODY\n"),
        ("hpp", 2, "namespace {\nint anonymous() { return 7; }\n}\n"),
        ("hpp", 1, "static int private_function() { return 7; }\n"),
        ("hpp", 1, "constexpr int constant() { return 7; }\n"),
    ],
)
def test_staged_headers_reject_function_definitions(
    comment_repository: tuple[Path, Path], extension: str, line: int, source: str
) -> None:
    root, index = comment_repository
    path = f"tests/fixtures/probe.{extension}"
    stage_files(root, {path: source}, index=index)
    assert_header_rejected(selected_hook(root, index), path, line)


def test_header_rule_uses_staged_blob(comment_repository: tuple[Path, Path]) -> None:
    root, index = comment_repository
    path = "include/sglang/probe.hpp"
    declaration = (
        "template<class T> T identity(T);\n"
        "struct Probe { Probe(); Probe(const Probe&) = delete; virtual void run() = 0; };\n"
        "void unavailable(int) = delete;\n"
        "inline int (*probe)(int){nullptr};\n"
        "inline constexpr int data[] = {1, 2, 3};\n"
    )
    definition = "inline int value() { return 7; }\n"
    stage_files(root, {path: declaration}, index=index)
    write_fixture(root, path, definition)
    assert_passed(selected_hook(root, index))

    git_fixture(root, "add", path, index=index)
    write_fixture(root, path, declaration)
    assert_header_rejected(selected_hook(root, index), path)


def test_cuda_header_rule_uses_staged_qualifier_macro(
    comment_repository: tuple[Path, Path],
) -> None:
    root, index = comment_repository
    qualifiers = "src/probe/qualifiers.cuh"
    probe = "src/probe/function.cuh"
    declaration = '#include "qualifiers.cuh"\nQUAL int probe();\n'
    qualifier_source = "#ifdef __CUDACC__\n#define QUAL __device__\n#else\n#define QUAL\n#endif\n"
    stage_files(root, {qualifiers: qualifier_source, probe: declaration}, index=index)
    write_fixture(root, qualifiers, "#define QUAL unexpected_token\n")
    assert_passed(selected_hook(root, index))

    definition = '#include "qualifiers.cuh"\nQUAL int probe() { return 0; }\n'
    stage_files(root, {probe: definition}, index=index)
    write_fixture(root, probe, declaration)
    assert_header_rejected(selected_hook(root, index), probe, 2)


@pytest.fixture
def style_repository(tmp_path: Path) -> Path:
    return seed_repository(tmp_path, {}, style=True)


ROOT_SOURCES = {"root.cpp": "int main() {\n    return 0;\n}\n", "root.py": "value = 42\n"}


STYLE_CASES = [
    (
        "root.c",
        "int main(void) {\n    return 0;\n}\n",
        "int main(void){\n    return 0;\n}\n",
        "int broken( ;\n",
    ),
    (
        "root.cpp",
        "int main() {\n    return 0;\n}\n",
        "int main(){\n    return 0;\n}\n",
        "int broken( ;\n",
    ),
    (
        "root.cu",
        "[[maybe_unused]] static __global__ void sync_threads() {\n    __syncthreads();\n}\n",
        "[[maybe_unused]] static __global__ void sync_threads(){\n    __syncthreads();\n}\n",
        "__global__ void broken( ;\n",
    ),
    ("root.py", "value = 42\n", "value=42\n", "def broken(:\n"),
]


@pytest.mark.parametrize(
    ("name", "clean", "dirty", "malformed"), STYLE_CASES, ids=[case[0] for case in STYLE_CASES]
)
def test_staged_format_and_syntax(
    style_repository: Path, name: str, clean: str, dirty: str, malformed: str
) -> None:
    root = style_repository
    index = copy_index(root)
    stage_files(root, {name: clean}, index=index)
    write_fixture(root, name, dirty)
    assert_passed(selected_hook(root, index))

    git_fixture(root, "add", "--", name, index=index)
    write_fixture(root, name, clean)
    assert_format_rejected(selected_hook(root, index), name)
    stage_files(root, {name: malformed}, index=index)
    write_fixture(root, name, clean)
    output = assert_rejected(selected_hook(root, index), name)
    assert "syntax" in output.lower() or "clang-diagnostic-error" in output, output


def assert_format_rejected(result: subprocess.CompletedProcess[str], name: str) -> None:
    assert_rejected(result, f"Format {name} from the repository root with:")


@pytest.mark.parametrize(("name", "dirty"), [("old.cpp", "int old=42;\n"), ("old.py", "old=42\n")])
def test_root_rename_checks_destination_format(tmp_path: Path, name: str, dirty: str) -> None:
    root = seed_repository(tmp_path, {name: dirty}, style=True)
    stage_files(root, {"new.py": "new = 42\n"})
    assert_passed(selected_hook(root))

    renamed = "renamed" + Path(name).suffix
    git_fixture(root, "mv", name, renamed)
    staged_names = git_fixture(root, "diff", "--cached", "--name-status", "-M")
    assert f"R100\t{name}\t{renamed}" in staged_names
    assert_format_rejected(selected_hook(root), renamed)


def test_initial_index_checks_root_files(tmp_path: Path) -> None:
    root = seed_repository(tmp_path, {"root.py": "value=42\n"}, style=True, commit=False)
    assert_format_rejected(selected_hook(root), "root.py")


def test_shared_policy_scan_skips_unrelated_path_only_check(tmp_path: Path) -> None:
    configuration = json.loads((ROOT / ".agents/gates.json").read_text(encoding="utf-8"))
    configuration["checks"] = [
        {
            "name": "Full-policy sentinel",
            "kind": "static",
            "paths": ["src/*.cpp"],
            "command": ["{python}", "-c", "print('full policy check reached')"],
        },
        {
            "name": "Path-only sentinel",
            "kind": "static",
            "paths": ["src/*.cpp"],
            "path_only": True,
            "command": ["{python}", "-c", "raise SystemExit('path-only check reached')"],
        },
    ]
    root = seed_repository(
        tmp_path, {".agents/gates.json": json.dumps(configuration, indent=2) + "\n"}
    )
    stage_files(root, {"AGENTS.md": "# Policy fixture\n"})
    result = selected_hook(root)
    output = result.stdout + result.stderr
    assert_passed(result)
    assert "full policy check reached" in output, output
    assert "path-only check reached" not in output, output

    stage_files(root, {"src/probe.cpp": "int value = 42;\n"})
    assert_rejected(selected_hook(root), "path-only check reached")


@pytest.mark.parametrize(
    ("name", "opening", "entry", "closing"),
    [
        (
            "multi.cpp",
            "int main() {\n    int value = 0;\n",
            "    value += {};\n",
            "    return value;\n}\n",
        ),
        ("multi.py", "", "entry_{0} = {0}\n", ""),
    ],
)
def test_format_checks_separate_staged_hunks(
    tmp_path: Path, name: str, opening: str, entry: str, closing: str
) -> None:
    lines = [entry.format(i) for i in range(1, 26)]
    lines[12] = lines[12].replace("= ", "=")
    root = seed_repository(tmp_path, {name: opening + "".join(lines) + closing}, style=True)
    lines.insert(0, entry.format(26))
    lines[-1] = entry.format(27)
    stage_files(root, {name: opening + "".join(lines) + closing})
    assert_passed(selected_hook(root))

    lines[-1] = lines[-1].replace("= ", "=")
    stage_files(root, {name: opening + "".join(lines) + closing})
    assert_format_rejected(selected_hook(root), name)


def test_tool_configuration_is_read_from_selected_index(style_repository: Path) -> None:
    root = style_repository
    index = copy_index(root)
    stage_files(root, ROOT_SOURCES, index=index)
    write_fixture(root, ".clang-format", "UnknownStyleOption: true\n")
    write_fixture(root, ".clang-tidy", "UnknownLintOption: true\n")
    write_fixture(root, "ruff.toml", "unknown-style-option = true\n")
    write_fixture(root, ".githooks/run", "exit 77\n")
    assert_passed(selected_hook(root, index))


def test_real_hook_discovers_pinned_tools_from_system_path(style_repository: Path) -> None:
    stage_files(style_repository, ROOT_SOURCES)
    assert_passed(selected_hook(style_repository, path="/usr/bin:/bin"))


def test_required_tool_missing_from_hook_path(tmp_path: Path) -> None:
    root = seed_repository(tmp_path / "source", {}, style=True)
    binaries = tmp_path / "bin"
    host_root = tmp_path / "missing-tools"
    binaries.mkdir()
    for command in ("git", "sh", "cp", "cmp", "mktemp", "rm"):
        executable = shutil.which(command)
        assert executable is not None, command
        (binaries / command).symlink_to(executable)
    python = host_root / ".pixi/envs/default/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text(f'#!/bin/sh\nexec {shlex.quote(sys.executable)} "$@"\n', encoding="utf-8")
    python.chmod(0o755)

    stage_files(root, {"README.md": "# Tool fixture\n"})
    assert_passed(selected_hook(root, path=binaries, host_root=host_root))

    stage_files(root, {"root.cpp": ROOT_SOURCES["root.cpp"]})
    assert_rejected(selected_hook(root, path=binaries, host_root=host_root), "clang-format")


@pytest.mark.parametrize(
    ("name", "source", "diagnostic"),
    [
        ("undefined.py", "value = undefined_name\n", "F821"),
        ("mapping.py", 'value = {"key": 1, "key": 2}\n', "F601"),
        (
            "default.py",
            "def consume(values: list[int] = []) -> int:\n    return len(values)\n",
            "B006",
        ),
        (
            "async_io.py",
            "import time\n\n\nasync def poll() -> None:\n    time.sleep(1)\n",
            "ASYNC251",
        ),
    ],
)
def test_staged_python_static_lint(
    style_repository: Path, name: str, source: str, diagnostic: str
) -> None:
    stage_files(style_repository, {name: source})
    assert_rejected(selected_hook(style_repository), name, diagnostic)


def test_staged_valid_python_fixture(style_repository: Path) -> None:
    source = (
        '"""Workload documents with # inside a literal."""\n'
        "\nimport json\nfrom pathlib import Path\n\nimport pytest\n"
        "\n\n@pytest.fixture\n"
        "def identity(tmp_path: Path) -> Path:\n"
        '    path = tmp_path / "identity.json"\n'
        '    path.write_text(json.dumps({"seed": 7}), encoding="utf-8")\n'
        "    return path\n"
        "\n\ndef test_identity(identity: Path) -> None:\n"
        "    assert isinstance(identity, Path)\n"
        '    assert json.loads(identity.read_text(encoding="utf-8")) == {"seed": 7}\n'
    )
    stage_files(style_repository, {"tests/test_workload.py": source})
    assert_passed(selected_hook(style_repository))


@pytest.mark.parametrize(
    ("name", "before", "removed", "diagnostic"),
    [
        ("binding.py", "value = 7\nanswer = value\n", "value = 7\n", "F821"),
        (
            "guard.cpp",
            (
                "int read() {\n"
                "    int* pointer = nullptr;\n"
                "    if (pointer == nullptr) {\n"
                "        return 0;\n"
                "    }\n"
                "    return *pointer;\n"
                "}\n"
            ),
            "    if (pointer == nullptr) {\n        return 0;\n    }\n",
            "clang-analyzer-core.NullDereference",
        ),
    ],
)
def test_staged_deletion_reports_new_diagnostic_on_untouched_line(
    tmp_path: Path, name: str, before: str, removed: str, diagnostic: str
) -> None:
    root = seed_repository(tmp_path, {name: before}, style=True)
    stage_files(root, {name: before.replace(removed, "")})
    write_fixture(root, name, before)

    assert_rejected(selected_hook(root), name, diagnostic)


def canonical_native_source(root: Path, name: str, source: str) -> None:
    result = run_fixture(["clang-format", f"--assume-filename={name}"], cwd=root, source=source)
    assert_passed(result)
    write_fixture(root, name, result.stdout)
    assert_passed(run_fixture(["clang-format", "--dry-run", "--Werror", name], cwd=root))
    git_fixture(root, "add", name)


def test_staged_cpp_analyzer_rejects_formatted_null_dereference(tmp_path: Path) -> None:
    header = (
        "#pragma once\n#include <filesystem>\n"
        "bool checkpoint_exists(const std::filesystem::path&);\n"
    )
    root = seed_repository(tmp_path, {"checkpoint_probe.h": header}, style=True)
    name = "checkpoint_probe.cpp"
    source = (
        '#include "checkpoint_probe.h"\n'
        "#include <system_error>\n\n"
        "bool checkpoint_exists(const std::filesystem::path& path) {\n"
        "    std::error_code error;\n"
        "    return std::filesystem::is_regular_file(path, error) && !error;\n"
        "}\n"
    )
    canonical_native_source(root, name, source)
    assert_passed(selected_hook(root))

    source += "\nint invalid_metadata() { int* value = nullptr; return *value; }\n"
    canonical_native_source(root, name, source)
    assert_rejected(selected_hook(root), name, "clang-analyzer-core.NullDereference")


def test_staged_cuda_analyzes_device_branch_with_cuda_13_4(tmp_path: Path) -> None:
    header = "#pragma once\nvoid launch_lane(int* output);\n"
    root = seed_repository(tmp_path, {"src/probe_sm100.h": header}, style=True)
    name = "src/probe_sm100.cu"
    guarded_ptx = '    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");\n'
    source = (
        '#include "probe_sm100.h"\n'
        "#include <cuda_fp8.h>\n"
        "#include <cuda_runtime.h>\n\n"
        "static_assert(CUDART_VERSION == 13040);\n\n"
        "static __global__ void store_lane(int* output) {\n"
        "    unsigned int lane = 0;\n"
        '    asm volatile("mov.u32 %0, %%laneid;" : "=r"(lane));\n'
        "#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000\n"
        f"{guarded_ptx}#endif\n"
        "    const __nv_fp8_e4m3 fp8(static_cast<float>(lane));\n"
        "    *output = static_cast<int>(static_cast<float>(fp8));\n"
        "}\n"
        "void launch_lane(int* output) { store_lane<<<1, 1>>>(output); }\n"
    )
    canonical_native_source(root, name, source)
    assert_passed(selected_hook(root))

    defect = "    int* pointer = nullptr;\n    *pointer = static_cast<int>(lane);\n"
    canonical_native_source(root, name, source.replace(guarded_ptx, guarded_ptx + defect))
    assert_rejected(selected_hook(root), name, "clang-analyzer-core.NullDereference")
