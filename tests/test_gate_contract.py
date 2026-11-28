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
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from types import ModuleType


ROOT = Path(__file__).resolve().parents[1]
CHECKER = ROOT / ".agents/checks/check_rules.py"
GATE_POLICY = ".agents/gates.json"
EXAMPLE_SKILL = "---\nname: example\ndescription: Example.\ndisable-model-invocation: true\n---\n"
EXPLICIT_SKILL_POLICY = "policy:\n  allow_implicit_invocation: false\n"


def clean_environment(index: Path | None = None) -> dict[str, str]:
    environment = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    environment.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull)
    if index is not None:
        environment["GIT_INDEX_FILE"] = str(index)
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
        assert_passed(result)
    else:
        assert_rejected(result, diagnostic, stderr_only=True)


def test_code_link_detects_drift_at_the_same_line(tmp_path: Path) -> None:
    root = tmp_path / "skills"
    guide = root / "example/SKILL.md"
    write_fixture(root, "example/agents/openai.yaml", EXPLICIT_SKILL_POLICY)
    source = root / "example/example.py"
    write_fixture(root, source, "def first():\n    return 1\n")
    link = checker("code-link", "example/example.py", "1", "--document", guide, "--root", root)
    assert_passed(link)
    write_fixture(root, guide, EXAMPLE_SKILL + link.stdout)
    assert_passed(checker("package", root))
    write_fixture(root, source, "def other():\n    return 1\n")
    assert_rejected(checker("package", root), "example.py#L1", "prefix is stale", stderr_only=True)


def write_fixture(root: Path, name: str | Path, contents: str) -> None:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")


class GateRepository:
    def __init__(self, root: Path, index: Path | None = None) -> None:
        self.root = root
        self.index = index

    @classmethod
    def init(cls, root: Path) -> GateRepository:
        root.mkdir(exist_ok=True)
        repo = cls(root)
        repo.git("init", "--quiet", "--template=")
        return repo

    def alternate(self, *, relative: bool = False) -> GateRepository:
        index = self.root / ".git/selected-index"
        shutil.copyfile(self.root / ".git/index", index)
        return GateRepository(self.root, index.relative_to(self.root) if relative else index)

    def git(self, *arguments: str) -> str:
        result = run_fixture(
            ["git", "-C", self.root, *arguments], environment=clean_environment(self.index)
        )
        assert_passed(result)
        return result.stdout

    def write(self, name: str | Path, contents: str) -> None:
        write_fixture(self.root, name, contents)

    def stage(self, files: dict[str, str]) -> None:
        for name, contents in files.items():
            self.write(name, contents)
        self.git("add", "--", *files)

    def hook(
        self,
        *,
        path: str | Path | None = None,
        host_root: Path | None = None,
        command: list[str | Path] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        environment = clean_environment(self.index)
        if path is not None:
            environment["PATH"] = str(path)
            environment.pop("CONDA_PREFIX", None)
            environment.pop("SGLANG_HOST_ROOT", None)
        if host_root is not None:
            environment["SGLANG_HOST_ROOT"] = str(host_root)
        indexes = {self.root / ".git/index"}
        if self.index is not None:
            indexes.add(self.root / self.index)
        original = {name: name.read_bytes() for name in indexes}
        worktree = worktree_files(self.root)
        if command is None:
            command = ["git", "-C", self.root, "hook", "run", "pre-commit"]
            cwd = self.root.parent
        else:
            cwd = self.root
        result = run_fixture(command, cwd=cwd, environment=environment)
        assert {name: name.read_bytes() for name in indexes} == original
        assert worktree_files(self.root) == worktree
        return result

    def staged_hook(
        self, files: dict[str, str], *, unstaged: dict[str, str] | None = None
    ) -> subprocess.CompletedProcess[str]:
        self.stage(files)
        if unstaged is not None:
            for name, contents in unstaged.items():
                self.write(name, contents)
        return self.hook()

    def stage_formatted(self, name: str, source: str) -> None:
        result = run_fixture(
            ["clang-format", f"--assume-filename={name}"], cwd=self.root, source=source
        )
        assert_passed(result)
        self.write(name, result.stdout)
        assert_passed(run_fixture(["clang-format", "--dry-run", "--Werror", name], cwd=self.root))
        self.git("add", name)


def gate_policy() -> dict[str, Any]:
    return json.loads((ROOT / GATE_POLICY).read_text(encoding="utf-8"))


def seed_repository(
    root: Path,
    files: dict[str, str] | None = None,
    *,
    style: bool = False,
    commit: bool = True,
    policy: dict[str, Any] | None = None,
) -> GateRepository:
    repo = GateRepository.init(root)
    repo.git("config", "core.hooksPath", ".githooks")
    names = [".githooks/pre-commit", ".githooks/commit-msg", ".githooks/run"]
    if style:
        names.extend([".clang-format", ".clang-tidy", "ruff.toml"])
    for name in names:
        target = root / name
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(ROOT / name, target)
    for helper in CHECKER.parent.glob("*.py"):
        repo.write(helper.relative_to(ROOT), helper.read_text(encoding="utf-8"))
    if policy is None:
        policy = gate_policy()
        policy["checks"] = [
            check
            for check in policy["checks"]
            if style and check["name"] == "Staged source lint and formatting"
        ]
        if style:
            assert len(policy["checks"]) == 1
    repo.write(GATE_POLICY, json.dumps(policy, indent=2) + "\n")
    if files is not None:
        for name, contents in files.items():
            repo.write(name, contents)
    repo.git("add", "--all")
    if commit:
        repo.git(
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
    return repo


def install_hook_tool(host_root: Path, name: str, script: str) -> None:
    path = host_root / ".pixi/envs/default/bin" / name
    write_fixture(host_root, path, script)
    path.chmod(0o755)


def install_hook_python(host_root: Path) -> None:
    install_hook_tool(host_root, "python", f'#!/bin/sh\nexec {shlex.quote(sys.executable)} "$@"\n')


def worktree_files(root: Path) -> dict[Path, tuple[int, bytes | None]]:
    paths = {}
    for child in root.iterdir():
        if child.name == ".git":
            continue
        for path in (child, *child.rglob("*")):
            contents = path.read_bytes() if path.is_file() else None
            paths[path.relative_to(root)] = path.stat().st_mode, contents
    return paths


def assert_passed(result: subprocess.CompletedProcess[str]) -> str:
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    return output


def assert_rejected(
    result: subprocess.CompletedProcess[str], *diagnostics: str, stderr_only: bool = False
) -> str:
    output = result.stdout + result.stderr
    assert result.returncode != 0, output
    diagnostic_output = result.stderr if stderr_only else output
    for diagnostic in diagnostics:
        assert diagnostic in diagnostic_output, output
    return output


def assert_header_rejected(
    result: subprocess.CompletedProcess[str], path: str, line: int = 1
) -> None:
    assert_rejected(
        result, f"{path}: Line {line} contains a function definition.", stderr_only=True
    )


@pytest.fixture
def alternate_index(tmp_path: Path) -> GateRepository:
    repo = seed_repository(tmp_path, {"src/legacy.h": "inline int value() { return 7; }\n"})
    selected = repo.alternate()
    selected.stage({"README.md": "# Updated fixture\n"})
    return selected


def test_header_rule_rejects_retained_internal_definition(alternate_index: GateRepository) -> None:
    assert_header_rejected(alternate_index.hook(), "src/legacy.h")


def test_header_rule_ignores_generated_and_unselected_headers(
    alternate_index: GateRepository,
) -> None:
    alternate_index.git("rm", "--cached", "src/legacy.h")
    alternate_index.write("build/generated.hpp", "int generated() { return 7; }\n")
    assert_passed(alternate_index.hook())
    assert (alternate_index.root / "src/legacy.h").is_file()


@pytest.mark.parametrize("path", ["src/new.cuh", "tests/probe.ipp", "tools/probe.tpp"])
def test_header_rule_accepts_internal_declarations(
    alternate_index: GateRepository, path: str
) -> None:
    alternate_index.git("rm", "--cached", "src/legacy.h")
    assert_passed(alternate_index.staged_hook({path: "#pragma once\nint declared();\n"}))
    assert (alternate_index.root / "src/legacy.h").is_file()


@pytest.fixture
def comment_repository(tmp_path: Path) -> GateRepository:
    return seed_repository(
        tmp_path,
        {"legacy.py": "# inherited\nvalue = 1\n", "removed.cpp": "int value; // inherited\n"},
    ).alternate()


def comment_diagnostic(name: str, line: int = 1) -> str:
    return f"{name}: Line {line} contains a comment"


def test_whole_staged_file_and_worktree_disagreement(comment_repository: GateRepository) -> None:
    name = "legacy.py"
    staged = "# inherited\nvalue = 2\n"
    clean = "value = 2\n"
    result = comment_repository.staged_hook({name: staged}, unstaged={name: clean})
    assert_rejected(result, comment_diagnostic(name), stderr_only=True)
    assert_passed(comment_repository.staged_hook({name: clean}, unstaged={name: staged}))


def test_shared_rule_change_ignores_untouched_and_deleted_comments(
    comment_repository: GateRepository,
) -> None:
    comment_repository.git("rm", "--cached", "removed.cpp")
    assert_passed(comment_repository.staged_hook({"AGENTS.md": "# Shared rules\n"}))


def test_rename_checks_destination_with_existing_comment(
    comment_repository: GateRepository,
) -> None:
    target = "renamed.cpp"
    comment_repository.git("mv", "removed.cpp", target)
    assert_rejected(comment_repository.hook(), comment_diagnostic(target), stderr_only=True)


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
    comment_repository: GateRepository, name: str, clean: str, comment: str
) -> None:
    assert_passed(comment_repository.staged_hook({name: clean}))
    result = comment_repository.staged_hook({name: clean + comment})
    assert_rejected(result, comment_diagnostic(name, clean.count("\n") + 1), stderr_only=True)


@pytest.fixture
def comment_parser(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(CHECKER.parent)
    return import_module("comment_rules")


def test_other_recognized_languages_use_comment_parsers(
    comment_repository: GateRepository, comment_parser: ModuleType
) -> None:
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
    assert_passed(comment_repository.staged_hook(dict.fromkeys(sources, "")))
    for name, (source, _) in sources.items():
        comment_repository.write(name, source)
        assert comment_parser.comment_errors(comment_repository.root / name, code=True) == []
    result = comment_repository.staged_hook(
        {name: source + comment for name, (source, comment) in sources.items()}
    )
    lines = {name: source.count("\n") + 1 for name, (source, _) in sources.items()}
    assert_rejected(
        result, *(comment_diagnostic(name, line) for name, line in lines.items()), stderr_only=True
    )
    for name, line in lines.items():
        for earlier_line in range(1, line):
            assert comment_diagnostic(name, earlier_line) not in result.stderr


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
    comment_repository: GateRepository, extension: str, line: int, source: str
) -> None:
    path = f"tests/fixtures/probe.{extension}"
    assert_header_rejected(comment_repository.staged_hook({path: source}), path, line)


def test_header_rule_uses_staged_blob(comment_repository: GateRepository) -> None:
    path = "include/sglang/probe.hpp"
    declaration = (
        "template<class T> T identity(T);\n"
        "struct Probe { Probe(); Probe(const Probe&) = delete; virtual void run() = 0; };\n"
        "void unavailable(int) = delete;\n"
        "inline int (*probe)(int){nullptr};\n"
        "inline constexpr int data[] = {1, 2, 3};\n"
    )
    definition = "inline int value() { return 7; }\n"
    assert_passed(comment_repository.staged_hook({path: declaration}, unstaged={path: definition}))
    result = comment_repository.staged_hook({path: definition}, unstaged={path: declaration})
    assert_header_rejected(result, path)


def test_cuda_header_rule_uses_staged_qualifier_macro(comment_repository: GateRepository) -> None:
    qualifiers = "src/probe/qualifiers.cuh"
    probe = "src/probe/function.cuh"
    declaration = '#include "qualifiers.cuh"\nQUAL int probe();\n'
    qualifier_source = "#ifdef __CUDACC__\n#define QUAL __device__\n#else\n#define QUAL\n#endif\n"
    assert_passed(
        comment_repository.staged_hook(
            {qualifiers: qualifier_source, probe: declaration},
            unstaged={qualifiers: "#define QUAL unexpected_token\n"},
        )
    )

    definition = '#include "qualifiers.cuh"\nQUAL int probe() { return 0; }\n'
    result = comment_repository.staged_hook({probe: definition}, unstaged={probe: declaration})
    assert_header_rejected(result, probe, 2)


@pytest.fixture
def style_repository(tmp_path: Path) -> GateRepository:
    return seed_repository(tmp_path, style=True)


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
    style_repository: GateRepository, name: str, clean: str, dirty: str, malformed: str
) -> None:
    repo = style_repository.alternate()
    assert_passed(repo.staged_hook({name: clean}, unstaged={name: dirty}))
    assert_format_rejected(repo.staged_hook({name: dirty}, unstaged={name: clean}), name)
    output = assert_rejected(repo.staged_hook({name: malformed}, unstaged={name: clean}), name)
    assert "syntax" in output.lower() or "clang-diagnostic-error" in output, output


def assert_format_rejected(result: subprocess.CompletedProcess[str], name: str) -> None:
    assert_rejected(result, f"Format {name} from the repository root with:")


@pytest.mark.parametrize(("name", "dirty"), [("old.cpp", "int old=42;\n"), ("old.py", "old=42\n")])
def test_root_rename_checks_destination_format(tmp_path: Path, name: str, dirty: str) -> None:
    repo = seed_repository(tmp_path, {name: dirty}, style=True)
    assert_passed(repo.staged_hook({"new.py": "new = 42\n"}))

    renamed = "renamed" + Path(name).suffix
    repo.git("mv", name, renamed)
    staged_names = repo.git("diff", "--cached", "--name-status", "-M")
    assert f"R100\t{name}\t{renamed}" in staged_names
    assert_format_rejected(repo.hook(), renamed)


def test_initial_index_checks_root_files(tmp_path: Path) -> None:
    repo = seed_repository(tmp_path, {"root.py": "value=42\n"}, style=True, commit=False)
    assert_format_rejected(repo.hook(), "root.py")


@pytest.mark.parametrize("changed", ["AGENTS.md", GATE_POLICY])
def test_shared_policy_scan_skips_unrelated_path_only_check(tmp_path: Path, changed: str) -> None:
    policy = gate_policy()
    policy["checks"] = [
        {
            "name": "Full-policy sentinel",
            "kind": "static",
            "paths": ["src/*.cpp"],
            "command": ["{python}", "-c", "import sys; print(*sys.argv[1:])", "{files}"],
        },
        {
            "name": "Path-only sentinel",
            "kind": "static",
            "paths": ["src/*.cpp"],
            "path_only": True,
            "command": ["{python}", "-c", "raise SystemExit('path-only check reached')"],
        },
    ]
    repo = seed_repository(tmp_path, {"src/existing.cpp": "int existing;\n"}, policy=policy)
    text = "# Policy fixture\n"
    if changed == GATE_POLICY:
        text = (repo.root / changed).read_text(encoding="utf-8") + "\n"
    output = assert_passed(repo.staged_hook({changed: text}))
    assert "src/existing.cpp" in output, output
    assert "path-only check reached" not in output, output

    assert_rejected(
        repo.staged_hook({"src/probe.cpp": "int value = 42;\n"}), "path-only check reached"
    )


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
    repo = seed_repository(tmp_path, {name: opening + "".join(lines) + closing}, style=True)
    lines.insert(0, entry.format(26))
    lines[-1] = entry.format(27)
    assert_passed(repo.staged_hook({name: opening + "".join(lines) + closing}))

    lines[-1] = lines[-1].replace("= ", "=")
    assert_format_rejected(repo.staged_hook({name: opening + "".join(lines) + closing}), name)


def test_tool_configuration_is_read_from_selected_index(style_repository: GateRepository) -> None:
    repo = style_repository.alternate()
    assert_passed(
        repo.staged_hook(
            ROOT_SOURCES,
            unstaged={
                ".clang-format": "UnknownStyleOption: true\n",
                ".clang-tidy": "UnknownLintOption: true\n",
                "ruff.toml": "unknown-style-option = true\n",
                ".githooks/run": "exit 77\n",
            },
        )
    )


def test_real_hook_bootstraps_from_system_only_path(style_repository: GateRepository) -> None:
    style_repository.stage(ROOT_SOURCES)
    assert_passed(style_repository.hook(path="/usr/bin:/bin"))


def test_required_tool_missing_from_hook_path(tmp_path: Path) -> None:
    repo = seed_repository(tmp_path / "source", style=True)
    binaries = tmp_path / "bin"
    host_root = tmp_path / "missing-tools"
    binaries.mkdir()
    for command in ("git", "sh", "cp", "cmp", "mktemp", "rm"):
        executable = shutil.which(command)
        assert executable is not None, command
        (binaries / command).symlink_to(executable)
    install_hook_python(host_root)

    repo.stage({"README.md": "# Tool fixture\n"})
    assert_passed(repo.hook(path=binaries, host_root=host_root))

    repo.stage({"root.cpp": ROOT_SOURCES["root.cpp"]})
    assert_rejected(repo.hook(path=binaries, host_root=host_root), "clang-format")


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
    style_repository: GateRepository, name: str, source: str, diagnostic: str
) -> None:
    assert_rejected(style_repository.staged_hook({name: source}), name, diagnostic)


def test_staged_valid_python_fixture(style_repository: GateRepository) -> None:
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
    assert_passed(style_repository.staged_hook({"tests/test_workload.py": source}))


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
    repo = seed_repository(tmp_path, {name: before}, style=True)
    result = repo.staged_hook({name: before.replace(removed, "")}, unstaged={name: before})
    assert_rejected(result, name, diagnostic)


def test_staged_cpp_analyzer_rejects_formatted_null_dereference(tmp_path: Path) -> None:
    header = (
        "#pragma once\n#include <filesystem>\n"
        "bool checkpoint_exists(const std::filesystem::path&);\n"
    )
    repo = seed_repository(tmp_path, {"checkpoint_probe.h": header}, style=True)
    name = "checkpoint_probe.cpp"
    source = (
        '#include "checkpoint_probe.h"\n'
        "#include <system_error>\n\n"
        "bool checkpoint_exists(const std::filesystem::path& path) {\n"
        "    std::error_code error;\n"
        "    return std::filesystem::is_regular_file(path, error) && !error;\n"
        "}\n"
    )
    repo.stage_formatted(name, source)
    assert_passed(repo.hook())

    source += "\nint invalid_metadata() { int* value = nullptr; return *value; }\n"
    repo.stage_formatted(name, source)
    assert_rejected(repo.hook(), name, "clang-analyzer-core.NullDereference")


def test_staged_cuda_analyzes_device_branch_with_cuda_13_4(tmp_path: Path) -> None:
    header = "#pragma once\nvoid launch_lane(int* output);\n"
    repo = seed_repository(tmp_path, {"src/probe_sm100.h": header}, style=True)
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
    repo.stage_formatted(name, source)
    assert_passed(repo.hook())

    defect = "    int* pointer = nullptr;\n    *pointer = static_cast<int>(lane);\n"
    repo.stage_formatted(name, source.replace(guarded_ptx, guarded_ptx + defect))
    assert_rejected(repo.hook(), name, "clang-analyzer-core.NullDereference")


@pytest.mark.parametrize("selection", ["default", "relative"])
def test_staged_bootstrap_and_commit_message(tmp_path: Path, selection: str) -> None:
    default = seed_repository(tmp_path, {"probe.py": "value = 0\n"})
    repo = default.alternate(relative=True) if selection == "relative" else default
    repo.stage({"probe.py": "value = 1\n"})
    if selection == "relative":
        default.stage({"probe.py": "value = (\n"})
    helper = ".agents/checks/gate_config.py"
    valid = (repo.root / helper).read_text(encoding="utf-8")
    staged_driver = ".agents/checks/staged_check.py"
    failure_status = 93
    for name in (".githooks/run", ".agents/checks/check_rules.py", helper, staged_driver):
        bad = f"exit {failure_status}\n" if name.endswith("/run") else f"exit({failure_status})\n"
        repo.write(name, bad)
    assert_passed(repo.hook())
    default.git("restore", "--", staged_driver)
    direct = [sys.executable, "-IB", repo.root / staged_driver, repo.root]
    assert_passed(repo.hook(command=direct))
    message = repo.root / "message"
    repo.write(message, "test: verify selected message checker\n")
    hook: list[str | Path] = ["git", "hook", "run", "commit-msg", "--", message]
    if selection == "relative":
        default.git("add", "--", ".agents/checks/check_rules.py")
        assert default.hook(command=hook).returncode == failure_status
    assert_passed(repo.hook(command=hook))
    repo.write(message, "invalid header\n")
    assert_rejected(repo.hook(command=hook), "Use a Conventional")
    repo.stage({helper: 'raise ValueError("invalid staged helper")\n'})
    repo.write(helper, valid)
    for command in (None, direct):
        assert_rejected(repo.hook(command=command), "invalid staged helper")
    repo.git("rm", "--cached", "-f", "--", helper)
    assert_rejected(repo.hook(), helper)


def test_frozen_policy_cannot_be_changed_or_removed(tmp_path: Path) -> None:
    repo = seed_repository(tmp_path)
    original = (repo.root / GATE_POLICY).read_text(encoding="utf-8")
    for field in ("baseline", "limits"):
        configuration = json.loads(original)
        if field == "baseline":
            configuration[field]["tree_sha256"] = "1" * 64
        else:
            configuration["baseline"]["maxima"]["directory_entries"] = 31
            configuration[field]["directory_entries"] = 31
        result = repo.staged_hook(
            {GATE_POLICY: json.dumps(configuration) + "\n"}, unstaged={GATE_POLICY: original}
        )
        assert_rejected(result, "cannot change the frozen adoption baseline or limits")
    repo.git("rm", "--cached", "-f", "--", GATE_POLICY)
    assert_rejected(repo.hook(), "cannot remove the frozen gate configuration")


def test_replacement_ref_cannot_hide_selected_blob(tmp_path: Path) -> None:
    repo = seed_repository(tmp_path)
    name = "probe.py"
    repo.stage({name: "value = (\n"})
    repo.write(name, "value = 1\n")
    blob = repo.git("rev-parse", f":{name}").strip()
    replacement = repo.git("hash-object", "-w", name).strip()
    repo.git("replace", blob, replacement)
    assert repo.git("show", f":{name}") == "value = 1\n"
    assert_rejected(repo.hook(), name, "Python syntax is invalid")


def test_retained_link_follows_selected_source_changes(tmp_path: Path) -> None:
    name = "probe.py"
    guide = '[Value](probe.py#L1)<!-- prefix: "value = 1" -->\n'
    repo = seed_repository(tmp_path, {name: "value = 1\n", "guide.md": guide})
    assert_rejected(repo.staged_hook({name: "other = 1\n"}), "guide.md", "prefix is stale")
    repo.git("rm", "-f", name)
    assert_rejected(repo.hook(), "guide.md", "link is missing")
    repo.git("restore", "--source=HEAD", "--staged", "--worktree", "--", name)
    repo.git("mv", name, "renamed.py")
    assert_rejected(repo.hook(), "guide.md", "link is missing")


def test_staged_skill_package_rejects_repository_only_link(tmp_path: Path) -> None:
    package = ".agents/skills/example/"
    name = package + "SKILL.md"
    repo = seed_repository(
        tmp_path,
        {
            name: EXAMPLE_SKILL,
            package + "agents/openai.yaml": EXPLICIT_SKILL_POLICY,
            "guide.md": "# Guide\n",
        },
    )
    result = repo.staged_hook({name: EXAMPLE_SKILL + "\n[Guide](../../../guide.md)\n"})
    assert_rejected(result, "link leaves the selected directory")


def test_physical_limits_on_the_selected_index(tmp_path: Path) -> None:
    repo = seed_repository(tmp_path)
    name = "probe.cpp"
    inside = "int probe() {\n" + "    (void)0;\n" * 97
    ending = "    return 0;\n}\n"
    width = "value = '" + "a" * 90 + "'\n"
    entries = {f"entries/{index}": "initial\n" for index in range(30)}
    exact = {name: inside + ending + "\n" * 900, "width.py": width, "guide.md": "word " * 3000}
    assert_passed(repo.staged_hook({**entries, **exact}))
    extra = dict(exact)
    extra.update({name: exact[name] + "\n", "width.py": width.replace("a'", "aa'")})
    extra.update({"entries/extra": "one more\n", "guide.md": "word " * 3001})
    result = repo.staged_hook(extra)
    assert_rejected(result, "31 entries", "more than 100 characters", "3001 visible words")
    assert "more than 1000 lines" in result.stderr
    repo.git("rm", "-f", "entries/extra")
    output = assert_rejected(
        repo.staged_hook({**exact, name: inside + "    (void)0;\n" + ending + "\n" * 899}),
        "Function probe() at line 1 exceeds",
    )
    assert "more than 1000 lines" not in output


@pytest.mark.parametrize(
    ("suffix", "definition", "declaration"),
    [
        ("cpp", "int probe() {\n    return 1;\n}\n", "int value;\n"),
        ("cu", "__global__ void probe() {\n    return;\n}\n", "int value;\n"),
        ("lua", "function probe()\n    return 1\nend\n", "local value = 1\n"),
        ("sh", "probe() {\n    return 1\n}\n", "value=1\n"),
    ],
)
def test_function_adapter_span(
    tmp_path: Path, suffix: str, definition: str, declaration: str
) -> None:
    path = tmp_path / f"probe.{suffix}"
    arguments = gate_policy()["function_adapters"][f".{suffix}"]
    values = {"{python}": sys.executable, "{file}": str(path)}
    command = [values.get(part, part) for part in arguments]
    for text, expected in ((definition, [(1, 3)]), (declaration, [])):
        write_fixture(tmp_path, path.name, text)
        result = run_fixture(command, cwd=ROOT)
        assert_passed(result)
        spans = json.loads(result.stdout)
        assert [(span["start"], span["end"]) for span in spans] == expected
        assert all(span["name"] for span in spans)


def test_configured_child_sees_selected_git_and_fresh_report(tmp_path: Path) -> None:
    name = "probe/input.py"
    policy = gate_policy()
    check = {
        "name": "Snapshot observer",
        "kind": "test",
        "paths": [name],
        "command": ["{python}", "observe.py", "{files}"],
        "report": "report.xml",
    }
    policy["checks"] = [check, check | {"name": "Unrelated observer", "paths": ["other/*"]}]
    script = (
        "import subprocess\nimport sys\nfrom pathlib import Path\n"
        "head = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip()\n"
        "assert head.decode() == Path('expected-head').read_text().strip()\n"
        "changes = subprocess.check_output(['git', 'diff', '--cached', '--name-only'])\n"
        "assert changes.splitlines() == [b'expected-head', b'probe/input.py']\n"
        "assert sys.argv[1:] == (['probe/input.py'] if Path('probe/input.py').exists() else [])\n"
        "for name in sys.argv[1:]:\n"
        "    blob = subprocess.check_output(['git', 'show', ':' + name])\n"
        "    assert blob == Path(name).read_bytes()\n"
        "    if Path(name).read_text() == 'value = 1\\n':\n"
        "        Path('report.xml').write_text('<testsuite><testcase/></testsuite>')\n"
    )
    report = "<testsuite><testcase/></testsuite>\n"
    files = {name: "value = 0\n", "observe.py": script, "report.xml": report}
    repo = seed_repository(tmp_path, files, policy=policy)
    expected = repo.git("rev-parse", "HEAD")
    output = assert_passed(
        repo.staged_hook(
            {name: "value = 1\n", "expected-head": expected}, unstaged={name: "value = 2\n"}
        )
    )
    assert "Running Snapshot observer" in output
    assert "Running Unrelated observer" not in output
    result = repo.staged_hook({name: "value = 2\n"})
    assert_rejected(result, "Running Snapshot observer", "local JUnit report")
    repo.git("rm", "-f", name)
    assert_rejected(repo.hook(), "Running Snapshot observer", "local JUnit report")


@pytest.mark.parametrize(
    ("document", "diagnostic"),
    [
        ("<testsuite/>", "zero cases"),
        ("<testsuite><testcase><skipped/></testcase></testsuite>", "skipped every case"),
        ("<testsuite><testcase><failure/></testcase></testsuite>", "failure or error"),
        ("<testsuite><testcase/><error/></testsuite>", "collection or suite failure"),
        ("<testsuite><testcase/><failure/></testsuite>", "collection or suite failure"),
        ("<testsuites errors='1'><testsuite><testcase/></testsuite></testsuites>", "suite failure"),
        ("<testsuites failures='1'><testcase/></testsuites>", "suite failure"),
        ("<testsuites><testsuite errors='1'><testcase/></testsuite></testsuites>", "suite failure"),
        ("<testsuites><testsuite failures='1'><testcase/></testsuite></testsuites>", "suite fail"),
        ("<testsuites><testsuite><testcase/></testsuite></testsuites>", None),
        ("<testsuite><testcase/><testcase><skipped/></testcase></testsuite>", None),
    ],
)
def test_required_junit_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, document: str, diagnostic: str | None
) -> None:
    monkeypatch.syspath_prepend(CHECKER.parent)
    config = import_module("gate_config")
    name = "report.xml"
    write_fixture(tmp_path, name, document)
    if diagnostic is None:
        config.test_report(tmp_path, name)
    else:
        with pytest.raises(ValueError, match=diagnostic):
            config.test_report(tmp_path, name)


@pytest.mark.parametrize("command", [["missing-proof-tool"], ["{python}", "-c", "exit(23)"]])
def test_required_static_check_fails_closed(tmp_path: Path, command: list[str]) -> None:
    name = "probe.py"
    policy = gate_policy()
    policy["checks"] = [
        {"name": "Required static", "kind": "static", "paths": [name], "command": command}
    ]
    repo = seed_repository(tmp_path, policy=policy)
    diagnostic = command[0] if len(command) == 1 else "23"
    assert_rejected(repo.staged_hook({name: "value = 1\n"}), "Running Required static", diagnostic)


def test_required_function_adapter_fails_closed(tmp_path: Path) -> None:
    policy = gate_policy()
    policy["checks"] = []
    policy["function_adapters"][".cpp"] = ["missing-proof-parser", "{file}"]
    repo = seed_repository(tmp_path, policy=policy)
    result = repo.staged_hook({"probe.cpp": "int value;\n" + "\n" * 100})
    assert_rejected(result, "missing-proof-parser")
