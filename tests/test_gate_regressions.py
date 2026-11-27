from __future__ import annotations

import json
import shlex
import shutil
import sys
from typing import TYPE_CHECKING

import pytest
from test_gate_contract import (
    assert_header_rejected,
    assert_passed,
    assert_rejected,
    canonical_native_source,
    checker,
    git_fixture,
    seed_repository,
    selected_hook,
    stage_files,
    write_fixture,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("explicit_adapter", [False, True])
def test_baseline_measures_inline_header(tmp_path: Path, *, explicit_adapter: bool) -> None:
    git_fixture(tmp_path, "init", "--quiet", "--template=")
    write_fixture(tmp_path, "probe.inl", "void probe() {\n    return;\n}\n")
    arguments: list[str | Path] = ["baseline", tmp_path]
    expected_span = 3
    if explicit_adapter:
        spans = json.dumps([{"name": "adapter probe", "start": 1, "end": 2}])
        adapters = {
            ".inl": ["{python}", "-c", f"print({spans!r})", "{file}"],
        }
        write_fixture(tmp_path, "adapters.json", json.dumps(adapters) + "\n")
        arguments.extend(["--adapters", tmp_path / "adapters.json"])
        expected_span = 2

    result = checker(*arguments)
    assert_passed(result)
    assert json.loads(result.stdout)["baseline"]["maxima"]["function_lines"] == expected_span


def test_staged_pure_virtual_destructor_is_a_declaration(tmp_path: Path) -> None:
    root = seed_repository(tmp_path, {})
    name = "probe.hpp"
    declaration = "struct Probe { virtual ~Probe() = 0; };\n"
    stage_files(root, {name: declaration})
    assert_passed(selected_hook(root))

    stage_files(root, {name: declaration.replace("= 0;", "= default;")})
    assert_header_rejected(selected_hook(root), name)


def test_staged_delete_must_be_first_declaration(tmp_path: Path) -> None:
    root = seed_repository(tmp_path, {}, style=True)
    name = "probe.hpp"
    declaration = "void probe(int) = delete;\n"
    canonical_native_source(root, name, "void probe(double);\n" + declaration)
    assert_passed(selected_hook(root))

    canonical_native_source(root, name, "void probe(int);\n" + declaration)
    assert_rejected(selected_hook(root), name, "deleted definition must be first declaration")


@pytest.mark.parametrize(
    ("name", "header_name"),
    [("probe.cpp", "<detail//probe.h>"), ("probe.hpp", '"detail//probe.h"')],
)
def test_staged_has_include_header_name_can_contain_slashes(
    tmp_path: Path, name: str, header_name: str
) -> None:
    root = seed_repository(tmp_path, {})
    conditional = f"#if __has_include({header_name})"
    stage_files(root, {name: f"{conditional}\nint probe;\n#endif\n"})
    assert_passed(selected_hook(root))

    stage_files(root, {name: f"{conditional} // comment\nint probe;\n#endif\n"})
    assert_rejected(selected_hook(root), f"{name}: Line 1 contains a comment")


def test_staged_ruff_diagnostics_ignore_output_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = seed_repository(tmp_path, {}, style=True)
    output = root / "ruff-output.json"
    write_fixture(root, output, "preserve this file\n")
    monkeypatch.setenv("RUFF_OUTPUT_FILE", str(output))
    name = "probe.py"
    config = (root / "ruff.toml").read_text(encoding="utf-8")
    stage_files(root, {"ruff.toml": config + "\n", name: "value = 7\n"})
    assert_passed(selected_hook(root))

    stage_files(root, {name: "value = missing\n"})
    assert_rejected(selected_hook(root), name, "F821")


@pytest.mark.parametrize("include_style", ["literal", "macro", "via_header"])
def test_staged_header_checks_consumers_with_different_macros(
    tmp_path: Path, include_style: str
) -> None:
    name = "probe.hpp"
    header = (
        "#pragma once\n\n"
        "inline constexpr int value = 0;\n\n"
        "#ifdef VARIABLE_VALUE\n"
        "    #define PROBE_POINTER nullptr\n"
        "#else\n"
        "    #define PROBE_POINTER &value\n"
        "#endif\n"
    )
    macro_include = '#define PROBE_HEADER "probe.hpp"\n#include PROBE_HEADER\n'
    includes = {
        "literal": '#include "probe.hpp"\n',
        "macro": macro_include,
        "via_header": '#include "bridge.hpp"\n',
    }
    fixture = {}
    if include_style == "via_header":
        fixture["bridge.hpp"] = f"#pragma once\n\n{macro_include}"
    source = (
        f"{includes[include_style]}\n"
        "int read_probe() {\n"
        "    const int* pointer = PROBE_POINTER;\n"
        "    return *pointer;\n"
        "}\n"
    )
    root = seed_repository(
        tmp_path,
        {
            **fixture,
            name: header.replace("nullptr", "&value"),
            "a.cpp": source,
            "b.cpp": "#define VARIABLE_VALUE\n" + source,
        },
        style=True,
    )
    stage_files(root, {name: header.replace("nullptr", "&value").replace("value = 0", "value = 7")})
    assert_passed(selected_hook(root))

    stage_files(root, {name: header})
    assert_rejected(
        selected_hook(root),
        "b.cpp:",
        "Dereference of null pointer",
        "clang-analyzer-core.NullDereference",
    )


def test_staged_header_deletion_checks_macro_include_consumers(tmp_path: Path) -> None:
    name = "probe.hpp"
    root = seed_repository(
        tmp_path,
        {
            name: "#pragma once\n",
            "probe.cpp": '#define PROBE_HEADER "probe.hpp"\n#include PROBE_HEADER\n',
        },
        style=True,
    )
    git_fixture(root, "rm", "--", name)
    assert_rejected(selected_hook(root), "probe.cpp:2:", "'probe.hpp' file not found")


def test_staged_hook_rejects_head_advance_during_change_selection(tmp_path: Path) -> None:
    root = seed_repository(tmp_path / "source", {"probe.py": "value = 0\n"})
    original = git_fixture(root, "rev-parse", "HEAD").strip()
    stage_files(root, {"probe.py": "value = 1\n"})
    tree = git_fixture(root, "write-tree").strip()
    advanced = git_fixture(
        root,
        "-c",
        "user.name=Gate fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "commit-tree",
        tree,
        "-p",
        original,
        "-m",
        "Advance fixture HEAD",
    ).strip()
    stage_files(root, {"probe.py": "value = 0\n", "README.md": "# Hook fixture\n"})
    git = shutil.which("git")
    assert git is not None
    host = tmp_path / "host"
    binaries = host / ".pixi/envs/default/bin"
    python = binaries / "python"
    wrapper = binaries / "git"
    write_fixture(host, python, f'#!/bin/sh\nexec {shlex.quote(sys.executable)} "$@"\n')
    write_fixture(
        host,
        wrapper,
        "#!/bin/sh\nset -eu\n"
        f"git={shlex.quote(git)}\nroot={shlex.quote(str(root))}\n"
        '"$git" "$@"\n'
        'case " $* " in\n'
        '    *" -C $root diff --cached --name-only "*)\n'
        f'        "$git" -C "$root" update-ref HEAD {advanced} {original};;\n'
        "esac\n",
    )
    python.chmod(0o755)
    wrapper.chmod(0o755)

    assert_rejected(selected_hook(root, host_root=host), "HEAD changed during verification")
    assert git_fixture(root, "rev-parse", "HEAD").strip() == advanced
