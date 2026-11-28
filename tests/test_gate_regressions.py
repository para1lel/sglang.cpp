from __future__ import annotations

import json
import shlex
import shutil
from typing import TYPE_CHECKING

import pytest
from test_gate_contract import (
    GateRepository,
    assert_header_rejected,
    assert_passed,
    assert_rejected,
    checker,
    comment_diagnostic,
    install_hook_python,
    install_hook_tool,
    seed_repository,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("explicit_adapter", [False, True])
def test_baseline_measures_inline_header(tmp_path: Path, *, explicit_adapter: bool) -> None:
    repo = GateRepository.init(tmp_path)
    repo.write("probe.inl", "void probe() {\n    return;\n}\n")
    arguments: list[str | Path] = ["baseline", tmp_path]
    expected_span = 3
    if explicit_adapter:
        spans = json.dumps([{"name": "adapter probe", "start": 1, "end": 2}])
        adapters = {".inl": ["{python}", "-c", f"print({spans!r})", "{file}"]}
        repo.write("adapters.json", json.dumps(adapters) + "\n")
        arguments.extend(["--adapters", tmp_path / "adapters.json"])
        expected_span = 2

    result = checker(*arguments)
    assert_passed(result)
    assert json.loads(result.stdout)["baseline"]["maxima"]["function_lines"] == expected_span


def test_staged_pure_virtual_destructor_is_a_declaration(tmp_path: Path) -> None:
    repo = seed_repository(tmp_path)
    name = "probe.hpp"
    declaration = "struct Probe { virtual ~Probe() = 0; };\n"
    assert_passed(repo.staged_hook({name: declaration}))
    definition = declaration.replace("= 0;", "= default;")
    assert_header_rejected(repo.staged_hook({name: definition}), name)


def test_staged_delete_must_be_first_declaration(tmp_path: Path) -> None:
    repo = seed_repository(tmp_path, style=True)
    name = "probe.hpp"
    declaration = "void probe(int) = delete;\n"
    repo.stage_formatted(name, "void probe(double);\n" + declaration)
    assert_passed(repo.hook())

    repo.stage_formatted(name, "void probe(int);\n" + declaration)
    assert_rejected(repo.hook(), name, "deleted definition must be first declaration")


@pytest.mark.parametrize(
    ("name", "header_name"),
    [("probe.cpp", "<detail//probe.h>"), ("probe.hpp", '"detail//probe.h"')],
)
def test_staged_has_include_header_name_can_contain_slashes(
    tmp_path: Path, name: str, header_name: str
) -> None:
    repo = seed_repository(tmp_path)
    conditional = f"#if __has_include({header_name})"
    body = "\nint probe;\n#endif\n"
    assert_passed(repo.staged_hook({name: conditional + body}))
    result = repo.staged_hook({name: conditional + " // comment" + body})
    assert_rejected(result, comment_diagnostic(name))


def test_staged_ruff_diagnostics_ignore_output_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = seed_repository(tmp_path, style=True)
    output = "ruff-output.json"
    repo.write(output, "preserve this file\n")
    monkeypatch.setenv("RUFF_OUTPUT_FILE", str(repo.root / output))
    name = "probe.py"
    config = (repo.root / "ruff.toml").read_text(encoding="utf-8")
    assert_passed(repo.staged_hook({"ruff.toml": config + "\n", name: "value = 7\n"}))
    assert_rejected(repo.staged_hook({name: "value = missing\n"}), name, "F821")


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
    repo = seed_repository(
        tmp_path,
        {
            **fixture,
            name: header.replace("nullptr", "&value"),
            "a.cpp": source,
            "b.cpp": "#define VARIABLE_VALUE\n" + source,
        },
        style=True,
    )
    clean = header.replace("nullptr", "&value").replace("value = 0", "value = 7")
    assert_passed(repo.staged_hook({name: clean}))
    assert_rejected(
        repo.staged_hook({name: header}),
        "b.cpp:",
        "Dereference of null pointer",
        "clang-analyzer-core.NullDereference",
    )


def test_staged_header_deletion_checks_macro_include_consumers(tmp_path: Path) -> None:
    name = "probe.hpp"
    repo = seed_repository(
        tmp_path,
        {
            name: "#pragma once\n",
            "probe.cpp": '#define PROBE_HEADER "probe.hpp"\n#include PROBE_HEADER\n',
        },
        style=True,
    )
    repo.git("rm", "--", name)
    assert_rejected(repo.hook(), "probe.cpp:2:", "'probe.hpp' file not found")


def test_staged_cuda_dynamic_shared_storage(tmp_path: Path) -> None:
    repo = seed_repository(tmp_path, style=True)
    name = "probe.cu"
    source = (
        "[[maybe_unused]] static __device__ int read_shared(int index) {\n"
        "    alignas(1024) extern __shared__ unsigned char storage[];\n"
        "    return storage[index];\n"
        "}\n"
    )
    repo.stage_formatted(name, source)
    assert_passed(repo.hook())

    globals_source = "namespace host {\nint storage;\n}\n__device__ int device_storage;\n\n"
    repo.stage_formatted(name, globals_source + source)
    result = repo.hook()
    assert_rejected(
        result,
        "probe.cu:2:5:",
        "probe.cu:4:16:",
        "cppcoreguidelines-avoid-non-const-global-variables",
    )
    assert "probe.cu:7:" not in result.stdout + result.stderr


def test_staged_hook_rejects_head_advance_during_change_selection(tmp_path: Path) -> None:
    repo = seed_repository(tmp_path / "source", {"probe.py": "value = 0\n"})
    original = repo.git("rev-parse", "HEAD").strip()
    repo.stage({"probe.py": "value = 1\n"})
    tree = repo.git("write-tree").strip()
    advanced = repo.git(
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
    repo.stage({"probe.py": "value = 0\n", "README.md": "# Hook fixture\n"})
    git = shutil.which("git")
    assert git is not None
    host = tmp_path / "host"
    install_hook_python(host)
    install_hook_tool(
        host,
        "git",
        "#!/bin/sh\nset -eu\n"
        f"git={shlex.quote(git)}\nroot={shlex.quote(str(repo.root))}\n"
        '"$git" "$@"\n'
        'case " $* " in\n'
        '    *" -C $root diff --cached --name-only "*)\n'
        f'        "$git" -C "$root" update-ref HEAD {advanced} {original};;\n'
        "esac\n",
    )

    assert_rejected(repo.hook(host_root=host), "HEAD changed during verification")
    assert repo.git("rev-parse", "HEAD").strip() == advanced
