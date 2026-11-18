"""Run with python -B tests/prove_handoff.py [--revision COMMIT] and retain stdout as JSONL."""

import argparse
import fnmatch
from functools import partial
from importlib.metadata import version
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import traceback

import handoff_cases as cases
from handoff_fixture import CONFIG, PRE_COMMIT, Fixture, copy_candidate, digest, invoke, record
from handoff_limits import boundaries, fixed_limits, function_adapter


TESTS = Path(__file__).resolve().parent
SOURCE = TESTS.parent


def checkout(work, revision):
    git = ["git", "--no-replace-objects"]
    commit = invoke(SOURCE, [*git, "rev-parse", "--verify", revision + "^{commit}"])
    identity = commit.stdout.decode().strip()
    hooks = invoke(SOURCE, [*git, "config", "--local", "--get", "core.hooksPath"])
    if hooks.stdout.decode().strip() != ".githooks":
        raise ValueError("Target core.hooksPath must select .githooks for the installation proof")
    clone = work / "checkout"
    invoke(work, [*git, "clone", "--quiet", "--no-local", "--no-checkout", "--template=",
                  "--", SOURCE, clone])
    invoke(clone, [*git, "checkout", "--quiet", "--detach", identity])
    proof = [*sorted(TESTS.glob("handoff_*.py")), Path(__file__)]
    packages = ("pytest", "markdown-it-py", "PyYAML", "tree-sitter", "tree-sitter-bash",
                "tree-sitter-cpp", "tree-sitter-cuda", "tree-sitter-lua")
    git_version = invoke(work, ["git", "--version"]).stdout.decode().strip()
    record(candidate=identity, target=str(SOURCE), hooks_path=".githooks", python=sys.executable,
           python_version=sys.version, git=git_version,
           versions={package: version(package) for package in packages},
           proof_sha256={path.name: digest(path.read_bytes()) for path in proof})
    return clone


def installed_policy(repo):
    checks = repo.config()["checks"]
    expected = [f"Running {check['name']}." for check in checks]
    if not any(check["kind"] == "test" for check in checks):
        raise AssertionError("The copied policy has no required host test")
    repo.git("add", "-A")
    repo.observe("unchanged installation on unborn index", PRE_COMMIT, contains=expected)
    repo.checkpoint()
    host = "tests/test_gate_contract.py"
    repo.stage(host, (repo.root / host).read_text() + "\n")
    expected = [f"Running {check['name']}." for check in checks if check["kind"] == "test"
                and any(fnmatch.fnmatchcase(host, pattern) for pattern in check["paths"])]
    if not expected:
        raise AssertionError("The ordinary host test file does not select any configured test")
    tree = repo.git("write-tree")
    command = ["git", "commit", "-qm", "test: verify installed hooks"]
    result = invoke(repo.root, command, check=False)
    output = (result.stdout + result.stderr).decode(errors="replace")
    record(label="real fixture commit with both hooks", command=command, output=output,
           exit=result.returncode, expected_tree=tree)
    if result.returncode or not all(marker in output for marker in [*expected, "Conventional"]):
        raise AssertionError(f"Fixture commit did not pass both configured hooks: {output}")
    if repo.git("rev-parse", "HEAD^{tree}") != tree:
        raise AssertionError("Fixture commit did not contain the selected tree")
    repo.observe("clean selected index", PRE_COMMIT, contains=("no changed files",))


def clean_build(repo):
    repo.git("config", "core.hooksPath", ".githooks")
    if repo.git("status", "--porcelain", "--ignored"):
        raise AssertionError("XMake control requires a clean checkout")
    checks = [check for check in repo.config()["checks"] if check["kind"] == "static"
              and "tools/build.sh" in check["command"]]
    if len(checks) != 1:
        raise AssertionError("Expected a configured static tools/build.sh command")
    check = checks[0]
    samples = ("xmake.lua", "tools/build.sh", "src/module.cc", "include/api.h",
               "models/core.cu", "models/core.cuh", "models/core.cpp", "models/api.h")
    for name in samples:
        if not any(fnmatch.fnmatchcase(name, pattern) for pattern in check["paths"]):
            raise AssertionError(f"The configured XMake gate misses {name}")
    filename = "tools/toolchain-probe.cpp"
    source = (repo.root / filename).read_text()
    good = source + "\n"
    bad = source + "\n#error HANDOFF_EXPECTED_BUILD_FAILURE\n"
    marker = f"Running {check['name']}."
    repo.stage(filename, good)
    repo.write(filename, bad)
    repo.observe("clean clone compiles the staged C++ probe", PRE_COMMIT,
                 contains=(marker, "sm_100.cubin", "CUDA runtime:"),
                 absent=("Running on physical",))
    repo.stage(filename, bad)
    repo.write(filename, good)
    repo.observe("staged compiler error rejects commit with valid working source", PRE_COMMIT,
                 error="HANDOFF_EXPECTED_BUILD_FAILURE", contains=(marker,))
    repo.git("restore", "--source=HEAD", "--staged", "--worktree", "--", filename)
    repo.stage(CONFIG, (repo.root / CONFIG).read_text() + "\n")
    test_markers = [f"Running {group['name']}." for group in repo.config()["checks"]
                    if group["kind"] == "test"]
    repo.observe("gate-config-only change runs build and host checks", PRE_COMMIT,
                 contains=(marker, "sm_100.cubin", *test_markers))


def run_case(name, function, *arguments):
    try:
        function(*arguments)
    except (AssertionError, OSError, ValueError, subprocess.SubprocessError) as error:
        stderr = getattr(error, "stderr", None) or b""
        record(group=name, passed=False, detail=traceback.format_exc(),
               stderr=stderr.decode(errors="replace"))
        return False
    record(group=name, passed=True)
    return True


def run(work, revision):
    clone = checkout(work, revision)
    seed = Fixture(work / "install")
    seed.root.mkdir()
    copy_candidate(clone, seed.root)
    seed.initialize()
    results = [run_case("production configuration", installed_policy, seed)]
    if (clone / "xmake.lua").is_file() or any(
        "tools/build.sh" in check["command"] for check in seed.config()["checks"]
    ):
        results.append(run_case("clean clone XMake hook", clean_build, Fixture(clone)))
    else:
        record(unavailable="clean clone XMake hook", reason="selected revision has no xmake.lua")
    record(synthetic_checks="Fixture commands replace configured children after installation proof")
    cases.controlled_checks(seed)
    groups = {
        "default index": partial(cases.index_conflicts, alternate=False),
        "relative alternate": partial(cases.index_conflicts, alternate=True),
        "staged bootstrap": cases.staged_bootstrap, "snapshot Git": cases.git_selection,
        "incoming links": cases.incoming_links, "frozen policy": cases.frozen_baseline,
        "commit message": cases.commit_message, "JUnit": cases.junit,
        "static child exit": partial(cases.required_static, missing=False),
        "missing static": partial(cases.required_static, missing=True),
        "package": cases.package_boundary, "replacement refs": cases.replaced_blob,
        "width and words": fixed_limits, "parser adapters": function_adapter,
    }
    for name, function in groups.items():
        with tempfile.TemporaryDirectory(dir=work) as directory:
            fixture = Path(directory) / "fixture"
            shutil.copytree(seed.root, fixture, symlinks=True)
            results.append(run_case(name, function, Fixture(fixture)))
    for raised in (False, True):
        name = f"adoption_{'above' if raised else 'below'}"
        results.append(run_case(name, boundaries, seed, work / name, raised))
    record(groups=len(results), passed=results.count(True), failed=results.count(False))
    return int(not all(results))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", default="HEAD")
    arguments = parser.parse_args()
    temp = Path(tempfile.gettempdir()).resolve()
    if temp.is_relative_to(SOURCE):
        raise ValueError("Select TMPDIR outside the target for the independent repositories")
    if os.environ.get("PYTHONOPTIMIZE"):
        raise ValueError("Run the proof with Python assertions enabled")
    with tempfile.TemporaryDirectory(prefix="sglang-handoff-", dir=temp) as directory:
        return run(Path(directory), arguments.revision)


if __name__ == "__main__":
    sys.exit(main())
