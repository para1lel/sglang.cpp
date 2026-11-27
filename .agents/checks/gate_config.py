import ast
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as element_tree


DEFAULT_LIMITS = {
    "directory_entries": 30,
    "code_file_lines": 1000,
    "function_lines": 100,
}
CONFIG_PATH = ".agents/gates.json"
DATA_SUFFIXES = {".json", ".toml", ".yaml", ".yml"}
PYTHON_SUFFIXES = {".py", ".pyi"}


def limits(configuration=None):
    return (configuration or {}).get("limits", DEFAULT_LIMITS)


def load(root):
    path = root / CONFIG_PATH
    if not path.is_file():
        return {}
    configuration = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(configuration, dict) or configuration.get("schema") != 1:
        raise ValueError("The gate configuration needs schema 1.")
    maximum = configuration.get("baseline", {}).get("maxima", {})
    selected = configuration.get("limits", {})
    for name, minimum in DEFAULT_LIMITS.items():
        value = maximum.get(name)
        if type(value) is not int or value < 0:
            raise ValueError(f"The baseline needs a nonnegative {name}.")
        if type(selected.get(name)) is not int or selected[name] != max(value, minimum):
            raise ValueError(f"The {name} limit must equal max(baseline, {minimum}).")
    fields = (
        ("function_adapters", dict), ("checks", list), ("shared_paths", list),
        ("code_suffixes", list), ("code_paths", list),
    )
    for key, expected in fields:
        if not isinstance(configuration.get(key, expected()), expected):
            raise ValueError(f"The gate configuration has an invalid {key}.")
    configuration["_root"] = str(root)
    return configuration


def function_source(path):
    return path.suffix not in DATA_SUFFIXES


def is_code(path, suffixes, configuration, root):
    configured = configuration.get("code_suffixes", [])
    if not all(isinstance(value, str) and value.startswith(".") for value in configured):
        raise ValueError("Configured code suffixes must start with a dot.")
    policy_root = Path(configuration.get("_root", root))
    relative_root = policy_root if path.is_relative_to(policy_root) else root
    return (
        path.suffix in suffixes or path.suffix in configured or path.parent.name == ".githooks"
        or matches(path.relative_to(relative_root).as_posix(), configuration.get("code_paths", []))
    )


def function_spans(
    path: Path,
    text: str,
    configuration: dict,
    *,
    default_adapter: list[str] | None = None,
) -> list[tuple[str, int, int]]:
    if not function_source(path):
        return []
    if path.suffix in PYTHON_SUFFIXES:
        tree = ast.parse(text, filename=str(path))
        return [
            (
                node.name,
                min([node.lineno, *[item.lineno for item in node.decorator_list]]),
                node.end_lineno,
            )
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
    suffix = ".sh" if path.parent.name == ".githooks" else path.suffix
    command = configuration.get("function_adapters", {}).get(suffix, default_adapter)
    if (
        not isinstance(command, list) or not command or "{file}" not in command
        or not all(isinstance(part, str) and part for part in command)
    ):
        raise ValueError(f"Configure a syntax-tree function adapter for {suffix}: {path}")
    arguments = [str(path) if part == "{file}" else part for part in command]
    arguments = [sys.executable if part == "{python}" else part for part in arguments]
    result = subprocess.run(
        arguments,
        cwd=configuration.get("_root"),
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    if result.returncode and (diagnostic := result.stderr.strip()):
        raise ValueError(diagnostic)
    result.check_returncode()
    spans = json.loads(result.stdout)
    if not isinstance(spans, list):
        raise ValueError("A function adapter must return a JSON list.")
    maximum = len(text.splitlines())
    for span in spans:
        if (
            not isinstance(span, dict) or not isinstance(span.get("name"), str)
            or type(span.get("start")) is not int or type(span.get("end")) is not int
            or not 1 <= span["start"] <= span["end"] <= maximum
        ):
            raise ValueError(f"The function adapter returned an invalid span for {path}.")
    return [(span["name"], span["start"], span["end"]) for span in spans]


def measure(
    root: Path,
    paths: list[Path],
    code_suffixes: set[str],
    configuration: dict,
    *,
    default_adapters: dict[str, list[str]],
) -> dict:
    configuration = {**configuration, "_root": str(root)}
    maximum = dict.fromkeys(DEFAULT_LIMITS, 0)
    children = {}
    identity = hashlib.sha256()
    for path in sorted(paths):
        if not path.resolve().is_relative_to(root):
            raise ValueError(f"A baseline resource leaves the maintained tree: {path}")
        relative = path.relative_to(root)
        for parent in relative.parents:
            children.setdefault(parent, set()).add(relative.parts[len(parent.parts)])
        contents = path.read_bytes()
        identity.update(relative.as_posix().encode("utf-8") + b"\0")
        identity.update(hashlib.sha256(contents).digest())
        default_adapter = default_adapters.get(path.suffix.lower())
        if not is_code(path, code_suffixes, configuration, root) and default_adapter is None:
            continue
        text = contents.decode("utf-8")
        maximum["code_file_lines"] = max(maximum["code_file_lines"], len(text.splitlines()))
        for _, start, end in function_spans(
            path,
            text,
            configuration,
            default_adapter=default_adapter,
        ):
            maximum["function_lines"] = max(maximum["function_lines"], end - start + 1)
    maximum["directory_entries"] = max((len(value) for value in children.values()), default=0)
    result = subprocess.run(
        ["git", "--no-replace-objects", "-C", str(root), "rev-parse", "--verify", "HEAD"],
        capture_output=True, text=True, check=False,
    )
    return {
        "schema": 1,
        "baseline": {
            "revision": result.stdout.strip() if result.returncode == 0 else None,
            "tree_sha256": identity.hexdigest(),
            "maxima": maximum,
        },
        "limits": {name: max(value, DEFAULT_LIMITS[name]) for name, value in maximum.items()},
        "function_adapters": configuration.get("function_adapters", {}),
        "code_suffixes": configuration.get("code_suffixes", []),
        "code_paths": configuration.get("code_paths", []),
        "checks": configuration.get("checks", []),
        "shared_paths": configuration.get("shared_paths", []),
    }


def matches(name, patterns):
    if not isinstance(patterns, list) or not all(isinstance(item, str) for item in patterns):
        raise ValueError("Check paths must be a list of repository-relative glob patterns.")
    return any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns)


def baseline_unchanged(configuration: dict, old_configuration: bytes) -> None:
    if not old_configuration:
        return
    if not configuration:
        raise ValueError("A normal commit cannot remove the frozen gate configuration.")
    previous = json.loads(old_configuration)
    for name in ("baseline", "limits"):
        if previous.get(name) != configuration.get(name):
            raise ValueError(
                "A normal commit cannot change the frozen adoption baseline or limits."
            )


def test_report(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("A test check must produce its configured local JUnit report.")
    tree = element_tree.parse(path)
    cases = tree.findall(".//testcase")
    if not cases or all(case.find("skipped") is not None for case in cases):
        raise ValueError("The required test check executed zero cases or skipped every case.")
    if any(case.find(tag) is not None for case in cases for tag in ("failure", "error")):
        raise ValueError("The test report contains a failure or error.")
    if any(tree.find(f".//{tag}") is not None for tag in ("error", "failure")):
        raise ValueError("The test report contains a collection or suite failure.")
    suites = (node for node in tree.iter() if node.tag in {"testsuite", "testsuites"})
    if any(int(suite.get(key, "0")) != 0 for suite in suites for key in ("errors", "failures")):
        raise ValueError("The test report contains a collection or suite failure.")


def command_arguments(command, files):
    if not isinstance(command, list) or not command or not all(
        isinstance(part, str) and part for part in command
    ):
        raise ValueError("Each configured check needs a nonempty command argument list.")
    arguments = []
    for part in command:
        if part == "{files}":
            arguments.extend(files)
        else:
            arguments.append(sys.executable if part == "{python}" else part)
    return arguments


def run_checks(root, configuration, changed, full):
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith("GIT_")
    }
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment.pop("PYTEST_ADDOPTS", None)
    environment.pop("PYTHONPATH", None)
    for check in configuration.get("checks", []):
        if not isinstance(check, dict) or check.get("kind") not in {"static", "test"}:
            raise ValueError("Each configured check needs kind static or test.")
        name = check.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("Each configured check needs a name.")
        all_files = full and not check.get("path_only")
        patterns = check.get("paths", [])
        affected = [path for path in changed if matches(path, patterns)]
        if not all_files and not affected:
            continue
        files = [] if all_files else [path for path in affected if (root / path).is_file()]
        command = command_arguments(check.get("command"), files)
        if "{files}" in check["command"] and not files:
            command = command_arguments(check["command"], [
                path.relative_to(root).as_posix() for path in sorted(root.rglob("*"))
                if path.is_file() and ".git" not in path.parts
                and matches(path.relative_to(root).as_posix(), patterns)
            ])
        report = check.get("report")
        if check["kind"] == "test":
            if not isinstance(report, str) or not report:
                raise ValueError(
                    "Test checks need a JUnit report path or a project report adapter."
                )
            destination = (root / report).resolve()
            if not destination.is_relative_to(root) or destination == root:
                raise ValueError("The test report must remain inside the isolated snapshot.")
            if destination.exists():
                destination.unlink()
            destination.parent.mkdir(parents=True, exist_ok=True)
        timeout = check.get("timeout", 120)
        if type(timeout) is not int or timeout <= 0:
            raise ValueError("A check timeout must be a positive number of seconds.")
        print(f"Running {name}.", flush=True)
        subprocess.run(command, cwd=root, env=environment, check=True, timeout=timeout)
        if check["kind"] == "test":
            test_report(root, report)
