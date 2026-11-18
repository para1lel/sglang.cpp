"""Measured adoption policies and independent physical-boundary witnesses."""

import json
import shutil
import sys

from handoff_fixture import CHECKS, CONFIG, PRE_COMMIT, Fixture


DEFAULTS = {"directory_entries": 30, "code_file_lines": 1000, "function_lines": 100}
MODULE = "limits/measured.py"


def source_text(file_lines, function_lines, value):
    body = ["@staticmethod", "def measured():"]
    body += ["    value = 1"] * (function_lines - 3)
    body.append(f"    return {value}")
    body += ["# outside the function"] * (file_lines - len(body))
    return "\n".join(body) + "\n"


def measured_fixture(installed, path, raised):
    directory, file_lines, function = (31, 1001, 101) if raised else (3, 6, 4)
    path.mkdir()
    repo = Fixture(path)
    repo.initialize()
    for index in range(directory):
        repo.write(f"entries/{index}", "initial\n")
    repo.write(MODULE, source_text(file_lines, function, 1))
    repo.checkpoint()
    policy = json.loads(repo.observe("measure adoption baseline", [
        sys.executable, "-IB", installed.root / CHECKS / "check_rules.py", "baseline", ".",
    ]))
    maxima = dict(zip(DEFAULTS, (directory, file_lines, function), strict=True))
    limits = {name: max(DEFAULTS[name], count) for name, count in maxima.items()}
    if policy["baseline"]["maxima"] != maxima or policy["limits"] != limits:
        raise AssertionError(f"Unexpected measurement for {maxima}: {policy}")
    if policy["baseline"]["revision"] != repo.git("rev-parse", "HEAD"):
        raise AssertionError("Measurement recorded a different adoption revision")
    shutil.copytree(installed.root / CHECKS, path / CHECKS)
    shutil.copytree(installed.root / ".githooks", path / ".githooks")
    policy["function_adapters"] = installed.config()["function_adapters"]
    repo.write(CONFIG, json.dumps(policy, indent=2) + "\n")
    repo.checkpoint()
    return repo, limits


def boundaries(installed, path, raised):
    repo, limits = measured_fixture(installed, path, raised)
    directory, file_lines, function = (limits[key] for key in DEFAULTS)
    for index in range(directory):
        repo.write(f"entries/{index}", "at limit\n")
    repo.git("add", "entries")
    repo.stage(MODULE, source_text(file_lines, function, 2))
    label = "raised" if raised else "default"
    repo.observe(f"{label} exact directory, file and function", PRE_COMMIT, contains=("Passed:",))
    repo.stage("entries/extra", "one too many\n")
    reason = f"entries: The directory has {directory + 1} entries. The limit is {directory}"
    repo.observe(f"{label} directory +1", PRE_COMMIT, error=reason)
    repo.git("rm", "-f", "entries/extra")
    repo.stage(MODULE, source_text(file_lines + 1, function, 2))
    repo.observe(f"{label} file +1", PRE_COMMIT, error=f"more than {file_lines} lines",
                 contains=(MODULE,), absent=("exceeds the function limit",))
    repo.stage(MODULE, source_text(file_lines, function + 1, 2))
    repo.observe(f"{label} decorated function +1 in fixed-size file", PRE_COMMIT,
                 error="Function measured at line 1 exceeds", absent=("more than",))


def fixed_limits(repo):
    repo.stage("probe/width.py", "#" * 100 + "\n")
    doc = "probe/words.md"
    repo.stage(doc, "word " * 3000 + "\n")
    repo.observe("exact line and document sizes", PRE_COMMIT, contains=("Passed:",))
    repo.stage("probe/width.py", "#" * 101 + "\n")
    repo.observe("line width +1", PRE_COMMIT, error="Line 1 has more than 100 characters")
    repo.stage("probe/width.py", "#" * 100 + "\n")
    repo.stage(doc, "word " * 3001 + "\n")
    repo.observe("document +1", PRE_COMMIT, error="3001 visible words", contains=(doc,))


def function_adapter(repo):
    adapters = repo.config()["function_adapters"]
    for suffix, (function, empty) in {
        ".cpp": ("int probe() {\n    return 1;\n}\n", "int value;\n"),
        ".cu": ("__global__ void probe() {\n    return;\n}\n", "int value;\n"),
        ".lua": ("function probe()\n    return 1\nend\n", "local value = 1\n"),
        ".sh": ("probe() {\n    return 1\n}\n", "value=1\n"),
    }.items():
        filename = f"probe/parser{suffix}"
        substitutions = {"{python}": sys.executable, "{file}": filename}
        argv = [substitutions.get(part, part) for part in adapters[suffix]]
        for text, expected in ((function, [(1, 3)]), (empty, [])):
            repo.write(filename, text)
            spans = json.loads(repo.observe(f"{suffix} spans {expected}", argv))
            observed = [(span["start"], span["end"]) for span in spans]
            if observed != expected or any(not span["name"] for span in spans):
                raise AssertionError(f"{suffix}: expected {expected}, got {spans}")
    limit = repo.config()["limits"]["function_lines"]
    filename = "probe/parser.cpp"
    inside = "int sample() {\n" + "    (void)0;\n" * (limit - 3)
    ending = "    return 0;\n}\n"
    repo.stage(filename, inside + ending + "// outside\n")
    repo.observe("parsed C++ function exactly at limit", PRE_COMMIT, contains=("Passed:",))
    repo.stage(filename, inside + "    (void)0;\n" + ending)
    repo.observe("parsed C++ function +1 in fixed-size file", PRE_COMMIT,
                 error="exceeds the function limit", contains=(filename,))
    control = repo.config()
    control["function_adapters"][".cpp"] = ["missing-handoff-parser", "{file}"]
    repo.write(CONFIG, json.dumps(control, indent=2) + "\n")
    repo.checkpoint()
    repo.stage(filename, "int value;\n" + "\n" * limit)
    repo.observe("required function parser unavailable", PRE_COMMIT, error="missing-handoff-parser")
