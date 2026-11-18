"""Public entry point controls run only by prove_handoff.py."""

import json
import shutil
import sys

from handoff_fixture import ALTERNATE, CHECKS, CONFIG, PRE_COMMIT


ITEM = "probe/left/item.py"
RIGHT = "probe/right/item.py"
SHARED = "probe/shared.txt"
GUIDE = "probe/reader.md"
SOURCE = "probe/documented.py"
REPORT = "probe/result.xml"
OBSERVER = """\
import os
from pathlib import Path
import subprocess
import sys

def git(*args):
    return subprocess.check_output(["git", *args])

expected = os.environ.get("HANDOFF_HEAD")
if expected:
    assert git("rev-parse", "HEAD").decode().strip() == expected
    names = git("diff", "--cached", "--name-only").decode().splitlines()
    assert sorted(names) == sorted(os.environ["HANDOFF_CHANGED"].splitlines()), names
for name in sys.argv[2:]:
    assert git("show", ":" + name) == Path(name).read_bytes()
print("observed", sys.argv[1], *sorted(sys.argv[2:]))
"""


def controlled_checks(repo):
    configuration = repo.config()
    configuration["shared_paths"] = [SHARED]
    configuration["checks"] = [
        {"name": side, "kind": "static", "paths": [f"probe/{side}/*"],
         "command": ["{python}", "probe/observe.py", side, "{files}"]}
        for side in ("left", "right")
    ]
    repo.write(CONFIG, json.dumps(configuration, indent=2) + "\n")
    repo.write("probe/observe.py", OBSERVER)
    repo.write(ITEM, "value = 0\n")
    repo.write(RIGHT, "value = 0\n")
    repo.write(SHARED, "before\n")
    repo.write(SOURCE, "value = 1\n")
    repo.write(GUIDE, '[value](documented.py#L1)<!-- prefix: "value = 1" -->\n')
    repo.checkpoint()


def expected_git(repo, *paths):
    repo.extra = {"HANDOFF_HEAD": repo.git("rev-parse", "HEAD"),
                  "HANDOFF_CHANGED": "\n".join(paths)}


def index_conflicts(repo, alternate):
    direct = [sys.executable, "-IB", repo.root / CHECKS / "staged_check.py", repo.root]
    if alternate:
        shutil.copy2(repo.root / ".git/index", repo.root / ALTERNATE)
    expected_git(repo, ITEM)
    sibling = f"{CHECKS}/gate_config.py"
    for valid in (True, False):
        staged, working = ("value = 7\n", "value = (\n")
        if not valid:
            staged, working = working, staged
        if alternate:
            repo.stage(ITEM, working)
        repo.stage(ITEM, staged, alternate=alternate)
        repo.write(ITEM, working)
        repo.write(sibling, 'raise RuntimeError("working helper loaded")\n')
        outcomes = {"contains": (f"observed left {ITEM}",), "absent": ("observed right",)}
        if not valid:
            outcomes = {"error": "Python syntax is invalid", "contains": (ITEM,)}
        mode = f"{'alternate' if alternate else 'default'} index, staged valid={valid}"
        for entry, argv in (("hook", PRE_COMMIT), ("direct", direct)):
            repo.observe(f"{entry}, {mode}", argv, alternate=alternate,
                         external=entry == "direct", **outcomes)


def staged_bootstrap(repo):
    repo.stage(ITEM, "value = 8\n")
    for name in (".githooks/run", f"{CHECKS}/staged_check.py", f"{CHECKS}/check_rules.py"):
        repo.write(name, "exit 93\n" if name.endswith("/run") else "raise SystemExit(93)\n")
    repo.observe("hook uses staged runner and checker", PRE_COMMIT,
                 contains=(f"observed left {ITEM}",))
    helper = f"{CHECKS}/gate_config.py"
    original = (repo.root / helper).read_text()
    repo.stage(helper, 'raise ValueError("invalid staged helper control")\n')
    repo.write(helper, original)
    repo.observe("invalid staged sibling", PRE_COMMIT, error="invalid staged helper control")
    driver = f"{CHECKS}/staged_check.py"
    repo.git("restore", "--", driver)
    repo.observe("direct helper rejects invalid staged sibling",
                 [sys.executable, "-IB", driver, "."], error="invalid staged helper control")
    repo.git("rm", "--cached", "-f", "--", helper)
    repo.observe("missing staged sibling with working copy", PRE_COMMIT, error="gate_config.py")


def git_selection(repo):
    repo.stage(ITEM, "value = 2\n")
    expected_git(repo, ITEM)
    repo.observe("module-only selection", PRE_COMMIT, contains=(f"observed left {ITEM}",),
                 absent=("observed right",))
    repo.git("rm", "-f", ITEM)
    repo.observe("deletion still selects owner", PRE_COMMIT, contains=("observed left\n",),
                 absent=("observed right",))
    repo.git("restore", "--source=HEAD", "--staged", "--worktree", "--", ITEM)
    repo.stage(SHARED, "after\n")
    expected_git(repo, SHARED)
    repo.observe("shared selection", PRE_COMMIT,
                 contains=(f"observed left {ITEM}", f"observed right {RIGHT}"))


def incoming_links(repo):
    repo.stage(SOURCE, "other = 1\n")
    repo.observe("incoming link stale", PRE_COMMIT, error="prefix is stale", contains=(GUIDE,))
    repo.git("rm", "-f", SOURCE)
    repo.observe("incoming link deleted", PRE_COMMIT, error="local link is missing",
                 contains=(GUIDE,))
    repo.git("restore", "--source=HEAD", "--staged", "--worktree", "--", SOURCE)
    repo.git("mv", SOURCE, "probe/renamed.py")
    repo.observe("incoming link renamed", PRE_COMMIT, error="local link is missing",
                 contains=(GUIDE,))


def frozen_baseline(repo):
    original = repo.config()
    for field in ("tree_sha256", "limits"):
        policy = repo.config()
        if field == "tree_sha256":
            policy["baseline"][field] = "1" * 64
        else:
            size = original["limits"]["directory_entries"] + 1
            policy["baseline"]["maxima"]["directory_entries"] = size
            policy["limits"]["directory_entries"] = size
        repo.stage(CONFIG, json.dumps(policy, indent=2) + "\n")
        repo.write(CONFIG, json.dumps(original, indent=2) + "\n")
        repo.observe(f"changed frozen {field}", PRE_COMMIT, error="frozen adoption")
    repo.git("rm", "--cached", "-f", CONFIG)
    repo.observe("removed frozen policy", PRE_COMMIT, error="cannot remove")


def commit_message(repo):
    filename = "probe/message"
    argv = ["git", "hook", "run", "commit-msg", "--", filename]
    repo.write(filename, "feat(api)!: change reply\n\nBREAKING CHANGE: use v2.\n")
    repo.write(f"{CHECKS}/check_rules.py", "raise SystemExit(97)\n")
    repo.observe("commit-msg accepts valid footer", argv, contains=("Conventional Commits",))
    repo.write(filename, "bad header\n")
    repo.observe("commit-msg rejects malformed header", argv, error="Use a Conventional")
    shutil.copy2(repo.root / ".git/index", repo.root / ALTERNATE)
    repo.git("add", "--", f"{CHECKS}/check_rules.py")
    repo.write(filename, "test: select alternate message checker\n")
    repo.observe("commit-msg uses alternate checker", argv, alternate=True,
                 contains=("Conventional Commits",))


def configure_command(repo, kind, command):
    policy = repo.config()
    policy["checks"] = [{"name": "proof-child", "kind": kind, "paths": ["probe/left/*"],
                         "command": command}]
    if kind == "test":
        policy["checks"][0]["report"] = REPORT
    repo.write(CONFIG, json.dumps(policy, indent=2) + "\n")
    repo.checkpoint()
    repo.stage(ITEM, "value = 3\n")


def required_static(repo, missing):
    command = ["missing-handoff-static-tool"] if missing else ["{python}", "-c", "exit(23)"]
    configure_command(repo, "static", command)
    reason = "missing-handoff-static-tool" if missing else "23"
    repo.observe(f"required static missing={missing}", PRE_COMMIT, error=reason,
                 contains=("Running proof-child.",))


def junit(repo):
    passed = '<testcase name="ran"/>'
    skip = '<testcase name="optional"><skipped/></testcase>'
    repo.write(REPORT, f"<testsuite>{passed}</testsuite>\n")
    script = "probe/write_report.py"
    repo.write(script, "from pathlib import Path\n"
               "source = Path('probe/left/cases.xml')\n"
               f"if source.exists():\n    Path('{REPORT}').write_bytes(source.read_bytes())\n")
    configure_command(repo, "test", ["{python}", script])
    repo.observe("stale report is removed before child", PRE_COMMIT, error="local JUnit report")

    def suite(cases, errors=0, failures=0):
        return f'<testsuite errors="{errors}" failures="{failures}">{cases}</testsuite>'

    def report(child, errors=0, failures=0):
        return f'<testsuites errors="{errors}" failures="{failures}">{child}</testsuites>'

    good = suite(passed)
    failed = '<testcase name="failed"><failure/></testcase>'
    for label, xml, failure in [
        ("zero", suite(""), "zero cases"), ("all skipped", suite(skip), "skipped every case"),
        ("failed", suite(failed), "failure or error"),
        ("executed plus skipped", suite(passed + skip), None),
        ("passing case alone", f"<testsuite>{passed}</testsuite>", None),
        ("suite error with a passing case", f"<testsuite>{passed}<error/></testsuite>",
         "suite failure"),
        ("suite failure with a passing case", suite(passed + "<failure/>"), "suite failure"),
        ("aggregate and nested counts zero", report(good), None),
        ("aggregate errors attribute only", report(good, errors=1), "suite failure"),
        ("aggregate failures attribute only", report(good, failures=1), "suite failure"),
        ("nested errors attribute only", report(suite(passed, errors=1)), "suite failure"),
        ("nested failures attribute only", report(suite(passed, failures=1)), "suite failure"),
    ]:
        repo.stage("probe/left/cases.xml", xml + "\n")
        repo.observe(f"JUnit {label}", PRE_COMMIT, error=failure,
                     contains=("Running proof-child.",))


def package_boundary(repo):
    package = ".agents/skills"
    skill = sorted((repo.root / package).glob("*/SKILL.md"))[0]
    argv = [sys.executable, "-IB", f"{CHECKS}/check_rules.py", "package", package,
            "--policy-root", "."]
    repo.observe("copied skills and policy", argv, contains=("Passed:",))
    name = str(skill.relative_to(repo.root))
    repo.stage(name, skill.read_text() + "\n[Fixture document](../../../probe/reader.md)\n")
    repo.observe("standalone package rejects repository-only resource", argv,
                 error="link leaves the selected directory")
    repo.observe("hook enforces copied package boundary", PRE_COMMIT,
                 error="link leaves the selected directory")


def replaced_blob(repo):
    bad = "probe/bad.py"
    repo.stage(bad, "value = (\n")
    repo.write(bad, "value = 9\n")
    original = repo.git("rev-parse", f":{bad}")
    substitute = repo.git("hash-object", "-w", bad)
    repo.git("replace", original, substitute)
    if repo.git("show", f":{bad}") != "value = 9":
        raise AssertionError("Git replacement control did not take effect")
    repo.observe("replacement cannot hide malformed staged blob", PRE_COMMIT,
                 error="Python syntax is invalid", contains=(bad,))
