"""Independent Git repositories and observations for the standalone handoff proof."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys


CONFIG = ".agents/gates.json"
CHECKS = ".agents/checks"
ALTERNATE = ".git/handoff.index"
PRE_COMMIT = ["git", "hook", "run", "pre-commit"]


def record(**values):
    print(json.dumps(values, sort_keys=True), flush=True)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def environment():
    clean = {
        key: value for key, value in os.environ.items()
        if not key.startswith(("GIT_", "HANDOFF_")) and key not in {"PYTEST_ADDOPTS", "PYTHONPATH"}
    }
    clean.update(
        GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull,
        GIT_OPTIONAL_LOCKS="0", GIT_TERMINAL_PROMPT="0", PYTHONDONTWRITEBYTECODE="1",
        PATH=str(Path(sys.executable).parent) + os.pathsep + clean["PATH"],
    )
    return clean


def file_identity(path):
    mode = path.lstat().st_mode
    if stat.S_ISLNK(mode):
        return [mode, os.readlink(path)]
    if stat.S_ISDIR(mode):
        return [mode]
    if stat.S_ISREG(mode):
        return [mode, digest(path.read_bytes())]
    raise ValueError(f"Unexpected fixture resource: {path}")


def state(root):
    manifest = {}
    for current, directories, files in os.walk(root):
        parent = Path(current)
        if parent == root:
            directories.remove(".git")
        for name in directories + files:
            path = parent / name
            manifest[path.relative_to(root).as_posix()] = file_identity(path)
    metadata = root / ".git"
    paths = [metadata / name for name in ("index", "HEAD", "config", "packed-refs")]
    paths += [root / ALTERNATE, *sorted((metadata / "refs").rglob("*"))]
    for path in paths:
        manifest[path.relative_to(root).as_posix()] = file_identity(path) if path.exists() else None
    return manifest


def invoke(cwd, argv, env=None, check=True):
    return subprocess.run(
        argv, cwd=cwd, env=environment() | (env or {}), capture_output=True,
        timeout=360, check=check,
    )


def source_files(source):
    arguments = ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"]
    names = invoke(source, arguments).stdout.split(b"\0")
    return sorted({os.fsdecode(name) for name in names if name})


def copy_candidate(source, destination):
    manifest = {}
    for name in source_files(source):
        original = source / name
        target = destination / name
        if original.is_symlink():
            raise ValueError(f"Review copying a symlink into the proof fixture: {name}")
        if not original.is_file():
            continue
        manifest[name] = file_identity(original)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, target)
        if file_identity(target) != manifest[name]:
            raise RuntimeError(f"Input changed during the copy: {name}")
    head = invoke(source, ["git", "rev-parse", "--verify", "HEAD"], check=False)
    record(source=str(source), head=head.stdout.decode().strip() or None, inputs=manifest,
           manifest_sha256=digest(json.dumps(manifest, sort_keys=True).encode()))
    return manifest


class Fixture:
    def __init__(self, root):
        self.root = root
        self.extra = {}

    def git(self, *arguments, alternate=False):
        env = {"GIT_INDEX_FILE": ALTERNATE} if alternate else {}
        return invoke(self.root, ["git", *arguments], env).stdout.decode().strip()

    def initialize(self):
        self.git("init", "--quiet", "--template=")
        self.git("config", "user.name", "Handoff fixture")
        self.git("config", "user.email", "handoff@example.invalid")
        self.git("config", "core.hooksPath", ".githooks")

    def checkpoint(self):
        self.git("add", "-A")
        self.git("-c", f"core.hooksPath={os.devnull}", "commit", "-qm", "test: seed fixture")

    def write(self, name, text):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def stage(self, name, text, alternate=False):
        self.write(name, text)
        self.git("add", "--", name, alternate=alternate)

    def config(self):
        return json.loads((self.root / CONFIG).read_bytes())

    def observe(self, label, argv, *, error=None, contains=(), absent=(), alternate=False,
                external=False):
        before = state(self.root)
        env = dict(self.extra)
        if alternate:
            env["GIT_INDEX_FILE"] = ALTERNATE
        result = invoke(self.root.parent if external else self.root, argv, env, check=False)
        output = (result.stdout + result.stderr).decode(errors="replace")
        after = state(self.root)
        mutations = [key for key in before | after if before.get(key) != after.get(key)]
        violations = []
        if mutations:
            violations.append(f"Changed files or Git metadata: {sorted(mutations)}")
        if (result.returncode == 0) != (error is None):
            violations.append(f"Unexpected exit {result.returncode}")
        required = (*contains, error) if error is not None else contains
        violations.extend(f"Missing {text!r}" for text in required if text not in output)
        violations.extend(f"Unexpected {text!r}" for text in absent if text in output)
        cwd = self.root.parent if external else self.root
        record(label=label, command=[str(arg) for arg in argv], cwd=str(cwd),
               env=env, exit=result.returncode, output=output, errors=violations,
               expected_error=error, contains=contains, absent=absent, changed=sorted(mutations),
               before=digest(json.dumps(before, sort_keys=True).encode()),
               after=digest(json.dumps(after, sort_keys=True).encode()),
               indexes={key: before.get(key) for key in (".git/index", ALTERNATE)})
        if violations:
            raise AssertionError(f"{label}: {', '.join(violations)}\n{output[-1600:]}")
        return result.stdout
