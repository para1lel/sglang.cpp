import shutil
import subprocess
from pathlib import Path


def invoke(
    command: list[str],
    root: Path,
    source: bytes | None = None,
    *,
    merge_stderr: bool = False,
    timeout: int = 30,
) -> subprocess.CompletedProcess[bytes]:
    executable = shutil.which(command[0])
    if executable is None:
        raise FileNotFoundError(f"Install the pinned gate dependency {command[0]} from AGENTS.md.")
    return subprocess.run(
        [str(Path(executable).resolve()), *command[1:]],
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT if merge_stderr else subprocess.PIPE,
        input=source,
        timeout=timeout,
        check=False,
    )
