import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import uuid


def git(repo, *arguments, allow_failure=False):
    result = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if result.returncode and not allow_failure:
        raise ValueError(result.stderr.strip() or "The Git command failed.")
    return result


def repository(path):
    result = git(Path(path).resolve(), "rev-parse", "--show-toplevel")
    return Path(result.stdout.strip()).resolve()


def current_head(repo):
    result = git(repo, "rev-parse", "--verify", "HEAD^{commit}", allow_failure=True)
    if result.returncode == 0:
        return result.stdout.strip()
    symbolic = git(repo, "symbolic-ref", "-q", "HEAD", allow_failure=True)
    if symbolic.returncode:
        raise ValueError("The repository has no valid HEAD.")
    reference = symbolic.stdout.strip()
    exists = git(repo, "show-ref", "--verify", "--quiet", reference, allow_failure=True)
    if exists.returncode != 1:
        raise ValueError("The repository has an invalid branch reference.")
    return None


def resolve_commit(repo, revision):
    if revision.startswith("-"):
        raise ValueError("A commit reference cannot start with a hyphen.")
    result = git(repo, "rev-parse", "--verify", f"{revision}^{{commit}}")
    return result.stdout.strip()


def write_state(path, state):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(state, stream, indent=2)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def read_state(path, repo):
    state = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(state, dict):
        raise ValueError("The session state must be a JSON object.")
    if state.get("version") != 1 or state.get("repository") != str(repo):
        raise ValueError("The session state does not match this repository.")
    interval = state.get("interval")
    if isinstance(interval, bool) or not isinstance(interval, int) or interval < 1:
        raise ValueError("The review interval must be a positive integer.")
    if not isinstance(state.get("reviewed"), list) or not isinstance(state.get("reviews"), list):
        raise ValueError("The session state has invalid review records.")
    if len(state["reviewed"]) % interval:
        raise ValueError("The session state has an incomplete recorded batch.")
    return state


def session_commits(repo, state):
    head = current_head(repo)
    base = state["base"]
    if head is None:
        if base is not None or state["reviewed"]:
            raise ValueError("The session history is no longer available.")
        return head, 0, []
    if base is not None:
        ancestor = git(repo, "merge-base", "--is-ancestor", base, head, allow_failure=True)
        if ancestor.returncode:
            raise ValueError(
                "The session base is no longer an ancestor. Recover the original base."
            )
    revision = f"{base}..{head}" if base is not None else head
    count = int(git(repo, "rev-list", "--count", revision).stdout.strip())
    commits = git(repo, "rev-list", "--topo-order", "--reverse", revision).stdout.splitlines()
    if len(commits) != count:
        raise ValueError("The commit list does not match the Git count.")
    reviewed = state["reviewed"]
    if len(set(reviewed)) != len(reviewed) or not set(reviewed).issubset(commits):
        raise ValueError("Reviewed commits changed. Resolve the rewritten history before review.")
    return head, count, commits


def batch_digest(commits):
    return hashlib.sha256("\n".join(commits).encode("ascii")).hexdigest()


def status(repo, path, state):
    head, count, commits = session_commits(repo, state)
    reviewed = set(state["reviewed"])
    remaining = [commit for commit in commits if commit not in reviewed]
    interval = state["interval"]
    batch = remaining[:interval] if len(remaining) >= interval else []
    return {
        "state_path": str(path),
        "base": state["base"],
        "head": head,
        "count": count,
        "interval": interval,
        "reviewed_count": len(reviewed),
        "due_batches": len(remaining) // interval,
        "next_boundary": len(reviewed) + interval,
        "next_batch": batch,
        "batch_digest": batch_digest(batch) if batch else None,
    }


def start(repo, arguments):
    if arguments.state:
        path = (repo / arguments.state).resolve()
    else:
        relative = f"meta-harness/sessions/{uuid.uuid4().hex}.json"
        location = git(repo, "rev-parse", "--git-path", relative).stdout.strip()
        path = Path(location)
        path = (repo / path).resolve() if not path.is_absolute() else path.resolve()
    if path.exists():
        state = read_state(path, repo)
        if arguments.interval is not None and arguments.interval != state["interval"]:
            raise ValueError("The existing session has a different review interval.")
        if arguments.base is not None:
            if resolve_commit(repo, arguments.base) != state["base"]:
                raise ValueError("The existing session has a different base.")
        return status(repo, path, state)
    interval = arguments.interval if arguments.interval is not None else 50
    if interval < 1:
        raise ValueError("The review interval must be positive.")
    base = resolve_commit(repo, arguments.base) if arguments.base else current_head(repo)
    state = {
        "version": 1,
        "repository": str(repo),
        "base": base,
        "interval": interval,
        "reviewed": [],
        "reviews": [],
    }
    snapshot = status(repo, path, state)
    write_state(path, state)
    return snapshot


def complete(repo, arguments):
    path = (repo / arguments.state).resolve()
    state = read_state(path, repo)
    for review in state["reviews"]:
        if review["digest"] == arguments.batch:
            snapshot = status(repo, path, state)
            snapshot["already_recorded"] = True
            return snapshot
    snapshot = status(repo, path, state)
    if snapshot["head"] != arguments.head:
        raise ValueError("The expected head does not match current HEAD. Examine the review again.")
    if not snapshot["next_batch"] or snapshot["batch_digest"] != arguments.batch:
        raise ValueError("The supplied batch does not match the next due review.")
    evidence_path = (repo / arguments.evidence).resolve()
    if not evidence_path.is_file():
        raise ValueError("The review evidence file does not exist.")
    evidence = evidence_path.read_text(encoding="utf-8")
    marker = f"<!-- review-batch: {arguments.batch} -->"
    body = evidence.replace(marker, "").strip()
    if marker not in evidence or len(body.split()) < 10:
        raise ValueError("Record the batch marker and actual findings before completion.")
    state["reviewed"].extend(snapshot["next_batch"])
    state["reviews"].append(
        {
            "digest": arguments.batch,
            "head": snapshot["head"],
            "evidence": str(evidence_path),
            "evidence_sha256": hashlib.sha256(evidence.encode("utf-8")).hexdigest(),
        }
    )
    write_state(path, state)
    return status(repo, path, state)


def main():
    parser = argparse.ArgumentParser(
        description="Track commit reviews for one development session."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    begin = commands.add_parser("start", help="Record the session base before development.")
    begin.add_argument("--repo", default=".")
    begin.add_argument("--state")
    begin.add_argument("--base")
    begin.add_argument("--interval", type=int)
    inspect = commands.add_parser("status", help="Show the Git count and next due batch.")
    inspect.add_argument("--repo", default=".")
    inspect.add_argument("--state", required=True)
    finish = commands.add_parser("complete", help="Record a completed review with evidence.")
    finish.add_argument("--repo", default=".")
    finish.add_argument("--state", required=True)
    finish.add_argument("--head", required=True, help="Full head identifier returned by status.")
    finish.add_argument("--batch", required=True)
    finish.add_argument("--evidence", required=True)
    arguments = parser.parse_args()
    try:
        repo = repository(arguments.repo)
        if arguments.command == "start":
            result = start(repo, arguments)
        elif arguments.command == "complete":
            result = complete(repo, arguments)
        else:
            path = (repo / arguments.state).resolve()
            result = status(repo, path, read_state(path, repo))
        print(json.dumps(result, indent=2))
        return 0
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
