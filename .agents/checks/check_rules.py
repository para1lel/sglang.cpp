import argparse
from html.parser import HTMLParser
import json
import os
from pathlib import Path, PureWindowsPath
import re
import shutil
import subprocess
import sys
import tomllib
from urllib.parse import unquote, urlsplit

from markdown_it import MarkdownIt
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import comment_rules
import function_spans
import gate_config


CODE_SUFFIXES = {
    ".bash", ".c", ".cc", ".cpp", ".cxx", ".cu", ".cuh", ".go", ".h", ".hh", ".hpp",
    ".hxx", ".java", ".js", ".json", ".jsx", ".lua", ".mjs", ".ps1", ".py", ".rs", ".sh",
    ".toml", ".ts", ".tsx", ".yaml", ".yml",
}
CODE_SUFFIXES.update(function_spans.HEADER_SUFFIXES | gate_config.PYTHON_SUFFIXES)
DOCUMENT_SUFFIXES = {".md", ".rst", ".txt", ".html"}
MARKDOWN = MarkdownIt("commonmark", {"html": True}).enable("table")
REFERENCE_MARKDOWN = MarkdownIt("commonmark", {"html": True}).enable("table")
REFERENCE_MARKDOWN.validateLink = lambda destination: True
PREFIX = re.compile(r'<!-- prefix: ("(?:[^"\\]|\\.)*") -->')


class VisibleText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.hidden = 0

    def handle_starttag(self, tag, attributes):
        if tag in {"head", "script", "style"}:
            self.hidden += 1
        if tag in {"p", "div", "li", "td", "th", "br", "pre"}:
            self.parts.append("\n")
        if tag == "img" and not self.hidden:
            self.parts.append(dict(attributes).get("alt", ""))

    def handle_endtag(self, tag):
        if tag in {"head", "script", "style"}:
            self.hidden = max(0, self.hidden - 1)
        if tag in {"p", "div", "li", "td", "th", "pre", "h1", "h2", "h3", "h4"}:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


def markdown_body(text):
    match = re.match(r"\A---\r?\n(.*?)\r?\n---(?:\r?\n|\Z)", text, re.DOTALL)
    return text[match.end():] if match else text


def rendered_text(text, suffix):
    if suffix == ".md":
        text = MARKDOWN.render(markdown_body(text))
    if suffix in {".md", ".html"}:
        parser = VisibleText()
        parser.feed(text)
        return "".join(parser.parts)
    return text


def repository_files(root, *, tracked=False):
    command = ["git", "-C", str(root), "ls-files", "--cached", "-z"]
    if not tracked:
        command.extend(("--others", "--exclude-standard"))
    result = subprocess.run(command, capture_output=True, check=True)
    paths = sorted(set(result.stdout.decode("utf-8").split("\0")) - {""})
    return [root / relative for relative in paths if (root / relative).is_file()]


def package_files(root):
    if not any(root.glob("*/SKILL.md")):
        raise ValueError("Select a package root that contains skill directories.")
    ignored = {".git", ".venv", "__pycache__"}
    paths = []
    for path in sorted(root.rglob("*")):
        if ignored.intersection(path.relative_to(root).parts):
            continue
        if not path.resolve().is_relative_to(root):
            raise ValueError(f"A package resource leaves its directory: {path}")
        if path.is_file():
            paths.append(path)
    return paths


def code_errors(path, text, configuration=None, header_spans=None):
    errors = []
    lines = text.splitlines()
    limits = gate_config.limits(configuration)
    if len(lines) > limits["code_file_lines"]:
        errors.append(f"The code file has more than {limits['code_file_lines']} lines.")
    for number, line in enumerate(lines, 1):
        if len(line) > 100:
            errors.append(f"Line {number} has more than 100 characters.")
    if path.suffix in gate_config.PYTHON_SUFFIXES or (
        configuration and gate_config.function_source(path) and lines
    ):
        try:
            spans = header_spans
            if spans is None:
                spans = gate_config.function_spans(path, text, configuration)
            for name, start, end in spans:
                if end - start + 1 > limits["function_lines"]:
                    errors.append(f"Function {name} at line {start} exceeds the function limit.")
        except (ValueError, subprocess.SubprocessError) as error:
            errors.append(str(error))
        except SyntaxError as error:
            errors.append(f"The Python syntax is invalid at line {error.lineno}.")
    try:
        if path.suffix == ".json":
            json.loads(text)
        elif path.suffix == ".toml":
            tomllib.loads(text)
        elif path.suffix in {".yaml", ".yml"}:
            yaml.safe_load(text)
    except (ValueError, yaml.YAMLError) as error:
        errors.append(f"The structured data syntax is invalid: {error}")
    if path.suffix in {".sh", ".bash"} or path.parent.name == ".githooks":
        interpreter = "bash" if lines and "bash" in lines[0] else "sh"
        shell = shutil.which(interpreter)
        if shell is None and shutil.which("git"):
            candidate = Path(shutil.which("git")).resolve().parents[1] / "usr/bin/sh.exe"
            shell = str(candidate) if candidate.is_file() else None
        if shell is None:
            errors.append("POSIX sh is required to examine hook syntax.")
        else:
            result = subprocess.run(
                [shell, "-n", path.as_posix()], capture_output=True, text=True, check=False,
            )
            if result.returncode:
                errors.append(f"The shell syntax is invalid: {result.stderr.strip()}")
    return errors


def local_target(root, document, destination):
    parsed = urlsplit(destination)
    if parsed.scheme == "file" or PureWindowsPath(unquote(destination)).drive:
        raise ValueError(f"Use a relative path for a local file link: {destination}")
    if parsed.scheme or parsed.netloc:
        return None
    if Path(unquote(parsed.path)).is_absolute():
        raise ValueError(f"Use a relative path for a local file link: {destination}")
    target = (document.parent / unquote(parsed.path)).resolve() if parsed.path else document
    if not target.is_relative_to(root):
        raise ValueError(f"The local link leaves the selected directory: {destination}")
    return target, parsed.fragment


def requirement_errors(root, path, text):
    errors = []
    pattern = r"^(?:-r\s*|--requirement[ =]+|-c\s*|--constraint[ =]+)(.+)$"
    for line in text.splitlines():
        match = re.match(pattern, line.split(" #", 1)[0].strip())
        if match is None:
            continue
        destination = match[1].strip().strip("\"'")
        try:
            resolved = local_target(root, path, destination)
            if resolved is None or not resolved[0].is_file():
                errors.append(f"The included requirements file is unavailable: {destination}")
        except ValueError as error:
            errors.append(str(error))
    return errors


def code_reference_error(target, fragment, following):
    anchor = re.fullmatch(r"L([1-9]\d*)(?:-L([1-9]\d*))?", fragment)
    if anchor is None:
        return "A code link must have a line anchor, such as #L12."
    if anchor[2] and int(anchor[2]) < int(anchor[1]):
        return "The code line range ends before its start."
    lines = target.read_text(encoding="utf-8").splitlines()
    first = int(anchor[1])
    last = int(anchor[2] or first)
    if first > len(lines) or last > len(lines):
        return "The code line anchor is outside the file."
    expected = lines[first - 1].lstrip()[:12]
    if not expected:
        return "A code link must refer to a nonblank line."
    match = PREFIX.fullmatch(following)
    if match is None:
        return "Put the prefix comment immediately after the code link."
    if json.loads(match[1]) != expected:
        return f"The code prefix is stale. The expected value is {json.dumps(expected)}."
    return None


def link_errors(root, path, text, configuration=None):
    errors = []
    for block in REFERENCE_MARKDOWN.parse(markdown_body(text)):
        children = block.children or []
        for index, token in enumerate(children):
            if token.type not in {"link_open", "image"}:
                continue
            destination = token.attrGet("href" if token.type == "link_open" else "src") or ""
            try:
                resolved = local_target(root, path, destination)
                if resolved is None:
                    continue
                target, fragment = resolved
                if not target.exists():
                    errors.append(f"The local link is missing: {destination}")
                    continue
                if token.type == "image" or not gate_config.is_code(
                    target, CODE_SUFFIXES, configuration or {}, root
                ):
                    continue
                closing = next(
                    position for position in range(index + 1, len(children))
                    if children[position].type == "link_close"
                )
                following = children[closing + 1].content if closing + 1 < len(children) else ""
                error = code_reference_error(target, fragment, following)
                if error:
                    errors.append(f"{destination}: {error}")
            except (ValueError, OSError, StopIteration) as error:
                errors.append(str(error))
    return errors


def skill_errors(path, text):
    match = re.match(r"\A---\n(.*?)\n---(?:\n|\Z)", text, re.DOTALL)
    if match is None:
        return ["The skill has no YAML frontmatter."]
    metadata = yaml.safe_load(match[1])
    if not isinstance(metadata, dict):
        return ["The skill frontmatter must be a mapping."]
    errors = []
    if metadata.get("name") != path.parent.name:
        errors.append("The skill name must match its directory.")
    if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", str(metadata.get("name", ""))):
        errors.append("The skill name must use lowercase letters, digits, and hyphens.")
    if len(str(metadata.get("name", ""))) >= 64:
        errors.append("The skill name must have fewer than 64 characters.")
    description = metadata.get("description")
    if not isinstance(description, str) or not description.strip():
        errors.append("The skill needs a description.")
    if metadata.get("disable-model-invocation") is not True:
        errors.append("The skill must disable model invocation.")
    policy_path = path.parent / "agents" / "openai.yaml"
    if not policy_path.is_file():
        return [*errors, "The skill needs a Codex invocation policy."]
    policy = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
    if not isinstance(policy, dict) or not isinstance(policy.get("policy"), dict):
        errors.append("The Codex policy must be a mapping.")
    elif policy["policy"].get("allow_implicit_invocation") is not False:
        errors.append("The Codex policy must disable implicit invocation.")
    if "[TODO:" in markdown_body(text):
        errors.append("Replace the unfinished skill scaffold.")
    return errors


def directory_errors(root, paths, changed=None, configuration=None):
    children = {}
    affected = set()
    for path in paths:
        relative = path.relative_to(root)
        for parent in relative.parents:
            children.setdefault(parent, set()).add(relative.parts[len(parent.parts)])
    for name in changed or []:
        affected.update(Path(name).parents)
    limit = gate_config.limits(configuration)["directory_entries"]
    return [
        f"{parent.as_posix()}: The directory has {len(entries)} entries. The limit is {limit}."
        for parent, entries in sorted(children.items())
        if len(entries) > limit and (changed is None or parent in affected)
    ]


def reference_targets(root, path):
    text = path.read_text(encoding="utf-8")
    destinations = []
    if path.suffix == ".md":
        for block in REFERENCE_MARKDOWN.parse(markdown_body(text)):
            for token in block.children or []:
                if token.type in {"link_open", "image"}:
                    destinations.append(token.attrGet("href") or token.attrGet("src") or "")
        if path.name == "SKILL.md":
            destinations.append("agents/openai.yaml")
    if path.name == "requirements.txt":
        pattern = r"^(?:-r\s*|--requirement[ =]+|-c\s*|--constraint[ =]+)(.+)$"
        for line in text.splitlines():
            match = re.match(pattern, line.split(" #", 1)[0].strip())
            if match:
                destinations.append(match[1].strip().strip("\"'"))
    targets = set()
    for destination in destinations:
        try:
            resolved = local_target(root, path, destination)
            if resolved:
                targets.add(resolved[0])
        except ValueError:
            continue
    return targets


def incremental_paths(root, paths, changed):
    selected = {root / name for name in changed}
    dependencies = {
        path: reference_targets(root, path) for path in paths
        if path.suffix == ".md" or path.name == "requirements.txt"
    }
    while True:
        incoming = {
            path for path, targets in dependencies.items() if targets.intersection(selected)
        }
        if incoming.issubset(selected):
            return sorted(path for path in selected if path.is_file() or path.is_symlink())
        selected.update(incoming)


def check_headers(root, paths):
    parsers = {}
    measured = {}
    failures = []
    for path in paths:
        suffix = path.suffix.lower()
        if suffix not in function_spans.HEADER_SUFFIXES or not path.is_file():
            continue
        relative = path.relative_to(root)
        if not path.resolve().is_relative_to(root):
            failures.append(f"{relative}: The file leaves the repository.")
            continue
        grammar = function_spans.GRAMMARS[suffix]
        if grammar not in parsers:
            parsers[grammar] = function_spans.parser_for(path)
        definitions, spans, error = function_spans.scan_header(path, parsers[grammar], root)
        for line in sorted(set(definitions)):
            failures.append(
                f"{relative}: Line {line} contains a function definition. "
                "Move it to a .cpp or .cu translation unit."
            )
        if error is not None:
            failures.append(
                f"{relative}: Line {error} has unsupported or invalid {grammar} syntax."
            )
        measured[path] = [(item["name"], item["start"], item["end"]) for item in spans]
    return failures, measured


def check(root, paths=None, all_paths=None, changed=None, configuration=None, policy=None,
          project_headers=True):
    failures = []
    documents = 0
    skills = 0
    maximum = 0
    policy = set() if policy is None else set(policy)
    header_paths = all_paths
    if paths is None:
        paths = repository_files(root)
        if all_paths is None and project_headers:
            header_paths = repository_files(root, tracked=True)
    all_paths = paths if all_paths is None else all_paths
    configuration = gate_config.load(root) if configuration is None else configuration
    failures.extend(directory_errors(root, all_paths, changed, configuration))
    header_spans = {}
    if project_headers:
        header_failures, header_spans = check_headers(
            root, all_paths if header_paths is None else header_paths,
        )
        failures.extend(header_failures)
    for path in paths:
        relative = path.relative_to(root)
        if not path.resolve().is_relative_to(root):
            failures.append(f"{relative}: The file leaves the repository.")
            continue
        if not path.is_file():
            continue
        is_code = path in header_spans or gate_config.is_code(
            path, CODE_SUFFIXES, configuration, root,
        )
        selected = relative.as_posix() in policy
        if path.suffix not in DOCUMENT_SUFFIXES and not is_code and not selected:
            continue
        errors = []
        if path.suffix in DOCUMENT_SUFFIXES or is_code:
            text = path.read_text(encoding="utf-8")
            adapter = configuration.get("function_adapters", {}).get(path.suffix)
            standard = ["{python}", ".agents/checks/function_spans.py", "{file}"]
            spans = header_spans.get(path) if adapter in (None, standard) else None
            errors = code_errors(path, text, configuration, spans) if is_code else []
            if path.suffix in DOCUMENT_SUFFIXES:
                documents += 1
                count = len(re.findall(r"\S+", rendered_text(text, path.suffix)))
                maximum = max(maximum, count)
                if count > 3000:
                    errors.append(f"The document has {count} visible words. The limit is 3,000.")
            if path.suffix == ".md":
                errors.extend(link_errors(root, path, text, configuration))
            if path.name == "requirements.txt":
                errors.extend(requirement_errors(root, path, text))
            if path.name == "SKILL.md":
                skills += 1
                errors.extend(skill_errors(path, text))
        if selected:
            errors.extend(comment_rules.comment_errors(path, is_code))
        failures.extend(f"{relative}: {error}" for error in errors)
    for failure in failures:
        print(failure, file=sys.stderr)
    if failures:
        return 1
    print(f"Passed: {len(paths)} files, {documents} documents, and {skills} skills.")
    print(f"The largest document has {maximum} visible words.")
    print("Review prose and remote source links separately.")
    return 0


def code_link(root, arguments):
    target = (root / arguments.file).resolve()
    document = (root / arguments.document).resolve()
    if not target.is_relative_to(root) or not document.is_relative_to(root):
        raise ValueError("The source and document must be inside the repository.")
    if not target.is_file() or not gate_config.is_code(
        target, CODE_SUFFIXES, gate_config.load(root), root
    ):
        raise ValueError("Select an existing code file.")
    lines = target.read_text(encoding="utf-8").splitlines()
    if arguments.line < 1 or arguments.line > len(lines):
        raise ValueError("Select a line inside the source file.")
    prefix = lines[arguments.line - 1].lstrip()[:12]
    if not prefix:
        raise ValueError("Select a nonblank source line.")
    relative = Path(os.path.relpath(target, document.parent)).as_posix()
    encoded = json.dumps(prefix, ensure_ascii=True).replace("<", "\\u003c").replace(">", "\\u003e")
    label = target.name.replace("[", "\\[").replace("]", "\\]")
    destination = relative.replace(" ", "%20")
    print(f"[{label}]({destination}#L{arguments.line})<!-- prefix: {encoded} -->")
    return 0


def commit_message(path):
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    header = r"[a-z][a-z0-9-]*(?:\([^()\r\n]*[^()\s][^()\r\n]*\))?!?: \S[^\r\n]*"
    if not lines or re.fullmatch(header, lines[0], re.IGNORECASE) is None:
        raise ValueError("Use a Conventional Commits header: type(scope): description.")
    if len(lines) > 1 and lines[1].strip():
        raise ValueError("Separate the commit header and body with a blank line.")
    marker = re.compile(r"breaking[ -]change\s*:", re.IGNORECASE)
    for position, line in enumerate(lines[1:], 1):
        if marker.match(line) is None:
            continue
        if re.fullmatch(r"BREAKING(?: CHANGE|-CHANGE): \S.*", line) is None:
            raise ValueError("Use an uppercase breaking-change marker and a description.")
        start = position
        while start > 0 and lines[start - 1].strip():
            start -= 1
        footer = r"(?:[a-z0-9-]+(?:: | #)|BREAKING(?: CHANGE|-CHANGE): )\S"
        if re.match(footer, lines[start], re.IGNORECASE) is None:
            raise ValueError("Separate a breaking-change footer block with a blank line.")
    print("Passed: Conventional Commits header, body separation, and breaking-change markers.")
    print("Review the change type, compatibility impact, and English separately.")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Repository rule checks and code links.")
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("check")
    inspect.add_argument("root", nargs="?", default=".")
    inspect.add_argument("--selection", type=Path)
    bundle = commands.add_parser("package")
    bundle.add_argument("root", nargs="?", default=Path(__file__).resolve().parents[2])
    bundle.add_argument("--policy-root", type=Path)
    baseline = commands.add_parser("baseline")
    baseline.add_argument("root", nargs="?", default=".")
    baseline.add_argument("--adapters", type=Path)
    baseline.add_argument("--code-suffix", action="append")
    baseline.add_argument("--code-path", action="append")
    prepare = commands.add_parser("code-link")
    prepare.add_argument("file")
    prepare.add_argument("line", type=int)
    prepare.add_argument("--document", required=True)
    prepare.add_argument("--root", default=".")
    message = commands.add_parser("commit-message")
    message.add_argument("file")
    arguments = parser.parse_args()
    try:
        if arguments.command == "commit-message":
            return commit_message(Path(arguments.file))
        root = Path(arguments.root).resolve()
        if arguments.command == "baseline":
            configuration = gate_config.load(root)
            if arguments.code_suffix:
                configuration["code_suffixes"] = arguments.code_suffix
            if arguments.code_path:
                configuration["code_paths"] = arguments.code_path
            if arguments.adapters:
                configuration["function_adapters"] = json.loads(
                    arguments.adapters.read_text(encoding="utf-8")
                )
            result = gate_config.measure(root, repository_files(root), CODE_SUFFIXES, configuration)
            print(json.dumps(result, indent=2))
            return 0
        if arguments.command == "package":
            configuration = (
                gate_config.load(arguments.policy_root.resolve()) if arguments.policy_root else None
            )
            return check(root, package_files(root), configuration=configuration,
                         project_headers=False)
        if arguments.command == "check" and arguments.selection:
            selection = json.loads(arguments.selection.read_text(encoding="utf-8"))
            paths = [root / name for name in selection["files"]]
            changed = None if selection["full"] else selection["changed"]
            selected = paths if changed is None else incremental_paths(root, paths, changed)
            return check(root, selected, paths, changed, policy=selection["changed"])
        return check(root) if arguments.command == "check" else code_link(root, arguments)
    except (OSError, ValueError, yaml.YAMLError, subprocess.SubprocessError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
