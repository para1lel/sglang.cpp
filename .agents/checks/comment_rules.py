import io
import os
import re
import tempfile
import tokenize
from pathlib import Path

import tree_sitter_bash
import tree_sitter_go
import tree_sitter_java
import tree_sitter_javascript
import tree_sitter_lua
import tree_sitter_powershell
import tree_sitter_rust
import tree_sitter_toml
import tree_sitter_typescript
import tree_sitter_yaml
from function_spans import HEADER_SUFFIXES
from tool_process import invoke
from tree_sitter import Language, Parser, Query, QueryCursor

C_KINDS = ({".c", ".c++", ".cc", ".cpp", ".cxx"} | HEADER_SUFFIXES) - {".cuh"}
CUDA_KINDS = {".cu", ".cuh"}
SHELL_KINDS = {".bash", ".sh"}
JAVASCRIPT_KINDS = {".js", ".jsx", ".mjs"}
SUFFIX_KINDS = {
    **dict.fromkeys(C_KINDS, "cpp"),
    **dict.fromkeys(CUDA_KINDS, "cuda"),
    **dict.fromkeys(SHELL_KINDS, "bash"),
    **dict.fromkeys(JAVASCRIPT_KINDS, "javascript"),
    ".go": "go",
    ".java": "java",
    ".json": "json",
    ".lua": "lua",
    ".ps1": "powershell",
    ".py": "python",
    ".pyi": "python",
    ".rs": "rust",
    ".toml": "toml",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".yaml": "yaml",
    ".yml": "yaml",
}
GRAMMARS = {
    "bash": tree_sitter_bash.language,
    "go": tree_sitter_go.language,
    "java": tree_sitter_java.language,
    "javascript": tree_sitter_javascript.language,
    "lua": tree_sitter_lua.language,
    "powershell": tree_sitter_powershell.language,
    "rust": tree_sitter_rust.language,
    "toml": tree_sitter_toml.language,
    "tsx": tree_sitter_typescript.language_tsx,
    "typescript": tree_sitter_typescript.language_typescript,
    "yaml": tree_sitter_yaml.language,
}
COMMENT_QUERIES = {
    **dict.fromkeys(("java", "rust"), "[(line_comment) (block_comment)] @comment"),
}
SHELL_INTERPRETERS = {
    "ash", "bash", "bsh", "dash", "ksh", "mksh", "pdksh", "posh", "sh", "yash", "zsh",
}
SHEBANG = re.compile(
    r"\A#![ \t]*(?:(?:/usr/bin/)?env[ \t]+(?:-{1,2}[A-Za-z0-9-]+[ \t]+)*|/?\S*/)?"
    r"([A-Za-z0-9_.+-]+)"
)
UNSUPPORTED = (
    "The {kind} code kind has no comment parser. Extend comment_rules.py before staging it."
)
COMMENT_FOUND = "Line {line} contains a comment. Remove it or express it as data."
NOT_UTF8 = "The staged content is not valid UTF-8, so its comments cannot be examined."
SPLICE = rb"\\[ \t]*\n"
COMMENT_START = re.compile(rb"(?:" + SPLICE + rb")*(/)(?:" + SPLICE + rb")*[/*]")
DIRECTIVE_SPACE = rb"(?:[ \t\v\f]|" + SPLICE + rb"|/\*.*?\*/)*"
HEADER_NAME = re.compile(
    rb"(?:#|%:)"
    + DIRECTIVE_SPACE
    + rb"(?P<directive>[a-z_]+(?:"
    + SPLICE
    + rb"[a-z_]+)*)"
    + DIRECTIVE_SPACE
    + rb"<(?P<name>(?:"
    + SPLICE
    + rb"|[^>\n])*)>",
    re.DOTALL,
)
HEADER_CHARACTER = re.compile(rb"(" + SPLICE + rb")|[^\n]")
HEADER_QUOTING = re.compile(rb"[\"']|//|/\*")
HEADER_DIRECTIVES = {b"include", b"include_next", b"import", b"embed"}


def classify(path, prefix):
    kind = SUFFIX_KINDS.get(path.suffix.lower())
    if kind:
        return kind
    if path.parent.name == ".githooks":
        return "bash"
    match = SHEBANG.match(prefix)
    if match is None:
        return None
    name = match.group(1)
    if name in SHELL_INTERPRETERS:
        return "bash"
    if name.startswith("python"):
        return "python"
    if name == "lua":
        return "lua"
    return None


def python_comment_lines(text):
    lines = set()
    tokens = tokenize.generate_tokens(io.StringIO(text).readline)
    try:
        for token in tokens:
            if token.type != tokenize.COMMENT:
                continue
            if token.start == (1, 0) and token.string.startswith("#!"):
                continue
            lines.add(token.start[0])
    except (tokenize.TokenError, SyntaxError, IndentationError, ValueError):
        pass
    return lines


def is_shebang(node):
    return (
        node.start_point.row == 0
        and node.start_point.column == 0
        and node.text.startswith(b"#!")
    )


def raw_token_starts(source: bytes, standard: str) -> list[tuple[int, int]]:
    with tempfile.NamedTemporaryFile() as handle:
        handle.write(source)
        handle.flush()
        language = "c" if standard == "c23" else "c++"
        result = invoke(
            [
                "clang++",
                "-cc1",
                "-x",
                language,
                "-std=" + standard,
                "-dump-raw-tokens",
                handle.name,
            ],
            Path.cwd(),
        )
        result.check_returncode()
        location = rb"Loc=<" + re.escape(os.fsencode(handle.name)) + rb":(\d+):(\d+)>"
    starts = [0, *(match.end() for match in re.finditer(rb"\n", source))]
    return [
        (int(row), starts[int(row) - 1] + int(column) - 1)
        for row, column in re.findall(location, result.stderr)
    ]


def c_comment_lines(source: bytes, standard: str) -> set[int]:
    if b"/" not in source:
        return set()
    while True:
        tokens = raw_token_starts(source, standard)
        normalized = bytearray(source)
        lines = set()
        at_start = True
        for index, (row, offset) in enumerate(tokens):
            end = tokens[index + 1][1] if index + 1 < len(tokens) else len(source)
            token = source[offset:end]
            comment = COMMENT_START.match(token)
            if comment:
                lines.add(row + token[: comment.start(1)].count(b"\n"))
                continue
            if at_start and (header := HEADER_NAME.match(source, offset)):
                directive = re.sub(SPLICE, b"", header["directive"])
                name = re.sub(SPLICE, b"", header["name"])
                if directive in HEADER_DIRECTIVES and HEADER_QUOTING.search(name):
                    start, stop = header.span("name")
                    normalized[start:stop] = HEADER_CHARACTER.sub(
                        lambda character: character[1] or b" ",
                        header["name"],
                    )
            if not token.strip() or token.startswith(b"\\"):
                at_start = at_start or b"\n" in re.sub(SPLICE, b"", token)
            else:
                at_start = False
        if normalized == source:
            return lines
        source = bytes(normalized)


def comment_lines(kind: str, text: str, standard: str = "c++23") -> set[int]:
    if kind == "python":
        return python_comment_lines(text)
    source = text.encode("utf-8")
    if kind in {"cpp", "cuda"}:
        return c_comment_lines(source, standard)
    grammar = Language(GRAMMARS[kind]())
    tree = Parser(grammar).parse(source)
    query = Query(grammar, COMMENT_QUERIES.get(kind, "(comment) @comment"))
    comments = QueryCursor(query).captures(tree.root_node).get("comment", ())
    return {node.start_point.row + 1 for node in comments if not is_shebang(node)}


def comment_errors(path, code=False):
    with path.open("rb") as handle:
        prefix = handle.read(128).decode("utf-8", errors="replace")
    kind = classify(path, prefix)
    if kind is None:
        if code:
            label = path.suffix or "extensionless"
            return [UNSUPPORTED.format(kind=label)]
        return []
    if kind == "json":
        return []
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return [NOT_UTF8]
    standard = "c23" if path.suffix.lower() == ".c" else "c++23"
    return [COMMENT_FOUND.format(line=line) for line in sorted(comment_lines(kind, text, standard))]
