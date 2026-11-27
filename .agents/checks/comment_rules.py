import io
import re
import tokenize

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
from native_tokens import inspect_tokens
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


def c_comment_lines(source: bytes, standard: str) -> set[int]:
    if b"/" not in source:
        return set()
    lines, _ = inspect_tokens(source, standard)
    return lines


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
