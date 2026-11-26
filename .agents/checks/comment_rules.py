import io
import re
import tokenize

from tree_sitter import Language, Parser

import tree_sitter_bash
import tree_sitter_cpp
import tree_sitter_cuda
import tree_sitter_go
import tree_sitter_java
import tree_sitter_javascript
import tree_sitter_lua
import tree_sitter_powershell
import tree_sitter_rust
import tree_sitter_toml
import tree_sitter_typescript
import tree_sitter_yaml

C_KINDS = {".c", ".c++", ".cc", ".cpp", ".cxx", ".h", ".h++", ".hh", ".hpp", ".hxx"}
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
    ".rs": "rust",
    ".toml": "toml",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".yaml": "yaml",
    ".yml": "yaml",
}
GRAMMARS = {
    "bash": tree_sitter_bash.language,
    "cpp": tree_sitter_cpp.language,
    "cuda": tree_sitter_cuda.language,
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
COMMENT_TYPES = {
    **dict.fromkeys(["java", "rust"], {"line_comment", "block_comment"}),
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


def skip_quoted(logical, index):
    length = len(logical)
    position = index + 1
    while position < length:
        char = logical[position]
        if char == "\\" and position + 1 < length:
            position += 2
            continue
        if char == "\n":
            return position
        if char == "\"":
            return position + 1
        position += 1
    return position


def skip_char(logical, index):
    length = len(logical)
    position = index + 1
    limit = min(length, index + 9)
    while position < limit:
        char = logical[position]
        if char == "\n":
            break
        if char == "\\":
            position += 2
            continue
        if char == "'":
            return position + 1
        position += 1
    return index + 1


def raw_prefix(logical, index):
    start = index
    while start > 0 and (logical[start - 1].isalnum() or logical[start - 1] == "_"):
        start -= 1
    return logical[start:index] in {"R", "uR", "UR", "LR", "u8R"}


def raw_string_end(logical, index):
    opening = logical.find("(", index + 1)
    if opening < 0 or opening - index > 17:
        return -1
    delimiter = logical[index + 1:opening]
    if any(character in " \t\n()\\" for character in delimiter):
        return -1
    closing = ")" + delimiter + "\""
    end = logical.find(closing, opening + 1)
    if end < 0:
        return -1
    return end + len(closing)


def skip_literal(logical, index, raw_strings):
    if logical[index] == "'":
        return skip_char(logical, index)
    if raw_strings and raw_prefix(logical, index):
        end = raw_string_end(logical, index)
        if end >= 0:
            return end
    return skip_quoted(logical, index)


def splice_lines(fragment):
    logical = []
    positions = []
    index = 0
    length = len(fragment)
    row = 0
    column = 0
    while index < length:
        if fragment.startswith("\\\r\n", index):
            index += 3
            row += 1
            column = 0
            continue
        if fragment.startswith("\\\n", index):
            index += 2
            row += 1
            column = 0
            continue
        char = fragment[index]
        logical.append(char)
        positions.append((row, column))
        index += 1
        if char == "\n":
            row += 1
            column = 0
        else:
            column += 1
    return "".join(logical), positions


def skip_directive(logical, index):
    length = len(logical)
    position = index + 1
    while position < length and logical[position] in " \t":
        position += 1
    if not logical.startswith("include", position):
        return index + 1
    position += len("include")
    while position < length and logical[position] in " \t":
        position += 1
    if position >= length or logical[position] != "<":
        return index + 1
    end = position
    while end < length and logical[end] not in ">\n":
        end += 1
    return end + 1 if end < length and logical[end] == ">" else end


def record_comment(positions, index, base_row, base_column, found):
    row, column = positions[index]
    if row == 0:
        column += base_column
    found[(base_row + row, column)] = base_row + row + 1


def lex_logical(logical, positions, base_row, base_column, found, raw_strings):
    index = 0
    length = len(logical)
    while index < length:
        char = logical[index]
        if char in "\"'":
            index = skip_literal(logical, index, raw_strings)
            continue
        if char == "#":
            index = skip_directive(logical, index)
            continue
        if logical.startswith("//", index):
            record_comment(positions, index, base_row, base_column, found)
            end = logical.find("\n", index)
            if end < 0:
                return
            index = end + 1
            continue
        if logical.startswith("/*", index):
            record_comment(positions, index, base_row, base_column, found)
            end = logical.find("*/", index + 2)
            if end < 0:
                return
            index = end + 2
            continue
        index += 1


def tree_comment_found(kind, text):
    tree = Parser(Language(GRAMMARS[kind]())).parse(text.encode("utf-8"))
    types = COMMENT_TYPES.get(kind, {"comment"})
    found = {}
    pending = [tree.root_node]
    while pending:
        node = pending.pop()
        pending.extend(reversed(node.children))
        if node.type in types and not is_shebang(node):
            point = node.start_point
            found[(point.row, point.column)] = point.row + 1
    return found


def comment_lines(kind, text, raw_strings=True):
    if kind == "python":
        return python_comment_lines(text)
    if kind in {"cpp", "cuda"}:
        found = {}
        logical, positions = splice_lines(text)
        lex_logical(logical, positions, 0, 0, found, raw_strings)
        return set(found.values())
    return set(tree_comment_found(kind, text).values())


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
    suffix = path.suffix.lower()
    raw_strings = kind == "cuda" or (kind == "cpp" and suffix != ".c")
    return [
        COMMENT_FOUND.format(line=line)
        for line in sorted(comment_lines(kind, text, raw_strings))
    ]
