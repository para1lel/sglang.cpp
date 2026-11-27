from __future__ import annotations

import importlib
import json
import re
import sys
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

from tool_process import invoke
from tree_sitter import Language, Parser, Query, QueryCursor

if TYPE_CHECKING:
    from tree_sitter import Node

HEADER_SUFFIXES = {".h", ".hh", ".hpp", ".hxx", ".h++", ".cuh", ".inl", ".ipp", ".tpp"}
GRAMMARS = {
    **dict.fromkeys(HEADER_SUFFIXES | {".cc", ".cpp", ".cxx"}, "cpp"),
    ".c": "c",
    ".sh": "bash",
    ".bash": "bash",
    ".lua": "lua",
    ".cu": "cuda",
    ".cuh": "cuda",
}
SPAN_QUERIES = {
    "cpp": "[(function_definition) (lambda_expression)] @function",
    "cuda": "[(function_definition) (lambda_expression)] @function",
    "lua": "[(function_definition) (function_declaration)] @function",
}
HEADER_QUERY = """
    [(function_definition) (lambda_expression)] @definition
    [(init_declarator value: [(delete_expression) (identifier)])
     (declaration default_value: [(delete_expression) (identifier)])] @initialized
    [(preproc_def value: (preproc_arg) @replacement)
     (preproc_function_def value: (preproc_arg) @replacement)]
    (preproc_include path: [(string_literal) (system_lib_string)] @include)
    [(MISSING) (ERROR)] @invalid
"""
QUALIFIER_QUERY = """
    (preproc_include path: [(string_literal) (system_lib_string)] @include)
    [(preproc_def name: (identifier)) (preproc_function_def name: (identifier))] @macro
"""
CUDA_IDENTIFIER_QUERY = """
    [(identifier) (type_identifier) (field_identifier) (namespace_identifier)] @identifier
"""
CUDA_QUALIFIERS = {
    b"__host__",
    b"__device__",
    b"__global__",
    b"__forceinline__",
    b"__inline_hint__",
}
EXPLICIT_CLASS_QUERY = '"template" @template'
TRIVIA = rb"(?:\s|/\*.*?\*/|//[^\n]*\n)*"
CLASS_START = re.compile(TRIVIA + rb"(?:class|struct|union)\b", re.DOTALL)
EXTERN_PREFIX = re.compile(rb"(?<!\w)(extern)" + TRIVIA + rb"\Z", re.DOTALL)
CLASS_SPECIFIERS = {"class_specifier", "struct_specifier", "union_specifier"}
DECLARATOR_OPERATORS = {
    "array_declarator",
    "function_declarator",
    "pointer_declarator",
    "pointer_type_declarator",
    "reference_declarator",
}


def parser_for(path):
    grammar = GRAMMARS[path.suffix.lower() or ".sh"]
    module = importlib.import_module(f"tree_sitter_{grammar}")
    return Parser(Language(module.language()))


@lru_cache
def query(language: Language, expression: str) -> Query:
    return Query(language, expression)


def captures(parser: Parser, root: Node, expression: str) -> dict[str, list[Node]]:
    return QueryCursor(query(parser.language, expression)).captures(root)


def function_nodes(root: Node, parser: Parser, grammar: str) -> list[Node]:
    expression = SPAN_QUERIES.get(grammar, "(function_definition) @function")
    return sorted(
        captures(parser, root, expression).get("function", ()),
        key=lambda node: (node.start_byte, -node.end_byte),
    )


def function_span(node):
    label = node.child_by_field_name("name") or node.child_by_field_name("declarator")
    name = label.text.decode().splitlines()[0] if label and label.text else node.type
    declaration = node
    while declaration.parent and declaration.parent.type == "template_declaration":
        declaration = declaration.parent
    return {
        "name": name,
        "start": declaration.start_point.row + 1,
        "end": node.end_point.row + (1 if node.end_point.column else 0),
    }


def declarator_operator(declarator):
    nearest = None
    while declarator:
        if declarator.type == "operator_cast":
            return "function_declarator"
        if declarator.type in DECLARATOR_OPERATORS:
            nearest = declarator.type
        child = declarator.child_by_field_name("declarator")
        if declarator.type in {"parenthesized_declarator", "reference_declarator"}:
            child = next((item for item in declarator.named_children if not item.is_extra), None)
        elif declarator.type == "qualified_identifier":
            child = declarator.child_by_field_name("name")
        declarator = child
    return nearest


def present_tokens(node):
    return [token.type for token in node.children if not token.is_extra and not token.is_missing]


def deleted_method(node: Node) -> bool:
    return any(
        child.type == "delete_method_clause"
        and not child.has_error
        and present_tokens(child) == ["=", "delete", ";"]
        for child in node.named_children
    )


def header_findings(
    parser: Parser, root: Node
) -> tuple[dict[str, list[Node]], list[Node], int | None]:
    matches = captures(parser, root, HEADER_QUERY)
    definitions = [node for node in matches.get("definition", ()) if not deleted_method(node)]
    valid_deletes = set()
    for declaration in matches.get("initialized", ()):
        value = declaration.child_by_field_name("value")
        value = value or declaration.child_by_field_name("default_value")
        if value.type != "delete_expression" and value.text != b"default":
            continue
        declarator = declaration.child_by_field_name("declarator")
        if declarator_operator(declarator) != "function_declarator":
            continue
        if value.type == "delete_expression" and present_tokens(value) == ["delete"]:
            valid_deletes.add(value)
        else:
            definitions.append(declaration)
    invalid = [
        node.start_point.row + 1
        for node in matches.get("invalid", ())
        if not (node.is_missing and node.parent in valid_deletes)
    ]
    return matches, definitions, min(invalid, default=None)


def explicit_class_prefixes(source, root, parser):
    for token in captures(parser, root, EXPLICIT_CLASS_QUERY).get("template", ()):
        if not CLASS_START.match(source, token.end_byte):
            continue
        tree = parser.parse(source[token.end_byte :]).root_node
        declaration = next((node for node in tree.named_children if not node.is_extra), None)
        if (
            declaration is None
            or declaration.has_error
            or declaration.type not in CLASS_SPECIFIERS
            or not declaration.child_by_field_name("name")
            or declaration.child_by_field_name("body")
            or declaration.next_sibling is None
            or declaration.next_sibling.is_missing
            or declaration.next_sibling.type != ";"
        ):
            continue
        previous = EXTERN_PREFIX.search(source[: token.start_byte])
        if previous:
            yield previous.span(1)
        yield token.start_byte, token.end_byte


def mask_token(source, node):
    source[node.start_byte:node.end_byte] = b" " * (node.end_byte - node.start_byte)


def included_header(include, owner, repository):
    if include.type == "string_literal":
        directories = (owner.parent, repository / "include", repository)
    elif include.type == "system_lib_string":
        directories = (repository / "include",)
    else:
        return None
    name = Path(include.text[1:-1].decode())
    if name.suffix.lower() not in HEADER_SUFFIXES:
        return None
    for directory in directories:
        candidate = (directory / name).resolve()
        if candidate.is_relative_to(repository) and candidate.is_file():
            return candidate
    return None


def qualifier_macros(path, repository, root, parser):
    pending = [(path.resolve(), root)]
    seen = {path.resolve()}
    aliases = set()
    other = set()
    while pending:
        owner, tree = pending.pop()
        matches = captures(parser, tree, QUALIFIER_QUERY)
        for include in matches.get("include", ()):
            header = included_header(include, owner, repository)
            if header and header not in seen:
                seen.add(header)
                pending.append((header, parser.parse(header.read_bytes()).root_node))
        for node in matches.get("macro", ()):
            name = node.child_by_field_name("name")
            if not name or name.is_missing:
                continue
            value = node.child_by_field_name("value")
            replacement = value.text if value else b""
            tokens = set(replacement.replace(b"\\\r\n", b"").replace(b"\\\n", b"").split())
            if node.type != "preproc_def" or not tokens <= CUDA_QUALIFIERS:
                other.add(name.text)
            elif tokens:
                aliases.add(name.text)
    return aliases - other


def parse_source(path, parser, repository):
    source = path.read_bytes()
    root = parser.parse(source).root_node
    grammar = GRAMMARS[path.suffix.lower() or ".sh"]
    if not root.has_error or grammar not in {"cpp", "cuda"}:
        return root
    if b"template" in source:
        normalized = bytearray(source)
        for start, end in explicit_class_prefixes(source, root, parser):
            normalized[start:end] = b" " * (end - start)
        if normalized != source:
            source = bytes(normalized)
            root = parser.parse(source).root_node
    if not root.has_error or grammar != "cuda":
        return root
    qualifiers = qualifier_macros(path, repository, root, parser) | {b"__inline_hint__"}
    normalized = bytearray(source)
    for node in captures(parser, root, CUDA_IDENTIFIER_QUERY).get("identifier", ()):
        if node.parent.type.startswith("preproc_"):
            continue
        if node.text in qualifiers:
            mask_token(normalized, node)
        elif node.text == b"__local_maxnreg__":
            normalized[node.start_byte : node.end_byte] = b"__launch_bounds__"
    return parser.parse(bytes(normalized)).root_node if normalized != source else root


def macro_definitions(value, parser):
    replacement = parser.parse(value.text).root_node
    _, definitions, _ = header_findings(parser, replacement)
    for node in definitions:
        if node.type == "function_definition":
            operator = declarator_operator(node.child_by_field_name("declarator"))
            if operator and operator != "function_declarator":
                continue
        yield value.start_point.row + function_span(node)["start"]


def expand_macros(path: Path, repository: Path, source: bytes) -> tuple[bytes, list[int]]:
    overlay = {
        "version": 0,
        "fallthrough": False,
        "roots": [
            {
                "type": "directory-remap",
                "name": str(repository),
                "external-contents": str(repository),
            },
        ],
    }
    with tempfile.NamedTemporaryFile() as configuration:
        configuration.write(json.dumps(overlay).encode())
        configuration.flush()
        result = invoke(
            [
                "clang++",
                "-cc1",
                "-x",
                "c++",
                "-std=c++23",
                "-E",
                "-undef",
                "-nostdsysteminc",
                "-nobuiltininc",
                "-ferror-limit",
                "0",
                "-ivfsoverlay",
                configuration.name,
                "-iquote",
                str(path.parent.resolve()),
                "-I",
                str(repository / "include"),
                "-I",
                str(repository),
                "-",
            ],
            repository,
            source,
        )
    lines = []
    original_rows = []
    row = 0
    for line in result.stdout.splitlines(keepends=True):
        marker = re.fullmatch(rb'# (\d+) "([^"]+)".*\n', line)
        if marker:
            row = int(marker[1]) if marker[2] == b"<stdin>" else 0
        elif row:
            lines.append(line)
            original_rows.append(row)
            row += 1
    return b"".join(lines), original_rows


def scan_header(path, parser, repository=None):
    repository = Path.cwd() if repository is None else repository.resolve()
    root = parse_source(path, parser, repository)
    matches, direct, error = header_findings(parser, root)
    definitions = [function_span(node)["start"] for node in direct]
    for value in matches.get("replacement", ()):
        definitions.extend(macro_definitions(value, parser))
    if (
        matches.get("replacement")
        or error is not None
        or any(included_header(node, path, repository) for node in matches.get("include", ()))
    ):
        expanded, rows = expand_macros(path, repository, root.text)
        if expanded:
            _, indirect, _ = header_findings(parser, parser.parse(expanded).root_node)
            definitions.extend(rows[function_span(node)["start"] - 1] for node in indirect)
    grammar = GRAMMARS[path.suffix.lower()]
    measured = [function_span(node) for node in function_nodes(root, parser, grammar)]
    return definitions, measured, error


def spans(path):
    grammar = GRAMMARS[path.suffix.lower() or ".sh"]
    parser = parser_for(path)
    if path.suffix.lower() in HEADER_SUFFIXES:
        _, measured, error = scan_header(path, parser)
        if error is None:
            return measured
    else:
        root = parse_source(path, parser, Path.cwd())
        if not root.has_error:
            return [function_span(node) for node in function_nodes(root, parser, grammar)]
    raise ValueError(f"{path}: invalid {grammar} syntax or unsupported grammar construct")


if __name__ == "__main__":
    print(json.dumps(spans(Path(sys.argv[1]))))
