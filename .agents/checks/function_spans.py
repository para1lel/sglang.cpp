import importlib
import json
from pathlib import Path
import sys

from tree_sitter import Language, Parser


HEADER_SUFFIXES = {".h", ".hh", ".hpp", ".hxx", ".h++", ".cuh", ".inl", ".ipp", ".tpp"}
GRAMMARS = {
    **dict.fromkeys(HEADER_SUFFIXES | {".cc", ".cpp", ".cxx"}, "cpp"),
    ".sh": "bash", ".bash": "bash", ".lua": "lua", ".cu": "cuda", ".cuh": "cuda",
}
FUNCTIONS = {"function_definition", "function_declaration", "lambda_expression"}
MACROS = {"preproc_def", "preproc_function_def"}
CUDA_QUALIFIERS = {
    b"__host__", b"__device__", b"__global__", b"__forceinline__", b"__inline_hint__",
}
IDENTIFIERS = {"identifier", "type_identifier", "field_identifier", "namespace_identifier"}
CLASS_SPECIFIERS = {"class_specifier", "struct_specifier", "union_specifier"}
CLASS_KEYWORDS = {b"class", b"struct", b"union"}
DECLARATOR_OPERATORS = {
    "array_declarator", "function_declarator", "pointer_declarator",
    "pointer_type_declarator", "reference_declarator",
}


def parser_for(path):
    grammar = GRAMMARS[path.suffix.lower() or ".sh"]
    module = importlib.import_module(f"tree_sitter_{grammar}")
    return Parser(Language(module.language()))


def nodes(root, named=True):
    pending = [root]
    while pending:
        node = pending.pop()
        pending.extend(reversed(node.named_children if named else node.children))
        yield node


def function_span(node):
    label = node.child_by_field_name("name") or node.child_by_field_name("declarator")
    name = label.text.decode().splitlines()[0] if label and label.text else node.type
    declaration = node
    while declaration.parent and declaration.parent.type == "template_declaration":
        declaration = declaration.parent
    return {
        "name": name, "start": declaration.start_point.row + 1,
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


def header_definition(node):
    if node.type == "function_definition":
        for child in node.named_children:
            if child.type == "delete_method_clause":
                if present_tokens(child) == ["=", "delete", ";"] and not child.has_error:
                    return "delete"
        return "function"
    if node.type == "lambda_expression":
        return "lambda"
    if node.type in {"init_declarator", "declaration"}:
        value = node.child_by_field_name("value") or node.child_by_field_name("default_value")
        declarator = node.child_by_field_name("declarator")
        if not value or declarator_operator(declarator) != "function_declarator":
            return None
        if value.type == "delete_expression":
            return "delete" if present_tokens(value) == ["delete"] else "function"
        if value.type == "identifier" and value.text == b"default":
            return "function"
    return None


def header_syntax_error(root):
    if not root.has_error:
        return None
    for node in nodes(root, named=False):
        if not node.is_error and not node.is_missing:
            continue
        parent = node.parent
        if (node.is_missing and parent.type == "delete_expression"
                and header_definition(parent.parent) == "delete"):
            continue
        return node.start_point.row + 1
    return None


def explicit_class_prefixes(source, root, parser):
    tokens = [node for node in nodes(root, named=False)
              if not node.child_count and not node.is_missing and node.type != "comment"]
    for index, token in enumerate(tokens[:-1]):
        keyword = tokens[index + 1]
        if token.type != "template" or keyword.text not in CLASS_KEYWORDS:
            continue
        end = next((tokens[i] for i in range(index + 2, len(tokens))
                    if tokens[i].type == ";"), None)
        if end is None:
            continue
        fragment = parser.parse(source[keyword.start_byte:end.end_byte]).root_node
        declarations = [node for node in fragment.named_children if not node.is_extra]
        if fragment.has_error or len(declarations) != 1:
            continue
        declaration = declarations[0]
        if (declaration.type not in CLASS_SPECIFIERS or not declaration.child_by_field_name("name")
                or declaration.child_by_field_name("body")):
            continue
        if index and tokens[index - 1].type == "extern":
            yield tokens[index - 1]
        yield token


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
        for node in nodes(tree):
            if node.type == "preproc_include":
                include = node.child_by_field_name("path")
                header = included_header(include, owner, repository) if include else None
                if header and header not in seen:
                    seen.add(header)
                    pending.append((header, parser.parse(header.read_bytes()).root_node))
            if node.type not in MACROS:
                continue
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
        for token in explicit_class_prefixes(source, root, parser):
            mask_token(normalized, token)
        if normalized != source:
            source = bytes(normalized)
            root = parser.parse(source).root_node
    if not root.has_error or grammar != "cuda":
        return root
    qualifiers = qualifier_macros(path, repository, root, parser) | {b"__inline_hint__"}
    normalized = bytearray(source)
    for node in nodes(root):
        if node.type not in IDENTIFIERS or node.parent.type.startswith("preproc_"):
            continue
        if node.text in qualifiers:
            mask_token(normalized, node)
        elif node.text == b"__local_maxnreg__":
            normalized[node.start_byte:node.end_byte] = b"__launch_bounds__"
    return parser.parse(bytes(normalized)).root_node if normalized != source else root


def macro_definitions(value, parser):
    replacement = parser.parse(value.text).root_node
    for node in nodes(replacement):
        kind = header_definition(node)
        if not kind or kind == "delete":
            continue
        if node.type == "function_definition":
            declarator = node.child_by_field_name("declarator")
            operator = declarator_operator(declarator)
            if operator and operator != "function_declarator":
                continue
        yield value.start_point.row + function_span(node)["start"]


def scan_header(path, parser, repository=None):
    root = parse_source(path, parser, Path.cwd() if repository is None else repository)
    definitions = []
    measured = []
    for node in nodes(root):
        kind = header_definition(node)
        span = function_span(node) if kind or node.type in FUNCTIONS else None
        if node.type in FUNCTIONS:
            measured.append(span)
        if kind and kind != "delete":
            definitions.append(span["start"])
        if node.type in MACROS:
            value = node.child_by_field_name("value")
            if value is not None:
                definitions.extend(macro_definitions(value, parser))
    return definitions, measured, header_syntax_error(root)


def spans(path):
    parser = parser_for(path)
    if path.suffix.lower() in HEADER_SUFFIXES:
        _, measured, error = scan_header(path, parser)
        if error is None:
            return measured
    else:
        root = parse_source(path, parser, Path.cwd())
        if not root.has_error:
            return [function_span(node) for node in nodes(root) if node.type in FUNCTIONS]
    grammar = GRAMMARS[path.suffix.lower() or ".sh"]
    raise ValueError(f"{path}: invalid {grammar} syntax or unsupported grammar construct")


if __name__ == "__main__":
    print(json.dumps(spans(Path(sys.argv[1]))))
