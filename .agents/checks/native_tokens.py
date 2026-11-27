import os
import re
import tempfile
from pathlib import Path

from tool_process import invoke

SPLICE = rb"\\[ \t]*\r?\n"
COMMENT_START = re.compile(rb"(?:" + SPLICE + rb")*(/)(?:" + SPLICE + rb")*[/*]")
DIRECTIVE_SPACE = rb"(?:[ \t\v\f]|" + SPLICE + rb"|/\*.*?\*/)*"
DIRECTIVE_IDENTIFIER = rb"[A-Za-z_0-9$\x80-\xff]+"
DIRECTIVE = re.compile(
    rb"(?:#|%:)"
    + DIRECTIVE_SPACE
    + rb"("
    + DIRECTIVE_IDENTIFIER
    + rb"(?:(?:"
    + SPLICE
    + rb")+"
    + DIRECTIVE_IDENTIFIER
    + rb")*)",
    re.DOTALL,
)
ANGLE_NAME = rb"<(?P<name>(?:" + SPLICE + rb"|[^>\r\n])+)>"
QUOTED_NAME = rb'"(?:' + SPLICE + rb'|[^"\r\n])+"'
HEADER_NAME = re.compile(DIRECTIVE_SPACE + ANGLE_NAME, re.DOTALL)
HAS_INCLUDE_NAME = re.compile(
    DIRECTIVE_SPACE
    + rb"\("
    + DIRECTIVE_SPACE
    + rb"(?P<operand>"
    + ANGLE_NAME
    + rb"|"
    + QUOTED_NAME
    + rb")"
    + DIRECTIVE_SPACE
    + rb"\)",
    re.DOTALL,
)
HEADER_CHARACTER = re.compile(rb"(" + SPLICE + rb")|[^\n]")
HEADER_QUOTING = re.compile(rb"[\"']|//|/\*")
HEADER_DIRECTIVES = {b"include", b"include_next", b"import", b"embed"}
CONDITIONAL_DIRECTIVES = {b"if", b"elif"}
INCLUDE_PREDICATES = {b"__has_include", b"__has_include_next"}


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


def blank_characters(source: bytes) -> bytes:
    return HEADER_CHARACTER.sub(lambda character: character[1] or b" ", source)


def mask_header(normalized: bytearray, header: re.Match[bytes] | None) -> None:
    if header is None or header["name"] is None:
        return
    if HEADER_QUOTING.search(re.sub(SPLICE, b"", header["name"])):
        start, stop = header.span("name")
        normalized[start:stop] = blank_characters(header["name"])


def inspect_tokens(source: bytes, standard: str) -> tuple[set[int], set[tuple[int, int]]]:
    operands = set()
    while True:
        tokens = raw_token_starts(source, standard)
        normalized = bytearray(source)
        lines = set()
        at_start = True
        conditional = False
        for index, (row, offset) in enumerate(tokens):
            end = tokens[index + 1][1] if index + 1 < len(tokens) else len(source)
            token = source[offset:end]
            comment = COMMENT_START.match(token)
            if comment:
                lines.add(row + token[: comment.start(1)].count(b"\n"))
                continue
            if at_start and (directive := DIRECTIVE.match(source, offset)):
                kind = re.sub(SPLICE, b"", directive[1])
                conditional = kind in CONDITIONAL_DIRECTIVES
                if kind in HEADER_DIRECTIVES:
                    mask_header(normalized, HEADER_NAME.match(source, directive.end()))
            if (
                conditional
                and re.sub(SPLICE, b"", token) in INCLUDE_PREDICATES
                and (operand := HAS_INCLUDE_NAME.match(source, end))
            ):
                mask_header(normalized, operand)
                operands.add(operand.span("operand"))
            whitespace = not token.strip() or token.startswith(b"\\")
            if whitespace and b"\n" in re.sub(SPLICE, b"", token):
                at_start = True
                conditional = False
            elif not whitespace:
                at_start = False
        if normalized == source:
            return lines, operands
        source = bytes(normalized)


def normalize_include_predicates(source: bytes, standard: str) -> bytes:
    if b"__has_include" not in re.sub(SPLICE, b"", source):
        return source
    _, operands = inspect_tokens(source, standard)
    if not operands:
        return source
    normalized = bytearray(source)
    for start, stop in operands:
        normalized[start:stop] = b"0" + blank_characters(source[start + 1 : stop])
    return bytes(normalized)
