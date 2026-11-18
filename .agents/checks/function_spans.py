import importlib
import json
from pathlib import Path
import sys

from tree_sitter import Language, Parser


GRAMMARS = {
    ".sh": "bash", ".bash": "bash", ".lua": "lua", ".cu": "cuda", ".cuh": "cuda",
    ".cc": "cpp", ".cpp": "cpp", ".cxx": "cpp", ".h": "cpp", ".hpp": "cpp",
}
FUNCTIONS = {"function_definition", "function_declaration", "lambda_expression"}


def spans(path):
    source = path.read_bytes()
    grammar = GRAMMARS[path.suffix or ".sh"]
    module = importlib.import_module(f"tree_sitter_{grammar}")
    tree = Parser(Language(module.language())).parse(source)
    if tree.root_node.has_error:
        raise ValueError(f"{path}: invalid {grammar} syntax or unsupported grammar construct")
    pending = [tree.root_node]
    result = []
    while pending:
        node = pending.pop()
        pending.extend(reversed(node.named_children))
        if node.type not in FUNCTIONS:
            continue
        label = node.child_by_field_name("name") or node.child_by_field_name("declarator")
        name = label.text.decode().splitlines()[0] if label else node.type
        declaration = node
        while declaration.parent and declaration.parent.type == "template_declaration":
            declaration = declaration.parent
        result.append({
            "name": name, "start": declaration.start_point.row + 1,
            "end": node.end_point.row + (1 if node.end_point.column else 0),
        })
    return result


if __name__ == "__main__":
    print(json.dumps(spans(Path(sys.argv[1]))))
