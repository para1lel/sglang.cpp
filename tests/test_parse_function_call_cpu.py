"""Exercise standalone Qwen tool parsing through the native HTTP listener."""

import argparse
from contextlib import closing
import http.client
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent / "text"))
from test_sampling_routes_cpu import receipts, send, server


ROUTE = "/parse_function_call"
TOOLS = [
    {"function": {"name": "ping"}},
    {"function": {"name": "lookup", "parameters": {
        "type": "object",
        "$defs": {"Lookup": {"type": "object", "required": ["hours"]}},
        "properties": {
            "city": {"type": "string"},
            "days": {"type": "integer"},
            "active": {"type": "boolean"},
            "filters": {"type": "object"},
            "structured": {"anyOf": [{"type": "string"}, {"type": "object"}]},
            "local_ref": {"anyOf": [{"type": "string"}, {"$ref": "#/$defs/Lookup"}]},
            "minimum_items": {"anyOf": [{"type": "string"}, {"type": "array", "minItems": 2}]},
            "untyped": {},
        },
    }}},
]
LOOKUP = (
    "<tool_call>\n<function=lookup>\n"
    "<parameter=city>\nZürich\n</parameter>\n"
    "<parameter=days>2</parameter>\n"
    "<parameter=active>false</parameter>\n"
    '<parameter=filters>{"hours":[8,9]}</parameter>\n'
    "</function>\n</tool_call>"
)
PING = "<tool_call><function=ping></function></tool_call>"
PREFIX = "  Checking.\n"
VALID = {"text": PREFIX + LOOKUP + PING + "\ntrailing", "tools": TOOLS,
         "tool_call_parser": "qwen3_coder"}
EXPECTED = [
    {"tool_index": 0, "name": "lookup", "parameters": {
        "city": "Zürich", "days": 2, "active": False, "filters": {"hours": [8, 9]},
    }},
    {"tool_index": 1, "name": "ping", "parameters": {}},
]


def successful_cases():
    for parser in ("qwen3_coder", "step3p5", "nanbeige"):
        yield parser, VALID | {"tool_call_parser": parser}, PREFIX, EXPECTED
    yield "ordinary text", VALID | {"text": "  雪 and rain\n"}, "  雪 and rain\n", []
    yield "omitted tools", {"text": PING, "tool_call_parser": "qwen3_coder"}, PING, []
    yield "malformed function", VALID | {
        "text": "Prefix<tool_call><function=broken</function></tool_call>",
    }, "Prefix", []
    yield "unknown function", VALID | {
        "text": "<tool_call><function=unlisted><parameter=days>2</parameter>"
                "</function></tool_call>",
    }, "", [{"tool_index": 0, "name": "unlisted", "parameters": {"days": "2"}}]
    yield "duplicate parameter and truncation", VALID | {
        "text": "<tool_call><function=lookup><parameter=days>2</parameter>"
                "<parameter=days>4",
    }, "", [{"tool_index": 0, "name": "lookup", "parameters": {"days": 4}}]
    for name, parameter, value, expected in (
        ("structured union", "structured", '{"hours":[8,9]}', {"hours": [8, 9]}),
        ("object literal", "filters", "{'hours': [8, 9]}", {"hours": [8, 9]}),
        ("union string", "structured", "open now", "open now"),
        ("untyped string", "untyped", "True", "True"),
        ("local ref match", "local_ref", '{"hours":[8,9]}', {"hours": [8, 9]}),
        ("local ref mismatch", "local_ref", '{"minutes":[8,9]}', '{"minutes":[8,9]}'),
        ("minItems match", "minimum_items", "[8,9]", [8, 9]),
        ("minItems mismatch", "minimum_items", "[8]", "[8]"),
    ):
        yield name, VALID | {
            "text": f"<tool_call><function=lookup><parameter={parameter}>{value}"
                    "</parameter></function></tool_call>",
        }, "", [{"tool_index": 0, "name": "lookup", "parameters": {parameter: expected}}]


def validation_cases():
    yield "invalid JSON", b"{", "JSON decode error"
    yield "missing text", {"tool_call_parser": "qwen3_coder"}, "text"
    yield "wrong text type", VALID | {"text": 42}, "text"
    yield "wrong tools type", VALID | {"tools": {}}, "tools"
    yield "missing tool function", VALID | {"tools": [{"type": "function"}]}, "function"
    yield "wrong parser type", VALID | {"tool_call_parser": 42}, "tool_call_parser"


def selection_error_cases():
    yield "unknown parser", VALID | {"tool_call_parser": "no-such-parser"}, (
        "Unsupported tool_call_parser: no-such-parser"
    )
    yield "omitted parser", {key: value for key, value in VALID.items()
                            if key != "tool_call_parser"}, "Unsupported tool_call_parser: None"
    yield "unserved model parser", VALID | {"tool_call_parser": "deepseek-v4"}, (
        "'deepseek-v4' output is parsed with its model's tokenizer; "
        "this server does not serve that model"
    )


class StandaloneToolParsing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.log = cls.output / "standalone.log"
        cls.port = cls.enterClassContext(server(cls.binary, cls.checkpoint, cls.log))

    def tearDown(self):
        self.assertEqual(receipts(self.log), [])

    def test_parsed_output(self):
        for name, payload, normal_text, calls in successful_cases():
            with self.subTest(case=name):
                status, response = send(self.port, ROUTE, payload)
                self.assertEqual(status, 200, response)
                for call in response["calls"]:
                    self.assertIsInstance(call["parameters"], str)
                    call["parameters"] = json.loads(call["parameters"])
                self.assertEqual(json.dumps(response, sort_keys=True), json.dumps({
                    "normal_text": normal_text, "calls": calls,
                }, sort_keys=True))

    def test_request_validation(self):
        for name, payload, diagnostic in validation_cases():
            with self.subTest(case=name):
                status, body = send(self.port, ROUTE, payload)
                self.assertEqual(status, 400, body)
                self.assertIn(diagnostic, body.pop("message"))
                self.assertEqual(body, {"object": "error", "type": "Bad Request",
                                        "param": None, "code": 400})

    def test_parser_selection(self):
        for name, payload, message in selection_error_cases():
            with self.subTest(case=name):
                self.assertEqual(send(self.port, ROUTE, payload), (400, {
                    "object": "error", "type": "400", "param": None, "code": 400,
                    "message": message,
                }))

    def test_get_is_rejected(self):
        with closing(http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)) as conn:
            conn.request("GET", ROUTE)
            with conn.getresponse() as response:
                self.assertEqual((response.status, response.getheader("Allow")), (405, "POST"))


if __name__ == "__main__":
    arguments = argparse.ArgumentParser()
    arguments.add_argument("--binary", type=Path, required=True)
    arguments.add_argument("--checkpoint", type=Path, required=True)
    arguments.add_argument("--output", type=Path, required=True)
    options = arguments.parse_args()
    options.output.mkdir(parents=True, exist_ok=True)
    StandaloneToolParsing.binary = options.binary.resolve()
    StandaloneToolParsing.checkpoint = options.checkpoint.resolve()
    StandaloneToolParsing.output = options.output.resolve()
    unittest.main(argv=[__file__], verbosity=2)
