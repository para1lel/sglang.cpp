"""Verify pinned SGLang count_tokens behavior with the Qwen3.8 CPU server."""

import argparse
from contextlib import closing
from http.client import HTTPConnection
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent / "text"))
from test_native_tokenization_cpu import post
from test_sampling_routes_cpu import receipts, server


ROUTE = "/v1/messages/count_tokens"
USER = {"role": "user", "content": "Hello, 你好 🌍 café"}
FOLLOW_UP = {"role": "user", "content": "next"}
THINKING = {"role": "assistant", "content": [
    {"type": "thinking", "thinking": "Private analysis", "signature": "opaque"},
    {"type": "text", "text": "Before"},
]}
CALL = {"role": "assistant", "content": [{
    "type": "tool_use", "name": "lookup", "id": "call-1",
    "input": {"key": "你好", "wide": 18446744073709551615,
              "lossy": 18446744073709551616, "nested": {"x": [True, None]}},
}]}
RESULT = {"role": "user", "content": [{
    "type": "tool_result", "tool_use_id": "call-1",
    "content": [{"type": "text", "text": "Before"}, {"type": "text", "text": " after "}],
}]}
TOOLS = [
    {"name": "lookup", "description": "Café <test>", "strict": True, "defer_loading": True,
     "input_schema": {"properties": {"query": {"type": "string"}}}},
    {"name": "ping", "input_schema": {"type": "object", "properties": {}}},
]


def request(*messages, **options):
    return {"model": "Qwen3.8-27B-FP8", "messages": list(messages), **options}


class AnthropicCount(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.log = cls.output / "anthropic-count.log"
        cls.port = cls.enterClassContext(server(cls.binary, cls.checkpoint, cls.log))

    def assert_count(self, payload, expected, port=None):
        self.assertEqual(post(port or self.port, ROUTE, payload),
                         (200, {"input_tokens": expected}))

    def assert_invalid(self, payload, message=None, headers=None):
        status, body = (post(self.port, ROUTE, payload) if headers is None else
                        post(self.port, ROUTE, payload, headers))
        self.assertEqual(status, 400, body)
        error = body["error"]
        self.assertIsInstance(error["message"], str)
        self.assertTrue(error["message"])
        self.assertEqual(body, {"type": "error", "error": {
            "type": "invalid_request_error", "message": message or error["message"],
        }})

    def test_count_converted_messages(self):
        inline = {"role": "system", "content": [
            {"type": "text", "text": "  inline  "}, {"type": "text", "text": "\t"},
        ]}
        system = [{"type": "text", "text": text} for text in ("Before", " ", " after ")]
        image = {"role": "user", "content": [
            {"type": "image", "source": {"url": "file:///not-read.png"}},
            {"type": "text", "text": "Before"},
        ]}
        assistant_result = {"role": "assistant", "content": [{
            "type": "tool_result", "content": [{"type": "image", "source": {"url": [1]}}],
        }]}
        for name, payload, expected in (
            ("unicode", request(USER, model="unserved-model", stream=True), 60),
            ("system", request(USER, inline, {"role": "user", "content": "again"},
                               system=system), 72),
            ("image", request(image), 56),
            ("tool history", request(USER, CALL, RESULT, FOLLOW_UP), 195),
            ("terminal tool use", request(USER, CALL), 65),
            ("tool use alone", request(CALL), 52),
            ("assistant tool result media", request(USER, assistant_result, FOLLOW_UP), 78),
            ("thinking history without parser", request(USER, THINKING, FOLLOW_UP), 76),
        ):
            with self.subTest(case=name):
                self.assert_count(payload, expected)
        self.assertEqual(receipts(self.log), [])

    def test_tools_and_output_effort(self):
        for options, expected in (
            ({"tools": TOOLS}, 361),
            ({"tools": TOOLS, "tool_choice": {"type": "tool", "name": "ping"}}, 296),
            ({"tools": TOOLS, "tool_choice": {"type": "none"}}, 60),
            ({"tools": [{"type": "web_search_20250305"}]}, 60),
            ({"output_config": {"effort": "low"}}, 48),
        ):
            with self.subTest(options=options):
                self.assert_count(request(USER, **options), expected)

    def test_server_tool_decimal_versions(self):
        for version in ("２０２５０３０５", "2٠𝟐５０3０5"):
            with self.subTest(version=version):
                self.assert_count(request(USER, tools=[{"type": f"web_search_{version}"}]), 60)
        for version in ("２０２５０３０", "２０２５０３０５５", "２０２５０３０²"):
            with self.subTest(version=version):
                self.assert_invalid(request(USER, tools=[{"type": f"web_search_{version}"}]))

    def test_task_budget_integer_coercion(self):
        def budget(total, remaining):
            return request(USER, output_config={"task_budget": {
                "type": "tokens", "total": total, "remaining": remaining,
            }})

        for total, remaining in (("  +1024  ", "1_000"), ("\u00a0+1_024.00\u00a0", "-0_0"),
                                 ("18446744073709551616", 0)):
            with self.subTest(total=total, remaining=remaining):
                self.assert_count(budget(total, remaining), 60)
        for total, remaining in (("1__024", 0), ("1e3", 0), ("１０２４", 0), ("\x1c1024", 0),
                                 ("1.0000000000000000001", 0), (1e19, 0), (0, 0), (1, "-1_000")):
            with self.subTest(total=total, remaining=remaining):
                self.assert_invalid(budget(total, remaining))
        self.assert_count(budget("9" * 4300, 0), 60)
        message = "Unable to parse input string as an integer, exceeded maximum size"
        self.assert_invalid(budget("9" * 4301, 0), f"output_config.task_budget.total: {message}")

    def test_count_json_float_overflow(self):
        prefix = b'{"model":"m","messages":[{"role":"user","content":"hi"}]'
        self.assert_count(prefix + b',"unused":1e309}', 53)
        self.assert_count(prefix + b',"unused":{"nested":[-1e309]}}', 53)
        self.assert_invalid(prefix + b',"output_config":{"task_budget":'
                            b'{"type":"tokens","total":1e309}}}',
                            "output_config.task_budget.total: Input should be a finite number")

    def test_qwen_reasoning_parser(self):
        log = self.output / "anthropic-count-qwen3.log"
        with server(self.binary, self.checkpoint, log, "--reasoning-parser", "qwen3") as port:
            self.assert_count(request(USER, THINKING, FOLLOW_UP), 81, port)
            self.assert_count(request(USER, thinking={"type": "disabled"}), 20, port)
            self.assert_count(request(USER, thinking={"type": "adaptive"}), 60, port)
            self.assertEqual(receipts(log), [])

    def test_validation(self):
        for payload, message in (
            ({"messages": [USER]}, "model: Field required"),
            ({"model": "m"}, "messages: Field required"),
            (request(), "Cannot apply chat template to an empty conversation. "
                        "Provide at least one message."),
            (request({"role": "user", "content": [{"type": "text"}]}), None),
            (request(USER, tools=TOOLS, tool_choice={"type": "tool", "name": "missing"}), None),
            (request(USER, thinking={"type": "adaptive"}),
             "Anthropic thinking is not supported for models without a reasoning parser"),
            (request(USER, output_config={"format": {"type": "json_schema"}}),
             "output_config.format.schema: Field required"),
            (b"{", None),
        ):
            with self.subTest(payload=payload):
                self.assert_invalid(payload, message)
        self.assert_invalid(request(USER),
                            "header.content-type: Unsupported Media Type: Only "
                            "'application/json' is allowed", {"Content-Type": "text/plain"})
        self.assert_count(request(USER), 60)

    def test_route_methods(self):
        for method, route, status, header in (
            ("GET", ROUTE, 405, "Allow"),
            ("POST", ROUTE + "/", 307, "Location"),
        ):
            with self.subTest(method=method, route=route):
                with closing(HTTPConnection("127.0.0.1", self.port, timeout=5)) as connection:
                    connection.request(method, route)
                    with connection.getresponse() as reply:
                        self.assertEqual(reply.status, status)
                        self.assertTrue(reply.getheader(header))
                        reply.read()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    AnthropicCount.binary = args.binary.resolve()
    AnthropicCount.checkpoint = args.checkpoint.resolve()
    AnthropicCount.output = args.output.resolve()
    AnthropicCount.output.mkdir(parents=True, exist_ok=True)
    unittest.main(argv=[__file__], verbosity=2)
