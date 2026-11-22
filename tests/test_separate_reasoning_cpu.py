"""Exercise standalone reasoning separation through native sockets."""

import argparse
from contextlib import closing
import http.client
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent / "text"))
from test_sampling_routes_cpu import receipts, send, server


ROUTE = "/separate_reasoning"


class StandaloneReasoning(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.log = cls.output / "standalone.log"
        cls.port = cls.enterClassContext(server(cls.binary, cls.checkpoint, cls.log))

    def check_split(self, source, parser, expected, **fields):
        status, body = send(self.port, ROUTE, {
            "text": source, "reasoning_parser": parser, **fields,
        })
        self.assertEqual((status, body), (200, expected))

    def test_parser_modes(self):
        for parser, reasoning, text in (
            ("qwen3", "", "雪</think> answer"),
            ("QwEn3-ThInKiNg", "雪", " answer"),
            ("minimax", "雪", " answer"),
        ):
            with self.subTest(parser=parser):
                self.check_split("雪</think> answer", parser,
                                 {"reasoning_text": reasoning, "text": text},
                                 force_reasoning=True, stream=True)
        self.assertEqual(receipts(self.log), [])

    def test_blocks_and_tool_interrupt(self):
        reasoning = "\r\n雪\x00"
        text = "<tool_call>opaque</think> answer"
        source = f"<think><think>{reasoning}{text}"
        flat = {"reasoning_text": reasoning, "text": text}
        self.check_split(source, "qwen3", flat, return_blocks=False)
        self.check_split(source, "qwen3", {
            **flat, "reasoning_blocks": [reasoning], "text_blocks": [text],
            "blocks": [{"type": "reasoning", "text": reasoning},
                       {"type": "text", "text": text}],
        }, return_blocks=True)
        self.check_split("<think></think>", "qwen3", {
            "reasoning_text": "", "text": "", "reasoning_blocks": [], "text_blocks": [""],
            "blocks": [{"type": "text", "text": ""}],
        }, return_blocks=True)

    def test_validation_and_method(self):
        for payload, diagnostic in (
            (b"{", "JSON decode error"),
            ({"text": "answer"}, "reasoning_parser"),
            ({"text": "answer", "reasoning_parser": "qwen3", "return_blocks": None},
             "return_blocks"),
        ):
            with self.subTest(payload=payload):
                status, body = send(self.port, ROUTE, payload)
                self.assertEqual(status, 400)
                self.assertEqual({key: value for key, value in body.items() if key != "message"},
                                 {"object": "error", "type": "Bad Request",
                                  "param": None, "code": 400})
                self.assertIn(diagnostic, body["message"])
        with closing(http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)) as conn:
            conn.request("GET", ROUTE)
            with conn.getresponse() as response:
                self.assertEqual(response.status, 405)
                self.assertEqual(response.getheader("Allow"), "POST")
                self.assertEqual(json.load(response), {"detail": "Method Not Allowed"})


if __name__ == "__main__":
    arguments = argparse.ArgumentParser()
    arguments.add_argument("--binary", type=Path, required=True)
    arguments.add_argument("--checkpoint", type=Path, required=True)
    arguments.add_argument("--output", type=Path, required=True)
    options = arguments.parse_args()
    options.output.mkdir(parents=True, exist_ok=True)
    StandaloneReasoning.binary = options.binary.resolve()
    StandaloneReasoning.checkpoint = options.checkpoint.resolve()
    StandaloneReasoning.output = options.output.resolve()
    unittest.main(argv=[__file__], verbosity=2)
