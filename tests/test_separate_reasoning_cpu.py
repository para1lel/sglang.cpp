"""Exercise standalone reasoning separation through native sockets."""

import argparse
from contextlib import closing
from email.parser import BytesHeaderParser
import http.client
import json
from pathlib import Path
import socket
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent / "text"))
from test_sampling_routes_cpu import receipts, send, server


ROUTE = "/separate_reasoning"
ENCODED_ROUTE = "/%73eparate_%72easoning"
RAW_QUERY = "?trace=%2f+%252F&trace=%3F?tail"


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

    def tearDown(self):
        self.assertEqual(receipts(self.log), [])

    def wire(self, method, target, body=b"", media=None):
        headers = [f"{method} {target} HTTP/1.1", f"Host: 127.0.0.1:{self.port}",
                   "Connection: close", f"Content-Length: {len(body)}"]
        if media is not None:
            headers.append(f"Content-Type: {media}")
        with socket.create_connection(("127.0.0.1", self.port), timeout=5) as client:
            client.sendall(("\r\n".join(headers) + "\r\n\r\n").encode("ascii") + body)
            with client.makefile("rb") as stream:
                head, boundary, data = stream.read().partition(b"\r\n\r\n")
        self.assertTrue(boundary)
        status_line, fields = head.split(b"\r\n", 1)
        return int(status_line.split()[1]), BytesHeaderParser().parsebytes(fields), data

    def test_decoded_post_and_media_type(self):
        payload = b'{"text":"<think>plan</think>answer","reasoning_parser":"qwen3"}'
        target = ENCODED_ROUTE + RAW_QUERY
        for media in ("application/json", "Application/Vnd.Example+Json; charset=utf-8"):
            with self.subTest(media=media):
                status, headers, data = self.wire("POST", target, payload, media)
                self.assertEqual((status, headers.get_content_type()), (200, "application/json"))
                self.assertEqual(json.loads(data), {"reasoning_text": "plan", "text": "answer"})
        for media in (None, "text/plain"):
            with self.subTest(media=media):
                status, headers, data = self.wire("POST", target, payload, media)
                self.assertEqual((status, headers.get_content_type()), (400, "application/json"))
                error = json.loads(data)
                self.assertTrue(error.pop("message"))
                self.assertEqual(error, {"object": "error", "type": "Bad Request",
                                         "param": None, "code": 400})
        for path in ("/%2573eparate_reasoning", ROUTE + "%3Fignored"):
            with self.subTest(path=path):
                status, _, _ = self.wire("POST", path + RAW_QUERY, payload, "application/json")
                self.assertEqual(status, 404)

    def test_slash_redirect_precedes_method_and_media_validation(self):
        for method, suffix in (("POST", "/"), ("DELETE", "%2F")):
            with self.subTest(method=method, suffix=suffix):
                status, headers, data = self.wire(method, ENCODED_ROUTE + suffix + RAW_QUERY,
                                                  b"ignored", "text/plain")
                self.assertEqual((status, headers["Content-Length"], data), (307, "0", b""))
                self.assertEqual(headers["Location"], f"http://127.0.0.1:{self.port}"
                                 + ROUTE + RAW_QUERY)

    def test_canonical_method_on_wire(self):
        for method in ("HEAD", "PUT", "DELETE", "OPTIONS"):
            with self.subTest(method=method):
                status, headers, data = self.wire(method, ENCODED_ROUTE + RAW_QUERY)
                self.assertEqual((status, headers["Allow"], headers.get_content_type()),
                                 (405, "POST", "application/json"))
                if method == "HEAD":
                    self.assertEqual(data, b"")
                else:
                    self.assertEqual(json.loads(data), {"detail": "Method Not Allowed"})

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
