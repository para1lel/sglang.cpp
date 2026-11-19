"""Loopback integration tests for a running sglang-server with the pinned checkpoint.

Run with python -B tests/test_native_streaming.py --port 17384.
"""

import argparse
from contextlib import closing
import http.client
import json
from pathlib import Path
import select
import socket
import struct
import unittest


FIXTURE = Path(__file__).resolve().parents[1] / "benchmarks/vllm/small-fp8.json"
LOOPBACK = "127.0.0.1"
COMPLETIONS = "/v1/completions"
HEADERS = {
    "Accept": "text/event-stream",
    "Connection": "close",
    "Content-Type": "application/json",
}
SOCKET_TIMEOUT = 12


def completion_request(prompt, budget):
    return {
        "model": "benchmark",
        "prompt": prompt,
        "max_tokens": budget,
        "temperature": 0.0,
        "seed": 0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
        "return_token_ids": True,
        "stream_interval": 1,
        "add_special_tokens": False,
    }


class NativeStreamingTest(unittest.TestCase):
    port = None

    @classmethod
    def setUpClass(cls):
        if cls.port is None:
            raise unittest.SkipTest("Run this script with --port against sglang-server")
        fixture = json.loads(FIXTURE.read_bytes())
        cls.prompt = fixture["prompt_token_ids"]
        cls.expected = fixture["expected_output_ids"]

    def connect(self):
        return http.client.HTTPConnection(LOOPBACK, self.port, timeout=SOCKET_TIMEOUT)

    def post(self, connection, body):
        encoded = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
        connection.request("POST", COMPLETIONS, encoded, HEADERS)

    def start_stream(self, response):
        self.assertEqual(response.status, 200)
        self.assertEqual(response.headers.get_content_type(), "text/event-stream")

    def read_event(self, response):
        line = response.readline()
        self.assertTrue(line.startswith(b"data: "), line)
        self.assertIn(response.readline(), (b"\n", b"\r\n"))
        payload = line.removeprefix(b"data: ").strip()
        return "[DONE]" if payload == b"[DONE]" else json.loads(payload)

    def check_choice(self, response, token_ids, finish_reason, prompt=None):
        event = self.read_event(response)
        self.assertNotIn("error", event)
        self.assertIsNone(event.get("usage"))
        choice, = event["choices"]
        self.assertEqual(choice["index"], 0)
        self.assertEqual(choice["token_ids"], token_ids)
        self.assertEqual(choice["finish_reason"], finish_reason)
        if prompt is None:
            self.assertNotIn("prompt_token_ids", choice)
        else:
            self.assertEqual(choice["prompt_token_ids"], prompt)

    def check_stream(self, response, prompt, expected):
        self.start_stream(response)
        for step, token in enumerate(expected):
            with self.subTest(step=step):
                self.check_choice(response, [token], None, prompt if step == 0 else None)
        self.check_choice(response, [], "length", None if expected else prompt)
        event = self.read_event(response)
        self.assertEqual(event["choices"], [])
        usage = event["usage"]
        self.assertEqual(usage["prompt_tokens"], len(prompt))
        self.assertEqual(usage["completion_tokens"], len(expected))
        self.assertEqual(usage["prompt_tokens_details"]["cached_tokens"], 0)
        self.assertEqual(self.read_event(response), "[DONE]")
        self.assertEqual(response.read(1), b"")

    def check_completion(self, prompt, expected):
        with closing(self.connect()) as connection:
            self.post(connection, completion_request(prompt, len(expected)))
            with connection.getresponse() as response:
                self.check_stream(response, prompt, expected)

    def test_health(self):
        with closing(self.connect()) as connection:
            connection.request("GET", "/health")
            with connection.getresponse() as response:
                self.assertEqual(response.status, 200)
                self.assertEqual(json.load(response), {})

    def test_exact_tokens_and_request_reset(self):
        self.check_completion(self.prompt, self.expected)
        self.check_completion([9419], [11, 353, 2688])
        self.check_completion(self.prompt, self.expected)

    def test_zero_budget(self):
        self.check_completion(self.prompt, [])

    def test_compatible_zero_temperature_spellings(self):
        request = json.dumps(completion_request(self.prompt, 0)).encode()
        for temperature in (b"0e0", b"1e-400", b"-1e-400"):
            with self.subTest(temperature=temperature), closing(self.connect()) as connection:
                body = request.replace(b'"temperature": 0.0', b'"temperature": ' + temperature)
                self.assertNotEqual(body, request)
                self.post(connection, body)
                with connection.getresponse() as response:
                    self.check_stream(response, self.prompt, [])

    def test_client_write_half_close(self):
        for after_headers in (False, True):
            with (
                self.subTest(after_response_headers=after_headers),
                closing(self.connect()) as connection,
            ):
                self.post(connection, completion_request(self.prompt, len(self.expected)))
                client_socket = connection.sock
                if not after_headers:
                    client_socket.shutdown(socket.SHUT_WR)
                with connection.getresponse() as response:
                    if after_headers:
                        client_socket.shutdown(socket.SHUT_WR)
                    self.check_stream(response, self.prompt, self.expected)

    def test_invalid_requests(self):
        valid = completion_request(self.prompt, 1)
        cases = {
            "empty body": b"",
            "malformed JSON": b'{"prompt":',
            "empty prompt": valid | {"prompt": []},
            "text prompt": valid | {"prompt": "hello"},
            "fractional token": valid | {"prompt": [1.5]},
            "token above vocabulary": valid | {"prompt": [248320]},
            "negative budget": valid | {"max_tokens": -1},
            "context overflow": valid | {"max_tokens": 262144},
            "unsupported stream option": valid | {"stream_interval": 2},
            "nonzero temperature": valid | {"temperature": 1e-300},
            "negative nonzero temperature": valid | {"temperature": -1e-300},
            "string temperature": valid | {"temperature": "0"},
        }
        for label, body in cases.items():
            with self.subTest(case=label), closing(self.connect()) as connection:
                self.post(connection, body)
                with connection.getresponse() as response:
                    self.assertEqual(response.status, 400)
                    self.assertEqual(response.headers.get_content_type(), "application/json")
                    self.assertTrue(json.load(response)["error"]["message"])
        self.check_completion(self.prompt, self.expected[:1])

    def test_strict_json_requests(self):
        valid = json.dumps(completion_request(self.prompt, 0)).encode()
        cases = {
            "duplicate key": b'{"seed":0,' + valid[1:],
            "escaped duplicate key": rb'{"s\u0065ed":0,' + valid[1:],
            "leading zero": valid.replace(b'"seed": 0', b'"seed": 00'),
            "leading plus": valid.replace(b'"seed": 0', b'"seed": +0'),
            "missing exponent": valid.replace(b'"temperature": 0.0', b'"temperature": 0e'),
            "invalid UTF-8": valid.replace(b'"benchmark"', b'"ben\xc0\xafchmark"'),
            "unpaired surrogate": valid.replace(b'"benchmark"', rb'"ben\ud800chmark"'),
            "invalid escape": valid.replace(b'"benchmark"', rb'"ben\qchmark"'),
            "unescaped control": valid.replace(b'"benchmark"', b'"ben\x01chmark"'),
            "depth": valid.replace(b'"seed": 0', b'"seed": ' + b"[" * 32 + b"0" + b"]" * 32),
            "trailing data": valid + b"null",
        }
        for label, body in cases.items():
            with self.subTest(case=label), closing(self.connect()) as connection:
                self.assertNotEqual(body, valid)
                self.post(connection, body)
                with connection.getresponse() as response:
                    self.assertEqual(response.status, 400)
                    self.assertEqual(response.headers.get_content_type(), "application/json")
                    self.assertEqual(json.load(response),
                                     {"error": {"message": "Invalid JSON request body"}})

    def test_single_active_request_and_disconnect(self):
        with closing(self.connect()) as queued, closing(self.connect()) as active:
            self.post(active, completion_request(self.prompt, 1024))
            active_socket = active.sock
            with active.getresponse() as response:
                self.start_stream(response)
                self.check_choice(response, self.expected[:1], None, self.prompt)
                self.post(queued, completion_request([9419], 1))
                self.check_choice(response, self.expected[1:2], None)
                readable, _, _ = select.select([queued.sock], [], [], 0.25)
                self.assertFalse(readable, "Queued request responded during active generation")
                active_socket.setsockopt(
                    socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0)
                )
            with queued.getresponse() as response:
                self.check_stream(response, [9419], [11])
        self.check_completion(self.prompt, self.expected[:2])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be in [1, 65535]")
    NativeStreamingTest.port = args.port
    unittest.main(argv=[__file__], verbosity=2)


if __name__ == "__main__":
    main()
