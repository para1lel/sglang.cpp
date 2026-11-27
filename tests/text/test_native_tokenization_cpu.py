"""Exercise native tokenization HTTP routes with the Qwen checkpoint and CPU server."""

from contextlib import closing
from email.parser import BytesHeaderParser
import http.client
import json
from pathlib import Path
import socket
from tempfile import TemporaryDirectory
import time
import unittest
from unittest.mock import patch

from test_native_generate_cpu import abort, generate_client, get
from test_sampling_routes_cpu import control_response, run_cpu_suite, server


TOKENIZE = ("/tokenize", "/v1/tokenize")
DETOKENIZE = ("/detokenize", "/v1/detokenize")
HEADERS = {"Content-Type": "application/json"}
CONTEXT = 262144
SPECIAL_TOKENS = [248045, 87, 248046]
UNICODE_TOKENS = [109266, 10838, 234, 235, 50203, 3825, 188]


def packet(response):
    assert response.headers.get_content_type() == "application/json", response.headers
    return response.status, json.load(response)


def post(port, path, body, headers=HEADERS):
    if not isinstance(body, bytes):
        body = json.dumps(body).encode()
    with control_response(port, "POST", path, body, headers) as response:
        return packet(response)


class NativeTokenization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.log = cls.output / "tokenization.log"
        cls.port = cls.enterClassContext(server(cls.binary, cls.checkpoint, cls.log))

    def check_error(self, result, status, category, message=None):
        actual_status, payload = result
        self.assertEqual(actual_status, status, payload)
        self.assertIsInstance(payload.get("message"), str)
        self.assertTrue(payload["message"])
        self.assertEqual(payload, {"object": "error", "type": category, "code": status,
                                   "param": None, "message": message or payload["message"]})

    def test_tokenize_shapes_and_checkpoint_ids(self):
        special = "<|im_start|>x<|im_end|>"
        unicode = "你好 🌍 café e\u0301\0"
        cases = (
            ({"prompt": special}, SPECIAL_TOKENS, 3),
            ({"prompt": [unicode, "", special], "add_special_tokens": False},
             [UNICODE_TOKENS, [], SPECIAL_TOKENS], [7, 0, 3]),
            ({"prompt": []}, [], []),
        )
        for route in TOKENIZE:
            for request, ids, count in cases:
                with self.subTest(route=route, request=request):
                    self.assertEqual(post(self.port, route, request), (
                        200, {"tokens": ids, "count": count, "max_model_len": CONTEXT}
                    ))
        self.assertNotIn("cpu admit", self.log.read_text())

    def test_detokenize_shapes_and_special_tokens(self):
        cases = (
            ({"tokens": SPECIAL_TOKENS}, "x"),
            ({"tokens": [UNICODE_TOKENS, [], [162], SPECIAL_TOKENS]},
             ["你好 🌍 café é\0", "", "\ufffd", "x"]),
            ({"tokens": SPECIAL_TOKENS, "skip_special_tokens": "off"},
             "<|im_start|>x<|im_end|>"),
            ({"tokens": []}, ""),
        )
        for route in DETOKENIZE:
            for request, text in cases:
                with self.subTest(route=route, request=request):
                    self.assertEqual(post(self.port, route, request), (200, {"text": text}))
        self.assertNotIn("cpu admit", self.log.read_text())

    def test_route_errors_and_recovery(self):
        for route in TOKENIZE + DETOKENIZE:
            valid = {"prompt": ""} if route in TOKENIZE else {"tokens": []}
            invalid = {"prompt": ["x", 1]} if route in TOKENIZE else {"tokens": [1, []]}
            for body, headers in ((b"{", HEADERS), (b"[]", HEADERS), (invalid, HEADERS),
                                  (valid, {"Content-Type": "text/plain"})):
                with self.subTest(route=route, body=body, headers=headers):
                    self.check_error(post(self.port, route, body, headers), 400, "Bad Request")
            self.assertEqual(post(self.port, route, valid, {
                "Content-Type": "Application/JSON; charset=utf-8"
            })[0], 200)
        for route in DETOKENIZE:
            self.check_error(post(self.port, route, {"tokens": [[], [248320]]}),
                             400, "BadRequestError",
                             "Invalid input: token id 248320 is out of vocabulary; "
                             "valid range is [0, 248320).")
        self.check_error(post(self.port, TOKENIZE[0], {"prompt": "x", "stream": True}),
                         501, "NotImplementedError")

    def test_score_methods_on_wire(self):
        body = b'{"detail":"Method Not Allowed"}'
        for method in ("GET", "PUT", "OPTIONS", "HEAD"):
            with self.subTest(method=method):
                request = (f"{method} /v1/score HTTP/1.1\r\n"
                           f"Host: 127.0.0.1:{self.port}\r\nConnection: close\r\n\r\n")
                with socket.create_connection(("127.0.0.1", self.port), timeout=5) as client:
                    client.sendall(request.encode("ascii"))
                    with client.makefile("rb") as stream:
                        head, boundary, data = stream.read().partition(b"\r\n\r\n")
                self.assertTrue(boundary)
                status_line, fields = head.split(b"\r\n", 1)
                headers = BytesHeaderParser().parsebytes(fields)
                self.assertEqual(int(status_line.split()[1]), 405)
                self.assertEqual(headers.get_all("Allow"), ["POST"])
                self.assertEqual(headers.get_content_type(), "application/json")
                self.assertEqual(headers["Content-Length"], str(len(body)))
                self.assertEqual(data, b"" if method == "HEAD" else body)
        status, scores = post(self.port, "/v1/score", {"items": [], "label_token_ids": [32]})
        self.assertEqual((status, scores["scores"]), (200, []))

    def test_missing_tokenizer_errors_and_retry(self):
        with TemporaryDirectory(dir=self.output) as directory:
            checkpoint = Path(directory)
            log = self.output / "missing-tokenizer.log"
            with server(self.binary, checkpoint, log, "--prefix-cache-entries", "0") as port:
                self.check_error(post(port, TOKENIZE[0], {"prompt": "x"}),
                                 500, "InternalServerError")
                self.check_error(post(port, DETOKENIZE[0], {"tokens": [87]}),
                                 500, "InternalServerError")
                self.assertEqual(post(port, DETOKENIZE[1], {"tokens": []}), (200, {"text": ""}))
                (checkpoint / "tokenizer.json").symlink_to(self.checkpoint / "tokenizer.json")
                self.assertEqual(post(port, DETOKENIZE[0], {"tokens": [87]}), (200, {"text": "x"}))
            self.assertNotIn("cpu admit", log.read_text())

    def test_shared_queue_overflow_and_reuse(self):
        cases = ((TOKENIZE[1], {"prompt": "x"},
                  {"tokens": [87], "count": 1, "max_model_len": CONTEXT}, DETOKENIZE[0],
                  {"tokens": []}),
                 (DETOKENIZE[1], {"tokens": [87]}, {"text": "x"}, TOKENIZE[0], {"prompt": ""}))
        for index, (queued_route, body, expected, overflow, overflow_body) in enumerate(cases):
            with self.subTest(route=queued_route), TemporaryDirectory(dir=self.output) as directory:
                gate = Path(directory) / "step"
                log = self.output / f"queue-{index}.log"
                with patch.dict("os.environ", SGLANG_TEST_BATCH_STEP_GATE=str(gate)), \
                     server(self.binary, self.checkpoint, log, "--max-active", "1") as port:
                    self.check_queue(port, log, queued_route, body, expected,
                                     overflow, overflow_body)
                self.assertEqual(log.read_text().count("cpu admit"), 1)

    def check_queue(self, port, log, route, body, expected, overflow, overflow_body):
        with generate_client(port, {
            "input_ids": [32], "sampling_params": {"sampling_seed": -41, "max_new_tokens": 2},
        }) as running, closing(http.client.HTTPConnection("127.0.0.1", port, timeout=5)) as queued:
            deadline = time.monotonic() + 5
            while "cpu admit slot=0" not in log.read_text():
                self.assertLess(time.monotonic(), deadline, log.read_text())
                time.sleep(0.01)
            queued.request("POST", route, json.dumps(body).encode(), HEADERS)
            deadline = time.monotonic() + 5
            while b'vllm:num_requests_waiting{engine="0"} 1' not in get(port, "/metrics")[1]:
                self.assertLess(time.monotonic(), deadline, log.read_text())
                time.sleep(0.01)
            self.check_error(post(port, overflow, overflow_body), 503, "ServiceUnavailableError")
            self.assertEqual(get(port, "/health")[0], 200)
            abort(running)
            with queued.getresponse() as response:
                self.assertEqual(packet(response), (200, expected))
            self.assertEqual(post(port, overflow, overflow_body)[0], 200)


if __name__ == "__main__":
    run_cpu_suite(NativeTokenization, __file__, __doc__)
