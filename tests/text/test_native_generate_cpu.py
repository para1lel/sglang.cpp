"""Exercise the native generate route through the host-only server."""

import argparse
from contextlib import closing
import http.client
import json
from pathlib import Path
import socket
import struct
import time
import unittest
from unittest.mock import patch

from test_native_chat import tokenizer_pieces
from test_native_text import EOS, HEADERS, MODEL, event
from test_sampling_routes_cpu import CHAT, MODELS, TEXT, body, receipts, send, server


GENERATE = "/generate"


def get(port, path):
    with closing(http.client.HTTPConnection("127.0.0.1", port, timeout=5)) as connection:
        connection.request("GET", path)
        with connection.getresponse() as response:
            return response.status, response.read()


def queue_client(port, budget):
    wire = json.dumps({
        "input_ids": [32], "stream": True, "sampling_params": {"max_new_tokens": budget},
    }).encode()
    client = socket.create_connection(("127.0.0.1", port), timeout=5)
    try:
        client.sendall(
            b"POST /generate HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n"
            b"Content-Type: application/json\r\nContent-Length: "
            + str(len(wire)).encode() + b"\r\n\r\n" + wire
        )
        return client
    except OSError:
        client.close()
        raise


def abort(client):
    client.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
    client.close()


class NativeGenerate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pieces = tokenizer_pieces(cls.checkpoint)
        cls.bytes = {piece[0]: token for token, piece in pieces.items() if len(piece) == 1}
        cls.eos = min(EOS)
        cls.script = [32, *(cls.bytes[byte] for byte in "⛄".encode()), cls.eos, 33, 34]
        scripted = cls.output / "token-ids.txt"
        tokens = " ".join(map(str, cls.script))
        scripted.write_text(f"-7 {tokens}\n11 32 {cls.eos} 33\n0 32\n")
        cls.enterClassContext(patch.dict("os.environ", SGLANG_TEST_CHAT_OUTPUT_IDS=str(scripted)))
        cls.log = cls.output / "native.log"
        cls.port = cls.enterClassContext(server(cls.binary, cls.checkpoint, cls.log,
                                               "--max-active", "1"))

    def check_packet(self, packet, text, ids, reason, *, prompt_tokens=1):
        self.assertEqual(set(packet), {"text", "output_ids", "meta_info"})
        self.assertEqual(packet["text"], text)
        self.assertEqual(packet["output_ids"], ids)
        meta = packet["meta_info"]
        self.assertIsInstance(meta["id"], str)
        self.assertTrue(meta["id"])
        expected = {"id": meta["id"], "prompt_tokens": prompt_tokens,
                    "completion_tokens": len(ids), "finish_reason": reason}
        if reason is not None:
            expected["cached_tokens"] = 0
        self.assertEqual(meta, expected)
        return meta["id"]

    def generate(self, payload, *, port=None):
        status, result = send(self.port if port is None else port, GENERATE, payload)
        self.assertEqual(status, 200, result)
        return result

    def test_wire_text_ids_and_cumulative_sse(self):
        options = {"max_new_tokens": 6, "temperature": 0, "ignore_eos": True,
                   "sampling_seed": -7, "stop_token_ids": [33]}
        common = {"sampling_params": options, "lora_path": None, "return_logprob": False,
                  "return_routed_experts": False, "logprob_start_len": -1}
        final_ids = self.script[:6]
        finish = {"type": "length", "length": 6}
        plain = self.generate(common | {"text": "A", "input_ids": None})
        token_result = self.generate(common | {"input_ids": [32], "text": None})
        packets = self.generate(common | {"input_ids": [32], "stream": True})
        different = {self.check_packet(packet, "A⛄B", final_ids, finish)
                     for packet in (plain, token_result, packets[-1])}
        self.assertEqual(len(different), 3)
        self.assertEqual(len(packets), 6)
        stream_id = packets[-1]["meta_info"]["id"]
        for index, text in enumerate(("A", "A", "A", "A⛄", "A⛄"), start=1):
            self.assertEqual(self.check_packet(packets[index - 1], text, final_ids[:index], None),
                             stream_id)

    def test_stop_and_zero_tokens(self):
        for stop, ids, text in ((None, [32, self.eos], "A"), (32, [32], "")):
            params = {"sampling_seed": 11, "max_new_tokens": 4}
            if stop is not None:
                params["stop_token_ids"] = [stop]
            for streaming in (False, True):
                with self.subTest(stop=stop, streaming=streaming):
                    result = self.generate({"text": "A", "stream": streaming,
                                            "sampling_params": params})
                    if streaming:
                        self.assertEqual(len(result), len(ids))
                        result = result[-1]
                    self.check_packet(result, text, ids, {"type": "stop", "matched": ids[-1]})
        for streaming in (False, True):
            result = self.generate({"input_ids": [0, 248319], "stream": streaming,
                                    "sampling_params": {"max_new_tokens": 0}})
            if streaming:
                result, = result
            self.check_packet(result, "", [], {"type": "length", "length": 0}, prompt_tokens=2)

    def test_sampling_defaults_and_forwarding(self):
        first = len(receipts(self.log))
        options = {"max_new_tokens": 2, "temperature": 0.625, "top_k": 7, "top_p": 0.75,
                   "min_p": 0.125, "sampling_seed": -(2**63), "stop_token_ids": [42], "n": 1}
        defaults = self.generate({"text": "A", "sampling_params": None})
        self.check_packet(defaults, "A" * 128, [32] * 128, {"type": "length", "length": 128})
        self.generate({"text": "A", "sampling_params": options, "cache_salt": "tenant"})
        self.generate({"text": "A", "sampling_params": {
            "max_new_tokens": 1, "temperature": None, "top_k": None, "top_p": None,
            "min_p": None, "sampling_seed": None, "ignore_eos": None, "stop_token_ids": None,
        }})
        implicit, explicit, nullable = receipts(self.log)[first:]
        expected = {"temperature": 1, "top_k": -1, "top_p": 1, "min_p": 0,
                    "seed": None, "budget": 128, "sampled": True, "prompt_tokens": 1,
                    "stops": sorted(EOS)}
        self.assertEqual(implicit, expected)
        self.assertEqual(nullable, expected | {"budget": 1})
        self.assertEqual(explicit, expected | {
            "temperature": 0.625, "top_k": 7, "top_p": 0.75, "min_p": 0.125,
            "seed": -(2**63), "budget": 2, "stops": sorted(EOS | {42}),
        })

    def test_strict_requests_and_framing(self):
        invalid = [
            ([], "JSON object"), ({}, "exactly one"), ({"text": "", "input_ids": None}, "text"),
            ({"text": ["A"]}, "text"), ({"text": "A", "input_ids": [32]}, "exactly one"),
            ({"input_ids": []}, "input_ids"), ({"input_ids": [[32]]}, "input_ids"),
            *[({"input_ids": [token]}, "input_ids") for token in (-1, 248320, True, 1.5)],
            *[({"text": "A", key: value}, key) for key, value in (
                ("model", MODEL), ("stream", 1), ("lora_path", "adapter"), ("return_logprob", True),
                ("return_routed_experts", True), ("logprob_start_len", 0), ("rid", "x"),
                ("image_data", []), ("sampling_params", [{}]), ("cache_salt", "x" * 257),
            )],
            *[({"text": "A", "sampling_params": {key: value}}, key) for key, value in (
                ("max_new_tokens", None), ("max_new_tokens", -1), ("max_new_tokens", 1.5),
                ("n", 2), ("top_p", 0), ("temperature", -1), ("min_p", 2), ("top_k", 0),
                ("sampling_seed", 2**63), ("ignore_eos", "yes"),
                ("stop", "X"), ("stop_regex", "X"), ("regex", "X"), ("frequency_penalty", 0),
            )],
            ({"text": "A", "sampling_params": {"stop_token_ids": [-1]}}, "vocabulary"),
            ({"input_ids": [0, 1], "sampling_params": {"max_new_tokens": 262143}}, "context"),
        ]
        first = len(receipts(self.log))
        for payload, field in invalid:
            with self.subTest(payload=str(payload)[:300]):
                status, result = send(self.port, GENERATE, payload)
                self.assertEqual(status, 400, result)
                self.assertIn(field, result["error"]["message"])
        for method, wire, headers, expected in (
            ("POST", None, HEADERS, 411),
            ("POST", b'{"text":"A",}', HEADERS, 400),
            ("POST", b'{"text":"A"}', {"Content-Type": "text/plain"}, 415),
            ("PUT", b'{"text":"A"}', HEADERS, 404),
        ):
            with self.subTest(method=method, headers=headers, wire=wire):
                with closing(http.client.HTTPConnection("127.0.0.1", self.port,
                                                        timeout=5)) as connection:
                    connection.putrequest(method, GENERATE)
                    for key, value in headers.items():
                        connection.putheader(key, value)
                    if wire is not None:
                        connection.putheader("Content-Length", str(len(wire)))
                    connection.endheaders(wire)
                    with connection.getresponse() as response:
                        status = response.status
                        content_type = response.headers.get_content_type()
                        data = response.read()
                self.assertEqual(status, expected, data)
                self.assertEqual(content_type, "application/json")
                self.assertIn("message", json.loads(data)["error"])
        self.assertEqual(len(receipts(self.log)), first)

    def test_models_and_existing_routes(self):
        status, data = get(self.port, MODELS)
        self.assertEqual(status, 200)
        model, = json.loads(data)["data"]
        self.assertEqual(model["id"], MODEL)
        for path in (TEXT, CHAT):
            status, packet = send(self.port, path, body(path, max_tokens=1, model=model["id"]))
            self.assertEqual(status, 200, packet)
            if path == CHAT:
                self.assertEqual(packet["model"], model["id"])
            self.assertEqual(packet["usage"]["completion_tokens"], 1)

    def wait_queue(self, port, running, waiting):
        deadline = time.monotonic() + 3
        expected = {f'vllm:num_requests_running{{engine="0"}} {running}',
                    f'vllm:num_requests_waiting{{engine="0"}} {waiting}'}
        while time.monotonic() < deadline:
            status, data = get(port, "/metrics")
            self.assertEqual(status, 200, data)
            if expected <= set(data.decode().splitlines()):
                return
            time.sleep(0.01)
        self.fail(f"Queue did not reach {running} running and {waiting} waiting: {data!r}")

    def test_queue_overflow_disconnect_and_reuse(self):
        log = self.output / "queue.log"
        with server(self.binary, self.checkpoint, log, "--max-active", "1",
                    "--prefix-cache-entries", "1") as port:
            with queue_client(port, 2048) as active, http.client.HTTPResponse(active) as reply:
                reply.begin()
                self.assertEqual(reply.status, 200)
                self.assertEqual(event(reply)["output_ids"], [32])
                with queue_client(port, 77) as dropped:
                    self.wait_queue(port, 1, 1)
                    status, error = send(port, GENERATE, {"text": "A"})
                    self.assertEqual(status, 503, error)
                    status, error = send(port, TEXT, body(TEXT, max_tokens=1))
                    self.assertEqual(status, 503, error)
                    abort(dropped)
                    self.wait_queue(port, 1, 0)
                with queue_client(port, 2) as queued:
                    self.wait_queue(port, 1, 1)
                    reply.close()
                    abort(active)
                    with http.client.HTTPResponse(queued) as resumed:
                        resumed.begin()
                        self.assertEqual(resumed.status, 200)
                        self.check_packet(event(resumed), "A", [32], None)
                        self.check_packet(event(resumed), "AA", [32, 32],
                                          {"type": "length", "length": 2})
                        self.assertEqual(event(resumed), "[DONE]")
                        self.assertEqual(resumed.read(), b"")
            self.wait_queue(port, 0, 0)
            self.check_packet(self.generate({"text": "A", "sampling_params": {"max_new_tokens": 1}},
                                            port=port),
                              "A", [32], {"type": "length", "length": 1})
        self.assertEqual([receipt["budget"] for receipt in receipts(log)], [2048, 2, 1])
        self.assertIn("cpu cancel slot=0 prompt=1 budget=2048", log.read_text())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    NativeGenerate.binary = args.binary.resolve()
    NativeGenerate.checkpoint = args.checkpoint.resolve()
    NativeGenerate.output = args.output.resolve()
    NativeGenerate.output.mkdir(parents=True, exist_ok=True)
    unittest.main(argv=[__file__], verbosity=2)
