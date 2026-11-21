"""Exercise the native generate route through the host-only server."""

import argparse
from contextlib import closing
import http.client
import json
from pathlib import Path
import select
import socket
import struct
from tempfile import TemporaryDirectory
import time
import unittest
from unittest.mock import patch

from test_native_chat import tokenizer_pieces
from test_native_text import EOS, HEADERS, MODEL, event
from test_sampling_routes_cpu import CHAT, MODELS, TEXT, body, receipts, send, server


GENERATE = "/generate"
ABORT_REQUEST = "/abort_request"
ABORTED = {"type": "abort", "message": "Aborted", "status_code": None, "err_type": None}
ABORTED_WAITING = {"type": "abort", "message": "Abort in waiting queue"}


def get(port, path):
    with closing(http.client.HTTPConnection("127.0.0.1", port, timeout=5)) as connection:
        connection.request("GET", path)
        with connection.getresponse() as response:
            return response.status, response.read()


def generate_client(port, payload):
    wire = json.dumps(payload).encode()
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


def queue_client(port, budget):
    return generate_client(port, {
        "input_ids": [32], "stream": True, "sampling_params": {"max_new_tokens": budget},
    })


def named(rid, *, budget=2, stream=True, seed=-41):
    return {
        "input_ids": [32], "rid": rid, "stream": stream,
        "sampling_params": {"max_new_tokens": budget, "sampling_seed": seed},
    }


def abort(client):
    client.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
    client.close()


def batch_frame(reply):
    line = reply.readline()
    if not line:
        return None
    if line == b":\n":
        assert reply.readline() == b"\n"
        return "keepalive"
    assert line.startswith(b"data: ") and line.endswith(b"\n"), line
    assert reply.readline() == b"\n"
    return "[DONE]" if line == b"data: [DONE]\n" else json.loads(line[6:])


class NativeGenerate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pieces = tokenizer_pieces(cls.checkpoint)
        cls.bytes = {piece[0]: token for token, piece in pieces.items() if len(piece) == 1}
        cls.eos = min(EOS)
        cls.script = [32, *(cls.bytes[byte] for byte in "⛄".encode()), cls.eos, 33, 34]
        scripted = cls.output / "token-ids.txt"
        tokens = " ".join(map(str, cls.script))
        scripted.write_text(f"-7 {tokens}\n11 32 {cls.eos} 33\n-41 32 33\n0 32\n")
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
            "repetition_penalty": None,
        }})
        implicit, explicit, nullable = receipts(self.log)[first:]
        expected = {"temperature": 1, "top_k": -1, "top_p": 1, "min_p": 0,
                    "frequency_penalty": 0, "presence_penalty": 0, "repetition_penalty": 1,
                    "seed": None, "budget": 128, "minimum": 0, "sampled": True,
                    "prompt_tokens": 1,
                    "stops": sorted(EOS), "logit_bias": []}
        self.assertEqual(implicit, expected)
        self.assertEqual(nullable, expected | {"budget": 1})
        self.assertEqual(explicit, expected | {
            "temperature": 0.625, "top_k": 7, "top_p": 0.75, "min_p": 0.125,
            "seed": -(2**63), "budget": 2, "stops": sorted(EOS | {42}),
        })

    def test_strict_requests_and_framing(self):
        invalid = [
            ([], "JSON object"), ({}, "exactly one"), ({"text": "", "input_ids": None}, "text"),
            ({"text": "A", "input_ids": [32]}, "exactly one"), ({"input_ids": []}, "input_ids"),
            *[({"input_ids": [token]}, "input_ids") for token in (-1, 248320, True, 1.5)],
            *[({"text": "A", key: value}, key) for key, value in (
                ("model", MODEL), ("stream", 1), ("lora_path", "adapter"), ("return_logprob", True),
                ("return_routed_experts", True), ("logprob_start_len", 0),
                ("image_data", []), ("sampling_params", [{}]), ("cache_salt", "x" * 257),
            )],
            *[({"text": "A", "sampling_params": {key: value}}, key) for key, value in (
                ("max_new_tokens", None), ("max_new_tokens", -1), ("max_new_tokens", 1.5),
                ("n", 0), ("top_p", 0), ("temperature", -1), ("min_p", 2), ("top_k", 0),
                ("sampling_seed", 2**63), ("ignore_eos", "yes"),
                ("stop", ["X", 1]), ("stop_regex", "X"), ("regex", 1),
                ("frequency_penalty", 2.01), ("repetition_penalty", 0),
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

    def test_regex_request_validation(self):
        invalid = (
            ({"regex": "["}, {}, "regex"),
            ({"regex": "A", "max_new_tokens": 0}, {}, "max_new_tokens"),
            ({"regex": "A", "ignore_eos": True}, {}, "model EOS"),
            ({"regex": "A", "stop_token_ids": [32]}, {}, "model EOS"),
            ({"regex": "A", "temperature": 0}, {"return_logprob": True}, "logprobs"),
        )
        first = len(receipts(self.log))
        for params, fields, message in invalid:
            with self.subTest(params=params, fields=fields):
                status, result = send(self.port, GENERATE, {
                    "text": "A", "stream": True, "sampling_params": params, **fields,
                })
                self.assertEqual(status, 400, result)
                self.assertIn(message, result["error"]["message"])
        status, result = send(self.port, GENERATE, {
            "input_ids": [[32], [33]], "sampling_params": [{"regex": "A"}, {"regex": False}],
        })
        self.assertEqual(status, 400, result)
        self.assertIn("regex must be a string", result["error"]["message"])
        self.assertEqual(len(receipts(self.log)), first)
        items = self.generate({
            "text": ["A", "B"], "sampling_params": [
                {"regex": None, "max_new_tokens": 1}, {"regex": "", "max_new_tokens": 1},
            ],
        })
        for item in items:
            self.check_packet(item, "A", [32], {"type": "length", "length": 1})

    def test_batch_shape_and_input_order(self):
        log = self.output / "batch-order.log"
        with server(self.binary, self.checkpoint, log, "--max-active", "2") as port:
            result = self.generate({
                "input_ids": [[32], [33, 33], [34, 34, 34]],
                "sampling_params": [{"max_new_tokens": 8, "sampling_seed": 0},
                                    {"max_new_tokens": 1, "sampling_seed": -7},
                                    {"max_new_tokens": 4, "sampling_seed": 11}],
                "cache_salt": ["first", "second", "third"],
            }, port=port)
            self.assertIsInstance(result, list)
            self.assertEqual(len(result), 3)
            ids = {
                self.check_packet(result[0], "AAAAAAAA", [32] * 8,
                                  {"type": "length", "length": 8}),
                self.check_packet(result[1], "A", [32], {"type": "length", "length": 1},
                                  prompt_tokens=2),
                self.check_packet(result[2], "A", [32, self.eos],
                                  {"type": "stop", "matched": self.eos}, prompt_tokens=3),
            }
            self.assertEqual(len(ids), 3)
            self.assertEqual([line for line in log.read_text().splitlines()
                              if line.startswith("cpu finish budget=")],
                             ["cpu finish budget=1", "cpu finish budget=4", "cpu finish budget=8"])
            options = {"max_new_tokens": 1, "sampling_seed": 0}
            result = self.generate({"text": ["A", "A B"], "sampling_params": options},
                                   port=port)
            self.assertEqual(len(result), 2)
            for packet, prompt_tokens in zip(result, (1, 2)):
                self.check_packet(packet, "A", [32], {"type": "length", "length": 1},
                                  prompt_tokens=prompt_tokens)
            for field, scalar in (("text", "A"), ("input_ids", [32])):
                with self.subTest(field=field):
                    common = {"sampling_params": options}
                    item = self.generate(common | {field: scalar}, port=port)
                    batch = self.generate(common | {field: [scalar]}, port=port)
                    self.assertIsInstance(batch, list)
                    singleton, = batch
                    self.assertNotEqual(singleton["meta_info"].pop("id"),
                                        item["meta_info"].pop("id"))
                    self.assertEqual(singleton, item)

    def test_invalid_batch_never_admits_a_lane(self):
        invalid = [
            ({"text": ["A", ""]}, "text"),
            ({"input_ids": [[32], [248320]]}, "input_ids"),
            ({"input_ids": [[32], 33]}, "input_ids"),
            ({"text": ["A", "B"], "sampling_params": [{"max_new_tokens": 1}, {"top_p": 0}]},
             "top_p"),
            ({"text": ["A", "B"], "sampling_params": [{}, {"repetition_penalty": 2.01}]},
             "repetition_penalty"),
            ({"text": ["A", "B"], "sampling_params": [{}]}, "sampling_params"),
            ({"text": ["A", "B"], "cache_salt": ["a", "x" * 257]}, "cache_salt"),
            ({"text": ["A", "B"], "cache_salt": ["a"]}, "cache_salt"),
            ({"input_ids": [[32], [33, 34]],
              "sampling_params": [{"max_new_tokens": 1}, {"max_new_tokens": 262143}]}, "context"),
            ({"input_ids": [[32], [33]], "sampling_params": {"max_new_tokens": 131073}},
             "total"),
            ({"text": ["A"] * 33}, "batch items"),
            ({"text": ["A", ""], "stream": True}, "text"),
        ]
        log = self.output / "batch-invalid.log"
        with server(self.binary, self.checkpoint, log, "--max-active", "2") as port:
            for payload, field in invalid:
                with self.subTest(field=field, payload=str(payload)[:240]):
                    status, result = send(port, GENERATE, payload)
                    self.assertEqual(status, 400, result)
                    self.assertIn(field, result["error"]["message"])
                self.assertNotIn("cpu admit", log.read_text())

    def test_batch_repetition_penalty_forwarding(self):
        first = len(receipts(self.log))
        options = {"max_new_tokens": 1, "temperature": 0}
        result = self.generate({"input_ids": [[32], [32, 33], [32, 33, 34]],
                                "sampling_params": [
                                    options | {"repetition_penalty": 2},
                                    options | {"repetition_penalty": 0.5},
                                    options | {"repetition_penalty": None},
                                ]})
        self.assertEqual([packet["output_ids"] for packet in result], [[32]] * 3)
        by_length = {seen["prompt_tokens"]: seen["repetition_penalty"]
                     for seen in receipts(self.log)[first:]}
        self.assertEqual(by_length, {1: 2, 2: 0.5, 3: 1})

    def check_batch_headers(self, reply):
        reply.begin()
        self.assertEqual(reply.status, 200)
        self.assertEqual(reply.headers.get_content_type(), "text/event-stream")
        self.assertTrue(reply.chunked)

    def check_batch_frame(self, packet, index, text, ids, finish, *, prompt_tokens):
        self.assertEqual(packet.pop("index"), index)
        cached = packet["meta_info"]["cached_tokens"]
        self.assertEqual(cached, 0)
        if finish is None:
            del packet["meta_info"]["cached_tokens"]
        return self.check_packet(packet, text, ids, finish, prompt_tokens=prompt_tokens)

    def test_batch_stream_cumulative_utf8_and_finish(self):
        log = self.output / "batch-stream.log"
        payload = {
            "input_ids": [[32], [33, 33], [34, 34, 34]],
            "sampling_params": [{"max_new_tokens": 8, "sampling_seed": -7},
                                {"max_new_tokens": 4, "sampling_seed": 11},
                                {"max_new_tokens": 0}],
            "cache_salt": ["first", "second", "third"],
        }
        with server(self.binary, self.checkpoint, log, "--max-active", "2") as port:
            plain = self.generate(payload, port=port)
            with generate_client(port, payload | {"stream": True}) as client, \
                 http.client.HTTPResponse(client) as reply:
                self.check_batch_headers(reply)
                frames = list(iter(lambda: batch_frame(reply), None))

        self.assertEqual(frames.pop(), "[DONE]")
        self.assertTrue(all(isinstance(frame, dict) for frame in frames), frames)
        finishes = [frame["index"] for frame in frames if frame["meta_info"]["finish_reason"]]
        self.assertCountEqual(finishes, (0, 1, 2))
        self.assertEqual(finishes[0], 1)
        expected = [
            (self.script[:5], ("A", "A", "A", "A⛄", "A⛄"),
             {"type": "stop", "matched": self.eos}),
            ([32, self.eos], ("A", "A"), {"type": "stop", "matched": self.eos}),
            ([], ("",), {"type": "length", "length": 0}),
        ]
        ids = set()
        for index, (tokens, texts, finish) in enumerate(expected):
            items = [frame for frame in frames if frame.get("index") == index]
            self.assertEqual(len(items), len(texts))
            lane_ids = set()
            for count, (item, text) in enumerate(zip(items, texts), 1):
                terminal = finish if count == len(texts) else None
                lane_ids.add(self.check_batch_frame(item, index, text, tokens[:count], terminal,
                                                    prompt_tokens=index + 1))
            self.assertEqual(len(lane_ids), 1)
            ids.update(lane_ids)
            self.assertEqual(items[-1], plain[index] | {
                "meta_info": plain[index]["meta_info"] | {"id": lane_ids.pop()},
            })
        self.assertEqual(len(ids), 3)

    def test_batch_stream_heartbeat_follows_wire_writes(self):
        log = self.output / "batch-heartbeat.log"
        controls = Path(self.enterContext(TemporaryDirectory(dir=self.output)))
        gate = controls / "step"
        clock = controls / "seconds"

        def tick(seconds):
            pending = clock.with_suffix(".next")
            pending.write_text(str(seconds))
            pending.replace(clock)

        tick(0)
        payload = {"text": ["A", "B"], "stream": True,
                   "sampling_params": [{"max_new_tokens": 2, "sampling_seed": -41},
                                       {"max_new_tokens": 1}]}
        with patch.dict("os.environ", SGLANG_TEST_BATCH_STEP_GATE=str(gate),
                        SGLANG_TEST_STREAM_CLOCK_FILE=str(clock)), \
             server(self.binary, self.checkpoint, log, "--max-active", "2") as port, \
             generate_client(port, payload) as client, http.client.HTTPResponse(client) as reply:
            self.check_batch_headers(reply)
            fast = batch_frame(reply)
            self.check_batch_frame(fast, 1, "A", [32], {"type": "length", "length": 1},
                                   prompt_tokens=1)
            self.assertFalse(select.select([client], [], [], 0.1)[0])
            tick(15)
            self.assertEqual(batch_frame(reply), "keepalive")
            self.assertFalse(select.select([client], [], [], 0.1)[0])
            tick(24)
            gate.with_suffix(".0").touch()
            first = self.check_batch_frame(batch_frame(reply), 0, "A", [32], None,
                                           prompt_tokens=1)
            self.assertFalse(select.select([client], [], [], 0.1)[0])
            tick(38)
            self.assertFalse(select.select([client], [], [], 0.1)[0])
            tick(39)
            self.assertEqual(batch_frame(reply), "keepalive")
            gate.with_suffix(".1").touch()
            last = self.check_batch_frame(batch_frame(reply), 0, "AB", [32, 33],
                                          {"type": "length", "length": 2}, prompt_tokens=1)
            self.assertEqual(first, last)
            self.assertEqual(batch_frame(reply), "[DONE]")
            self.assertIsNone(batch_frame(reply))

    def test_batch_stream_admission_failure_before_and_after_headers(self):
        log = self.output / "batch-stream-error.log"
        with patch.dict("os.environ", SGLANG_TEST_BATCH_ADMISSION_FAILURE="-900"), \
             server(self.binary, self.checkpoint, log, "--max-active", "2",
                    "--prefix-cache-entries", "1") as port:
            failing = {"max_new_tokens": 7, "sampling_seed": -900}
            early = {"text": ["A"], "stream": True, "sampling_params": failing}
            status, error = send(port, GENERATE, early)
            self.assertEqual(status, 503)
            self.assertEqual(error, {"error": {"message": "Injected batch admission failure"}})
            self.assertEqual(receipts(log), [])
            late = {"input_ids": [[32], [33, 33], [34], [35]], "stream": True,
                    "sampling_params": [{"max_new_tokens": 2048}, {"max_new_tokens": 1},
                                        failing, {"max_new_tokens": 9}]}
            with generate_client(port, late) as client, http.client.HTTPResponse(client) as reply:
                self.check_batch_headers(reply)
                frames = list(iter(lambda: batch_frame(reply), None))
            self.assertEqual(frames.pop(), {"error": {"message": "Native decoder failed",
                                                      "type": "server_error", "code": 500,
                                                      "retryable": False}})
            done, = [frame for frame in frames if frame["index"] == 1]
            self.check_batch_frame(done, 1, "A", [32], {"type": "length", "length": 1},
                                   prompt_tokens=2)
            self.assertEqual(sorted(item["budget"] for item in receipts(log)), [1, 2048])
            admitted, = [line for line in log.read_text().splitlines()
                         if line.startswith("cpu admit ") and line.endswith(" budget=2048")]
            self.assertIn(admitted.replace("cpu admit ", "cpu cancel ", 1), log.read_text())
            self.check_batch_reuse(port)

    def test_batch_stream_disconnect_after_partial_output(self):
        log = self.output / "batch-stream-disconnect.log"
        payload = {"input_ids": [[32], [33, 33], [34]], "stream": True,
                   "sampling_params": {"max_new_tokens": 2048}}
        with server(self.binary, self.checkpoint, log, "--max-active", "2",
                    "--prefix-cache-entries", "1") as port, \
             generate_client(port, payload) as client:
            with http.client.HTTPResponse(client) as reply:
                self.check_batch_headers(reply)
                first = batch_frame(reply)
                self.assertEqual(first["output_ids"], [32])
                self.assertIsNone(first["meta_info"]["finish_reason"])
            with queue_client(port, 1) as scalar:
                self.wait_queue(port, 2, 0)
                abort(client)
                with http.client.HTTPResponse(scalar) as reply:
                    self.check_batch_headers(reply)
                    self.check_packet(batch_frame(reply), "A", [32],
                                      {"type": "length", "length": 1})
                    self.assertEqual(batch_frame(reply), "[DONE]")
            self.wait_queue(port, 0, 0)
            lines = log.read_text().splitlines()
            admitted = {line for line in lines
                        if line.startswith("cpu admit ") and line.endswith(" budget=2048")}
            cancelled = {line for line in lines if line.startswith("cpu cancel ")}
            self.assertEqual(cancelled, {line.replace("cpu admit ", "cpu cancel ", 1)
                                         for line in admitted})
            seen = receipts(log)
            self.assertEqual([item["budget"] for item in seen], [2048, 2048, 1])
            self.assertCountEqual([item["prompt_tokens"] for item in seen[:2]], [1, 2])

    def check_batch_reuse(self, port):
        self.wait_queue(port, 0, 0)
        result = self.generate({"text": ["A", "B"], "sampling_params": {"max_new_tokens": 1}},
                               port=port)
        self.assertEqual(len(result), 2)
        for packet in result:
            self.check_packet(packet, "A", [32], {"type": "length", "length": 1})

    def test_batch_admission_failure_cancels_running_sibling(self):
        log = self.output / "batch-admission-failure.log"
        with patch.dict("os.environ", SGLANG_TEST_BATCH_ADMISSION_FAILURE="-900"), \
             server(self.binary, self.checkpoint, log, "--max-active", "2",
                    "--prefix-cache-entries", "1") as port:
            status, result = send(port, GENERATE, {
                "input_ids": [[32], [33, 33], [34], [35]],
                "sampling_params": [{"max_new_tokens": 2048, "sampling_seed": 0},
                                    {"max_new_tokens": 1},
                                    {"max_new_tokens": 7, "sampling_seed": -900},
                                    {"max_new_tokens": 9}],
            })
            self.assertEqual(status, 503, result)
            self.assertEqual(result, {"error": {"message": "Injected batch admission failure"}})
            log_text = log.read_text()
            admitted, = [line for line in log_text.splitlines()
                         if line.startswith("cpu admit ") and line.endswith(" budget=2048")]
            markers = (admitted, "cpu finish budget=1", "cpu reject seed=-900",
                       admitted.replace("cpu admit ", "cpu cancel ", 1))
            offsets = [log_text.index(marker) for marker in markers]
            self.assertEqual(offsets, sorted(offsets))
            self.assertEqual(sorted(receipt["budget"] for receipt in receipts(log)), [1, 2048])
            self.check_batch_reuse(port)
        self.assertIn("cpu admit slot=1 prompt=1 budget=1", log.read_text())

    def test_batch_disconnect_cancels_all_admitted_lanes(self):
        log = self.output / "batch-disconnect.log"
        with server(self.binary, self.checkpoint, log, "--max-active", "2",
                    "--prefix-cache-entries", "1") as port:
            with generate_client(port, {
                "input_ids": [[32], [33, 33], [34]],
                "sampling_params": {"max_new_tokens": 2048, "sampling_seed": 0},
            }) as client:
                deadline = time.monotonic() + 3
                while True:
                    admitted = {line for line in log.read_text().splitlines()
                                if line.startswith("cpu admit ") and line.endswith(" budget=2048")}
                    if len(admitted) == 2:
                        break
                    self.assertLess(time.monotonic(), deadline, log.read_text())
                    time.sleep(0.01)
                with queue_client(port, 1) as scalar:
                    self.wait_queue(port, 2, 0)
                    self.assertEqual(sorted(item["prompt_tokens"] for item in receipts(log)),
                                     [1, 2])
                    abort(client)
                    with http.client.HTTPResponse(scalar) as reply:
                        reply.begin()
                        self.assertEqual(reply.status, 200)
                        self.check_packet(event(reply), "A", [32],
                                          {"type": "length", "length": 1})
                        self.assertEqual(event(reply), "[DONE]")
            self.wait_queue(port, 0, 0)
            cancelled = {line for line in log.read_text().splitlines()
                         if line.startswith("cpu cancel ")}
            self.assertEqual(cancelled, {line.replace("cpu admit ", "cpu cancel ", 1)
                                         for line in admitted})
            self.assertEqual([item["budget"] for item in receipts(log)], [2048, 2048, 1])
            self.check_batch_reuse(port)

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

    def start_abort_server(self, *, active=1, fail=False):
        controls = Path(self.enterContext(TemporaryDirectory(dir=self.output)))
        gate = controls / "step"
        log = self.output / f"{self._testMethodName}.log"
        environment = {"SGLANG_TEST_BATCH_STEP_GATE": str(gate)}
        if fail:
            environment["SGLANG_TEST_BATCH_ADMISSION_FAILURE"] = "-900"
        self.enterContext(patch.dict("os.environ", environment))
        port = self.enterContext(server(self.binary, self.checkpoint, log,
                                        "--max-active", str(active),
                                        "--prefix-cache-entries", "1"))
        return port, gate, log

    def abort_request(self, port, payload, *, status=200):
        with closing(http.client.HTTPConnection("127.0.0.1", port, timeout=5)) as connection:
            wire = payload if isinstance(payload, bytes) else json.dumps(payload)
            connection.request("POST", ABORT_REQUEST, wire, HEADERS)
            with connection.getresponse() as reply:
                body = reply.read()
                self.assertEqual(reply.status, status, body)
                if status == 200:
                    self.assertEqual(body, b"")
                else:
                    self.assertIn("error", json.loads(body))

    def check_named_finish(self, packet, rid, tokens, reason, *, index=None):
        text = "AB"[:len(tokens)]
        if index is None:
            checked = self.check_packet(packet, text, tokens, reason)
        else:
            checked = self.check_batch_frame(packet, index, text, tokens, reason, prompt_tokens=1)
        self.assertEqual(checked, rid)

    def check_aborted_stream(self, reply, rid, tokens, *, waiting=False):
        self.check_batch_headers(reply)
        self.check_named_finish(batch_frame(reply), rid, tokens,
                                ABORTED_WAITING if waiting else ABORTED)
        self.assertEqual(batch_frame(reply), "[DONE]")
        self.assertIsNone(batch_frame(reply))

    def check_reused_ids(self, port, ids):
        values = self.generate({"input_ids": [[32]] * len(ids), "rid": ids,
                                "sampling_params": {"max_new_tokens": 0}}, port=port)
        for packet, rid in zip(values, ids, strict=True):
            self.check_named_finish(packet, rid, [], {"type": "length", "length": 0})

    def test_request_id_shapes_and_abort_body(self):
        port, _, log = self.start_abort_server()
        zero = named(None, budget=0, seed=None, stream=False)
        omitted = {key: value for key, value in zero.items() if key != "rid"}
        allocated = [self.generate(payload, port=port)["meta_info"]["id"]
                     for payload in (zero, omitted)]
        self.assertEqual(len(set(allocated)), 2)
        self.assertTrue(all(isinstance(rid, str) and rid for rid in allocated))
        for rid, expected in (("caller", ["caller_0", "caller_1"]),
                              ([" exact ", "ʀid"], [" exact ", "ʀid"])):
            result = self.generate({"input_ids": [[32], [33]], "rid": rid,
                                    "sampling_params": {"max_new_tokens": 0}}, port=port)
            self.assertEqual([row["meta_info"]["id"] for row in result], expected)
        for rid in (3, True, {}, [], [""], ["only"], ["dup", "dup"], ["valid", 42]):
            with self.subTest(rid=rid):
                status, result = send(port, GENERATE, named(rid) | {"input_ids": [[32], [33]]})
                self.assertEqual(status, 400, result)
                self.assertRegex(result["error"]["message"], "rid|request ID")
        for payload in ([], {"rid": 3}, {"abort_all": 1}, {"abort_all": "yes"},
                        b'{"rid":"invalid",}'):
            with self.subTest(payload=payload):
                self.abort_request(port, payload, status=400)
        for payload in ({}, {"rid": None}, {"rid": ""}, {"rid": "unknown"},
                        {"abort_all": False}):
            self.abort_request(port, payload)
        self.assertFalse([row for row in receipts(log) if row["budget"]])

    def test_named_abort_bypasses_full_queue_and_keeps_partial_sse(self):
        port, gate, log = self.start_abort_server()
        gate.with_suffix(".0").touch()
        with generate_client(port, named("Active")) as active, \
             http.client.HTTPResponse(active) as stream:
            self.check_batch_headers(stream)
            self.check_named_finish(batch_frame(stream), "Active", [32], None)
            with generate_client(port, named("waiting", budget=77)) as queued, \
                 http.client.HTTPResponse(queued) as waiting:
                self.wait_queue(port, 1, 1)
                status, error = send(port, GENERATE, named("overflow", budget=0, stream=False))
                self.assertEqual(status, 503, error)
                for rid in ("unknown", "active", ""):
                    self.abort_request(port, {"rid": rid})
                self.wait_queue(port, 1, 1)
                self.abort_request(port, {"rid": "waiting"})
                self.abort_request(port, {"rid": "Active"})
                self.abort_request(port, {"rid": "Active"})
                self.check_aborted_stream(stream, "Active", [32])
                self.check_aborted_stream(waiting, "waiting", [], waiting=True)
        self.wait_queue(port, 0, 0)
        self.abort_request(port, {"rid": "Active"})
        self.check_reused_ids(port, ["Active", "waiting"])
        self.assertEqual([row["budget"] for row in receipts(log) if row["budget"]], [2])
        self.assertIn("cpu cancel slot=0 prompt=1 budget=2", log.read_text())

    def test_partially_aborted_http_batch_keeps_its_live_sibling(self):
        port, gate, log = self.start_abort_server()
        gate.with_suffix(".0").touch()
        ids = ["live", "queued-item"]
        payload = named(ids) | {"input_ids": [[32], [33]]}
        with generate_client(port, named("active")) as active, \
             http.client.HTTPResponse(active) as stream:
            self.check_batch_headers(stream)
            self.check_named_finish(batch_frame(stream), "active", [32], None)
            with generate_client(port, payload) as queued, \
                 http.client.HTTPResponse(queued) as batch:
                self.wait_queue(port, 1, 1)
                self.abort_request(port, {"rid": ids[1]})
                self.wait_queue(port, 1, 1)
                self.abort_request(port, {"rid": "active"})
                self.check_aborted_stream(stream, "active", [32])
                self.check_batch_headers(batch)
                first = [batch_frame(batch), batch_frame(batch)]
                for index, packet in enumerate(sorted(first, key=lambda item: item["index"])):
                    tokens, reason = ([32], None) if index == 0 else ([], ABORTED_WAITING)
                    self.check_named_finish(packet, ids[index], tokens, reason, index=index)
                gate.with_suffix(".1").touch()
                finish = {"type": "length", "length": 2}
                self.check_named_finish(batch_frame(batch), ids[0], [32, 33], finish, index=0)
                self.assertEqual(batch_frame(batch), "[DONE]")
                self.assertIsNone(batch_frame(batch))
        self.assertEqual([row["budget"] for row in receipts(log)], [2, 2])

    def test_aborted_http_queue_entry_releases_capacity_while_active_request_runs(self):
        port, gate, _ = self.start_abort_server()
        gate.with_suffix(".0").touch()
        with generate_client(port, named("active")) as active, \
             http.client.HTTPResponse(active) as stream:
            self.check_batch_headers(stream)
            self.check_named_finish(batch_frame(stream), "active", [32], None)
            with generate_client(port, named("queued", budget=77)) as queued, \
                 http.client.HTTPResponse(queued) as waiting:
                self.wait_queue(port, 1, 1)
                self.abort_request(port, {"rid": "queued"})
                self.wait_queue(port, 1, 0)
                self.check_aborted_stream(waiting, "queued", [], waiting=True)
            self.abort_request(port, {"rid": "active"})
            self.check_aborted_stream(stream, "active", [32])

    def test_live_collision_does_not_reserve_other_batch_ids(self):
        port, gate, log = self.start_abort_server(active=2)
        gate.with_suffix(".0").touch()
        held = "gen-1"
        with generate_client(port, named(held)) as client, \
             http.client.HTTPResponse(client) as reply:
            self.check_batch_headers(reply)
            self.check_named_finish(batch_frame(reply), held, [32], None)
            for colliding in (held, ["new", held], [held, "new"]):
                prompts = [[32], [33]] if isinstance(colliding, list) else [32]
                status, result = send(port, GENERATE, named(colliding) | {"input_ids": prompts})
                self.assertEqual(status, 400, result)
                self.assertRegex(result["error"]["message"], "rid|request ID")
            auto = self.generate(named(None, budget=0, seed=None, stream=False), port=port)
            self.check_named_finish(auto, "gen-2", [], {"type": "length", "length": 0})
            self.check_reused_ids(port, ["new"])
            self.abort_request(port, {"rid": held})
            self.check_aborted_stream(reply, held, [32])
        self.wait_queue(port, 0, 0)
        self.check_reused_ids(port, [held, "new"])
        admitted = [row["budget"] for row in receipts(log) if row["budget"]]
        self.assertEqual(admitted, [2])

    def test_batch_item_abort_and_abort_all(self):
        port, gate, _ = self.start_abort_server(active=2)
        gate.with_suffix(".0").touch()
        request_body = named("batch") | {"input_ids": [[32], [33], [34]]}
        with generate_client(port, request_body) as client, \
             http.client.HTTPResponse(client) as stream:
            self.check_batch_headers(stream)
            first = [batch_frame(stream), batch_frame(stream)]
            for index, packet in enumerate(sorted(first, key=lambda item: item["index"])):
                self.check_named_finish(packet, f"batch_{index}", [32], None, index=index)
            self.abort_request(port, {"rid": "batch_2"})
            self.check_named_finish(batch_frame(stream), "batch_2", [], ABORTED_WAITING, index=2)
            self.abort_request(port, {"rid": "batch_0"})
            lane = batch_frame(stream)
            self.check_named_finish(lane, "batch_0", [32], ABORTED, index=0)
            self.abort_request(port, {"rid": "missing", "abort_all": True})
            lane = batch_frame(stream)
            self.check_named_finish(lane, "batch_1", [32], ABORTED, index=1)
            self.assertEqual(batch_frame(stream), "[DONE]")
            self.assertIsNone(batch_frame(stream))
        self.wait_queue(port, 0, 0)
        self.check_reused_ids(port, ["batch_0", "batch_1", "batch_2"])

    def test_named_prefix_abort_only_matches_dispatched_siblings(self):
        port, gate, _ = self.start_abort_server(active=2)
        gate.with_suffix(".0").touch()
        ids = ["prefix", "prefix-running", "prefix-waiting"]
        payload = named(ids) | {"input_ids": [[32], [33], [34]]}
        with generate_client(port, payload) as client, http.client.HTTPResponse(client) as reply:
            self.check_batch_headers(reply)
            first = [batch_frame(reply), batch_frame(reply)]
            self.assertCountEqual([packet["meta_info"]["id"] for packet in first], ids[:2])
            self.abort_request(port, {"rid": "pre"})
            self.abort_request(port, {"rid": ids[0]})
            terminal = [batch_frame(reply) for _ in ids]
            for index, packet in enumerate(sorted(terminal, key=lambda item: item["index"])):
                self.check_named_finish(packet, ids[index], [32],
                                        ABORTED if index < 2 else None, index=index)
            gate.with_suffix(".1").touch()
            finish = {"type": "length", "length": 2}
            self.check_named_finish(batch_frame(reply), ids[2], [32, 33], finish, index=2)
            self.assertEqual(batch_frame(reply), "[DONE]")
            self.assertIsNone(batch_frame(reply))

    def test_finished_batch_item_reuse_survives_old_request_cleanup(self):
        self.check_id_reuse_with_owner(busy_owner=False)

    def test_finished_batch_item_reuse_with_busy_owner(self):
        self.check_id_reuse_with_owner(busy_owner=True)

    def check_id_reuse_with_owner(self, *, busy_owner):
        port, gate, log = self.start_abort_server(active=2)
        first_step = gate.with_suffix(".0")
        finish = {"type": "length", "length": 1}
        old_ids = ["reusable", "old-sibling"] if busy_owner else ["old-sibling", "reusable"]
        reuse_index = old_ids.index("reusable")
        sibling_index = old_ids.index("old-sibling")
        payload = named(old_ids) | {
            "input_ids": [[32], [33]],
            "sampling_params": [named(rid, budget=1 if rid == "reusable" else 2)["sampling_params"]
                                for rid in old_ids],
        }
        first_step.touch()
        with generate_client(port, named("parked", budget=3)) as parked, \
             http.client.HTTPResponse(parked) as parked_reply:
            self.check_batch_headers(parked_reply)
            self.check_named_finish(batch_frame(parked_reply), "parked", [32], None)
            first_step.unlink()
            log_offset = len(log.read_bytes())
            with generate_client(port, payload) as old, http.client.HTTPResponse(old) as stream:
                self.check_batch_headers(stream)
                self.wait_queue(port, 2, 0)
                self.abort_request(port, {"rid": "parked"})
                self.check_named_finish(batch_frame(parked_reply), "parked", [32], ABORTED)
                self.assertEqual(batch_frame(parked_reply), "[DONE]")
                self.assertIsNone(batch_frame(parked_reply))
                deadline = time.monotonic() + 3
                while True:
                    admitted = [line for line in log.read_bytes()[log_offset:].splitlines()
                                if line.startswith(b"cpu admit ")]
                    if len(admitted) == 2:
                        break
                    self.assertLess(time.monotonic(), deadline, admitted)
                    time.sleep(0.01)
                first_step.touch()
                first = {packet["index"]: packet for packet in
                         (batch_frame(stream), batch_frame(stream))}
                self.check_named_finish(first[reuse_index], "reusable", [32], finish,
                                        index=reuse_index)
                self.check_named_finish(first[sibling_index], "old-sibling", [32], None,
                                        index=sibling_index)
                self.check_live_reused_registration(port, stream, sibling_index)
        self.wait_queue(port, 0, 0)
        self.check_reused_ids(port, ["reusable", "old-sibling", "companion"])
        self.wait_queue(port, 0, 0)

    def check_live_reused_registration(self, port, stream, sibling_index):
        payload = named(["reusable", "companion"]) | {
            "input_ids": [[32], [32]],
            "sampling_params": [named("reusable")["sampling_params"], {"max_new_tokens": 0}],
        }
        duplicate = named(["reusable", "duplicate-probe"], budget=0, stream=False) | {
            "input_ids": [[32], [32]],
        }
        with generate_client(port, payload) as new, http.client.HTTPResponse(new) as reply:
            self.check_batch_headers(reply)
            self.check_named_finish(batch_frame(reply), "reusable", [32], None, index=0)
            self.wait_queue(port, 2, 0)
            self.abort_request(port, {"rid": "old-sibling"})
            self.check_named_finish(batch_frame(stream), "old-sibling", [32], ABORTED,
                                    index=sibling_index)
            self.assertEqual(batch_frame(stream), "[DONE]")
            self.assertIsNone(batch_frame(stream))
            self.wait_queue(port, 1, 0)
            self.check_named_finish(batch_frame(reply), "companion", [],
                                    {"type": "length", "length": 0}, index=1)
            status, result = send(port, GENERATE, duplicate)
            self.assertEqual(status, 400, result)
            self.assertIn("Duplicate request ID", result["error"]["message"])
            self.abort_request(port, {"rid": "reusable"})
            self.check_named_finish(batch_frame(reply), "reusable", [32], ABORTED, index=0)
            self.assertEqual(batch_frame(reply), "[DONE]")
            self.assertIsNone(batch_frame(reply))

    def test_ids_released_after_batch_failure(self):
        port, _, log = self.start_abort_server(active=2, fail=True)
        ids = ["running", "done", "failed", "tail"]
        options = [{"max_new_tokens": 20, "sampling_seed": -41}, {"max_new_tokens": 1},
                   {"max_new_tokens": 7, "sampling_seed": -900}, {"max_new_tokens": 9}]
        invalid = named(ids) | {"input_ids": [[32], [33], [34], [-1]]}
        status, result = send(port, GENERATE, invalid)
        self.assertEqual(status, 400, result)
        self.wait_queue(port, 0, 0)
        self.check_reused_ids(port, ids)
        for stream in (False, True):
            with self.subTest(stream=stream):
                payload = {"input_ids": [[32], [33], [34], [35]], "rid": ids, "stream": stream,
                           "sampling_params": options}
                with generate_client(port, payload) as client, \
                     http.client.HTTPResponse(client) as reply:
                    reply.begin()
                    if stream:
                        self.assertEqual(reply.status, 200)
                        lane = batch_frame(reply)
                        finish = {"type": "length", "length": 1}
                        self.check_named_finish(lane, "done", [32], finish, index=1)
                        self.assertIn("error", batch_frame(reply))
                        self.assertIsNone(batch_frame(reply))
                    else:
                        self.assertEqual(reply.status, 503)
                        self.assertIn("Injected batch admission failure", reply.read().decode())
                self.wait_queue(port, 0, 0)
                self.check_reused_ids(port, ids)
        log_text = log.read_text()
        self.assertEqual(log_text.count("cpu reject seed=-900"), 2)
        cancelled = [line for line in log_text.splitlines() if line.startswith("cpu cancel ")]
        self.assertEqual(len(cancelled), 2)
        self.assertTrue(all(line.endswith(" prompt=1 budget=20") for line in cancelled))
        self.assertNotIn(" budget=9\n", log_text)

    def test_ids_released_after_queued_and_stream_disconnect(self):
        port, gate, log = self.start_abort_server(active=2)
        gate.with_suffix(".0").touch()
        ids = ["disconnected-0", "disconnected-1", "disconnected-2"]
        payload = named(ids, budget=20) | {"input_ids": [[32], [33], [34]]}
        with generate_client(port, payload) as client:
            with http.client.HTTPResponse(client) as reply:
                self.check_batch_headers(reply)
                first = [batch_frame(reply), batch_frame(reply)]
                self.assertCountEqual([frame["meta_info"]["id"] for frame in first], ids[:2])
            with generate_client(port, named("slot-drop", budget=76)) as slot:
                self.wait_queue(port, 2, 0)
                with generate_client(port, named("queued-drop", budget=77)) as queued:
                    self.wait_queue(port, 2, 1)
                    abort(queued)
                self.wait_queue(port, 2, 0)
                abort(slot)
            self.wait_queue(port, 1, 0)
            abort(client)
        self.wait_queue(port, 0, 0)
        self.check_reused_ids(port, ids + ["slot-drop", "queued-drop"])
        seen = [row["budget"] for row in receipts(log) if row["budget"]]
        self.assertEqual(seen, [20, 20])
        cancelled = [line for line in log.read_text().splitlines()
                     if line.startswith("cpu cancel ")]
        self.assertCountEqual(cancelled, [f"cpu cancel slot={slot} prompt=1 budget=20"
                                         for slot in (0, 1)])

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
