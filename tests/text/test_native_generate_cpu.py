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
            ({"text": "A", "input_ids": [32]}, "exactly one"), ({"input_ids": []}, "input_ids"),
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
