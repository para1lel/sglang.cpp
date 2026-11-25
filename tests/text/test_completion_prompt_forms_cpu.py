"""Exercise four /v1/completions prompt forms through the CPU host server."""

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

from test_native_text import HEADERS, MODEL, event, events, exchange, legacy, raw, usage
from test_sampling_routes_cpu import receipts, server


class CompletionPrompts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.log = cls.output / "prompt-forms.log"
        cls.port = cls.enterClassContext(server(cls.binary, cls.checkpoint, cls.log,
                                               "--max-active", "2", sampling=False))

    def plain(self, prompt, budget, *, port=None, **options):
        mime, data = exchange(port or self.port, raw(prompt, budget, **options))
        self.assertEqual(mime, "application/json")
        return json.loads(data)

    def check_json(self, packet, prompts, outputs, reasons):
        self.assertEqual(set(packet), {"choices", "usage"})
        self.assertEqual(packet["usage"], usage(sum(prompts, []), sum(outputs, [])))
        for index, (choice, prompt, ids, reason) in enumerate(
            zip(packet["choices"], prompts, outputs, reasons, strict=True)
        ):
            self.assertEqual(choice, {"index": index, "text": choice["text"],
                                      "prompt_token_ids": prompt, "token_ids": ids,
                                      "finish_reason": reason})

    def check_stream(self, frames, expected, *, include_usage):
        if include_usage:
            self.assertEqual(frames[-1], {"choices": [], "usage": expected["usage"]})
            frames = frames[:-1]
        choices = expected["choices"]
        seen = [[] for _ in choices]
        for packet in frames:
            self.assertEqual(set(packet), {"choices"})
            choice, = packet["choices"]
            index = choice["index"]
            self.assertIn(index, range(len(choices)))
            seen[index].append(choice)
        for track, choice in zip(seen, choices, strict=True):
            self.assertTrue(track)
            self.assertEqual(track[0]["prompt_token_ids"], choice["prompt_token_ids"])
            self.assertTrue(all("prompt_token_ids" not in item for item in track[1:]))
            self.assertEqual(sum((item["token_ids"] for item in track), []), choice["token_ids"])
            self.assertEqual("".join(item["text"] for item in track), choice["text"])
            self.assertTrue(all(item["finish_reason"] is None for item in track[:-1]))
            self.assertEqual(track[-1]["finish_reason"], choice["finish_reason"])
            self.assertEqual(track[-1]["token_ids"], [])

    def test_four_forms_in_both_modes(self):
        hi = [12675]
        mixed = [12675, 220, 96748]
        inputs = (
            ("Hi 世界", [mixed], [[172, 253, 239]]),
            (mixed, [mixed], [[172, 253, 239]]),
            (["Hi", "Hi 世界", "Hi"], [hi, mixed, hi], [[248046, 33, 172], [172, 253, 239],
                                                       [248046, 33, 172]]),
            ([hi, mixed, hi], [hi, mixed, hi], [[248046, 33, 172], [172, 253, 239],
                                              [248046, 33, 172]]),
        )
        packets = []
        for prompt, ids, outputs in inputs:
            with self.subTest(prompt=prompt):
                packet = self.plain(prompt, 3, cache_salt="openai-forms")
                self.check_json(packet, ids, outputs, ["length"] * len(ids))
                self.check_stream(events(self.port, raw(prompt, 3, stream=True,
                                                       stream_options={"include_usage": True})),
                                  packet, include_usage=True)
                packets.append(packet)
        if len(packets) == len(inputs):
            self.assertEqual(packets[0], packets[1])
            self.assertEqual(packets[2], packets[3])

    def test_zero_budget_and_usage_options(self):
        for prompt, ids in (("Hi", [12675]), ([248319], [248319]),
                            (["Hi"], [12675]), ([[248319]], [248319])):
            with self.subTest(prompt=prompt):
                packet = self.plain(prompt, 0, n=1)
                self.check_json(packet, [ids], [[]], ["length"])
                for options in ({}, {"stream_options": {"include_usage": False}},
                                {"stream_options": {"include_usage": True}}):
                    frames = events(self.port, raw(prompt, 0, stream=True, **options))
                    self.check_stream(frames, packet, include_usage=options.get(
                        "stream_options", {}).get("include_usage", False))
        prompts = [[32]] * 32
        self.check_json(self.plain(prompts, 0), prompts, [[]] * 32, ["length"] * 32)

    def test_text_only_stream_and_inclusive_output_limit(self):
        prompts = [[32] * 7, [33] * 7]
        options = {"return_token_ids": False, "stop_token_ids": [32], "ignore_eos": False}
        counts = usage([32] * 14, [32, 32])
        choices = [{"index": index, "text": "", "finish_reason": "stop"} for index in range(2)]
        self.assertEqual(self.plain(prompts, 131072, **options),
                         {"choices": choices, "usage": counts})
        frames = events(self.port, raw(prompts, 131072, stream=True, **options,
                                       stream_options={"include_usage": True}))
        self.assertCountEqual(frames[:-1], [{"choices": [choice]} for choice in choices])
        self.assertEqual(frames[-1], {"choices": [], "usage": counts})

    def test_invalid_requests_stop_before_admission(self):
        before = self.log.read_text()
        cases = (
            ("null prompt", {"prompt": None}),
            ("empty text", {"prompt": ""}),
            ("empty vector", {"prompt": []}),
            ("empty item", {"prompt": ["Hi", ""]}),
            ("empty ID item", {"prompt": [[32], []]}),
            ("mixed text and IDs", {"prompt": ["Hi", [32]]}),
            ("mixed text and number", {"prompt": [32, "Hi"]}),
            ("mixed IDs and number", {"prompt": [[32], 32]}),
            ("mixed IDs and text", {"prompt": [[32], "Hi"]}),
            ("nested too deeply", {"prompt": [[32], [[33]]]}),
            ("negative flat ID", {"prompt": [-1]}),
            ("invalid flat vocabulary", {"prompt": [248320]}),
            ("invalid batch vocabulary", {"prompt": [[32], [248320]]}),
            ("noninteger batch ID", {"prompt": [[32], [33.0]]}),
            ("boolean batch ID", {"prompt": [[32], [True]]}),
            ("too many prompts", {"prompt": ["Hi"] * 33}),
            ("batch output total", {"prompt": [[32]] * 3, "max_tokens": 87382}),
            ("per-item ID context", {"prompt": [[32], [33] * 131073], "max_tokens": 131072}),
            ("per-item text context", {"prompt": ["Hi", " x" * 131073], "max_tokens": 131072}),
            ("flat context", {"prompt": [32, 33], "max_tokens": 262143}),
            ("text byte limit", {"prompt": ["Hi", "é" * 524289]}),
            ("multiple choices", {"n": 2}),
            ("noninteger choice count", {"n": 1.0}),
            ("wrong model", {"model": "invalid"}),
            ("nonstream usage", {"stream": False, "stream_options": {"include_usage": True}}),
            ("invalid usage type", {"stream_options": {"include_usage": 1}}),
            ("unrecognized usage option", {"stream_options": {"include_usage": True, "other": 0}}),
            ("invalid stop ID", {"stop_token_ids": [248320]}),
            ("invalid sampling", {"top_p": 0}),
            ("oversize cache salt", {"cache_salt": "a" * 257}),
        )
        for label, fields in cases:
            with self.subTest(label=label):
                payload = raw([[32], [33]], 1, stream=True) | fields
                mime, data = exchange(self.port, payload, 400)
                self.assertEqual(mime, "application/json")
                message = json.loads(data)["error"]["message"]
                self.assertTrue(message)
                if "context" in label:
                    self.assertIn("context", message)
        self.assertEqual(self.log.read_text(), before)

    def test_stop_is_per_choice_and_usage_counts_retained_ids(self):
        prompts = [[32] * 4, [33] * 7]
        for stops in ({"stop": "A"}, {"stop_token_ids": [32]}):
            with self.subTest(stops=stops):
                options = {"ignore_eos": False, **stops}
                packet = self.plain(prompts, 2, **options)
                self.check_json(packet, prompts, [[253, 239], [32]], ["length", "stop"])
                self.assertEqual(packet["choices"][1]["text"], "")
                frames = events(self.port, raw(prompts, 2, stream=True, **options,
                                               stream_options={"include_usage": True}))
                self.check_stream(frames, packet, include_usage=True)

    def gated_server(self, width=2):
        log = self.output / f"{self._testMethodName}.log"
        controls = Path(self.enterContext(TemporaryDirectory(dir=self.output)))
        gate = controls / "step"
        self.enterContext(patch.dict("os.environ", SGLANG_TEST_BATCH_STEP_GATE=str(gate)))
        port = self.enterContext(server(self.binary, self.checkpoint, log, "--max-active",
                                        str(width), sampling=False))
        return port, gate, log

    def connect(self, port, payload):
        wire = json.dumps(payload).encode()
        client = self.enterContext(socket.create_connection(("127.0.0.1", port), timeout=5))
        client.sendall(
            b"POST /v1/completions HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n"
            b"Content-Type: application/json\r\nContent-Length: " + str(len(wire)).encode()
            + b"\r\n\r\n" + wire
        )
        return client

    def begin_stream(self, client):
        reply = self.enterContext(http.client.HTTPResponse(client))
        reply.begin()
        self.assertEqual(reply.status, 200)
        self.assertEqual(reply.headers.get_content_type(), "text/event-stream")
        self.assertEqual(reply.getheader("Transfer-Encoding"), "chunked")
        return reply

    def test_incremental_stream_with_out_of_order_finish(self):
        port, gate, _ = self.gated_server()
        prompts = [[32] * 4, [33] * 7]
        payload = raw(prompts, 2, stream=True, seed=-41, ignore_eos=False, stop_token_ids=[32],
                      stream_options={"include_usage": True})
        client = self.connect(port, payload)
        reply = self.begin_stream(client)
        gate.with_name(gate.name + ".0").touch()
        first = []
        for _ in range(3):
            first.append(event(reply))
        choices = [packet["choices"][0] for packet in first]
        self.assertIn({"index": 1, "text": "", "token_ids": [], "finish_reason": "stop"}, choices)
        self.assertEqual([choice["token_ids"] for choice in choices if choice["index"] == 0],
                         [[253]])
        self.assertFalse(select.select([client], [], [], 0.1)[0])
        gate.with_name(gate.name + ".1").touch()
        rest = list(iter(lambda: event(reply), "[DONE]"))
        self.assertEqual(reply.read(1), b"")
        expected = self.plain(prompts, 2, ignore_eos=False, stop_token_ids=[32])
        self.check_stream(first + rest, expected, include_usage=True)
        evidence = {"before_second_step": first, "after_second_step": rest + ["[DONE]"]}
        (self.output / "incremental-stream.json").write_text(json.dumps(evidence, indent=2) + "\n")

    def wait_queue(self, port, running, waiting):
        deadline = time.monotonic() + 5
        expected = {f'vllm:num_requests_running{{engine="0"}} {running}',
                    f'vllm:num_requests_waiting{{engine="0"}} {waiting}'}
        while time.monotonic() < deadline:
            with closing(http.client.HTTPConnection("127.0.0.1", port, timeout=5)) as connection:
                connection.request("GET", "/metrics", headers=HEADERS)
                with connection.getresponse() as reply:
                    self.assertEqual(reply.status, 200)
                    metrics = reply.read().decode().splitlines()
            if expected <= set(metrics):
                return
            time.sleep(0.01)
        self.fail(f"Missing metrics {expected!r} in {metrics!r}")

    def disconnect(self, client):
        client.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
        client.close()

    def test_active_disconnect_retires_items_and_skips_unsubmitted_tail(self):
        port, gate, log = self.gated_server()
        gate.with_name(gate.name + ".0").touch()
        payload = raw([[32] * 4, [33] * 7, [34] * 9], 4, stream=True, seed=-41)
        active = self.connect(port, payload)
        reply = self.begin_stream(active)
        self.assertEqual({event(reply)["choices"][0]["index"] for _ in range(2)}, {0, 1})
        self.wait_queue(port, 1, 0)
        reply.close()
        self.disconnect(active)
        self.wait_queue(port, 0, 0)
        lines = log.read_text().splitlines()
        admitted = {line for line in lines if line.startswith("cpu admit ")}
        cancelled = {line for line in lines if line.startswith("cpu cancel ")}
        self.assertEqual(len(admitted), 2)
        self.assertEqual(cancelled, {line.replace("cpu admit ", "cpu cancel ", 1)
                                     for line in admitted})
        self.assertCountEqual([line.split(" prompt=")[1] for line in admitted],
                              ["4 budget=4", "7 budget=4"])
        followup = self.plain([[32], [33]], 1, port=port)
        self.check_json(followup, [[32], [33]], [[248046], [248046]], ["length", "length"])

    def test_queued_disconnect_never_admits_either_prompt(self):
        port, gate, log = self.gated_server(width=1)
        active = self.connect(port, raw([32] * 7, 1, stream=True, seed=-41))
        reply = self.begin_stream(active)
        queued = self.connect(port, raw([[35] * 5, [36] * 6], 6))
        self.wait_queue(port, 1, 1)
        self.disconnect(queued)
        self.wait_queue(port, 1, 0)
        gate.with_name(gate.name + ".0").touch()
        self.assertEqual(event(reply)["choices"][0]["token_ids"], [32])
        self.assertEqual(event(reply)["choices"][0]["finish_reason"], "length")
        self.assertEqual(event(reply), "[DONE]")
        self.assertEqual(reply.read(1), b"")
        self.wait_queue(port, 0, 0)
        admissions = [line for line in log.read_text().splitlines()
                      if line.startswith("cpu admit ")]
        self.assertEqual(admissions, ["cpu admit slot=0 prompt=7 budget=1"])

    def test_inherited_sampling_and_singleton_routes(self):
        log = self.output / "inherited-sampling.log"
        with server(self.binary, self.checkpoint, log, "--max-active", "2") as port:
            mime, data = exchange(port, {"model": MODEL, "prompt": "Hi"})
            self.assertEqual(mime, "application/json")
            packet = json.loads(data)
            self.assertNotIn("prompt_token_ids", packet["choices"][0])
            self.assertNotIn("token_ids", packet["choices"][0])
            self.assertEqual(packet["choices"][0]["finish_reason"], "stop")
            self.assertEqual(packet["usage"], usage([12675], [32, 248046]))
            options = {"temperature": 0.75, "top_p": 0.875, "top_k": 13, "min_p": 0.125,
                       "seed": -9, "cache_salt": "salt"}
            output = self.plain([[32], [33, 34]], 1, port=port, **options)
            self.check_json(output, [[32], [33, 34]], [[32], [32]], ["length", "length"])
            default, *batch = receipts(log)
            self.assertEqual(default["budget"], 16)
            self.assertEqual(default["temperature"], 1)
            self.assertCountEqual([row["prompt_tokens"] for row in batch], [1, 2])
            for row in batch:
                self.assertEqual({key: row[key] for key in options if key != "cache_salt"},
                                 {key: value for key, value in options.items()
                                  if key != "cache_salt"})
                self.assertTrue(row["sampled"])
        frames = events(self.port, legacy([32], 1))
        self.assertEqual(frames, [
            {"choices": [{"index": 0, "token_ids": [248046], "prompt_token_ids": [32],
                          "finish_reason": None}]},
            {"choices": [{"index": 0, "token_ids": [], "finish_reason": "length"}]},
            {"choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 1,
                                     "prompt_tokens_details": {"cached_tokens": 0}}},
        ])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options, tests = parser.parse_known_args()
    CompletionPrompts.binary = options.server.resolve(strict=True)
    CompletionPrompts.checkpoint = options.checkpoint.resolve(strict=True)
    CompletionPrompts.output = options.output.resolve()
    CompletionPrompts.output.mkdir(parents=True, exist_ok=True)
    unittest.main(argv=[__file__, *tests], verbosity=2)
