"""Check native parallel sampling with the scripted CPU decoder."""

from __future__ import annotations

import json
import time
import unittest
from contextlib import closing
from typing import TYPE_CHECKING
from unittest.mock import patch
from uuid import UUID

from test_native_generate_cpu import (
    ABORT_REQUEST,
    ABORTED,
    GENERATE,
    abort,
    batch_frame,
    generate_client,
    named,
    raw_response,
)
from test_native_generated_logprobs_cpu import scripted_server
from test_sampling_routes_cpu import receipts, run_cpu_suite, send, server, wire

if TYPE_CHECKING:
    from http.client import HTTPResponse


class NativeParallelSampling(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        scripts = {-41: [32, 33], -42: [34, 35]}
        cls.log = cls.output / "parallel.log"
        cls.port = cls.enterClassContext(
            scripted_server(cls, "token-ids.txt", scripts, cls.log, "--max-active", "1")
        )

    def generate(self, payload, *, port=None):
        status, result = send(self.port if port is None else port, GENERATE, payload)
        self.assertEqual(status, 200, result)
        return result

    def sample_ids(self, packets):
        ids = {item["meta_info"]["id"] for item in packets}
        self.assertEqual(len(ids), len(packets))
        for rid in ids:
            parsed = UUID(rid)
            self.assertEqual(parsed.hex, rid)
            self.assertEqual(parsed.version, 4)
        return ids

    def first_two_samples(self, stream: HTTPResponse) -> list[dict[str, object]]:
        self.assertEqual(stream.status, 200)
        frames = sorted((batch_frame(stream), batch_frame(stream)), key=lambda item: item["index"])
        self.assertEqual([item["index"] for item in frames], [0, 1])
        self.assertTrue(all(item["meta_info"]["finish_reason"] is None for item in frames))
        return frames

    def check_cancelled_sample(self, stream: HTTPResponse, rid: str) -> None:
        terminal = batch_frame(stream)
        self.assertEqual(terminal["meta_info"]["id"], rid)
        self.assertEqual(terminal["meta_info"]["finish_reason"], ABORTED)
        self.assertEqual(terminal["output_ids"], [32])

    def check_id_conflict(self, port: int, payload: dict[str, object], rid: str) -> None:
        status, result = send(port, GENERATE, payload)
        self.assertEqual(status, 400, result)
        self.assertIn(f"Duplicate request ID detected: {rid}", result["error"]["message"])

    def test_scalar_results_and_n_one(self):
        params = {"n": 2, "max_new_tokens": 2, "sampling_seed": -41}
        for prompt in ({"text": "A"}, {"input_ids": [32]}):
            with self.subTest(prompt=prompt):
                scalar = self.generate(prompt | {"rid": "one",
                                                "sampling_params": params | {"n": 1}})
                self.assertEqual(scalar["output_ids"], [32, 33])
                self.assertEqual(scalar["meta_info"]["id"], "one")
                self.assertNotIn("index", scalar)
                nullable = self.generate(prompt | {"sampling_params": params | {"n": None}})
                self.assertEqual(nullable["output_ids"], scalar["output_ids"])
                payload = prompt | {"rid": ["one"], "cache_salt": ["tenant"],
                                    "sampling_params": [params]}
                plain = self.generate(payload)
                frames = self.generate(payload | {"stream": True})
                self.assertEqual(len(plain), 2)
                self.assertEqual(len(frames), 4)
                final_frames = [item for item in frames if item["meta_info"]["finish_reason"]]
                self.assertTrue(self.sample_ids(plain).isdisjoint(self.sample_ids(final_frames)))
                for index, item in enumerate(plain):
                    self.assertEqual(item["output_ids"], scalar["output_ids"])
                    self.assertNotIn("index", item)
                    own = [frame for frame in frames if frame["index"] == index]
                    self.assertEqual([frame["output_ids"] for frame in own], [[32], [32, 33]])
                    rid = own[0]["meta_info"]["id"]
                    self.assertEqual(own[-1], item | {"index": index, "meta_info":
                                                     item["meta_info"] | {"id": rid}})

    def test_prompt_major_batch_and_seed_forwarding(self):
        params = [{"n": 2, "max_new_tokens": 2, "sampling_seed": -41},
                  {"n": 2, "max_new_tokens": 1, "sampling_seed": -42}]
        for prompt in ({"text": ["A", "B"]}, {"input_ids": [[32], [33, 33]]}):
            with self.subTest(prompt=prompt):
                first = len(receipts(self.log))
                payload = prompt | {"rid": ["first", "second"], "sampling_params": params}
                result = self.generate(payload)
                self.assertEqual([item["output_ids"] for item in result],
                                 [[32, 33], [32, 33], [34], [34]])
                self.sample_ids(result)
                seen = receipts(self.log)[first:]
                self.assertEqual([item["seed"] for item in seen], [-41, -41, -42, -42])
                self.assertEqual([item["budget"] for item in seen], [2, 2, 1, 1])
                count = 2 if "input_ids" in prompt else 1
                self.assertEqual([item["meta_info"]["prompt_tokens"] for item in result],
                                 [1, 1, count, count])
        maximum = self.generate({"text": "A", "rid": "limit",
                                 "sampling_params": {"n": 32, "max_new_tokens": 0}})
        self.assertEqual(len(maximum), 32)
        self.sample_ids(maximum)
        self.assertTrue(all(item["output_ids"] == [] for item in maximum))
        first = len(receipts(self.log))
        self.generate({"text": "A", "sampling_params": {"n": 2, "max_new_tokens": 1}})
        self.assertEqual([item["seed"] for item in receipts(self.log)[first:]], [None, None])

    def test_broadcast_logprobs_and_text_stop(self):
        result = self.generate({"text": "A", "return_logprob": True, "top_logprobs_num": 1,
                                "sampling_params": {"n": 2, "temperature": 0,
                                                    "max_new_tokens": 1, "sampling_seed": -41}})
        self.assertEqual(len(result), 2)
        for item in result:
            meta = item["meta_info"]
            self.assertEqual(meta["output_token_logprobs_length"], 1)
            self.assertEqual(meta["output_token_logprobs"][0][1], item["output_ids"][0])
        frames = self.generate({"input_ids": [32], "stream": True,
                                "sampling_params": {"n": 2, "stop": "B", "max_new_tokens": 4,
                                                    "sampling_seed": -41}})
        self.assertEqual(len(frames), 4)
        for index in range(2):
            own = [item for item in frames if item["index"] == index]
            self.assertEqual([item["text"] for item in own], ["A", "A"])
            self.assertEqual(own[-1]["output_ids"], [32, 33])
            self.assertEqual(own[-1]["meta_info"]["finish_reason"],
                             {"type": "stop", "matched": "B"})

    def test_expansion_validation_precedes_decoder_admission(self):
        first = len(receipts(self.log))
        invalid = [
            *[({"text": "A", "sampling_params": {"n": count}}, "sampling_params.n")
              for count in (-1, 0, True, 2.0, "2", 33)],
            ({"text": ["A", "B"], "sampling_params": [{"n": 2}, {"n": None}]}, "same"),
            ({"text": ["A", "B"], "sampling_params": {"n": 17}}, "at most 32"),
            ({"text": ["A", "B"], "sampling_params": [
                {"n": 2, "max_new_tokens": 1}, {"n": 2, "max_new_tokens": 131072}
            ]}, "262144"),
            ({"text": ["A", "B"], "rid": ["same", "same"],
              "sampling_params": {"n": 2}}, "Duplicate request ID"),
            ({"text": "A", "rid": ["first", "second"],
              "sampling_params": {"n": 2}}, "one rid string per batch item"),
            ({"text": "A", "rid": [3], "sampling_params": {"n": 2}},
             "rid string for every batch item"),
            *[({"text": ["A", "B"], field: values, "sampling_params": {"n": 2}}, field)
              for field, values in (("return_logprob", [False, False]),
                                    ("logprob_start_len", [-1, -1]),
                                    ("top_logprobs_num", [0, 0]))],
        ]
        for payload, message in invalid:
            with self.subTest(payload=payload):
                status, result = send(self.port, GENERATE, payload | {"stream": True})
                self.assertEqual(status, 400, result)
                self.assertIn(message, result["error"]["message"])
        self.assertEqual(len(receipts(self.log)), first)
        result = self.generate({"text": "A", "rid": "same",
                                "sampling_params": {"n": 2, "max_new_tokens": 0}})
        self.assertEqual(len(result), 2)
        self.sample_ids(result)

    def test_named_abort_and_disconnect(self):
        gate = self.output / "step"
        log = self.output / "gated.log"
        payload = {"input_ids": [32], "stream": True, "rid": "parallel",
                   "sampling_params": {"n": 3, "sampling_seed": -41, "max_new_tokens": 2}}
        gate.with_suffix(".0").touch()
        with patch.dict("os.environ", SGLANG_TEST_BATCH_STEP_GATE=str(gate)), \
             server(self.binary, self.checkpoint, log, "--max-active", "2") as port:
            with closing(generate_client(port, payload)) as client, raw_response(client) as stream:
                first = self.first_two_samples(stream)
                self.assertTrue(all(item["output_ids"] == [32] for item in first))
                admitted = [line for line in log.read_text().splitlines()
                            if line.startswith("cpu admit ")]
                self.assertEqual(admitted, ["cpu admit slot=0 prompt=1 budget=2",
                                            "cpu admit slot=1 prompt=1 budget=2"])
                ids = self.sample_ids(first)
                rid = first[1]["meta_info"]["id"]
                self.abort_id(port, payload["rid"])
                self.abort_id(port, rid)
                next_frames = {item["index"]: item for item in
                               (batch_frame(stream), batch_frame(stream))}
                self.assertEqual(next_frames.keys(), {1, 2})
                terminal = next_frames[1]
                self.assertEqual(terminal["index"], 1)
                self.assertEqual(terminal["meta_info"]["id"], rid)
                self.assertEqual(terminal["output_ids"], [32])
                self.assertEqual(terminal["meta_info"]["finish_reason"], ABORTED)
                third = next_frames[2]
                self.assertEqual(third["output_ids"], [32])
                third_id = third["meta_info"]["id"]
                self.assertNotIn(third_id, ids)
                gate.with_suffix(".1").touch()
                remaining = {item["index"]: item for item in
                             (batch_frame(stream), batch_frame(stream))}
                self.assertEqual(remaining.keys(), {0, 2})
                for index, expected_id in ((0, first[0]["meta_info"]["id"]), (2, third_id)):
                    self.assertEqual(remaining[index]["meta_info"]["id"], expected_id)
                    self.assertEqual(remaining[index]["output_ids"], [32, 33])
                    self.assertEqual(remaining[index]["meta_info"]["finish_reason"],
                                     {"type": "length", "length": 2})
                self.assertEqual(batch_frame(stream), "[DONE]")
                self.assertIsNone(batch_frame(stream))
            gate.with_suffix(".1").unlink()
            self.check_disconnect(port, payload, log)

    def abort_id(self, port, rid):
        status, _, body = wire(
            port,
            "POST",
            ABORT_REQUEST,
            json.dumps({"rid": rid}),
            {"Content-Type": "application/json"},
        )
        self.assertEqual(status, 200)
        self.assertEqual(body, b"")

    def test_normalized_ids_checked_against_live_requests(self):
        gate = self.output / "duplicate-step"
        gate.with_suffix(".0").touch()
        log = self.output / "duplicate.log"
        active = named(["collision_2"]) | {"input_ids": [[32]]}
        with (
            patch.dict("os.environ", SGLANG_TEST_BATCH_STEP_GATE=str(gate)),
            server(self.binary, self.checkpoint, log, "--max-active", "2") as port,
            closing(generate_client(port, active)) as client,
            raw_response(client) as stream,
        ):
            self.assertEqual(batch_frame(stream)["meta_info"]["id"], "collision_2")
            parallel = {"text": "A", "sampling_params": {"n": 3, "max_new_tokens": 0}}
            for rid in ("collision", ["collision_2"]):
                self.check_id_conflict(port, parallel | {"rid": rid}, "collision_2")
            self.assertEqual(len(receipts(log)), 1)
            result = self.generate(parallel | {"rid": "collision_2"}, port=port)
            self.assertEqual(len(result), 3)
            self.sample_ids(result)
            self.abort_id(port, "collision_2")
            self.assertEqual(batch_frame(stream)["meta_info"]["finish_reason"], ABORTED)
            self.assertEqual(batch_frame(stream), "[DONE]")

    def test_terminal_child_reused_as_normalized_parent_while_sibling_runs(self):
        gate = self.output / "terminal-parent-step"
        gate.with_suffix(".0").touch()
        log = self.output / "terminal-parent.log"
        active = named("owner", budget=41)
        active["sampling_params"]["n"] = 2
        with (
            patch.dict("os.environ", SGLANG_TEST_BATCH_STEP_GATE=str(gate)),
            server(self.binary, self.checkpoint, log, "--max-active", "3") as port,
            closing(generate_client(port, active)) as client,
            raw_response(client) as stream,
        ):
            first = self.first_two_samples(stream)
            old_ids = self.sample_ids(first)
            retired, sibling = (item["meta_info"]["id"] for item in first)
            parallel = {"input_ids": [[32], [32]], "rid": [retired, "unowned"],
                        "sampling_params": {"n": 2, "max_new_tokens": 0}}
            admitted = len(receipts(log))
            self.check_id_conflict(port, parallel, retired)
            self.assertEqual(len(receipts(log)), admitted)

            self.abort_id(port, retired)
            self.check_cancelled_sample(stream, retired)
            self.abort_id(port, retired)
            result = self.generate(parallel, port=port)
            self.assertEqual(len(result), 4)
            self.assertTrue(self.sample_ids(result).isdisjoint(old_ids | {"unowned"}))
            self.assertTrue(all(item["output_ids"] == [] for item in result))
            self.check_id_conflict(port, parallel | {"rid": [retired, sibling]}, sibling)

            replacement = {"input_ids": [[32], [32]], "rid": [retired, "companion"],
                           "sampling_params": [{"max_new_tokens": 42, "sampling_seed": -41},
                                               {"max_new_tokens": 0}], "stream": True}
            with (
                closing(generate_client(port, replacement)) as other,
                raw_response(other) as new_stream,
            ):
                self.assertEqual(new_stream.status, 200)
                first_replacement = batch_frame(new_stream)
                self.assertEqual(first_replacement["index"], 0)
                self.assertEqual(first_replacement["meta_info"]["id"], retired)
                self.assertIsNone(first_replacement["meta_info"]["finish_reason"])
                self.abort_id(port, sibling)
                self.check_cancelled_sample(stream, sibling)
                self.assertEqual(batch_frame(stream), "[DONE]")
                self.assertIsNone(batch_frame(stream))
                companion = batch_frame(new_stream)
                self.assertEqual(companion["meta_info"]["id"], "companion")
                status, error = send(port, GENERATE, named(retired, budget=0, stream=False))
                self.assertEqual(status, 400, error)
                self.assertIn("Duplicate request ID", error["error"]["message"])
                self.abort_id(port, retired)
                self.check_cancelled_sample(new_stream, retired)
                self.assertEqual(batch_frame(new_stream), "[DONE]")
                self.assertIsNone(batch_frame(new_stream))

    def check_disconnect(self, port, payload, log):
        with closing(generate_client(port, payload)) as client:
            with raw_response(client) as stream:
                self.assertEqual(stream.status, 200)
                self.assertEqual({batch_frame(stream)["index"] for _ in range(2)}, {0, 1})
            abort(client)
        deadline = time.monotonic() + 5
        while log.read_text().count("cpu cancel ") < 3 and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(log.read_text().count("cpu cancel "), 3)
        result = self.generate(payload | {"stream": False, "sampling_params": {
            "n": 3, "max_new_tokens": 0,
        }}, port=port)
        self.assertEqual(len(result), 3)
        self.sample_ids(result)
        self.assertEqual([item["budget"] for item in receipts(log)], [2, 2, 2, 2, 2, 0, 0, 0])


if __name__ == "__main__":
    run_cpu_suite(NativeParallelSampling, __file__, __doc__)
