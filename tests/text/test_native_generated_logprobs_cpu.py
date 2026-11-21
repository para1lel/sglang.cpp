"""Check native score metadata through the HTTP worker with scripted decoder scores."""

import argparse
from contextlib import closing
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from test_native_chat import request, tokenizer_pieces
from test_native_generate_cpu import ABORTED, ABORT_REQUEST, GENERATE, batch_frame
from test_sampling_routes_cpu import receipts, server


def prompts(budget, **parameters):
    return {"input_ids": [32], "sampling_params": {
        "temperature": 0, "sampling_seed": 217, "ignore_eos": False,
        "max_new_tokens": budget, **parameters,
    }}


class NativeGeneratedLogprobs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pieces = tokenizer_pieces(cls.checkpoint)
        ids = {piece: token for token, piece in cls.pieces.items()}
        cls.hello = ids[b"hello"]
        cls.snow = [ids[bytes([byte])] for byte in "⛄".encode()]
        cls.stop_id = ids[b"Z"]
        script = cls.output / "generated-logprobs.ids"
        normal = [cls.hello, *cls.snow, cls.stop_id]
        script.write_text("217 " + " ".join(map(str, normal)) + "\n-41 32 33 34\n")
        cls.gate = cls.output / "generated-logprobs-gate"
        Path(f"{cls.gate}.0").touch()
        cls.enterClassContext(patch.dict("os.environ", SGLANG_TEST_CHAT_OUTPUT_IDS=str(script),
                                         SGLANG_TEST_BATCH_STEP_GATE=str(cls.gate)))
        cls.log = cls.output / "generated-logprobs.log"
        cls.port = cls.enterClassContext(server(cls.binary, cls.checkpoint, cls.log,
                                               "--max-active", "2"))

    def exchange(self, payload, expected=200, *, port=None):
        connection, response = request(port or self.port, GENERATE, payload)
        with closing(connection), response:
            self.assertEqual(response.status, expected)
            if expected != 200 or not payload.get("stream"):
                self.assertEqual(response.headers.get_content_type(), "application/json")
                return json.load(response)
            self.assertEqual(response.headers.get_content_type(), "text/event-stream")
            frames = []
            while (frame := batch_frame(response)) != "[DONE]":
                self.assertIsNotNone(frame, "Stream ended before DONE")
                if frame != "keepalive":
                    frames.append(frame)
            self.assertEqual(response.read(), b"")
            return frames

    def scores(self, frame, top, text=False):
        meta = frame["meta_info"]
        ids = frame["output_ids"]
        self.assertEqual(meta["completion_tokens"], len(ids))
        if top is None:
            self.assertFalse(any("logprob" in key for key in meta))
            return
        self.assertEqual(meta["input_token_logprobs"], [])
        self.assertEqual(meta["output_token_logprobs_length"], len(ids))
        selected = [[-0.25 * ordinal, token, self.pieces[token].decode(errors="replace")
                     if text else None] for ordinal, token in enumerate(ids, 1)]
        self.assertEqual(meta["output_token_logprobs"], selected)
        if top == 0:
            self.assertNotIn("input_top_logprobs", meta)
            self.assertNotIn("output_top_logprobs", meta)
            return
        self.assertEqual(meta["input_top_logprobs"], [])
        expected = [[selected_row] + [[selected_row[0] - rank, rank, None]
                                      for rank in range(1, top)] for selected_row in selected]
        self.assertEqual(meta["output_top_logprobs"], expected)

    def test_partial_and_id_stops(self):
        partial = prompts(7, stop="llo") | {
            "return_logprob": True, "return_text_in_logprobs": True, "stream": True,
        }
        frame, = self.exchange(partial)
        self.assertEqual((frame["text"], frame["output_ids"]), ("he", [self.hello]))
        self.assertEqual(frame["meta_info"]["finish_reason"], {"type": "stop", "matched": "llo"})
        self.scores(frame, 0, text=True)
        id_stop = prompts(7, stop_token_ids=[self.stop_id]) | {
            "return_logprob": True, "top_logprobs_num": 20, "stream": True,
        }
        frames = self.exchange(id_stop)
        self.assertEqual([frame["output_ids"] for frame in frames],
                         [[self.hello, *self.snow, self.stop_id][:length]
                          for length in range(1, 6)])
        self.assertEqual(frames[-1]["text"], "hello⛄")
        self.assertEqual(frames[-1]["meta_info"]["finish_reason"],
                         {"type": "stop", "matched": self.stop_id})
        for frame in frames:
            self.assertNotIn("index", frame)
            self.scores(frame, 20)
        partial_utf8 = self.exchange(prompts(2) | {
            "return_logprob": True, "return_text_in_logprobs": True,
        })
        self.assertEqual(partial_utf8["text"], "hello�")
        self.scores(partial_utf8, 0, text=True)

    def test_mixed_batch_zero_budget_and_opt_out(self):
        payload = prompts(4) | {
            "input_ids": [[32], [33], [34]],
            "return_logprob": [True, False, True], "top_logprobs_num": [20, 12, 3],
            "logprob_start_len": [-1, None, -1], "token_ids_logprob": None,
            "sampling_params": [prompts(count)["sampling_params"] for count in (4, 4, 0)],
        }
        expected_tops = [20, None, 3]
        plain = self.exchange(payload)
        for index, frame in enumerate(plain):
            self.assertNotIn("index", frame)
            self.scores(frame, expected_tops[index])
        self.assertEqual(plain[2]["output_ids"], [])
        frames = self.exchange(payload | {"stream": True})
        finals = {}
        for frame in frames:
            index = frame["index"]
            self.scores(frame, expected_tops[index])
            self.assertEqual(frame["output_ids"], plain[index]["output_ids"][
                :frame["meta_info"]["completion_tokens"]])
            if frame["meta_info"]["finish_reason"] is not None:
                self.assertNotIn(index, finals)
                finals[index] = frame
        self.assertEqual(set(finals), {0, 1, 2})

        implicit = self.exchange(prompts(4) | {"rid": "opt-out-parity", "stream": True})
        explicit = self.exchange(prompts(4) | {
            "rid": "opt-out-parity", "stream": True, "return_logprob": False,
            "top_logprobs_num": 20, "return_text_in_logprobs": True,
        })
        self.assertEqual(explicit, implicit)
        for frame in explicit:
            self.scores(frame, None)

    def test_abort_and_id_reuse(self):
        payload = prompts(8, sampling_seed=-41) | {
            "rid": "scored-abort", "return_logprob": True, "top_logprobs_num": 2, "stream": True,
        }
        connection, response = request(self.port, GENERATE, payload)
        with closing(connection), response:
            self.assertEqual(response.status, 200)
            first = batch_frame(response)
            self.assertEqual(first["output_ids"], [32])
            self.scores(first, 2)
            abort_connection, abort_reply = request(
                self.port, ABORT_REQUEST, {"rid": "scored-abort"})
            with closing(abort_connection), abort_reply:
                self.assertEqual(abort_reply.status, 200)
                self.assertEqual(abort_reply.read(), b"")
            final = batch_frame(response)
            self.assertEqual(final["meta_info"]["finish_reason"], ABORTED)
            self.assertEqual(final["output_ids"], [32])
            self.scores(final, 2)
            self.assertEqual(batch_frame(response), "[DONE]")
            self.assertEqual(response.read(), b"")
        reused = self.exchange(payload | {"sampling_params": prompts(1)["sampling_params"],
                                          "stream": False})
        self.assertEqual(reused["output_ids"], [self.hello])
        self.scores(reused, 2)

    def test_validation_before_any_admission(self):
        start = len(receipts(self.log))
        scalar = prompts(1) | {"return_logprob": True, "stream": True}
        for fields, error in [
            ({"return_logprob": "yes"}, "return_logprob"),
            ({"return_logprob": [True]}, "return_logprob"),
            ({"return_text_in_logprobs": [True]}, "return_text_in_logprobs"),
            ({"logprob_start_len": 0}, "logprob_start_len"),
            ({"token_ids_logprob": []}, "token_ids_logprob"),
            ({"top_logprobs_num": -1}, "top_logprobs_num"),
            ({"top_logprobs_num": 21}, "top_logprobs_num"),
            ({"top_logprobs_num": 1.5}, "top_logprobs_num"),
            ({"sampling_params": {"max_new_tokens": 1}}, "greedy"),
            ({"input_ids": [[32], [33]], "top_logprobs_num": [20, "bad"]}, "top_logprobs_num"),
            ({"input_ids": [[32], [33]], "logprob_start_len": [-1]}, "logprob_start_len"),
        ]:
            with self.subTest(fields=fields):
                result = self.exchange(scalar | fields, 400)
                self.assertIn(error, result["error"]["message"])
        self.assertEqual(len(receipts(self.log)), start)
        greedy = self.exchange(scalar | {"sampling_params": {"top_k": 1, "max_new_tokens": 1}})
        self.scores(greedy[-1], 0)
        mtp_log = self.output / "generated-logprobs-mtp.log"
        with server(self.binary, self.checkpoint, mtp_log, "--max-active", "1",
                    "--mtp-steps", "1") as port:
            result = self.exchange(scalar, 400, port=port)
            self.assertIn("MTP disabled", result["error"]["message"])
            self.assertEqual(receipts(mtp_log), [])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments, remaining = parser.parse_known_args()
    NativeGeneratedLogprobs.binary = arguments.binary.resolve()
    NativeGeneratedLogprobs.checkpoint = arguments.checkpoint.resolve()
    NativeGeneratedLogprobs.output = arguments.output.resolve()
    NativeGeneratedLogprobs.output.mkdir(parents=True, exist_ok=True)
    unittest.main(argv=[__file__, *remaining])
