"""Exercise native text stops with the real tokenizer, scheduler and loopback HTTP."""

import argparse
from contextlib import closing
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from test_native_chat import request, tokenizer_pieces
from test_native_generate_cpu import GENERATE, batch_frame
from test_sampling_routes_cpu import TEXT, body, receipts, server


def options(seed, budget, **extra):
    return {"temperature": 0, "sampling_seed": seed, "max_new_tokens": budget, **extra}


def stopped(match):
    return {"type": "stop", "matched": match}


def length(count):
    return {"type": "length", "length": count}


class NativeTextStops(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pieces = tokenizer_pieces(cls.checkpoint)
        by_piece = {piece: token for token, piece in pieces.items() if piece}
        cls.letters = [by_piece[bytes([letter])] for letter in b"ABCDEQZ"]
        cls.hello = by_piece[b"hello"]
        cls.snow = [by_piece[bytes([byte])] for byte in "⛄".encode()]
        cls.eos = 248046
        cls.special = 248047
        a, b, c, d, e, q, z = cls.letters
        scripts = {
            120: [cls.hello, q],
            121: [z, a, b, a, b, q],
            122: [a, *cls.snow, b, q],
            123: [a, b, c, d, e, q],
            124: [a, cls.snow[0], b],
            125: [a, b, a, q],
            126: [a, cls.eos, b],
            127: [a, cls.special, b, q],
        }
        script = cls.output / "text-stops.ids"
        script.write_text("".join(f"{seed} {' '.join(map(str, ids))}\n"
                                  for seed, ids in scripts.items()), encoding="ascii")
        cls.enterClassContext(patch.dict("os.environ", SGLANG_TEST_CHAT_OUTPUT_IDS=str(script),
                                         SGLANG_TEST_SCRIPTED_MTP="1"))
        cls.log = cls.output / "text-stops.log"
        cls.port = cls.enterClassContext(server(cls.binary, cls.checkpoint, cls.log,
                                               "--max-active", "2"))

    def exchange(self, params, stream, *, port=None, inputs=None, expected_status=200, **fields):
        packet = {"input_ids": inputs or [32], "sampling_params": params, "stream": stream,
                  **fields}
        connection, response = request(self.port if port is None else port, GENERATE, packet)
        with closing(connection), response:
            self.assertEqual(response.status, expected_status)
            if not stream or expected_status != 200:
                self.assertEqual(response.headers.get_content_type(), "application/json")
                return json.load(response)
            self.assertEqual(response.headers.get_content_type(), "text/event-stream")
            frames = []
            while (frame := batch_frame(response)) != "[DONE]":
                self.assertIsNotNone(frame, "EOF before DONE")
                if frame != "keepalive":
                    frames.append(frame)
            self.assertEqual(response.read(), b"")
            return frames

    def terminal(self, packet, text, ids, reason, prompts=1):
        self.assertEqual(packet["text"], text)
        self.assertEqual(packet["output_ids"], ids)
        meta = packet["meta_info"]
        self.assertEqual(meta["finish_reason"], reason)
        self.assertEqual(meta["prompt_tokens"], prompts)
        self.assertEqual(meta["completion_tokens"], len(ids))
        self.assertEqual(meta["cached_tokens"], 0)
        self.assertTrue(meta["id"])

    def sequence(self, frames, text, ids, reason, *, intermediate=None, prompts=1):
        self.assertTrue(frames)
        final = frames[-1]
        self.terminal(final, text, ids, reason, prompts)
        for frame in frames[:-1]:
            self.assertEqual(frame["meta_info"]["id"], final["meta_info"]["id"])
            self.assertIsNone(frame["meta_info"]["finish_reason"])
            self.assertTrue(text.startswith(frame["text"]), frames)
            count = frame["meta_info"]["completion_tokens"]
            self.assertEqual(frame["output_ids"], ids[:count])
        if intermediate is not None:
            self.assertEqual([frame["text"] for frame in frames[:-1]], intermediate)

    def scalar(self, params, text, ids, reason, intermediate, **kwargs):
        plain = self.exchange(params, False, **kwargs)
        self.assertNotIn("index", plain)
        self.terminal(plain, text, ids, reason)
        frames = self.exchange(params, True, **kwargs)
        self.assertTrue(all("index" not in frame for frame in frames))
        self.sequence(frames, text, ids, reason, intermediate=intermediate)

    def test_partial_token_and_cross_token_overlap(self):
        a, b, _, _, _, _, z = self.letters
        self.scalar(options(120, 4, stop="ll"), "he", [self.hello], stopped("ll"), [])
        self.scalar(options(121, 8, stop=["BAB", "ABA"]), "Z", [z, a, b, a],
                    stopped("ABA"), ["Z", "Z", "Z"])

    def test_utf8_and_length_releases_held_prefix(self):
        a, b, *_ = self.letters
        self.scalar(options(122, 8, stop="⛄B"), "A", [a, *self.snow, b],
                    stopped("⛄B"), ["A"] * 4)
        self.scalar(options(122, 4, stop="⛄B"), "A⛄", [a, *self.snow],
                    length(4), ["A"] * 3)
        self.scalar(options(124, 2, stop="⛄"), "A�", [a, self.snow[0]], length(2), ["A"])
        self.scalar(options(125, 4, stop="ABAB"), "ABAQ", [a, b, a, self.letters[5]],
                    length(4), ["", "", ""])

    def test_scalar_id_only_fast_path(self):
        a, b, *_ = self.letters
        for stop in (None, []):
            with self.subTest(stop=stop):
                params = options(123, 6, stop=stop, stop_token_ids=[b])
                self.scalar(params, "A", [a, b], stopped(b), ["A"])
        self.scalar(options(124, 4, stop_token_ids=[b]), "A�", [a, self.snow[0], b],
                    stopped(b), ["A", "A"])

    def test_no_stop_trim_text_and_length(self):
        a, b, c, *_ = self.letters
        for value, text in ((False, "A"), (None, "A"), (True, "ABC")):
            with self.subTest(value=value):
                self.scalar(options(123, 6, stop="BC", no_stop_trim=value),
                            text, [a, b, c], stopped("BC"), ["A", "A"])
        self.scalar(options(120, 4, stop="ll", no_stop_trim=True),
                    "hell", [self.hello], stopped("ll"), [])
        self.scalar(options(123, 2, stop="BC", no_stop_trim=True), "AB", [a, b],
                    length(2), ["A"])
        self.scalar(options(122, 8, stop="⛄B", no_stop_trim=True), "A⛄B",
                    [a, *self.snow, b], stopped("⛄B"), ["A"] * 4)

    def test_no_stop_trim_token_ids_and_eos(self):
        a, b, *_ = self.letters
        for value, text in ((False, "A"), (True, "AB")):
            with self.subTest(value=value):
                self.scalar(options(123, 6, stop_token_ids=[b], no_stop_trim=value),
                            text, [a, b], stopped(b), ["A"])
                self.scalar(options(126, 6, no_stop_trim=value),
                            "A", [a, self.eos], stopped(self.eos), ["A"])
        self.scalar(options(123, 6, stop_token_ids=[a], no_stop_trim=True),
                    "A", [a], stopped(a), [])

    def test_skip_special_tokens_defaults_and_terminal_ids(self):
        a, b, *_ = self.letters
        special = "<|object_ref_start|>"
        for params in ({}, {"skip_special_tokens": None}, {"skip_special_tokens": True},
                       {"skip_special_tokens": False}):
            with self.subTest(params=params):
                visible = params.get("skip_special_tokens") is False
                prefix = "A" + (special if visible else "")
                self.scalar(options(127, 3, **params), prefix + "B", [a, self.special, b],
                            length(3), ["A", prefix])
        for trim, text in ((False, "A"), (True, "A<|im_end|>")):
            self.scalar(options(126, 6, skip_special_tokens=False, no_stop_trim=trim),
                        text, [a, self.eos], stopped(self.eos), ["A"])
        self.scalar(options(127, 6, stop_token_ids=[self.special], skip_special_tokens=False,
                            no_stop_trim=True), "A" + special, [a, self.special],
                    stopped(self.special), ["A"])

    def test_skip_special_tokens_text_match_and_indexed_scores(self):
        a, b, *_ = self.letters
        special = "<|object_ref_start|>"
        params = [options(127, 4, stop=special, skip_special_tokens=False),
                  options(127, 4, stop=special, skip_special_tokens=False, no_stop_trim=True),
                  options(127, 3, stop=special)]
        scores = {"return_logprob": True, "top_logprobs_num": 2,
                  "return_text_in_logprobs": True}
        inputs = [[self.special], [self.special, a], [self.special, a, b]]
        plain = self.exchange(params, False, inputs=inputs, **scores)
        frames = self.exchange(params, True, inputs=inputs, **scores)
        expected = [("A", [a, self.special], stopped(special)),
                    ("A" + special, [a, self.special], stopped(special)),
                    ("AB", [a, self.special, b], length(3))]
        self.assertEqual(len(plain), len(expected))
        self.assertEqual({frame["index"] for frame in frames}, set(range(len(expected))))
        for index, (text, ids, reason) in enumerate(expected):
            with self.subTest(index=index):
                self.terminal(plain[index], text, ids, reason, index + 1)
                lane = [frame for frame in frames if frame["index"] == index]
                self.sequence(lane, text, ids, reason, prompts=index + 1)
                for frame in [plain[index], *lane]:
                    meta = frame["meta_info"]
                    selected = meta["output_token_logprobs"]
                    self.assertEqual([entry[1] for entry in selected], frame["output_ids"])
                    if len(selected) > 1:
                        self.assertEqual(selected[1], [-0.5, self.special, special])

    def test_invalid_skip_special_tokens_does_not_admit_batch(self):
        before = len(receipts(self.log))
        for value in (0, "false", [], {}):
            with self.subTest(value=value):
                params = [options(127, 2), options(127, 2, skip_special_tokens=value)]
                error = self.exchange(params, True, inputs=[[32], [33]], expected_status=400)
                self.assertIn("skip_special_tokens", error["error"]["message"])
        self.assertEqual(len(receipts(self.log)), before)

    def test_no_stop_trim_broadcast_to_parallel_samples(self):
        a, b, c, *_ = self.letters
        params = options(123, 6, n=2, stop="BC", no_stop_trim=True)
        inputs = [[a], [a, b]]
        plain = self.exchange(params, False, inputs=inputs)
        frames = self.exchange(params, True, inputs=inputs)
        self.assertEqual(len(plain), 4)
        self.assertEqual({frame["index"] for frame in frames}, set(range(4)))
        for index, packet in enumerate(plain):
            self.assertNotIn("index", packet)
            prompts = len(inputs[index // 2])
            self.terminal(packet, "ABC", [a, b, c], stopped("BC"), prompts)
            lane = [frame for frame in frames if frame["index"] == index]
            self.sequence(lane, "ABC", [a, b, c], stopped("BC"), prompts=prompts)

    def test_no_stop_trim_indexed_json_and_scored_sse(self):
        a, b, c, *_ = self.letters
        params = [options(123, 6, stop="BC", no_stop_trim=True),
                  options(123, 6, stop="QZ", stop_token_ids=[b], no_stop_trim=True),
                  options(123, 6, stop="BC", no_stop_trim=False)]
        inputs = [[a], [a, b], [a, b, c]]
        scores = {"return_logprob": [True, True, False],
                  "top_logprobs_num": [2, 2, 0], "return_text_in_logprobs": True}
        plain = self.exchange(params, False, inputs=inputs, **scores)
        frames = self.exchange(params, True, inputs=inputs, **scores)
        expected = [("ABC", [a, b, c], stopped("BC")), ("AB", [a, b], stopped(b)),
                    ("A", [a, b, c], stopped("BC"))]
        self.assertEqual(len(plain), len(expected))
        self.assertEqual({frame["index"] for frame in frames}, set(range(len(expected))))
        for index, (text, ids, reason) in enumerate(expected):
            with self.subTest(index=index):
                self.assertNotIn("index", plain[index])
                self.terminal(plain[index], text, ids, reason, index + 1)
                lane = [frame for frame in frames if frame["index"] == index]
                self.sequence(lane, text, ids, reason, prompts=index + 1)
                for frame in [plain[index], *lane]:
                    meta = frame["meta_info"]
                    if index == 2:
                        self.assertFalse(any("logprob" in key for key in meta))
                        continue
                    selected = [[-0.25 * ordinal, token, chr(ord("A") + self.letters.index(token))]
                                for ordinal, token in enumerate(frame["output_ids"], 1)]
                    self.assertEqual(meta["output_token_logprobs_length"], len(selected))
                    self.assertEqual(meta["output_token_logprobs"], selected)
                    alternatives = [[row, [row[0] - 1, 1, '"']] for row in selected]
                    self.assertEqual(meta["output_top_logprobs"], alternatives)

    def test_invalid_no_stop_trim_does_not_admit_batch(self):
        before = len(receipts(self.log))
        for value in (0, "true", [], {}):
            with self.subTest(value=value):
                params = [options(123, 2), options(123, 2, no_stop_trim=value)]
                error = self.exchange(params, True, inputs=[[32], [33]], expected_status=400)
                self.assertIn("no_stop_trim", error["error"]["message"])
        self.assertEqual(len(receipts(self.log)), before)

    def test_per_item_batch_json_and_sse(self):
        a, b, c, d, *_ = self.letters
        params = [options(123, 6, stop="BC"), options(123, 4, stop=["DX"]),
                  options(120, 4, stop="ell")]
        inputs = [[a], [a, b], [a, b, c]]
        expected = [("A", [a, b, c], stopped("BC")), ("ABCD", [a, b, c, d], length(4)),
                    ("h", [self.hello], stopped("ell"))]
        plain = self.exchange(params, False, inputs=inputs)
        frames = self.exchange(params, True, inputs=inputs)
        self.assertEqual(len(plain), len(expected))
        self.assertEqual({frame["index"] for frame in frames}, set(range(len(expected))))
        for index, (text, ids, reason) in enumerate(expected):
            with self.subTest(index=index):
                self.assertNotIn("index", plain[index])
                self.terminal(plain[index], text, ids, reason, index + 1)
                lane = [frame for frame in frames if frame["index"] == index]
                self.sequence(lane, text, ids, reason, prompts=index + 1)

    def test_invalid_stop_does_not_admit_batch(self):
        before = len(receipts(self.log))
        for value in (3, {"text": "B"}, ["B", None], ["B"] * 33):
            with self.subTest(value=value):
                params = [options(123, 2, stop="B"), options(123, 2, stop=value)]
                error = self.exchange(params, True, inputs=[[32], [33]], expected_status=400)
                self.assertIn("stop", error["error"]["message"])
        self.assertEqual(len(receipts(self.log)), before)

    def check_openai(self, port, seed, stop, text, ids, reason, budget):
        payload = body(TEXT, seed=seed, stop=stop, max_tokens=budget, temperature=0)
        connection, response = request(port, TEXT, payload)
        with closing(connection), response:
            self.assertEqual(response.status, 200)
            result = json.load(response)
        choice, = result["choices"]
        self.assertEqual((choice["text"], choice["token_ids"], choice["finish_reason"]),
                         (text, ids, reason))
        self.assertEqual(result["usage"]["completion_tokens"], len(ids))

    def test_openai_nonstream_keeps_ids_and_reason(self):
        a, b, c, *_ = self.letters
        for seed, stop, text, ids, reason, budget in (
            (120, "ll", "he", [self.hello], "stop", 4),
            (123, "BC", "A", [a, b, c], "stop", 5),
            (123, "BC", "AB", [a, b], "length", 2),
        ):
            with self.subTest(seed=seed, stop=stop, budget=budget):
                self.check_openai(self.port, seed, stop, text, ids, reason, budget)

    def test_mtp_clips_before_callback_and_delivers_terminal(self):
        a, b, c, d, e, q, _ = self.letters
        log = self.output / "text-stops-mtp.log"
        with server(self.binary, self.checkpoint, log, "--max-active", "2",
                    "--mtp-steps", "3") as port:
            for budget in (4, 7):
                with self.subTest(budget=budget):
                    start = len(log.read_text(encoding="utf-8").splitlines())
                    self.scalar(options(123, budget, stop="BC"), "A", [a, b, c],
                                stopped("BC"), ["A"], port=port)
                    lines = log.read_text(encoding="utf-8").splitlines()[start:]
                    expected = "cpu scripted mtp tokens=" + ("3 terminal=1" if budget == 4
                                                              else "4 terminal=0")
                    self.assertEqual(lines.count(expected), 2)
                    cancels = [line for line in lines if line.startswith("cpu cancel slot=")]
                    self.assertEqual(len(cancels), 0 if budget == 4 else 2)
            self.scalar(options(123, 6, stop="BC", stop_token_ids=[d]), "A", [a, b, c],
                        stopped("BC"), ["A"], port=port)
            self.scalar(options(123, 7, stop="BC", no_stop_trim=True), "ABC", [a, b, c],
                        stopped("BC"), ["A"], port=port)
            special_stop = "<|object_ref"
            self.scalar(options(127, 4, stop=special_stop, skip_special_tokens=False,
                                no_stop_trim=True),
                        "A" + special_stop, [a, self.special], stopped(special_stop),
                        ["A"], port=port)
            self.check_openai(port, 123, "BC", "A", [a, b, c], "stop", 6)
            self.scalar(options(123, 5, stop="QZ"), "ABCDE", [a, b, c, d, e],
                        length(5), ["A"], port=port)
            self.scalar(options(123, 6, stop="QZ"), "ABCDEQ", [a, b, c, d, e, q],
                        length(6), ["A", "ABCDE"], port=port)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args, unittest_args = parser.parse_known_args()
    NativeTextStops.binary = args.binary.resolve()
    NativeTextStops.checkpoint = args.checkpoint.resolve()
    NativeTextStops.output = args.output.resolve()
    NativeTextStops.output.mkdir(parents=True, exist_ok=True)
    unittest.main(argv=[__file__, *unittest_args])


if __name__ == "__main__":
    main()
