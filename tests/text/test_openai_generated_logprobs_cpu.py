"""Exercise generated logprobs through the OpenAI routes with a scripted decoder."""

from collections import defaultdict
import os
import unittest
from unittest.mock import patch

from test_native_text import legacy, stream_options
from test_sampling_routes_cpu import (
    CHAT, TEXT, body, checked_body, post_response, run_cpu_suite, server, sse_frames,
)


class OpenAILogprobs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        scripts = cls.output / "scripted-scores.txt"
        scripts.write_text("11 32 33 34 35 36\n12 172 254 106 115 5513\n", encoding="ascii")
        cls.enterClassContext(patch.dict(os.environ, SGLANG_TEST_CHAT_OUTPUT_IDS=str(scripts),
                                         SGLANG_TEST_SCRIPTED_MTP="1"))
        cls.port = cls.enterClassContext(server(cls.binary, cls.checkpoint,
                                               cls.output / "scores.log", "--max-active", "4",
                                               "--mtp-steps", "0", sampling=False))

    def payload(self, route, top=3, **options):
        score = {"logprobs": top} if route == TEXT else {
            "logprobs": True, "top_logprobs": top,
        }
        defaults = {"temperature": 0, "ignore_eos": True, "max_tokens": 5}
        return body(route, **(defaults | score | options))

    def exchange(self, route, payload, port=None, status=200):
        with post_response(port or self.port, route, payload) as response:
            self.assertEqual(response.status, status, (route, payload))
            frames = sse_frames(response) if status == 200 and payload.get("stream") else None
            return checked_body(self, response, frames)

    def pair(self, route, payload, port=None):
        answer = self.exchange(route, payload, port)
        frames = self.exchange(route, payload | stream_options(), port)
        self.assertEqual(frames[-1], {**frames[-1], "choices": [], "usage": answer["usage"]})
        grouped = defaultdict(list)
        for frame in frames[:-1]:
            for choice in frame["choices"]:
                grouped[choice["index"]].append(choice)
        self.assertEqual(sorted(grouped), list(range(len(answer["choices"]))))
        for choice in answer["choices"]:
            chunks = grouped[choice["index"]]
            self.assertEqual(chunks[-1]["finish_reason"], choice["finish_reason"])
            if not (payload.get("echo") and len(chunks) == 1):
                self.assertIsNone(chunks[-1]["logprobs"])
            self.assertTrue(all(chunk["finish_reason"] is None for chunk in chunks[:-1]))
            self.assertEqual([token for chunk in chunks for token in chunk.get("token_ids", [])],
                             choice.get("token_ids", []))
            rows = [chunk["logprobs"] for chunk in chunks if chunk["logprobs"] is not None]
            for field, expected in choice["logprobs"].items():
                self.assertEqual([entry for row in rows for entry in row[field]], expected)
            if route == CHAT:
                self.assertEqual(chunks[0]["delta"], {"role": "assistant"})
                self.assertIsNone(chunks[0]["logprobs"])
                text = "".join(chunk["delta"].get("content") or "" for chunk in chunks)
                self.assertEqual(text, choice["message"]["content"] or "")
            else:
                self.assertEqual("".join(chunk["text"] for chunk in chunks), choice["text"])
        return answer, grouped

    def selected(self, route, choice):
        scores = choice["logprobs"]
        if route == TEXT:
            return scores["token_logprobs"]
        return [row["logprob"] for row in scores["content"]]

    def visible(self, route, choice):
        return choice["text"] if route == TEXT else choice["message"]["content"] or ""

    def test_indexed_prompt_forms_and_top_twenty(self):
        for scalar, batch in (("Test", ["Test", "Several extra words"]),
                              ([32], [[32], [33, 34, 35]])):
            with self.subTest(prompt=batch):
                many, _ = self.pair(TEXT, self.payload(TEXT, top=20, prompt=batch))
                first, _ = self.pair(TEXT, self.payload(TEXT, top=20, prompt=scalar))
                self.assertEqual(first["choices"][0]["logprobs"], many["choices"][0]["logprobs"])
                expected_usage = 0
                for index, prompt in enumerate(batch):
                    single = self.exchange(TEXT, self.payload(TEXT, top=20, prompt=prompt))
                    choice = many["choices"][index]
                    reference = single["choices"][0]
                    for field in ("text", "token_ids", "logprobs", "finish_reason"):
                        self.assertEqual(choice[field], reference[field])
                    expected_usage += single["usage"]["prompt_tokens"]
                    self.assertEqual(choice["logprobs"]["text_offset"], [-1] * 5)
                    for selected, alternatives in zip(self.selected(TEXT, choice),
                                                      choice["logprobs"]["top_logprobs"]):
                        self.assertEqual(sorted(alternatives.values(), reverse=True),
                                         [selected - rank for rank in range(20)])
                self.assertNotEqual(many["choices"][0]["token_ids"],
                                    many["choices"][1]["token_ids"])
                self.assertEqual(many["usage"]["prompt_tokens"], expected_usage)
                self.assertEqual(many["usage"]["completion_tokens"], 10)

    def test_indexed_choices_share_prompt_usage(self):
        for route in (TEXT, CHAT):
            scalar = self.exchange(route, self.payload(route, seed=11, max_tokens=3))
            reference = scalar["choices"][0]
            prompt_tokens = scalar["usage"]["prompt_tokens"]
            for count in (2, 4):
                with self.subTest(route=route, n=count):
                    answer, grouped = self.pair(
                        route, self.payload(route, seed=11, max_tokens=3, n=count))
                    self.assertEqual([choice["index"] for choice in answer["choices"]],
                                     list(range(count)))
                    self.assertEqual(answer["usage"]["prompt_tokens"], prompt_tokens)
                    self.assertEqual(answer["usage"]["completion_tokens"], 3 * count)
                    self.assertEqual(answer["usage"]["total_tokens"], prompt_tokens + 3 * count)
                    for choice in answer["choices"]:
                        self.assertEqual(choice, reference | {"index": choice["index"]})
                        self.assertGreater(len(grouped[choice["index"]]), 1)

    def test_chat_and_completion_bytes_and_stops(self):
        for route in (TEXT, CHAT):
            with self.subTest(route=route):
                answer, chunks = self.pair(route, self.payload(route, top=20, seed=12))
                choice = answer["choices"][0]
                self.assertEqual(choice["token_ids"], [172, 254, 106, 115, 5513])
                self.assertEqual(self.visible(route, choice), "𠮷�")
                self.assertEqual(self.selected(route, choice), [-0.25, -0.5, -0.75, -1, -1.25])
                self.assertEqual(sum(chunk["logprobs"] is not None for chunk in chunks[0]), 5)
                if route == CHAT:
                    rows = choice["logprobs"]["content"]
                    self.assertEqual([row["bytes"] for row in rows],
                                     [[240], [160], [174], [183], [239, 191, 189]])
                    self.assertEqual([row["token"] for row in rows], ["�"] * 5)
                    for row in rows:
                        self.assertEqual([score["logprob"] for score in row["top_logprobs"]],
                                         [row["logprob"] - rank for rank in range(20)])
                else:
                    self.assertEqual(choice["logprobs"]["tokens"], ["ð", "\xa0", "®", "·", "�"])

                for options, count in (({"stop": "BC"}, 3),
                                       ({"stop_token_ids": [33], "ignore_eos": False}, 2)):
                    with self.subTest(route=route, stops=options):
                        answer, chunks = self.pair(route, self.payload(
                            route, seed=11, return_token_ids=False, **options))
                        choice = answer["choices"][0]
                        self.assertNotIn("token_ids", choice)
                        self.assertEqual(choice["finish_reason"], "stop")
                        self.assertEqual(self.visible(route, choice), "A")
                        self.assertEqual(answer["usage"]["completion_tokens"], count)
                        self.assertEqual(self.selected(route, choice),
                                         [-0.25 * index for index in range(1, count + 1)])
                        hidden_scores = [chunk for chunk in chunks[0] if chunk["logprobs"] and
                                         not (chunk["text"] if route == TEXT else
                                              chunk["delta"].get("content"))]
                        self.assertEqual(len(hidden_scores), count - 1)

    def test_selected_only_empty_and_greedy_top_k_one(self):
        for route in (TEXT, CHAT):
            for budget in (0, 2):
                with self.subTest(route=route, budget=budget):
                    answer, _ = self.pair(route, self.payload(
                        route, top=0, max_tokens=budget, temperature=1, top_k=1, seed=11))
                    choice = answer["choices"][0]
                    self.assertEqual(choice["token_ids"], [32, 33][:budget])
                    self.assertEqual(self.selected(route, choice), [-0.25, -0.5][:budget])
                    self.assertEqual(answer["usage"]["completion_tokens"], budget)
                    rows = choice["logprobs"]
                    if route == TEXT:
                        self.assertEqual(rows["top_logprobs"], [])
                    else:
                        self.assertTrue(all(row["top_logprobs"] == [] for row in rows["content"]))

    def test_echo_prompt_forms_usage_and_stops(self):
        cases = (("BC\n", ["BC\n"]), (["BC", "  A\n"], ["BC", "  A\n"]),
                 ([248046, 33, 34], ["BC"]), ([[248046, 33, 34], [32]], ["BC", "A"]))
        for prompt, rendered in cases:
            for budget in (0, 4):
                for top in (0, 2):
                    with self.subTest(prompt=prompt, budget=budget, top=top):
                        payload = self.payload(TEXT, top=top, prompt=prompt, echo=True,
                                               seed=11, max_tokens=budget, stop="BC")
                        answer, chunks = self.pair(TEXT, payload)
                        count = 3 if budget else 0
                        self.assertEqual(answer["usage"]["completion_tokens"],
                                         count * len(rendered))
                        for choice, text in zip(answer["choices"], rendered):
                            self.assertEqual(choice["text"], text + ("A" if budget else ""))
                            self.assertEqual(choice["finish_reason"], "stop" if budget else
                                             "length")
                            self.assertEqual(chunks[choice["index"]][0]["text"],
                                             text + ("A" if budget else ""))
                            ids = choice["prompt_token_ids"]
                            prompt_values = [None] + [-1.5 * offset
                                                      for offset in range(1, len(ids))]
                            expected = (prompt_values if top else []) + [
                                -0.25 * offset for offset in range(1, count + 1)
                            ]
                            scores = choice["logprobs"]
                            self.assertEqual(scores["token_logprobs"], expected)
                            self.assertEqual(scores["text_offset"], [-1] * len(expected))
                            if top:
                                self.assertIsNone(scores["top_logprobs"][0])
                                self.assertEqual(scores["top_logprobs"][1:len(ids)],
                                                 [{"A": -0.125, "B": -0.25}] * (len(ids) - 1))
                            else:
                                self.assertEqual(scores["top_logprobs"], [])

    def test_echo_unscored_and_stop_never_crosses_prompt(self):
        for echo in (False, True):
            answer = self.exchange(TEXT, body(TEXT, prompt="B", seed=11, echo=echo,
                                              temperature=0, max_tokens=1, stop="BA"))
            choice = answer["choices"][0]
            self.assertEqual(choice["text"], "BA" if echo else "A")
            self.assertEqual(choice["finish_reason"], "length")
            self.assertNotIn("logprobs", choice)
        answer = self.exchange(TEXT, body(TEXT, prompt=[248046], echo=True, max_tokens=0))
        self.assertEqual(answer["choices"][0]["text"], "")
        self.assertEqual(answer["usage"]["completion_tokens"], 0)

    def test_validation_opt_out_and_frozen_legacy(self):
        for route, invalid in (
            (TEXT, ({"logprobs": False}, {"logprobs": -1}, {"logprobs": 21},
                    {"logprobs": 1.5}, {"echo": "true"}, {"n": 0}, {"n": 5}, {"regex": "A+"},
                    {"temperature": .5})),
            (CHAT, ({"logprobs": 1}, {"top_logprobs": -1}, {"top_logprobs": 21},
                    {"top_logprobs": 1.5}, {"n": 0}, {"n": 5}, {"regex": "A+"},
                    {"temperature": .5})),
        ):
            for options in invalid:
                with self.subTest(route=route, options=options):
                    answer = self.exchange(route, self.payload(route, stream=True, **options),
                                           status=400)
                    self.assertTrue(answer["error"]["message"])
            plain = self.payload(route, logprobs=None if route == TEXT else False)
            for answer in (self.exchange(route, plain),
                           *self.exchange(route, plain | stream_options())):
                self.assertTrue(all("logprobs" not in choice for choice in answer["choices"]))
        accepted = self.exchange(TEXT, legacy([32, 33], 2))
        self.assertEqual(accepted[-1]["usage"]["completion_tokens"], 2)
        rejected = self.exchange(TEXT, legacy([32, 33], 2, logprobs=0), status=400)
        self.assertIn("frozen benchmark", rejected["error"]["message"])

    def test_implicit_mtp_scores_scalar_and_indexed_choices(self):
        log = self.output / "scores-default.log"
        with server(self.binary, self.checkpoint, log, "--max-active", "2",
                    sampling=False) as port:
            for route in (TEXT, CHAT):
                with self.subTest(route=route):
                    payload = self.payload(route, seed=11)
                    offset = log.stat().st_size
                    result, chunks = self.pair(route, payload, port)
                    choice, = result["choices"]
                    self.assertEqual(choice["token_ids"], [32, 33, 34, 35, 36])
                    self.assertEqual(self.selected(route, choice), [-0.25, -0.5, -0.75, -1, -1.25])
                    self.assertEqual(sum(chunk["logprobs"] is not None for chunk in chunks[0]), 5)
                    receipt = b"cpu scripted mtp tokens=4 terminal=1\n"
                    self.assertEqual(log.read_bytes()[offset:].count(receipt), 2)
                    plain = payload | {"logprobs": None if route == TEXT else False}
                    result = self.exchange(route, plain, port)
                    unscored, = result["choices"]
                    self.assertEqual(unscored["token_ids"], choice["token_ids"])
                    self.assertNotIn("logprobs", unscored)
                    self.assertEqual(log.read_bytes()[offset:].count(receipt), 3)
                    for budget, mtp_rounds in ((5, 4), (1, 0)):
                        with self.subTest(budget=budget):
                            offset = log.stat().st_size
                            batch = payload | {"n": 2, "max_tokens": budget}
                            grouped, _ = self.pair(route, batch, port)
                            self.assertEqual([item["index"] for item in grouped["choices"]], [0, 1])
                            self.assertEqual(grouped["usage"]["completion_tokens"], 2 * budget)
                            for item in grouped["choices"]:
                                self.assertEqual(item["token_ids"], choice["token_ids"][:budget])
                                self.assertEqual(self.selected(route, item),
                                                 self.selected(route, choice)[:budget])
                                self.assertEqual(item["finish_reason"], "length")
                            log_tail = log.read_bytes()[offset:]
                            self.assertEqual(log_tail.count(b"cpu scripted mtp tokens="),
                                             mtp_rounds)
                            self.assertEqual(log_tail.count(receipt), mtp_rounds)

    def test_explicit_speculative_boundary(self):
        log = self.output / "scores-mtp.log"
        with server(self.binary, self.checkpoint, log, "--max-active", "2", "--mtp-steps", "3",
                    sampling=False) as port:
            for route in (TEXT, CHAT):
                with self.subTest(route=route):
                    before = log.stat().st_size
                    invalid = self.payload(route, seed=11, stream=True, temperature=0.5)
                    error = self.exchange(route, invalid, port, status=400)
                    self.assertIn("greedy", error["error"]["message"])
                    self.assertNotIn(b"cpu admit", log.read_bytes()[before:])
                    result, chunks = self.pair(route, self.payload(route, seed=11, max_tokens=6),
                                               port)
                    choice, = result["choices"]
                    self.assertEqual(choice["token_ids"], [32, 33, 34, 35, 36, 32])
                    self.assertEqual(self.selected(route, choice),
                                     [-0.25, -0.5, -0.75, -1, -1.25, -1.5])
                    self.assertEqual(sum(chunk["logprobs"] is not None for chunk in chunks[0]), 6)
                    self.assertEqual(result["usage"]["completion_tokens"], 6)
                    self.assertEqual(choice["finish_reason"], "length")
                    scored_log = log.read_bytes()[before:]
                    self.assertEqual(scored_log.count(b"cpu scripted mtp tokens=4 terminal=0\n"), 2)
                    self.assertEqual(scored_log.count(b"cpu scripted mtp tokens="), 2)
                    plain = self.payload(route, seed=11, stream=True,
                                         logprobs=None if route == TEXT else False, stop="BC")
                    chunks = [choice for frame in self.exchange(route, plain, port)
                              for choice in frame["choices"]]
                    self.assertEqual(chunks[-1]["finish_reason"], "stop")
                    self.assertTrue(all("logprobs" not in choice for choice in chunks))
            self.assertIn("cpu scripted mtp tokens=", log.read_text())


if __name__ == "__main__":
    run_cpu_suite(OpenAILogprobs, __file__, __doc__, server_option=True, select_tests=True)
