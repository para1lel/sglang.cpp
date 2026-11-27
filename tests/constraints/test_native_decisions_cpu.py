"""Check the Decisions adapter with the checkpoint tokenizer and scripted scores."""

import argparse
from contextlib import closing
import json
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "text"))
from test_native_chat import request
from test_sampling_routes_cpu import server


QUESTIONS = [
    {"id": "kind", "type": "choice", "question": "Category?",
     "options": [{"name": "technical"}, {"name": "billing", "description": ["invoice"]}]},
    {"id": "severity", "type": "score", "question": "Severity?",
     "levels": ["Calm", "Frustrated", "Very angry"]},
    {"id": "issue", "type": "yes_no", "question": "Is there a problem?"},
]


class NativeDecisions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.port = cls.enterClassContext(server(cls.binary, cls.checkpoint, cls.output / "cpu.log"))

    def exchange(self, payload, status=200):
        connection, reply = request(self.port, "/v1/decisions", payload)
        with closing(connection), reply:
            self.assertEqual(reply.status, status)
            return json.load(reply)

    def test_labels_assignment_and_temperature(self):
        payload = {"input": {"message": "Laptop is broken.", "tags": ["é", 1.0]},
                   "questions": QUESTIONS, "return_prompt_token_ids": True}
        ordinary = self.exchange(payload)
        colder = self.exchange({**payload, "temperature": 0.5})
        self.assertEqual(list(ordinary["answers"]), ["kind", "severity", "issue"])
        self.assertEqual(ordinary["answers"]["kind"]["choice"], "technical")
        for key, label_ids in zip(ordinary["answers"], ([32, 33], [15, 16, 17], [9405, 2083])):
            answer = ordinary["answers"][key]
            cold = colder["answers"][key]
            self.assertEqual(answer["label_token_ids"], label_ids)
            self.assertEqual(answer["prompt_token_ids"], cold["prompt_token_ids"])
            mass = sum(math.exp(-2 - index) for index in range(len(label_ids)))
            self.assertAlmostEqual(answer["label_mass"], mass)
            self.assertEqual(answer["label_mass"], cold["label_mass"])
            self.assertAlmostEqual(sum(answer["probabilities"].values()), 1.0)
            first = next(iter(answer["probabilities"]))
            self.assertLess(answer["probabilities"][first], cold["probabilities"][first])
        severity = ordinary["answers"]["severity"]
        self.assertAlmostEqual(severity["score"], sum(
            int(level) * probability for level, probability in severity["probabilities"].items()
        ))
        self.assertEqual(ordinary["usage"]["completion_tokens"], 0)
        self.assertEqual(ordinary["usage"]["prompt_tokens"], sum(
            len(answer["prompt_token_ids"]) for answer in ordinary["answers"].values()
        ))

    def test_empty_and_invalid_options(self):
        payload = {"input": "Laptop is broken.", "questions": QUESTIONS}
        for changed in ({"input": []}, {"questions": []}, {"temperature": 0},
                        {"label_token_ids": [32, 33]}, {"questions": [QUESTIONS[0]] * 1025}):
            with self.subTest(changed=list(changed)):
                self.exchange({**payload, **changed}, 400)
        for options in ([], [{"name": "only"}], [{"name": " Straße "}, {"name": "STRASSE"}]):
            with self.subTest(options=options):
                self.exchange({**payload, "questions": [{**QUESTIONS[0], "options": options}]}, 400)

    def test_prompt_format_version(self):
        payload = {"input": "Laptop is broken.", "questions": QUESTIONS,
                   "return_prompt_token_ids": True}
        baseline = self.exchange(payload)
        self.assertEqual(baseline["prompt_format_version"], 1)
        for value in (1, None, True, 1.0, "1", "0_1", "0__1", "+0_1.000", "\u00a01\u202f"):
            with self.subTest(value=value):
                self.assertEqual(self.exchange({**payload, "prompt_format_version": value}),
                                 baseline)
        for value in (2, -1, 0, False, 1.5, "2", "1e0", "one", [], {}):
            with self.subTest(value=value):
                self.exchange({**payload, "prompt_format_version": value}, 400)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("binary", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    NativeDecisions.binary = arguments.binary
    NativeDecisions.checkpoint = arguments.checkpoint
    NativeDecisions.output = arguments.output
    unittest.main(argv=[__file__])
