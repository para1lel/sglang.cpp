"""Check independent token display and score bytes through real HTTP formatters."""

import json
import sys
import unittest
from http import HTTPStatus
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "text"))

from test_native_chat import tokenizer_pieces
from test_native_text import MODEL, stream_options
from test_sampling_routes_cpu import (
    CHAT,
    body,
    checked_body,
    post_response,
    run_cpu_suite,
    server,
    sse_frames,
)

NATIVE = "/generate"
RESPONSES = "/v1/responses"
SAMPLED = (158, 159, 97055, 248077)
RANK_ORDER = (158, 97055, 159, 248077, 5513)
RAW = {158: b"\xe2", 159: b"\xe3", 97055: "雪".encode(), 248077: b"", 5513: "�".encode()}
DISPLAY = {158: "�", 159: "�", 97055: "雪", 248077: "", 5513: "�"}
VISIBLE = "��雪"


def ranked_ids(selected: int) -> tuple[int, ...]:
    return selected, *(token for token in RANK_ORDER if token != selected)


def script_line(key: int, ids: tuple[int, ...]) -> str:
    return " ".join(map(str, (key, *ids))) + "\n"


class LogprobProjection(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        pieces = tokenizer_pieces(cls.checkpoint)
        for token, raw in RAW.items():
            if raw:
                assert pieces[token] == raw
            else:
                assert token not in pieces
        output = cls.output / "projection-output.ids"
        output.write_text(script_line(0, SAMPLED), encoding="ascii")
        ranked = cls.output / "projection-ranked.ids"
        ranked.write_text(
            "".join(script_line(token, ranked_ids(token)) for token in SAMPLED), encoding="ascii"
        )
        cls.enterClassContext(
            patch.dict(
                "os.environ",
                SGLANG_TEST_CHAT_OUTPUT_IDS=str(output),
                SGLANG_TEST_CHAT_TOP_IDS=str(ranked),
            )
        )
        cls.port = cls.enterClassContext(
            server(
                cls.binary,
                cls.checkpoint,
                cls.output / "projection-server.log",
                "--mtp-steps",
                "0",
                sampling=False,
            )
        )

    def exchange(
        self,
        route: str,
        payload: dict[str, object],
        status: HTTPStatus = HTTPStatus.OK,
    ) -> dict[str, object] | list[dict[str, object]]:
        with post_response(self.port, route, payload) as response:
            self.assertEqual(
                response.status, status, response.read() if response.status != status else None
            )
            if status != HTTPStatus.OK or not payload.get("stream"):
                return checked_body(self, response)
            if route != RESPONSES:
                return checked_body(self, response, sse_frames(response))
            self.assertEqual(response.headers.get_content_type(), "text/event-stream")
            frames = []
            while name := response.readline():
                self.assertTrue(name.startswith(b"event: "), name)
                data = response.readline()
                self.assertTrue(data.startswith(b"data: "), data)
                self.assertEqual(response.readline(), b"\n")
                event = json.loads(data[6:])
                self.assertEqual(name[7:].strip().decode("ascii"), event["type"])
                frames.append(event)
            return frames

    def test_chat_raw_bytes_and_all_ranked_ids(self) -> None:
        request = body(CHAT, max_tokens=len(SAMPLED), temperature=0, logprobs=True, top_logprobs=5)
        answer = self.exchange(CHAT, request)["choices"][0]
        self.assertEqual(answer["token_ids"], list(SAMPLED))
        self.assertEqual(answer["message"]["content"], VISIBLE)
        rows = []
        for ordinal, token in enumerate(SAMPLED, 1):
            value = -0.25 * ordinal
            row = {"token": DISPLAY[token], "bytes": list(RAW[token]), "logprob": value}
            rows.append(
                row
                | {
                    "top_logprobs": [
                        {"token": DISPLAY[alt], "bytes": list(RAW[alt]), "logprob": value - rank}
                        for rank, alt in enumerate(ranked_ids(token))
                    ]
                }
            )
        self.assertEqual(answer["logprobs"], {"content": rows})
        frames = self.exchange(CHAT, request | stream_options())
        choices = [choice for frame in frames for choice in frame["choices"]]
        self.assertEqual(choices[0]["delta"], {"role": "assistant"})
        self.assertIsNone(choices[0]["logprobs"])
        self.assertIsNone(choices[-1]["logprobs"])
        self.assertEqual("".join(c["delta"].get("content") or "" for c in choices), VISIBLE)
        self.assertEqual(
            [row for c in choices if c["logprobs"] for row in c["logprobs"]["content"]], rows
        )
        empty = self.exchange(CHAT, request | {"max_tokens": 0, "top_logprobs": 0})
        self.assertEqual(empty["choices"][0]["logprobs"], {"content": []})

    def test_native_display_id_triples_and_cumulative_frames(self) -> None:
        request = {
            "input_ids": [32],
            "return_logprob": True,
            "top_logprobs_num": 5,
            "sampling_params": {"max_new_tokens": len(SAMPLED), "temperature": 0},
        }
        for include_text in (False, True):
            with self.subTest(include_text=include_text):
                request["return_text_in_logprobs"] = include_text
                blocking = self.exchange(NATIVE, request)
                frames = self.exchange(NATIVE, request | {"stream": True})
                selected, alternatives = [], []
                for ordinal, token in enumerate(SAMPLED, 1):
                    value = -0.25 * ordinal
                    selected.append([value, token, DISPLAY[token] if include_text else None])
                    alternatives.append(
                        [
                            [value - rank, alt, DISPLAY[alt] if include_text else None]
                            for rank, alt in enumerate(ranked_ids(token))
                        ]
                    )
                self.assertEqual(blocking["text"], VISIBLE)
                self.assertEqual([len(frame["output_ids"]) for frame in frames], [1, 2, 3, 4])
                for response in [*frames, blocking]:
                    count = len(response["output_ids"])
                    meta = response["meta_info"]
                    self.assertEqual(response["output_ids"], list(SAMPLED[:count]))
                    self.assertEqual(meta["output_token_logprobs_length"], count)
                    self.assertEqual(meta["output_token_logprobs"], selected[:count])
                    self.assertEqual(meta["output_top_logprobs"], alternatives[:count])
                self.assertEqual(frames[-1]["text"], VISIBLE)
        empty_request = request | {"sampling_params": {"max_new_tokens": 0, "temperature": 0}}
        empty = self.exchange(NATIVE, empty_request)
        self.assertEqual(empty["output_ids"], [])
        self.assertEqual(empty["meta_info"]["output_token_logprobs"], [])
        self.assertEqual(empty["meta_info"]["output_top_logprobs"], [])

    def test_responses_display_bytes_dedup_and_stream_guard(self) -> None:
        request = {
            "model": MODEL,
            "input": "Test",
            "reasoning": {"effort": "none"},
            "max_output_tokens": len(SAMPLED) + 2,
            "temperature": 0,
            "store": False,
        }
        scored = request | {"include": ["message.output_text.logprobs"]}
        for width in (0, 3, 4, 5):
            with self.subTest(width=width):
                answer = self.exchange(RESPONSES, scored | {"top_logprobs": width})
                (item,) = answer["output"]
                (part,) = item["content"]
                self.assertEqual(part["text"], VISIBLE)
                self.assertEqual(answer["usage"]["output_tokens"], len(SAMPLED))
                rows = []
                for ordinal, token in enumerate(SAMPLED, 1):
                    value = -0.25 * ordinal
                    distinct = {
                        DISPLAY[alt]: value - rank
                        for rank, alt in enumerate(ranked_ids(token)[:width])
                    }
                    top = [
                        {"token": text, "bytes": list(text.encode()), "logprob": score}
                        for text, score in distinct.items()
                    ]
                    text = DISPLAY[token]
                    rows.append(
                        {
                            "token": text,
                            "bytes": list(text.encode()),
                            "logprob": value,
                            "top_logprobs": top,
                        }
                    )
                self.assertEqual(part["logprobs"], rows)
        empty = self.exchange(RESPONSES, scored | {"max_output_tokens": 2})
        self.assertEqual(empty["output"], [])
        self.assertEqual(empty["usage"]["output_tokens"], 0)
        rejected = self.exchange(
            RESPONSES, scored | {"stream": True}, status=HTTPStatus.BAD_REQUEST
        )
        self.assertIn("logprobs are not supported in streaming mode", rejected["error"]["message"])
        events = self.exchange(RESPONSES, request | {"stream": True})
        self.assertEqual([event["sequence_number"] for event in events], list(range(len(events))))
        deltas = [event for event in events if event["type"] == "response.output_text.delta"]
        self.assertEqual("".join(event["delta"] for event in deltas), VISIBLE)
        self.assertTrue(all(event["logprobs"] == [] for event in deltas))
        self.assertEqual(events[-1]["type"], "response.incomplete")
        (item,) = events[-1]["response"]["output"]
        (part,) = item["content"]
        self.assertEqual(part["text"], VISIBLE)
        self.assertIsNone(part["logprobs"])


if __name__ == "__main__":
    run_cpu_suite(LogprobProjection, __file__, __doc__, select_tests=True)
