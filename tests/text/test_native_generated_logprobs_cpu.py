"""Check native score metadata through the HTTP worker with scripted decoder scores."""

from __future__ import annotations

import unittest
from contextlib import contextmanager
from dataclasses import dataclass
from http import HTTPStatus
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

from test_native_chat import tokenizer_pieces
from test_native_generate_cpu import ABORT_REQUEST, ABORTED, GENERATE, batch_frame, batch_frames
from test_sampling_routes_cpu import (
    checked_body,
    post_response,
    receipts,
    run_cpu_suite,
    server,
    sse_frames,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Mapping, Sequence
    from typing import Protocol

    class ServerPaths(Protocol):
        binary: Path
        checkpoint: Path
        output: Path


@contextmanager
def scripted_server(
    test: type[ServerPaths],
    name: str,
    scripts: Mapping[int, Sequence[int]],
    log: Path,
    *flags: str,
    sampling: bool = True,
    **environment: str,
) -> Generator[int, None, None]:
    script = test.output / name
    script.write_text(
        "".join(f"{seed} {' '.join(map(str, ids))}\n" for seed, ids in scripts.items()),
        encoding="ascii",
    )
    with (
        patch.dict("os.environ", SGLANG_TEST_CHAT_OUTPUT_IDS=str(script), **environment),
        server(test.binary, test.checkpoint, log, *flags, sampling=sampling) as port,
    ):
        yield port


@dataclass(frozen=True)
class ReplyChecks:
    status: int = HTTPStatus.OK
    native_eof: str = "Stream ended before DONE"
    read_error: bool = False


def checked_exchange(
    test: unittest.TestCase,
    route: str,
    payload: dict[str, object],
    port: int,
    checks: ReplyChecks,
) -> dict[str, object] | list[dict[str, object]]:
    with post_response(port, route, payload) as response:
        detail: bytes | tuple[str, dict[str, object]] = (route, payload)
        if checks.read_error and response.status != checks.status:
            detail = response.read()
        test.assertEqual(response.status, checks.status, detail)
        frames = None
        if checks.status == HTTPStatus.OK and payload.get("stream"):
            frames = (
                batch_frames(response, checks.native_eof)
                if route == GENERATE
                else sse_frames(response)
            )
        return checked_body(test, response, frames)


def assert_output_logprobs(
    test: unittest.TestCase,
    frame: dict[str, object],
    token_text: Callable[[int], str | None],
    top: int,
    *,
    alternative_text: str | None = None,
) -> None:
    meta = frame["meta_info"]
    ids = frame["output_ids"]
    selected = [[-0.25 * ordinal, token, token_text(token)] for ordinal, token in enumerate(ids, 1)]
    test.assertEqual(meta["output_token_logprobs_length"], len(ids))
    test.assertEqual(meta["output_token_logprobs"], selected)
    if top == 0:
        test.assertNotIn("output_top_logprobs", meta)
        return
    expected = [
        [row] + [[row[0] - rank, rank, alternative_text] for rank in range(1, top)]
        for row in selected
    ]
    test.assertEqual(meta["output_top_logprobs"], expected)


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
        scripts = {217: [cls.hello, *cls.snow, cls.stop_id], -41: [32, 33, 34]}
        cls.gate = cls.output / "generated-logprobs-gate"
        Path(f"{cls.gate}.0").touch()
        cls.log = cls.output / "generated-logprobs.log"
        flags = ("--max-active", "2", "--mtp-steps", "0")
        with_script = scripted_server(
            cls,
            "generated-logprobs.ids",
            scripts,
            cls.log,
            *flags,
            SGLANG_TEST_BATCH_STEP_GATE=str(cls.gate),
            SGLANG_TEST_SCRIPTED_MTP="1",
        )
        cls.port = cls.enterClassContext(with_script)

    def exchange(self, payload, expected=200, *, port=None):
        checks = ReplyChecks(status=expected, read_error=True)
        port = self.port if port is None else port
        return checked_exchange(self, GENERATE, payload, port, checks)

    def scores(self, frame, top, text=False, prompt=(), start=0):
        meta = frame["meta_info"]
        ids = frame["output_ids"]
        self.assertEqual(meta["completion_tokens"], len(ids))
        if top is None:
            self.assertFalse(any("logprob" in key for key in meta))
            return

        def token_text(token: int) -> str | None:
            return self.pieces[token].decode(errors="replace") if text else None

        prompt_scores = [
            [None if index == start else -1.5 * index, token, token_text(token)]
            for index, token in enumerate(prompt)
            if index >= start
        ]
        self.assertEqual(meta["input_token_logprobs"], prompt_scores)
        assert_output_logprobs(self, frame, token_text, top)
        if top == 0:
            self.assertNotIn("input_top_logprobs", meta)
            return
        prompt_top = [
            [-0.125 * rank, 31 + rank, token_text(31 + rank)] for rank in range(1, top + 1)
        ]
        self.assertEqual(meta["input_top_logprobs"],
                         [None] + [prompt_top] * (len(prompt_scores) - 1)
                         if prompt_scores else [])

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

    def test_prompt_suffix_zero_budget_and_independent_text(self):
        ids = [self.hello, *self.snow, self.stop_id]
        payload = prompts(0) | {
            "input_ids": ids, "return_logprob": True, "return_text_in_logprobs": True,
            "rid": "prompt-only",
        }
        for start, top in [(0, 0), (1, 2), (4, 1), (5, 2)]:
            with self.subTest(start=start, top=top):
                body = payload | {"logprob_start_len": start, "top_logprobs_num": top}
                final = self.exchange(body)
                self.assertEqual(final["output_ids"], [])
                self.assertEqual(final["meta_info"]["finish_reason"],
                                 {"type": "length", "length": 0})
                self.scores(final, top, text=True, prompt=ids, start=start)
                self.assertEqual(self.exchange(body | {"stream": True}), [final])

    def test_prompt_batch_and_parallel_cumulative_scores(self):
        ids = [[32, 33, 34], [34, 35, 36], [36, 37, 38]]
        batch = prompts(2) | {
            "input_ids": ids, "return_logprob": [True, True, False],
            "logprob_start_len": [1, 0, "ignored"], "top_logprobs_num": [0, 2, 20],
        }
        parallel = prompts(2, n=2) | {
            "input_ids": None, "text": ["hello", "helloZ"], "return_logprob": True,
            "logprob_start_len": 0, "top_logprobs_num": 2,
        }
        for body, expected in [
            (batch, [(0, ids[0], 1), (2, ids[1], 0), (None, (), 0)]),
            (parallel, [(2, [self.hello], 0)] * 2 + [(2, [self.hello, self.stop_id], 0)] * 2),
        ]:
            with self.subTest(body=body):
                plain = self.exchange(body)
                self.assertEqual(len(plain), len(expected))
                for frame, (top, prompt, start) in zip(plain, expected):
                    self.scores(frame, top, prompt=prompt, start=start)
                    frame["meta_info"].pop("id")
                finals = {}
                stream_ids = {}
                for frame in self.exchange(body | {"stream": True}):
                    index = frame.pop("index")
                    top, prompt, start = expected[index]
                    self.scores(frame, top, prompt=prompt, start=start)
                    rid = frame["meta_info"].pop("id")
                    self.assertEqual(rid, stream_ids.setdefault(index, rid))
                    self.assertEqual(frame["output_ids"],
                                     plain[index]["output_ids"][:len(frame["output_ids"])])
                    if frame["meta_info"]["finish_reason"]:
                        self.assertNotIn(index, finals)
                        finals[index] = frame
                self.assertEqual(finals, dict(enumerate(plain)))
                self.assertEqual(len(set(stream_ids.values())), len(expected))

    def test_prompt_start_opt_out(self):
        payload = prompts(2) | {"rid": "opt-out-offset"}
        plain = self.exchange(payload)
        for start in (0, 3, -12, True, "ignored", None):
            with self.subTest(start=start):
                self.assertEqual(self.exchange(payload | {"return_logprob": False,
                                                          "logprob_start_len": start}), plain)

    def test_abort_and_id_reuse(self):
        payload = prompts(8, sampling_seed=-41) | {
            "rid": "scored-abort", "return_logprob": True, "top_logprobs_num": 2, "stream": True,
            "input_ids": [33, 34, 35], "logprob_start_len": 1,
        }
        with post_response(self.port, GENERATE, payload) as response:
            self.assertEqual(response.status, 200)
            first = batch_frame(response)
            self.assertEqual(first["output_ids"], [32])
            self.scores(first, 2, prompt=payload["input_ids"], start=1)
            with post_response(self.port, ABORT_REQUEST, {"rid": "scored-abort"}) as abort_reply:
                self.assertEqual(abort_reply.status, 200)
                self.assertEqual(abort_reply.read(), b"")
            final = batch_frame(response)
            self.assertEqual(final["meta_info"]["finish_reason"], ABORTED)
            self.assertEqual(final["output_ids"], [32])
            self.scores(final, 2, prompt=payload["input_ids"], start=1)
            self.assertEqual(batch_frame(response), "[DONE]")
            self.assertEqual(response.read(), b"")
        reused = self.exchange(payload | {"sampling_params": prompts(1)["sampling_params"],
                                          "stream": False})
        self.assertEqual(reused["output_ids"], [self.hello])
        self.scores(reused, 2, prompt=payload["input_ids"], start=1)

    def check_scored_stream(self, frames, ids, lengths, prompt, *, index=None):
        self.assertEqual([len(frame["output_ids"]) for frame in frames], lengths)
        for frame, length in zip(frames, lengths):
            if index is None:
                self.assertNotIn("index", frame)
            else:
                self.assertEqual(frame["index"], index)
            self.assertEqual(frame["output_ids"], ids[:length])
            self.assertEqual(frame["meta_info"]["finish_reason"],
                             None if length < len(ids) else {"type": "length", "length": length})
            self.scores(frame, 2, prompt=prompt)

    def test_implicit_mtp_scores_scalar_and_mixed_batch(self):
        log = self.output / "generated-logprobs-default.log"
        prompt = [32, 33]
        ids = [self.hello, *self.snow, self.stop_id]
        payload = prompts(5) | {"input_ids": prompt, "return_logprob": True,
                                "top_logprobs_num": 2, "logprob_start_len": 0}
        receipt = b"cpu scripted mtp tokens=4 terminal=1\n"
        with server(self.binary, self.checkpoint, log, "--max-active", "2") as port:
            final = self.exchange(payload, port=port)
            self.assertEqual(final["output_ids"], ids)
            self.scores(final, 2, prompt=prompt)
            frames = self.exchange(payload | {"stream": True}, port=port)
            self.check_scored_stream(frames, ids, [1, 5], prompt)
            self.assertEqual(log.read_bytes().count(receipt), 2)
            plain = self.exchange(payload | {"return_logprob": False}, port=port)
            self.assertEqual(plain["output_ids"], ids)
            self.scores(plain, None)
            self.assertEqual(log.read_bytes().count(receipt), 3)
            offset = log.stat().st_size
            batch = payload | {
                "input_ids": [prompt, prompt], "return_logprob": [True, False],
                "sampling_params": [prompts(length)["sampling_params"] for length in (5, 1)],
            }
            mixed = self.exchange(batch, port=port)
            self.assertEqual([result["output_ids"] for result in mixed], [ids, ids[:1]])
            self.scores(mixed[0], 2, prompt=prompt)
            self.scores(mixed[1], None)
            grouped = {0: [], 1: []}
            for frame in self.exchange(batch | {"stream": True}, port=port):
                grouped[frame["index"]].append(frame)
            self.check_scored_stream(grouped[0], ids, [1, 5], prompt, index=0)
            short, = grouped[1]
            self.assertEqual(short["output_ids"], ids[:1])
            self.assertEqual(short["meta_info"]["finish_reason"], {"type": "length", "length": 1})
            self.scores(short, None)
            self.assertEqual(len({frame["meta_info"]["id"] for frame in grouped[0]}), 1)
            self.assertNotEqual(grouped[0][0]["meta_info"]["id"], short["meta_info"]["id"])
            log_tail = log.read_bytes()[offset:]
            self.assertEqual(log_tail.count(b"cpu scripted mtp tokens="), 2)
            self.assertEqual(log_tail.count(receipt), 2)

    def test_validation_before_any_admission(self):
        start = len(receipts(self.log))
        scalar = prompts(1) | {"return_logprob": True, "stream": True}
        for fields, error in [
            ({"return_logprob": "yes"}, "return_logprob"),
            ({"return_logprob": [True]}, "return_logprob"),
            ({"return_text_in_logprobs": [True]}, "return_text_in_logprobs"),
            *[({"logprob_start_len": value}, "logprob_start_len")
              for value in (-2, 2, True, 0.5, "0")],
            ({"token_ids_logprob": []}, "token_ids_logprob"),
            ({"top_logprobs_num": -1}, "top_logprobs_num"),
            ({"top_logprobs_num": 21}, "top_logprobs_num"),
            ({"top_logprobs_num": 1.5}, "top_logprobs_num"),
            ({"sampling_params": {"max_new_tokens": 1}}, "greedy"),
            ({"input_ids": [[32], [33]], "top_logprobs_num": [20, "bad"]}, "top_logprobs_num"),
            ({"input_ids": [[32], [33]], "logprob_start_len": [-1]}, "logprob_start_len"),
            ({"input_ids": [[32], [33]], "logprob_start_len": [0, 2]}, "logprob_start_len"),
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
            error = self.exchange(scalar | {"sampling_params": {"max_new_tokens": 5}},
                                  400, port=port)
            self.assertIn("greedy", error["error"]["message"])
            self.assertEqual(receipts(mtp_log), [])
            prompt = [32, 33]
            ids = [self.hello, *self.snow, self.stop_id]
            scored = prompts(5) | {"input_ids": prompt, "return_logprob": True,
                                   "top_logprobs_num": 2, "logprob_start_len": 0}
            result = self.exchange(scored, port=port)
            self.assertEqual(result["output_ids"], ids)
            self.scores(result, 2, prompt=prompt)
            frames = self.exchange(scored | {"stream": True}, port=port)
            self.check_scored_stream(frames, ids, [1, 3, 5], prompt)
            self.assertEqual(
                mtp_log.read_bytes().count(b"cpu scripted mtp tokens=2 terminal=0\n"), 2)
            self.assertEqual(
                mtp_log.read_bytes().count(b"cpu scripted mtp tokens=2 terminal=1\n"), 2)


if __name__ == "__main__":
    run_cpu_suite(NativeGeneratedLogprobs, __file__, __doc__, select_tests=True, verbosity=1)
