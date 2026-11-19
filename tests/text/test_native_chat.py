"""Check the native Layer 1 wire contract against frozen checkpoint fixtures."""

import argparse
from contextlib import closing
import http.client
import json
from pathlib import Path
import socket
import struct
import unittest

from test_native_text import checkpoint_pieces


MODEL = "Qwen3.8-27B-FP8"
CHAT = "/v1/chat/completions"
RAW = "/v1/completions"
FIXTURE = Path(__file__).with_name("chat-layer1.json")
CHECKPOINT = Path("/data0/shared") / MODEL
HEADERS = {"Content-Type": "application/json", "Connection": "close"}
GREEDY = {"temperature": 0.0, "seed": 0}


def request(port, route, body):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    encoded = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
    try:
        connection.request("POST", route, encoded, HEADERS)
        return connection, connection.getresponse()
    except BaseException:
        connection.close()
        raise


def json_response(port, route, body, status=200):
    connection, response = request(port, route, body)
    with closing(connection), response:
        assert response.status == status, (response.status, response.read())
        assert response.headers.get_content_type() == "application/json"
        return json.load(response)


def event(response):
    frame = response.readline()
    assert frame.startswith(b"data: "), frame
    assert response.readline() in (b"\n", b"\r\n")
    payload = frame.removeprefix(b"data: ").rstrip(b"\r\n")
    return "[DONE]" if payload == b"[DONE]" else json.loads(payload)


def metadata(packet, object_name):
    assert packet["id"].startswith("chatcmpl-")
    assert packet["object"] == object_name
    assert packet["model"] == MODEL
    assert type(packet["created"]) is int and packet["created"] > 0
    return packet["id"], packet["created"]


def one_choice(packet):
    choice, = packet["choices"]
    assert choice["index"] == 0
    return choice


def usage(packet, prompt_count, completion_count):
    assert packet["usage"] == {
        "prompt_tokens": prompt_count,
        "completion_tokens": completion_count,
        "total_tokens": prompt_count + completion_count,
        "prompt_tokens_details": {"cached_tokens": 0},
    }


def chat_json(port, body, prompt_ids, *, completion_count=None):
    return check_chat_json(json_response(port, CHAT, body), body, prompt_ids, completion_count)


def check_chat_json(packet, body, prompt_ids, completion_count):
    identifier, _ = metadata(packet, "chat.completion")
    choice = one_choice(packet)
    assert set(choice["message"]) == {"role", "content"}
    assert choice["message"]["role"] == "assistant"
    fields = {"index", "message", "finish_reason"}
    ids = []
    if body.get("return_token_ids", False):
        fields.update(("prompt_token_ids", "token_ids"))
        assert choice["prompt_token_ids"] == prompt_ids
        ids = choice["token_ids"]
        if completion_count is None:
            completion_count = len(ids)
        assert len(ids) == completion_count
    assert choice.keys() == fields, choice
    assert completion_count is not None
    usage(packet, len(prompt_ids), completion_count)
    return identifier, ids, choice["message"]["content"], choice["finish_reason"]


def chat_sse(port, body, prompt_ids, *, completion_count=None, frames=None):
    body = body | {"stream": True}
    include_usage = body.get("stream_options", {}).get("include_usage", False)
    diagnostic = body.get("return_token_ids", False)
    connection, response = request(port, CHAT, body)
    with closing(connection), response:
        assert response.status == 200, (response.status, response.read())
        assert response.headers.get_content_type() == "text/event-stream"
        role = event(response)
        if frames is not None:
            frames.append(role)
        identity = metadata(role, "chat.completion.chunk")
        assert one_choice(role) == {
            "index": 0, "delta": {"role": "assistant"}, "finish_reason": None
        }, ("Expected assistant role frame", role)
        ids, text = [], []
        first = True
        while True:
            packet = event(response)
            if frames is not None:
                frames.append(packet)
            assert metadata(packet, "chat.completion.chunk") == identity
            assert "usage" not in packet
            choice = one_choice(packet)
            if diagnostic:
                if first:
                    assert choice["prompt_token_ids"] == prompt_ids
                else:
                    assert "prompt_token_ids" not in choice
                ids.extend(choice["token_ids"])
            else:
                assert "prompt_token_ids" not in choice and "token_ids" not in choice
            first = False
            reason = choice["finish_reason"]
            if reason is not None:
                assert reason in ("stop", "length")
                assert choice["delta"] == {}
                assert choice.get("token_ids", []) == []
                break
            assert set(choice["delta"]) == {"content"}
            assert type(choice["delta"]["content"]) is str
            text.append(choice["delta"]["content"])
        if include_usage:
            packet = event(response)
            if frames is not None:
                frames.append(packet)
            assert metadata(packet, "chat.completion.chunk") == identity
            assert packet["choices"] == []
            if completion_count is None:
                assert diagnostic
                completion_count = len(ids)
            usage(packet, len(prompt_ids), completion_count)
        assert event(response) == "[DONE]"
        if frames is not None:
            frames.append("[DONE]")
        assert response.read(1) == b""
    return identity[0], ids, "".join(text), reason


def reset_after_first_token(port, case):
    body = chat_body(case, 64) | {"stream": True}
    with closing(http.client.HTTPConnection("127.0.0.1", port, timeout=30)) as connection:
        connection.request("POST", CHAT, json.dumps(body), HEADERS)
        with connection.sock.dup() as control, connection.getresponse() as response:
            assert response.status == 200
            assert response.headers.get_content_type() == "text/event-stream"
            role, first = event(response), event(response)
            assert metadata(role, "chat.completion.chunk") == (
                metadata(first, "chat.completion.chunk")
            )
            assert one_choice(role)["delta"] == {"role": "assistant"}
            choice = one_choice(first)
            assert choice["prompt_token_ids"] == case["ids"]
            assert len(choice["token_ids"]) == 1
            assert choice["finish_reason"] is None
            control.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
    return [role, first]


def fixtures():
    cases = json.loads(FIXTURE.read_text(encoding="utf-8"))["cases"]
    for case in cases:
        case["rendered"] = "".join(case["rendered"])
        for message in case["body"]["messages"]:
            for call in message.get("tool_calls", ()):
                function = call["function"]
                if isinstance(function["arguments"], list):
                    function["arguments"] = "".join(function["arguments"])
    return cases


def tokenizer_pieces(checkpoint):
    sha = json.loads(FIXTURE.read_text(encoding="utf-8"))["assets_sha256"]["tokenizer.json"]
    return checkpoint_pieces(checkpoint / "tokenizer.json", sha)


def chat_body(case, count):
    return case["body"] | {
        "model": MODEL, "max_tokens": count, "return_token_ids": True, "ignore_eos": True,
        **GREEDY,
    }


def raw_body(case, count):
    return {
        "model": MODEL, "prompt": case["rendered"], "max_tokens": count,
        "ignore_eos": True, "return_token_ids": True, **GREEDY,
    }


class NativeChatTest(unittest.TestCase):
    port = None
    pieces = None
    require_partial = False
    evidence = None

    @classmethod
    def setUpClass(cls):
        if cls.port is None:
            raise unittest.SkipTest("Run with --port against the native Layer 1 server")

    def test_checkpoint_prompt_ids(self):
        for case in fixtures():
            with self.subTest(case=case["name"]):
                raw = one_choice(json_response(self.port, RAW, raw_body(case, 0)))
                prompt = case["ids"]
                body = chat_body(case, 0)
                packet = json_response(self.port, CHAT, body)
                choice = one_choice(packet)
                if self.evidence is not None:
                    self.evidence["prompts"].append({
                        "case": case["name"], "expected": prompt, "raw": raw["prompt_token_ids"],
                        "chat": choice["prompt_token_ids"],
                    })
                self.assertEqual(raw["prompt_token_ids"], prompt)
                self.assertEqual(choice["prompt_token_ids"], prompt)
                _, ids, content, reason = check_chat_json(packet, body, prompt, 0)
                self.assertEqual((ids, content, reason), ([], "", "length"))

    def test_public_completion_content(self):
        case = fixtures()[0]
        for budget in (1, 2, 7):
            with self.subTest(budget=budget):
                raw = json_response(self.port, RAW, raw_body(case, budget))
                choice = one_choice(raw)
                ids = choice["token_ids"]
                data = b"".join(self.pieces[token] for token in ids)
                expected = data.decode("utf-8", errors="replace")
                body = chat_body(case, budget) | {"return_token_ids": False}
                observed_json = chat_json(self.port, body, case["ids"], completion_count=budget)
                stream = body | {"stream_options": {"include_usage": True}}
                observed_sse = chat_sse(self.port, stream, case["ids"], completion_count=budget)
                if self.evidence is not None:
                    self.evidence["public"].append({
                        "budget": budget, "ids": ids, "bytes_hex": data.hex(),
                        "expected": expected, "json": observed_json[2:], "sse": observed_sse[2:],
                    })
                self.assertEqual(choice["prompt_token_ids"], case["ids"])
                self.assertEqual((len(ids), choice["finish_reason"]), (budget, "length"))
                self.assertEqual(choice["text"], expected)
                if self.require_partial and budget in (1, 2):
                    self.assertEqual(data, b"\xf0\x9f"[:budget])
                    self.assertEqual(expected, "\ufffd")
                self.assertTrue(expected)
                self.assertEqual(observed_json[1:], ([], expected, "length"))
                self.assertEqual(observed_sse[1:], ([], expected, "length"))

    def test_json_sse_raw_and_stop(self):
        case = fixtures()[0]
        body = chat_body(case, 7)
        ids = case["ids"]
        json_id, tokens, text, reason = chat_json(self.port, body, ids)
        sse_id, *stream = chat_sse(
            self.port, body | {"stream_options": {"include_usage": True}}, ids
        )
        self.assertEqual((tokens, text, reason), tuple(stream))
        self.assertNotEqual(json_id, sse_id)
        self.assertEqual((len(tokens), reason), (7, "length"))
        raw = json_response(self.port, RAW, raw_body(case, 7))
        raw_choice = one_choice(raw)
        self.assertEqual(raw_choice["prompt_token_ids"], ids)
        self.assertEqual(raw_choice["token_ids"], tokens)
        self.assertEqual(raw_choice["text"], text)
        usage(raw, len(ids), len(tokens))
        stopped = body | {"ignore_eos": False, "stop_token_ids": [tokens[0]]}
        self.assertEqual(chat_json(self.port, stopped, ids)[1:], ([tokens[0]], "", "stop"))
        stopped_stream = stopped | {"stream_options": {"include_usage": True}}
        self.assertEqual(chat_sse(self.port, stopped_stream, ids)[1:], ([tokens[0]], "", "stop"))
        text_stop = body | {"stop": [text[0]]}
        _, text_stop_ids, stopped_text, stopped_reason = chat_json(self.port, text_stop, ids)
        self.assertEqual((stopped_text, stopped_reason), ("", "stop"))
        self.assertTrue(text_stop_ids)
        self.assertEqual(text_stop_ids, tokens[:len(text_stop_ids)])
        self.assertEqual(chat_sse(self.port, text_stop | {
            "stream_options": {"include_usage": True}
        }, ids)[1:], (text_stop_ids, stopped_text, stopped_reason))
        empty = chat_body(case, 0)
        del empty["max_tokens"]
        empty.update(max_completion_tokens=0, stream_options={"include_usage": False})
        self.assertEqual(chat_sse(self.port, empty, ids)[1:], ([], "", "length"))
        no_diagnostics = body | {"return_token_ids": False}
        self.assertEqual(chat_sse(self.port, no_diagnostics, ids)[1:], ([], text, reason))
        with_usage = no_diagnostics | {"stream_options": {"include_usage": True}}
        self.assertEqual(
            chat_sse(self.port, with_usage, ids, completion_count=len(tokens))[1:],
            ([], text, reason),
        )

    def test_disconnect_and_reuse(self):
        case = fixtures()[0]
        body = chat_body(case, 7)
        first = chat_json(self.port, body, case["ids"])
        role, token = reset_after_first_token(self.port, case)
        self.assertEqual(one_choice(token)["token_ids"], first[1][:1])
        replay = chat_json(self.port, body, case["ids"])
        self.assertNotEqual(replay[0], role["id"])
        self.assertEqual(replay[1:], first[1:])

    def test_sampling_json_sse_and_raw(self):
        case = fixtures()[0]
        body = chat_body(case, 5)
        options = {"temperature": 1.25, "top_p": 0.87, "top_k": 13, "seed": -(1 << 63)}
        request = body | options
        _, ids, text, reason = chat_json(self.port, request, case["ids"], completion_count=5)
        self.assertEqual(reason, "length")
        _, streamed_ids, streamed_text, stream_reason = chat_sse(
            self.port, request | {"stream_options": {"include_usage": True}},
            case["ids"], completion_count=5)
        self.assertEqual((streamed_ids, streamed_text, stream_reason), (ids, text, reason))
        raw = one_choice(json_response(self.port, RAW, raw_body(case, 5) | options))
        self.assertEqual((raw["token_ids"], raw["text"], raw["finish_reason"]),
                         (ids, text, reason))

    def test_request_errors(self):
        user = {"role": "user", "content": "question"}
        valid = {"model": MODEL, "messages": [user], "max_tokens": 0}
        cases = [
            (b'{"messages":', "Invalid JSON"),
            (b'{"messages":[{"role":"user","content":"\\ud800"}]}', "Invalid JSON"),
            (b'{"messages":[{"role":"user","content":"\xff"}]}', "Invalid JSON"),
            (valid | {"model": MODEL.lower()}, "Expected model"),
            (valid | {"messages": []}, "No messages"),
            (valid | {"messages": [user | {"role": "User"}, user]}, "Unexpected message role"),
            (valid | {"messages": [user | {"role": "developer"}, user]}, "Unexpected message"),
            (valid | {"messages": [user, {"role": "system", "content": "late"}]},
             "System message must be at the beginning"),
            (valid | {"messages": [{"role": "assistant", "content": "answer"}]}, "No user query"),
            (valid | {"messages": [user | {"content": {"text": "object"}}]}, "content type"),
            (valid | {"messages": [user | {"content": [{"type": "Text", "text": "x"}]}]},
             "Unexpected item type"),
            (valid | {"messages": [user | {"content": [{"type": "image", "image": "x"}]}]},
             "Media payloads"),
            (valid | {"messages": [{"role": "system", "content": [{"type": "video"}]}, user]},
             "System message cannot contain videos"),
            (valid | {"messages": [user | {"content": "<tool_response>x</tool_response>"}]},
             "No user query"),
            (valid | {"messages": [user, {"role": "assistant", "tool_calls": [
                {"function": {"name": "lookup", "arguments": "[0]"}}
            ]}]}, "arguments must be an object"),
            (valid | {"max_tokens": 262144}, "262144-token context"),
            (valid | {"reasoning_effort": "LOW"}, "Unexpected reasoning effort"),
            (valid | {"tool_choice": "AUTO"}, "Unsupported tool_choice"),
            (valid | {"temperature": -1}, "finite and nonnegative"),
            (valid | {"top_p": 1.1}, "top_p must be in"),
            (valid | {"top_k": -2}, "top_k must be -1"),
            (valid | {"seed": 1 << 63}, "signed 64-bit integer"),
            (valid | {"n": 2}, "n=1"),
            (valid | {"stop": [1]}, "Expected a nonempty stop string"),
            (valid | {"stop": [""]}, "Expected a nonempty stop string"),
            (valid | {"stream_options": {"include_usage": True}}, "for a stream"),
            (valid | {"stop_token_ids": [248320]}, "outside the vocabulary"),
        ]
        for body, message in cases:
            with self.subTest(body=body):
                packet = json_response(self.port, CHAT, body, status=400)
                self.assertIn(message, packet["error"]["message"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    options, unittest_args = parser.parse_known_args()
    NativeChatTest.port = options.port
    NativeChatTest.pieces = tokenizer_pieces(options.checkpoint)
    unittest.main(argv=[__file__, *unittest_args])
