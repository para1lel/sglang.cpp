"""Check Qwen chat tokenization through both native HTTP aliases on CPU."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent / "text"))
from test_native_tokenization_cpu import CONTEXT, TOKENIZE, post
from test_sampling_routes_cpu import server


USER = {"role": "user", "content": "你好 🌍 café"}
ASSISTANT = {"role": "assistant", "content": "继续", "reasoning_content": "private"}
NEXT = {"role": "user", "content": "again"}
TOOL_USER = {"role": "user", "content": "Hello, 你好 🌍"}
USER_IDS = [248045, 846, 198, 109266, 10838, 234, 235, 50203, 248046, 198]
OPEN_ASSISTANT = [248045, 74455, 198, 248068, 198]
CLOSED_THINKING = [248045, 74455, 198, 248068, 271, 248069, 271]
XHIGH_IDS = [
    248045, 8678, 198, 24342, 286, 4879, 369, 716, 310, 830, 11553, 13, 5044, 1683,
    15060, 1472, 279, 3274, 11, 9307, 1328, 30800, 11, 2814, 47675, 25605, 11, 321,
    60445, 55404, 11, 27224, 11, 321, 30246, 303, 279, 1534, 4087, 13, 248046, 198,
]
LOW_IDS = [
    248045, 8678, 198, 24342, 286, 4879, 369, 716, 310, 3238, 13, 13262, 678, 7047,
    9522, 321, 10419, 11, 6992, 5774, 310, 279, 16198, 1973, 24366, 24150, 362,
    13, 248046, 198,
]
MEDIUM_IDS = USER_IDS + OPEN_ASSISTANT
ASSISTANT_IDS = OPEN_ASSISTANT + [1929, 198, 248069, 271, 97625, 248046, 198]
NEXT_IDS = [248045, 846, 198, 31685, 248046, 198]
TOOL_HISTORY_IDS = USER_IDS + [
    248045, 74455, 198, 547, 271, 248058, 198, 27, 1628, 28, 20377, 29, 198, 27, 15704,
    28, 8656, 29, 198, 98116, 198, 510, 15704, 29, 198, 510, 1628, 29, 198, 248059,
    248046, 198, 248045, 846, 198, 248066, 198, 100253, 198, 248067, 248046, 198,
] + NEXT_IDS
MEDIUM = {"reasoning_effort": "medium"}


def chat(*messages, **options):
    return {"messages": list(messages), **options}


def history(function, content="ok"):
    return {"role": "assistant", "content": content,
            "tool_calls": [{"function": {"name": "lookup", **function}}]}


def option_cases():
    yield "checkpoint default", chat(USER), XHIGH_IDS + MEDIUM_IDS
    yield "low effort", chat(USER, reasoning_effort="low"), LOW_IDS + MEDIUM_IDS
    yield "kwargs override effort", chat(USER, reasoning_effort="low",
                                        chat_template_kwargs=MEDIUM), MEDIUM_IDS
    yield "nested effort precedence", chat(
        USER, reasoning_effort="xhigh", reasoning={"effort": "medium"},
        thinking={"effort": "low"},
    ), LOW_IDS + MEDIUM_IDS
    yield "kwargs disable thinking", chat(USER, reasoning_effort="xhigh",
                                         chat_template_kwargs={"enable_thinking": False}), (
        USER_IDS + CLOSED_THINKING
    )
    yield "explicit no generation", chat(USER, **MEDIUM, add_generation_prompt=False), USER_IDS
    yield "default assistant rewrite", chat(USER, ASSISTANT, **MEDIUM), (
        USER_IDS + [248045, 846, 198, 97625, 248046, 198] + OPEN_ASSISTANT
    )
    yield "explicit generation keeps assistant", chat(
        USER, ASSISTANT, **MEDIUM, add_generation_prompt=True,
    ), USER_IDS + ASSISTANT_IDS + OPEN_ASSISTANT
    yield "explicit no generation keeps assistant", chat(
        USER, ASSISTANT, **MEDIUM, add_generation_prompt=False, add_special_tokens=True,
    ), USER_IDS + ASSISTANT_IDS
    yield "separately encoded continuation", chat(
        USER, ASSISTANT | {"content": "\n继续  "}, **MEDIUM, continue_final_message=True,
    ), MEDIUM_IDS + [198, 97625, 256]
    yield "nested continuation trims the rendered assistant", chat(
        TOOL_USER,
        {"role": "assistant", "content": "  prefix \n", "reasoning_content": " separate thought "},
        add_generation_prompt=False, chat_template_kwargs={"continue_final_message": True},
    ), XHIGH_IDS + [
        248045, 846, 198, 9419, 11, 220, 109266, 10838, 234, 235, 248046, 198,
        248045, 74455, 198, 248068, 198, 323, 65319, 3272, 198, 248069, 271, 11510,
    ]
    yield "preserved historical thinking", chat(
        USER, ASSISTANT, NEXT, **MEDIUM, add_generation_prompt=False,
    ), USER_IDS + ASSISTANT_IDS + NEXT_IDS
    yield "disabled historical thinking", chat(
        USER, ASSISTANT, NEXT, **MEDIUM, add_generation_prompt=False,
        chat_template_kwargs={"preserve_thinking": False},
    ), USER_IDS + [248045, 74455, 198, 97625, 248046, 198] + NEXT_IDS
    yield "flattened text parts", chat(
        {"role": "system", "content": [{"type": "text", "text": "Be brief."}]},
        USER | {"content": [{"type": "text", "text": "你好 "},
                            {"type": "text", "text": "🌍 café"}]},
        chat_template_kwargs={"enable_thinking": False},
    ), [248045, 8678, 198, 3320, 9522, 13, 248046, 198] + USER_IDS + CLOSED_THINKING
    yield "tool call and result history", chat(
        USER, history({"arguments": '{"city":"北京"}'}), {"role": "tool", "content": "晴"},
        NEXT, **MEDIUM, add_generation_prompt=False,
        chat_template_kwargs={"preserve_thinking": False},
    ), TOOL_HISTORY_IDS


def tool_cases():
    lookup = {"function": {
        "name": "lookup",
        "parameters": {"properties": {"q": {"type": "string"}}, "type": "object"},
        "description": "Café & <query>", "strict": False,
    }}
    ping = {"function": {"name": "ping"}}
    tools = [lookup, ping]
    yield "tools", chat(TOOL_USER, tools=tools), 336, (
        "85181913b32235b6d05e5039668355465e9dbfb6743733e22e57ffe72bf5d8d5"
    )
    yield "named tool", chat(TOOL_USER, tools=tools, tool_choice=ping), 282, (
        "201067a6cf300ce7d5878e510397d5914b5cf2df034e154127eb4fdc7e13c8e3"
    )
    yield "deferred tool", chat(TOOL_USER, tools=[{"defer_loading": True, **lookup}]), 322, (
        "bf598924ac21fd501c43df79fc164de2cc0e06f5da7a994d977e9f4d1e82e6df"
    )
    next_turn = {"role": "user", "content": "Next."}
    encoded_wide = history({"arguments": '{"n":18446744073709551616,"m":-9223372036854775809}'}, "")
    direct_wide = history({"arguments": {"n": 18446744073709551616, "m": -9223372036854775809}}, "")
    yield ("wide integers in encoded tool arguments", chat(TOOL_USER, encoded_wide, next_turn),
           152, "11654f32652b9ca9c86135bb1cc0620a121c7d7c81587933e02b14eb2abf5ec0")
    yield ("wide integers in object tool arguments", chat(TOOL_USER, direct_wide, next_turn),
           148, "8bcd97dbabf79563d049ce75205972674605cf16bfa2a524ca54753a41e232b3")
    yield "underflow in encoded tool arguments", chat(
        TOOL_USER, history({"arguments": '{"n":1e-999}'}, ""), next_turn,
    ), 101, "da115e1477bc2e8f14aeb4c93597e31868d193bc1fa9b46e569ece8b9777f279"
    media = {"role": "user", "content": [
        {"type": "image_url", "image_url": "file:///DO-NOT-READ.png"},
        {"type": "video_url", "video_url": {"url": "https://invalid.example/video"}},
        {"type": "text", "text": "Describe both."},
    ]}
    yield "numbered media", chat(media, media, chat_template_kwargs={"add_vision_id": True}), 95, (
        "76626f7fc5f95de4663b2723f1e3bac4ee208aed48f98ee380944219f5e2ece9"
    )


def error_cases():
    yield "no input", {}, 400, "Bad Request", "Exactly one"
    yield "both inputs", chat(USER, prompt=""), 400, "Bad Request", "Exactly one"
    yield "unknown role", chat(USER | {"role": "unknown"}), 400, "Bad Request", None
    yield "bad generation flag", chat(USER, add_generation_prompt="maybe"), 400, "Bad Request", None
    yield "streaming", chat(USER, stream=True), 501, "NotImplementedError", "streaming"
    yield "schema before streaming", chat({"role": "user"}, stream=True), 400, "Bad Request", None
    yield "empty chat", chat(), 400, "BadRequestError", "Messages cannot be empty."
    yield ("continuation conflict",
           chat(USER, ASSISTANT, continue_final_message=True, add_generation_prompt=False),
           400, "BadRequestError", "continue_final_message needs add_generation_prompt")
    yield ("required tool missing", chat(USER, tool_choice="required"),
           400, "BadRequestError", "Tools cannot be empty")
    yield ("custom template blocked", chat(USER, chat_template_kwargs={"chat_template": "x"}),
           400, "BadRequestError", "Request-supplied chat_template is not")
    yield "bad arguments before assistant rewrite", chat(
        USER, history({"arguments": "broken"}),
    ), 400, "BadRequestError", "function.arguments"
    yield "non-object historical arguments", chat(
        USER, history({"arguments": "[]"}), NEXT,
    ), 400, "BadRequestError", "function.arguments"
    yield ("missing historical arguments", chat(USER, history({}), NEXT),
           400, "BadRequestError", None)
    yield ("null historical arguments", chat(USER, history({"arguments": None}), NEXT),
           400, "BadRequestError", None)
    yield "malformed media before streaming", chat({"role": "user", "content": [
        {"type": "video_url", "video_url": {"url": "file:///unused.mp4", "fps": 0}},
    ]}, stream=True), 400, "Bad Request", None
    yield "incomplete tool reference", chat({
        "role": "user", "content": [{"type": "tool_reference"}],
    }), 400, "Bad Request", None
    invalid_history = history({"arguments": "{}"}, "")
    invalid_history["tool_calls"][0]["id"] = 42
    yield ("numeric historical tool call ID", chat(USER, invalid_history, NEXT),
           400, "Bad Request", None)
    yield "non-numeric temperature", chat(USER, temperature="warm"), 400, "BadRequestError", None
    yield ("completion budget over context", chat(USER, max_completion_tokens=CONTEXT + 1),
           400, "BadRequestError", None)
    yield "JSON schema response format without schema", chat(
        USER, response_format={"type": "json_schema", "json_schema": {"name": "x"}},
    ), 400, "BadRequestError", None
    yield ("sampling mask without metadata", chat(USER, return_sampling_mask=True),
           400, "BadRequestError", None)
    yield ("template tokenize argument collision",
           chat(USER, chat_template_kwargs={"tokenize": False}),
           400, "BadRequestError", None)
    yield ("template messages argument collision",
           chat(USER, chat_template_kwargs={"messages": []}),
           400, "BadRequestError", None)


class ChatTokenization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.log = cls.output / "chat-tokenization.log"
        cls.port = cls.enterClassContext(server(cls.binary, cls.checkpoint, cls.log))

    def tearDown(self):
        self.assertNotIn("cpu admit", self.log.read_text())

    def test_checkpoint_options(self):
        for route in TOKENIZE:
            for name, request, ids in option_cases():
                with self.subTest(route=route, case=name):
                    self.assertEqual(post(self.port, route, request), (
                        200, {"tokens": ids, "count": len(ids), "max_model_len": CONTEXT}
                    ))

    def test_tools_and_media(self):
        for route in TOKENIZE:
            for name, request, count, digest in tool_cases():
                with self.subTest(route=route, case=name):
                    status, body = post(self.port, route, request)
                    self.assertEqual(status, 200, body)
                    tokens = body["tokens"]
                    self.assertEqual(body, {
                        "tokens": tokens, "count": count, "max_model_len": CONTEXT,
                    })
                    self.assertEqual(len(tokens), count)
                    self.assertEqual(
                        sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest(),
                        digest, tokens,
                    )

    def test_direct_tool_arguments_preserve_underflow_sign(self):
        request = (
            b'{"messages":[{"role":"user","content":"Find a result."},'
            b'{"role":"assistant","content":"","tool_calls":['
            b'{"function":{"name":"lookup","arguments":{"n":%s}}}]},'
            b'{"role":"user","content":"Next."}]}'
        )
        cases = (
            (b"1e-999", 98, "ef83ffa74c4efc19588c2920f07a0766e533117e29a3366bdbd4a119a6adcc5b"),
            (b"-1e-999", 99, "51d754c5d26337b80bec49e244d77aa784033be55fe8f65ccbf9327e00cd30ae"),
        )
        for route in TOKENIZE:
            for literal, count, digest in cases:
                with self.subTest(route=route, literal=literal):
                    status, body = post(self.port, route, request % literal)
                    self.assertEqual(status, 200, body)
                    tokens = body["tokens"]
                    self.assertEqual((len(tokens), body["count"], body["max_model_len"]),
                                     (count, count, CONTEXT))
                    self.assertEqual(
                        sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest(),
                        digest, tokens,
                    )

    def test_validation_and_raw_recovery(self):
        for route in TOKENIZE:
            for name, request, expected_status, category, fragment in error_cases():
                with self.subTest(route=route, case=name):
                    status, body = post(self.port, route, request)
                    self.assertEqual(status, expected_status, body)
                    message = body.get("message")
                    self.assertIsInstance(message, str)
                    self.assertTrue(message)
                    if fragment is not None:
                        self.assertIn(fragment, message)
                    self.assertEqual(body, {"object": "error", "type": category,
                                            "code": status, "message": message, "param": None})
            with self.subTest(route=route, case="raw batch after validation errors"):
                self.assertEqual(post(self.port, route, {
                    "prompt": ["", "<|im_end|>", "x"], "messages": None,
                }), (200, {"tokens": [[], [248046], [87]], "count": [0, 1, 1],
                           "max_model_len": CONTEXT}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ChatTokenization.binary = args.binary.resolve()
    ChatTokenization.checkpoint = args.checkpoint.resolve()
    ChatTokenization.output = args.output.resolve()
    ChatTokenization.output.mkdir(parents=True, exist_ok=True)
    unittest.main(argv=[__file__], verbosity=2)
