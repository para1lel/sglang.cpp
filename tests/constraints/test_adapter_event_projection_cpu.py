"""Check dialect-specific event order, cancellation, abort and media dispatch."""

import base64
import json
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from test_messages import CHAT, MODEL, ROUTE, frames, visual_case
from test_sampling_routes_cpu import post_response, run_cpu_suite, server, wire

RESPONSES = "/v1/responses"
OLLAMA = "/api/generate"
RESPONSE_REQUEST = {
    "model": MODEL,
    "input": "Say AB.",
    "max_output_tokens": 5,
    "reasoning": {"effort": "none"},
    "temperature": 0,
}
OLLAMA_REQUEST = {"model": MODEL, "prompt": "Say AB.", "options": {"num_predict": 3}}


class AdapterEvents(unittest.TestCase):
    def start(self, *, gated: bool = False) -> int:
        directory = Path(self.enterContext(TemporaryDirectory(dir=self.output)))
        script = directory / "output.ids"
        script.write_text("0 32 33 34\n")
        environment = {"SGLANG_TEST_CHAT_OUTPUT_IDS": str(script)}
        if gated:
            self.gate = directory / "step.0"
            environment.update(
                SGLANG_TEST_BATCH_STEP_GATE=str(self.gate.with_suffix("")),
                SGLANG_TEST_BATCH_STEP_GATE_UNSEEDED="1",
            )
        self.enterContext(patch.dict("os.environ", environment))
        return self.enterContext(
            server(
                self.binary,
                self.checkpoint,
                self.output / f"{self._testMethodName}.log",
                "--enable-response-store",
                sampling=False,
            )
        )

    def test_responses_order_and_finish(self) -> None:
        port = self.start()
        payload = RESPONSE_REQUEST | {"stream": True}
        with post_response(port, RESPONSES, payload) as reply:
            self.assertEqual(reply.status, 200)
            self.assertEqual(reply.headers.get_content_type(), "text/event-stream")
            events = list(frames(reply))
        names = [event["type"] for event in events]
        self.assertEqual(
            names[:4],
            [
                "response.created",
                "response.in_progress",
                "response.output_item.added",
                "response.content_part.added",
            ],
        )
        self.assertEqual(names[4:-4], ["response.output_text.delta"] * 3)
        self.assertEqual(
            names[-4:],
            [
                "response.output_text.done",
                "response.content_part.done",
                "response.output_item.done",
                "response.incomplete",
            ],
        )
        self.assertEqual([event["sequence_number"] for event in events], list(range(len(events))))
        self.assertEqual("".join(event["delta"] for event in events[4:-4]), "ABC")
        self.assertEqual(events[-4]["text"], "ABC")
        final = events[-1]["response"]
        self.assertEqual(final["incomplete_details"], {"reason": "max_output_tokens"})
        self.assertEqual(final["usage"]["output_tokens"], 3)

    def test_responses_background_cancel(self) -> None:
        port = self.start(gated=True)
        payload = RESPONSE_REQUEST | {"background": True, "store": True}
        with post_response(port, RESPONSES, payload) as reply:
            self.assertEqual(reply.status, 200)
            queued = json.load(reply)
        self.assertEqual(queued["status"], "queued")
        path = f"{RESPONSES}/{queued['id']}"
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            status, _, body = wire(port, "GET", path)
            self.assertEqual(status, 200, body)
            if json.loads(body)["status"] == "in_progress":
                break
            time.sleep(0.01)
        else:
            self.fail("Background response did not enter in_progress")
        status, _, body = wire(port, "POST", path + "/cancel", b"{}")
        self.assertEqual(status, 200, body)
        self.assertEqual(json.loads(body)["status"], "cancelled")
        self.gate.touch()
        status, _, snapshot = wire(port, "GET", path)
        self.assertEqual(status, 200, snapshot)
        self.assertEqual(json.loads(snapshot), json.loads(body))

    def test_ollama_abort_keeps_partial_ndjson(self) -> None:
        port = self.start(gated=True)
        self.gate.touch()
        with post_response(port, OLLAMA, OLLAMA_REQUEST) as reply:
            self.assertEqual(reply.status, 200)
            self.assertEqual(reply.headers.get_content_type(), "application/x-ndjson")
            first = json.loads(reply.readline())
            self.assertEqual(
                (first["response"], first["done"], first["done_reason"]), ("A", False, None)
            )
            status, _, body = wire(
                port,
                "POST",
                "/abort_request",
                b'{"abort_all":true}',
                {"Content-Type": "application/json"},
            )
            self.assertEqual((status, body), (200, b""))
            self.assertEqual(reply.readline(), b"")

    def test_media_routes_dispatch_and_validate(self) -> None:
        port = self.start()
        messages, chat = visual_case()
        url = chat["messages"][-1]["content"][0]["image_url"]["url"]
        responses = RESPONSE_REQUEST | {
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_image", "image_url": url},
                        {"type": "input_text", "text": "Name the color."},
                    ],
                }
            ]
        }
        rgb = base64.b64encode(b"\xff\x00\x00" * (32 * 32)).decode()
        image = chat["messages"][-1]["content"][0]["image_url"]
        image.update(url="data:application/x-sglang-rgb8;base64," + rgb, height=32, width=32)
        for route, payload in ((ROUTE, messages), (CHAT, chat), (RESPONSES, responses)):
            with self.subTest(route=route), post_response(port, route, payload) as reply:
                status, body = reply.status, json.load(reply)
                self.assertEqual(reply.headers.get_content_type(), "application/json")
                if route == CHAT:
                    self.assertEqual(status, 500, body)
                    self.assertEqual(body["error"]["message"], "Native request failed")
                else:
                    self.assertEqual(status, 400, body)
                    self.assertEqual(
                        body["error"]["message"],
                        "Expected an enabled inline media data URL with matching MIME",
                    )
        log = self.output / f"{self._testMethodName}.log"
        self.assertIn("The CPU decoder does not accept media", log.read_text())
        responses["input"][0]["content"][0]["image_url"] = False
        with post_response(port, RESPONSES, responses) as reply:
            self.assertEqual(reply.status, 400)
            self.assertEqual(json.load(reply)["error"]["param"], "input")


if __name__ == "__main__":
    run_cpu_suite(AdapterEvents, __file__, __doc__, select_tests=True)
