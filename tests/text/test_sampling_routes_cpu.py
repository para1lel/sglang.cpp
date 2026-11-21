"""Exercise public text and chat sampling routes with the CPU decoder double."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, contextmanager
import http.client
import json
import os
from pathlib import Path
import socket
import subprocess
from tempfile import TemporaryDirectory
from threading import Barrier
import time
import unittest

from test_native_chat import request
from test_native_text import MODEL, event, legacy


TEXT = "/v1/completions"
CHAT = "/v1/chat/completions"
MODELS = "/v1/models"
SCRIPT = [32, 248046, 33, 248044, 34, 35, 36]
OPTIONS_PREFIX = "cpu sampling {"
CHAT_CONTEXT_RESERVE = 2
CHAT_PROMPT_HEADROOM = 6
CHAT_TEST_PROMPT_TOKENS = 53


def body(route, **options):
    prompt = {"prompt": "Test"} if route == TEXT else {
        "messages": [{"role": "user", "content": "Test"}],
    }
    return {"model": MODEL, "return_token_ids": True, **prompt, **options}


def send(port, route, payload):
    connection, response = request(port, route, payload)
    with closing(connection), response:
        if response.headers.get_content_type() == "text/event-stream":
            events = []
            for packet in iter(lambda: event(response), "[DONE]"):
                events.append(packet)
            assert response.read() == b""
            return response.status, events
        return response.status, json.load(response)


def receipts(log):
    return [json.loads(line.removeprefix("cpu sampling ")) for line in log.read_text().splitlines()
            if line.startswith(OPTIONS_PREFIX)]


@contextmanager
def server(binary, checkpoint, log, *flags, sampling=True):
    with closing(socket.socket()) as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    command = [str(binary), "--checkpoint", str(checkpoint), "--port", str(port), *flags]
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    if sampling:
        environment["SGLANG_TEST_SAMPLING_RECEIPTS"] = "1"
    else:
        environment.pop("SGLANG_TEST_SAMPLING_RECEIPTS", None)
    with log.open("wb") as output:
        process = subprocess.Popen(command, env=environment, stdout=output, stderr=output)
        try:
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise AssertionError(f"server exited {process.returncode}: {log.read_text()}")
                try:
                    with closing(http.client.HTTPConnection("127.0.0.1", port, timeout=1)) as conn:
                        conn.request("GET", "/health")
                        with conn.getresponse() as reply:
                            if reply.status == 200:
                                reply.read()
                                break
                except OSError:
                    time.sleep(0.02)
            else:
                raise AssertionError(f"server did not become ready: {log.read_text()}")
            yield port
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)


class Routes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((cls.checkpoint / "generation_config.json").read_text())
        model = json.loads((cls.checkpoint / "config.json").read_text())["text_config"]
        cls.context = model["max_position_embeddings"]
        cls.vocab = model["vocab_size"]
        cls.eos = sorted(cls.config["eos_token_id"])
        cls.log = cls.output / "serial.log"
        cls.port = cls.enterClassContext(server(cls.binary, cls.checkpoint, cls.log,
                                               "--max-active", "1"))

    def exchange(self, route, payload, *, status=200, port=None, log=None):
        port = self.port if port is None else port
        log = self.log if log is None else log
        before = len(receipts(log))
        actual, value = send(port, route, payload)
        seen = receipts(log)[before:]
        self.assertEqual(actual, status, (route, payload, value))
        if status != 200:
            self.assertEqual(seen, [])
            self.assertIn("error", value)
            return
        self.assertEqual(len(seen), 1, (route, payload, seen))
        return value, seen[0]

    def selection(self, seen, *, temperature=1, top_k=-1, top_p=1, min_p=0, sampled=True):
        if top_k == -1:
            self.assertIn(seen["top_k"], (-1, self.vocab))
        else:
            self.assertEqual(seen["top_k"], top_k)
        self.assertAlmostEqual(seen["temperature"], temperature, places=6)
        self.assertAlmostEqual(seen["top_p"], top_p, places=6)
        self.assertAlmostEqual(seen["min_p"], min_p, places=6)
        self.assertIs(seen["sampled"], sampled)

    def completion(self, packet, seen, count, reason):
        choice, = packet["choices"]
        self.assertEqual(choice["finish_reason"], reason)
        self.assertEqual(
            choice["token_ids"], [SCRIPT[index % len(SCRIPT)] for index in range(count)]
        )
        self.assertEqual(packet["usage"]["completion_tokens"], count)
        self.assertEqual(packet["usage"]["prompt_tokens"], seen["prompt_tokens"])

    def check_model_defaults(self, seen, config):
        self.selection(seen, temperature=config["temperature"], top_k=config["top_k"],
                       top_p=config["top_p"])

    def check_chat_default_budget(self, seen, prompt_tokens=CHAT_TEST_PROMPT_TOKENS):
        self.assertEqual(seen["prompt_tokens"], prompt_tokens)
        self.assertEqual(seen["budget"], self.context - CHAT_CONTEXT_RESERVE - prompt_tokens)

    def control(self, port, path, *, method="GET", payload=None, status=200,
                content_type="application/json"):
        with closing(http.client.HTTPConnection("127.0.0.1", port, timeout=5)) as connection:
            connection.request(method, path, payload)
            with connection.getresponse() as response:
                data = response.read()
                self.assertEqual(response.status, status, (method, path, data))
                self.assertEqual(response.headers.get_content_type(), content_type)
                self.assertEqual(response.getheader("Content-Length"), str(len(data)))
                return json.loads(data) if content_type == "application/json" else data

    def check_model_discovery(self, port, log, *, prefix_controls):
        started = int(time.time())
        catalog = self.control(port, MODELS)
        self.assertEqual(set(catalog), {"object", "data"})
        self.assertEqual(catalog["object"], "list")
        self.assertIsInstance(catalog["data"], list)
        card, = catalog["data"]
        self.assertIs(type(card["created"]), int)
        self.assertLessEqual(started, card["created"])
        self.assertLessEqual(card["created"], int(time.time()))
        self.assertEqual(card, {
            "id": MODEL, "object": "model", "created": card["created"],
            "owned_by": "sglang", "root": MODEL, "parent": None,
            "max_model_len": self.context,
        })
        self.assertIs(type(card["max_model_len"]), int)
        for route in (TEXT, CHAT):
            with self.subTest(route=route):
                payload = body(route, model=card["id"], max_tokens=1)
                packet, seen = self.exchange(route, payload, port=port, log=log)
                self.completion(packet, seen, 1, "length")
        for method, payload in (("POST", None), ("GET", b"{}")):
            error = self.control(port, MODELS, method=method, payload=payload, status=404)
            self.assertIn("error", error)
        self.assertEqual(self.control(port, "/health")["max_active"], 1)
        self.assertEqual(self.control(port, "/ready"), {"ready": True})
        if prefix_controls:
            metrics = self.control(port, "/metrics", content_type="text/plain")
            self.assertIn(b"sglang_cpp:prefix_cache_entries", metrics)
        else:
            self.assertIn("error", self.control(port, "/metrics", status=404))

    def test_model_discovery(self):
        with self.subTest(prefix_controls=False):
            self.check_model_discovery(self.port, self.log, prefix_controls=False)
        with self.subTest(prefix_controls=True):
            log = self.output / "models-prefix.log"
            with server(self.binary, self.checkpoint, log, "--max-active", "1",
                        "--prefix-cache-entries", "1") as port:
                self.check_model_discovery(port, log, prefix_controls=True)

    def test_text_omitted_sampling_and_budget(self):
        packet, seen = self.exchange(TEXT, body(TEXT, ignore_eos=True))
        self.selection(seen)
        self.assertEqual(seen["budget"], 16)
        self.assertIsNone(seen["seed"])
        self.assertEqual(seen["stops"], [])
        self.completion(packet, seen, 16, "length")
        explicit_packet, explicit = self.exchange(TEXT, body(
            TEXT, ignore_eos=True, temperature=1, top_k=-1, top_p=1, min_p=0, max_tokens=16))
        self.selection(explicit)
        self.assertEqual(explicit["budget"], seen["budget"])
        self.completion(explicit_packet, explicit, 16, "length")

    def test_chat_omitted_sampling_and_budget(self):
        packet, seen = self.exchange(CHAT, body(CHAT))
        self.assertEqual(seen["stops"], self.eos)
        self.assertIsNone(seen["seed"])
        self.completion(packet, seen, 2, "stop")
        with self.subTest(control="sampling"):
            self.check_model_defaults(seen, self.config)
        with self.subTest(control="budget"):
            self.check_chat_default_budget(seen)
        explicit_packet, explicit = self.exchange(CHAT, body(
            CHAT, **{key: self.config[key] for key in ("temperature", "top_k", "top_p")}))
        self.check_model_defaults(explicit, self.config)
        self.check_chat_default_budget(explicit)
        self.completion(explicit_packet, explicit, 2, "stop")

    def test_chat_default_budget_prompt_boundary(self):
        limit = self.context - CHAT_PROMPT_HEADROOM
        for prompt_tokens in (limit - 1, limit):
            with self.subTest(prompt_tokens=prompt_tokens):
                content = "Test" + " a" * (prompt_tokens - CHAT_TEST_PROMPT_TOKENS)
                payload = body(CHAT, messages=[{"role": "user", "content": content}])
                if prompt_tokens == limit:
                    self.exchange(CHAT, payload, status=400)
                else:
                    packet, seen = self.exchange(CHAT, payload)
                    self.check_chat_default_budget(seen, prompt_tokens)
                    self.completion(packet, seen, 2, "stop")

    def test_chat_model_defaults_use_checkpoint_config(self):
        log = self.output / "checkpoint-override.log"
        with TemporaryDirectory(prefix="checkpoint-override-", dir=self.output) as directory:
            checkpoint = Path(directory)
            for name in (
                "config.json", "tokenizer.json", "chat_template.jinja", "tokenizer_config.json"
            ):
                (checkpoint / name).symlink_to(self.checkpoint / name)
            config = {**self.config, "temperature": 0.625, "top_k": 7, "top_p": 0.8}
            (checkpoint / "generation_config.json").write_text(json.dumps(config))
            with server(self.binary, checkpoint, log, "--max-active", "1") as port:
                _, chat = self.exchange(CHAT, body(CHAT, max_tokens=1), port=port, log=log)
                with self.subTest(route=CHAT):
                    self.check_model_defaults(chat, config)
                _, text = self.exchange(TEXT, body(TEXT, max_tokens=1), port=port, log=log)
                with self.subTest(route=TEXT):
                    self.selection(text)

    def test_chat_openai_defaults(self):
        log = self.output / "openai.log"
        with server(self.binary, self.checkpoint, log, "--max-active", "1",
                    "--sampling-defaults", "openai") as port:
            _, omitted = self.exchange(CHAT, body(CHAT, max_tokens=1), port=port, log=log)
            self.selection(omitted)
            _, explicit = self.exchange(CHAT, body(CHAT, max_tokens=1, top_k=7),
                                        port=port, log=log)
            self.selection(explicit, top_k=7)

    def test_explicit_sampling_controls_positive(self):
        for route in (TEXT, CHAT):
            with self.subTest(route=route):
                packet, seen = self.exchange(route, body(
                    route, max_tokens=1, temperature=0.625, top_k=7, top_p=0.8, min_p=0.1))
                self.selection(seen, temperature=0.625, top_k=7, top_p=0.8, min_p=0.1)
                self.completion(packet, seen, 1, "length")

    def test_chat_partial_override(self):
        _, seen = self.exchange(CHAT, body(CHAT, max_tokens=1, top_k=-1))
        self.selection(seen, temperature=self.config["temperature"], top_p=self.config["top_p"])

    def test_chat_nulls_use_defaults(self):
        packet, seen = self.exchange(CHAT, body(CHAT, temperature=None, top_k=None,
                                                 top_p=None, min_p=None, max_tokens=None,
                                                 max_completion_tokens=None))
        self.check_model_defaults(seen, self.config)
        self.check_chat_default_budget(seen)
        self.completion(packet, seen, 2, "stop")

    def test_text_nulls_rejected_positive(self):
        for field in ("temperature", "top_k", "top_p", "min_p", "max_tokens"):
            with self.subTest(field=field):
                self.exchange(TEXT, body(TEXT, **{field: None}), status=400)

    def test_invalid_controls_positive(self):
        for route in (TEXT, CHAT):
            for field, value in (("temperature", -0.1), ("top_k", 0), ("top_p", 0),
                                 ("top_p", 1.1), ("min_p", 1.1), ("max_tokens", -1),
                                 ("stop_token_ids", [self.vocab])):
                with self.subTest(route=route, field=field, value=value):
                    self.exchange(route, body(route, **{field: value}), status=400)

    def test_greedy_boundary_positive(self):
        for route in (TEXT, CHAT):
            for options, expected in (
                ({"temperature": 0}, False),
                ({"temperature": 0.0000009}, False),
                ({"temperature": 0.000001}, True),
                ({"temperature": 1, "top_k": 1}, False),
            ):
                with self.subTest(route=route, options=options):
                    _, seen = self.exchange(route, body(route, max_tokens=1, **options))
                    self.assertIs(seen["sampled"], expected)

    def test_seed_transport_positive(self):
        for route in (TEXT, CHAT):
            for options, expected in (({}, None), ({"seed": 0}, 0),
                                      ({"seed": -(2**63)}, -(2**63)),
                                      ({"seed": 2**63 - 1}, 2**63 - 1)):
                with self.subTest(route=route, seed=options):
                    _, seen = self.exchange(route, body(route, max_tokens=1, temperature=0.75,
                                                        top_k=7, **options))
                    self.assertIs(seen["sampled"], True)
                    self.assertEqual(seen["seed"], expected)
            for out_of_range in (-(2**63) - 1, 2**63):
                with self.subTest(route=route, out_of_range=out_of_range):
                    self.exchange(route, body(route, seed=out_of_range), status=400)

    def test_wire_null_seed_is_unspecified(self):
        for route in (TEXT, CHAT):
            with self.subTest(route=route):
                _, seen = self.exchange(route, body(route, max_tokens=1, temperature=0.75,
                                                    top_k=7, seed=None))
                self.assertIs(seen["sampled"], True)
                self.assertIsNone(seen["seed"])

    def test_explicit_budgets_positive(self):
        for route, field in ((TEXT, "max_tokens"), (CHAT, "max_tokens"),
                             (CHAT, "max_completion_tokens")):
            for budget in (0, 1):
                with self.subTest(route=route, field=field, budget=budget):
                    packet, seen = self.exchange(route, body(route, **{field: budget}))
                    self.assertEqual(seen["budget"], budget)
                    self.completion(packet, seen, budget, "length")

    def test_chat_budget_alias_precedence(self):
        for new, old, expected in ((0, 9, 0), (1, 0, 1), (None, 1, 1), (1, None, 1)):
            with self.subTest(max_completion_tokens=new, max_tokens=old):
                packet, seen = self.exchange(CHAT, body(CHAT, max_completion_tokens=new,
                                                        max_tokens=old))
                self.assertEqual(seen["budget"], expected)
                self.completion(packet, seen, expected, "length")

    def test_eos_and_explicit_stop_at_budget_positive(self):
        for route in (TEXT, CHAT):
            for count, options, stops in ((2, {}, self.eos),
                                          (1, {"stop_token_ids": [SCRIPT[0]]},
                                           [SCRIPT[0], *self.eos]),
                                          (4, {"ignore_eos": True}, [])):
                with self.subTest(route=route, options=options):
                    packet, seen = self.exchange(route, body(route, max_tokens=count, **options))
                    self.assertEqual(seen["stops"], stops)
                    self.completion(packet, seen, count, "stop" if stops else "length")
                    if options.get("stop_token_ids"):
                        choice, = packet["choices"]
                        text = choice["text"] if route == TEXT else choice["message"]["content"]
                        self.assertEqual(text, "")

    def test_ignore_eos_also_bypasses_explicit_stops(self):
        for route in (TEXT, CHAT):
            with self.subTest(route=route):
                packet, seen = self.exchange(route, body(route, max_tokens=4, ignore_eos=True,
                                                        stop_token_ids=[SCRIPT[0]]))
                self.assertEqual(seen["stops"], [])
                self.completion(packet, seen, 4, "length")

    def test_sse_stop_and_usage_positive(self):
        for route in (TEXT, CHAT):
            with self.subTest(route=route):
                packets, seen = self.exchange(route, body(
                    route, max_tokens=2, stream=True, stream_options={"include_usage": True}))
                choices = [choice for packet in packets for choice in packet["choices"]]
                self.assertEqual(
                    [token for choice in choices for token in choice.get("token_ids", [])],
                    SCRIPT[:2]
                )
                self.assertEqual([choice["finish_reason"] for choice in choices
                                  if choice["finish_reason"] is not None], ["stop"])
                self.assertEqual(packets[-1]["usage"]["completion_tokens"], 2)
                self.assertEqual(packets[-1]["usage"]["prompt_tokens"], seen["prompt_tokens"])

    def test_benchmark_explicit_greedy_positive(self):
        payload = legacy([32], 1)
        packets, seen = self.exchange(TEXT, payload)
        self.selection(seen, temperature=0, sampled=False)
        self.assertEqual(seen["budget"], 1)
        self.assertEqual(packets[-1]["usage"]["completion_tokens"], 1)
        self.exchange(TEXT, payload | {"seed": 1}, status=400)

    def test_mixed_batch_preserves_request_options_positive(self):
        log = self.output / "mixed.log"
        with server(self.binary, self.checkpoint, log, "--max-active", "2") as port:
            barrier = Barrier(2)

            def call(route, **options):
                barrier.wait(timeout=5)
                return send(port, route, body(
                    route, max_tokens=24, ignore_eos=True, **options))

            with ThreadPoolExecutor(max_workers=2) as clients:
                text = clients.submit(call, TEXT, temperature=0, seed=11)
                chat = clients.submit(call, CHAT, temperature=0.625, top_k=7, top_p=0.8, seed=22)
                results = {11: text.result(), 22: chat.result()}
            by_seed = {seen["seed"]: seen for seen in receipts(log)}
            self.assertEqual(set(by_seed), {11, 22})
            self.selection(by_seed[11], temperature=0, sampled=False)
            self.selection(by_seed[22], temperature=0.625, top_k=7, top_p=0.8)
            self.assertIn("cpu sampling mixed step", log.read_text())
            for seed, (status, packet) in results.items():
                self.assertEqual(status, 200, packet)
                self.completion(packet, by_seed[seed], 24, "length")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options, tests = parser.parse_known_args()
    Routes.binary = options.server.resolve(strict=True)
    Routes.checkpoint = options.checkpoint.resolve(strict=True)
    Routes.output = options.output.resolve()
    Routes.output.mkdir(parents=True, exist_ok=True)
    unittest.main(argv=[__file__, *tests], verbosity=2)
