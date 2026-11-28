"""Exercise public text and chat sampling routes with the CPU decoder double."""

import argparse
import http.client
import json
import os
import socket
import subprocess
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, contextmanager
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Barrier
from unittest.mock import patch

from test_native_chat import request
from test_native_text import MODEL, event, legacy

TEXT = "/v1/completions"
CHAT = "/v1/chat/completions"
MODELS = "/v1/models"
MODEL_INFO = ("/model_info", "/get_model_info")
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


def greedy_payload(route, fields, *, budget, stream=None):
    streaming = {} if stream is None else {"stream": stream}
    if route == "/generate":
        return {"input_ids": [32, 220], **streaming, "sampling_params": {
            "max_new_tokens": budget, "temperature": 0, **fields,
        }}
    return body(route, max_tokens=budget, temperature=0, **streaming, **fields)


@contextmanager
def control_response(port, method, path, payload=None, headers=None):
    with closing(http.client.HTTPConnection("127.0.0.1", port, timeout=5)) as connection:
        connection.request(method, path, payload, headers or {})
        with connection.getresponse() as response:
            yield response


def wire(port, method, path, payload=None, headers=None):
    with control_response(port, method, path, payload, headers) as response:
        return response.status, response.headers, response.read()


def sse_frames(response):
    return iter(lambda: event(response), "[DONE]")


def checked_body(test, response, frames=None):
    test.assertEqual(response.headers.get_content_type(),
                     "application/json" if frames is None else "text/event-stream")
    if frames is None:
        return json.load(response)
    packets = list(frames)
    test.assertEqual(response.read(), b"")
    return packets


@contextmanager
def post_response(port, route, payload):
    connection, response = request(port, route, payload)
    with closing(connection), response:
        yield response


def send(port, route, payload):
    with post_response(port, route, payload) as response:
        if response.headers.get_content_type() == "text/event-stream":
            events = list(sse_frames(response))
            assert response.read() == b""
            return response.status, events
        return response.status, json.load(response)


def receipts(log):
    return [json.loads(line.removeprefix("cpu sampling ")) for line in log.read_text().splitlines()
            if line.startswith(OPTIONS_PREFIX)]


@contextmanager
def server(binary, checkpoint, log, *flags, sampling=True, with_process=False):
    with closing(socket.socket()) as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    command = [str(binary), "--checkpoint", str(checkpoint), "--port", str(port), *flags]
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    if sampling:
        environment["SGLANG_TEST_SAMPLING_RECEIPTS"] = "1"
    else:
        environment.pop("SGLANG_TEST_SAMPLING_RECEIPTS", None)
    with TemporaryDirectory(prefix="sglang-cpu-") as runtime, log.open("wb") as output:
        environment["XDG_RUNTIME_DIR"] = runtime
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
            yield (port, process) if with_process else port
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)


class Routes(unittest.TestCase):
    def test_default_mtp_and_target_only_launch(self):
        data = self.control(self.port, "/server_info")
        self.assertEqual(data["fp8_prefill"], "exact")
        self.assertEqual(data["mtp_steps"], 4)
        self.assertIs(data["mtp_int8_head"], False)
        self.assertIs(data["certified_int8_verify_head"], False)
        log = self.output / "draft-head-k0.log"
        with server(self.binary, self.checkpoint, log, "--mtp-steps", "0") as port:
            data = self.control(port, "/server_info")
            self.assertEqual(data["mtp_steps"], 0)
            self.assertIs(data["mtp_int8_head"], False)
            self.assertIs(data["certified_int8_verify_head"], False)
            response, receipt = self.exchange(
                TEXT, body(TEXT, max_tokens=1, temperature=0, ignore_eos=True),
                port=port, log=log,
            )
            self.completion(response, receipt, 1, "length")

    def test_scheduler_budget_default_and_override(self):
        for flag, budget in (((), 32768), (("--max-num-batched-tokens", "8192"), 8192)):
            with self.subTest(budget=budget):
                log = self.output / f"scheduler-budget-{budget}.log"
                with server(self.binary, self.checkpoint, log, *flag) as port:
                    with closing(http.client.HTTPConnection("127.0.0.1", port, timeout=5)) as conn:
                        conn.request("GET", "/health")
                        with conn.getresponse() as response:
                            self.assertEqual(response.status, 200)
                            self.assertEqual(json.load(response), {
                                "max_active": 8, "max_num_batched_tokens": budget,
                            })

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

    def test_sparse_logit_bias(self):
        for route in ("/generate", TEXT, CHAT):
            payload = partial(greedy_payload, route, budget=1)

            with self.subTest(route=route):
                for fields in ({}, {"logit_bias": None}, {"logit_bias": {}},
                               {"logit_bias": {"32": 4, "032": -0.0}}):
                    _, seen = self.exchange(route, payload(fields))
                    self.assertEqual(seen["logit_bias"], [])
                _, seen = self.exchange(route, payload({"logit_bias": {
                    "95": -2.25, "32": 1, " +0032 ": 0.125, str(self.vocab - 1): 2**70,
                }}))
                self.assertEqual(seen["logit_bias"], [
                    [32, 0.125], [95, -2.25], [self.vocab - 1, 2**70],
                ])
                _, seen = self.exchange(route, payload({"logit_bias": {
                    " +0032 ": 0.125, "32": 1,
                }}))
                self.assertEqual(seen["logit_bias"], [[32, 1]])
                for invalid in ([], 1, {"": 1}, {"32x": 1}, {"+-0": 1}, {"-1": 1},
                                {str(self.vocab): 1}, {"32": True}, {"32": "1"},
                                {"32": None}, {"32": 1e31}, {"32": -1e31}):
                    self.exchange(route, {**payload({"logit_bias": invalid}), "stream": True},
                                  status=400)
                duplicate = json.dumps(payload({"logit_bias": {"32": 1}})).replace(
                    '"32": 1', '"32": 1, "32": 2',
                ).encode()
                self.exchange(route, duplicate, status=400)

    def test_native_schema_validation_precedes_submission(self):
        schema = json.dumps({"type": "null"})
        invalid = (
            *[({"json_schema": value}, {}, "must be a string")
              for value in ({}, False, 0, [])],
            ({"json_schema": "{"}, {}, "must contain valid JSON"),
            ({"json_schema": '{"type":"null","type":"string"}'}, {}, "must contain valid JSON"),
            ({"json_schema": "true"}, {}, "JSON Schema"),
            ({"json_schema": '{"$ref":"#"}'}, {}, "JSON Schema"),
            ({"json_schema": schema, "regex": "A"}, {}, "Use one of"),
            ({"json_schema": schema, "regex": False}, {}, "regex must be a string"),
            ({"json_schema": schema, "max_new_tokens": 0}, {}, "max_new_tokens"),
            ({"json_schema": '{"enum":["a b c d"]}', "max_new_tokens": 1}, {},
             "max_new_tokens"),
            ({"json_schema": schema, "ignore_eos": True}, {}, "model EOS"),
            ({"json_schema": schema, "stop_token_ids": []}, {}, "model EOS"),
            ({"json_schema": schema, "temperature": 0}, {"return_logprob": True}, "logprobs"),
            ([{"json_schema": schema}, {"json_schema": "["}], {"input_ids": [[32], [33]]},
             "must contain valid JSON"),
            ({}, {"json_schema": schema}, "Unsupported"),
        )
        first = len(receipts(self.log))
        for params, fields, message in invalid:
            for stream in (False, True):
                with self.subTest(params=params, fields=fields, stream=stream):
                    payload = {"input_ids": [32], "sampling_params": params,
                               "stream": stream, **fields}
                    status, result = send(self.port, "/generate", payload)
                    self.assertEqual(status, 400, result)
                    self.assertIn(message, result["error"]["message"])
        self.assertEqual(len(receipts(self.log)), first)

    def test_native_schema_null_and_empty_use_plain_sampling(self):
        for fields in ({}, {"json_schema": None}, {"json_schema": ""}):
            with self.subTest(fields=fields):
                item, _ = self.exchange("/generate", {
                    "input_ids": [32], "sampling_params": {"max_new_tokens": 1, **fields},
                })
                self.assertEqual(item["output_ids"], [32])
                self.assertEqual(item["meta_info"]["finish_reason"],
                                 {"type": "length", "length": 1})

    def test_output_history_penalties(self):
        names = ("frequency_penalty", "presence_penalty")
        for route in ("/generate", TEXT, CHAT):
            payload = partial(greedy_payload, route, budget=1)

            for values in ((0, 0), (-2, 2), (2, -2), (-0.29, 1.125), (None, None)):
                with self.subTest(route=route, values=values):
                    request_body = payload(dict(zip(names, values)))
                    if values[0] is None and route != "/generate":
                        self.exchange(route, request_body, status=400)
                    else:
                        _, seen = self.exchange(route, request_body)
                        self.assertEqual(tuple(seen[name] for name in names),
                                         (0, 0) if values[0] is None else values)

            for name in names:
                for invalid in (-2.01, 2.01, True, "0.5", []):
                    with self.subTest(route=route, name=name, invalid=invalid):
                        self.exchange(route, {**payload({name: invalid}), "stream": True},
                                      status=400)

    def test_minimum_output(self):
        for route in ("/generate", TEXT, CHAT):
            native = route == "/generate"
            name = "min_new_tokens" if native else "min_tokens"
            other = "min_tokens" if native else "min_new_tokens"

            payload = partial(greedy_payload, route, budget=3, stream=False)

            with self.subTest(route=route):
                _, implicit = self.exchange(route, payload({}))
                _, zero = self.exchange(route, payload({name: 0}))
                self.assertEqual(implicit, zero)
                self.assertEqual(zero["minimum"], 0)
                for value in (1, 3):
                    _, seen = self.exchange(route, payload({name: value}, stream=True))
                    self.assertEqual(seen["minimum"], value)
                for value in (-1, 4, True, "1", 1.5, [], 2**40):
                    self.exchange(route, payload({name: value}, stream=True), status=400)
                self.exchange(route, payload({name: 1}, budget=0), status=400)
                self.exchange(route, payload({other: 1}), status=400)
                if native:
                    _, seen = self.exchange(route, payload({name: None}))
                    self.assertEqual(seen, zero)
                else:
                    self.exchange(route, payload({name: None}), status=400)
        self.exchange(CHAT, body(CHAT, max_tokens=3, max_completion_tokens=1, min_tokens=2),
                      status=400)

    def test_repetition_penalty(self):
        for route in ("/generate", TEXT, CHAT):
            payload = partial(greedy_payload, route, budget=3, stream=False)

            with self.subTest(route=route):
                _, default = self.exchange(route, payload({}))
                _, neutral = self.exchange(route, payload({"repetition_penalty": 1}))
                self.assertEqual(default, neutral)
                self.assertEqual(neutral["repetition_penalty"], 1)
                for value in (5e-324, 1e-46, 0.375, 2):
                    _, seen = self.exchange(route, payload({
                        "repetition_penalty": value, "frequency_penalty": -0.25,
                        "presence_penalty": 0.5, "logit_bias": {"32": 0.125},
                    }, stream=True))
                    self.assertEqual(seen["repetition_penalty"], value)
                    self.assertEqual(seen["frequency_penalty"], -0.25)
                    self.assertEqual(seen["presence_penalty"], 0.5)
                    self.assertEqual(seen["logit_bias"], [[32, 0.125]])
                for invalid in (0, -0.01, 2.01, True, "0.5", [], {}):
                    self.exchange(route, payload({"repetition_penalty": invalid}, stream=True),
                                  status=400)
                null = payload({"repetition_penalty": None})
                if route == TEXT:
                    self.exchange(route, null, status=400)
                else:
                    _, seen = self.exchange(route, null)
                    self.assertEqual(seen, default)

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
        actual, headers, data = wire(port, method, path, payload)
        self.assertEqual(actual, status, (method, path, data))
        self.assertEqual(headers.get_content_type(), content_type)
        self.assertEqual(headers["Content-Length"], str(len(data)))
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
        with self.subTest(prefix_controls=True):
            self.check_model_discovery(self.port, self.log, prefix_controls=True)
        with self.subTest(prefix_controls=False):
            log = self.output / "models-no-prefix.log"
            with server(self.binary, self.checkpoint, log, "--max-active", "1",
                        "--prefix-cache-entries", "0") as port:
                self.check_model_discovery(port, log, prefix_controls=False)

    def check_model_info(self, port, checkpoint, *, reasoning=None, tools=None):
        config = json.loads((checkpoint / "config.json").read_text())
        card, = self.control(port, MODELS)["data"]
        expected = {
            "model_path": str(checkpoint),
            "served_model_name": card["id"],
            "tokenizer_path": str(checkpoint),
            "is_generation": True,
            "preferred_sampling_params": None,
            "weight_version": "default",
            "load_format": "safetensors",
            "reasoning_parser": reasoning,
            "tool_call_parser": tools,
            "disaggregation_mode": "null",
            "has_image_understanding": True,
            "has_audio_understanding": False,
            "model_type": config["model_type"],
            "architectures": config["architectures"],
        }
        for path in MODEL_INFO:
            with self.subTest(path=path):
                info = self.control(port, path)
                self.assertEqual(info, expected)
                for field in ("is_generation", "has_image_understanding",
                              "has_audio_understanding"):
                    self.assertIs(info[field], expected[field])

    def test_model_info(self):
        self.check_model_info(self.port, self.checkpoint)

    def test_model_info_parser_flags(self):
        cases = (
            (("--reasoning-parser", "qwen3"), "qwen3", None),
            (("--tool-call-parser", "qwen3_coder"), None, "qwen3_coder"),
            (("--reasoning-parser", "qwen3", "--tool-call-parser", "qwen3_coder"),
             "qwen3", "qwen3_coder"),
        )
        with TemporaryDirectory(prefix="model-info-", dir=self.output) as directory:
            checkpoint = Path(directory) / 'checkpoint 你好 "with quotes"'
            checkpoint.symlink_to(self.checkpoint, target_is_directory=True)
            for flags, reasoning, tools in cases:
                with self.subTest(reasoning=reasoning, tools=tools):
                    log = self.output / f"model-info-{reasoning}-{tools}.log"
                    with server(self.binary, checkpoint, log, *flags) as port:
                        self.check_model_info(port, checkpoint, reasoning=reasoning, tools=tools)
                    self.assertEqual(receipts(log), [])

    def test_model_info_http_routing(self):
        for path in MODEL_INFO:
            with self.subTest(path=path):
                self.assert_redirect("GET", path + "/?x=1", path + "?x=1")
                self.assertEqual(self.control(self.port, path + "?x=1"),
                                 self.control(self.port, path))
                self.assertIn("error", self.control(self.port, path + "_extra", status=404))
            self.assertIn("error", self.control(self.port, path, payload=b"{}", status=404))
            for method in ("POST", "PUT", "HEAD", "OPTIONS", "GE(T"):
                with self.subTest(path=path, method=method):
                    with socket.create_connection(("127.0.0.1", self.port), timeout=5) as client:
                        client.sendall(f"{method} {path} HTTP/1.1\r\nHost: localhost\r\n\r\n"
                                       .encode("ascii"))
                        with client.makefile("rb") as response:
                            status, _, data = response.read().partition(b"\r\n\r\n")
                    if method == "GE(T":
                        self.assertTrue(status.startswith(b"HTTP/1.1 400 "))
                        self.assertIn("error", json.loads(data))
                    else:
                        self.assertTrue(status.startswith(b"HTTP/1.1 405 "))
                        self.assertIn(b"\r\nAllow: GET\r\n", status)
                        self.assertIn(b"\r\nContent-Type: application/json\r\n", status)
                        if method == "HEAD":
                            self.assertEqual(data, b"")
                        else:
                            self.assertEqual(json.loads(data), {"detail": "Method Not Allowed"})

    def assert_redirect(self, method, target, location):
        status, headers, body = wire(self.port, method, target)
        self.assertEqual(status, 307, (target, body))
        self.assertEqual(headers["Location"], f"http://127.0.0.1:{self.port}{location}")
        self.assertEqual((headers["Content-Length"], body), ("0", b""))

    def test_model_path_decoded_once_after_raw_query_split(self):
        card = self.control(self.port, f"/v1/%6dodels%2F{MODEL}?q=%3F?tail")
        self.assertEqual((card["id"], card["object"]), (MODEL, "model"))
        for suffix, model in (
            ("%2551wen3.8-27B-FP8", "%51wen3.8-27B-FP8"),
            ("a%2Fb%3F%23+c?ignored=%2F", "a/b?#+c"), ("", ""), (MODEL + "/", MODEL + "/"),
        ):
            with self.subTest(suffix=suffix):
                result = self.control(self.port, f"/v1/models/{suffix}", status=404)
                self.assertEqual(result["error"], {
                    "message": f"The model '{model}' does not exist",
                    "type": "invalid_request_error", "param": "model", "code": "model_not_found",
                })
        self.assertEqual(wire(self.port, "GET", f"/v1/models%252F{MODEL}")[0], 404)

    def test_encoded_put_and_redirect(self):
        self.assert_redirect("PUT", "/%67enerate%2F?x=%23?y", "/generate?x=%23?y")
        target = "/%67enerate?stream=false&trace=%2F"
        status, headers, data = wire(self.port, "DELETE", target)
        self.assertEqual((status, set(headers["Allow"].split(", "))), (405, {"POST", "PUT"}))
        self.assertEqual(json.loads(data), {"detail": "Method Not Allowed"})
        payload = {"input_ids": [32], "stream": True,
                   "sampling_params": {"max_new_tokens": 1, "temperature": 0}}
        status, headers, data = wire(self.port, "PUT", target, json.dumps(payload),
                                     {"Content-Type": "application/json"})
        self.assertEqual(status, 200, data)
        self.assertEqual(headers["Content-Type"], "text/event-stream")
        frames = data.split(b"\n\n")
        self.assertEqual(frames[-2:], [b"data: [DONE]", b""])
        final = json.loads(frames[-3].removeprefix(b"data: "))
        self.assertEqual(len(final["output_ids"]), 1)
        self.assertEqual(final["meta_info"]["finish_reason"], {"type": "length", "length": 1})

    def test_streaming_session_open_points_to_generate_sse(self):
        session = {"session_id": "cpu-sse", "capacity_of_str_len": 16}
        status, error = send(self.port, "/open_session", session | {"streaming": True})
        self.assertEqual(status, 400)
        self.assertEqual(error, {
            "error": {"message": "Streaming session opens are unsupported. "
                                 "Use stream: true on /generate for SSE."},
        })
        self.assertEqual(send(self.port, "/open_session", session), (200, "cpu-sse"))
        status, frames = send(self.port, "/generate", {
            "input_ids": [32], "session_params": {"id": "cpu-sse"}, "stream": True,
            "sampling_params": {"max_new_tokens": 1},
        })
        self.assertEqual(status, 200)
        self.assertEqual(len(frames[-1]["output_ids"]), 1)
        self.assertEqual(frames[-1]["meta_info"]["finish_reason"], {"type": "length", "length": 1})

    def test_server_info_aliases_and_redirects(self):
        expected = self.control(self.port, "/server_info")
        self.assertEqual(expected["port"], self.port)
        self.assertEqual(expected["internal_states"][0]["num_running_reqs"], 0)
        for path in ("/server_info", "/get_server_info"):
            with self.subTest(path=path):
                self.exchange(TEXT, body(TEXT, max_tokens=1, ignore_eos=True))
                expected["internal_states"][0]["generation_tokens_total"] += 1
                encoded = path.replace("_info", "%5finfo")
                self.assertEqual(self.control(self.port, encoded + "?query=%3f"), expected)
                self.assert_redirect("GET", encoded + "%2f?q=%252F", path + "?q=%252F")
                status, headers, data = wire(self.port, "HEAD", encoded)
                self.assertEqual((status, headers["Allow"], data), (405, "GET", b""))

    def test_cuda_graph_launch_defaults_and_eager_override(self):
        graph = ("--batch-cuda-graphs",)
        eager = ("--eager",)
        runtime = Path(self.enterContext(TemporaryDirectory(prefix="sglang-socket-")))
        lmcache_socket = ("--lmcache-socket", str(runtime / "missing.sock"))
        no_prefix = ("--prefix-cache-entries", "0")
        cases = (
            ("default", (), True, True),
            ("legacy", graph, True, True),
            ("eager", eager, False, True),
            ("legacy-then-eager", (*graph, *eager), False, True),
            ("eager-then-legacy", (*eager, *graph), False, True),
            ("lmcache-socket", lmcache_socket, True, True),
            ("eager-lmcache-socket", (*eager, *lmcache_socket), False, True),
            ("no-prefix", no_prefix, True, False),
            ("eager-no-prefix", (*eager, *no_prefix), False, False),
        )
        for name, flags, graphs, lmcache_enabled in cases:
            with self.subTest(name=name):
                log = self.output / f"graphs-launch-{name}.log"
                with server(self.binary, self.checkpoint, log, *flags) as port:
                    info = self.control(port, "/server_info")
                    self.assertIs(info["batch_cuda_graphs"], graphs)
                    self.assertIs(info["eager"], not graphs)
                    self.assertIs(info["lmcache_enabled"], lmcache_enabled)
                    self.assertIs(info["lmcache_available"], False)
                    self.assertEqual(info["prefix_cache_entries"], 0)
                    self.assertIs(info["enable_multimodal"], True)
                    expected_receipt = f"cpu batch cuda_graphs={int(graphs)}"
                    self.assertIn(expected_receipt, log.read_text().splitlines())

    def test_mtp_launch_depth_with_lmcache(self):
        runtime = Path(self.enterContext(TemporaryDirectory(prefix="sglang-socket-")))
        socket_flag = ("--lmcache-socket", str(runtime / "missing.sock"))
        cases = (("lmcache", socket_flag, 4),
                 ("lmcache-target-only", (*socket_flag, "--mtp-k", "0"), 0))
        for name, flags, depth in cases:
            with self.subTest(name=name):
                log = self.output / f"mtp-launch-{name}.log"
                with server(self.binary, self.checkpoint, log, *flags) as port:
                    info = self.control(port, "/server_info")
                    self.assertEqual(info["mtp_steps"], depth)
                    self.assertIs(info["lmcache_enabled"], True)
                    self.assertIs(info["lmcache_available"], False)

    def test_flush_cache_methods_and_timeout(self):
        flushed = (b"Cache flushed.\nPlease check backend logs for more details. "
                   b"(When there are running or waiting requests, the operation will not be "
                   b"performed.)\n")
        log = self.output / "flush-disabled.log"
        port = self.enterContext(server(self.binary, self.checkpoint, log,
                                        "--prefix-cache-entries", "0"))
        for method, query in (("GET", ""), ("POST", "?timeout=bad&time%6Fut=0"),
                              ("GET", "?timeout=1e%2B3")):
            with self.subTest(method=method, query=query):
                status, headers, data = wire(port, method, "/flush%5fcache" + query)
                self.assertEqual((status, data), (200, flushed))
                self.assertEqual(headers["Content-Length"], str(len(flushed)))
                self.assertIsNone(headers["Content-Type"])
        error = self.control(self.port, "/flush_cache", status=503)
        self.assertEqual(error["error"]["message"], "LMCache namespace reset was not acknowledged")
        self.assert_redirect("POST", "/flush_cache%2F?timeout=%FF", "/flush_cache?timeout=%FF")
        status, headers, data = wire(self.port, "HEAD", "/flush_cache")
        self.assertEqual((status, set(headers["Allow"].split(", ")), data),
                         (405, {"GET", "POST"}, b""))
        for method, value, decoded, diagnostic in (
            ("GET", "1e+3", "1e 3", "valid number"),
            ("POST", "-1", "-1", "greater than or equal to 0"),
            ("GET", "NaN", "NaN", "greater than or equal to 0"),
            ("POST", "0&timeout=%FF", "\ufffd", "valid number"),
        ):
            with self.subTest(method=method, value=value):
                error = self.control(self.port, "/flush_cache?timeout=" + value,
                                     method=method, status=400)
                message = error.pop("message")
                self.assertEqual(error, {
                    "object": "error", "type": "Bad Request", "param": None, "code": 400,
                })
                self.assertIn("query.timeout", message)
                self.assertIn(diagnostic, message)
                self.assertIn(decoded, message)

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
            config = {**self.config, "temperature": 0.625, "top_k": 7, "top_p": 0.8,
                      "repetition_penalty": 1.25}
            (checkpoint / "generation_config.json").write_text(json.dumps(config))
            with server(self.binary, checkpoint, log, "--max-active", "1",
                        "--prefix-cache-entries", "0") as port:
                _, chat = self.exchange(CHAT, body(CHAT, max_tokens=1), port=port, log=log)
                with self.subTest(route=CHAT):
                    self.check_model_defaults(chat, config)
                    self.assertEqual(chat["repetition_penalty"], 1.25)
                    for value, expected in ((None, 1.25), (1, 1)):
                        payload = body(CHAT, max_tokens=1, repetition_penalty=value)
                        _, seen = self.exchange(CHAT, payload, port=port, log=log)
                        self.assertEqual(seen["repetition_penalty"], expected)
                _, text = self.exchange(TEXT, body(TEXT, max_tokens=1), port=port, log=log)
                with self.subTest(route=TEXT):
                    self.selection(text)
                    self.assertEqual(text["repetition_penalty"], 1)

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

    def test_default_completion_burst_reserves_idle_workers(self) -> None:
        self.check_completion_burst((), active=8, row_budget=32768, assert_overload=True)

    def test_explicit_sixteen_completion_workers(self) -> None:
        self.check_completion_burst(
            ("--max-active", "16", "--max-num-batched-tokens", "8192"),
            active=16,
            row_budget=8192,
            assert_overload=False,
        )

    def check_completion_burst(
        self, flags: tuple[str, ...], *, active: int, row_budget: int, assert_overload: bool
    ) -> None:
        width = 16
        barrier = Barrier(width)
        payload = body(TEXT, prompt=[32], max_tokens=1, temperature=0, seed=-41, ignore_eos=True)
        log = self.output / f"burst-{active}.log"
        with TemporaryDirectory(dir=self.output) as directory:
            gate = Path(directory) / "step"
            with (
                patch.dict(os.environ, SGLANG_TEST_BATCH_STEP_GATE=str(gate)),
                server(self.binary, self.checkpoint, log, *flags) as port,
            ):
                self.assertEqual(
                    self.control(port, "/health"),
                    {"max_active": active, "max_num_batched_tokens": row_budget},
                )

                def launch() -> tuple[int, dict[str, object]]:
                    barrier.wait(timeout=5)
                    return send(port, TEXT, payload)

                expected = {
                    f'vllm:num_requests_running{{engine="0"}} {active}',
                    f'vllm:num_requests_waiting{{engine="0"}} {width - active}',
                }
                with ThreadPoolExecutor(max_workers=width) as clients:
                    requests = [clients.submit(launch) for _ in range(width)]
                    try:
                        deadline = time.monotonic() + 3
                        while True:
                            metrics = self.control(port, "/metrics", content_type="text/plain")
                            if expected <= set(metrics.decode().splitlines()):
                                break
                            finished = [client.result() for client in requests if client.done()]
                            self.assertEqual(
                                finished, [], "A burst client completed before release"
                            )
                            self.assertLess(time.monotonic(), deadline, metrics)
                            time.sleep(0.01)
                        if assert_overload:
                            status, error = send(port, TEXT, payload)
                            self.assertEqual(status, 503)
                            self.assertEqual(
                                error["error"]["message"], "Native completion queue is full"
                            )
                    finally:
                        gate.with_suffix(".0").touch()
                    for client in requests:
                        status, result = client.result()
                        self.assertEqual(status, 200, result)
                        self.assertEqual(result["usage"]["completion_tokens"], 1)
                self.assertEqual(len(receipts(log)), width)

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


def run_cpu_suite(test_case, script, description, *, server_option=False,
                  select_tests=False, verbosity=2):
    parser = argparse.ArgumentParser(description=description)
    binary = "server" if server_option else "binary"
    parser.add_argument(f"--{binary}", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options, tests = parser.parse_known_args() if select_tests else (parser.parse_args(), [])
    test_case.binary = getattr(options, binary).resolve(strict=server_option)
    test_case.checkpoint = options.checkpoint.resolve(strict=server_option)
    test_case.output = options.output.resolve()
    test_case.output.mkdir(parents=True, exist_ok=True)
    unittest.main(argv=[script, *tests], verbosity=verbosity)


if __name__ == "__main__":
    run_cpu_suite(Routes, __file__, __doc__, server_option=True, select_tests=True)
