import importlib.util
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from urllib.parse import urlsplit


DIRECTORY = Path(__file__).resolve().parents[1] / "benchmarks" / "streaming"
spec = importlib.util.spec_from_file_location("streaming_client", DIRECTORY / "client.py")
client = importlib.util.module_from_spec(spec)
spec.loader.exec_module(client)


def choice(token_ids, prompt=None, reason=None):
    return {"choices": [{"index": 0, "token_ids": token_ids,
                         "prompt_token_ids": prompt, "finish_reason": reason}]}


def frame(value):
    return b"data: " + client.json_bytes(value) + b"\r\n\r\n"


class Reads:
    length = None

    def __init__(self, *reads):
        self.reads = iter(reads)

    def read1(self, _size):
        return next(self.reads, b"")


class StreamingClientTest(unittest.TestCase):
    def test_clock_tracks_completed_frames_and_shares_read_timestamp(self):
        response = Reads(b"data: ear", b"ly\r\n\r\ndata: same\n\n:keep",
                         b"alive\n\n", b"data: unfinished")
        with patch.object(client, "perf_counter_ns", side_effect=(10, 20, 30, 40, 50)):
            events = list(client.sse_frames(response))
        self.assertEqual([(event["ns"], event["lines"]) for event in events], [
            (20, ["data: early"]), (20, ["data: same"]), (30, [":keepalive"]), (50, []),
        ])
        self.assertEqual(bytes.fromhex(events[-1]["unterminated"]), b"data: unfinished")

    def test_live_http_delta_chunks_metadata_and_done(self):
        case = {"workload": "W1", "prompt_token_ids": [14, 25], "output_tokens": 3}
        request_body = client.json_bytes(client.request_body(case))
        usage = {"prompt_tokens": 2, "completion_tokens": 3,
                 "prompt_tokens_details": {"cached_tokens": 0}}
        data = (b": heartbeat\n\n" + frame(choice([], [14, 25])) + frame(choice([57, 58]))
                + frame(choice([59], reason="length")) + frame({"choices": [], "usage": usage}))

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                size = int(self.headers["Content-Length"])
                self.server.received = (self.path, self.rfile.read(size))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                length = len(self.server.payload) + self.server.extra
                self.send_header("Content-Length", str(length))
                self.end_headers()
                self.wfile.write(self.server.payload)

            def log_message(self, *_):
                pass

        with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            try:
                origin = urlsplit(f"http://127.0.0.1:{server.server_port}")
                terminal = b"data: [DONE]\n\n"
                server.extra = 0
                server.payload = data + terminal + b": final heartbeat\n\n"
                result = client.measure(origin, case, request_body)
                self.assertEqual(server.received, ("/v1/completions", request_body))
                self.assertNotIn("error", result)
                self.assertEqual(result["output_ids"], [57, 58, 59])
                self.assertEqual([size for _, size in result["token_chunks"]], [2, 1])
                first, last = (chunk[0] for chunk in result["token_chunks"])
                self.assertEqual(result["metrics"]["tpot_ms"], (last - first) / 2e6)
                self.assertEqual(result["metrics"]["e2e_ms"],
                                 (result["done_ns"] - result["submit_ns"]) / 1e6)
                self.assertEqual(result["done_ns"], result["events"][-2]["ns"])
                self.assertEqual(result["events"][-1]["lines"], [": final heartbeat"])
                self.assertEqual(result["response_end"]["body_bytes"], len(server.payload))
                self.assertGreaterEqual(result["response_end"]["ns"], result["events"][-1]["ns"])
                server.payload = data + terminal
                server.extra = 128
                failed = client.measure(origin, case, request_body)
                self.assertIn("128 more expected", failed["error"])
                self.assertIn("done_ns", failed)
                self.assertNotIn("response_end", failed)
                self.assertNotIn("metrics", failed)
                server.extra = 0
                for tail in (terminal, frame({"error": "failed after DONE"}), b"data:\n\n"):
                    with self.subTest(tail=tail):
                        server.payload = data + terminal + tail + b": trailing heartbeat\n\n"
                        failed = client.measure(origin, case, request_body)
                        self.assertIn("Data after [DONE]", failed["error"])
                        self.assertEqual(failed["done_ns"], failed["events"][-3]["ns"])
                        self.assertEqual(failed["events"][-1]["lines"], [": trailing heartbeat"])
                        self.assertNotIn("metrics", failed)
                server.payload = data
                failed = client.measure(origin, case, request_body)
                self.assertIn("Stream ended before [DONE]", failed["error"])
                self.assertEqual(failed["output_ids"], [57, 58, 59])
                self.assertNotIn("metrics", failed)
                suffix = frame({"choices": [], "usage": usage}) + b"data: [DONE]\n\n"
                late = frame(choice([])) + frame(choice([57, 58, 59], [14, 25], "length"))
                repeated = (
                    frame(choice([], [14, 25])) + frame(choice([57, 58, 59], [14, 25], "length"))
                )
                for payload, error in (
                    (late, "First completion choice must echo prompt token IDs"),
                    (repeated, "Prompt IDs repeated"),
                ):
                    with self.subTest(error=error):
                        server.payload = payload + suffix
                        failed = client.measure(origin, case, request_body)
                        self.assertIn(error, failed["error"])
                        self.assertNotIn("metrics", failed)
            finally:
                server.shutdown()
                worker.join(timeout=2)

    def test_missing_ids_prompt_mutation_and_prefix_hits_fail(self):
        completion = client.Completion()
        with self.assertRaisesRegex(ValueError, "explicit delta"):
            completion.consume({"choices": [{"index": 0, "text": "abc"}]}, 1)
        completion.consume(choice([12], [13], "length"), 2)
        completion.consume({"choices": [], "usage": {
            "prompt_tokens": 1, "completion_tokens": 1,
            "prompt_tokens_details": {"cached_tokens": 1},
        }}, 3)
        with self.assertRaisesRegex(ValueError, "Prompt IDs absent or changed"):
            completion.validate({"prompt_token_ids": [14], "output_tokens": 1})
        with self.assertRaisesRegex(ValueError, "zero reused"):
            completion.validate({"prompt_token_ids": [13], "output_tokens": 1})

    def test_threshold_formula_and_no_gate_for_failed_grid(self):
        key = "output_tokens_per_second"
        results = [{"phase": "measured", "output_ids": [0],
                    "metrics": dict.fromkeys((key, "ttft_ms", "tpot_ms", "e2e_ms"), count)}
                   for count in range(96, 105)]
        report = client.summarize(results, True)
        self.assertEqual(report[key]["p50"], 100)
        self.assertEqual(report[key]["p95"], 103.6)
        self.assertAlmostEqual(report["thresholds"]["strict_throughput_ratio_gt"],
                               1 + 2 * (7.5 ** 0.5) / 100)
        self.assertAlmostEqual(report["thresholds"]["p95_latency_ceilings_ms"]["ttft_ms"],
                               103.6 + 2 * (7.5 ** 0.5))
        at_throughput_limit = client.compare(report, report)
        self.assertFalse(at_throughput_limit["throughput_passed"])
        candidate = {**report, key: {"p50": 110}}
        self.assertTrue(client.compare(candidate, report)["passed"])
        candidate["ttft_ms"] = {"p95": 120}
        comparison = client.compare(candidate, report)
        self.assertTrue(comparison["throughput_passed"])
        self.assertFalse(comparison["p95_latency"]["ttft_ms"]["passed"])
        self.assertFalse(comparison["passed"])
        results[0]["error"] = "short generation"
        self.assertNotIn("thresholds", client.summarize(results, True))

    def test_fixed_workload_request_body(self):
        cases = client.load_cases(DIRECTORY)
        for name, (case, _) in cases.items():
            request = json.loads(client.json_bytes(client.request_body(case)))
            self.assertEqual((len(request["prompt"]), request["max_tokens"]), client.SHAPES[name])
            self.assertEqual(request["temperature"], 0.0)
            self.assertTrue(request["ignore_eos"])
            self.assertTrue(request["return_token_ids"])

    def test_published_reference_drives_candidate_comparison(self):
        path = DIRECTORY / "gpu3-reference.json"
        workloads = json.loads(path.read_bytes())["workloads"]

        def candidate_measure(_url, case, _body):
            baseline = workloads[case["workload"]]
            metrics = dict(baseline["thresholds"]["p95_latency_ceilings_ms"])
            throughput = baseline["output_tokens_per_second"]["p50"]
            metrics["output_tokens_per_second"] = throughput * 1.02
            return {"metrics": metrics, "output_ids": baseline["greedy_output_ids"]}

        with TemporaryDirectory() as directory:
            output = Path(directory) / "candidate"
            arguments = [
                "client.py", "--url", "http://127.0.0.1:1", "--output", str(output),
                "--reference", str(path),
            ]
            with patch("sys.argv", arguments), patch.object(
                client, "measure", side_effect=candidate_measure
            ):
                self.assertEqual(client.main(), 0)
            reports = json.loads((output / "summary.json").read_bytes())["workloads"]
            for name, report in reports.items():
                self.assertEqual(report["warmups_successful"], 1)
                self.assertEqual(report["successful"], 1)
                self.assertEqual(report["greedy_output_ids"], workloads[name]["greedy_output_ids"])
                self.assertAlmostEqual(report["comparison"]["throughput_ratio"], 1.02)
                self.assertTrue(report["comparison"]["passed"])
                self.assertIsNone(report["output_tokens_per_second"]["sample_stdev"])
                self.assertNotIn("thresholds", report)
            events = (output / "events.jsonl").read_text().splitlines()
            records = [json.loads(line) for line in events]
            self.assertEqual([(row["phase"], row["workload"], row["cohort"]) for row in records],
                             [(phase, name, 1) for phase in ("warmup", "measured")
                              for name in ("W1", "W2")])


if __name__ == "__main__":
    unittest.main()
