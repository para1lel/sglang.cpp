"""Check generation health through the production HTTP queue and CPU decoder double."""

from contextlib import closing
import http.client
from pathlib import Path
import select
import socket
from tempfile import mkdtemp
import time
import unittest
from unittest.mock import patch

from test_native_generate_cpu import ABORTED, abort, batch_frame, generate_client, get, named
from test_sampling_routes_cpu import post_response, receipts, run_cpu_suite, send, server


HEALTH = "/health_generate"
HEALTH_ADMIT = "cpu health prompt=0"
HEALTH_TOKEN = "cpu health token"
HEALTH_CANCEL = "cpu cancel slot=0 prompt=1 budget=1"


class GenerationHealth(unittest.TestCase):
    def setUp(self):
        self.directory = Path(mkdtemp(
            prefix=f"{self._testMethodName}-", dir=self.output,
        ))
        self.log = self.directory / "server.log"
        self.clock = self.directory / "clock"
        self.set_clock(1)
        self.gate = self.directory / "step"
        self.enterContext(patch.dict("os.environ", {
            "SGLANG_TEST_STREAM_CLOCK_FILE": str(self.clock),
            "SGLANG_TEST_BATCH_STEP_GATE": str(self.gate),
            "SGLANG_TEST_HEALTH_CONTROL": str(self.directory),
        }))
        self.port, self.process = self.enterContext(server(
            self.binary, self.checkpoint, self.log,
            "--max-active", "1", with_process=True,
        ))

    def set_clock(self, seconds):
        update = self.directory / "clock.next"
        update.write_text(f"{seconds}\n")
        update.replace(self.clock)

    def mark(self, name):
        (self.directory / name).touch()

    def wait_receipt(self, text, count=1):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            output = self.log.read_text()
            if output.count(text) >= count:
                return output
            time.sleep(0.01)
        self.fail(f"Expected {count} receipts for {text!r}: {output}")

    def connection(self, path=HEALTH):
        peer = self.enterContext(closing(socket.create_connection(
            ("127.0.0.1", self.port), timeout=4,
        )))
        peer.sendall(f"GET {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n"
                     .encode())
        return peer

    def check_probe(self, peer, status):
        with http.client.HTTPResponse(peer) as response:
            response.begin()
            self.assertEqual(response.status, status)
            self.assertEqual(response.getheader("Content-Length"), "0")
            self.assertEqual(response.getheader("Connection"), "close")
            self.assertEqual(response.read(), b"")

    def expect_pending(self, *peers):
        readable, _, _ = select.select(peers, [], [], 0.12)
        self.assertEqual(readable, [], self.log.read_text())

    def check_controls(self):
        self.assertEqual(get(self.port, "/health")[0], 200)
        self.assertEqual(get(self.port, "/ready"), (200, b'{"ready":true}'))

    def check_reuse(self):
        status, response = send(self.port, "/generate", named(
            "reused-lane", budget=1, stream=False, seed=7,
        ))
        self.assertEqual(status, 200, response)
        self.assertEqual(len(response["output_ids"]), 1)

    def check_idle_reset(self):
        status, response = send(self.port, "/reset_prefix_cache", {})
        self.assertEqual(status, 503, response)
        self.assertEqual(response["error"]["message"],
                         "LMCache namespace reset was not acknowledged")

    def test_idle_probe_executes_one_greedy_token(self):
        probe = self.connection()
        self.wait_receipt(HEALTH_ADMIT)
        self.expect_pending(probe)
        self.check_controls()
        self.assertEqual(send(self.port, "/reset_prefix_cache", {})[0], 409)
        before = get(self.port, "/metrics")
        self.assertEqual(before[0], 200)
        self.mark("release")
        self.check_probe(probe, 200)
        self.assertEqual(get(self.port, "/metrics"), before)
        receipt, = receipts(self.log)
        self.assertEqual(receipt["prompt_tokens"], 1)
        self.assertEqual(receipt["budget"], 1)
        self.assertIs(receipt["sampled"], False)
        self.assertEqual(self.log.read_text().count(HEALTH_TOKEN), 1)
        self.check_idle_reset()

    def test_idle_terminal_without_token_is_unhealthy(self):
        self.mark("finish-empty")
        self.mark("release")
        self.check_probe(self.connection(), 503)
        self.wait_receipt(HEALTH_ADMIT)
        self.assertNotIn(HEALTH_TOKEN, self.log.read_text())
        self.check_controls()
        self.check_reuse()

    def test_admission_failure_is_unhealthy_and_recoverable(self):
        self.mark("reject")
        self.check_probe(self.connection(), 503)
        self.wait_receipt(HEALTH_ADMIT)
        self.assertNotIn(HEALTH_TOKEN, self.log.read_text())
        self.check_controls()
        self.check_reuse()

    def open_generation(self, rid, budget):
        payload = named(rid, budget=budget)
        payload["sampling_params"]["ignore_eos"] = True
        client = self.enterContext(closing(generate_client(self.port, payload)))
        response = self.enterContext(http.client.HTTPResponse(client))
        response.begin()
        self.assertEqual(response.status, 200)
        return response

    def check_generation_length(self, generation, length):
        final = batch_frame(generation)
        self.assertEqual(len(final["output_ids"]), length)
        self.assertEqual(final["meta_info"]["finish_reason"], {"type": "length", "length": length})
        self.assertEqual(batch_frame(generation), "[DONE]")

    def test_busy_probe_requires_fresh_native_progress(self):
        self.mark("step.0")
        generation = self.open_generation("busy", budget=3)
        first = batch_frame(generation)
        self.assertEqual(len(first["output_ids"]), 1)
        self.assertIsNone(first["meta_info"]["finish_reason"])
        self.set_clock(2)
        first_probe = self.connection()
        self.expect_pending(first_probe)
        queued_probe = self.connection()
        self.expect_pending(queued_probe)
        self.check_probe(self.connection(), 503)
        self.check_controls()
        self.set_clock(3)
        self.mark("step.1")
        self.check_probe(first_probe, 200)
        self.check_probe(queued_probe, 200)
        second = batch_frame(generation)
        self.assertEqual(len(second["output_ids"]), 2)
        self.assertIsNone(second["meta_info"]["finish_reason"])
        self.assertNotIn("cpu finish", self.log.read_text())
        self.mark("step.2")
        self.check_generation_length(generation, 3)
        self.assertNotIn(HEALTH_ADMIT, self.log.read_text())

    def test_saturated_probes_expire_from_their_own_receipt_times(self):
        generation = self.open_generation("saturated", budget=1)
        self.wait_receipt("cpu admit slot=0 prompt=1 budget=1")
        self.set_clock(2)
        active = self.connection()
        self.expect_pending(active)
        self.set_clock(10)
        waiting = self.connection()
        self.expect_pending(waiting)
        self.check_probe(self.connection(), 503)
        self.set_clock(21)
        self.expect_pending(active, waiting)
        self.set_clock(22)
        self.check_probe(active, 503)
        self.set_clock(29)
        self.expect_pending(waiting)
        self.set_clock(30)
        self.check_probe(waiting, 503)
        self.assertEqual(batch_frame(generation), "keepalive")
        self.assertNotIn("cpu finish", self.log.read_text())
        self.mark("step.0")
        self.check_generation_length(generation, 1)
        self.assertNotIn(HEALTH_ADMIT, self.log.read_text())

    def test_timeout_retires_idle_probe_and_releases_lane(self):
        probe = self.connection()
        self.wait_receipt(HEALTH_ADMIT)
        self.set_clock(22)
        self.check_probe(probe, 503)
        self.wait_receipt(HEALTH_CANCEL)
        self.assertNotIn(HEALTH_TOKEN, self.log.read_text())
        self.check_idle_reset()
        self.check_reuse()

    def test_failed_worker_rejects_new_and_waiting_probes(self):
        active = self.connection()
        self.wait_receipt(HEALTH_ADMIT)
        waiting = self.connection()
        self.expect_pending(waiting)
        self.mark("fail")
        self.mark("release")
        self.check_probe(active, 503)
        self.check_probe(waiting, 503)
        self.check_probe(self.connection(), 503)
        self.assertEqual(get(self.port, "/health")[0], 200)
        self.assertEqual(get(self.port, "/ready"),
                         (503, b'{"ready":false,"reason":"worker_failed"}'))
        self.assertEqual(self.log.read_text().count(HEALTH_ADMIT), 1)
        self.assertNotIn(HEALTH_TOKEN, self.log.read_text())

    def test_abort_responsive_while_busy_probe_waits(self):
        generation = self.open_generation("abort-health", budget=3)
        self.wait_receipt("cpu admit slot=0 prompt=1 budget=3")
        probe = self.connection()
        self.expect_pending(probe)
        self.check_controls()
        with post_response(self.port, "/abort_request", {"rid": "abort-health"}) as response:
            self.assertEqual(response.status, 200)
            self.assertEqual(response.read(), b"")
        packet = batch_frame(generation)
        self.assertEqual(packet["output_ids"], [])
        self.assertEqual(packet["meta_info"]["finish_reason"], ABORTED)
        self.assertEqual(batch_frame(generation), "[DONE]")
        self.wait_receipt("cpu cancel slot=0 prompt=1 budget=3")
        self.wait_receipt(HEALTH_ADMIT)
        self.expect_pending(probe)
        self.set_clock(2)
        self.mark("release")
        self.check_probe(probe, 200)
        self.assertEqual(self.log.read_text().count(HEALTH_TOKEN), 1)

    def test_concurrent_probe_disconnect_and_queue_reuse(self):
        active = self.connection()
        self.wait_receipt(HEALTH_ADMIT)
        waiting = self.connection()
        self.expect_pending(waiting)
        self.check_probe(self.connection(), 503)
        abort(waiting)
        replacement = self.connection()
        self.expect_pending(replacement)
        abort(active)
        self.wait_receipt(HEALTH_CANCEL)
        self.wait_receipt(HEALTH_ADMIT, count=2)
        self.expect_pending(replacement)
        self.mark("release")
        self.check_probe(replacement, 200)
        self.assertEqual(self.log.read_text().count(HEALTH_ADMIT), 2)
        self.assertEqual(self.log.read_text().count(HEALTH_TOKEN), 1)
        self.check_reuse()

    def test_shutdown_releases_active_and_queued_probes(self):
        active = self.connection()
        self.wait_receipt(HEALTH_ADMIT)
        waiting = self.connection()
        self.expect_pending(waiting)
        self.check_probe(self.connection(), 503)
        self.process.terminate()
        self.check_probe(active, 503)
        self.check_probe(waiting, 503)
        self.assertEqual(self.process.wait(timeout=4), 0, self.log.read_text())
        self.assertEqual(self.log.read_text().count(HEALTH_ADMIT), 1)
        self.assertNotIn(HEALTH_TOKEN, self.log.read_text())

    def test_probe_accepts_only_empty_get(self):
        for method, payload, status in (
            ("POST", b"", 405), ("HEAD", b"", 405), ("GET", b"x", 404),
        ):
            with self.subTest(method=method), closing(http.client.HTTPConnection(
                "127.0.0.1", self.port, timeout=4,
            )) as connection:
                connection.request(method, HEALTH, body=payload)
                with connection.getresponse() as response:
                    self.assertEqual(response.status, status)
                    if status == 405:
                        self.assertEqual(response.getheader("Allow"), "GET")
                    response.read()
        self.assertNotIn(HEALTH_ADMIT, self.log.read_text())


if __name__ == "__main__":
    run_cpu_suite(GenerationHealth, __file__, __doc__)
