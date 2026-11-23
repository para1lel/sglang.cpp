"""Opt-in live acceptance for native HTTP batching on a reserved B200.

Select the GPU before launching this script. Run it with a release server, the pinned
checkpoint and Nsight Systems on PATH. Use --width 2 --width 4 for the scheduler checks.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, contextmanager, ExitStack
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import select
import shutil
import signal
import socket
import sqlite3
import struct
import subprocess
import threading
import time


FIXTURES = Path(__file__).resolve().parent / "fixtures"
FIXTURE_DIGESTS = {
    "inputs.json.fixture": "20572af61c8f1fbbfef8cc3a6aede848b2cedd2e36f520819f10990a6e5bdf07",
    "serial-oracle.json.fixture":
        "b5c4ed9c72162f163e0492c954da7d26e9e6ea499b189a54bd2a7225bbb8b36b",
}
LOOPBACK = "127.0.0.1"
TRACE = re.compile(rb"^batch width=(\d+) decode=(\d+) request_bytes=(\d+)$", re.M)
ADMISSION_WINDOW = 0.75


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def load_fixtures():
    records = {}
    for filename, expected in FIXTURE_DIGESTS.items():
        data = (FIXTURES / filename).read_bytes()
        require(hashlib.sha256(data).hexdigest() == expected, f"Frozen fixture changed: {filename}")
        records[filename] = json.loads(data)
    inputs, oracle = records["inputs.json.fixture"], records["serial-oracle.json.fixture"]
    for prompt, expected in zip(inputs, oracle, strict=True):
        require(all(expected[key] == value for key, value in prompt.items()), prompt["name"])
        stop_step = prompt["stop_step"]
        output = expected["output"]
        if stop_step is None:
            require(expected["finish_reason"] == "length" and len(output) == prompt["max_tokens"],
                    f"Invalid length fixture: {prompt['name']}")
        else:
            require(expected["finish_reason"] == "stop" and
                    output.index(expected["stop_token_id"]) == stop_step and
                    len(output) == stop_step + 1 < prompt["max_tokens"],
                    f"Invalid stop fixture: {prompt['name']}")
    return {case["name"]: case for case in oracle}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def request_body(case, budget=None):
    body = {
        "model": "benchmark",
        "prompt": case["prompt"],
        "max_tokens": case["max_tokens"] if budget is None else budget,
        "temperature": 0.0,
        "seed": 0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
        "return_token_ids": True,
        "stream_interval": 1,
        "add_special_tokens": False,
    }
    if "stop_token_id" in case:
        body["stop_token_ids"] = [case["stop_token_id"]]
    return body


class Stream:
    def __init__(self, port, case, budget=None, timeout=30):
        self.connection = http.client.HTTPConnection(LOOPBACK, port, timeout=timeout)
        self.case = case
        self.request = request_body(case, budget)
        self.events = []
        self.response = None
        self.reset_sent = False

    def __enter__(self):
        self.connection.connect()
        self.sock = self.connection.sock
        return self

    def __exit__(self, *_):
        if self.response is not None:
            self.response.close()
        self.connection.close()
        self.sock.close()

    def post(self):
        self.connection.request("POST", "/v1/completions", json.dumps(self.request), {
            "Accept": "text/event-stream",
            "Content-Type": "application/json",
            "Connection": "close",
        })

    def open(self):
        self.response = self.connection.getresponse()
        if self.response.status != 200:
            raise AssertionError(f"HTTP {self.response.status}: {self.response.read()!r}")
        require(self.response.headers.get_content_type() == "text/event-stream", "Expected SSE")

    def event(self):
        line = self.response.readline()
        require(line.startswith(b"data: "), f"Expected an SSE data line: {line!r}")
        require(self.response.readline() in (b"\n", b"\r\n"), "Unterminated SSE event")
        payload = line.removeprefix(b"data: ").strip()
        event = "[DONE]" if payload == b"[DONE]" else json.loads(payload)
        self.events.append(event)
        require(not isinstance(event, dict) or "error" not in event, f"SSE error: {event!r}")
        return event

    def choice(self, ids, finish=None, first=False):
        expected = {"index": 0, "token_ids": ids, "finish_reason": finish}
        if first:
            expected["prompt_token_ids"] = self.request["prompt"]
        event = self.event()
        require(isinstance(event, dict) and event.get("choices") == [expected] and
                event.get("usage") is None,
                f"{self.case['name']}: expected {expected!r}, got {event!r}")

    def complete(self):
        self.open()
        output = self.case["output"]
        for position, token in enumerate(output):
            self.choice([token], first=position == 0)
        self.choice([], self.case["finish_reason"])
        usage_event = self.event()
        expected = {
            "prompt_tokens": len(self.request["prompt"]),
            "completion_tokens": len(output),
            "prompt_tokens_details": {"cached_tokens": 0},
        }
        require(isinstance(usage_event, dict) and usage_event.get("choices") == [] and
                usage_event.get("usage") == expected, f"Unexpected usage event: {usage_event!r}")
        require(self.event() == "[DONE]" and self.response.read(1) == b"", "Invalid SSE end")

    def reset(self):
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
        self.__exit__()
        self.reset_sent = True

    def evidence(self):
        return {"fixture": self.case["name"], "request": self.request,
                "events": self.events, "reset_sent": self.reset_sent}


def choose_port():
    with socket.socket() as listener:
        listener.bind((LOOPBACK, 0))
        return listener.getsockname()[1]


def wait_until_ready(process, port, log_path):
    deadline = time.monotonic() + 120
    ready = f"Listening on http://{LOOPBACK}:{port}\n".encode()
    while time.monotonic() < deadline:
        require(process.poll() is None, f"Server exited before readiness. See {log_path}")
        if ready in log_path.read_bytes():
            try:
                with closing(http.client.HTTPConnection(LOOPBACK, port, timeout=2)) as connection:
                    connection.request("GET", "/health")
                    with connection.getresponse() as response:
                        require(response.status == 200, f"Unhealthy: HTTP {response.status}")
                        health = json.load(response)
                        require(isinstance(health, dict), f"Invalid health response: {health!r}")
                        return health
            except (ConnectionRefusedError, TimeoutError, http.client.RemoteDisconnected):
                pass
        time.sleep(0.1)
    raise TimeoutError(f"Server readiness. See {log_path}")


def stop_server(process, pid_file):
    if process.poll() is not None:
        return
    try:
        if pid_file is None or not pid_file.exists():
            process.terminate()
        else:
            os.kill(int(pid_file.read_text()), signal.SIGTERM)
        process.wait(timeout=60)
    except ProcessLookupError:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=15)
        raise


@contextmanager
def server(args, evidence, width, profile=False, *, flags=(), expected_health=None):
    evidence.mkdir()
    port = choose_port()
    command = [str(args.binary), "--checkpoint", str(args.checkpoint), "--device",
               str(args.device), "--port", str(port), "--mtp-steps", "0"]
    if width is not None:
        command += ["--max-active", width]
    if width in ("2", "4"):
        command.append("--trace-batches")
    command.extend(flags)
    pid_file = None
    if profile:
        pid_file = evidence / "server.pid"
        command = [
            args.nsys, "profile", "--trace=cuda", "--sample=none", "--cpuctxsw=none",
            "--cuda-graph-trace=graph", "--export=sqlite", "-o", str(evidence / "graph"),
            "/bin/sh", "-c", 'printf "%s\\n" "$$" > "$1"; shift; exec "$@"',
            "sglang-profile", str(pid_file), *command,
        ]
    write_json(evidence / "command.json", command)
    log_path = evidence / "server.log"
    with log_path.open("wb") as log:
        process = subprocess.Popen(command, cwd=evidence, stdout=log, stderr=log,
                                   start_new_session=True)
        try:
            health = wait_until_ready(process, port, log_path)
            if expected_health is not None:
                write_json(evidence / "health.json", health)
                require(all(health.get(key) == value for key, value in expected_health.items()),
                        f"Expected server limits {expected_health!r}, got {health!r}")
            yield port, log_path
        finally:
            stop_server(process, pid_file)
    require(process.returncode == 0, f"Server exit {process.returncode}. See {log_path}")


def request(port, case):
    with Stream(port, case) as stream:
        stream.post()
        stream.complete()
    return stream


def serial_cases(port, cases, evidence):
    references, responses = {}, {}
    for key, case in cases.items():
        output = []
        with Stream(port, case) as stream:
            stream.post()
            stream.open()
            while True:
                event = stream.event()
                choice, = event["choices"]
                if not output:
                    require(choice["prompt_token_ids"] == case["prompt"], case["name"])
                require(choice["index"] == 0 and event.get("usage") is None, case["name"])
                ids = choice["token_ids"]
                if not ids:
                    finish = choice["finish_reason"]
                    break
                require(choice["finish_reason"] is None, case["name"])
                output.extend(ids)
                require(len(output) <= case["max_tokens"], case["name"])
            require(stream.event() == {"choices": [], "usage": {
                "prompt_tokens": len(case["prompt"]),
                "completion_tokens": len(output),
                "prompt_tokens_details": {"cached_tokens": 0},
            }}, case["name"])
            require(stream.event() == "[DONE]" and stream.response.read(1) == b"", case["name"])
        require(0 < len(output) <= case["max_tokens"]
                and ((finish == "length" and len(output) == case["max_tokens"])
                     or (finish == "stop" and output[-1] == case.get("stop_token_id"))),
                case["name"])
        references[key] = {**case, "output": output, "finish_reason": finish}
        responses[key] = stream.evidence()
    write_json(evidence, responses)
    return references


def trace_rows(log_path, offset, capacity):
    with log_path.open("rb") as log:
        log.seek(offset)
        rows = [tuple(map(int, row)) for row in TRACE.findall(log.read())]
    require(rows, f"No scheduler trace since byte {offset} in {log_path}")
    require(all(0 <= decode <= width <= capacity for width, decode, _ in rows),
            f"Scheduler exceeded width {capacity}: {rows!r}")
    return rows


def record_batch(path, streams, rows):
    write_json(path, {
        "streams": [stream.evidence() for stream in streams],
        "scheduler_rows": [{"width": width, "decode": decode, "request_bytes": size}
                           for width, decode, size in rows],
        "max_width": max(width for width, _, _ in rows),
        "max_decode": max(decode for _, decode, _ in rows),
    })


def burst(port, log_path, evidence, cases, full_decode):
    capacity = len(cases)
    offset = log_path.stat().st_size
    barrier = threading.Barrier(capacity)

    def client(case):
        with Stream(port, case) as stream:
            barrier.wait(timeout=10)
            stream.post()
            stream.complete()
        return stream

    with ThreadPoolExecutor(max_workers=capacity) as pool:
        streams = list(pool.map(client, cases))
    rows = trace_rows(log_path, offset, capacity)
    record_batch(evidence, streams, rows)
    require(any(width == capacity for width, _, _ in rows), f"Missing width {capacity}: {evidence}")
    if full_decode:
        require(any(decode == capacity for _, decode, _ in rows),
                f"No overlapping decode at width {capacity}: {evidence}")


def await_retirement(log_path, offset, capacity):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        with log_path.open("rb") as log:
            log.seek(offset)
            matches = TRACE.findall(log.read())
        if matches and int(matches[-1][2]) == 0:
            return trace_rows(log_path, offset, capacity)[-1]
        time.sleep(0.01)
    raise AssertionError(f"Request allocations did not return to zero. See {log_path}")


def check_admission(port, log_path, evidence, cases, capacity):
    active, queued_case = cases["held-out"], cases["numbers"]
    offset = log_path.stat().st_size
    with ExitStack() as stack:
        blockers = [stack.enter_context(Stream(port, active, budget=1024))
                    for _ in range(capacity)]
        for blocker in blockers:
            blocker.post()
        for blocker in blockers:
            blocker.open()
            blocker.choice(active["output"][:1], first=True)
            blocker.choice(active["output"][1:2])
        queued = stack.enter_context(Stream(port, queued_case))
        queued.post()
        require(not select.select([queued.sock], [], [], ADMISSION_WINDOW)[0],
                f"An extra stream responded with {capacity} active decoders")
        for blocker in blockers:
            blocker.choice(active["output"][2:3])
        retirement_offset = log_path.stat().st_size
        blockers[0].reset()
        queued.complete()
        for blocker in blockers[1:]:
            blocker.choice(active["output"][3:4])
            blocker.reset()
    retired = await_retirement(log_path, retirement_offset, capacity)
    replacement = request(port, active)
    record_batch(evidence, [*blockers, queued, replacement],
                 trace_rows(log_path, offset, capacity))
    write_json(evidence.with_stem("admission"), {
        "queued_readability_timeout_seconds": ADMISSION_WINDOW,
        "request_retirement": retired,
    })


def check_singleton_graph(args, evidence, width, cases):
    flags = ("--batch-cuda-graphs", "--trace-batches")
    with server(args, evidence, width, profile=True, flags=flags) as (port, log):
        cases = serial_cases(port, cases, evidence / "serial.json")
        offset = log.stat().st_size
        streams = [request(port, case).evidence() for case in cases.values()]
        write_json(evidence / "serial-oracle.json", streams)
    trace = log.read_bytes()
    traced = trace[offset:].count(b"batch graph_slots=1\n")
    total_traced = trace.count(b"batch graph_slots=1\n")
    database = evidence / "graph.sqlite"
    with closing(sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True)) as connection:
        launches, = connection.execute("""
            SELECT count(*)
            FROM CUPTI_ACTIVITY_KIND_RUNTIME AS api
            JOIN StringIds AS names ON api.nameId = names.id
            WHERE names.value LIKE 'cudaGraphLaunch%'
        """).fetchone()
    expected = sum(len(case["output"]) - 1 for case in cases.values())
    write_json(evidence / "graph-launches.json", {
        "expected_streaming": expected, "traced_streaming": traced,
        "observed_total": launches, "traced_total": total_traced,
    })
    require(launches == total_traced and traced == expected,
            f"{width}: expected {expected} launches, got {launches} with {traced} traced")


def check_parallel(args, evidence, width, cases):
    mixed = ["numbers", "code"] if width == "2" else ["facts", "numbers", "code", "facts"]
    page = ["page-boundary", "held-out", "colors", "facts"][:int(width)]
    with server(args, evidence, width) as (port, log):
        cases = serial_cases(port, cases, evidence / "serial.json")
        burst(port, log, evidence / "mixed-stop.json", [cases[name] for name in mixed],
              full_decode=True)
        burst(port, log, evidence / "mixed-page.json", [cases[name] for name in page],
              full_decode=False)
        check_admission(port, log, evidence / "cancel-and-reuse.json", cases, int(width))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True, help="release sglang-server")
    parser.add_argument("--checkpoint", type=Path, default=Path("/data0/shared/Qwen3.8-27B-FP8"))
    parser.add_argument("--device", type=int, default=0, help="CUDA ordinal within visible GPUs")
    parser.add_argument("--evidence", type=Path, required=True,
                        help="new directory for raw results")
    parser.add_argument("--width", action="append", choices=("1", "2", "4"),
                        help="repeat to select modes, defaults to all three")
    parser.add_argument("--nsys", default="nsys", help="Nsight Systems CLI for width-one graphs")
    args = parser.parse_args()
    args.binary = args.binary.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.evidence = args.evidence.resolve()
    widths = tuple(dict.fromkeys(args.width or ("1", "2", "4")))
    if "1" in widths:
        nsys = shutil.which(args.nsys)
        if not nsys:
            parser.error("Nsight Systems is required for explicit width-1 graph checks")
        args.nsys = str(Path(nsys).absolute())
    cases = load_fixtures()
    args.evidence.mkdir(parents=True)
    with args.binary.open("rb") as binary:
        write_json(args.evidence / "identity.json", {
            "binary_sha256": hashlib.file_digest(binary, "sha256").hexdigest(),
            "fixture_sha256": FIXTURE_DIGESTS,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "widths": widths,
        })
    for width in widths:
        print(f"Checking native serving width {width}", flush=True)
        evidence = args.evidence / f"width-{width}"
        if width == "1":
            check_singleton_graph(args, evidence, width, cases)
        else:
            check_parallel(args, evidence, width, cases)
    write_json(args.evidence / "status.json", {"result": "passed"})


if __name__ == "__main__":
    main()
