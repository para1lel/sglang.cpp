import argparse
from contextlib import closing, contextmanager
from http.client import HTTPConnection
import json
import os
from pathlib import Path
import socket
import statistics
import struct
import subprocess
import time
from urllib.parse import urlsplit


from benchmarks.streaming.client import measure, request_body


ROOT = Path(__file__).resolve().parent


ARMS = ("serial", "base", "candidate")
HEADERS = {"Content-Type": "application/json", "Connection": "close"}


def save(destination, data):
    destination.write_text(json.dumps(data, indent=2) + "\n")


@contextmanager
def server(arm, depth, folder):
    with socket.socket() as port_source:
        port_source.bind(("127.0.0.1", 0))
        port = port_source.getsockname()[1]
    builds = Path(os.environ["MTP_BUILDS"])
    names = {"serial": "mtp-serial-control", "base": "mtp-parallel-control",
             "candidate": "mtp-adaptive"}
    binary = builds / names[arm] / "out/bin/sglang-server"
    with (folder / "server.log").open("wb") as log:
        process = subprocess.Popen([binary, "--port", str(port), "--max-active", "1",
                                    "--mtp-steps", str(depth), "--checkpoint",
                                    os.environ["CHECKPOINT"]], stdout=log, stderr=log)
        try:
            deadline = time.monotonic() + 90
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"{binary} exited before health")
                with closing(HTTPConnection("127.0.0.1", port, timeout=1)) as connection:
                    try:
                        connection.request("GET", "/health")
                        response = connection.getresponse()
                        if response.status == 200 and response.read() == b"{}":
                            break
                    except (ConnectionError, TimeoutError):
                        time.sleep(0.2)
            else:
                raise RuntimeError(f"{binary} health timeout")
            yield port
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def exchange(port, body, cut=None):
    ids, reason, usage, done = [], None, None, False
    with closing(HTTPConnection("127.0.0.1", port, timeout=90)) as connection:
        connection.connect()
        transport = connection.sock
        connection.request("POST", "/v1/completions", json.dumps(body), HEADERS)
        with connection.getresponse() as response:
            assert response.status == 200, response.read()
            for line in response:
                if not line.startswith(b"data: "):
                    continue
                payload = line[6:].strip()
                if payload == b"[DONE]":
                    done = True
                    break
                event = json.loads(payload)
                assert "error" not in event, event
                if not event["choices"]:
                    usage = event["usage"]
                for choice in event["choices"]:
                    ids.extend(choice.get("token_ids") or ())
                    reason = choice.get("finish_reason") or reason
                if cut is not None and len(ids) >= cut:
                    transport.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER,
                                         struct.pack("ii", 1, 0))
                    return {"ids": ids, "cancelled": True}
    assert done and reason and usage is not None
    return {"ids": ids, "reason": reason, "usage": usage}


def controls(port, case, expected):
    body = request_body(case)
    stop_index = next(index for index in range(2, len(expected))
                      if expected[index] not in expected[:index])
    stopped = exchange(port, {**body, "stop_token_ids": [expected[stop_index]]})
    assert stopped["ids"] == expected[:stop_index + 1] and stopped["reason"] == "stop"
    cancelled = exchange(port, {**body, "max_tokens": 4096}, cut=5)
    assert cancelled["ids"] == expected[:5]
    recovered = exchange(port, body)
    assert recovered["ids"] == expected and recovered["reason"] == "length"
    return {"stopped": stopped, "cancelled": cancelled, "recovered": recovered}


def block(run, cohort, depth, order, cases, reference, repeats):
    for arm in order:
        folder = run / f"http-{cohort}-K{depth}-{arm}"
        folder.mkdir()
        with server(arm, depth, folder) as port:
            address = urlsplit(f"http://127.0.0.1:{port}")
            samples = {name: [] for name in cases}
            for repeat in range(-1, repeats):
                for name, case in cases.items():
                    body = json.dumps(request_body(case)).encode()
                    result = measure(address, case, body)
                    save(folder / f"{name}-{repeat}.json", result)
                    assert "error" not in result, result.get("error")
                    output = result["output_ids"]
                    assert reference.setdefault(name, output) == output, (arm, depth, name)
                    if repeat >= 0:
                        samples[name].append(result["metrics"])
            if depth in (0, 4):
                save(folder / "controls.json", controls(port, cases["W1"], reference["W1"]))
        for name, values in samples.items():
            medians = {metric: statistics.median(row[metric] for row in values)
                       for metric in ("ttft_ms", "tpot_ms", "e2e_ms")}
            print(cohort, arm, depth, name, medians, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--depths", nargs="+", type=int, default=[0, 4, 1, 2, 3])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--reverse", nargs="+", type=int, default=[])
    parser.add_argument("--fixtures", type=Path, default=ROOT.parent / "recipes/mtp-serving.json")
    args = parser.parse_args()
    run = Path(os.environ["ADAPTIVE_TEST_RUN"])
    cases = json.loads(args.fixtures.read_text())
    save(run / "fixtures.json", cases)
    reference = {}
    for index, depth in enumerate(args.depths):
        order = ARMS[index % 3:] + ARMS[:index % 3]
        block(run, 0, depth, order, cases, reference, args.repeats)
        if depth in args.reverse:
            block(run, 1, depth, tuple(reversed(order)), cases, reference, args.repeats)
    save(run / "greedy.json", reference)


if __name__ == "__main__":
    main()
