"""Run native chat checks through the server executable and record the evidence."""

import argparse
import codecs
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, contextmanager
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import socket
import struct
import subprocess
import sys
from threading import Barrier, Condition
import time

from test_native_chat import (
    CHAT, HEADERS, MODEL, RAW, NativeChatTest, chat_body, chat_json, chat_sse, event, fixtures,
    json_response, metadata, one_choice, raw_body, tokenizer_pieces, usage,
)


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "batch"))
from check_graph_evidence import LIFECYCLE, count, trace


STEP = re.compile(r"batch width=(\d+) decode=(\d+) request_bytes=(\d+)")
MASK = re.compile(r"batch graph_slots=(\d+)")
CPU_SLOT = re.compile(r"cpu (admit|cancel) slot=(\d+) prompt=(\d+) budget=(\d+)")


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


@contextmanager
def server(binary, checkpoint, out, width, *, graph=False, mtp=False, observer=None):
    with closing(socket.socket()) as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    out.mkdir()
    command = [
        str(binary), "--checkpoint", str(checkpoint), "--port", str(port),
        "--max-active", str(width), "--trace-batches",
    ]
    if graph:
        command.append("--batch-cuda-graphs")
    if mtp:
        command.extend(("--mtp-steps", "4"))
    environment = os.environ.copy()
    if observer:
        environment["LD_PRELOAD"] = str(observer)
        environment["BATCH_GRAPH_ACTIVITY"] = str(out / "activity.tsv")
    with (out / "server.log").open("wb") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment)
        try:
            deadline = time.monotonic() + 90
            while True:
                assert process.poll() is None, (process.returncode, out)
                try:
                    with closing(http.client.HTTPConnection("127.0.0.1", port, timeout=1)) as conn:
                        conn.request("GET", "/health")
                        with conn.getresponse() as reply:
                            if (reply.status, reply.read()) == (200, b"{}"):
                                break
                except (OSError, http.client.HTTPException):
                    pass
                assert time.monotonic() < deadline, out
                time.sleep(0.1)
            proc = Path(f"/proc/{process.pid}")
            sha = digest(binary)
            assert digest(proc / "exe") == sha
            maps = (proc / "maps").read_text()
            foreign = ("libpython", "libtorch", "libtokenizers", "libonig")
            assert not any(lib in maps for lib in foreign)
            (out / "maps.txt").write_text(maps)
            save(out / "launch.json", {
                "command": command, "sha256": sha, "pid": process.pid,
                "device": environment.get("CUDA_VISIBLE_DEVICES"),
                "observer_sha256": digest(observer) if observer else None,
            })
            yield port
            assert process.poll() is None, process.returncode
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=12)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            (out / "server-exit.txt").write_text(f"{process.returncode}\n")
    assert process.returncode == 0, (process.returncode, out)


def legacy_body(case, budget):
    return {
        "model": "benchmark", "prompt": case["ids"], "stream": True, "temperature": 0,
        "max_tokens": budget, "ignore_eos": True, "return_token_ids": True,
        "stream_interval": 1, "add_special_tokens": False, "seed": 0,
        "stream_options": {"include_usage": True},
    }


def legacy(port, case, budget, on_token=None, stop=None):
    body = legacy_body(case, budget)
    if stop is not None:
        body["stop_token_ids"] = [stop]
    with closing(http.client.HTTPConnection("127.0.0.1", port, timeout=45)) as conn:
        conn.request("POST", RAW, json.dumps(body), HEADERS)
        with conn.getresponse() as reply:
            assert reply.status == 200 and reply.headers.get_content_type() == "text/event-stream"
            frames = []
            while True:
                frame = event(reply)
                if frame == "[DONE]":
                    break
                frames.append(frame)
                if frame["choices"] and one_choice(frame)["token_ids"] and on_token:
                    on_token()
            assert reply.read() == b""
    choices = [one_choice(frame) for frame in frames[:-1]]
    assert choices[0]["prompt_token_ids"] == case["ids"]
    assert all("prompt_token_ids" not in choice for choice in choices[1:])
    ids = [token for choice in choices for token in choice["token_ids"]]
    assert all(choice["finish_reason"] is None for choice in choices[:-1])
    assert choices[-1]["token_ids"] == []
    assert frames[-1] == {"choices": [], "usage": {
        "prompt_tokens": len(case["ids"]), "completion_tokens": len(ids),
        "prompt_tokens_details": {"cached_tokens": 0},
    }}
    return ids, choices[-1]["finish_reason"], frames


def baseline(port, case, budget):
    _, ids, text, finish = chat_json(port, chat_body(case, budget), case["ids"])
    raw = json_response(port, RAW, raw_body(case, budget))
    choice = one_choice(raw)
    assert choice["prompt_token_ids"] == case["ids"]
    assert (ids, text, finish) == (choice["token_ids"], choice["text"], choice["finish_reason"])
    usage(raw, len(case["ids"]), len(ids))
    assert len(ids) == budget and finish == "length"
    return ids, text, finish


def batch_trace(out):
    steps, masks = [], []
    for line in (out / "server.log").read_text().splitlines():
        if match := STEP.fullmatch(line):
            steps.append(tuple(map(int, match.groups())))
        elif match := MASK.fullmatch(line):
            mask = int(match[1])
            assert steps and 0 < mask < 16 and mask.bit_count() == steps[-1][1], line
            masks.append(mask)
    return steps, masks


def checkpoint(out):
    steps, masks = batch_trace(out)
    return len(steps), len(masks)


def mixed(port, out, selected, width):
    requests = list(zip(
        ("chat-sse", "legacy", "raw-json", "chat-json"),
        selected, (26, 23, 20, 17), strict=True,
    ))[:width]
    wanted = [baseline(port, case, budget) for _, case, budget in requests]
    assert len({len(case["ids"]) for _, case, _ in requests}) == width
    barrier = Barrier(width, timeout=10)

    def lane(request, reference):
        route, case, budget = request
        barrier.wait()
        if route == "chat-sse":
            body = chat_body(case, budget) | {"stream_options": {"include_usage": True}}
            actual = chat_sse(port, body, case["ids"])[1:]
        elif route == "legacy":
            ids, finish, _ = legacy(port, case, budget)
            assert (ids, finish) == (reference[0], reference[2])
            actual = ids, None, finish
        elif route == "raw-json":
            packet = json_response(port, RAW, raw_body(case, budget))
            choice = one_choice(packet)
            assert choice["prompt_token_ids"] == case["ids"]
            usage(packet, len(case["ids"]), budget)
            actual = choice["token_ids"], choice["text"], choice["finish_reason"]
        else:
            actual = chat_json(port, chat_body(case, budget), case["ids"])[1:]
        if route != "legacy":
            assert actual == reference, (route, actual, reference)
        return {"route": route, "case": case["name"], "prompt_ids": case["ids"],
                "reference": reference, "observed": actual}

    begin = checkpoint(out)
    with ThreadPoolExecutor(max_workers=width) as pool:
        results = list(pool.map(lane, requests, wanted))
    return {"begin": begin, "end": checkpoint(out), "lanes": results}


def cancel(port, case, reference, frame_count, before_reset):
    body = chat_body(case, 4096) | {"stream": True}
    with closing(http.client.HTTPConnection("127.0.0.1", port, timeout=45)) as conn:
        conn.connect()
        with conn.sock.dup() as control:
            conn.request("POST", CHAT, json.dumps(body), HEADERS)
            with conn.getresponse() as reply:
                assert reply.status == 200
                assert reply.headers.get_content_type() == "text/event-stream"
                role = event(reply)
                identity = metadata(role, "chat.completion.chunk")
                assert one_choice(role)["delta"] == {"role": "assistant"}
                observed = []
                for index in range(frame_count):
                    packet = event(reply)
                    assert metadata(packet, "chat.completion.chunk") == identity
                    choice = one_choice(packet)
                    assert choice["finish_reason"] is None
                    assert choice["token_ids"] == reference[index:index + 1]
                    if index == 0:
                        assert choice["prompt_token_ids"] == case["ids"]
                    observed.append(choice)
                mark = before_reset()
                control.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
    return mark, observed


def churn(port, out, selected, graph):
    case, *peers = selected
    reference = baseline(port, case, 17)
    expected = [legacy(port, peer, 96) for peer in peers]
    barrier = Barrier(4, timeout=10)
    changed = Condition()
    ticks = [[] for _ in peers]

    def stream_peer(index):
        def received():
            with changed:
                ticks[index].append(time.monotonic_ns())
                changed.notify_all()

        barrier.wait()
        return legacy(port, peers[index], 96, on_token=received)

    with ThreadPoolExecutor(max_workers=3) as pool:
        pending = [pool.submit(stream_peer, index) for index in range(len(peers))]
        begin = checkpoint(out)
        barrier.wait()
        frames = 13 if graph else 15
        mark, prefix = cancel(port, case, reference[0], frames, lambda: checkpoint(out))
        reset = time.monotonic_ns()
        with changed:
            assert changed.wait_for(
                lambda: all(sum(stamp > reset for stamp in row) >= 3 for row in ticks),
                timeout=10,
            ), ticks
        assert all(not future.done() for future in pending), "Peer exited before readmission"
        readmit_start_ns = time.monotonic_ns()
        readmit = checkpoint(out)
        assert chat_sse(port, chat_body(case, 17), case["ids"])[1:] == reference
        complete = checkpoint(out)
        readmit_end_ns = time.monotonic_ns()
        assert [future.result() for future in pending] == expected
        assert all(row[-1] > readmit_end_ns for row in ticks), "Peer exited during readmission"
    if graph:
        _, masks = batch_trace(out)
        before = set(m for m in masks[begin[1]:mark[1]] if m.bit_count() == 4)
        after = set(m for m in masks[readmit[1]:complete[1]] if m.bit_count() == 4)
        assert before == after == {15}, (before, after)
    return {"begin": begin, "before_reset": mark, "reset_ns": reset,
            "readmit": readmit, "readmit_complete": complete, "prefix": prefix,
            "readmit_start_ns": readmit_start_ns, "readmit_end_ns": readmit_end_ns,
            "peer_ticks": ticks, "peer_output_ids": [entry[0] for entry in expected],
            "replacement": reference}


def verify_cpu_cancellation(out, cancellation, pieces):
    prefix = cancellation["prefix"]
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    for choice in prefix:
        assert choice["delta"]["content"] == decoder.decode(pieces[choice["token_ids"][0]])
    pending = decoder.getstate()[0]
    assert pending, "Expected a partial UTF-8 scalar at TCP reset"
    operations = [
        (match[1], *map(int, match.groups()[1:]))
        for line in (out / "server.log").read_text().splitlines()
        if (match := CPU_SLOT.fullmatch(line)) and int(match[3]) == 17
    ]
    [index] = [i for i, entry in enumerate(operations)
               if entry[0] == "admit" and entry[2:] == (17, 4096)]
    _, slot, _, _ = operations[index]
    assert operations[index:index + 3] == [
        ("admit", slot, 17, 4096), ("cancel", slot, 17, 4096), ("admit", slot, 17, 17),
    ]
    return {"slot": slot, "pending_utf8_hex": pending.hex()}


def raw_wires(port, out, case):
    request = raw_body(case, 7)
    stream = request | {"stream": True, "stream_options": {"include_usage": True}}
    benchmark = legacy_body(case, 7)
    controls = {
        "raw": request, "raw-stream": stream, "raw-zero": stream | {"max_tokens": 0},
        "raw-error": request | {"prompt": []},
        "raw-options-error": request | {"stream_options": {"include_usage": True}},
        "legacy": benchmark, "legacy-zero": benchmark | {"max_tokens": 0},
        "legacy-stop": benchmark | {"stop_token_ids": [198]},
        "legacy-error": benchmark | {"prompt": []},
        "legacy-options-error": benchmark | {"stream_options": {"include_usage": False}},
        "malformed": b'{"model":',
    }
    envelope = {}
    for name, body in controls.items():
        wire = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
        with closing(http.client.HTTPConnection("127.0.0.1", port, timeout=45)) as conn:
            conn.request("POST", RAW, wire, HEADERS)
            with conn.getresponse() as reply:
                payload = reply.read()
                status = 400 if "error" in name or name == "malformed" else 200
                assert reply.status == status, (name, reply.status, payload)
                envelope[name] = [reply.status, reply.getheader("Content-Type")]
                assert payload
                (out / f"{name}.wire").write_bytes(payload)
    save(out / "raw-envelopes.json", envelope)
    return envelope


def verify_zero(port, case):
    response = chat_json(port, chat_body(case, 0), case["ids"])[1:]
    assert response == ([], "", "length")
    frames = []
    body = chat_body(case, 0) | {"stream_options": {"include_usage": True}}
    assert chat_sse(port, body, case["ids"], frames=frames)[1:] == response
    assert baseline(port, case, 17)[1]
    return frames


def verify_graph(out, cohorts, replacement):
    steps, masks = batch_trace(out)
    gpu, runtime = trace(out / "activity.tsv")
    executions = list(zip(masks, gpu, strict=True))
    assert steps and steps[-1][2] == 0
    by_width = defaultdict(set)
    for mask, graph_id in executions:
        by_width[mask.bit_count()].add(graph_id)
    assert set(by_width) == {2, 3, 4}
    assert all(len(ids) == 1 for ids in by_width.values())
    assert len(set.union(*by_width.values())) == 3
    assert all(count(runtime, name) == 3 for name in LIFECYCLE)
    observed = {}
    for width, cohort in cohorts.items():
        start, end = cohort["begin"][1], cohort["end"][1]
        matching = [(mask, gid) for mask, gid in executions[start:end] if mask.bit_count() == width]
        assert len(matching) >= 2, (width, matching)
        observed[width] = matching
    before, after = replacement["begin"][1], replacement["before_reset"][1]
    start, end = replacement["readmit"][1], replacement["readmit_complete"][1]
    removed = [(mask, gid) for mask, gid in executions[before:after] if mask == 15]
    readmitted = [(mask, gid) for mask, gid in executions[start:end] if mask == 15]
    assert removed and readmitted and set(removed) == set(readmitted)
    return {"cohorts": observed, "before_reset": removed, "readmitted": readmitted,
            "widths": dict(Counter(mask.bit_count() for mask in masks)),
            "runtime": dict(Counter(runtime)), "final_request_bytes": steps[-1][2]}


def run_checks(port, out, pieces, *, require_partial):
    NativeChatTest.port = port
    NativeChatTest.pieces = pieces
    NativeChatTest.require_partial = require_partial
    evidence = {"prompts": [], "public": []}
    NativeChatTest.evidence = evidence
    with (out / "maintained.log").open("w") as log:
        import unittest

        suite = unittest.defaultTestLoader.loadTestsFromTestCase(NativeChatTest)
        expected = suite.countTestCases()
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    save(out / "maintained.json", evidence)
    assert expected > 0 and result.testsRun == expected and result.wasSuccessful(), out
    assert not result.skipped, out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--control", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--observer", type=Path)
    args = parser.parse_args()
    graph = args.mode == "gpu"
    if not args.control or (graph and not args.observer):
        parser.error("--control is required, and GPU mode requires --observer")
    all_cases = {case["name"]: case for case in fixtures()}
    selected = [all_cases[name] for name in (
        "plain", "literal-control-markers", "case-and-order-tools", "text-parts-concatenation",
    )]
    main_case = selected[0]
    out = args.output.resolve()
    out.mkdir()
    candidate = out / "candidate"
    control = out / "control"
    args.checkpoint = args.checkpoint.resolve()
    observer = args.observer.resolve() if args.observer else None
    pieces = tokenizer_pieces(args.checkpoint)
    with server(args.binary.resolve(), args.checkpoint, candidate, 4, graph=graph,
                observer=observer) as port:
        pre = raw_wires(port, candidate, main_case)
        run_checks(port, candidate, pieces, require_partial=not graph)
        save(candidate / "chat-zero.json", verify_zero(port, main_case))
        cohorts = {width: mixed(port, candidate, selected, width) for width in (2, 3, 4)}
        save(candidate / "mixed.json", cohorts)
        cancellation = churn(port, candidate, selected, graph)
        save(candidate / "cancellation.json", cancellation)
        post = candidate / "after-replacement"
        post.mkdir()
        assert raw_wires(port, post, main_case) == pre
        if graph:
            ordinary = [baseline(port, case, 17) for case in selected]
            correction_case = {"ids": [760, 6511, 314, 9338, 369]}
            correction = legacy(port, correction_case, 16, stop=198)
            assert correction[:2] == ([11751, 13, 198], "stop")
    if graph:
        save(candidate / "graph.json", verify_graph(candidate, cohorts, cancellation))
    else:
        cancellation["cpu"] = verify_cpu_cancellation(candidate, cancellation, pieces)
        steps, masks = batch_trace(candidate)
        assert not masks and steps[-1][2] == 0
        for width, cohort in cohorts.items():
            start, end = cohort["begin"][0], cohort["end"][0]
            assert sum(step[:2] == (width, width) for step in steps[start:end]) >= 2, width
        save(candidate / "cancellation.json", cancellation)
    with server(args.control.resolve(), args.checkpoint, control, 4, graph=graph) as port:
        assert raw_wires(port, control, main_case) == pre
    for name in pre:
        wire = (candidate / f"{name}.wire").read_bytes()
        assert wire == (control / f"{name}.wire").read_bytes()
        assert wire == (post / f"{name}.wire").read_bytes()
    save(out / "raw-parity.json", {
        name: digest(candidate / f"{name}.wire") for name in pre
    })
    if graph:
        mtp = out / "mtp4"
        with server(args.binary.resolve(), args.checkpoint, mtp, 1, mtp=True) as port:
            actual = [baseline(port, case, 17) for case in selected]
            assert actual == ordinary
            assert legacy(port, correction_case, 16, stop=198) == correction
        save(mtp / "greedy.json", {"prompts": [case["ids"] for case in selected],
                                  "ordinary": ordinary, "mtp4": actual,
                                  "correction": correction})
    print(f"PASS native chat {args.mode}: mixed widths 2, 3, 4, reset and replacement, raw parity",
          flush=True)


if __name__ == "__main__":
    main()
