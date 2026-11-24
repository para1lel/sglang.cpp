"""Run dynamic K0 serving checks against a release server using --trace-batches."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
import json
import os
from pathlib import Path
import re
import select
import subprocess
import sys
import threading
import time


sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_native as native


FIXTURES = native.load_fixtures()
TEXT_CASES = Path(__file__).resolve().parents[1] / "text/cases.json"
TEXT = json.loads(TEXT_CASES.read_text())["cases"]["chatml"]
DIRECT = ("held-out", "facts", "colors", "page-boundary")


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def raw_body():
    return {
        "model": "Qwen3.8-27B-FP8", "prompt": TEXT["text"], "max_tokens": 24, "stream": True,
        "stream_options": {"include_usage": True}, "return_token_ids": True,
    }


def consume(stream, exact, preposted=False):
    sent = time.perf_counter_ns()
    if not preposted:
        stream.post()
    stream.open()
    timestamps = []
    tokens = []
    finish = usage = None
    while True:
        event = stream.event()
        if event == "[DONE]":
            done = time.perf_counter_ns()
            break
        if event["choices"]:
            assert finish is None and usage is None
            choice, = event["choices"]
            assert choice["index"] == 0
            ids = choice["token_ids"]
            if ids:
                assert len(ids) == 1 and choice["finish_reason"] is None
                tokens.extend(ids)
                timestamps.append(time.perf_counter_ns())
            else:
                assert choice["finish_reason"] in ("length", "stop")
                finish = choice["finish_reason"]
        else:
            assert finish and usage is None
            usage = event["usage"]
    assert stream.response.read(1) == b""
    prompt = stream.request["prompt"]
    expected_prompt = stream.case["prompt"] if isinstance(prompt, str) else prompt
    assert stream.events[0]["choices"][0]["prompt_token_ids"] == expected_prompt
    assert usage["completion_tokens"] == len(tokens)
    assert usage["prompt_tokens"] == len(expected_prompt)
    assert usage["prompt_tokens_details"] == {"cached_tokens": 0}
    if isinstance(prompt, str):
        assert usage["total_tokens"] == len(expected_prompt) + len(tokens)
    else:
        assert set(usage) == {"prompt_tokens", "completion_tokens", "prompt_tokens_details"}
    stop = stream.request.get("stop_token_ids", [])
    if stream.case.get("raw"):
        stop = [248044, 248046]
        assert tokens == [248046] and finish == "stop"
        assert stream.events[len(tokens) - 1]["choices"][0]["text"] == ""
    elif exact and "stop_token_id" in stream.case:
        assert tokens == stream.case["output"] and finish == "stop"
    else:
        if exact and not stream.case.get("sampling"):
            expected = stream.case["output"]
            assert tokens[:len(expected)] == expected[:len(tokens)]
        assert finish == "stop" or len(tokens) == stream.request["max_tokens"]
    assert (tokens[-1] in stop) == (finish == "stop")
    return {**stream.evidence(), "tokens": tokens, "finish": finish, "usage": usage,
            "sent_ns": sent, "tokens_ns": timestamps, "done_ns": done}


def start(port, case, barrier, exact):
    with native.Stream(port, case) as stream:
        if case.get("raw"):
            stream.request = raw_body()
        elif sampling := case.get("sampling"):
            stream.request = sampling
        barrier.wait(timeout=15)
        return consume(stream, exact)


def parallel(port, cases, exact):
    barrier = threading.Barrier(len(cases))
    with ThreadPoolExecutor(max_workers=len(cases)) as pool:
        work = [pool.submit(start, port, case, barrier, exact) for case in cases]
        return [item.result() for item in work]


def cancel_and_replace(port, width, exact):
    with ExitStack() as stack, ThreadPoolExecutor(max_workers=width) as pool:
        cancel = stack.enter_context(native.Stream(port, FIXTURES["held-out"], budget=4096))
        peers = [stack.enter_context(native.Stream(port, FIXTURES[name], budget=36))
                 for name in DIRECT[1:width]]
        replacement = stack.enter_context(native.Stream(port, FIXTURES["code"]))
        gate = threading.Barrier(width)

        def post(stream):
            gate.wait(timeout=15)
            stream.post()

        work = [pool.submit(post, stream) for stream in [cancel, *peers]]
        for item in work:
            item.result()
        work = [pool.submit(consume, stream, exact, True) for stream in peers]
        cancel.open()
        for index in range(4):
            event = cancel.event()
            choice, = event["choices"]
            assert choice["finish_reason"] is None and len(choice["token_ids"]) == 1
            if index == 0:
                assert choice["prompt_token_ids"] == cancel.case["prompt"]
            if exact:
                assert choice["token_ids"] == cancel.case["output"][index:index + 1]
        replacement.post()
        assert not select.select([replacement.sock], [], [], 0.06)[0], "Extra lane admitted early"
        rst_ns = time.perf_counter_ns()
        cancel.reset()
        next_response = consume(replacement, exact, True)
        ongoing = [item.result() for item in work]
        assert any(item["tokens_ns"][-1] > next_response["tokens_ns"][0] for item in ongoing), (
            "Replacement did not overlap a surviving lane")
    return {"cancel": cancel.evidence(), "reset_ns": rst_ns,
            "replacement": next_response, "peers": ongoing}


def scheduler(path, offset):
    rows = []
    for line in path.read_bytes()[offset:].splitlines():
        if step := native.TRACE.fullmatch(line):
            width, decode, request_bytes = map(int, step.groups())
            rows.append({"width": width, "decode": decode, "request_bytes": request_bytes,
                         "graph_slots": 0})
        elif prefill := re.fullmatch(rb"batch prefill_rows=([\d,]+) workspace_bytes=(\d+)", line):
            rows[-1]["prefill"] = [int(value) for value in prefill[1].split(b",")]
            rows[-1]["workspace_bytes"] = int(prefill[2])
        elif groups := re.fullmatch(rb"batch layerwise_prefill_groups=(\d+) rows=(\d+)", line):
            rows[-1]["layerwise_prefill_groups"], rows[-1]["layerwise_prefill_rows"] = map(
                int, groups.groups())
        elif graph := re.fullmatch(rb"batch graph_slots=(\d+)", line):
            rows[-1]["graph_slots"] = int(graph[1])
        elif sampling := re.fullmatch(rb"batch sampling_workspace_bytes=([1-9]\d*)", line):
            assert rows and rows[-1]["width"] and "sampling_workspace_bytes" not in rows[-1], line
            rows[-1]["sampling_workspace_bytes"] = int(sampling[1])
        elif target := re.fullmatch(rb"batch target_rows=(\d+) step_budget=(\d+)", line):
            assert rows and rows[-1]["width"] and "target_rows" not in rows[-1], line
            rows[-1]["target_rows"], rows[-1]["step_budget"] = map(int, target.groups())
        elif verify := re.fullmatch(
            rb"batch verify lanes=(\d+) rows=(\d+) peak_request_bytes=(\d+)", line
        ):
            assert rows and rows[-1]["width"] and "verification_rows" not in rows[-1], line
            rows[-1]["verification_lanes"], rows[-1]["verification_rows"] = map(
                int, verify.groups()[:2])
            assert int(verify[3]) >= rows[-1]["request_bytes"], line
        elif mtp := re.fullmatch(
            rb"batch mtp slot=(\d+) drafted=(\d+) accepted=(\d+) verified=(\d+) "
            rb"replayed=(\d+) boundary=([01])",
            line,
        ):
            assert rows and "verification_rows" in rows[-1], line
            slots = rows[-1].setdefault("mtp_slots", {})
            slot, drafted, accepted, verified, replayed, boundary = map(int, mtp.groups())
            assert slot not in slots, line
            slots[slot] = {"drafted": drafted, "accepted": accepted, "verified": verified,
                           "replayed": replayed, "boundary": boundary}
        elif peak := re.fullmatch(rb"batch peak_request_bytes=(\d+)", line):
            assert rows and rows[-1]["width"] and "peak_request_bytes" not in rows[-1], line
            peak_bytes = int(peak[1])
            assert peak_bytes >= rows[-1]["request_bytes"], line
            rows[-1]["peak_request_bytes"] = peak_bytes
        else:
            assert not line.startswith(b"batch "), line
    assert rows and rows[-1]["request_bytes"] == 0
    return rows


def check(port, out, maximum_width, graph_enabled):
    log = out / "server.log"
    result = {}
    for width in range(1, maximum_width + 1):
        cases = [FIXTURES[name] for name in DIRECT[:width]]
        offset = log.stat().st_size
        streams = parallel(port, cases, exact=False)
        rows = scheduler(log, offset)
        assert any(row["width"] == row["decode"] == width for row in rows), (width, rows)
        assert all(row["graph_slots"].bit_count() <= int(graph_enabled) for row in rows)
        if width == 1 and graph_enabled:
            assert any(row["graph_slots"] for row in rows)
        result[f"width{width}"] = {"streams": streams, "rows": rows}
        save(out / "http.json", result)
    offset = log.stat().st_size
    mixed = [{"name": "chatml-eos", "raw": True, "prompt": TEXT["ids"], "max_tokens": 24},
             FIXTURES["numbers"],
             FIXTURES["held-out"], FIXTURES["page-boundary"]][:maximum_width]
    streams = parallel(port, mixed, exact=False)
    result["independent-stops"] = {"streams": streams, "rows": scheduler(log, offset)}
    save(out / "http.json", result)
    greedy = {**FIXTURES["held-out"], "max_tokens": 32}
    sampled = {"name": "seeded-sampling", "prompt": TEXT["ids"], "max_tokens": 32, "sampling": {
        **raw_body(), "max_tokens": 32, "ignore_eos": True,
        "temperature": 0.8, "top_k": 20, "seed": 42,
    }}
    offset = log.stat().st_size
    streams = parallel(port, [greedy, sampled], exact=False)
    assert all(stream["finish"] == "length" and len(stream["tokens"]) == 32 for stream in streams)
    rows = scheduler(log, offset)
    joint_decode = [row for row in rows
                    if row["decode"] == 2 and row.get("sampling_workspace_bytes")]
    assert joint_decode and all(row["graph_slots"] == 0 for row in joint_decode), rows
    result["sampling-and-replay"] = {"streams": streams, "rows": rows}
    save(out / "http.json", result)
    offset = log.stat().st_size
    result["cancel-and-replace"] = cancel_and_replace(port, maximum_width, exact=False)
    result["cancel-and-replace"]["rows"] = scheduler(log, offset)
    result["recovery"] = parallel(port, [FIXTURES["code"]], exact=False)[0]
    replacement = result["cancel-and-replace"]["replacement"]
    assert replacement["tokens"] == result["recovery"]["tokens"]
    assert replacement["finish"] == result["recovery"]["finish"]
    save(out / "http.json", result)


@contextmanager
def server(args):
    args.output.mkdir()
    port = native.choose_port()
    command = [str(args.binary.resolve()), "--checkpoint", str(args.checkpoint.resolve()),
               "--device", "0", "--port", str(port), "--max-active", str(args.width),
               "--mtp-steps", "0", "--trace-batches"]
    if getattr(args, "max_num_batched_tokens", None) is not None:
        command += ["--max-num-batched-tokens", str(args.max_num_batched_tokens)]
    if args.route == "eager":
        command.append("--eager")
    environment = dict(os.environ, LD_PRELOAD=str(args.activity.resolve()),
                       BATCH_GRAPH_ACTIVITY=str((args.output / "cupti.tsv").resolve()))
    log_path = args.output / "server.log"
    save(args.output / "command.json", command)
    with log_path.open("wb") as log:
        process = subprocess.Popen(command, env=environment, stdout=log, stderr=log,
                                   start_new_session=True)
        try:
            native.wait_until_ready(process, port, log_path)
            yield port
        finally:
            native.stop_server(process, None)
    assert process.returncode == 0, log_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("port", type=int, help="Existing server port, or 0 to start a server")
    parser.add_argument("output", type=Path)
    parser.add_argument("width", type=int, choices=(2, 3, 4))
    parser.add_argument("route", choices=("eager", "graph"))
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--activity", type=Path)
    args = parser.parse_args()
    if args.port:
        check(args.port, args.output, args.width, args.route == "graph")
        return
    if not all((args.binary, args.checkpoint, args.activity)):
        parser.error("Starting a server requires --binary, --checkpoint and --activity")
    with server(args) as port:
        check(port, args.output, args.width, args.route == "graph")


if __name__ == "__main__":
    main()
