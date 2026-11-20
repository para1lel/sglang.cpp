"""Compare sampled HTTP output and stop behavior against native K0."""

import argparse
from base64 import b64encode
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, closing, contextmanager
import http.client
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from threading import Barrier
import time


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from batch.run_native import Stream, choose_port, require, stop_server, wait_until_ready, write_json

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "prefix"))
from live_controls import ENTRIES, Controls


CHECKPOINT = Path("/data0/shared/Qwen3.8-27B-FP8")
TOKEN_BUDGET = 32768
NO_DEVICE = "Get current CUDA device: no CUDA-capable device is detected"
TEXT = "/v1/completions"
CHAT = "/v1/chat/completions"
COMMON = {"model": "Qwen3.8-27B-FP8", "return_token_ids": True}
MAIN = COMMON | {
    "prompt": "The first three prime numbers are", "max_tokens": 20, "ignore_eos": True,
    "temperature": 0.83, "top_p": 0.9, "top_k": 48, "min_p": 0.015, "seed": -(2**63),
}
EOS = COMMON | {
    "messages": [{"role": "user", "content": "Reply with exactly the word yes."}],
    "enable_thinking": False, "max_tokens": 24, "temperature": 0.4, "top_k": 20, "seed": 14,
}
MASKED_CHOICES = ("Please reply with yes today.", "Please reply with no today.")
MASKED = MAIN | {
    "prompt": "Choose one reply.", "max_tokens": 12,
    "ebnf": "root ::= " + " | ".join(map(json.dumps, MASKED_CHOICES)),
    "top_p": 1, "min_p": 0, "ignore_eos": False,
}
LONG = MAIN | {
    "prompt": "The first three prime numbers are 2, 3, and 5.\n" * 48 + MAIN["prompt"],
    "max_tokens": 12,
}
COUNTERS = re.compile(rb"drafted=(\d+) accepted=(\d+) verified=(\d+) replayed=(\d+)")
LANES = re.compile(rb"^batch verify lanes=(\d+) rows=(\d+) ", re.M)
GRAPH = re.compile(rb"^batch graph_slots=\d+", re.M)


@contextmanager
def server(args, output, width, depth, prefill="exact", flags=(), token_budget=TOKEN_BUDGET):
    output.mkdir()
    port = choose_port()
    command = [str(args.server), "--checkpoint", str(args.checkpoint), "--device", "0",
               "--port", str(port), "--max-active", str(width), "--mtp-steps", str(depth),
               "--max-num-batched-tokens", str(token_budget), "--fp8-prefill", prefill,
               "--trace-batches", *flags]
    write_json(output / "command.json", command)
    log = output / "server.log"
    with log.open("wb") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=stream, start_new_session=True)
        try:
            health = wait_until_ready(process, port, log)
            write_json(output / "health.json", health)
            require(health.get("max_active") == width and
                    health.get("max_num_batched_tokens") == token_budget, health)
            yield port, log
        finally:
            stop_server(process, None)


def post(port, route, body, stream=True, barrier=None, status=200):
    if stream:
        body = body | {"stream": True, "stream_options": {"include_usage": True}}
    with closing(http.client.HTTPConnection("127.0.0.1", port, timeout=60)) as connection:
        if barrier:
            barrier.wait(timeout=15)
        connection.request("POST", route, json.dumps(body),
                           {"Content-Type": "application/json", "Connection": "close"})
        with connection.getresponse() as response:
            require(response.status == status, (response.status, response.read()
                                                if response.status != status else None))
            if not stream:
                return json.load(response)
            require(response.headers.get_content_type() == "text/event-stream", response.headers)
            frames = []
            while line := response.readline():
                require(line.startswith(b"data: ") and response.readline() == b"\n", line)
                if line == b"data: [DONE]\n":
                    require(response.read(1) == b"", "Bytes after [DONE]")
                    return frames
                frames.append(json.loads(line[6:]))
    raise AssertionError("Missing SSE [DONE]")


def transcript(frames):
    tokens, prompts, content, finishes, usages = [], [], [], [], []
    for frame in frames:
        if "prompt_token_ids" in frame:
            prompts.append(frame["prompt_token_ids"])
        for choice in frame["choices"]:
            tokens.extend(choice.get("token_ids", []))
            if "prompt_token_ids" in choice:
                prompts.append(choice["prompt_token_ids"])
            if choice["finish_reason"]:
                finishes.append(choice["finish_reason"])
            message = choice.get("delta", choice.get("message", {}))
            content.append(choice.get("text", message.get("content") or ""))
        if frame.get("usage") is not None:
            usages.append(frame["usage"])
    prompt, = prompts
    finish, = finishes
    usage, = usages
    require(usage["completion_tokens"] == len(tokens), usage)
    return {"prompt": prompt, "tokens": tokens, "text": "".join(content),
            "finish": finish, "usage": usage}


def cohort(port, cases):
    barrier = Barrier(len(cases))

    def client(case):
        return post(port, *case, barrier=barrier)

    with ThreadPoolExecutor(max_workers=len(cases)) as pool:
        return list(pool.map(client, cases))


def stop_case(wanted):
    ids = wanted["tokens"]
    index, token = next((index, token) for index, token in enumerate(ids)
                        if index >= 3 and token not in ids[:index])
    return MAIN | {"stop_token_ids": [token], "ignore_eos": False, "max_tokens": index + 1}


def media_body():
    pixels = b64encode(bytes((195, 30, 30)) * (64 * 64)).decode()
    image = {"type": "image_url", "image_url": {
        "url": f"data:application/x-sglang-rgb8;base64,{pixels}", "width": 64, "height": 64,
    }}
    text = {"type": "text", "text": "What color is the image? Reply with one word."}
    return EOS | {"messages": [{"role": "user", "content": [image, text]}],
                  "max_tokens": 4, "ignore_eos": True}


def check_media(port, log):
    body = media_body()
    offset = log.stat().st_size
    error = post(port, CHAT, body | {"mtp_steps": 1}, stream=False, status=400)
    require("Unsupported chat completion option: mtp_steps" in error["error"]["message"], error)
    require(log.stat().st_size == offset, "Rejected media request produced generation traces")
    frames = post(port, CHAT, body)
    row = transcript(frames)
    require(len(row["tokens"]) == body["max_tokens"] and row["finish"] == "length", row)
    require(not COUNTERS.search(log.read_bytes()[offset:]), "Media request used MTP")
    return {"error": error, "wire": frames, "transcript": row}


def sampled_stream(port, body):
    stream = Stream(port, {"name": "sampled", "prompt": [], "max_tokens": body["max_tokens"]})
    stream.request = body | {"stream": True, "stream_options": {"include_usage": True}}
    return stream


def first_token(stream, wanted):
    stream.open()
    frame = stream.event()
    choice, = frame["choices"]
    require(choice["token_ids"] == wanted["tokens"][:1] and choice["finish_reason"] is None,
            frame)


def check_cancellation(port, log, width, depth, wanted):
    offset = log.stat().st_size
    with sampled_stream(port, MAIN | {"max_tokens": 4096}) as stream:
        stream.post()
        stream.open()
        emitted = []
        while len(emitted) < 5:
            choice, = stream.event()["choices"]
            require(choice["finish_reason"] is None, choice)
            emitted.extend(choice["token_ids"])
        require(emitted == wanted["tokens"][:len(emitted)], emitted)
        stream.reset()
        interrupted = stream.evidence()
    trace = log.read_bytes()[offset:]
    if depth:
        require(COUNTERS.search(trace), "Interrupted request had no verifier trace")
    if width == 1:
        deadline = time.monotonic() + 15
        while b"batch width=0 decode=0" not in trace and time.monotonic() < deadline:
            time.sleep(0.02)
            trace = log.read_bytes()[offset:]
        require(b"batch width=0 decode=0" in trace, "Request did not retire after TCP reset")
    replay = post(port, TEXT, MAIN)
    require(transcript(replay) == wanted, "TCP reset changed the next seeded request")
    return {"interrupted": interrupted, "before_replay_trace": trace.decode(), "replay": replay}


def run_cases(port, log, width, depth, prefill):
    main = post(port, TEXT, MAIN)
    wanted = transcript(main)
    stop = stop_case(wanted)
    cases = {
        "main": main,
        "json": [post(port, TEXT, MAIN, stream=False)],
        "stop-at-length": post(port, TEXT, stop),
        "natural-eos": post(port, CHAT, EOS),
        "ignore-natural-eos": post(port, CHAT, EOS | {"ignore_eos": True, "max_tokens": 4}),
        "one-token": post(port, TEXT, MAIN | {"max_tokens": 1}),
        "argmax-filter": post(port, TEXT, MAIN | {"top_p": 1, "top_k": 1, "min_p": 1}),
        "greedy": post(port, TEXT, MAIN | {"temperature": 0}),
    }
    cases["long"] = post(port, TEXT, LONG)
    offset = log.stat().st_size
    cases["mask"] = post(port, TEXT, MASKED)
    mask_rounds = COUNTERS.findall(log.read_bytes()[offset:])
    require(any(int(row[0]) > 0 and int(row[2]) > 0 for row in mask_rounds) == bool(depth),
            "Unexpected constrained proposal and verification activity")
    rows = {name: transcript(frames) for name, frames in cases.items()}
    require(rows["main"] == rows["json"], "JSON and SSE differ")
    require(rows["main"]["finish"] == "length" and len(wanted["tokens"]) == MAIN["max_tokens"],
            rows["main"])
    require(rows["stop-at-length"]["finish"] == "stop" and
            rows["stop-at-length"]["tokens"] == wanted["tokens"][:stop["max_tokens"]],
            rows["stop-at-length"])
    require(rows["natural-eos"]["finish"] == "stop" and
            rows["natural-eos"]["tokens"] == [9405, 248046], rows["natural-eos"])
    require(rows["ignore-natural-eos"]["finish"] == "length" and
            rows["ignore-natural-eos"]["tokens"][:2] == rows["natural-eos"]["tokens"],
            rows["ignore-natural-eos"])
    require(rows["one-token"]["tokens"] == wanted["tokens"][:1], rows["one-token"])
    require(rows["argmax-filter"] == rows["greedy"], "Top-1 sampling changed greedy output")
    require(rows["mask"]["text"] in MASKED_CHOICES and rows["mask"]["finish"] == "stop",
            rows["mask"])
    require(len(rows["long"]["prompt"]) >= 512 and
            len(rows["long"]["tokens"]) == LONG["max_tokens"], (prefill, rows["long"]))
    if width == 4:
        other = MAIN | {"seed": 2**63 - 1}
        mixed = [(TEXT, other), (CHAT, EOS), (TEXT, MAIN), (TEXT, MASKED)]
        cases["mixed"] = cohort(port, mixed)
        rows["mixed"] = [transcript(frame) for frame in cases["mixed"]]
        require(rows["mixed"][2] == wanted, "Concurrent neighbors changed the same seed")
        require(rows["mixed"][0]["tokens"] != wanted["tokens"], "Seed fixture did not diverge")
        require(rows["mixed"][1] == rows["natural-eos"] and rows["mixed"][3] == rows["mask"],
                "Concurrent neighbors changed EOS or mask behavior")
    errors = [post(port, TEXT, MAIN | {"min_p": value}, stream=False, status=400)
              for value in (-0.01, 1.01, "0.2")]
    require(all("min_p" in error["error"]["message"] for error in errors), errors)
    return {
        "wire": cases, "transcripts": rows, "errors": errors, "stop_request": stop,
        "media": check_media(port, log),
        "cancellation": check_cancellation(port, log, width, depth, wanted),
    }


def check_against_control(rows, control, context):
    for name, row in rows.items():
        require(row == control[name], f"{context}: changed {name}")


def check_configuration(args, width, prefill, mode, depths, flags=(), exact=None):
    observed = {}
    media_control = None
    for depth in depths:
        output = args.evidence / f"w{width}-{prefill}-{mode}-k{depth}"
        with server(args, output, width, depth, prefill, flags) as (port, log):
            evidence = run_cases(port, log, width, depth, prefill)
        write_json(output / "wire.json", evidence)
        rows = evidence["transcripts"]
        media = evidence["media"]["transcript"]
        if depth:
            require(media == media_control, f"W{width} {prefill} {mode} K{depth}: changed media")
        else:
            media_control = media
        if width == 1:
            rows = rows | {"media": media}
        if depth:
            control = observed[0] if exact is None else exact[depth]
            check_against_control(rows, control, f"W{width} {prefill} {mode} K{depth}")
        observed[depth] = rows
        summary = {"depth": depth, "width": width, "prefill": prefill, "mode": mode}
        trace = log.read_bytes()
        graph_steps = len(GRAPH.findall(trace))
        summary["graph_steps"] = graph_steps
        counts = tuple(sum(map(int, values)) for values in zip(*COUNTERS.findall(trace)))
        summary["counters"] = dict(zip(("drafted", "accepted", "verified", "replayed"), counts))
        if depth:
            require(counts and counts[0] > counts[1] > 0 and counts[3] > 0, summary)
            lanes = max((int(lane) for lane, _ in LANES.findall(trace)), default=0)
            require(lanes >= min(width, 2), f"No expected verifier lanes at W{width} K{depth}")
            for step in trace.split(b"\nbatch width=")[1:]:
                match = LANES.search(step)
                if match and int(match[1]) > 1:
                    require(not GRAPH.search(step), "Multiple verifier lanes replayed a graph")
            summary["concurrent_verifiers"] = lanes
        if mode != "graph" or depth == 0:
            require(bool(graph_steps) == (mode == "graph"), summary)
        print(json.dumps(summary), flush=True)
    return observed


def check_startup(args):
    cases = [
        (["--mtp-steps", "1"], NO_DEVICE),
        (["--max-active", "5", "--mtp-steps", "1"], NO_DEVICE),
        (["--max-active", "32", "--mtp-k", "4"], NO_DEVICE),
        (["--max-active", "32", "--prefix-cache-entries", "2"], NO_DEVICE),
        (["--max-active", "33", "--mtp-steps", "1"], "--max-active in [1, 32]"),
        (["--max-active", "32", "--mtp-steps", "5"], "--mtp-steps in [0, 4]"),
        (["--max-active", "4", "--mtp-steps", "4", "--prefix-cache-entries", "2"],
         "--prefix-cache-entries requires --mtp-steps 0. Omit --eager."),
        (["--max-active", "32", "--mtp-steps", "1", "--prefix-cache-entries", "2"],
         "--prefix-cache-entries requires --mtp-steps 0. Omit --eager."),
        (["--max-active", "1", "--mtp-steps", "1", "--eager"], NO_DEVICE),
        (["--max-active", "4", "--mtp-steps", "4", "--eager", "--batch-cuda-graphs"],
         NO_DEVICE),
        (["--max-active", "1", "--batch-cuda-graphs"], NO_DEVICE),
    ]
    records = []
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    for flags, diagnostic in cases:
        command = [str(args.server), "--checkpoint", str(args.evidence / "absent"),
                   "--max-num-batched-tokens", str(TOKEN_BUDGET), *flags]
        result = subprocess.run(command, env=environment, capture_output=True, text=True,
                                timeout=10, check=False)
        records.append({"command": command, "status": result.returncode,
                        "expected": diagnostic, "stdout": result.stdout, "stderr": result.stderr})
        write_json(args.evidence / "startup.json", records)
        require(result.returncode != 0 and diagnostic in result.stderr, records[-1])


def finish_stream(stream):
    stream.open()
    while stream.event() != "[DONE]":
        continue
    require(stream.response.read(1) == b"", "Bytes after [DONE]")
    return transcript(stream.events[:-1])


def check_queue_and_reset(args):
    output = args.evidence / "w2-exact-queue"
    flags = ("--batch-cuda-graphs", "--prefix-cache-entries", "2")
    with server(args, output, 2, 0, flags=flags) as (port, _):
        controls = Controls(port, output)
        body = LONG | {"cache_salt": "sampled-queue"}
        wanted = transcript(post(port, TEXT, body))
        with ExitStack() as streams:
            active = [streams.enter_context(sampled_stream(port, body | {"max_tokens": 4096}))
                      for _ in range(2)]
            for stream in active:
                stream.post()
                first_token(stream, wanted)
            controls.wait_for("active", 2, 0)
            queued, survivor = [streams.enter_context(sampled_stream(port, body)) for _ in range(2)]
            for stream in (queued, survivor):
                stream.post()
            controls.wait_for("busy", 2, 2)
            controls.reset("busy-reset", 409)
            queued.reset()
            controls.wait_for("queued-cancelled", 2, 1)
            for stream in active:
                stream.reset()
            replay = finish_stream(survivor)
            records = [stream.evidence() for stream in [*active, queued, survivor]]
        before = controls.wait_for("retired", 0, 0)
        controls.reset("idle-reset")
        after = controls.metrics("after-reset")
        replay_after_reset = transcript(post(port, TEXT, body))
        write_json(output / "wire.json", {"streams": records, "cold": wanted, "survivor": replay,
                                        "after_reset": replay_after_reset})
        require(before[ENTRIES] > 0 and after[ENTRIES] == 0, (before, after))
        for key in ("prompt", "tokens", "text", "finish"):
            require(replay[key] == wanted[key] == replay_after_reset[key], key)
        warm_cache = replay["usage"]["prompt_tokens_details"]["cached_tokens"]
        reset_cache = replay_after_reset["usage"]["prompt_tokens_details"]["cached_tokens"]
        require(warm_cache > 0 and reset_cache == 0, (replay, replay_after_reset))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    args.evidence.mkdir(parents=True)
    check_startup(args)
    graph_flag = ("--batch-cuda-graphs",)
    singleton = check_configuration(args, 1, "exact", "default", range(5))
    eager = check_configuration(args, 1, "exact", "eager", (0, 4), (*graph_flag, "--eager"))
    for depth in eager:
        require(singleton[depth] == eager[depth], f"Explicit eager changed capacity-one K{depth}")
    depths = (0, 1, 4)
    exact_graph = check_configuration(args, 4, "exact", "graph", range(5), graph_flag)
    exact_eager = check_configuration(args, 4, "exact", "eager", depths)
    tensor_graph = check_configuration(args, 4, "tensor-core", "graph", depths,
                                       graph_flag, exact=exact_graph)
    tensor_eager = check_configuration(args, 4, "tensor-core", "eager", depths, exact=exact_eager)
    for depth in depths:
        require(exact_graph[depth] == exact_eager[depth], f"Exact graph/eager K{depth} differs")
        require(tensor_graph[depth] == tensor_eager[depth], f"Tensor graph/eager K{depth} differs")
    write_json(args.evidence / "precision-controls.json",
               {"exact_k0": exact_graph[0]["long"]["tokens"],
                "tensor_k0": tensor_graph[0]["long"]["tokens"]})
    check_queue_and_reset(args)


if __name__ == "__main__":
    main()
