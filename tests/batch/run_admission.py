"""Check effective scheduler caps and live admission on a reserved B200."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, contextmanager
import hashlib
from http.client import HTTPConnection
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_native as native
import test_batch_graph as http
from media import http_routes as media


DEFAULT_ACTIVE = 8
MAX_ACTIVE = 32
DEFAULT_TOKENS = 32768
MAX_TOKENS = 32768
SMALL_TOKENS = 64
CHECKPOINT_INTERVAL = 8192
TOKEN_OPTION = "--max-num-batched-tokens"
NO_DEVICE = "Get current CUDA device: no CUDA-capable device is detected"


@contextmanager
def server(args, label, flags, *, binary=None, environment=None, cache_entries=0):
    output = args.evidence / label
    output.mkdir()
    port = native.choose_port()
    command = [str(binary or args.binary), "--checkpoint", str(args.checkpoint),
               "--port", str(port), "--trace-batches", "--prefix-cache-entries", str(cache_entries),
               *flags]
    log_path = output / "server.log"
    native.write_json(output / "command.json", command)
    with log_path.open("wb") as log:
        process = subprocess.Popen(command, cwd=output, stdout=log, stderr=log, env=environment,
                                   start_new_session=True)
        try:
            health = native.wait_until_ready(process, port, log_path)
            native.write_json(output / "health.json", health)
            yield port, log_path, output, health
        finally:
            native.stop_server(process, None)
    assert process.returncode == 0, log_path


def check_caps(health, active, tokens):
    for name, expected in (("max_active", active), ("max_num_batched_tokens", tokens)):
        actual = health.get(name)
        assert type(actual) is int and actual == expected, (
            f"/health {name}: expected {expected}, got {actual!r}", health)


def scheduler(log, offset, width):
    native.await_retirement(log, offset, width)
    steps = http.scheduler(log, offset)
    for step in steps:
        step.setdefault("prefill", [])
        assert 0 <= step["decode"] <= step["width"] <= width, step
        assert not step["width"] or step["prefill"], step
        assert sum(row > 0 for row in step["prefill"]) + step["decode"] <= step["width"], step
    return steps


def check_step_budget(steps, capacity):
    charges = []
    for step in steps:
        verified = step.get("verification_rows", 0)
        verification_lanes = step.get("verification_lanes", 0)
        assert 0 <= verification_lanes <= step["decode"] and verified >= verification_lanes, step
        slots = step.get("mtp_slots", {})
        assert all(0 <= slot < len(step["prefill"]) for slot in slots), step
        assert sum(entry["verified"] > 0 for entry in slots.values()) == verification_lanes, step
        assert sum(entry["verified"] for entry in slots.values()) == verified, step
        charged = sum(step["prefill"]) + step["decode"] - verification_lanes + verified
        if step["width"]:
            assert step["step_budget"] == capacity and step["target_rows"] == charged, step
        assert charged <= capacity, (capacity, step)
        charges.append(charged)
    return charges


def check_k0_rows(steps, capacity, prompt_rows, decode_rows):
    charged = check_step_budget(steps, capacity)
    assert sum(sum(step["prefill"]) for step in steps) == prompt_rows, (prompt_rows, steps)
    assert sum(step["decode"] for step in steps) == decode_rows, (decode_rows, steps)
    return max(charged)


def prompt_case(name, length, lane=0):
    tokens = http.FIXTURES["held-out"]["prompt"]
    repetitions = (lane + length + len(tokens) - 1) // len(tokens)
    return {"name": name, "prompt": (tokens * repetitions)[lane:lane + length],
            "max_tokens": 1, "output": []}


def cohort(port, cases):
    barrier = threading.Barrier(len(cases))

    def complete(case):
        with native.Stream(port, case, timeout=180) as stream:
            barrier.wait(timeout=15)
            return http.consume(stream, exact=False)

    with ThreadPoolExecutor(max_workers=len(cases)) as pool:
        return list(pool.map(complete, cases))


def check_ordinary(port, log, out, width, tokens, label, cases):
    offset = log.stat().st_size
    streams = cohort(port, cases)
    steps = scheduler(log, offset, width)
    native.write_json(out / f"{label}.json", {"streams": streams, "rows": steps})
    return check_k0_rows(steps, tokens, sum(len(case["prompt"]) for case in cases),
                         sum(len(stream["tokens"]) - 1 for stream in streams))


def check_bounded(args):
    cap = args.small_tokens
    flags = ["--max-active", "4", "--mtp-steps", "0", TOKEN_OPTION, str(cap)]
    with server(args, "bounded", flags) as (port, log, out, health):
        check_caps(health, 4, cap)
        for length in (cap - 1, cap, cap + 1):
            case = prompt_case(f"prompt-{length}", length)
            peak = check_ordinary(port, log, out, 4, cap, case["name"], [case])
            assert peak == min(length, cap), (case["name"], peak)
        check_mixed(port, log, out, cap)


def check_mixed(port, log, out, cap):
    short = native.serial_cases(port, {"held-out": http.FIXTURES["held-out"]},
                                out / "mixed-reference.json")["held-out"]
    offset = log.stat().st_size
    prompts = [prompt_case("short-tail", cap - 3), prompt_case("long-tail", cap + 5, 3)]
    with native.Stream(port, short, budget=1024) as decoder:
        decoder.post()
        decoder.open()
        decoder.choice(short["output"][:1], first=True)
        streams = cohort(port, prompts)
        decoder.choice(short["output"][1:2])
        decoder.reset()
        progress = decoder.evidence()
    steps = scheduler(log, offset, 4)
    native.write_json(out / "mixed.json", {"streams": streams, "cancelled": progress,
                                          "rows": steps})
    charged = check_step_budget(steps, cap)
    assert sum(sum(step["prefill"]) for step in steps) == (
        len(short["prompt"]) + sum(len(case["prompt"]) for case in prompts)), steps
    assert any(step["decode"] > 0 and sum(step["prefill"]) > 0 and charge == cap
               for step, charge in zip(steps, charged, strict=True)), steps
    assert all(stream["finish"] == "length" for stream in streams), streams
    check_ordinary(port, log, out, 4, cap, "reuse", [prompt_case("reuse", cap + 1)])


def check_default(args):
    with server(args, "default", ["--mtp-steps", "0"]) as (port, log, out, health):
        check_caps(health, DEFAULT_ACTIVE, DEFAULT_TOKENS)
        cases = native.serial_cases(port, http.FIXTURES, out / "serial-reference.json")
        offset = log.stat().st_size
        native.burst(port, log, out / "low-concurrency.json",
                     [cases["held-out"], cases["facts"]], full_decode=True)
        check_step_budget(scheduler(log, offset, DEFAULT_ACTIVE), DEFAULT_TOKENS)
        check_default_media(port, log, out, cases["held-out"])
        native.check_admission(port, log, out / "queue-and-reuse.json", cases, DEFAULT_ACTIVE)


def check_default_media(port, log, out, fixture):
    offset = log.stat().st_size
    with native.Stream(port, fixture, budget=1024, timeout=180) as decoder:
        decoder.post()
        decoder.open()
        decoder.choice(fixture["output"][:1], first=True)
        reply = media.capture(argparse.Namespace(evidence=out), "default-media", port,
                              media.CHAT, media.chat(media.mixed_content(), 1))
        media.received(reply, 1)
        decoder.choice(fixture["output"][1:2])
        decoder.reset()
    steps = scheduler(log, offset, DEFAULT_ACTIVE)
    native.write_json(out / "mixed-media.json", {"text": decoder.evidence(), "rows": steps})
    check_step_budget(steps, DEFAULT_TOKENS)
    assert any(step["decode"] > 0 and sum(step["prefill"]) > 0 for step in steps), steps


def check_pressure(args):
    flags = ["--mtp-steps", "0", "--max-active", str(MAX_ACTIVE)]
    with server(args, "pressure", flags) as (port, log, out, health):
        check_caps(health, MAX_ACTIVE, DEFAULT_TOKENS)
        count = DEFAULT_TOKENS // MAX_ACTIVE
        cases = [prompt_case(f"lane-{lane}", count + (lane == 0), lane)
                 for lane in range(MAX_ACTIVE)]
        check_ordinary(port, log, out, MAX_ACTIVE, DEFAULT_TOKENS, "cohort", cases)


def check_explicit(args, width):
    flags = ["--max-active", str(width), "--mtp-steps", "0"]
    with server(args, f"width-{width}", flags) as (port, log, out, health):
        check_caps(health, width, DEFAULT_TOKENS)
        cases = native.serial_cases(port, http.FIXTURES, out / "serial-reference.json")
        if width == 1:
            response = native.request(port, cases["held-out"])
            native.write_json(out / "serial.json", response.evidence())
            reply = media.capture(argparse.Namespace(evidence=out), "serial-media", port,
                                  media.CHAT, media.chat(media.mixed_content(), 1))
            media.received(reply, 1)
        else:
            native.burst(port, log, out / "full-width.json",
                         [cases[name] for name in http.DIRECT], full_decode=True)
        native.check_admission(port, log, out / "queue-and-reuse.json", cases, width)


def check_mtp(args):
    cap = 4
    flags = ["--max-active", "4", "--mtp-steps", "1", TOKEN_OPTION, str(cap)]
    with server(args, "mtp", flags) as (port, log, out, health):
        check_caps(health, 4, cap)
        fixture = http.FIXTURES["held-out"]
        offset = log.stat().st_size
        streams = http.parallel(port, [fixture, fixture], exact=True)
        steps = scheduler(log, offset, 4)
        native.write_json(out / "cohort.json", {"streams": streams, "rows": steps})
        assert max(check_step_budget(steps, cap)) == cap, steps
        verified_slots = {
            slot for step in steps for slot, entry in step.get("mtp_slots", {}).items()
            if entry["verified"] == 2
        }
        assert len(verified_slots) == 2, steps
        assert sum(sum(step["prefill"]) for step in steps) == 2 * len(fixture["prompt"]), steps


def check_wide_mtp(args):
    flags = ["--mtp-steps", "1"]
    with server(args, "wide-mtp", flags) as (port, log, out, health):
        check_caps(health, DEFAULT_ACTIVE, DEFAULT_TOKENS)
        cases = native.serial_cases(port, http.FIXTURES, out / "serial-reference.json")
        offset = log.stat().st_size
        native.check_admission(port, log, out / "queue-and-reuse.json", cases, DEFAULT_ACTIVE)
        steps = scheduler(log, offset, DEFAULT_ACTIVE)
        native.write_json(out / "steps.json", steps)
        check_step_budget(steps, DEFAULT_TOKENS)
        assert max(step.get("verification_lanes", 0) for step in steps) > 4, (
            "No scheduler iteration verified more than one group of lanes")
        for step in steps:
            boundaries = sum(entry["boundary"] for entry in step.get("mtp_slots", {}).values())
            assert step.get("verification_lanes", 0) + boundaries == step["decode"], step
        slots = {slot for step in steps for slot, entry in step.get("mtp_slots", {}).items()
                 if entry["drafted"] == 1 and entry["verified"] == 2}
        assert slots == set(range(DEFAULT_ACTIVE)), slots


def check_eager_cache(args):
    flags = ["--eager", "--max-active", "4"]
    case = prompt_case("eager-cache", CHECKPOINT_INTERVAL + 1)
    body = {"model": media.MODEL, "prompt": case["prompt"], "max_tokens": 8,
            "return_token_ids": True, "temperature": 0, "ignore_eos": True}
    prime = body | {"prompt": case["prompt"][:-1], "max_tokens": 1}
    with server(args, "eager-cache", flags, cache_entries=2) as (port, log, out, health):
        check_caps(health, 4, DEFAULT_TOKENS)
        evidence = argparse.Namespace(evidence=out)
        info_status, _, raw = media.exchange(port, "/server_info")
        info = json.loads(raw)
        native.write_json(out / "server-info.json", info)
        assert info_status == 200 and info["prefix_cache_entries"] == 2, info
        assert info["eager"] and not info["batch_cuda_graphs"] and info["mtp_steps"] == 4, info
        cold = media.capture(evidence, "cold", port, media.TEXT, body | {"cache_salt": "cold"})
        media.capture(evidence, "prime", port, media.TEXT, prime)
        warm = media.capture(evidence, "warm", port, media.TEXT, body)
        assert cold["usage"]["prompt_tokens_details"]["cached_tokens"] == 0, cold
        assert warm["usage"]["prompt_tokens_details"]["cached_tokens"] == CHECKPOINT_INTERVAL, warm
        for reply in (cold, warm):
            choice, = reply["choices"]
            assert choice["finish_reason"] == "length" and len(choice["token_ids"]) == 8, choice
        assert cold["choices"] == warm["choices"], (cold, warm)
        native.await_retirement(log, 0, 4)
        trace = log.read_bytes()
        assert b"batch graph_slots=" not in trace, log
        assert re.search(rb"batch mtp .* drafted=[1-9]\d* accepted=", trace), log


def check_cli(args):
    valid = [[], ["--max-active", "1"], ["--max-active", "4"],
             ["--max-active", str(DEFAULT_ACTIVE), TOKEN_OPTION, str(DEFAULT_TOKENS)],
             ["--max-active", str(MAX_ACTIVE)],
             [TOKEN_OPTION, str(MAX_TOKENS)],
             [TOKEN_OPTION, "8192"],
             [TOKEN_OPTION, str(args.small_tokens)],
             ["--max-active", "1", "--eager"], ["--max-active", "4", "--mtp-steps", "1"],
             ["--max-active", "4", "--mtp-steps", "1", "--eager"],
             ["--mtp-steps", "1"], ["--max-active", str(DEFAULT_ACTIVE), "--mtp-steps", "1"],
             ["--mtp-steps", "1", "--prefix-cache-entries", "2"],
             ["--eager", "--prefix-cache-entries", "2"],
             ["--max-active", str(DEFAULT_ACTIVE), "--mtp-k", "4"]]
    invalid = [([TOKEN_OPTION, value], TOKEN_OPTION)
               for value in ("0", "-1", "wide", str(MAX_TOKENS + 1), "2147483648")]
    invalid.extend([(["--max-active", "33"], "--max-active"),
                    (["--mtp-steps", "5"], "--mtp-steps"),
                    (["--mtp-k", "5"], "--mtp-steps")])
    command = [str(args.binary), "--checkpoint", str(args.evidence / "no-checkpoint")]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    records = []
    for flags, expected in [*((flags, NO_DEVICE) for flags in valid), *invalid]:
        result = subprocess.run([*command, *flags], env=env, capture_output=True, text=True,
                                timeout=15)
        passed = result.returncode == 1 and expected in result.stderr
        if expected != NO_DEVICE:
            passed = (passed and NO_DEVICE not in result.stderr and
                      "Invalid argument " not in result.stderr)
        records.append({"argv": [*command, *flags], "expected": expected, "exit": result.returncode,
                        "stderr": result.stderr, "passed": passed})
    native.write_json(args.evidence / "cli.json", records)
    assert all(record["passed"] for record in records), [
        record for record in records if not record["passed"]]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def build_fault_probe(args):
    out = args.evidence
    build = args.binary.parent.parent
    record = build / ".deps/sglang-server/linux/x86_64/release/sglang-server.d"
    sections = dict(re.findall(r"^    (files|values) = \{(.*?)^    \}", record.read_text(),
                               re.M | re.S))
    files = re.findall(r'"([^"\n]+)"', sections["files"])
    compiler, *flags = re.findall(r'"([^"\n]+)"', sections["values"])
    target = (args.source / "xmake.lua").read_text().split('target("sglang-server")', 1)[1]
    target = re.split(r"\n(?:option|target)\(", target, maxsplit=1)[0]
    sources = re.findall(r'"(apps/[^"]+\.cpp)"', target)
    objects = [next(name for name in files if name.endswith(f"/{source}.o")) for source in sources]
    assert objects and any("libsglang_decoder.a" in name for name in files), record
    baseline, wrapped, wrapper = (out / name for name in (
        "baseline-server", "fault-server", "kv_admission_fault.o"))
    wrapper_source = Path(__file__).resolve().with_name("kv_admission_fault.cpp")
    toolkit = Path(os.environ["CONDA_PREFIX"]) / "targets/x86_64-linux"
    compile_wrapper = [compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                       "-Iinclude", "-isystem", str(toolkit / "include"), "-c",
                       str(wrapper_source), "-o", str(wrapper)]
    command = [compiler, *objects]
    clean = [*command, *flags, "-o", str(baseline)]
    fault = [*command, str(wrapper), "-Wl,--wrap=cudaMalloc", "-Wl,--wrap=cudaFree",
             "-Wl,--wrap=cudaStreamSynchronize", *flags, "-o", str(wrapped)]
    native.write_json(out / "link.json", [compile_wrapper, clean, fault])
    for step in (compile_wrapper, clean, fault):
        subprocess.run(step, cwd=args.source, check=True)
    inputs = (args.binary, baseline, wrapped, wrapper_source, record,
              *(args.source / name for name in files))
    hashes = {str(path): digest(path) for path in inputs}
    native.write_json(out / "link-sha256.json", hashes)
    assert hashes[str(args.binary)] == hashes[str(baseline)], "Unmodified server link differs"
    return wrapped


def exchange(port, evidence, name, request=None):
    raw = None if request is None else json.dumps(request).encode()
    if raw is not None:
        (evidence / f"{name}.request").write_bytes(raw)
    with closing(HTTPConnection(native.LOOPBACK, port, timeout=45)) as connection:
        if request is None:
            connection.request("GET", "/health")
        else:
            connection.request("POST", "/v1/completions", raw, {
                "Accept": "text/event-stream",
                "Content-Type": "application/json",
                "Connection": "close",
            })
        with connection.getresponse() as response:
            status, body = response.status, response.read()
    (evidence / f"{name}.response").write_bytes(body)
    return status, body


def check_fault(port, evidence, marker, log, case):
    terminal = case != "kv-capacity"
    fixtures = http.FIXTURES
    short = native.request_body(fixtures["facts"], budget=1)
    fault = native.request_body({"prompt": [13, 198, 279, 826], "max_tokens": 1021})
    before = exchange(port, evidence, "before", short)
    assert before[0] == 200 and b"data: [DONE]" in before[1], before
    fixture = fixtures["held-out"]
    offset = log.stat().st_size
    with native.Stream(port, fixture, budget=512, timeout=45) as peer:
        peer.post()
        peer.open()
        peer.choice(fixture["output"][:1], first=True)
        marker.touch()
        try:
            status, body = exchange(port, evidence, "fault", fault)
        finally:
            marker.unlink()
        after = exchange(port, evidence, "new-request", short)
        health, caps = exchange(port, evidence, "post-fault-health")
        assert health == 200, health
        check_caps(json.loads(caps), DEFAULT_ACTIVE, DEFAULT_TOKENS)
        messages = log.read_text()
        assert messages.count("test: first KV payload allocated bytes=16777216\n") == 1
        code = 700 if case == "kv-context" else 2
        fault_line = f"test: second KV payload fault bytes=16777216 status={code}\n".encode()
        assert log.read_bytes().count(fault_line) == 1
        assert 500 <= status < 600 and b"[DONE]" not in body, (status, body)
        statuses = {"before": before[0], "fault": status, "fresh": after[0], "health": health}
        native.write_json(evidence / "statuses.json", statuses)
        if terminal:
            peer_tail = peer.response.read()
            (evidence / "peer-tail.sse").write_bytes(peer_tail)
            assert b"Native decoder failed" in peer_tail and b"[DONE]" not in peer_tail, peer_tail
            assert "test: terminal context fault persisted at synchronization\n" in log.read_text()
            assert (status, after[0]) == (500, 500), (
                "Terminal fault must fail the trigger and fresh request", statuses, body, after[1])
            assert b"[DONE]" not in after[1], after
        else:
            assert after == before, ("Fresh request must survive capacity failure", after)
            assert status == 503, (status, body)
            assert messages.count("test: partial KV payload freed\n") == 1
            peer.choice(fixture["output"][1:2])
            after_fault = log.read_bytes().split(fault_line, maxsplit=1)[1]
            assert any(int(decode) > 0 for _, decode, _ in native.TRACE.findall(after_fault)), (
                "No peer decode after the allocation fault")
            peer.reset()
        native.write_json(evidence / "peer.json", peer.evidence())
    if not terminal:
        native.await_retirement(log, offset, DEFAULT_ACTIVE)
        reuse = exchange(port, evidence, "after-cancel", short)
        assert reuse == before, ("Cancelled slot must allow reuse", reuse)
        native.await_retirement(log, offset, DEFAULT_ACTIVE)


def check_kv_fault(args, case):
    marker = args.evidence / case / "armed"
    environment = {**os.environ, "BATCH_KV_FAULT_MARKER": str(marker), "BATCH_KV_FAULT_MODE": case}
    flags = ["--mtp-steps", "0"]
    with server(args, case, flags, binary=args.fault_binary, environment=environment) as (
        port, log, evidence, health
    ):
        check_caps(health, DEFAULT_ACTIVE, DEFAULT_TOKENS)
        check_fault(port, evidence, marker, log, case)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("/data0/shared/Qwen3.8-27B-FP8"))
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--source", type=Path, help="Matching XMake project for the KV fault cases")
    parser.add_argument("--small-tokens", type=int, default=SMALL_TOKENS)
    checks = {"cli": check_cli, "default": check_default, "pressure": check_pressure,
              "width-1": lambda args: check_explicit(args, 1),
              "width-4": lambda args: check_explicit(args, 4), "bounded": check_bounded,
              "mtp": check_mtp, "wide-mtp": check_wide_mtp, "eager-cache": check_eager_cache}
    faults = ("kv-capacity", "kv-context", "kv-drain")
    parser.add_argument("--case", choices=[*checks, *faults], action="append")
    args = parser.parse_args()
    if not 8 <= args.small_tokens < 128:
        parser.error("--small-tokens must fit the mixed prompt and bind below the 128-row control")
    args.binary = args.binary.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.evidence = args.evidence.resolve()
    selected = args.case or [name for name in checks if name != "pressure"]
    has_fault = any(name in faults for name in selected)
    if has_fault and not args.source:
        parser.error("--source is required for the KV fault cases")
    args.evidence.mkdir()
    native.write_json(args.evidence / "identity.json", {
        "binary_sha256": digest(args.binary),
        "fixture_sha256": native.FIXTURE_DIGESTS,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cases": selected,
    })
    if has_fault:
        args.source = args.source.resolve()
        args.fault_binary = build_fault_probe(args)
    for name in selected:
        if name in faults:
            check_kv_fault(args, name)
        else:
            checks[name](args)
        print(f"Scheduler admission {name} passed", flush=True)


if __name__ == "__main__":
    main()
