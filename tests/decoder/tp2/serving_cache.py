"""Check rank-local direct prefix transfers and generated IDs on a private daemon pair."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, closing, contextmanager
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import monotonic, sleep

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "tests"), str(ROOT / "tests" / "prefix"),
                str(ROOT / "tests" / "text")]
from lmcache_relay import DRAFT, GDN, GQA, PREFIX_DETAIL, RANK_GET, RANK_PUT, READY, RESET
from lmcache_relay import Relay, control
from media import http_routes as http


def free_port():
    with closing(socket.socket()) as sock:
        sock.bind(("127.0.0.1", 0))
        return str(sock.getsockname()[1])


@contextmanager
def daemon(args, work, gpu, rank):
    socket_path = work / f"daemon-r{rank}.sock"
    log_path = args.evidence / f"daemon-r{rank}.log"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, XDG_RUNTIME_DIR=str(work),
               PYTHONPATH=os.environ.get("LMCACHE_EXTRA_PYTHONPATH", ""))
    command = ["bash", str(ROOT / "tools/lmcache/serve.sh"), "--rank", str(rank), "--socket",
               str(socket_path), "--port", free_port(), "--memory-gib", "1"]
    with log_path.open("wb") as out:
        proc = subprocess.Popen(command, env=env, stdout=out, stderr=subprocess.STDOUT,
                                start_new_session=True)
        try:
            deadline = monotonic() + 60
            while monotonic() < deadline and proc.poll() is None:
                try:
                    reply = control(socket_path, f"sglang-r{rank}", READY)
                    if reply == (0, b"0.5.8+native-v3"):
                        break
                except (FileNotFoundError, ConnectionRefusedError):
                    pass
                sleep(0.1)
            else:
                raise RuntimeError(f"Daemon {rank} failed readiness, see {log_path}")
            yield socket_path
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()


def capture(args, port, phase, prompt, count, cached, salt):
    body = http.native_text(prompt, count) | {"cache_salt": salt}
    result = http.capture(args, phase, port, http.GENERATE, body)
    return http.native_result(result, prompt, count, cached)[:2]


def receipt(events, rank, gpu, opcode, endpoint):
    got = [entry for entry in events if entry["opcode"] == opcode]
    assert got and all(e["status"] == 0 for e in got), (rank, opcode, got)
    total = [sum(e["receipt"][i] for e in got) for i in range(4)]
    expected = [16896 * endpoint, 76992512, (endpoint - 1) * 2112 if rank == 0 else 0,
                endpoint if opcode == RANK_PUT else 0]
    assert total == expected, (rank, opcode, total, expected)
    families = {key[0] for e in got for key in e["keys"]}
    assert families == ({GQA, GDN, DRAFT} if rank == 0 else {GQA, GDN}), (rank, families)
    for e in got:
        assert e["device"] == gpu.removeprefix("GPU-").replace("-", "")
        assert e["namespace"] == f"sglang-r{rank}"
        assert e["allocations"] > 0 and e["forwarded"]
    return total


def metrics(port):
    status, _, raw = http.exchange(port, "/metrics")
    assert status == 200
    marker = '{engine="0"} '
    values = {}
    for line in raw.decode().splitlines():
        if line.startswith("sglang_cpp:lmcache_") and marker in line:
            key, value = line.removeprefix("sglang_cpp:lmcache_").split(marker)
            values[key] = int(value)
    return values


def pending(port):
    return metrics(port)["pending_stores"]


def check_counters(before, after, totals, action):
    direction = "read" if action == "get" else "written"
    operation = "restores" if action == "get" else "stores"
    for rank in range(2):
        key = f"rank{rank}_{operation}_total"
        assert after[key] - before[key] == 1, (key, before[key], after[key])
        for i, family in enumerate(("gqa", "gdn", "draft")):
            key = f"rank{rank}_{family}_{direction}_bytes_total"
            expected = totals[f"rank{rank}_{action}"][i]
            assert after[key] - before[key] == expected, (key, expected, before[key], after[key])
    for i, family in enumerate(("gqa", "gdn", "draft")):
        key = f"{family}_{direction}_bytes_total"
        expected = sum(totals[f"rank{rank}_{action}"][i] for rank in range(2))
        assert after[key] - before[key] == expected, (key, expected, before[key], after[key])


def drain(port):
    deadline = monotonic() + 20
    while monotonic() < deadline:
        if pending(port) == 0:
            return
        sleep(0.02)
    raise TimeoutError("Pending LMCache stores did not drain")


def delayed_store(args, port, prefix, relays, offsets, baseline, gpu):
    barrier = relays[1].arm(RANK_PUT)
    with ThreadPoolExecutor(max_workers=1) as worker:
        prime = worker.submit(capture, args, port, "prime", prefix, 1, 0, "subject")
        try:
            barrier.wait()
            deadline = monotonic() + 10
            while monotonic() < deadline:
                completed = relays[0].since(offsets[0])
                midpoint = metrics(port)
                if (any(e["opcode"] == RANK_PUT and e["forwarded"] for e in completed)
                        and midpoint["rank0_draft_written_bytes_total"]
                        > baseline["rank0_draft_written_bytes_total"]):
                    break
                sleep(0.02)
            else:
                raise AssertionError("Rank 0 store did not retire before rank 1")
            physical = receipt(completed, 0, gpu, RANK_PUT, len(prefix))
            deltas = {key: value - baseline[key] for key, value in midpoint.items()}
            expected = {"pending_stores": 1, "rank0_stores_total": 1, "rank1_stores_total": 0,
                        "stores_total": 0, "gqa_written_tokens_total": 0,
                        "declined_stores_total": 0, "io_errors_total": 0}
            for index, family in enumerate(("gqa", "gdn", "draft")):
                expected[f"rank0_{family}_written_bytes_total"] = physical[index]
                expected[f"rank1_{family}_written_bytes_total"] = 0
                expected[f"{family}_written_bytes_total"] = 0
            observed = {key: deltas[key] for key in expected}
            assert observed == expected, (observed, expected)
        finally:
            barrier.release.set()
        prime.result()
    drain(port)
    return {"held": midpoint["pending_stores"], "drained": pending(port), "deltas": observed}


def exercise(args, work, devices, sockets, relays):
    path = str(work / "relay.sock")
    prefix = [760, 6511, 314, 9338, 369, 220, 17, 478, 220, 17, 283, 13, 198]
    prompt = prefix + [760, 6511, 314, 6124, 369]
    with http.serve(args, "ranked-cache", "--tp", "2", "--eager", "--mtp-steps", "4",
                    "--lmcache-socket", path) as (port, _):
        status, _, raw = http.exchange(port, "/server_info")
        (args.evidence / "server-info.json").write_bytes(raw)
        info = json.loads(raw)
        assert status == 200 and info["tp_size"] == info["tensor_parallel"] == 2
        assert info["lmcache_available"] and info["lmcache_enabled"] and info["mtp_steps"] == 4
        if not args.store_only:
            cold = capture(args, port, "cold", prompt, 8, 0, "reference")
            drain(port)
        offsets = [len(relay.since()) for relay in relays]
        pre_prime = metrics(port)
        held = delayed_store(args, port, prefix, relays, offsets, pre_prime, devices[0])
        totals = {"pending_store": held}
        for rank, relay in enumerate(relays):
            totals[f"rank{rank}_store"] = receipt(relay.since(offsets[rank]), rank,
                                                  devices[rank], RANK_PUT, len(prefix))
        pre_restore = metrics(port)
        check_counters(pre_prime, pre_restore, totals, "store")
        assert pre_restore["stores_total"] - pre_prime["stores_total"] == 1
        token_counter = "gqa_written_tokens_total"
        assert pre_restore[token_counter] - pre_prime[token_counter] == len(prefix)
        if args.store_only:
            (args.evidence / "rank-receipts.json").write_text(json.dumps(totals, indent=2) + "\n")
            (args.evidence / "metrics.json").write_text(json.dumps({
                "baseline": pre_prime, "settled": pre_restore,
            }, indent=2) + "\n")
            return
        offsets = [len(relay.since()) for relay in relays]
        hit = capture(args, port, "restored", prompt, 8, len(prefix), "subject")
        drain(port)
        assert hit == cold, (hit, cold)
        for rank, relay in enumerate(relays):
            events = relay.since(offsets[rank])
            assert any(e["opcode"] == PREFIX_DETAIL and e["status"] == 0 for e in events)
            totals[f"rank{rank}_get"] = receipt(events, rank, devices[rank], RANK_GET, len(prefix))
        post_restore = metrics(port)
        check_counters(pre_restore, post_restore, totals, "get")
        (args.evidence / "metrics.json").write_text(json.dumps({
            "pre_prime": pre_prime, "pre_restore": pre_restore, "post_restore": post_restore,
        }, indent=2) + "\n")
        committed = prompt + hit[0][:-1]
        continued = committed + [220, 17]
        reference = capture(args, port, "tail-cold", continued, 4, 0, "tail-reference")
        drain(port)
        offsets = [len(relay.since()) for relay in relays]
        tail = capture(args, port, "tail-restored", continued, 4, len(committed), "subject")
        assert tail == reference, (tail, reference)
        drain(port)
        for rank, relay in enumerate(relays):
            totals[f"rank{rank}_tail_get"] = receipt(relay.since(offsets[rank]), rank,
                                                     devices[rank], RANK_GET, len(committed))
        assert control(sockets[1], "sglang-r1", RESET)[0] == 0
        fallback = capture(args, port, "missing-rank", prompt, 8, 0, "subject")
        assert fallback == cold, (fallback, cold)
        drain(port)
        (args.evidence / "rank-receipts.json").write_text(json.dumps(totals, indent=2) + "\n")


def interrupted(signum, frame):
    raise SystemExit(128 + signum)


def main():
    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store-only", action="store_true")
    for key in ("server", "checkpoint", "evidence"):
        parser.add_argument(key, type=Path)
    args = parser.parse_args()
    devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    assert len(devices) == 2 and devices[0] != devices[1]
    args.evidence.mkdir(exist_ok=True, parents=True)
    with TemporaryDirectory(prefix="tp2-serving-", dir="/tmp") as directory, ExitStack() as stack:
        work = Path(directory)
        sockets = [stack.enter_context(daemon(args, work, gpu, i)) for i, gpu in enumerate(devices)]
        relays = [stack.enter_context(Relay(work / f"relay-r{r}.sock", p))
                  for r, p in enumerate(sockets)]
        try:
            exercise(args, work, devices, sockets, relays)
        finally:
            for rank, relay in enumerate(relays):
                events = {"events": relay.since(), "errors": relay.errors}
                (args.evidence / f"relay-r{rank}.json").write_text(
                    json.dumps(events, default=bytes.hex, indent=2) + "\n")
        assert not any(relay.errors for relay in relays)


if __name__ == "__main__":
    main()
