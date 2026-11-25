#!/usr/bin/env python3
"""Trace only the max-active=1 native worker's SSE sendto calls on Linux."""

from copy import copy
from contextlib import contextmanager
from decimal import Decimal
import json
from pathlib import Path
import os
import re
import sys
from time import monotonic_ns, time_ns


@contextmanager
def traced_worker(args, relay_path):
    from media import http_routes as http

    prefix = args.evidence.resolve() / "single-slot.sendto"
    launcher = copy(args)
    launcher.server = Path(__file__)
    before = time_ns() - monotonic_ns()
    with http.server_process(
            launcher, "single-slot", "--lmcache-socket", str(relay_path), "--max-active", "1",
            str(prefix), str(args.server.resolve())) as (port, _, _):
        yield port, prefix
    # strace reports syscall entry on CLOCK_REALTIME. Reject a clock step during the run.
    assert abs(time_ns() - monotonic_ns() - before) < 1_000_000, "Realtime clock offset changed"


def terminal_times(prefix, label):
    calls = []
    request_id = json.dumps('"id":"' + label + '"')[1:-1]
    terminal = r'\"finish_reason\":{'
    for path in prefix.parent.glob(prefix.name + ".*"):
        for line in path.read_text().splitlines():
            if request_id not in line or terminal not in line or "data: " not in line:
                continue
            sent = re.fullmatch(r'(\d+\.\d+) sendto\(\d+, ".*", (\d+), MSG_NOSIGNAL, '
                                r'NULL, 0\) += (\d+) <(\d+\.\d+)>', line)
            assert sent and sent[2] == sent[3], (label, line)
            start = int(Decimal(sent[1]) * 10**9)
            duration = int(Decimal(sent[4]) * 10**9)
            calls.append((start, start + duration))
    (call,) = calls
    return call


def require_deferred_send(label, start, opened):
    assert start > opened, (label, "terminal send preceded STORE barrier release", start, opened)


if __name__ == "__main__":
    prefix, server = sys.argv[-2:]
    # -D keeps the native process the harness's direct child for readiness and termination.
    # Separate logs avoid split syscall lines when request threads overlap.
    os.execvp("strace", ["strace", "-D", "-ff", "--seccomp-bpf", "-qq", "-s", "4096",
                         "-e", "trace=sendto", "--absolute-timestamps=unix,ns",
                         "--syscall-times=ns", "-o", prefix, server, *sys.argv[1:-2]])
