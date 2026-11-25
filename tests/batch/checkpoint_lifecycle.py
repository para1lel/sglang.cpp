"""Exercise public checkpoint cadence and real K4 session continuation on a reserved B200."""

import argparse
from contextlib import closing
from http.client import HTTPConnection
import json
import os
from pathlib import Path
import re
import socket
import struct
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from media import http_routes as http
from live_controls import Controls, STORES


STEP = 8192
DRAFT_ROWS = "sglang_cpp:lmcache_draft_restored_rows_total"
WRITTEN = ("sglang_cpp:lmcache_gqa_written_tokens_total",
           "sglang_cpp:lmcache_gqa_written_bytes_total",
           "sglang_cpp:lmcache_gdn_written_bytes_total",
           "sglang_cpp:lmcache_draft_written_bytes_total")
PREFILL = re.compile(rb"(?m)^batch prefill_rows=([\d,]+) workspace_bytes=")
PACKET = re.compile(rb"(?m)^batch mtp slot=\d+ drafted=(\d+) accepted=(\d+) verified=(\d+)")
ROOT = Path(__file__).resolve().parents[2]


def fixture(row):
    return json.loads((ROOT / f"benchmarks/streaming/{row}.json").read_text())["prompt_token_ids"]


def settled(cache, label, expected=None):
    metrics = cache.wait_for(label, 0, 0, stores=0 if expected is None else expected)
    if expected is not None:
        assert metrics[STORES] == expected, (label, expected, metrics)
    assert metrics["sglang_cpp:lmcache_io_errors_total"] == 0, (label, metrics)
    assert metrics["sglang_cpp:lmcache_declined_stores_total"] == 0, (label, metrics)
    return metrics


def reset(cache, label):
    settled(cache, f"{label}-before-reset")
    cache.reset(f"{label}-reset")


def generate(args, port, label, body):
    raw = http.capture(args, label, port, http.GENERATE, body)
    if not body.get("stream"):
        return raw
    *partial, last = http.stream_frames(raw)
    assert all(frame["meta_info"]["finish_reason"] is None for frame in partial)
    assert all(last["output_ids"][:len(frame["output_ids"])] == frame["output_ids"]
               for frame in partial)
    return last


def snapshot_steps(args, log, label, offset):
    trace = log.read_bytes()[offset:]
    (args.evidence / f"{label}.batch.log").write_bytes(trace)
    return trace


def open_session(args, port, session):
    reply = http.capture(args, f"open-{session}", port, "/open_session",
                         {"session_id": session, "capacity_of_str_len": STEP * 3})
    assert reply == session, reply


def turn(ids, session, rid, budget, parent=None, *, stream=False, **options):
    body = http.native_text(ids, budget)
    body["sampling_params"].update(options)
    params = {"id": session}
    if parent is not None:
        params["rid"] = parent
    return body | {"rid": rid, "session_params": params, "stream": stream}


def test_w1(args, cache, log, prompt):
    port = cache.port
    reset(cache, "w1")
    open_session(args, port, "w1")
    before = settled(cache, "w1-before")[STORES]
    start = log.stat().st_size
    w1 = generate(args, port, "w1-parent", turn(prompt, "w1", "w1-parent", 128, stream=True))
    output, _, _ = http.native_result(w1, prompt, 128, 0)
    stored = settled(cache, "w1-after-done", before + 1)
    trace = snapshot_steps(args, log, "w1-parent", start)
    assert b"batch width=1 decode=1 request_bytes=0\n" in trace, trace
    assert any(int(draft) == 4 and int(rows) == 5 for draft, _, rows in PACKET.findall(trace))
    replay = generate(args, port, "w1-raw-replay", http.native_text(prompt, 128))
    replay_ids, _, _ = http.native_result(replay, prompt, 128, 0)
    assert replay_ids == output, (replay_ids, output)
    replayed = settled(cache, "w1-after-replay", stored[STORES] + 1)
    # LMCache acknowledges the duplicate terminal PUT with no new immutable records.
    for key in WRITTEN:
        assert replayed[key] == stored[key], (key, stored[key], replayed[key])
    for branch, suffix in (("a", [13, 220, 32]), ("b", [11, 220, 33])):
        body = turn(suffix, "w1", f"w1-branch-{branch}", 8, "w1-parent", stream=branch == "a")
        assembled = prompt + output + suffix
        before_draft = settled(cache, f"before-draft-{branch}")[DRAFT_ROWS]
        start = log.stat().st_size
        resumed = generate(args, port, f"w1-branch-{branch}", body)
        resumed_ids, _, _ = http.native_result(resumed, assembled, 8, 255)
        after = settled(cache, f"after-draft-{branch}")
        trace = snapshot_steps(args, log, f"w1-branch-{branch}", start)
        assert sum(sum(map(int, rows.split(b","))) for rows in PREFILL.findall(trace)) == 4
        assert after[DRAFT_ROWS] == before_draft + 254, after
        cold = generate(args, port, f"w1-branch-{branch}-cold",
                        http.native_text(assembled) | {"cache_salt": f"cold-{branch}"})
        http.native_result(cold, assembled, 8, 0)
        assert resumed_ids == cold["output_ids"], (resumed_ids, cold["output_ids"])


def periodic(args, cache, log, base):
    port = cache.port
    for p, g, count in ((STEP - 1, 1, 1), (STEP, 1, 1), (STEP + 1, 1, 2),
                        (STEP - 1, 8, 2), (STEP * 2 + 1, 1, 3)):
        label = f"period-{p}-{g}"
        reset(cache, label)
        open_session(args, port, label)
        before = settled(cache, f"{label}-before")[STORES]
        offset = log.stat().st_size
        prompt = base[:p]
        parent = generate(args, port, label, turn(prompt, label, label, g))
        output, _, _ = http.native_result(parent, prompt, g, 0)
        settled(cache, f"{label}-after", before + count)
        trace = snapshot_steps(args, log, label, offset)
        if g > 1:
            packets = PACKET.findall(trace)
            assert packets[0] == (b"0", b"0", b"1"), (label, packets)
            assert any(draft == b"4" and rows == b"5" for draft, _, rows in packets), packets
            assert b"batch width=1 decode=1 request_bytes=0\n" in trace, trace
        suffix = [31] if output[-1] != 31 else [32]
        child = generate(args, port, f"{label}-child", turn(suffix, label, label + "-child", 1,
                                                          label))
        http.native_result(child, prompt + output + suffix, 1, p + g - 1)
        processed = prompt + output[:-1]
        # Probe deepest first so a probe cannot create a missing shallower checkpoint.
        for cutoff in reversed(range(STEP, len(processed), STEP)):
            off_branch = processed[:cutoff] + [31 if processed[cutoff] != 31 else 32]
            branch = generate(args, port, f"{label}-boundary-{cutoff}",
                              http.native_text(off_branch, 1))
            http.native_result(branch, off_branch, 1, cutoff)


def abort_turn(args, cache, log, prompt, *, disconnect, retained=0):
    port = cache.port
    key = "disconnect" if disconnect else "abort"
    open_session(args, port, key)
    before = settled(cache, f"{key}-before")[STORES]
    job = turn(prompt, key, key, 1024, stream=True) | {"cache_salt": key}
    with closing(HTTPConnection("127.0.0.1", port, timeout=90)) as connection:
        connection.connect()
        transport = connection.sock
        transport.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
        connection.request("POST", http.GENERATE, json.dumps(job), http.HEADERS)
        with connection.getresponse() as response:
            assert response.status == 200
            assert response.headers.get_content_type() == "text/event-stream"
            line = response.readline()
            while not line.startswith(b"data: "):
                assert line, "EOF before native output"
                line = response.readline()
            frame = json.loads(line[6:])
            assert frame["output_ids"] and frame["meta_info"]["finish_reason"] is None, frame
            (args.evidence / f"{key}-first.sse").write_bytes(line)
            offset = log.stat().st_size
            if disconnect:
                transport.close()
            else:
                status, _, body = http.exchange(port, "/abort_request", {"rid": key})
                assert status == 200 and not body, (status, body)
                raw = line + response.read()
                (args.evidence / "abort-terminal.sse").write_bytes(raw)
                assert http.stream_frames(raw)[-1]["meta_info"]["finish_reason"]["type"] == "abort"
    http.wait_retired(log, offset)
    settled(cache, f"{key}-after", before + bool(retained))
    child = turn([31], key, f"{key}-child", 1, key)
    error = http.capture(args, f"{key}-unusable-parent", port, http.GENERATE, child, 400)
    assert "Invalid request session id" in error["error"]["message"], error
    probe = http.native_text(prompt + frame["output_ids"] + [31], 1) | {"cache_salt": key}
    reply = generate(args, port, f"{key}-unusable-checkpoint", probe)
    http.native_result(reply, probe["input_ids"], 1, retained)


def mid_packet_stop(args, cache, log, prompt):
    port = cache.port
    reset(cache, "stop")
    offset = log.stat().st_size
    baseline = generate(args, port, "stop-baseline", http.native_text(prompt, 12))
    ids, _, _ = http.native_result(baseline, prompt, 12, 0)
    baseline_trace = snapshot_steps(args, log, "stop-baseline", offset)
    draft, accepted, rows = map(int, PACKET.findall(baseline_trace)[0])
    assert (draft, rows) == (4, 5) and accepted >= 3, baseline_trace
    prefixes = http.capture(args, "stop-pieces", port, "/detokenize", {
        "tokens": [ids[:n] for n in range(1, 5)], "skip_special_tokens": False})["text"]
    matches = {}
    for count in (3, 4):
        text, previous = prefixes[count - 1], prefixes[count - 2]
        stop = text[-12:]
        assert len(stop) >= 2 and stop not in previous and text.startswith(previous), prefixes
        matches[count] = stop

    for count, other in ((4, 3), (3, 4)):
        session = f"stop-{count}"
        stop = matches[count]
        open_session(args, port, session)
        before = settled(cache, f"{session}-before")[STORES]
        offset = log.stat().st_size
        parent = turn(prompt, session, f"{session}-parent", 12, stream=True,
                      stop=[stop, matches[other]]) | {"cache_salt": session}
        final = generate(args, port, f"{session}-mid-packet", parent)
        assert final["output_ids"] == ids[:count], final
        assert final["meta_info"]["finish_reason"] == {"type": "stop", "matched": stop}, final
        visible = prefixes[count - 1].split(stop, 1)[0]
        assert final["text"] == visible, final
        assert final["meta_info"]["completion_tokens"] == count, final
        settled(cache, f"{session}-after", before + 1)
        trace = snapshot_steps(args, log, f"{session}-mid-packet", offset)
        packet, = PACKET.findall(trace)
        assert tuple(map(int, packet)) == (4, count - 1, 5), trace

        suffix = [31 if ids[count] != 31 else 32]
        child = turn(suffix, session, f"{session}-child", 8, f"{session}-parent",
                     stream=True) | {"cache_salt": session}
        reply = generate(args, port, f"{session}-continuation", child)
        assembled = prompt + ids[:count] + suffix
        cached = len(prompt) + count - 1
        resumed, _, _ = http.native_result(reply, assembled, 8, cached)
        cold = generate(args, port, f"{session}-continuation-cold",
                        http.native_text(assembled) | {"cache_salt": f"cold-{session}"})
        cold_ids, _, _ = http.native_result(cold, assembled, 8, 0)
        assert resumed == cold_ids, (resumed, cold_ids)

        rejected = http.native_text(prompt + ids[:6] + [33], 1) | {"cache_salt": session}
        reply = generate(args, port, f"{session}-past-cutoff", rejected)
        http.native_result(reply, rejected["input_ids"], 1, cached)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("server", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--lmcache-socket", type=Path, required=True,
                        help="socket of a dedicated LMCache daemon whose namespace may be reset")
    args = parser.parse_args()
    endpoint = args.lmcache_socket.resolve()
    runtime = Path(os.environ.get("XDG_RUNTIME_DIR", ""))
    defaults = [Path(f"/tmp/sglang-lmcache-{os.geteuid()}.sock")]
    if runtime.is_absolute():
        defaults.append(runtime / "sglang-lmcache.sock")
    if not endpoint.is_socket() or any(p.exists() and endpoint.samefile(p) for p in defaults):
        parser.error("--lmcache-socket must name a live, dedicated, non-default socket")
    args.evidence.mkdir()
    prompt = fixture("W1")
    with http.serve(args, "checkpoint-lifecycle", "--lmcache-socket", str(endpoint)) as (port, log):
        http.server_policy(args, port, "default", True)
        cache = Controls(port, args.evidence)
        test_w1(args, cache, log, prompt)
        periodic(args, cache, log, fixture("W2") * 9)
        abort_turn(args, cache, log, prompt, disconnect=False)
        abort_turn(args, cache, log, fixture("W2") * 4 + [31], disconnect=True, retained=STEP)
        mid_packet_stop(args, cache, log, prompt)
        settled(cache, "lifecycle-complete")
    print("Public GDN checkpoint cadence, cancellation, and K4 continuation passed.")


if __name__ == "__main__":
    main()
