"""Audit a real eager W1 transport A/B against the frozen native trace and wire corpus."""

import argparse
import csv
from collections import Counter
import gzip
import json
from pathlib import Path
import re
import shlex

from cohorts import cache_activity, client, cold_cache, load_lanes, reference, validate_cohort
from k4 import NATIVE_MTP_LINE, native_activity
from lmcache_evidence import CAPACITY_BYTES, FRAME, LMCACHE_SLC3_READY, LMCACHE_VERSION
from native import owners, rank_table
from saved_evidence import digest, load, require
import transport


TRACE_SHA = '1fee72670f7480f76273b75855ca64c8b8a01fa515ef25c2997e0c3adeeff9d7'
SSE_SHA = 'dbdb17e132e2310e7740d2aeb22bb09e5e361b62c8af91ddfb2f47bafc5e075b'
LEGACY_BINARIES = {'9c912b8012411627596886425fe3b1b7a2182019647b031bcf00dbf7dcf04c1e',
                   '7430d9c1cba05f884b8908e76df7ef7802cb351b6919a8de5340caaf66d8f3cb'}


def process_image(run, exe, pid, checksum, devices):
    stat = exe.stat()
    for when, name in (('before', 'compute.csv'), ('after', 'compute-end.csv')):
        proc = run / f'proc-exe-{when}'
        inode = f'{stat.st_dev}:{stat.st_ino}'
        if checksum not in LEGACY_BINARIES or proc.exists():
            require(proc.read_text().splitlines() == [str(exe.resolve()), inode,
                                                      f'{checksum}  /proc/{pid}/exe'],
                    'Running process path, inode or contents differ from the declared executable')
        with (run / name).open(newline='') as stream:
            rows = [row for row in csv.reader(stream, skipinitialspace=True) if row[1] == str(pid)]
        require(sorted(row[0] for row in rows) == sorted(devices)
                and all(Path(row[2]).is_absolute() and exe.samefile(row[2]) for row in rows),
                'NVML process image differs from the rank-local executable')


def executable(run, pid, devices):
    argv = shlex.split((run / 'server-command.sh').read_text())
    require(argv and Path(argv[0]).is_absolute(), 'Missing native executable path')
    exe = Path(argv[0])
    require((run / 'binary.sha256').read_text() == f'{exe}: OK\n'
            and (run / 'live-binary.sha256').read_text() == f'/proc/{pid}/exe: OK\n',
            'Missing prelaunch or live executable attestation')
    checksum = digest(exe)
    if checksum not in LEGACY_BINARIES:
        require((run / 'executable.sha256').read_text() == f'{checksum}  {exe}\n',
                'Wrong live native binary digest')
    process_image(run, exe, pid, checksum, devices)
    return checksum, argv


def verify_progress(text, requests):
    native_activity(text, require_groups=True)
    slots = {}
    for row in re.findall(r'^batch prefill_rows=([\d,]+) ', text, re.M):
        for slot, tokens in enumerate(map(int, row.split(','))):
            if tokens:
                require(slot not in slots, 'Duplicate W1 prompt slot')
                slots[slot] = 1  # The final prompt row produces the first returned token.
    for line in text.splitlines():
        match = NATIVE_MTP_LINE.fullmatch(line)
        if match:
            slot, _, accepted, _, _, boundary = map(int, match.groups())
            require(slot in slots and not boundary, 'Unexpected W1 verification slot or boundary')
            slots[slot] += accepted + 1
    require(sorted(slots.values()) == sorted(r['usage']['completion_tokens'] for r in requests),
            'Verification events do not explain every completed W1 lane')


def target_widths(text, *, grouped):
    widths = Counter(int(rows) for rows in re.findall(r'^batch verify lanes=\d+ rows=(\d+) ',
                                                      text, re.M) if int(rows))
    batches = [[int(n) for n in row.split(',') if int(n)] for row in
               re.findall(r'^batch prefill_rows=([\d,]+) ', text, re.M)]
    require([row for row in batches if row] == [[128, 128]], 'Unexpected W1 prefill grants')
    groups = re.findall(r'^batch layerwise_prefill_groups=(\d+) rows=(\d+)$', text, re.M)
    require(groups == ([('1', '256')] if grouped else []), 'Unexpected physical W1 prefill group')
    widths.update([256] if grouped else [128, 128])
    return {rows * transport.HIDDEN: steps * 128 for rows, steps in widths.items()}


def oracle(directory, lanes, policy):
    log, wire = directory / 'server.log', directory / 'trace.json.gz'
    require(digest(log) == TRACE_SHA and digest(wire) == SSE_SHA, 'Unpinned W1 target trace')
    with gzip.open(wire, 'rt') as stream:
        raw = json.load(stream)
    validate_cohort(raw, lanes, policy)
    verify_progress(log.read_text(), raw['requests'])
    outputs = [r['output_ids'] for r in raw['requests']]
    return target_widths(log.read_text(), grouped=False), outputs


def daemon(run, rank, gpu):
    record = load(run / f'peer-{rank}-before.json')
    require(record == load(run / f'peer-{rank}-after.json'), 'LMCache endpoint changed')
    namespace = f'sglang-r{rank}'
    version = LMCACHE_SLC3_READY.encode()
    endpoint = Path(record['socket'])
    require(endpoint.is_absolute() and endpoint.name == f'sglang-lmcache-r{rank}.sock'
            and record['namespace'] == namespace and record['wire'] == 'SLC3'
            and record['version'] == LMCACHE_VERSION
            and record['ready_version'] == LMCACHE_SLC3_READY
            and record['frame'] == (FRAME.pack(b'SLC3', 0, len(version)) + version).hex(),
            'Invalid authenticated READY frame or private ranked endpoint')
    require(record['peer_uid'] == run.stat().st_uid and type(record['peer_start_ticks']) is int
            and record['peer_start_ticks'] > 0 and record['peer_pid'] == record['group_pid'] > 1
            and re.fullmatch(r'\d+:[1-9]\d*', record['socket_id']), 'Invalid peer credentials')
    device = gpu[4:].replace('-', '').lower()
    message = (f'sglang GPU LMCache at {endpoint}, device {device}, '
               f'L1 capacity {CAPACITY_BYTES} bytes')
    log = (run / f'run-logs/daemon{rank}.log').read_text()
    require(log.count(message) == 1, 'Wrong daemon device, endpoint or L1 capacity')
    return record


def runtime(run):
    devices = [line.split()[1] for line in (run / 'reservation/gpu.txt').read_text().splitlines()]
    log = (run / 'run-logs/server.log').read_text()
    pid = re.search(r'^tp2 rank=0 device=0 uuid=[\da-fA-F-]+ pid=(\d+) ', log, re.M)
    require(pid and len(devices) == len(set(devices)) == 2, 'Two physical TP ranks are required')
    server = int(pid[1])
    ranks = rank_table(log, devices, server)
    checksum, argv = executable(run, server, devices)
    peers = [daemon(run, rank, device) for rank, device in enumerate(devices)]
    require(len({peer['socket_id'] for peer in peers}) == 2, 'Duplicated daemon socket')
    for name in ('compute.csv', 'compute-end.csv'):
        owners(run / name, devices, [p['peer_pid'] for p in peers], server)
    info = load(run / 'server-info.json')
    require(info['tp_size'] == 2 and info['mtp_steps'] == 4 and info['lmcache_enabled'] is True
            and info['lmcache_available'] is True, 'Expected native MTP4 and two live cache ranks')
    return dict(devices=devices, ranks=ranks, peers=peers, info=info, log=log,
                sha=checksum, argv=argv, pids={server, *(p['peer_pid'] for p in peers)})


def candidate_widths(run, clean, lanes, policy, golden, expect_vendor):
    traced = runtime(run)
    require(traced['devices'] == clean['devices'] and traced['info']['batch_cuda_graphs'] is False
            and traced['sha'] == clean['sha'] and traced['pids'].isdisjoint(clean['pids'])
            and traced['argv'] == [*clean['argv'], '--trace-batches'],
            'Planner trace and timed cohort must use the same ranks, binary, and arguments')
    wire = run / 'cohorts/trace.json.gz'
    with gzip.open(wire, 'rt') as stream:
        raw = json.load(stream)
    require((raw['row'], raw['tag']) == ('W1', 'trace'), 'Wrong planner trace cohort')
    validate_cohort(raw, lanes, policy)
    client.verify_round_ids(raw, golden)
    cached = cold_cache.replay(raw, cold_cache.LMCACHE_DRAIN)
    cache_activity.replay(raw['cold_cache'], cached, None)
    verify_progress(traced['log'], raw['requests'])
    expected = target_widths(traced['log'], grouped=True)
    _, delta, widths = transport.replay(raw['cold_cache'], None, batch_cuda_graphs=False)
    transport.evidence(traced['log'], delta, widths, expected, expect_vendor=expect_vendor)
    return expected


def verify(run, fixtures, trace, *, expect_vendor, planner_trace=None):
    clean = runtime(run)
    require('--trace-batches' not in clean['argv'], 'Timed W1 enabled diagnostic tracing')
    legacy = clean['sha'] in LEGACY_BINARIES
    reasons = transport.LEGACY_REASONS if legacy else transport.REASONS
    lanes = load_lanes(fixtures, 'W1')
    policy = reference.reference()['policy']
    expected, golden = oracle(trace, lanes, policy)
    if not legacy:
        require(planner_trace is not None, 'Current binary needs its own executed planner trace')
        expected = candidate_widths(planner_trace, clean, lanes, policy, golden, expect_vendor)
    previous = cached = transfers = None
    cohorts = {}
    for tag in ('warmup-01', 'measured-01'):
        path = run / 'cohorts' / f'{tag}.json.gz'
        with gzip.open(path, 'rt') as stream:
            raw = json.load(stream)
        require((raw['row'], raw['tag']) == ('W1', tag), 'Wrong transport workload')
        metrics = validate_cohort(raw, lanes, policy)
        client.verify_round_ids(raw, golden)
        cached = cold_cache.replay(raw, cold_cache.LMCACHE_DRAIN, cached)
        transfers = cache_activity.replay(raw['cold_cache'], cached, transfers)
        previous, delta, widths = transport.replay(
            raw['cold_cache'], previous, batch_cuda_graphs=clean['info']['batch_cuda_graphs'],
            reasons=reasons)
        dispatch = transport.evidence(clean['log'], delta, widths, expected,
                                      expect_vendor=expect_vendor)
        cohorts[tag] = {'raw_sha256': digest(path), 'both_rank_deltas': delta, 'widths': widths,
                        'dispatch': dispatch, 'makespan_ns': metrics['makespan_ns']}
    return {'ranks': clean['ranks'], 'peers': clean['peers'], 'cohorts': cohorts,
            'expected_boundaries': expected}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('fixtures', type=Path)
    parser.add_argument('trace', type=Path)
    parser.add_argument('expected', choices=['vendor', 'nccl'])
    parser.add_argument('--planner-trace', type=Path, help='Executed grouped W1 trace')
    args = parser.parse_args()
    result = verify(args.run, args.fixtures, args.trace, expect_vendor=args.expected == 'vendor',
                    planner_trace=args.planner_trace)
    print(json.dumps(result, indent=2))
