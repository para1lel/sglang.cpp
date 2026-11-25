"""Replay TP2 eager all-reduce enqueue counters at both cold-cohort boundaries."""

import re

from saved_evidence import require


LEGACY_REASONS = ('fallback_disabled', 'fallback_setup_unavailable', 'fallback_oversize',
                  'fallback_unaligned', 'fallback_after_error')
REASONS = (*LEGACY_REASONS, 'fallback_capture_mismatch')
OUTCOMES = {'calls', 'mnnvl_oneshot', 'nccl', 'vendor_failed', 'bytes_payload', 'bytes_aggregated'}
COUNTERS = OUTCOMES.union(REASONS)
SERIES = re.compile(r'tp2_ar_([a-z_]+)_total\{engine="0",rank="([01])"\} ([0-9]+)')
WIDTH = re.compile(r'tp2_ar_width_(calls|payload_bytes)_total\{engine="0",rank="([01])",'
                   r'width="([1-9][0-9]*)"(?:,transport="(mnnvl|nccl)")?\} ([0-9]+)')
HIDDEN = 5120
CAP = 1024 * 1024


def snapshot(text, reasons=REASONS):
    counters = OUTCOMES.union(reasons)
    ranks = [{}, {}]
    widths = [{}, {}]
    for line in text.splitlines():
        if not line.startswith('tp2_ar_'):
            continue
        width = WIDTH.fullmatch(line)
        if width:
            name, rank, count, mode, value = width.groups()
            require(bool(mode) == (name == 'calls') and int(count) % HIDDEN == 0,
                    'Malformed width observation')
            values = widths[int(rank)].setdefault(int(count), {})
            name = mode or name
        else:
            series = SERIES.fullmatch(line)
            require(series, 'Malformed rank-local transport counter')
            name, rank, value = series.groups()
            require(name in counters, 'Unknown transport counter')
            values = ranks[int(rank)]
        require(name not in values, 'Duplicate transport counter')
        values[name] = int(value)
    for values, rows in zip(ranks, widths, strict=True):
        require(values.keys() == counters, 'Missing rank-local transport counters')
        require(values['vendor_failed'] == 0, 'A vendor enqueue failed in this server')
        require(values['calls'] == values['mnnvl_oneshot'] + values['nccl']
                and values['nccl'] == sum(values[name] for name in reasons),
                'Transport outcomes and fallback reasons disagree')
        for count, row in rows.items():
            require(row.keys() == {'nccl', 'mnnvl', 'payload_bytes'} and row['payload_bytes']
                    == (row['nccl'] + row['mnnvl']) * count * 2, 'Invalid width payload')
        require(sum(r['nccl'] for r in rows.values()) == values['nccl']
                and sum(r['mnnvl'] for r in rows.values()) == values['mnnvl_oneshot']
                and sum(r['payload_bytes'] for r in rows.values()) == values['bytes_payload']
                and values['bytes_aggregated'] == 2 * values['bytes_payload'],
                'Aggregate transport counters differ from actual width traffic')
    require(ranks[0] == ranks[1] and widths[0] == widths[1], 'Physical ranks disagree')
    return ranks[0], widths[0]


def replay(receipt, previous, *, batch_cuda_graphs, reasons=REASONS):
    require(batch_cuda_graphs is False, 'Enqueue counters cannot attest CUDA graph replays')
    before, cleared, after = (snapshot(receipt[name], reasons)
                              for name in ('before', 'cleared', 'after'))
    require(before == cleared and (previous is None or before == previous),
            'Transport counters changed outside the cold cohort')
    delta = {k: after[0][k] - cleared[0][k] for k in sorted(after[0])}
    widths = {}
    require(cleared[1].keys() <= after[1].keys(), 'Historical transport width disappeared')
    for count, end in after[1].items():
        initial = cleared[1].get(count, {})
        row = {k: end[k] - initial.get(k, 0) for k in end}
        require(min(row.values()) >= 0, 'Transport width regressed')
        if row['mnnvl'] + row['nccl']:
            widths[count] = row
    require(min(delta.values()) >= 0 and delta['calls'] > 0, 'Regressed or empty cold cohort')
    return after, delta, widths


def evidence(log, delta, widths, expected, *, expect_vendor):
    reasons = {'no_vmm', 'device_unsupported', 'multicast_create_failed'}
    setup = re.findall(r'^tp2_transport status=(mnnvl|nccl) ranks=2 devices=0,1 '
                       r'oneshot_max_bytes=(\d+) workspace_bytes_per_rank=(\d+) '
                       r'reason=(\w+)$', log, re.M)
    first = re.findall(r'^tp2_transport width=(\d+) transport=(mnnvl|nccl)$', log, re.M)
    lines = [line for line in log.splitlines() if line.startswith('tp2_transport ')]
    observed = {int(count): mode for count, mode in first}
    require(len(setup) <= 1 and len(observed) == len(first) and len(lines) == len(setup + first),
            'Malformed or duplicate transport log receipt')
    require(observed.keys() == widths.keys() == expected.keys(), 'Unexpected target widths')
    for count, row in widths.items():
        require(row['mnnvl'] + row['nccl'] == expected[count], 'Wrong boundary multiplicity')
        mode = 'mnnvl' if expect_vendor and count * 4 <= CAP else 'nccl'
        require(row[mode] == expected[count] and observed[count] == mode,
                'Transport selection disagrees with observed width and capacity')
    if expect_vendor:
        require(setup and setup[0][0:2] == ('mnnvl', str(CAP)) and setup[0][3] == 'ok'
                and int(setup[0][2]) >= CAP * 3 + 2048, 'Missing successful vendor setup')
        require(delta['mnnvl_oneshot'] and delta['nccl'] == delta['fallback_oversize'],
                'Unexpected vendor fallback')
    else:
        require(not setup or (setup[0][0:2] == ('nccl', str(CAP)) and setup[0][3] in reasons),
                'Unexpected setup status in NCCL control')
        reason = 'fallback_setup_unavailable' if setup else 'fallback_disabled'
        require(delta['nccl'] == delta['calls'] == delta[reason], 'Unexplained NCCL fallback')
    return {'setup': setup, 'first_use_by_elements': observed, 'aggregate_cap_bytes': CAP}
