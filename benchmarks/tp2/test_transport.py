"""Reject missing, capacity-inconsistent or graph-stale TP2 transport receipts."""

import pytest

from transport import COUNTERS, LEGACY_REASONS, evidence, replay, snapshot
from verify_transport import process_image, target_widths, verify_progress


WIDTHS = {51200: 128, 655360: 256}
SETUP = ('tp2_transport status=mnnvl ranks=2 devices=0,1 oneshot_max_bytes=1048576 '
         'workspace_bytes_per_rank=4194304 reason=ok')
LOG = (SETUP + '\ntp2_transport width=51200 transport=mnnvl\n'
       'tp2_transport width=655360 transport=nccl')


def samples(widths):
    parts = []
    for rank in range(2):
        values = dict.fromkeys(COUNTERS, 0)
        for count, calls in widths.items():
            used = 'mnnvl' if count * 4 <= 1024 * 1024 else 'nccl'
            for mode in ('mnnvl', 'nccl'):
                label = f'engine="0",rank="{rank}",width="{count}",transport="{mode}"'
                parts.append(f'tp2_ar_width_calls_total{{{label}}} {calls if mode == used else 0}')
            label = f'engine="0",rank="{rank}",width="{count}"'
            payload = count * calls * 2
            parts.append(f'tp2_ar_width_payload_bytes_total{{{label}}} {payload}')
            values['bytes_payload'] += payload
            values['bytes_aggregated'] += payload * 2
            values['calls'] += calls
            values['mnnvl_oneshot' if used == 'mnnvl' else 'nccl'] += calls
            if used == 'nccl':
                values['fallback_oversize'] += calls
        for name, value in values.items():
            parts.append(f'tp2_ar_{name}_total{{engine="0",rank="{rank}"}} {value}')
    return '\n'.join(parts)


def test_widths_and_bytes_require_two_whole_matching_rank_receipts():
    text = samples(WIDTHS)
    zero = samples({})
    receipt = {'before': zero, 'cleared': zero, 'after': text}
    end, delta, widths = replay(receipt, None, batch_cuda_graphs=False)
    assert evidence(LOG, delta, widths, WIDTHS, expect_vendor=True)['setup']
    with pytest.raises(ValueError, match='outside the cold cohort'):
        replay(receipt, end, batch_cuda_graphs=False)
    with pytest.raises(ValueError, match='graph replays'):
        replay(receipt, None, batch_cuda_graphs=True)
    with pytest.raises(ValueError, match='Invalid width payload'):
        snapshot(text.replace('335544320', '112640', 1))
    with pytest.raises(ValueError, match='reasons disagree'):
        snapshot(text.replace('fallback_oversize_total{engine="0",rank="1"} 256',
                              'fallback_oversize_total{engine="0",rank="1"} 0'))
    with pytest.raises(ValueError, match='Missing rank-local'):
        snapshot('\n'.join(line for line in text.splitlines() if 'rank="1"' not in line))


def test_widths_must_match_independent_target_activity_and_setup():
    delta, widths = snapshot(samples(WIDTHS))
    with pytest.raises(ValueError, match='Wrong boundary multiplicity'):
        evidence(LOG, delta, widths, WIDTHS | {51200: 1}, expect_vendor=True)
    with pytest.raises(ValueError, match='Unexpected target widths'):
        evidence(LOG, delta, widths, {51200: 128}, expect_vendor=True)
    with pytest.raises(ValueError, match='selection disagrees'):
        evidence(LOG.replace('655360 transport=nccl', '655360 transport=mnnvl'), delta,
                 widths, WIDTHS, expect_vendor=True)
    with pytest.raises(ValueError, match='Missing successful'):
        evidence(LOG.replace('4194304', '8'), delta, widths, WIDTHS, expect_vendor=True)


def test_legacy_capture_counter_absence_cannot_downgrade_a_new_binary():
    revised = samples(WIDTHS)
    legacy = '\n'.join(line for line in revised.splitlines()
                       if 'fallback_capture_mismatch' not in line)
    snapshot(legacy, LEGACY_REASONS)
    with pytest.raises(ValueError, match='Missing rank-local'):
        snapshot(legacy)
    with pytest.raises(ValueError, match='Unknown transport counter'):
        snapshot(revised, LEGACY_REASONS)


def test_physical_prefill_group_requires_observed_layerwise_execution():
    grants = 'batch prefill_rows=0,128,128,0 workspace_bytes=100\n'
    receipt = 'batch layerwise_prefill_groups=1 rows=256\n'
    assert target_widths(grants, grouped=False) == {655360: 256}
    assert target_widths(grants + receipt, grouped=True) == {1310720: 128}
    with pytest.raises(ValueError, match='physical W1 prefill group'):
        target_widths(grants, grouped=True)
    with pytest.raises(ValueError, match='physical W1 prefill group'):
        target_widths(grants + receipt, grouped=False)


def test_truncated_or_unrelated_valid_verification_cannot_explain_full_w1_streams():
    grants = 'batch prefill_rows=128,128 workspace_bytes=100\n'
    header = 'batch verify lanes=2 rows=10 peak_request_bytes=100\n'
    events = ''.join(f'batch mtp slot={s} drafted=4 accepted=4 verified=5 replayed=0 boundary=0\n'
                     for s in (0, 1))
    requests = [{'usage': {'completion_tokens': 128}}] * 2
    with pytest.raises(ValueError, match='every completed W1 lane'):
        verify_progress(grants + header + events, requests)
    with pytest.raises(ValueError, match='verification slot'):
        verify_progress(grants + header + events.replace('slot=1', 'slot=5'), requests)


def test_nvml_and_live_process_must_attest_the_same_immutable_file(tmp_path):
    exe = tmp_path / 'server'
    exe.write_bytes(b'actual')
    link = tmp_path / 'installed-server'
    link.symlink_to(exe)
    info = exe.stat()
    digest = 'x' * 64
    live = f'{exe}\n{info.st_dev}:{info.st_ino}\n{digest}  /proc/7/exe\n'
    for when, csv_name in (('before', 'compute.csv'), ('after', 'compute-end.csv')):
        (tmp_path / f'proc-exe-{when}').write_text(live)
        (tmp_path / csv_name).write_text(f'GPU-0, 7, {link}, 1 MiB\nGPU-1, 7, {exe}, 1 MiB\n')
    process_image(tmp_path, exe, 7, digest, ['GPU-0', 'GPU-1'])
    (tmp_path / 'compute-end.csv').write_text(f'GPU-0, 7, /bin/true, 1 MiB\nGPU-1, 7, {exe}, 1\n')
    with pytest.raises(ValueError, match='NVML process image'):
        process_image(tmp_path, exe, 7, digest, ['GPU-0', 'GPU-1'])
