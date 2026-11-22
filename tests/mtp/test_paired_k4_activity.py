"""Validate stock batched MTP trace events during offline replay."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_paired_identity import k4, write_k4_row, write_json


HEADER = "batch verify lanes=1 rows=5 peak_request_bytes=4096"
FULL = "batch mtp slot=4 drafted=4 accepted=4 verified=5 replayed=0 boundary=0"
PARTIAL = "batch mtp slot=7 drafted=2 accepted=1 verified=3 replayed=2 boundary=0"
BOUNDARY = "batch mtp slot=15 drafted=0 accepted=0 verified=0 replayed=0 boundary=1"


def test_native_trace_mixed_lanes_and_short_rounds():
    trace = "\n".join((
        "batch width=3 decode=3 request_bytes=4096",
        "batch verify lanes=2 rows=8 peak_request_bytes=4096",
        FULL, PARTIAL, BOUNDARY,
        "batch width=2 decode=2 request_bytes=4096",
        "batch verify lanes=2 rows=4 peak_request_bytes=4096",
        "batch mtp slot=4 drafted=0 accepted=0 verified=1 replayed=0 boundary=0",
        "batch mtp slot=7 drafted=3 accepted=3 verified=3 replayed=0 boundary=0",
        "batch width=0 decode=0 request_bytes=0",
    ))
    assert k4.native_activity(trace, True) == {"rounds": 5, "depth_four_rounds": 1}


def test_unframed_complete_event_remains_readable_in_legacy_replay():
    assert k4.native_activity(FULL, False) == {"rounds": 1, "depth_four_rounds": 1}
    with pytest.raises(ValueError, match="has no verification header"):
        k4.native_activity(FULL, True)


@pytest.mark.parametrize(("partial", "message"), [
    ("batch mtp slot=4 drafted=4 accepted=4 verified=5", "Malformed native MTP event"),
    (FULL.replace("verified=5", "verified=5.25"), "Malformed native MTP event"),
    (FULL + " trailing", "Malformed native MTP event"),
    (FULL.replace("slot=4", "slot=32"), "Invalid native MTP slot"),
    (FULL.replace("accepted=4", "accepted=5"), "Invalid native MTP round"),
    (FULL.replace("drafted=4", "drafted=3"), "Invalid native MTP round"),
    (FULL.replace("replayed=0", "replayed=1"), "Invalid native MTP round"),
    (FULL.replace("boundary=0", "boundary=1"), "Invalid native MTP boundary"),
])
def test_offline_rejects_invalid_event_with_matching_activity_summary(tmp_path, partial, message):
    write_k4_row(tmp_path, opt_in=True, receipt_version=1)
    assert k4.assess("W1", tmp_path, tmp_path)["passed"]
    trace = tmp_path / "native-k4-trace"
    (trace / "server.log").write_text(f"{HEADER}\n{partial}\n")
    write_json(trace / "mtp-activity.json", {"rounds": 1, "depth_four_rounds": 1})
    with pytest.raises(ValueError, match=message):
        k4.assess("W1", tmp_path, tmp_path)


@pytest.mark.parametrize(("trace", "message"), [
    (f"{HEADER}\n{FULL}\n{HEADER}\n", "Incomplete verification group"),
    (f"{HEADER}\n{FULL}\n{HEADER}\n{PARTIAL}\n", "Incomplete verification group"),
    (f"{HEADER}\n{FULL}\n{FULL}\n", "repeats a slot"),
    (f"{HEADER}\n{FULL}\n{PARTIAL}\n", "Excess verification events"),
    (f"{HEADER}\n{FULL}\n{HEADER.replace('rows=5', 'rows=5.25')}\n",
     "Malformed verification header"),
    (f"{HEADER.replace('rows=5', 'rows=6')}\n{FULL}\n", "Invalid verification header counts"),
    (f"{HEADER.replace('rows=5', 'rows=3')}\n{PARTIAL}\n{BOUNDARY}\n",
     "No native depth-four verification"),
])
def test_native_trace_rejects_inconsistent_group(trace, message):
    with pytest.raises(ValueError, match=message):
        k4.native_activity(trace, True)
