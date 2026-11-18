from collections import Counter

import pytest

from benchmarks.streaming.client import json_bytes
from benchmarks.variants.client import replay_cohort, verify_same_engine_greedy
from benchmarks.variants.verify import (
    MTP_GDN_DECODE, MTP_GPU_MARKERS, verify_counters, verify_mtp_profile, verify_prefix_profile,
)


def raw_lane(output, cached, finish):
    prompt = [11, 12, 13]
    case = {"prompt_token_ids": prompt, "output_tokens": 2, "cached_tokens": 1}
    identity = {"lane": 0, "request_sha256": "fixture"}
    usage = {"prompt_tokens": 3, "completion_tokens": 2,
             "prompt_tokens_details": {"cached_tokens": cached}}
    choices = [
        {"index": 0, "token_ids": [], "prompt_token_ids": prompt, "finish_reason": None},
        {"index": 0, "token_ids": output[:1], "finish_reason": None},
        {"index": 0, "token_ids": output[1:], "finish_reason": finish},
    ]
    messages = [json_bytes({"choices": [choice]}).decode() for choice in choices]
    messages.extend((json_bytes({"choices": [], "usage": usage}).decode(), "[DONE]"))
    raw = identity | {
        "connect_ns": 10, "submit_ns": 20, "sent_ns": 30, "headers_ns": 40, "eof_ns": 100,
        "http_status": 200, "response_headers": [("Content-Type", "text/event-stream")],
        "events": [{"read": index, "ns": 40 + 10 * index, "lines": [f"data: {payload}"]}
                   for index, payload in enumerate(messages, 1)],
    }
    return raw, (identity, b"", case)


@pytest.mark.parametrize(("output", "cached", "finish", "error"), [
    ([21, 31], 0, "length", "reused prompt tokens"),
    ([21, 31], 1, "stop", "Expected length stop"),
    ([21], 1, "length", "Expected length stop"),
])
def test_variant_wire_rejects_bad_counts_cache_and_finish(output, cached, finish, error):
    raw, lane = raw_lane(output, cached, finish)
    with pytest.raises(ValueError, match=error):
        replay_cohort({"instrumentation": {}, "barrier_ns": 15, "requests": [raw]}, [lane])


def test_variant_same_engine_ids_are_separate_from_wire():
    raw, lane = raw_lane([21, 31], 1, "length")
    report = replay_cohort({"instrumentation": {}, "barrier_ns": 15, "requests": [raw]}, [lane])
    verify_same_engine_greedy(report, [[21, 31]])
    with pytest.raises(ValueError, match="greedy mismatch at output index 1"):
        verify_same_engine_greedy(report, [[21, 30]])


def test_mtp_counters_require_target_verification():
    before = dict.fromkeys((
        "num_requests_running", "num_requests_waiting", "kv_cache_usage_perc",
        "prompt_tokens_total", "generation_tokens_total", "external_prefix_cache_hits_total",
        "prefix_cache_hits_total", "request_prefill_kv_computed_tokens_count",
        "request_prefill_kv_computed_tokens_sum", "spec_decode_num_drafts_total",
        "spec_decode_num_draft_tokens_total", "spec_decode_num_verified_tokens_total",
        "spec_decode_num_accepted_tokens_total",
        "spec_decode_num_verified_tokens_per_pos_total_pos0",
        "spec_decode_num_accepted_tokens_per_pos_total_pos0",
    ), 0.0)
    after = before | {
        "prompt_tokens_total": 32768.0, "generation_tokens_total": 4096.0,
        "request_prefill_kv_computed_tokens_count": 4.0,
        "request_prefill_kv_computed_tokens_sum": 32768.0,
        "spec_decode_num_drafts_total": 2048.0, "spec_decode_num_draft_tokens_total": 2048.0,
        "spec_decode_num_verified_tokens_total": 2048.0,
        "spec_decode_num_verified_tokens_per_pos_total_pos0": 2048.0,
        "spec_decode_num_accepted_tokens_total": 2048.0,
        "spec_decode_num_accepted_tokens_per_pos_total_pos0": 2048.0,
    }
    assert verify_counters(before, after, False)["speculation"]["acceptance_rate"] == 1.0
    with pytest.raises(ValueError, match="target verification"):
        verify_counters(before, after | {"spec_decode_num_verified_tokens_total": 0.0}, False)


def test_prefix_prefill_requires_state_restore_kernel():
    steps = [[4, 7168, 0, 0], [0, 0, 4, 4], [0, 0, 4, 4], [0, 0, 4, 4]]
    signatures = Counter({"preprocess_mamba_align_fused_kernel": 4})
    with pytest.raises(ValueError, match="GDN state restore dispatch"):
        verify_prefix_profile("prefill", steps, signatures)


def test_mtp_decode_requires_draft_and_target_kernels():
    steps = [[0, 0, 4, 8]] * 4
    trt = "fmhaSm100aKernel_QBfloat16KvE4m3OBfloat16H256PagedKvCausalP64ExampleForGen"
    signatures = Counter({trt: 68, MTP_GDN_DECODE: 192})
    signatures.update(dict.fromkeys(MTP_GPU_MARKERS.values(), 4))
    assert verify_mtp_profile("decode", steps, signatures)["speculative_gdn_bf16_fp32"] == 192
    signatures[MTP_GPU_MARKERS["target_verify"]] = 0
    with pytest.raises(ValueError, match="Incomplete MTP draft and target verification"):
        verify_mtp_profile("decode", steps, signatures)
