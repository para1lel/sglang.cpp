import json
import runpy
import statistics
from pathlib import Path
from types import SimpleNamespace

import pytest


BENCHMARK = Path(__file__).resolve().parents[1] / "benchmarks/vllm"
RUNNER = runpy.run_path(str(BENCHMARK / "measure.py"))
FP8, BF16 = "torch.float8_e4m3fn", "torch.bfloat16"
PAGED_PREFILL = ("paged_prefill", "fa2", BF16, (FP8, FP8), "HND")
TRTLLM_DECODE = ("trtllm_decode", "trtllm-gen", BF16, (FP8, FP8), "HND")
GDN_PREFILL = "_forward_flashinfer_prefill"
GDN_DECODE = "_forward_core_decode_non_spec"


def test_fp8_fixture_preserves_workload_and_pins_precision():
    load = RUNNER["load_case"]
    bf16_case = load(BENCHMARK / "small.json")
    fp8_case = load(BENCHMARK / "small-fp8.json")
    fp8_options = dict(kv_cache_dtype="fp8_e4m3",
                       expected_output_ids=[11751, 13, 198, 760, 6511, 314, 9564, 369])
    assert fp8_case == bf16_case | fp8_options

    config = RUNNER["engine_config"]
    historical = json.loads((BENCHMARK / "small-result.json").read_text())
    assert config(bf16_case) == historical["config"]
    assert config(fp8_case) == historical["config"] | dict(
        kv_cache_dtype="fp8_e4m3",
        attention_config=dict(backend="FLASHINFER", disable_flashinfer_q_quantization=True),
    )


@pytest.mark.parametrize(
    ("updates", "error"),
    [
        ({"kv_cache_dtype": "mxfp8"}, "KV cache"),
        ({"kv_cache_dtype": "fp8"}, "KV cache"),
        ({"expected_output_ids": None}, "expected output IDs"),
        ({"expected_output_ids": [11751]}, "expected output IDs"),
        ({"expected_output_ids": [True] * 8}, "expected output IDs"),
        ({"expected_output_ids": [248320] * 8}, "expected output IDs"),
    ],
)
def test_fp8_fixture_rejects_invalid_precision_or_ids(tmp_path, updates, error):
    case = json.loads((BENCHMARK / "small-fp8.json").read_text())
    path = tmp_path / "case.json"
    path.write_text(json.dumps(case | updates))
    with pytest.raises(ValueError, match=error):
        RUNNER["load_case"](path)


@pytest.mark.parametrize("wrong_request", [0, 2])
def test_fp8_rejects_wrong_ids_during_warmup_and_timing(wrong_request):
    case = RUNNER["load_case"](BENCHMARK / "small-fp8.json")
    expected = case["expected_output_ids"]
    requests = iter(range(case["warmup"] + case["repeats"]))

    def generate(prompts, params, *, use_tqdm):
        tokens = expected[:-1] + [0] if next(requests) == wrong_request else expected
        output = SimpleNamespace(token_ids=tokens, finish_reason="length")
        return [SimpleNamespace(finished=True, prompt_token_ids=case["prompt_token_ids"],
                                outputs=[output])]

    with pytest.raises(RuntimeError, match="Expected greedy IDs"):
        RUNNER["measure"](SimpleNamespace(generate=generate), case, sampling_params=dict)


def observed_state(metadata, gdn_branches, flashinfer_calls):
    layer = dict(backend="FLASHINFER", storage_dtype="torch.uint8", logical_dtype=FP8,
                 scale_dtypes=["torch.float32"] * 3, scales=[1.0] * 3, host_scales=[1.0] * 3)
    gdn = dict(conv_dtype=BF16, recurrent_dtype="torch.float32", prefill="flashinfer",
               packed_recurrent_decode=True)
    return dict(cache_dtype="fp8_e4m3", cache_bytes=1 << 30, prefix_caching=False,
                eager=True, speculative=False, gqa=dict.fromkeys(range(3, 64, 4), layer),
                gdn=[gdn] * 48, gqa_metadata=metadata, gdn_branches=gdn_branches,
                flashinfer_calls=flashinfer_calls)


@pytest.mark.parametrize(
    ("prompt_length", "output_length", "metadata", "gdn_branches", "prefills", "decodes"),
    [
        (1, 2, [(0, 1, "LBHNC")] * 2, [GDN_PREFILL, GDN_DECODE], 0, 32),
        (1, 1, [(0, 1, "LBHNC")], [GDN_PREFILL], 0, 16),
        (5, 1, [(5, 0, "LBHNC")], [GDN_PREFILL], 16, 0),
        (255, 1, [(255, 0, "LBHNC")], [GDN_PREFILL], 16, 0),
    ],
)
def test_fp8_dispatch_validates_only_exercised_phases(
    prompt_length, output_length, metadata, gdn_branches, prefills, decodes
):
    case = dict(prompt_token_ids=[760] * prompt_length, output_tokens=output_length)
    calls = [PAGED_PREFILL] * prefills + [TRTLLM_DECODE] * decodes
    state = observed_state(metadata, gdn_branches, calls)
    validate = RUNNER["validate_fp8_dispatch"]
    result = validate(state, case)
    assert result["prefill"]["calls"] == prefills
    assert result["decode"]["calls"] == decodes
    if not prefills:
        assert result["prefill"] == dict(entry=None, backend=None, calls=0)
    if not decodes:
        assert result["decode"] == dict(entry=None, backend=None, calls=0)
    with pytest.raises(RuntimeError, match="Unexpected FlashInfer dispatch"):
        validate(state | {"flashinfer_calls": calls[:-1]}, case)
    with pytest.raises(RuntimeError, match="Unexpected prefill/decode"):
        validate(state | {"gdn_branches": [GDN_DECODE] * output_length}, case)


def test_fp8_dispatch_rejects_fallback_and_scale_drift():
    case = RUNNER["load_case"](BENCHMARK / "small-fp8.json")
    state = observed_state(
        [(5, 0, "LBHNC")] + [(0, 1, "LBHNC")] * 7,
        [GDN_PREFILL] + [GDN_DECODE] * 7,
        [PAGED_PREFILL] * 16 + [TRTLLM_DECODE] * 112,
    )

    validate = RUNNER["validate_fp8_dispatch"]
    assert validate(state, case)["gqa_kv_dtype"] == FP8
    with pytest.raises(RuntimeError, match="Unexpected FlashInfer dispatch"):
        fallback = ("trtllm_decode", "xqa", *TRTLLM_DECODE[2:])
        validate(state | {"flashinfer_calls": [PAGED_PREFILL] * 16 + [fallback] * 112}, case)
    with pytest.raises(RuntimeError, match="Unexpected FlashInfer call dtypes"):
        fallback = ("trtllm_decode", "trtllm-gen", BF16, (BF16, BF16), "HND")
        validate(state | {"flashinfer_calls": [PAGED_PREFILL] * 16 + [fallback] * 112}, case)
    with pytest.raises(RuntimeError, match="Unexpected GQA cache or scales"):
        state["gqa"][3] = state["gqa"][3] | {"host_scales": [1.0, 2.0, 1.0]}
        validate(state, case)


def test_archived_fp8_result_matches_fixture_and_recomputes_rates():
    case = RUNNER["load_case"](BENCHMARK / "small-fp8.json")
    result = json.loads((BENCHMARK / "small-fp8-result.json").read_text())
    assert result["case"] == case
    assert result["config"] == RUNNER["engine_config"](case)
    assert result["output_ids"] == case["expected_output_ids"]
    assert result["requests_matching_output_ids"] == case["warmup"] + case["repeats"]
    assert result["gpu_uuid"] == "GPU-9a94015a-6a0d-1ae1-82db-b6736b77562e"
    assert result["run"] == "run.cctk0V"
    assert result["exit_code"] == 0
    samples = result["latency_ms"]
    assert len(samples) == case["repeats"] and all(sample > 0 for sample in samples)
    assert result["median_ms"] == statistics.median(samples)
    assert result["total_request_ms"] == pytest.approx(sum(samples))
    tokens = case["output_tokens"] * case["repeats"]
    assert result["total_output_tokens"] == tokens
    assert result["output_tokens_per_second"] == pytest.approx(tokens * 1000 / sum(samples))
