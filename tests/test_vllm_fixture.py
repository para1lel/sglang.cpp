import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


BASELINE = Path(__file__).resolve().parents[1] / "benchmarks/vllm"


@pytest.fixture(scope="module")
def runner():
    spec = importlib.util.spec_from_file_location("offline_measure", BASELINE / "measure.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_small_fixture(runner):
    assert runner.load_case(BASELINE / "small.json") == {
        "prompt_token_ids": [760, 6511, 314, 9338, 369],
        "output_tokens": 8,
        "warmup": 2,
        "repeats": 7,
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("output_tokens", 0), ("output_tokens", 65), ("output_tokens", True),
        ("prompt_token_ids", []), ("prompt_token_ids", [248320]),
        ("prompt_token_ids", [1] * 249),
        ("warmup", 0), ("warmup", 11), ("repeats", 0), ("repeats", 101),
    ],
)
def test_rejects_invalid_fixture(runner, tmp_path, field, value):
    case = json.loads((BASELINE / "small.json").read_text())
    case[field] = value
    path = tmp_path / "case.json"
    path.write_text(json.dumps(case))
    with pytest.raises(ValueError):
        runner.load_case(path)


def completion(tokens, finish_reason="length"):
    output = SimpleNamespace(token_ids=tokens, finish_reason=finish_reason)
    return SimpleNamespace(finished=True, prompt_token_ids=[9], outputs=[output])


def test_measure_counts_only_timed_requests(runner, monkeypatch):
    case = {"prompt_token_ids": [9], "output_tokens": 2, "warmup": 2, "repeats": 3}
    durations_ms = iter([100, 200, 8, 4, 28])
    clock = Mock(return_value=0)
    monkeypatch.setattr(runner, "perf_counter_ns", clock)

    def generate(prompts, params, *, use_tqdm):
        assert prompts == [{"prompt_token_ids": [9]}]
        assert params == dict(temperature=0, max_tokens=2, ignore_eos=True, detokenize=False)
        clock.return_value += next(durations_ms) * 1_000_000
        return [completion([5, 7])]

    engine = SimpleNamespace(generate=Mock(side_effect=generate))
    result = runner.measure(engine, case, sampling_params=dict)
    assert engine.generate.call_count == 5
    assert result["output_ids"] == [5, 7]
    assert result["latency_ms"] == [8, 4, 28]
    assert (result["min_ms"], result["median_ms"], result["max_ms"]) == (4, 8, 28)
    assert result["total_request_ms"] == 40
    assert result["total_output_tokens"] == 6
    assert result["output_tokens_per_second"] == 150
    if "output_tokens_per_median_request_second" in result:
        assert result["output_tokens_per_median_request_second"] == 250


@pytest.mark.parametrize(
    ("tokens", "reason", "diagnostic"),
    [
        ([5, 6], "length", "token IDs changed"),
        ([5], "length", "requested length"),
        ([5, 7], "stop", "requested length"),
    ],
)
def test_measure_rejects_inconsistent_output(runner, tokens, reason, diagnostic):
    case = {"prompt_token_ids": [9], "output_tokens": 2, "warmup": 1, "repeats": 2}
    replies = [[completion([5, 7])], [completion([5, 7])], [completion(tokens, reason)]]
    engine = SimpleNamespace(generate=Mock(side_effect=replies))
    with pytest.raises(RuntimeError, match=diagnostic):
        runner.measure(engine, case, sampling_params=dict)
