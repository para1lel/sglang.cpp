"""Measure completed offline token-ID requests with one resident vLLM engine."""

import json
import os
import statistics
import sys
from pathlib import Path
from time import perf_counter_ns


def load_case(path):
    case = json.loads(path.read_text())
    ids = case["prompt_token_ids"]
    counts = (case["output_tokens"], case["warmup"], case["repeats"])
    if not all(type(value) is int for value in counts):
        raise ValueError("Counts must be integers")
    if not isinstance(ids, list) or not ids or not all(
        type(token) is int and 0 <= token < 248320 for token in ids
    ):
        raise ValueError("Supply text token IDs from the checkpoint vocabulary")
    if not (1 <= counts[0] <= 64 and len(ids) + counts[0] <= 256):
        raise ValueError("Limit requests to 256 total tokens and 64 output tokens")
    if not (1 <= counts[1] <= 10 and 1 <= counts[2] <= 100):
        raise ValueError("Use 1–10 warmups and 1–100 timed requests")
    return case


def measure(engine, case, sampling_params=None):
    if sampling_params is None:
        from vllm import SamplingParams as sampling_params

    params = sampling_params(
        temperature=0, max_tokens=case["output_tokens"], ignore_eos=True, detokenize=False
    )
    prompt = {"prompt_token_ids": case["prompt_token_ids"]}
    reference = None
    elapsed_ms = []
    for iteration in range(case["warmup"] + case["repeats"]):
        start = perf_counter_ns()
        request, = engine.generate([prompt], params, use_tqdm=False)
        duration = (perf_counter_ns() - start) / 1e6
        output, = request.outputs
        tokens = list(output.token_ids)
        if not request.finished or request.prompt_token_ids != prompt["prompt_token_ids"]:
            raise RuntimeError("Incomplete or modified input request")
        if len(tokens) != case["output_tokens"] or output.finish_reason != "length":
            raise RuntimeError("Generation stopped before the requested length")
        if reference is not None and tokens != reference:
            raise RuntimeError("Greedy output token IDs changed across repetitions")
        reference = tokens
        if iteration >= case["warmup"]:
            elapsed_ms.append(duration)
    median = statistics.median(elapsed_ms)
    total_ms = sum(elapsed_ms)
    total_tokens = case["output_tokens"] * len(elapsed_ms)
    return dict(output_ids=reference, latency_ms=elapsed_ms, median_ms=median,
                min_ms=min(elapsed_ms), max_ms=max(elapsed_ms),
                output_tokens_per_second=total_tokens * 1000 / total_ms,
                output_tokens_per_median_request_second=case["output_tokens"] * 1000 / median,
                total_output_tokens=total_tokens, total_request_ms=total_ms,
                timing_scope="offline serial smoke, summed completed LLM.generate calls")


def main():
    import torch
    import vllm
    from vllm import LLM

    case_path, model, destination = map(Path, sys.argv[1:])
    case = load_case(case_path)
    if torch.cuda.device_count() != 1 or "B200" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Expected exactly one visible NVIDIA B200")
    pin = Path(sys.prefix).parent.joinpath("artifact.txt").read_text().splitlines()[0]
    if pin != "81a8cb063e6247fe254d7caf7e98a6691d71022a" or pin[:9] not in vllm.__version__:
        raise RuntimeError("The installed vLLM does not match the benchmark pin")
    config = dict(tensor_parallel_size=1, dtype="bfloat16", seed=0, enforce_eager=True,
                  max_model_len=256, max_num_batched_tokens=256, max_num_seqs=1,
                  kv_cache_memory_bytes=1 << 30, gpu_memory_utilization=0.30,
                  enable_prefix_caching=False, language_model_only=True,
                  skip_tokenizer_init=True, generation_config="vllm")
    engine = LLM(model=str(model.resolve(strict=True)), **config)
    result = measure(engine, case)
    result.update(case=case, model=str(model), config=config, vllm=vllm.__version__,
                  source_commit=pin, torch=str(torch.__version__), cuda=torch.version.cuda,
                  gpu_uuid=os.environ["CUDA_VISIBLE_DEVICES"], gpu=torch.cuda.get_device_name(0))
    destination.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
