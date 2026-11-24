"""Measure completed offline token-ID requests with one resident vLLM engine."""

import json
import os
import platform
import statistics
import sys
from collections import Counter
from contextlib import ExitStack
from importlib.metadata import version
from pathlib import Path
from time import perf_counter_ns
from unittest.mock import patch


def valid_token_ids(ids):
    return isinstance(ids, list) and bool(ids) and all(
        type(token) is int and 0 <= token < 248320 for token in ids
    )


def load_case(path):
    case = json.loads(path.read_text())
    ids = case["prompt_token_ids"]
    counts = (case["output_tokens"], case["warmup"], case["repeats"])
    if not all(type(value) is int for value in counts):
        raise ValueError("Counts must be integers")
    if not valid_token_ids(ids):
        raise ValueError("Supply text token IDs from the checkpoint vocabulary")
    if not (1 <= counts[0] <= 64 and len(ids) + counts[0] <= 256):
        raise ValueError("Limit requests to 256 total tokens and 64 output tokens")
    if not (1 <= counts[1] <= 10 and 1 <= counts[2] <= 100):
        raise ValueError("Use 1–10 warmups and 1–100 timed requests")
    dtype = case.get("kv_cache_dtype", "bfloat16")
    if dtype not in ("bfloat16", "fp8_e4m3"):
        raise ValueError("Use bfloat16 or fp8_e4m3 KV cache")
    if "expected_output_ids" in case or dtype == "fp8_e4m3":
        expected = case.get("expected_output_ids")
        if not valid_token_ids(expected) or len(expected) != counts[0]:
            raise ValueError("Supply expected output IDs for the full requested length")
    return case


def engine_config(case):
    config = dict(tensor_parallel_size=1, dtype="bfloat16", seed=0, enforce_eager=True,
                  max_model_len=256, max_num_batched_tokens=256, max_num_seqs=1,
                  kv_cache_memory_bytes=1 << 30, gpu_memory_utilization=0.30,
                  enable_prefix_caching=False, language_model_only=True,
                  skip_tokenizer_init=True, generation_config="vllm")
    if "kv_cache_dtype" in case:
        config["kv_cache_dtype"] = case["kv_cache_dtype"]
    if case.get("kv_cache_dtype") == "fp8_e4m3":
        config["attention_config"] = dict(
            backend="FLASHINFER", disable_flashinfer_q_quantization=True
        )
    return config


def greedy_params(sampling_params, case):
    return sampling_params(
        temperature=0, max_tokens=case["output_tokens"], ignore_eos=True, detokenize=False
    )


def completed_ids(request, case):
    output, = request.outputs
    tokens = list(output.token_ids)
    if not request.finished or request.prompt_token_ids != case["prompt_token_ids"]:
        raise RuntimeError("Incomplete or modified input request")
    if len(tokens) != case["output_tokens"] or output.finish_reason != "length":
        raise RuntimeError("Generation stopped before the requested length")
    if "expected_output_ids" in case and tokens != case["expected_output_ids"]:
        raise RuntimeError(f"Expected greedy IDs {case['expected_output_ids']}, got {tokens}")
    return tokens


def measure(engine, case, sampling_params=None, *, warmup_request=None):
    if sampling_params is None:
        from vllm import SamplingParams as sampling_params

    params = greedy_params(sampling_params, case)
    prompt = {"prompt_token_ids": case["prompt_token_ids"]}
    reference = None
    elapsed_ms = []
    for iteration in range(case["warmup"] + case["repeats"]):
        start = perf_counter_ns()
        if iteration == 0 and warmup_request is not None:
            request = warmup_request
        else:
            request, = engine.generate([prompt], params, use_tqdm=False)
        duration = (perf_counter_ns() - start) / 1e6
        tokens = completed_ids(request, case)
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


def checkpoint_scale_audit(model):
    index = json.loads((model / "model.safetensors.index.json").read_text())
    config = json.loads((model / "config.json").read_text())
    names = [name for name in index["weight_map"] if "scale" in name.lower()]
    audit = dict(count=len(names), suffixes=sorted({name.rsplit(".", 1)[-1] for name in names}),
                 kv_cache_scheme=config["quantization_config"].get("kv_cache_scheme"))
    if audit != dict(count=407, suffixes=["weight_scale_inv"], kv_cache_scheme=None):
        raise RuntimeError(f"Unexpected checkpoint scale policy: {audit}")
    return audit


def install_fp8_observer(worker):
    from vllm.forward_context import get_forward_context
    from vllm.v1.attention.backends import flashinfer

    layers = worker.get_model().language_model.model.layers
    gqa = {index: layer.self_attn.attn for index, layer in enumerate(layers)
           if layer.layer_type == "full_attention"}
    gdn = [layer.linear_attn for layer in layers if layer.layer_type == "linear_attention"]
    config = worker.vllm_config
    state = dict(cache_dtype=config.cache_config.cache_dtype,
                 cache_bytes=config.cache_config.kv_cache_memory_bytes,
                 prefix_caching=config.cache_config.enable_prefix_caching,
                 eager=config.model_config.enforce_eager,
                 speculative=config.speculative_config is not None,
                 gqa={}, gdn=[], gqa_metadata=[], gdn_branches=[], flashinfer_calls=[])
    for index, attention in gqa.items():
        backend = attention.get_attn_backend()
        scale_names = ("q", "k", "v")
        state["gqa"][index] = dict(
            backend=backend.get_name(),
            storage_dtype=str(attention.kv_cache.dtype),
            logical_dtype=str(backend.get_dtype_for_flashinfer(attention.impl.kv_cache_dtype)),
            scale_dtypes=[str(getattr(attention, f"_{name}_scale").dtype)
                          for name in scale_names],
            scales=[getattr(attention, f"_{name}_scale").item() for name in scale_names],
            host_scales=[getattr(attention, f"_{name}_scale_float") for name in scale_names],
        )
    for attention in gdn:
        state["gdn"].append(dict(
            conv_dtype=str(attention.kv_cache[0].dtype),
            recurrent_dtype=str(attention.kv_cache[1].dtype),
            prefill=attention.gdn_prefill_backend,
            packed_recurrent_decode=attention.enable_packed_recurrent_decode,
        ))

    def record(entry, query, cache, backend, layout):
        tensors = cache if isinstance(cache, tuple) else (cache,)
        dtypes = tuple(str(tensor.dtype) for tensor in tensors)
        state["flashinfer_calls"].append((entry, backend, str(query.dtype), dtypes, layout))

    def native_prefill(wrapper, query, cache, *args, **kwargs):
        record("paged_prefill", query, cache, wrapper._backend, wrapper._kv_layout)
        return original_prefill(wrapper, query, cache, *args, **kwargs)

    def trtllm_decode(*args, **kwargs):
        record("trtllm_decode", kwargs["query"], kwargs["kv_cache"],
               kwargs["backend"], kwargs["kv_layout"])
        return original_decode(*args, **kwargs)

    def metadata_hook(attention, _args):
        metadata = get_forward_context().attn_metadata[attention.layer_name]
        state["gqa_metadata"].append(
            (metadata.num_prefill_tokens, metadata.num_decode_tokens, metadata.kv_cache_layout.name)
        )

    def trace_gdn(method_name):
        original = getattr(gdn[0], method_name)

        def traced(*args, **kwargs):
            state["gdn_branches"].append(method_name)
            return original(*args, **kwargs)

        return traced

    native = flashinfer.BatchPrefillWithPagedKVCacheWrapper
    original_prefill = native.run
    original_decode = flashinfer.trtllm_batch_decode_with_kv_cache
    with ExitStack() as stack:
        stack.enter_context(patch.object(native, "run", native_prefill))
        stack.enter_context(
            patch.object(flashinfer, "trtllm_batch_decode_with_kv_cache", trtllm_decode)
        )
        for name in ("_forward_flashinfer_prefill", "_forward_core_decode_non_spec"):
            stack.enter_context(patch.object(gdn[0], name, trace_gdn(name)))
        stack.enter_context(gqa[3].register_forward_pre_hook(metadata_hook))
        worker.fp8_benchmark_observer = (stack.pop_all(), state)


def finish_fp8_observer(worker):
    stack, state = worker.fp8_benchmark_observer
    del worker.fp8_benchmark_observer
    stack.close()
    state["loaded_native_paths"] = sorted({
        filename for line in Path("/proc/self/maps").read_text().splitlines()
        if len(parts := line.split(maxsplit=5)) == 6
        and (filename := parts[5]).startswith("/") and ".so" in filename
        and any(name in filename for name in ("/vllm/", "/flashinfer", "/libcuda.so"))
    })
    return state


def validate_fp8_dispatch(state, case):
    fp8 = "torch.float8_e4m3fn"
    bf16 = "torch.bfloat16"
    if (state["cache_dtype"], state["cache_bytes"], state["prefix_caching"],
            state["eager"], state["speculative"]) != ("fp8_e4m3", 1 << 30, False, True, False):
        raise RuntimeError("Unexpected effective cache, eager, or speculative settings")
    if set(state["gqa"]) != set(range(3, 64, 4)):
        raise RuntimeError("Unexpected GQA layer indices")
    for index, layer in state["gqa"].items():
        if (layer["backend"] != "FLASHINFER" or layer["logical_dtype"] != fp8
                or layer["storage_dtype"] not in ("torch.uint8", fp8)
                or layer["scale_dtypes"] != ["torch.float32"] * 3
                or layer["scales"] != [1.0] * 3 or layer["host_scales"] != [1.0] * 3):
            raise RuntimeError(f"Unexpected GQA cache or scales in layer {index}: {layer}")
    gdn = dict(conv_dtype=bf16, recurrent_dtype="torch.float32", prefill="flashinfer",
               packed_recurrent_decode=True)
    if state["gdn"] != [gdn] * 48:
        raise RuntimeError("Unexpected GDN state precision or configuration")

    prompt_length = len(case["prompt_token_ids"])
    generated_decode_steps = case["output_tokens"] - 1
    gqa_prefill_steps = int(prompt_length > 1)
    gqa_decode_steps = case["output_tokens"] - gqa_prefill_steps
    metadata = ([(prompt_length, 0, "LBHNC")] * gqa_prefill_steps
                + [(0, 1, "LBHNC")] * gqa_decode_steps)
    branches = (["_forward_flashinfer_prefill"]
                + ["_forward_core_decode_non_spec"] * generated_decode_steps)
    if list(map(tuple, state["gqa_metadata"])) != metadata or state["gdn_branches"] != branches:
        raise RuntimeError("Unexpected prefill/decode steps or cache layout")
    prefill = dict(entry="paged_prefill", backend="fa2", calls=16 * gqa_prefill_steps)
    decode = dict(entry="trtllm_decode", backend="trtllm-gen", calls=16 * gqa_decode_steps)
    expected = Counter({
        (phase["entry"], phase["backend"]): phase["calls"]
        for phase in (prefill, decode) if phase["calls"]
    })
    observed = Counter()
    for entry, backend, query, cache, layout in state["flashinfer_calls"]:
        if query != bf16 or tuple(cache) != (fp8, fp8) or layout != "HND":
            raise RuntimeError(
                f"Unexpected FlashInfer call dtypes or layout: {entry}, {query}, {cache}, {layout}"
            )
        observed[entry, backend] += 1
    if observed != expected:
        raise RuntimeError(f"Unexpected FlashInfer dispatch: {observed}, expected {expected}")
    for phase in (prefill, decode):
        if not phase["calls"]:
            phase.update(entry=None, backend=None)
    return dict(gqa_backend="FLASHINFER", gqa_layer_count=len(state["gqa"]),
                gqa_kv_dtype=fp8, query_dtype=bf16, cache_layout="LBHNC", kernel_layout="HND",
                prefill=prefill, decode=decode, kv_scale_granularity="per_tensor",
                q_scale=1.0, k_scale=1.0, v_scale=1.0, gdn=gdn)


def observe_fp8(engine, case, sampling_params, destination):
    engine.collective_rpc(install_fp8_observer)
    try:
        request, = engine.generate([{"prompt_token_ids": case["prompt_token_ids"]}],
                                   greedy_params(sampling_params, case), use_tqdm=False)
    finally:
        state, = engine.collective_rpc(finish_fp8_observer)
        destination.with_name("fp8-dispatch.json").write_text(json.dumps(state, indent=2) + "\n")
    completed_ids(request, case)
    return request, validate_fp8_dispatch(state, case)


def main():
    case_path, model, destination = map(Path, sys.argv[1:])
    case = load_case(case_path) | {"warmup": 1, "repeats": 1}
    fp8 = case.get("kv_cache_dtype") == "fp8_e4m3"
    if fp8:
        os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

    import torch
    import vllm
    from vllm import LLM, SamplingParams

    if torch.cuda.device_count() != 1 or "B200" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Expected exactly one visible NVIDIA B200")
    pin = Path(sys.prefix).parent.joinpath("artifact.txt").read_text().splitlines()[0]
    if pin != "81a8cb063e6247fe254d7caf7e98a6691d71022a" or pin[:9] not in vllm.__version__:
        raise RuntimeError("The installed vLLM does not match the benchmark pin")
    config = engine_config(case)
    if fp8:
        scale_audit = checkpoint_scale_audit(model)
    engine = LLM(model=str(model.resolve(strict=True)), **config)
    warmup_request, dispatch = observe_fp8(engine, case, SamplingParams, destination) if fp8 else (
        None, None)
    result = measure(engine, case, SamplingParams, warmup_request=warmup_request)
    result.update(protocol="offline-single-round-v2", case=case, model=str(model),
                  config=config, vllm=vllm.__version__,
                  source_commit=pin, torch=str(torch.__version__), cuda=torch.version.cuda,
                  gpu_uuid=os.environ["CUDA_VISIBLE_DEVICES"], gpu=torch.cuda.get_device_name(0))
    if fp8:
        result.update(python=platform.python_version(), flashinfer=version("flashinfer-python"),
                      checkpoint_scale_audit=scale_audit,
                      requests_matching_output_ids=case["warmup"] + case["repeats"])
        result["effective"] = dispatch
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
