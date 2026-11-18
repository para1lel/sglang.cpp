import collections
import gzip
import json
import sys
from pathlib import Path


FA2_PREFILL = (
    "void flashinfer::BatchPrefillWithPagedKVCacheKernel<true, "
    "flashinfer::KernelTraits<(flashinfer::MaskMode)1, 64u, 1u, 4u, 16u, 16u, 4u, 1u, "
    "(flashinfer::PosEncodingMode)0, __nv_bfloat16, __nv_fp8_e4m3, __nv_bfloat16, float, int, "
    "flashinfer::DefaultAttention<false, false, false, false>, true, false, false>, "
    "PagedParams>(PagedParams)"
)
TRTLLM_GEN_DECODE = (
    "fmhaSm100aKernel_QBfloat16KvE4m3OBfloat16H256PagedKvCausalP64"
    "MultiCtasKvCgaVarSeqSeparateTransKVQ8Kv128StaticGroupedSwapsAbForGen"
)
GDN_PREFILL_PREFIX = (
    "kernel_cutlass_kernel_flashinfergdn_kernelsblackwellgated_delta_net_chunked"
    "GatedDeltaNetChunkedKernel_"
)
GDN_DECODE = "fused_recurrent_gated_delta_rule_packed_decode_kernel"


def audit(snapshot):
    state, = snapshot["results"]
    gqa = state["gqa"].values()
    gdn = state["gdn"].values()
    assert state["model_dtype"] == "torch.bfloat16"
    assert state["cache"]["cache_dtype"] == "fp8_e4m3"
    assert not state["cache"]["enable_prefix_caching"]
    assert state["attention"]["disable_flashinfer_q_quantization"]
    assert not state["enforce_eager"] and not state["speculative"]
    assert len(gqa) == 16 and len(gdn) == 48
    for layer in gqa:
        assert layer["backend"] == "FLASHINFER", layer
        assert layer["kernel_cache_dtype"] == "torch.float8_e4m3fn", layer
        assert set(layer["scales"]) == {"q", "k", "v"}, layer
        for scale in layer["scales"].values():
            assert scale == {"tensor": 1.0, "dtype": "torch.float32", "host": 1.0}, scale
    for layer in gdn:
        assert layer["convolution_dtype"] == "torch.bfloat16", layer
        assert layer["recurrence_dtype"] == "torch.float32", layer


def trace_summary(folder, name):
    traces = [
        path for suffix in (".json", ".json.gz")
        for path in folder.glob(f"{name}_*.pt.trace{suffix}")
    ]
    if len(traces) != 1:
        raise ValueError(f"{name} requires exactly one profiler trace, found {len(traces)}")
    path, = traces
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        events = json.load(stream)["traceEvents"]
    kernels = collections.Counter()
    graph_launches = collections.Counter()
    for event in events:
        if event.get("ph") != "X":
            continue
        category = event.get("cat")
        event_name = event.get("name", "")
        if category == "kernel":
            kernels[event_name] += 1
        elif (category, event_name) in {
            ("cuda_runtime", "cudaGraphLaunch"), ("cuda_driver", "cuGraphLaunch"),
        }:
            graph_launches[event_name] += 1
    gdn_prefill = sum(
        count for kernel, count in kernels.items() if kernel.startswith(GDN_PREFILL_PREFIX)
    )
    for route, actual, expected in (
        ("BF16-Q/E4M3-KV FA2 prefill", kernels[FA2_PREFILL], 16),
        ("BF16-Q/E4M3-KV TRTLLM-Gen decode", kernels[TRTLLM_GEN_DECODE], 32),
        ("Blackwell chunked GDN prefill", gdn_prefill, 48),
        ("packed recurrent GDN decode", kernels[GDN_DECODE], 96),
    ):
        if actual != expected:
            raise ValueError(f"{name} {route}: expected {expected} kernel calls, found {actual}")
    if not graph_launches:
        raise ValueError(f"{name} requires CUDA graph launches")
    return {"trace_files": [path.name], "kernels": kernels, "graph_launches": graph_launches}


if __name__ == "__main__":
    mode, path = sys.argv[1:3]
    if mode == "audit":
        audit(json.loads(Path(path).read_bytes()))
    elif mode == "trace":
        try:
            summary = trace_summary(Path(path), sys.argv[3])
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise SystemExit(str(error)) from error
        print(json.dumps(summary, indent=2))
    else:
        raise SystemExit(f"Unknown check: {mode}")
