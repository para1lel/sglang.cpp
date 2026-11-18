"""Match the GPU dispatch and scheduler steps of the pinned vLLM baseline."""

import collections
import gzip
import json
import re

from benchmarks.concurrent.reference import digest
from benchmarks.streaming.check_vllm import GDN_DECODE, GDN_PREFILL_PREFIX


STEP = re.compile(r"execute_context_(\d+)\((\d+)\)_generation_(\d+)\((\d+)\)")
TRT_DECODE_PREFIX = "fmhaSm100aKernel_QBfloat16KvE4m3OBfloat16H256PagedKvCausal"
GRAPH_LAUNCHES = {("cuda_runtime", "cudaGraphLaunch"), ("cuda_driver", "cuGraphLaunch")}
ATTENTION_OR_GDN = ("batchprefill", "batchdecode", "fmhasm100", "flash_fwd",
                    "gated_delta_rule", "gateddeltanet")


def kernel_route(name):
    if name.startswith("void flashinfer::BatchPrefillWithPagedKVCacheKernel<"):
        if ", __nv_bfloat16, __nv_fp8_e4m3, __nv_bfloat16, float, int," in name:
            return "fa2_bf16_e4m3"
    elif name.startswith(TRT_DECODE_PREFIX) and name.endswith("ForGen"):
        return "trtllm_gen_bf16_e4m3"
    elif name.startswith(GDN_PREFILL_PREFIX):
        return "blackwell_chunked_gdn"
    elif name == GDN_DECODE:
        return "packed_gdn"
    lower = name.lower()
    if (any(marker in lower for marker in ATTENTION_OR_GDN)
            and not any(marker in lower for marker in ("prepare", "postprocess", "norm"))):
        raise ValueError(f"Unrecognized attention or GDN kernel: {name}")
    return None


def inspect_events(events, phase, concurrency, token_budget):
    steps, routes, signatures = [], collections.Counter(), collections.Counter()
    launches = 0
    for event in events:
        if event.get("ph") != "X":
            continue
        category, name = event["cat"], event.get("name", "")
        if category == "user_annotation" and (match := STEP.fullmatch(name)):
            steps.append([int(value) for value in match.groups()])
        elif (category, name) in GRAPH_LAUNCHES:
            launches += 1
        elif category == "kernel" and (route := kernel_route(name)):
            routes[route] += 1
            signatures[name] += 1
    if len(steps) != 4 or launches < 4:
        raise ValueError(f"Expected four steps with CUDA graph activity: {steps}, {launches}")
    if phase == "prefill":
        if not all(0 < requests <= concurrency and 0 < tokens <= token_budget
                   for requests, tokens, _, _ in steps):
            raise ValueError(f"Unexpected prefill scheduler steps: {steps}")
        mixed = sum(decode_requests > 0 for _, _, decode_requests, _ in steps)
        expected = {"fa2_bf16_e4m3": 64, "blackwell_chunked_gdn": 192}
        if mixed:
            expected["trtllm_gen_bf16_e4m3"] = 16 * mixed
    elif phase == "decode":
        if steps != [[0, 0, concurrency, concurrency]] * 4:
            raise ValueError(f"Expected steady decode at full batch {concurrency}: {steps}")
        expected = {"trtllm_gen_bf16_e4m3": 64, "packed_gdn": 192}
    else:
        raise ValueError(f"Unknown phase {phase!r}")
    if routes != expected:
        raise ValueError(f"Unexpected kernel counts for {phase}: {dict(routes)}")
    return {"steps_prefill_requests_tokens_decode_requests_tokens": steps, "routes": dict(routes),
            "graph_launches": launches, "kernel_signatures": dict(signatures)}


def verify_profile(root, expected, phase, concurrency, token_budget):
    data = (root / expected["trace"]).read_bytes()
    if digest(data) != expected["sha256"]:
        raise ValueError(f"{phase}: trace digest mismatch")
    report = inspect_events(json.loads(gzip.decompress(data))["traceEvents"],
                            phase, concurrency, token_budget)
    if report != {key: expected[key] for key in report}:
        raise ValueError(f"{phase}: trace does not match frozen dispatch")
    return report
