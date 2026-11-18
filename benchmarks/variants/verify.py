"""Verify recorded W3 cohorts, effective configuration, engine counters and dispatch."""

import argparse
import collections
import gzip
import json
import math
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path

from benchmarks.concurrent.client import tags
from benchmarks.concurrent.profile import GRAPH_LAUNCHES, STEP, kernel_route
from benchmarks.concurrent.reference import digest, goldens, lane_manifest
from benchmarks.variants.client import replay_cohort, verify_same_engine_greedy
from benchmarks.variants.fixtures import (
    OUTPUT, POLICY, PREFIX, PRIOR_BUNDLE_SHA256, PROMPT, ROWS, TOKENIZER_SHA256, load_requests,
)


GPU_UUID = "GPU-7e25f95a-29b7-9f94-ce9a-2a2871cf7c0c"
PIN = "81a8cb063e6247fe254d7caf7e98a6691d71022a"
CHECKPOINT = "b54fe44144f1701a9369e8b912997ee5cab77cd176c8f97ad3fe63e78cf3c303"
PROMETHEUS = re.compile(r'^vllm:(\w+)\{([^}]+)\} ([^\s]+)$', re.MULTILINE)
COHORT_METRICS = {
    "num_requests_running", "num_requests_waiting", "kv_cache_usage_perc",
    "prompt_tokens_total", "generation_tokens_total", "external_prefix_cache_hits_total",
    "prefix_cache_hits_total", "request_prefill_kv_computed_tokens_count",
    "request_prefill_kv_computed_tokens_sum",
}


def read(path):
    return json.loads(path.read_bytes())


def read_gzip(path):
    with gzip.open(path, "rb") as stream:
        return json.load(stream)


def verify_worker(run, name):
    directory = run / name
    if (run / "vllm-pin.txt").read_text().strip() != PIN:
        raise ValueError("vLLM source pin changed")
    if digest((run / "checkpoint.sha256").read_bytes()) != CHECKPOINT:
        raise ValueError("Checkpoint manifest changed")
    if GPU_UUID not in (run / "gpu-before.csv").read_text():
        raise ValueError("GPU UUID differs from archived W3")
    snapshot, = read(directory / "worker-before.json")["results"]
    prefix = name == "W3-prefix"
    mode = snapshot["variant"]
    cache = snapshot["cache"]
    config = snapshot["config"]
    attention = snapshot["attention"]
    if (snapshot["vllm"] != "0.34.0rc1.dev787+g81a8cb063"
            or snapshot["model_dtype"] != "torch.bfloat16"
            or cache != {"cache_dtype": "fp8_e4m3", "enable_prefix_caching": prefix,
                         "kv_cache_memory_bytes": 137438953472,
                         "mamba_cache_dtype": "bfloat16", "mamba_ssm_cache_dtype": "float32"}
            or attention != {"backend": "FLASHINFER", "disable_flashinfer_q_quantization": True}
            or config["quantization"] != "fp8" or config["kv_offloading_size"] is not None
            or config["kv_transfer"] or "cpu_offload_gb=0.0" not in config["cpu_weight_offload"]
            or "offload_group_size=0" not in config["cpu_weight_offload"]
            or [config[f"{size}_parallel_size"] for size in ("tensor", "pipeline", "data")]
                != [1, 1, 1]
            or not config["chunked_prefill"]
            or (config["max_model_len"], config["max_num_batched_tokens"],
                config["max_num_seqs"]) != (132096, 8192, 32)
            or mode["block_size"] != 1600
            or snapshot["enforce_eager"] or snapshot["compilation"]["mode"] != "NONE"
            or snapshot["compilation"]["cudagraph_mode"] != "FULL_AND_PIECEWISE"):
        raise ValueError("Unexpected live GQA, cache or scheduler settings")
    if prefix:
        if (snapshot["speculative"] or mode["speculative_tokens"]
                or (mode["mamba_cache_mode"], mode["mamba_block_size"]) != ("align", 1600)):
            raise ValueError("Prefix row requires align state caching and speculation off")
    elif (not snapshot["speculative"] or mode["speculative_method"] != "mtp"
          or mode["speculative_tokens"] != 1
          or (mode["mamba_cache_mode"], mode["mamba_block_size"]) != ("none", 132096)
          or not any("MTP" in arch for arch in mode["draft_architectures"])):
        raise ValueError("MTP row requires an actual one-token MTP draft architecture")
    scale = {"tensor": 1.0, "dtype": "torch.float32", "host": 1.0}
    if len(snapshot["gqa"]) != 16 or any(
        layer["backend"] != "FLASHINFER" or layer["kernel_cache_dtype"] != "torch.float8_e4m3fn"
        or layer["scales"] != dict.fromkeys(("q", "k", "v"), scale)
        for layer in snapshot["gqa"].values()
    ):
        raise ValueError("GQA runtime layout or scale differs from the FP8 contract")
    if len(snapshot["gdn"]) != 48 or any(
        layer["convolution_dtype"] != "torch.bfloat16"
        or layer["recurrence_dtype"] != "torch.float32"
        or layer["prefill_backend"] != "flashinfer" for layer in snapshot["gdn"].values()
    ):
        raise ValueError("GDN state precision or prefill backend differs from the contract")
    pool = snapshot["pool"]
    capacity = pool["rows"]["W3"]
    if (capacity["length_budget"] != PROMPT + OUTPUT
            or not 0 < capacity["required_blocks"] < pool["blocks"] - 1
            or any(group["host_resident"] for group in capacity["groups"])):
        raise ValueError("Insufficient live GPU page capacity for W3")
    return snapshot


def metric_samples(path):
    samples = {}
    for key, raw_labels, raw_value in PROMETHEUS.findall(path.read_text()):
        if ('engine="0"' not in raw_labels.split(",")
                or key not in COHORT_METRICS and not key.startswith("spec_decode_num_")):
            continue
        position = re.search(r'(?:^|,)position="(\d+)"(?:,|$)', raw_labels)
        if position:
            if position[1] != "0":
                raise ValueError(f"Unexpected speculative position in {path}")
            key += "_pos0"
        if key in samples:
            raise ValueError(f"Duplicate metric {key} in {path}")
        samples[key] = float(raw_value)
    return samples


def verify_counters(before, after, prefix):
    def delta(key):
        change = after[key] - before[key]
        if not math.isfinite(change) or not change.is_integer() or change < 0:
            raise ValueError(f"Invalid counter delta for {key}: {change}")
        return int(change)

    for sample in (before, after):
        if sample["num_requests_running"] or sample["num_requests_waiting"]:
            raise ValueError("Engine is not idle at the cohort counter snapshot")
    expected = {
        "prompt_tokens_total": 4 * PROMPT,
        "generation_tokens_total": 4 * OUTPUT,
        "external_prefix_cache_hits_total": 0,
        "prefix_cache_hits_total": 4 * PREFIX if prefix else 0,
        "request_prefill_kv_computed_tokens_count": 4,
        "request_prefill_kv_computed_tokens_sum": 4 * (PROMPT - PREFIX if prefix else PROMPT),
    }
    observed = {key: delta(key) for key in expected}
    if observed != expected:
        raise ValueError(f"Engine counters do not match the request and reuse contract: {observed}")
    observed["kv_usage_fraction_before"] = before["kv_cache_usage_perc"]
    observed["kv_usage_fraction_after"] = after["kv_cache_usage_perc"]
    keys = {name: f"spec_decode_num_{name}_total"
            for name in ("drafts", "draft_tokens", "verified_tokens", "accepted_tokens")}
    if prefix:
        if any(key in after and after[key] for key in keys.values()):
            raise ValueError("Speculative activity in the prefix-only row")
        return observed
    counts = {name: delta(key) for name, key in keys.items()}
    if not (0 < counts["accepted_tokens"] <= counts["verified_tokens"]
            <= counts["draft_tokens"] == counts["drafts"]):
        raise ValueError(f"Missing actual draft or target verification work: {counts}")
    for field in ("accepted_tokens", "verified_tokens"):
        if delta(f"{keys[field].removesuffix('_total')}_per_pos_total_pos0") != counts[field]:
            raise ValueError(f"Single-token speculative position counts disagree for {field}")
    observed["speculation"] = {
        **counts, "acceptance_rate": counts["accepted_tokens"] / counts["draft_tokens"],
        "mean_accepted_draft_length": counts["accepted_tokens"] / counts["drafts"],
    }
    return observed


def profile_stats(path):
    steps, signatures = [], collections.Counter()
    launches = 0
    for event in read_gzip(path)["traceEvents"]:
        if event.get("ph") != "X":
            continue
        category, name = event["cat"], event["name"]
        if category == "user_annotation" and (step := STEP.fullmatch(name)):
            steps.append([int(value) for value in step.groups()])
        if (category, name) in GRAPH_LAUNCHES:
            launches += 1
        if category == "kernel":
            signatures[name] += 1
    if len(steps) != 4 or launches < 4:
        raise ValueError(f"{path.name}: missing four scheduler steps or graph launches")
    return steps, launches, signatures


def verify_prefix_profile(phase, steps, signatures):
    routes = collections.Counter()
    for name, count in signatures.items():
        if route := kernel_route(name):
            routes[route] += count
    context = sum(prefill_reqs > 0 for prefill_reqs, _, _, _ in steps)
    generation = sum(decode_reqs > 0 for _, _, decode_reqs, _ in steps)
    decode_only = sum(prefill_reqs == 0 and decode_reqs > 0
                      for prefill_reqs, _, decode_reqs, _ in steps)
    if phase == "prefill":
        if sum(step[0] for step in steps) != 4 or any(
            tokens != requests * (PROMPT - PREFIX) for requests, tokens, _, _ in steps
        ):
            raise ValueError(f"Prefix scheduler replay span differs from cache hit: {steps}")
        for kernel in ("preprocess_mamba_align_fused_kernel", "precopy_mamba_align_fused_kernel"):
            if signatures[kernel] != len(steps):
                raise ValueError(f"Missing GDN state restore dispatch: {kernel}")
    elif steps != [[0, 0, 4, 4]] * 4:
        raise ValueError(f"Expected four full-batch prefix decode steps: {steps}")
    expected = {}
    if context:
        expected.update(fa2_bf16_e4m3=16 * context, blackwell_chunked_gdn=48 * context)
    if generation:
        expected["trtllm_gen_bf16_e4m3"] = 16 * generation
    if decode_only:
        expected["packed_gdn"] = 48 * decode_only
    if routes != expected:
        raise ValueError(f"Unexpected {phase} routes for prefix row: {dict(routes)}")
    return dict(routes)


MTP_GDN_DECODE = (
    "void (anonymous namespace)::gdn_decode_post_conv_mtp_kernel<float, 3, false, 1>"
    "(__nv_bfloat16 const*, __nv_bfloat16 const*, __nv_bfloat16 const*, float const*,"
)
MTP_GPU_MARKERS = {
    "draft_input": "_gather_draft_sampled_kernel",
    "target_verify": "_rejection_kernel",
    "draft_argmax_partial": "_argmax_partial",
    "draft_argmax_finish": "_argmax_finish",
}


def verify_mtp_profile(phase, steps, signatures):
    routes = collections.Counter()
    for name, count in signatures.items():
        if name.startswith(MTP_GDN_DECODE):
            routes["speculative_gdn_bf16_fp32"] += count
        elif name == "fused_sigmoid_gating_delta_rule_update_kernel":
            routes["mixed_recurrent_gdn"] += count
        elif route := kernel_route(name):
            routes[route] += count
    if phase == "prefill":
        if not all(0 < requests <= 4 and 0 < tokens and requests + decode <= 4
                   and tokens + output == 8192 and output == 2 * decode
                   for requests, tokens, decode, output in steps):
            raise ValueError(f"Invalid MTP prefill scheduler steps: {steps}")
        mixed = sum(decode > 0 for _, _, decode, _ in steps)
        expected = {"fa2_bf16_e4m3": 68, "blackwell_chunked_gdn": 192}
        if mixed:
            expected.update(trtllm_gen_bf16_e4m3=17 * mixed, mixed_recurrent_gdn=48 * mixed)
        proposals, verifications = 4, mixed
    else:
        if steps != [[0, 0, 4, 8]] * 4:
            raise ValueError(f"Expected full-batch two-position MTP verification: {steps}")
        expected = {"trtllm_gen_bf16_e4m3": 68, "speculative_gdn_bf16_fp32": 192}
        proposals, verifications = 4, 4
    if routes != expected:
        raise ValueError(f"Unexpected MTP {phase} routes: {dict(routes)}")
    markers = {key: signatures[kernel] for key, kernel in MTP_GPU_MARKERS.items()}
    if markers != dict.fromkeys(("draft_input", "target_verify"), verifications) | dict.fromkeys(
        ("draft_argmax_partial", "draft_argmax_finish"), proposals
    ):
        raise ValueError(f"Incomplete MTP draft and target verification kernels: {markers}")
    return dict(routes) | markers


def independently_summarize(cohorts):
    def summary(values):
        ordered = sorted(values)
        lower, fraction = divmod((len(ordered) - 1) * .95, 1)
        lower = int(lower)
        upper = min(lower + 1, len(ordered) - 1)
        p95 = ordered[lower] + fraction * (ordered[upper] - ordered[lower])
        return {"n": len(values), "p50": statistics.median(values), "p95": p95,
                "min": min(values), "max": max(values),
                "sample_stdev": statistics.stdev(values), "samples": values}

    rates = collections.defaultdict(list)
    durations = collections.defaultdict(list)
    for cohort in cohorts:
        records = cohort["requests"]
        span = (max(row["done_ns"] for row in records)
                - min(row["submit_ns"] for row in records)) / 1e9
        tokens = sum(len(row["output_ids"]) for row in records)
        prompts = sum(row["usage"]["prompt_tokens"] for row in records)
        rates["output_tokens_per_second"].append(tokens / span)
        rates["requests_per_second"].append(len(records) / span)
        rates["aggregate_tokens_per_second"].append((tokens + prompts) / span)
        for row in records:
            durations["ttft_ms"].append((row["first_ns"] - row["submit_ns"]) / 1e6)
            durations["tpot_ms"].append((row["last_ns"] - row["first_ns"]) / 1023 / 1e6)
            durations["e2e_ms"].append((row["done_ns"] - row["submit_ns"]) / 1e6)
    return {"throughput": {key: summary(values) for key, values in rates.items()},
            "latency_ms": {key: summary(values) for key, values in durations.items()}}


def freeze_limits(stats):
    output = stats["throughput"]["output_tokens_per_second"]
    ratio = 1 + max(.01, 2 * output["sample_stdev"] / output["p50"])
    return {"strict_throughput_ratio_gt": ratio,
            "strict_median_output_tokens_per_second_gt": output["p50"] * ratio,
            "p95_latency_ceilings_ms": {
                name: metric["p95"] + max(.01 * metric["p95"], 2 * metric["sample_stdev"])
                for name, metric in stats["latency_ms"].items()},
            }


def check_reset(controls, tag, before_ns):
    reset = read(controls / f"{tag}-cache-reset.json")
    if (reset["status"] != 200 or json.loads(reset["body"]) != {"success": True}
            or reset["ack_ns"] >= before_ns):
        raise ValueError(f"Cache reset did not finish before {tag}")


def verify_fixtures(fixtures):
    manifest = read(fixtures / "manifest.json")
    if (manifest["tokenizer_sha256"] != TOKENIZER_SHA256
            or manifest["prior_bundle_sha256"] != PRIOR_BUNDLE_SHA256):
        raise ValueError("Variant fixture lost tokenizer or prior archive provenance")
    for key, entry in manifest["requests"].items():
        if (not entry["decoded_text_roundtrips"]
                or digest((fixtures / f"{key}.txt").read_bytes()) != entry["text_sha256"]):
            raise ValueError(f"{key}: frozen text identity or tokenizer round-trip changed")
    prime, = load_requests(fixtures, "prime")
    common = prime[2]["prompt_token_ids"][:PREFIX]
    prefix = load_requests(fixtures, "prefix")
    cold = load_requests(fixtures, "cold")
    branches = set()
    for index, (warm_lane, cold_lane) in enumerate(zip(prefix, cold, strict=True)):
        warm, control = warm_lane[2], cold_lane[2]
        if (len(warm["prompt_token_ids"]) != PROMPT or warm["output_tokens"] != OUTPUT
                or warm["prompt_token_ids"][:PREFIX] != common or warm["cached_tokens"] != PREFIX
                or control != warm | {"cached_tokens": 0}
                or json.loads(cold_lane[1])["cache_salt"] != f"w3-prefix-cold-control-{index}"
                or "cache_salt" in json.loads(warm_lane[1])):
            raise ValueError(f"Warm and cold prefix lane {index} differ beyond the cache control")
        branches.add(warm["prompt_token_ids"][PREFIX])
    if (len(branches) != 4 or prime[2]["prompt_token_ids"][PREFIX] in branches
            or len(prime[2]["prompt_token_ids"]) != PREFIX + 1
            or prime[2]["output_tokens"] != 1 or prime[2]["cached_tokens"] != 0):
        raise ValueError("Prefix prime extends a shared branch or has the wrong shape")
    w3 = [lane["request_sha256"] for lane in lane_manifest()["rows"]["W3"]["lanes"]]
    mtp = load_requests(fixtures, "mtp")
    if (w3 != manifest["prior_w3_request_sha256"]
            or w3 != [lane["request_sha256"] for lane, _, _ in mtp]
            or any(case["cached_tokens"] for _, _, case in mtp)):
        raise ValueError("MTP request identities differ from archived W3")


def verify_capture(run, name):
    directory, fixtures = run / name, run / "fixtures"
    capture = directory / "capture"
    controls = capture / "controls"
    prefix = name == "W3-prefix"
    expected = read(capture / "output-ids.json")
    lanes = load_requests(fixtures, "prefix" if prefix else "mtp")
    if prefix:
        cold = read_gzip(controls / "cold-control.json.gz")
        check_reset(controls, "cold-control", cold["barrier_ns"])
        verify_same_engine_greedy(replay_cohort(cold, load_requests(fixtures, "cold")), expected)
    elif expected != goldens("W3"):
        raise ValueError("MTP goldens differ from archived no-MTP W3 goldens")
    reports, counters = {}, {}
    registration = datetime.fromisoformat((run / "fixtures-frozen-utc.txt").read_text().strip())
    for tag in tags(POLICY) + ["diagnostic"]:
        raw = read_gzip(capture / f"{tag}.json.gz")
        if raw["tag"] != tag or datetime.fromisoformat(raw["started_utc"]) <= registration:
            raise ValueError(f"Wrong recorded cohort for {tag}")
        if prefix:
            prime = read_gzip(controls / f"{tag}-prime.json.gz")
            check_reset(controls, tag, prime["barrier_ns"])
            primed = replay_cohort(prime, load_requests(fixtures, "prime"))
            if primed["requests"][0]["done_ns"] >= raw["barrier_ns"]:
                raise ValueError(f"Prime completed after the cohort started: {tag}")
        replayed = replay_cohort(raw, lanes)
        verify_same_engine_greedy(replayed, expected)
        saved = read(capture / f"{tag}-replayed.json")
        if json.loads(json.dumps(replayed)) != saved:
            raise ValueError(f"{tag}: capture timing or token records differ from raw SSE")
        reports[tag] = replayed
        counters[tag] = verify_counters(
            metric_samples(controls / f"{tag}-before.prom"),
            metric_samples(controls / f"{tag}-after.prom"), prefix,
        )
    stats = independently_summarize(
        [report for tag, report in reports.items() if tag.startswith("measured")]
    )
    captured = read(capture / "summary.json")
    fixture_digest = digest((fixtures / "manifest.json").read_bytes())
    if (captured["status"] != "captured" or captured["fixture_manifest_sha256"] != fixture_digest
            or any(captured[key] != values for key, values in stats.items())):
        raise ValueError("Independent timing does not reproduce the saved capture")
    return stats, counters


def verify(run, name, frozen_utc):
    worker = verify_worker(run, name)
    verify_fixtures(run / "fixtures")
    stats, counters = verify_capture(run, name)
    profiles = {}
    inspect_variant = verify_prefix_profile if name == "W3-prefix" else verify_mtp_profile
    for phase in ("prefill", "decode"):
        trace, = (run / name / "profile").glob(f"diagnostic-{phase}*.pt.trace.json.gz")
        steps, launches, signatures = profile_stats(trace)
        routes = inspect_variant(phase, steps, signatures)
        profiles[phase] = {"file": str(trace.relative_to(run)),
                           "sha256": digest(trace.read_bytes()), "scheduler_steps": steps,
                           "graph_launches": launches, "routes": routes}
    fixture_sha = digest((run / "fixtures" / "manifest.json").read_bytes())
    return {"row": name, "status": "verified_frozen",
            "frozen_utc": frozen_utc, "gpu_uuid": GPU_UUID,
            "vllm_pin": PIN, "checkpoint_manifest_sha256": CHECKPOINT,
            "fixture_manifest_sha256": fixture_sha,
            "output_ids_file": f"{name}/capture/output-ids.json",
            "output_ids_sha256": digest((run / name / "capture" / "output-ids.json").read_bytes()),
            "effective_variant": worker["variant"], "capacity": worker["pool"]["rows"]["W3"],
            **stats, "thresholds": freeze_limits(stats), "counters": counters, "profiles": profiles}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("row", choices=ROWS)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    frozen = read(args.run / args.row / "frozen.json") if args.check else None
    utc = frozen["frozen_utc"] if args.check else datetime.now(timezone.utc).isoformat()
    result = verify(args.run, args.row, utc)
    if args.check and result != frozen:
        raise ValueError("Replayed evidence differs from the frozen result")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
