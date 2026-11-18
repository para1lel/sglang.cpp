"""Check the public native decoder against a frozen pinned-vLLM oracle."""

import argparse
import fcntl
import hashlib
import json
import math
import os
import statistics
import subprocess
from pathlib import Path


FP8_ORACLE = "pinned-vllm-fp8-e4m3-gqa-kv-raw-logits"
GPU_INDEX = "1"
GPU_UUID = "GPU-9a94015a-6a0d-1ae1-82db-b6736b77562e"
BASELINE = Path("/data1/shared/dongwu.chen/environments/vllm-81a8cb063e62")
CHECKPOINT = Path("/data0/shared/Qwen3.8-27B-FP8")
TOP_SCORES = 64


def sha256(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def command_output(*command):
    return subprocess.check_output(command, text=True, timeout=20)


def check_gpu(evidence, stage, gpu_index=GPU_INDEX, gpu_uuid=GPU_UUID):
    info = command_output(
        "nvidia-smi", "-i", gpu_index,
        "--query-gpu=index,uuid,name,pci.bus_id,compute_cap,driver_version,"
        "memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    )
    compute = command_output(
        "nvidia-smi", "-i", gpu_uuid,
        "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits",
    )
    (evidence / f"gpu-{stage}.csv").write_text(info)
    (evidence / f"compute-{stage}.csv").write_text(compute)
    index, uuid, name, _, capability, _, _, _, free, busy = (
        field.strip() for field in info.strip().split(",")
    )
    require((index, uuid, name, capability) == (gpu_index, gpu_uuid, "NVIDIA B200", "10.0"),
            f"GPU identity mismatch: {info}")
    require(not compute.strip() and int(free) >= 65000 and int(busy) == 0,
            f"GPU is occupied, active, or has less than 65000 MiB free: {info}{compute}")


def bf16_ulp(value):
    return math.ldexp(1.0, max(math.frexp(value)[1] - 8, -133)) if value else 2.0 ** -133


def validate_scores(rows, replicas, output_ids):
    require(len(rows) == len(output_ids), "The result lacks a score row for each generated token")
    reports = []
    for step, (row, token) in enumerate(zip(rows, output_ids, strict=True)):
        reference = replicas[0]["rows"][step]
        expected = {entry["token_id"]: entry["raw_logit"] for entry in reference["top_k"]}
        repeated = [
            {entry["token_id"]: entry["raw_logit"] for entry in replica["rows"][step]["top_k"]}
            for replica in replicas[1:]
        ]
        candidate = {entry["token_id"]: entry["logit"] for entry in row}
        require(len(row) == TOP_SCORES and len(candidate) == len(row),
                f"Step {step} lacks {TOP_SCORES} distinct score entries")
        require(all(type(key) is int and 0 <= key < 248320 for key in candidate),
                f"Step {step} returned an invalid vocabulary ID")
        require(
            all(type(value) in (int, float) and math.isfinite(value)
                for value in candidate.values()),
            f"Step {step} returned a nonfinite raw logit",
        )
        scores = list(candidate.values())
        require(scores == sorted(scores, reverse=True) and candidate.get(token) == scores[0],
                f"Step {step} does not expose the generated token as a maximum raw logit")
        require(scores[0] > scores[1] or reference["top1_minus_top2"] == 0,
                f"Step {step} lost the positive greedy margin")
        measured = []
        for token_id, reference_score in expected.items():
            repeat_delta = max((abs(reference_score - values.get(token_id, reference_score))
                                for values in repeated), default=0.0)
            budget = 0.5 + max(2 * bf16_ulp(reference_score), 4 * repeat_delta)
            actual = candidate.get(token_id)
            delta = abs(actual - reference_score) if actual is not None else None
            lower_bound = max(0.0, reference_score - scores[-1])
            measured.append(dict(token_id=token_id, reference=reference_score, candidate=actual,
                                 abs_delta=delta, budget=budget,
                                 absent_delta_lower_bound=lower_bound if actual is None else None,
                                 violation=(delta if actual is not None else lower_bound) > budget))
        known_deltas = [entry for entry in measured if entry["abs_delta"] is not None]
        reports.append(dict(
            step=step, phase=reference["phase"], prefix_length=reference["prefix_length"],
            token_id=token, candidate_margin=scores[0] - scores[1],
            reference_margin=reference["top1_minus_top2"],
            reference_top_k_in_candidate=len(expected.keys() & candidate.keys()),
            top16_overlap=len(expected.keys() & {entry["token_id"] for entry in row[:16]}),
            max_abs_delta=max(entry["abs_delta"] for entry in known_deltas),
            max_delta_ratio=max(entry["abs_delta"] / entry["budget"] for entry in known_deltas),
            scores=measured,
        ))
    return reports


def validate(candidate, request, warmup, repeats, with_scores):
    case = request["case"]
    reference = request["unscored"]
    expected_ids = reference["output_ids"]
    output_ids = candidate["output_token_ids"]
    require(candidate["prompt_token_ids"] == case["prompt_token_ids"], "Prompt token IDs differ")
    require(output_ids == expected_ids, f"Output token IDs differ: {output_ids} != {expected_ids}")
    require(candidate["finish_reason"] == reference["finish_reason"], "Finish reason differs")
    if reference["finish_reason"] == "stop":
        require(
            output_ids[-1] == reference["stop_reason"] and
            len(output_ids) < case["max_tokens"],
            "Configured stop ID or generation length differs",
        )
    effective = candidate["device"], candidate["warmup"], candidate["repeats"]
    require(effective == (0, warmup, repeats), "Unexpected device, warmup or repetition count")
    times = candidate["request_ms"]
    require(len(times) == repeats and all(math.isfinite(value) and value > 0 for value in times),
            "Missing or invalid request latency samples")
    rate = 1000 * len(output_ids) * repeats / sum(times)
    require(math.isclose(candidate["output_tokens_per_second"], rate, rel_tol=1e-8),
            "Reported throughput differs from completed-request samples")
    require(math.isfinite(candidate["load_ms"]) and candidate["load_ms"] > 0,
            "Missing or invalid model load time")
    model_bytes = candidate["model_device_allocation_bytes"]
    require(0 < model_bytes <= candidate["peak_device_allocation_bytes"],
            "Invalid device allocation metrics")
    if with_scores:
        return validate_scores(candidate["step_top_logits"], request["scored_replicas"], output_ids)
    require("step_top_logits" not in candidate, "Unexpected score observation in timing result")
    return []


def run_case(args, request, env, evidence, warmup, repeats, with_scores):
    case = request["case"]
    prompt = ",".join(map(str, case["prompt_token_ids"]))
    command = [str(args.binary), "--checkpoint", str(args.checkpoint), "--device", "0",
               "--max-new-tokens", str(case["max_tokens"]), "--warmup", str(warmup),
               "--repeats", str(repeats)]
    if case["name"] == "alternate":
        prompt_file = evidence / "alternate.prompt"
        prompt_file.write_text(" ".join(map(str, case["prompt_token_ids"])) + "\n")
        command.extend(("--prompt-file", str(prompt_file)))
    else:
        command.extend(("--prompt-ids", prompt))
    if case["stop_token_ids"]:
        stop, = case["stop_token_ids"]
        command.extend(("--stop-token", str(stop)))
    if with_scores:
        command.extend(("--top-logits", str(TOP_SCORES)))
    prefix = evidence / case["name"]
    write_json(prefix.with_suffix(".command.json"), command)
    with (
        prefix.with_suffix(".json").open("w") as output,
        prefix.with_suffix(".stderr").open("w") as errors,
    ):
        status = subprocess.run(command, env=env, cwd=evidence, stdout=output, stderr=errors,
                                timeout=args.timeout, check=False)
    require(status.returncode == 0, f"Native CLI exited with status {status.returncode}")
    candidate = json.loads(prefix.with_suffix(".json").read_text())
    rows = validate(candidate, request, warmup, repeats, with_scores)
    violations = sum(entry["violation"] for row in rows for entry in row["scores"])
    missing = sum(entry["candidate"] is None for row in rows for entry in row["scores"])
    return dict(output_ids=candidate["output_token_ids"], finish_reason=candidate["finish_reason"],
                score_violations=violations, missing_reference_scores=missing, scores=rows,
                median_ms=statistics.median(candidate["request_ms"]))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--oracle-sha256", required=True)
    parser.add_argument("--gpu-index", default=GPU_INDEX)
    parser.add_argument("--gpu-uuid", default=GPU_UUID)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--case", choices=("primary", "alternate", "single", "configured_stop"),
                        action="append")
    parser.add_argument("--timeout", type=float, default=300)
    return parser.parse_args()


def main():
    args = parse_args()
    args.binary = args.binary.resolve(strict=True)
    args.checkpoint = args.checkpoint.resolve(strict=True)
    args.oracle = args.oracle.resolve(strict=True)
    args.evidence = args.evidence.resolve()
    require(sha256(args.oracle) == args.oracle_sha256, "Frozen oracle digest mismatch")
    oracle = json.loads(args.oracle.read_text())
    require(oracle["label"] == FP8_ORACLE and oracle["gpu_uuid"] == GPU_UUID
            and args.checkpoint == Path(oracle["model"]), "Oracle provenance mismatch")
    args.evidence.mkdir(parents=True)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=args.gpu_uuid, CUDA_DEVICE_ORDER="PCI_BUS_ID")
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    identity = dict(binary=str(args.binary), binary_sha256=sha256(args.binary),
                    oracle=str(args.oracle), oracle_sha256=args.oracle_sha256,
                    gpu_index=args.gpu_index, gpu_uuid=args.gpu_uuid, cuda_ordinal=0)
    write_json(args.evidence / "identity.json", identity)
    reports = {}
    names = args.case or list(oracle["requests"])
    with (BASELINE / f"{args.gpu_uuid}.lock").open("r") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        check_gpu(args.evidence, "before", args.gpu_index, args.gpu_uuid)
        for name in names:
            try:
                request = oracle["requests"][name]
                reports[name] = run_case(args, request, env, args.evidence, 0, 1, True)
            except (OSError, ValueError, KeyError, subprocess.TimeoutExpired) as error:
                reports[name] = dict(error=str(error))
            write_json(args.evidence / "comparison.json", reports)
        check_gpu(args.evidence, "after", args.gpu_index, args.gpu_uuid)
    require(sha256(args.binary) == identity["binary_sha256"], "Binary changed during verification")
    failed = [
        name for name, report in reports.items()
        if "error" in report or report["score_violations"] or report["missing_reference_scores"]
    ]
    for name, report in reports.items():
        if name in failed:
            print(f"{name}: FAIL ({report.get('error', 'raw logit comparison failed')})")
        else:
            print(f"{name}: {report['finish_reason']}, IDs {report['output_ids']}, "
                  f"unobserved reference scores {report['missing_reference_scores']}")
    return bool(failed)


if __name__ == "__main__":
    raise SystemExit(main())
