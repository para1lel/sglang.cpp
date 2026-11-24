"""Collect paired concurrent cohorts and compare their replayed makespans."""

from statistics import median
from unittest.mock import patch

from benchmarks.concurrent import client
from measure_concurrent import main as collect


K4_MAKESPAN_CRITERION = "native-median-at-most-110-percent"


def makespan_criterion(identity):
    if "makespan_criterion" not in identity:
        return None
    criterion = identity["makespan_criterion"]
    if criterion != K4_MAKESPAN_CRITERION:
        raise ValueError("Unknown K4 makespan criterion")
    return criterion


def cohort_makespan_ns(records):
    if any(type(record["submit_ns"]) is not int or type(record["done_ns"]) is not int
           or record["done_ns"] <= record["submit_ns"] for record in records):
        raise ValueError("Expected positive client makespan from integer nanosecond timestamps")
    earliest = min(record["submit_ns"] for record in records)
    latest = max(record["done_ns"] for record in records)
    return latest - earliest


def validate_cohort(raw, lanes, policy):
    instrumentation = raw["instrumentation"]
    if raw["errors"] or any(key in instrumentation
                            for key in ("error", "stop_error", "deadline_exceeded")):
        raise ValueError(f'Cohort {raw["tag"]} failed: {raw["errors"] or instrumentation}')
    records = [client.replay_lane(source, item)
               for source, item in zip(raw["requests"], lanes, strict=True)]
    stream_ids = [item["stream_id"] for item in records if item["stream_id"] is not None]
    if len(set(stream_ids)) != len(stream_ids):
        raise ValueError("Completion IDs must be distinct across the cohort")
    earliest = min(item["submit_ns"] for item in records)
    span = (max(item["submit_ns"] for item in records) - earliest) / 1e6
    earliest_done = min(item["done_ns"] for item in records)
    if (type(raw["barrier_ns"]) is not int
            or not max(item["connect_ns"] for item in records) <= raw["barrier_ns"] <= earliest
            or max(item["sent_ns"] for item in records) >= earliest_done
            or span > policy["max_submission_span_ms"]):
        raise ValueError("Cohort submissions failed overlap bounds")
    makespan = cohort_makespan_ns(records)
    duration = makespan / 1e9
    prompt = sum(record["usage"]["prompt_tokens"] for record in records)
    output = sum(len(record["output_ids"]) for record in records)
    return {"makespan_ns": makespan,
            "output_tokens_per_second": output / duration,
            "requests_per_second": len(records) / duration,
            "aggregate_tokens_per_second": (prompt + output) / duration,
            "submission_span_ms": span,
            "all_streams_overlap_ms":
                (earliest_done - max(item["headers_ns"] for item in records)) / 1e6,
            "all_generating_overlap_ms":
                (earliest_done - max(item["first_ns"] for item in records)) / 1e6,
            "request_latencies": [record["metrics"] for record in records]}


def compare(native, vllm, eligible, *, criterion=None):
    if criterion is not None and criterion != K4_MAKESPAN_CRITERION:
        raise ValueError("Unknown K4 makespan criterion")
    if eligible and (not native or native.keys() != vllm.keys()
                     or any(len(samples) != len(vllm[name]) for name, samples in native.items())):
        raise ValueError("Paired makespan requires matching workload and measured sample counts")
    workloads = {}
    for name, samples in native.items():
        control = vllm.get(name)
        native_median = median(samples)
        vllm_median = median(control) if control is not None else None
        strict_passed = native_median < vllm_median if eligible else None
        workloads[name] = {
            "native": {"samples_ns": samples, "median_ns": native_median},
            "vllm": {"samples_ns": control, "median_ns": vllm_median}
            if control is not None else None,
            "passed": strict_passed,
        }
        if criterion is not None and eligible:
            workloads[name].update(
                passed=10 * native_median <= 11 * vllm_median,
                native_to_vllm_median_ratio=native_median / vllm_median,
                strict_passed=strict_passed,
            )
    result = {"passed": all(row["passed"] for row in workloads.values()) if eligible else None,
              "workloads": workloads}
    if criterion is not None:
        result["makespan_criterion"] = criterion
    return result


if __name__ == "__main__":
    with patch.object(client, "validate_cohort", validate_cohort):
        raise SystemExit(collect())
