---
name: principle-performance
description: Rules for repeatable benchmarks, profiling, and supported performance claims.
disable-model-invocation: true
---

# Performance engineering principles

Use this domain with the project's adopted general rules.
Establish correct behavior before you compare performance.

For GPU optimization, cache identity, unstable timing, and performance-claim review,
read [GPU measurement](references/gpu-measurement.md).

## Define the measurement

State the workload, metric, baseline, and system configuration.
Choose a workload that represents the actual use.
Identify which work is inside the measured region.
Keep setup, compilation, transfer, and execution costs explicit.
Include each cost in the result that claims to cover it.
Measure loading, residency, and first-request costs when the acceptance contract includes them.

Use primary references for the selected tools:

- [Google Benchmark user guide](https://google.github.io/benchmark/user_guide.html).
- [Nsight Systems user guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html).
- [Nsight Compute profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html).
- [CUDA programming guide](https://docs.nvidia.com/cuda/cuda-programming-guide/).

## Establish a valid baseline

Run the intended implementation.
Make sure that it produces the required result.
Isolate reference outputs from candidate mutations.
Give each candidate an independent input set when candidates can change storage.
Preserve required aliases and offsets within each input set.
Keep validation copies outside the timed region unless the metric explicitly includes them.
A reference that shares the candidate's defect cannot establish correctness.
Use comparable data, precision, hardware, and tool settings for both implementations.
Separate measurements at startup from measurements during steady operation.
Use warmup when the claim concerns steady operation.
Complete asynchronous work at the measurement boundaries.
Make sure that the timer measures the intended work.

Repeat enough measurements to characterize variation.
Report the aggregation method and relevant spread.
Keep the commands and raw results needed to repeat the comparison.

## Explain the result

Use profiles or other direct evidence to identify the limiting resource.
For an unexpected latency failure in compiled code, examine the relevant frontend,
instruction, resource, synchronization, and launch evidence before assigning a cause.
A failed latency ratio alone does not identify the responsible layer.
Determine whether the proposed change addresses that limit.
Distinguish useful work from retries, caches, skipped work, and measurement overhead.
Make sure that the result persists on the workload covered by the claim.

Report absolute measurements with a relative improvement.
State the measured configuration and the claim's limits.
When acceptance applies to each workload, keep each correctness and threshold result visible.
Aggregate improvements cannot close a required case that still fails.
A correct rejection diagnostic does not establish support for the rejected workload.
Treat an unavailable target or an unexecuted benchmark as missing evidence.
Do not present an estimate as a measured result.

## Keep useful knowledge

Keep repeatable measurement procedures and supported design decisions in current component documents.
Keep baseline identities, commands, raw samples, hardware, and tool versions in ignored evidence artifacts.
Record the relevant summary and limits in the commit or handoff.
Preserve evidence needed for later review in a developer-approved archive.
Historical measurements do not establish performance of the current revision.
When an observer is corrected, identify which claims failed and which earlier numerical evidence remains valid.
Repeat the affected observation with fixed operation inputs instead of discarding valid evidence without cause.
Use project lessons for failed optimization attempts that establish a useful constraint.
For an optional reusable measurement correction, use an externally supplied `meta-mode` workflow.
Treat the library target as a separate task input.
