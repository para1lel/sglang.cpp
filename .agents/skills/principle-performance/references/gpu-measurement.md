# GPU optimization and measurement

Read this reference when planning a GPU optimization, diagnosing unstable timing, or reporting a GPU performance claim.
Select the project's existing benchmark when its timing contract is suitable.

## Fix the comparison

Identify baseline and candidate source, workload parameters, numerical policy, references, and expected useful output.
Keep assertions, input distributions, tolerances, and accepted workload coverage fixed during comparison.
An intentional contract change needs separate review and evidence.
Unexpected large speedups require a check for missing work, skipped branches, stale caches, and changed timing boundaries.

Establish correctness before accepting performance.
Verify actual test collection and execution.
Zero collected cases, all cases skipped, cancellation, timeout, or a missing required tool do not supply passing evidence.
Inspect effective environment settings when they can downgrade numerical failures to warnings.
A green report can conceal a weaker assertion policy.

Keep candidate inputs independent when either implementation mutates storage.
Preserve required aliases and offsets within each input set.
Record actual imported source and relevant JIT, precompiled-package, and autotune cache identities.
When stale artifacts can conceal a source change, isolate or invalidate the affected cache.
Use a writable disposable cache when possible.
Keep unrelated shared caches and environments intact.
Exercise varying batches when compile-time specialization can cause repeated production compilation.

## Form and test one useful hypothesis

Use a timeline or kernel profile to identify the limiting work.
Possible causes include memory traffic, compute, synchronization, launch overhead, and resource occupancy.
Instruction presence does not establish overlap.
Higher occupancy alone does not establish lower latency.
Record the evidence that supports the hypothesis.
Make a bounded change and compare it against the same baseline.
Keep a supported result or record why the candidate was dropped or split.
Failed attempts can establish useful limits.

Add shape specialization when the contract and measured bottlenecks justify distinct paths.
Verify each path and its eligibility guard.
Use quantified bottleneck analysis to explain the result.
A claim that further optimization is blocked needs a measured baseline, a reasoned candidate, check results, and an identified bound.
An unavailable profiler or one losing candidate is insufficient.
Use the project's permitted implementation choices; evaluate their complete costs and numerical contract.

## Control the environment and timing

Record GPU model, architecture, driver, toolkit, compiler, device assignment, clocks, thermal state, and competing work.
A shared machine needs a reliable lease or equivalent coordination for the measured device.
Two independent locks that do not exclude each other provide no isolation.
Match the target hardware covered by the claim.
Measurements on another architecture are supplementary observations with their own limits.

Run baseline and candidate alternately in the same measurement session.
Warm up the work appropriate to a steady-state claim.
State whether L2 is warm or cold, how cache state is established, and whether that work is timed.
Use a suitable timer, such as events or the benchmark's supported profiling interface, and record any fallback.
Complete asynchronous work at timing boundaries.
Establish that the requested computation occurred within those boundaries.
Compile, startup, host, transfer, and device time need separate identities when measured separately.
An end-to-end claim needs its complete end-to-end measurement.

Repeat the comparison and report absolute times, the relative result, aggregation, and spread.
Use a noise criterion suitable for the observed distribution.
The historical CPU probe replaced a preemption-sensitive variation ratio with IQR/median while retaining both measurements.
That observation does not establish a universal statistic or threshold.
Busy devices, unstable clocks, or large variation can leave the result inconclusive.
Do not accept a noisy number as evidence merely because its speedup looks large.

Compare each required workload against its own acceptance criterion.
An aggregate improvement does not close a failed required case.
Speedups from separate experiments do not add or multiply into a measured total.
Remeasure the combined implementation against its baseline.

## Use profiles to explain

Use the installed version's supported profiler options.
For NVIDIA work, use a system timeline for launch, overlap, transfer, and host gaps.
Use kernel-level metrics for occupancy, DRAM and L2 traffic, stalls, bank conflicts, and instruction mix.
For AMD work, use its target-specific profiler and resource metrics, including register and local-storage usage.
Keep architecture-specific interpretation with the target evidence.

Write quantified observations before assigning a bottleneck.
A throughput percentage alone does not prove its cause.
Profiling instrumentation can perturb execution.
Use separate benchmark timing for latency comparisons when instrumentation changes the measured work.
Keep raw profiles with the fixed source and environment that produced them.

## Review a claim

Give the hardware identity, workload parameters, baseline and candidate revisions, absolute times, timer, cache conditions, and boundaries.
Include warmup, repetition, aggregation, noise, and per-case results.
State unexecuted targets and relevant resource-contention limits.
Independently verify the source and measured execution path when reviewing acceptance.
Static instruction checks, mathematical comparisons, launch observations, and timing establish different facts.
For layout, gradients, masks, state, autotune, index width, or replay, use the project's selected GPU verification guidance.
The optional Blackwell domain supplies a kernel verification reference for those obligations.

## Historical sources

The migration source is meta-harness `b9d9427a9d827aa6c6964f24d89da01c14183c60`.
Recover exact paths with `git show REV:PATH`:

- `skills/meta-harness/templates/_shared/skills/kernel-opt-loop/SKILL.md`
- `skills/meta-harness/templates/_shared/skills/profile-gpu/SKILL.md`
- `skills/meta-harness/templates/_shared/skills/perf-claim/SKILL.md`
- `skills/meta-harness/pitfalls/ai-infra.yaml`
- `skills/meta-harness/runtime/agent/perf.py`
- `skills/meta-harness/runtime/agent/bench_runner.py`
- `skills/meta-harness/runtime/agent/remote.py`
- `docs/spike-2026-11-15.md`
- `docs/overnight-2026-11-15.md`

Those sources record earlier experiments and implementation limits.
Their times and verdicts are not measurements of the current library or a new adopting project.
