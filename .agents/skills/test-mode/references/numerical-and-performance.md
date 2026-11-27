# Verify numerical and performance claims

Read this reference for numerical operations, GPU paths, optimization, and timing.
Use relevant supplied domain guidance for backend-specific semantics and tools.
Select checks for changed contracts rather than every possible parameter combination.

## Preserve the numerical question

Identify operation inputs, output shape, layout, dtype, accumulation policy, and allowed error.
Use a trusted independent reference for the property tested.
Use exact comparisons for discrete metadata and required bitwise behavior.
Justify numerical tolerances from the contract and reference precision.
Control reduced-precision reference behavior, including TF32 when relevant.
Wider tolerances do not explain a divergence.

Consider representative supported boundaries:

- Empty, unit, tail, non-power-of-two, and maximum supported geometry.
- Contiguous and supported noncontiguous strides, aliases, offsets, and alignment.
- Optional parameters, dispatch choices, and required fallbacks.
- Supported dtypes, nonfinite inputs, large values, and small positive values.
- Gradients for each differentiable input when backward behavior is part of the contract.
- Variable lengths, initial and final state, and relevant chunk boundaries.

Choose cases that expose distinct behavior.
Large-offset evidence can use a proof or a suitable probe when allocation costs prevent direct execution.
Preserve source reads and conflicting addresses when checking races.
Use applicable memory, race, synchronization, or instruction evidence for those obligations.

For suspected uninitialized reads, poison relevant unwritten or padding storage when it can expose the defect.
Confirm the poison reaches the real allocation and operation.
An allocator patch that misses the production allocation cannot establish the claim.
Keep poison and instrumentation outside performance measurements.

When a changed autotune policy can select different generated implementations, assess affected legal configurations.
Report which configurations ran.
Identify capped subsets as such.
Validate compilation and cache policy so stale artifacts cannot hide the candidate.

For graph-related changes, examine replay after relevant input, plan, or backend changes.
For lifetime changes, consider setup failure, growth, cancellation, closure, and delayed completion.
Use [resource ownership](../../coding-mode/references/resource-lifetimes.md) to identify the transition that needs evidence.
An eager pass or numerical match alone cannot establish graph lifetime safety.

## Diagnose before changing the contract

Preserve the failing input, source identity, applicable seed, expected result, and actual result.
Minimize the example while preserving the failed invariant.
Locate the earliest incorrect state or operation.
Use synchronization diagnostics to locate asynchronous failures when needed.
Use backend memory and synchronization tools for illegal access or race questions.
Check offset overflow, precision, padding reads, stale compilation, and partial-warp observation when relevant.

An unavailable diagnostic remains a limitation.
Do not substitute tolerance changes, reduced workloads, or a different oracle for the original question.
Return a required semantic change to main for an explicit contract decision.

## Measure a fixed comparison

Establish correctness for the exact performance candidate and workload before timing.
Freeze reference behavior, input distribution, public signatures, tolerances, and timing boundaries.
Keep acceptance criteria fixed across candidates.
A complete replacement with a vendor operation does not satisfy an assignment to optimize the existing implementation.
Other authorized implementation choices remain governed by the task.

Compare baseline and candidate with matching effective policies, hardware, inputs, and cache conditions.
Interleave comparisons in one session when drift can affect the result.
Keep mutable inputs independent between implementations and preserve required internal aliases.
Measure complete useful work.
Include compilation, transfer, launch, synchronization, replay, or setup in metrics that claim those costs.
Separate cold startup from steady state.

Use a timer that measures the intended asynchronous work.
Complete work at measurement boundaries.
Record warmup, repetition, aggregation, and spread.
Define noise criteria before interpreting the result.
Noisy data does not establish improvement or regression.
Performance thresholds belong to the project contract, not universal defaults.

## Support the claim

Report:

- Baseline and candidate identities with absolute times and relative results.
- Hardware, driver, compiler, relevant library versions, and effective policy.
- Workload parameters, precision, cache conditions, timer, and timing boundary.
- Warmup, repetitions, aggregation, spread, and noise status.
- Every required workload's result, including failures and regressions.
- The bottleneck explanation, raw evidence identity, and claim limitations.

Use profiles to locate bottlenecks.
Uninstrumented latency claims require evidence beyond profiler timings.
Instruction presence alone cannot prove overlap.
High occupancy alone cannot prove speed.
Investigate unexplained large speedups for skipped work, early exits, stale code, or invalid timing.
Show that the measured path actually executed the changed branch.

Keep each required workload failure visible.
Aggregate gains cannot close failed required cases.
Valid measurements can show no improvement.
Do not add or multiply speedups from separate attempts.
Results on different hardware do not establish the target result.
Kernel-only timing supports a kernel claim; application claims require their own complete boundary.

## Bound an optimization conclusion

Keep or reject a candidate using correctness, repeatable benefit, explanatory evidence, and added complexity.
Specialize only when workload or bottleneck differences justify the branch and its maintenance cost.
Compare specializations against the same baseline.
Record failed attempts when they establish a reusable constraint.

A claimed optimization limit requires a measured baseline and reasoned candidate.
Give the actual check result with a quantified bound or specific obstacle.
An unavailable profiler or one losing candidate does not establish that no improvement is possible.
Return estimates as estimates and unexecuted measurements as missing evidence.
