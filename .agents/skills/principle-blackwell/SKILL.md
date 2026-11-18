---
name: principle-blackwell
description: Blackwell hardware references and rules based on evidence from GPU kernel development.
disable-model-invocation: true
---

# Blackwell engineering principles

Use this domain with the project's adopted general rules.
Read the source project's lessons and relevant kernel design before changes.

## Establish the target

Record the GPU model, compute capability, driver, CUDA version, and compiler version.
Select reference material for that target.
Examine feature support before an implementation for one architecture.
The Blackwell name alone does not establish support for an instruction or programming model.

Use these primary sources:

- [Blackwell tuning guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html).
- [CUDA programming guide](https://docs.nvidia.com/cuda/cuda-programming-guide/).
- [PTX instruction set](https://docs.nvidia.com/cuda/parallel-thread-execution/).
- [CUTLASS source and examples](https://github.com/NVIDIA/cutlass).
- [Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html).
- [Nsight Compute profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html).

Examine the version of a source before you use its API or hardware claims.
Keep exact source references with the project decision.

## Establish correctness

Keep a trusted reference implementation.
Examine output values, shapes, layouts, and data types.
Choose numerical tolerances from the operation's requirements.
Include boundary sizes, partial tiles, and relevant alignment cases.
Examine synchronization and memory access with tools suitable for the target.

Treat asynchronous operations, memory reuse, and barrier phases as explicit correctness obligations.
Examine the full producer and consumer path after a pipeline change.
A successful launch does not establish numerical or memory correctness.

For operator contracts, gradients, layouts, variable length, state, masks, autotune, and replay,
read [kernel verification](references/kernel-verification.md).
Use its controls and execution limits when reviewing a kernel change.

## Complete reads and resource lifetimes

Describe asynchronous work as publication, issue, completion, consumption, and reuse or release.
Use the target's required synchronization and proxy fences at each relevant boundary.
An allocated buffer or scalar temporary does not establish completion of its preceding reads.

Classify uniform entry separately from divergence in the body.
Complete shared control reads before later writes can replace their source values.
Examine loop bounds, predicates, macro arguments, and returns to uniform work as related sites.
Place a collective barrier only where all required participants can reach it.
Use the enclosing uniform join when an inner scope is divergent.

Establish the required driver context on each host thread before descriptor encoding or launch.
Restore temporary host state after success and failure.
Keep context ownership separate from stream, descriptor, and storage lifetimes.
Examine logical tensor views and descriptors before raw-pointer use, including output and empty-work paths.

Give native resources an explicit owner during setup as well as normal execution.
Unwind partially acquired resources when setup fails before handle publication.
Coordinate closure with admitted submissions and retire resources only after their actual users complete.
Wrapper destruction alone does not establish GPU completion.
Preserve the owning context through deferred cleanup.
Track graph ownership separately from each in-flight replay.
Keep resources while a graph can use them, and until submitted replays complete.
Use conservative retention when completion cannot be established, and report its reclamation limit.
Keep the complete old allocation set until every replacement is acquired and published.
Remove each allocation owner only after its release succeeds.
Keep failed and unattempted releases available for cleanup.
Use fault injection to exercise partial acquisition, growth, and release failures.

Reject unsupported geometry before products, offsets, or metadata submissions can overflow.
Apply that validation at direct internal entries as well as public wrappers.

## Verify the actual observation path

Examine launch symbols in the compiled objects before relying on launch counters or rejection guards.
Use a separately compiled launch as a positive observation control.
Source names and manual wrapper calls do not establish interception of the compiler's launch entry.
Use the actual toolkit's handle types and matching name-lookup APIs.
Keep numerical evidence separate from claims made by an incomplete observer.

## Optimize from evidence

Record a repeatable baseline before an optimization.
Profile the actual limiting work.
Select data movement, tensor operations, memory layouts, and scheduling from that evidence.
Examine the target's requirements for TMA, TMEM, and tensor instructions before use.
Examine generated instructions when the result depends on compiler lowering.

Measure the complete operation as well as the kernel when the application claim requires both.
Account for launch, transfer, setup, and synchronization costs in the relevant result.
Examine correctness again after each meaningful optimization.
Keep an optimization only when the measured benefit supports its added cost.
Keep eligibility and resource guards specific to the complete operation.
Examine the final resource use after code generation.
Use a correct fallback when an optimized path cannot meet its contract.
One successful tile or descriptor does not establish legality for a wider configuration.

## Keep useful knowledge

Keep reusable failures and measured design constraints in the source project's lessons and component documents.
Keep raw runs, samples, and sanitizer output in ignored artifacts.
Record hardware and tool versions with each finding.
For an optional reusable hardware correction, use an externally supplied `meta-mode` workflow.
Treat the library target as a separate task input.
Separate documented hardware behavior from an observation on one workload.
