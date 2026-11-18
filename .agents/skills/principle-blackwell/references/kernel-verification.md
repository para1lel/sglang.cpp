# Kernel contracts and verification

Read this reference when changing a kernel's input domain, numerical policy, dispatch, state, or memory behavior.
Use the source project's supported contract to select applicable cases.
This reference does not require a particular test framework or an exhaustive parameter product.

## Define the operator contract

Record the public entry, supported shapes, strides, dtypes, layouts, device targets, and optional arguments.
Describe variable-length metadata, initial and final state, outputs, and required gradients.
Specify input conversion, computation and accumulation precision, result storage, and nonfinite behavior.
State each reference's scope and the justified numerical error criterion.
Describe backend selection, fallback conditions, and cross-implementation invariants.
Identify affected shape ranges, architectures, compiler versions, and callers.
A passing test result is not an impact analysis.

Derive verification cases from these obligations.
Keep reference implementations, input distributions, assertions, tolerances, and timing boundaries fixed during a candidate comparison.
Review an intentional contract change explicitly.
A precision or algorithm change needs its own justification, even when the old tolerance still passes.
Use the project's authorization rules for changes to its acceptance contract.

## Select cases that separate dimensions

| Contract branch | Useful exposing cases |
| --- | --- |
| Layout | Contiguous and supported noncontiguous strides; sequence-first and head-first layouts when supported. |
| Dimensions | Different key/value widths and query/value head counts; distinguish dimensions that equal-sized tests can confuse. |
| Tails and alignment | Length 0 or 1 when supported, non-power-of-two lengths, partial tiles, and chunk sizes not divisible by tile size. |
| Optional arguments | Omitted parameters, `None`, and values that select specialized code, such as a scale of 1. |
| Variable length | Sequence boundaries, empty and single-token sequences when supported, and irregular `cu_seqlens`. |
| State | Initial-state input, final-state output, chunk transitions, and whole versus segmented execution with state passing. |
| Gradients | Every differentiable input and every differentiable output, including initial and final state. |
| Numerics | Every supported precision policy, large and small magnitudes, required nonfinite behavior, and masked padding. |
| Routing | Optimized and fallback implementations, dispatch enabled and disabled, and each changed backend. |
| Address range | A supported shape or a suitable address probe that crosses the relevant signed 32-bit offset boundary. |

Use independent seeded input sets for repeatable failures.
Give multiple outputs independent upstream gradients when one loss could hide an unused output.
Enable gradients for initial state when the contract makes it differentiable.
Compare against an independent mathematical implementation and the operation's stepwise definition where useful.
Two references that share a defect do not establish correctness.

For causal operations, perturb future tokens and examine the required prefix outputs and gradients.
For stateful operations, compare whole execution with segmented execution and state transfer.
Use exact equality for discrete state and contractually exact behavior.
Use justified tolerances for numerical comparisons.
Record why a branch does not apply rather than implying that it was exercised.

## Poison uninitialized and masked storage

Unwritten outputs can pass when allocator contents happen to resemble the expected values.
In an isolated correctness run, fill relevant newly allocated floating or complex storage with NaN before kernel use.
Cover the allocation paths the operation actually uses.
The old implementation wrapped `empty`, `empty_like`, `empty_strided`, and `Tensor.new_empty`.
That list is an example, not proof that other allocation paths are covered.

Use a known unwritten-output defect to establish that the instrumentation detects the failure.
Also run a correct implementation with fully written valid outputs as a passing control.
Verify that the wrapper actually executes on the tested device and allocation path.
When instrumentation is required, report a missing import, failed fill, or unsupported allocator as unavailable or failed evidence.
Propagate the failure instead of silently continuing.
Keep poisoning outside timing measurements and restore temporary instrumentation after the probe.

Separately put NaN in padding or masked inputs that the contract declares irrelevant.
Check that valid outputs remain unaffected.
Choose a selection or masked-access operation that preserves the required nonfinite semantics.
Multiplication by zero cannot neutralize an already evaluated NaN or infinity.
Inspect whether a selection expression still evaluates an unsafe load or arithmetic operation.

The retired poisoning plugin swallowed fill errors.
Its implementation is not a valid template for failure handling.
A wrapper being loaded does not establish successful instrumentation.

## Verify configurations and actual execution

Exercise every autotune configuration that remains eligible for the claimed supported input domain.
Identify the launched kernel, configuration identity, shape, compiler, and target architecture.
Do not infer complete coverage from the configuration that the tuner happened to choose.
An index cap or a missing launch record leaves a coverage gap.
A failed supported configuration is a defect; an intentionally unsupported configuration needs an explicit legality guard.

Repeat affected cases across the relevant compiler and hardware matrix.
Keep unexecuted combinations visible.
Check version-specific autotuner APIs before pinning configurations.
A plugin that returns after an unavailable import has not established autotune coverage.
Positive controls must reach the actual compiled launch path.

Repeat the same input when determinism is required.
Bitwise agreement can expose some races but does not prove race freedom.
Use memory, race, or synchronization tools appropriate to the defect.
Numerical success does not replace ordering evidence.

Python line coverage does not measure a compiled JIT kernel body.
Establish launcher reachability and device execution with suitable probes.
Limit any coverage exclusion to identified compiled regions.
Do not exempt ordinary Python control flow merely because it calls a JIT kernel.

## Preserve production numerical policy

Use a mathematical reference with the precision required by the contract.
Disable TF32 for a reference when TF32 would change that reference's intended arithmetic.
Keep that choice separate from the production kernel's declared precision.
A global setting that forces candidate kernels into IEEE arithmetic can conceal a production TF32 failure.
Record effective settings and compare the production configuration as it will run.
Do not change tolerances before locating the first incorrect assumption or divergent intermediate.

## Trace address width and graph lifetimes

Trace program identifiers, loop indices, tuple assignments, and derived offsets through address arithmetic.
Promote operands before a potentially overflowing multiplication or addition.
Casting an already overflowed result is too late.
Justify bounded narrower arithmetic from the supported geometry.
A source pattern or regular expression cannot prove that every derived address is safe.
Use direct internal-entry guards as well as public validation.

For graph changes, capture and replay with changed supported inputs, plans, and backend choices.
Examine captured buffers, descriptors, graph ownership, and in-flight replay lifetimes.
An eager run alone does not establish replay validity.

## Diagnose a failure

Reduce to one input shape and fixed source, environment, and seed.
Use synchronous launch diagnostics to locate an asynchronous failure when appropriate.
Use memory, race, and synchronization probes for their respective questions.
Inspect accumulators, division, exponential range, masks, and padding for nonfinite failures.
Inspect every index-times-stride expression for large-shape failures.
In warp-specialized code, a diagnostic restricted to thread zero observes only the warp containing that thread.
Establish that the suspected producer or consumer actually reaches the diagnostic.
Check imported source and JIT/autotune cache identity before attributing a stale artifact to new code.

## Historical sources and limits

These lessons were migrated from meta-harness base `b9d9427a9d827aa6c6964f24d89da01c14183c60` on November 17, 2026.
Recover exact files with `git show REV:PATH` in that repository.
The following are historical source identities, not installed dependencies:

- `skills/meta-harness/packs/kernel-lib/pack.yaml`
- `skills/meta-harness/packs/kernel-lib/design/kernel-contract.md`
- `skills/meta-harness/packs/kernel-lib/skills/kernel-contract/SKILL.md`
- `skills/meta-harness/packs/kernel-lib/skills/kernel-correctness/SKILL.md`
- `skills/meta-harness/packs/kernel-lib/checks/kernel_checks.py`
- `skills/meta-harness/packs/kernel-lib/bugs/kernel.yaml`
- `skills/meta-harness/templates/_shared/skills/debug-kernel-crash/SKILL.md`
- `skills/meta-harness/runtime/agent/mh_poison.py`
- `skills/meta-harness/runtime/agent/mh_autotune.py`
- `skills/meta-harness/runtime/agent/correctness.py`
- `skills/meta-harness/pitfalls/ai-infra.yaml`
- `docs/overnight-2026-11-15.md`

The original pack combined reusable detection ideas with framework-specific policy.
This reference retains the algorithms and their limits.
It does not retain global test quotas, universal test-first requirements, coverage targets, or vendor-library bans.
