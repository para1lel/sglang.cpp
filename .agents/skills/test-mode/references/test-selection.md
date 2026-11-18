# Select verification by contract and risk

Use this procedure for a bounded implementation review or verification assignment.
Keep the acceptance question explicit before choosing an instrument.

## Prefer a complete static argument

Start with the changed contract and affected execution path.
For each obligation, identify its inputs, assumptions, state transitions, and observable outcome.
Use source analysis when those facts establish the result.
Examples include a local bounds proof, an unreachable invalid state, or a link target verified against fixed source.
Record the reasoning and the exact source examined.

Static review is insufficient when essential behavior depends on unavailable runtime facts.
Examples include dynamic dispatch, a loaded binary, asynchronous completion, environmental policy, and measured latency.
Identify that gap and select a check that can distinguish success from failure.
An assertion that the code looks correct is not a proof.
A passing test is also limited to its exercised inputs and observation path.

## Reuse the module's checks

Group checks by the production contract they protect.
Use current module verification commands and affected-consumer mappings.
Keep integration, hardware, and performance groups separate.
Ask doc to repair a stale procedure through main.

| Change | Initial selection | Reason to broaden |
| --- | --- | --- |
| Local module behavior | Static review and applicable module checks. | Changed callers or boundary assumptions. |
| Shared interface or representation | Owner checks and necessary consumers. | Other modules share the changed contract. |
| Fixture or test configuration | Affected collection and execution. | Discovery or result interpretation changes. |
| Lifetime, ordering, or cache | Relevant transitions and suitable observers. | Concurrent users, replay, or publication changes. |
| Build, package, or dependency | Effective build and installed-path checks. | Transitive consumers or source identity changes. |
| Instructions or documents | Semantic, language, metadata, and link review. | Changed tools or executable procedures. |

Use a full suite for a broad shared change, release requirement, or explicit request.
A handoff alone does not require a full run.
Repeat an expensive check after a relevant change, failure, or unresolved concern.
Once the necessary checks pass, additional repetitions need a reason.

Keep module commands reproducible, with prerequisites, expected behavior, and conditions for broader checks.
Have doc maintain those procedures as current contracts.
Keep the actual run record in ignored evidence.

## Add only distinct failure detection

Before adding a case, identify the behavior or invariant absent from existing coverage.
Extend its existing owner when practical.
Use representative boundaries and interactions.
Avoid an unjustified full product of shapes, dtypes, devices, or options.
Use assertions that distinguish the relevant outcome.
Avoid checks that merely restate implementation text or count generated items.

For a serious defect, retain its exposing public-path regression and a valid control.
When feasible, show the original implementation fails for the intended reason.
A syntax error, missing dependency, or unrelated failure does not establish that regression.
If source analysis already proves the original failure, record that fact accurately.
Do not patch missing production integration merely to make a test pass.

Preserve required behavior when merging or removing redundant tests.
Retire a case when its contract is explicitly superseded or another case retains its distinct protection.
Record intentional changes to expected behavior and the contract reason.
Have an independent review assess changes to tolerances, assertions, fixture behavior, and test configuration.
Keep a known valid control when reducing regressions.

Testing can precede implementation when it helps.
It is not a required sequence for every change.
Neither coverage percentages, test totals, source-to-test ratios, nor mutation counts are universal acceptance targets.
Use project-specific requirements only when they apply to the assignment.

## Use controls when they answer a question

A negative control deliberately violates the invariant that a check claims to protect.
Run the valid candidate with the same instrument as a positive control.
Record the exact perturbation, expected failure, actual result, and restoration or disposal of the isolated copy.
Equivalent perturbations do not establish detection of changed behavior.

For a non-obvious detector, a control can expose a blind observer or wrong execution path.
For a new gate, exercise an actual failing staged input and a valid staged control.
For ordinary changes with a complete proof, no mutation is required.
Do not add tests merely because a reviewer can invent arbitrary mutations.

Validate public reachability through the actual entry and branch.
A controlled removal of a call can be useful when runtime reachability is uncertain.
Use a static path argument when it establishes the same obligation completely.
Keep the measured observer's own reachability separate from the implementation's reachability.

## Choose an instrument that can see the defect

Use source or instruction assertions for exact ordering or instruction-selection contracts.
Use race, memory, or synchronization checks for those obligations.
A numerical comparison alone cannot establish race freedom.
Preserve shared addresses and conflicting accesses in a race probe.

For source-introspecting collection or compilation, freeze source through the operation.
Check imported implementation paths when a runner uses another checkout or installed package.
Use environment and artifact checks when stale caches can hide source changes.
For production claims, initialize the production policy and establish that the changed branch ran.

If a selected run collects zero cases or skips every case, record no executed acceptance evidence.
Treat timeout and missing required tools as unresolved verification.
A failed selected check blocks the claim it was selected to establish.
Return the remaining obligation to main without replacing it with a nearby passing fixture.
