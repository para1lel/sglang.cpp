# Select verification by contract and risk

Use this procedure for a bounded implementation review or verification assignment.
Keep the acceptance question explicit before choosing an instrument.

## Prefer a complete static argument

Start with the changed contract and affected execution path.
For each obligation, identify inputs, assumptions, state transitions, and observable outcome.
Use source analysis when those facts establish the result.
Examples include local bounds proofs, unreachable invalid states, and link targets verified against fixed source.
Record the reasoning and exact source examined.

Static review is insufficient when essential behavior depends on unavailable runtime facts.
Examples include dynamic dispatch, loaded binaries, asynchronous completion, environmental policy, and measured latency.
Identify the gap and choose a check that distinguishes success from failure.
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
Repeat expensive checks after a relevant change, failure, or unresolved concern.
Once necessary checks pass, justify additional repetitions.

Keep module commands reproducible, with prerequisites, expected behavior, and conditions for broader checks.
Have doc maintain those procedures as current contracts.
Keep the actual run record in ignored evidence.

## Add only distinct failure detection

Before adding a case, identify the uncovered behavior or invariant.
Extend its existing owner when practical.
Use representative boundaries and interactions.
Avoid unjustified full combinations of shapes, dtypes, devices, or options.
Use assertions that distinguish the relevant outcome.
Avoid checks that merely restate implementation text or count generated items.

For a serious defect, retain its exposing public-path regression and a valid control.
When feasible, show the original implementation fails for the intended reason.
Syntax errors, missing dependencies, and unrelated failures cannot establish that regression.
If source analysis proves the original failure, record that fact.
Do not patch missing production integration merely to make a test pass.

Preserve required behavior when merging or removing redundant tests.
Retire a case when its contract is explicitly superseded or another case retains its distinct protection.
Record intentional expectation changes and their contract reasons.
Have an independent review assess changes to tolerances, assertions, fixture behavior, and test configuration.
Keep a known valid control when reducing regressions.

Testing may precede implementation when helpful.
Testing first is optional.
Coverage percentages, test totals, source-to-test ratios, and mutation counts have no universal acceptance targets.
Use project-specific requirements only when they apply to the assignment.

## Use controls when they answer a question

A negative control deliberately violates the invariant under test.
Run the valid candidate with the same instrument as a positive control.
Record the exact perturbation, expected failure, actual result, and restoration or disposal of the isolated copy.
Equivalent perturbations cannot establish changed-behavior detection.

For a non-obvious detector, a control can expose a blind observer or wrong execution path.
For a new gate, exercise an actual failing staged input and a valid staged control.
For ordinary changes with a complete proof, no mutation is required.
Do not add tests merely because a reviewer can invent arbitrary mutations.

Validate public reachability through the actual entry and branch.
If runtime reachability is uncertain, controlled call removal can help.
Use a static path argument when it fully establishes the same obligation.
Keep measured observer reachability separate from implementation reachability.

## Choose an instrument that can see the defect

Use source or instruction assertions for exact ordering or instruction-selection contracts.
Use race, memory, or synchronization checks for those obligations.
A numerical comparison alone cannot establish race freedom.
Preserve shared addresses and conflicting accesses in a race probe.

Freeze source during source-introspecting collection or compilation.
Check imported implementation paths when the runner uses another checkout or installed package.
Check environment and artifacts when stale caches can hide source changes.
For production claims, initialize production policy and verify that the changed branch ran.

If a selected run collects zero cases or skips all cases, record no executed acceptance evidence.
Treat timeout and missing required tools as unresolved verification.
A failed selected check blocks its intended claim.
Return the obligation to main without substituting a nearby passing fixture.
