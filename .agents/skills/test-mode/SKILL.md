---
name: test-mode
description: Review correctness and verify distinct behaviors with proportionate checks and fixed evidence.
disable-model-invocation: true
---

# Test mode

Determine whether the candidate meets its assigned contract.
Prefer static review when source and invariants can prove correctness.
Execute checks for behavior that static review cannot establish.
Own only assigned test, fixture, benchmark, check, and ignored evidence paths.

Read [the worker contract](../main-mode/references/delegation.md#worker-contract).
Main supplies the exact candidate and explicit mode instructions.
Main owns the maintained target's branches, worktree registration, selected index, commits, and integration.
Review without repairing production files.
Return production defects to main for coding.

## Establish the obligations

Read repository instructions, requirements, affected contracts, relevant lessons, and the candidate diff.
Use [coding's standards](../coding-mode/references/code-standards.md) when reviewing implementation.
For build or observer claims, read [execution identity](../coding-mode/references/execution-identity.md).
For cleanup, concurrency, or asynchronous state, read [resource lifetimes](../coding-mode/references/resource-lifetimes.md).
Read only the relevant supplied domain references.

Identify the observable behavior and necessary invariants the candidate changes.
Separate required outcomes from added author claims.
Assess the approach, its assumptions, and any relevant simpler alternative.
Check existing tests and evidence before adding work.

Read [test selection](references/test-selection.md) to choose static review, existing checks, or new cases.
Read [evidence integrity](references/evidence.md) before accepting execution, observer, or gate results.
For numerical, GPU, optimization, or timing claims, also read [numerical and performance checks](references/numerical-and-performance.md).

## Review first

Trace required behavior from public entry through the changed implementation to its observable result.
Give the preconditions and proof for each statically provable obligation.
Check boundaries, callers, effective configuration, error paths, ownership, and compatibility.
Do not label a source reading as an executed test.

Use a fresh worker for independent acceptance review when main requests one.
Keep the reviewed working tree stable.
Run perturbations in an assigned isolated copy or [disposable fixture repository](../main-mode/references/delegation.md#disposable-git-fixtures).
Test can prepare independent fixtures; main registers any target-linked worktree.
An independent reviewer can write the assigned report and separately assigned tests.
The reviewer cannot repair the candidate while claiming to review the original diff.

## Select proportionate verification

Use existing module checks when they already protect the contract.
Add a test only for a distinct observable behavior or necessary invariant.
Keep meaningful failure detection while reducing duplicate cases.
There is no mandatory TDD sequence, test count, coverage percentage, or mutation for every change.
An instruction-only or mechanically obvious change can finish with supported static and link review.

Use negative controls when they can establish that a check detects the relevant defect.
Preserve independent expected results before running the candidate.
An actual failing control is required when validating a new or changed enforcement gate.
A gate that checks itself with the same defective assumption needs an independent control.

For selected execution, zero cases, all skips, timeout, and missing tools provide no passing acceptance evidence.
Keep reasoned static conclusions separate from empty runs.
Check the actual production configuration and changed branch when the claim concerns production behavior.

## Report findings and results

Bind the result to the exact source, diff, configuration, and evidence identities.
For each required outcome, give its static argument or actual check and result.
Distinguish passed, failed, unavailable, skipped, and unexecuted checks.
State whether acceptance is supported, changes are required, or evidence is insufficient.

For each defect, give its trigger, observable impact, source location, and proposed correction.
Distinguish introduced and pre-existing defects, optional improvements, and verification gaps.
A serious demonstrated correctness defect remains blocking.
On repeat review, reassess prior findings and the changed diff without adding unrelated demands.
Expand for affected consumers or new correctness evidence.

Store raw logs, samples, sanitizer output, and review reports in assigned ignored artifacts.
Use `artifacts/` when the project has no established evidence location and main assigns one.
Send durable verification procedures and lesson conclusions to doc through main.
Return test changes, exact candidate identity, command results, evidence hashes, limitations, and the weakest unverified assumption.
Do not stage or commit target changes, publish, or declare global completion.
