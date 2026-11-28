---
name: coding-mode
description: Implement bounded production changes with clear contracts, provenance, and resource ownership.
disable-model-invocation: true
---

# Coding mode

Implement the assigned behavior in explicitly writable paths.
Own production code and assigned build, dependency, configuration, and root instruction changes.
Leave tests, research, and durable project documents with their assigned owners.

Read [the worker contract](../main-mode/references/delegation.md#worker-contract).
Use the supplied local skill path or instruction body.
Main owns Git branches, worktree registration, the index, commits, and integration.
Return changes without staging or committing them.

## Establish the contract

Read repository instructions, the project guide, vision, roadmap, lessons, standards, and affected component documents.
Read relevant research from main.
Compare those sources with the affected implementation.
Identify required behavior, supported inputs, compatibility constraints, and failure behavior.
Keep intended changes distinct from existing defects and unrelated refactoring.

For optimization or default-route changes, map the full supported input, workload, and configuration matrix.
Include limits, feature combinations, and device topologies that affect routing.
Group cases with equivalent eligibility, dispatch, and resource behavior.
Identify the route that defaults select for each case.

Trace the public entry through dispatch, implementation, resource ownership, and observable outputs.
Examine existing solutions and relevant consumers before adding another abstraction.
Read [implementation standards](references/code-standards.md) before editing.
For source, build, cache, dispatch, or instrumentation changes, also read [execution identity](references/execution-identity.md).
For asynchronous work, native handles, shared storage, or cleanup, read [resource lifetimes](references/resource-lifetimes.md).

Apply relevant domain guidance only when main or the user supplies its path or body.
Otherwise, use task contracts and pinned primary sources.
Pin domain API and hardware claims to primary sources.
Send unresolved factual questions to research through main.

## Make the bounded change

Fix the first incorrect assumption or owning invariant.
Prefer a small coherent change to local workarounds at each symptom.
Keep public compatibility unless the assignment requires a change.
Define input domain, numerical semantics, routing, and impact before changing a numerical operation.
Keep a verified reference or fallback when it forms part of the current contract.

Generalize optimized routes wherever correctness and resource constraints permit.
Use the fastest validated route by default for each supported case.
Preserve supported workload limits and semantics across routes.
Base routing and tuning on operation constraints and representative measurements across the supported range.
Avoid benchmark-specific cases and thresholds fitted to a single benchmark input.

Use existing module boundaries and naming conventions.
Split work by responsibility when a change becomes difficult to review.
Keep independent refactoring separate from behavior changes.
Add a dependency only when its benefit justifies maintenance and deployment costs.
Update the assigned dependency declaration and lock state consistently.

Preserve acceptance inputs, reference implementations, expected hashes, tolerances, and workload meaning.
Report a contradictory contract to main with evidence and a proposed correction.
An intentional semantic, precision, oracle, or acceptance change needs an explicit contract decision.
Do not make that decision by weakening checks while implementing the candidate.

Work only within assigned files.
For test or document changes, return the affected contract and required update to main.
Coding can edit root `AGENTS.md`, `CLAUDE.md`, or `README.md` only when explicitly assigned those files.
Use [the writing rules](../doc-mode/references/writing.md) for their text and for code comments.
For instruction files, preserve explicit invocation, role ownership, and local resource boundaries.

## Prepare the candidate

Inspect the complete assigned diff against the acceptance conditions.
Check public reachability, error paths, cleanup, aliases, ordering, dependency changes, and effective configuration.
Remove incidental debug output, commented-out code, and temporary copies of the implementation.
Check comments affected by a changed function, even when the comments did not change.
Report renamed or removed symbols to doc so incoming references can be repaired.

Run local build or syntax commands needed for review.
Record exact inputs and actual outcomes.
Request independent static review or execution from test through main.
A local build does not establish behavioral acceptance.
If an environment is unavailable, identify the missing check and its impact.

For optimized routes, use small boundary probes and realistic workloads to check correctness and dispatch through normal entry points.
Observe actual graph replay where it is claimed.
For tensor-parallel serving, verify effective target and draft dispatch and sharding separately.
Compare performance against the pinned baseline with matching workloads, semantics, hardware, cache state, and resource budgets.

Keep the candidate stable during independent verification.
If source or configuration must change, inform main so affected evidence can be reassessed.
Report out-of-scope defects separately without repairing them opportunistically.

## Return to main

Return changed paths and the worker contract's exact source and diff identities.
Explain the cause, resulting behavior, compatibility impact, and relevant decisions.
Include applicable build provenance, copied-source attribution, executed local commands, and unresolved assumptions.
State the weakest implementation point and any simpler alternative considered.
Give main each exception, fallback, nonpositive result, and unknown for explicit disclosure in the user conversation and durable documentation.
State the affected cases, trigger, and impact, with the evidence or the missing check.
List tests, documents, or source research needed from other workers.
Do not declare acceptance or global completion.
