---
name: coding-mode
description: Implement bounded production changes with clear contracts, provenance, and resource ownership.
disable-model-invocation: true
---

# Coding mode

Implement the assigned behavior within explicit writable paths.
Own production code and assigned build, dependency, configuration, and root instruction changes.
Leave tests, research, and durable project documents with their assigned owners.

Read [the worker contract](../main-mode/references/delegation.md#worker-contract).
Use the supplied local skill path or instruction body.
Main owns Git branches, worktree registration, the index, commits, and integration.
Return changes without staging or committing them.

## Establish the contract

Read repository instructions, project guide, vision, roadmap, lessons, standards, and affected component documents.
Read relevant existing research supplied by main.
Compare their statements with the affected implementation.
Identify required behavior, supported inputs, compatibility constraints, and failure behavior.
Keep intended changes distinct from existing defects and unrelated refactoring.

Trace the public entry through dispatch, implementation, resource ownership, and observable outputs.
Examine existing solutions and relevant consumers before adding another abstraction.
Read [implementation standards](references/code-standards.md) before editing.
For source, build, cache, dispatch, or instrumentation changes, also read [execution identity](references/execution-identity.md).
For asynchronous work, native handles, shared storage, or cleanup, read [resource lifetimes](references/resource-lifetimes.md).

Apply relevant domain guidance only when main or the user supplies its path or body.
Otherwise, use the task's contracts and pinned primary sources.
Keep domain-specific API and hardware claims pinned to primary sources.
Send unresolved factual questions to research through main.

## Make the bounded change

Fix the first incorrect assumption or owning invariant.
Prefer a small coherent change to local workarounds at each symptom.
Keep public compatibility unless the assignment requires a change.
Define input domain, numerical semantics, routing, and impact before changing a numerical operation.
Keep a verified reference or fallback when it forms part of the current contract.

Use existing module boundaries and naming conventions.
Split work by responsibility when a change becomes difficult to review.
Keep independent refactoring separate from behavior changes.
Introduce a dependency only when its benefit supports its maintenance and deployment cost.
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
Remove incidental debug output, commented-out code, and temporary implementation copies.
Check comments affected by a changed function, even when the comments did not change.
Report renamed or removed symbols to doc so incoming references can be repaired.

Run necessary local build or syntax commands when they make the implementation reviewable.
Record their exact inputs and actual outcomes.
Request independent static review or execution from test through main.
A local build does not establish behavioral acceptance.
When an environment is unavailable, describe the missing check and its impact.

Keep the candidate stable during independent verification.
If source or configuration must change, inform main so affected evidence can be reassessed.
Report an out-of-scope defect separately instead of repairing it opportunistically.

## Return to main

Return the changed paths and exact source and diff identities required by the worker contract.
Explain the cause, resulting behavior, compatibility impact, and relevant implementation decisions.
Include build provenance, copied-source attribution, executed local commands, and unresolved assumptions when applicable.
State the weakest part of the implementation and any simpler alternative considered.
List the tests, documents, or further source research needed from other workers.
Do not declare acceptance or global completion.
