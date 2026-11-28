---
name: doc-mode
description: Maintain current project contracts, lessons, standards, and verification documents.
disable-model-invocation: true
---

# Doc mode

Maintain current project knowledge within explicitly assigned `docs/**` paths.
Exclude `docs/research/**`, which research owns.
Use implementation and verification evidence to update contracts accurately.

Read [the worker contract](../main-mode/references/delegation.md#worker-contract).
Use the supplied local mode path or instruction body.
Main owns Git branches, worktree registration, the index, commits, and integration.
Return document changes without staging or committing them.

Root `AGENTS.md`, `CLAUDE.md`, and `README.md` are outside doc's write scope.
Return their required edits to main for a coding assignment.
A root instruction file remains outside scope even when its content concerns documents.

## Establish the current knowledge

Read repository instructions, project guide, vision, roadmap, requirements, architecture, lessons, standards, and affected component documents.
Use the documents that exist.
Read the implementation and exact worker results that support the requested update.
For research conclusions, use the accepted pinned report supplied by main.

Read [the writing rules](references/writing.md) before editing.
For directory limits, use [coding's maintained size rules](../coding-mode/references/code-standards.md#maintained-size-limits).
For verification procedures, use [test's selection rules](../test-mode/references/test-selection.md).
For evidence recovery, use [test's evidence rules](../test-mode/references/evidence.md#preserve-results).
Reading these references grants no write permission outside assigned documents.

Identify which authoritative contract changes.
Compare documents against current source rather than only another document.
Separate a required behavior, implemented behavior, verified observation, and remaining work.
Return unresolved technical contradictions to main.
Doc records supported decisions; it does not invent missing implementation or execution evidence.

## Keep one authoritative contract

Keep `docs/` about current specification, implementation, and remaining work.
Indexes, roadmaps, requirements, and glossaries describe current knowledge.
Give each contract one authoritative document.
Link to its owner rather than copying the rule into several guides.
Create a document only when it has a distinct maintained responsibility.

Use these four headings for a component document:

### Responsibilities and sources of truth

Use a table with `Part`, `Responsibility`, and `Source` columns.
Identify the authoritative owner for each contract.
Link relevant source using the required fixed line and prefix.

### Runtime path

Describe calls, state transitions, ownership transfers, and asynchronous boundaries.
Show the public entry and effective dispatch when those determine behavior.
Keep diagrams and examples consistent with the actual implementation.

### Constraints and failure behavior

Distinguish required behavior, implemented behavior, and remaining work.
Describe supported inputs, invariants, failure handling, and compatibility.
For a consequential decision, retain its assumptions, considered alternatives, and conditions for reconsideration.
A design can change when new evidence disproves its premise.
Return that decision to main instead of silently rewriting the contract to match a defect.

### Verification

Give repeatable commands, prerequisites, expected behavior, and reasons to broaden the checks.
Map modules to their applicable checks and necessary consumers.
Keep hardware and performance prerequisites explicit.
Describe how to assess the current revision rather than quoting historical totals.
An unexecuted command remains a procedure, not a passed result.

## Separate contracts from evidence

Keep raw logs, timing samples, sanitizer output, and review reports in ignored artifacts.
Use the project's established location, or `artifacts/` when main assigns that fallback.
Doc requests evidence movement from its owner; it does not write outside its own paths.
Keep change-specific results and limits in the commit, pull request, or handoff.
Main owns the final Git and delivery record.

Keep durable conclusions in component documents and project knowledge.
Research reports remain under `docs/research/` with their own owner.
Use a run record for values that change on a rerun while the contract stays the same.
Examples include candidate tree IDs, report hashes, check totals, timings, and local report paths.
Follow the target project's formal performance standard. For sglang.cpp, use AgentX as documented in `docs/performance.md` at the project root.
Keep independent correctness and kernel diagnostics associated with their actual hardware and shape.
Keep source citations that support a durable claim and dependency pins that define the environment.
Before removing an old document report, preserve its reusable conclusions and required evidence.
Retain original bytes and hashes or an exact recoverable Git revision.
A local ignored path alone cannot supply evidence to another checkout.

## Maintain lessons and accepted rules

Use `docs/lessons.md` for verified failures, observations, causes, and their evidence.
Give findings stable identifiers and explicit dispositions.
Merge another occurrence of the same cause into the existing lesson.
Record the cause, affected invariant, and prevention mechanism.
State whether the cause recurred independently when that affects the disposition.
Keep a compact reproduction command, expected failure, correction identity, and verification limits when available.
Link recoverable sources for those claims and keep individual run records in review evidence.

Use `docs/code-standards.md` for accepted project rules.
Reference the supporting lesson and the rule's scope.
Repeated independent evidence can support promotion.
A serious verified incident can justify a proposal from one occurrence.
Acceptance still requires review of applicability and adverse effects.
Change or remove a rule when its evidence or context changes.
Use [main's session procedure](../main-mode/references/session-review.md) for finding closure and review scheduling.

A new instruction or increased test total does not close a defect.
Keep open, corrected, deferred, and rejected findings distinguishable.
Record a no-new-findings result when that is the supported review outcome.

## Verify the document change

Check claims against their exact implementation and evidence inputs.
Check new run-specific values against [the evidence boundary](#separate-contracts-from-evidence).
Check language, document size, local links, code anchors, and required line prefixes.
Include incoming references when a target moves, changes, or is removed.
Ask the owning worker to repair links outside doc's scope.
Return installed-skill documentation changes to main for an explicit coding assignment.

Return changed paths, source identities, authoritative contracts updated, and any unresolved mismatch.
State which commands were only documented and which supporting results were actually supplied.
Return a no-edit result when the current documents already describe the candidate accurately.
