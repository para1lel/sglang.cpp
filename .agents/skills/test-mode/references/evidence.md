# Keep verification evidence fixed

Use this reference for execution reports, build claims, observers, and repository gates.
Evidence is meaningful only for identified inputs and an instrument that reaches the claimed behavior.

## Freeze expected evidence

Record the candidate before verification.
Include full source revision, applied diff, relevant untracked bytes, configuration, dependencies, and artifact identities.
Use [the worker contract](../../main-mode/references/delegation.md#worker-contract) for uncommitted candidates.
Identify the exact commands, working directory, interpreter, options, and relevant environment.
Keep secrets out of the record.

Freeze reference outputs, expected hashes, input distribution, assertions, tolerances, and workload meaning before evaluating the candidate.
Keep independent references separate from candidate mutations.
Validate copied inputs against recorded expected hashes.
Rehashing changed bytes and accepting the new hash does not validate earlier evidence.
Report an intentional expectation change as a contract change for review.

Build success requires fixed source, effective options, patches, dependencies, and produced artifact hashes.
Publish that association only after the build and required probes succeed.
Distinguish library identity from the consuming executable's identity.
Read pinned Git objects with replacement processing disabled.
Preserve replacement refs rather than removing them.

For a remote run, identify the transferred snapshot and remote effective configuration.
Verify the copied input bytes before accepting returned output.
Record actual hardware, tool versions, selected device, and resource-sharing conditions when they affect the claim.
A transport error, incomplete run, or absent artifact is not a passing result.

## Establish execution and observation

Use the production configuration for a production claim.
Check effective dispatch and numerical policies.
Establish the actual changed branch with a distinguishing observation.
A compiled feature, public wrapper call, or passing neighboring fixture alone is insufficient.
Confirm the loaded or imported implementation belongs to the intended snapshot.

Validate an observer with a positive control that reaches its actual boundary.
For compiled interception, exercise a separately compiled call or launch through that boundary.
Confirm that the observer reports work or rejects the intended control.
Zero counters from an incomplete observer do not establish absence of work.

State each reference's scope and independence.
An independent mathematical implementation checks numerical meaning.
A pinned application implementation can check compatible application behavior.
A simple verified runtime path can isolate dispatch, cache, replay, or state behavior.
Shared faulty code limits the claim each comparison can establish.

When a pinned reference cannot run in the candidate environment, use a separate compatible environment within the task's constraints.
Keep the reference source and required dependency pins fixed.
Verify that the loaded packages and binaries come from the pinned inputs.
Run a known control through the reference path needed for the claim, including device execution when required.
Report the reference as unavailable if no permitted environment can run that control.

When an observer is repaired, identify the claims affected by its earlier gap.
Retain independent valid numerical evidence.
Repeat the affected observation with fixed operation inputs.
Do not erase all earlier evidence or accept all earlier claims without that assessment.

## Examine the selected index

Mechanically testable repository requirements belong in incremental pre-commit gates.
Install or update them through an authorized coding assignment.
Preserve existing hook behavior.
Use a companion `commit-msg` gate for message syntax supplied after pre-commit.
Language compliance and semantic review remain separate obligations.

Validate a fixed snapshot of the selected index, including an alternate `GIT_INDEX_FILE`.
For the maintained target, have main identify and freeze that selection.
Use staged checker code and staged tool configuration when they belong to the candidate.
Read staged blobs with replacement processing disabled.
Keep the caller's index bytes, worktree, replacement refs, and configuration intact.
Gates inspect; they do not rewrite source or repair the index.

An unstaged repair must not conceal an invalid staged file.
An unstaged defect must not replace a valid staged input.
Include additions, deletions, renames, affected directory counts, incoming references, and necessary consumers.
Check document, code, function, and directory limits, code links, syntax, metadata, and configured static analysis.
Use [coding's configured size limits](../../coding-mode/references/code-standards.md#maintained-size-limits), including the target's frozen adoption values.
Use the language's parser or verifier for function boundaries.
Validate copied-skill package boundaries when installed resources change.

Broaden structural checks for shared rules, gate code, configuration, and dependency declarations.
Keep unrelated expensive tests outside a commit hook.
Return nonzero for a violated requirement, unavailable required tool, or failed child command.
Main must stop dependent staging, commits, publication, and acceptance claims after failure.
A failed command followed by a successful unrelated command is still a failed requirement.

For a gate change, run an actual failing staged case and a valid staged control.
Include staged and unstaged disagreement, alternate indexes, changed link targets, and child-command failure when those contracts change.
For gate and session verification, test can create [assigned disposable Git fixtures](../../main-mode/references/delegation.md#disposable-git-fixtures).
Independent temporary fixtures require no preparation by main.
Workers do not stage fixtures in the shared development index.
Prefer a small copyable check to a new orchestration runtime.
Keep semantic limits explicit; a size checker cannot certify English or runtime correctness.

## Bind review and integration

Record reviewed source, diff identity, requirements, configuration, findings, and verification limits.
Keep reviewer identity separate from author identity for independent reviews.
If integration changes the candidate, reassess affected conclusions against the resulting identity.
Conflicts, rebases, or a changed dependency can alter behavior even when a patch identifier matches.
Have main provide the final candidate for necessary repeat checks.

Review test and fixture changes for hidden deselection or altered outcomes.
When test configuration can conceal existing regressions, use trusted baseline tests and configuration in an isolated comparison.
Document legitimate contract amendments rather than silently weakening the acceptance conditions.
A previous verdict for another candidate is stale.

Keep findings open until a correction or agreed deferral has the necessary evidence.
A mechanically complete review record proves identity, not correctness.
An unresolved serious defect needs developer agreement before affected delivery.

## Preserve results

Keep raw output, samples, sanitizer records, and review reports in assigned ignored artifacts.
Record actual pass, failure, timeout, skipped, unavailable, and unexecuted states separately.
For selected test runs, zero collected cases or all-skipped cases provide no passing acceptance evidence.
A reasoned static result remains a separate result.

Keep original bytes and hashes for evidence referenced by findings or session records.
If evidence moves, retain an exact recovery location.
Another checkout needs an approved archive or recoverable Git revision, not only a local ignored path.
Send reusable conclusions, reproduction commands, and current verification procedures to doc through main.
Historical test counts and timings do not establish the current revision's behavior.
