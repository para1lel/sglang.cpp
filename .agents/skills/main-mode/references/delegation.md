# Delegate bounded engineering work

Main owns this coordination contract.
Workers read it before accepting an assignment.
Reading a linked role reference does not grant that role's write permissions.

## Dispatch inputs

Supply the following information in the worker prompt:

- The goal, relevant user instructions, acceptance conditions, and excluded work.
- The target repository and assigned mode's local `SKILL.md` path, or its full instruction body.
- Required reference paths or bodies when the worker cannot access them directly.
- Explicit writable paths, read-only inputs, exclusions, and separate output paths.
- The source snapshot, existing changes, dependency results, and expected return format.
- Available tools, environment, hardware, time constraints, and actual authorization.

Use the host's available worker or subagent interface.
No particular tool name, launcher, account, model, or sandbox is assumed.
The prompt supplies instructions explicitly; it does not ask the worker to discover explicit-only skills.
If filesystem access differs, include the required text and snapshot inputs.
Report missing capabilities without claiming enforcement that the host does not provide.

Use minimal raw inputs for independent review.
Keep an author's hypothesis separate from the requirements and observations.
A review worker must assess the approach and relevant alternatives without a forced answer.

## Establish disjoint ownership

Give every writable path one active owner.
List exact files or bounded subtrees with explicit exclusions.
Read access can overlap.
Separate reports, temporary output, build directories, and caches when concurrent work could overwrite them.
Assign one writer to any shared progress record.

Default role boundaries are:

| Role | Possible writes | Exclusions |
| --- | --- | --- |
| Research | Assigned `docs/research/**`. | All other project files. |
| Coding | Assigned production, build, dependency, instruction, and root files. | Research, durable `docs/**`, and test-owned files. |
| Test | Assigned tests, fixtures, benchmarks, check resources, and ignored verification artifacts. | Production repairs, durable documents, and research. |
| Doc | Assigned `docs/**`. | `docs/research/**`, root documents, instructions outside `docs/`, and implementation files. |

These boundaries limit an assignment; they do not grant every listed path automatically.
For mixed files, assign the entire file to one worker at a time.
A build file with test configuration still needs one owner.
Have the other worker return a proposed change or review.
A worker reports an out-of-scope finding to main instead of repairing it.

Only main owns the maintained target's branch selection, Git worktree registration, indexes, commits, and integration.
Workers can inspect target Git through read-only commands.
They do not stage, commit, switch branches, merge, rebase, reset, or alter Git configuration in that repository.
Main registers any worktree linked to the target repository.
Workers return target changes without committing.
Preserve pre-existing user changes.

If overlapping work becomes necessary, stop writes to that path.
Return the current diff and identity.
Main records an ownership transfer and supplies the new owner's exact inputs.
Resume only after both owners know the boundary.

## Disposable Git fixtures

Test can create and manipulate independent repositories within explicitly assigned disposable fixture paths.
This includes fixture initialization, configuration, branches, indexes, commits, and history changes for gate or session verification.
Test can create an ordinary `TemporaryDirectory` fixture without main preparing its contents.
Fixture repositories must have independent Git metadata, even when seeded from a target snapshot.
Use fixture-local Git paths and environment so inherited target variables cannot redirect writes.
Keep the target's Git metadata, indexes, branches, registered worktrees, and session records unchanged.
Only main registers target-linked worktrees and performs integration Git.
Fixture commits remain verification artifacts outside target history and target session counts.

## Worker contract

Accept an assignment only when its purpose, writable paths, and input identity are explicit.
Identify missing information that blocks safe progress.
Continue independent work within the assignment when possible.
Use the supplied repository instructions and relevant role-owned references.

For each result, return:

- Completed acceptance conditions and any remaining obligation.
- Changed, added, renamed, and deleted paths within the assignment.
- The exact source identity, change identity, and relevant dependency identities.
- Observations, executed commands, static reasoning, and actual outcomes.
- Evidence paths and hashes when artifacts exist.
- Limitations, unavailable tools, unexecuted checks, and blocking findings.
- The weakest supported assumption and a simpler alternative when relevant.

Bind each conclusion to the inputs it examined.
For a clean snapshot, report the full commit identifier and relevant configuration.
For uncommitted work, report full `HEAD`, a digest of the binary diff, and hashes of relevant untracked file bytes.
Include renames, deletion states, modes, submodule revisions, and other material inputs.
Record pre-existing changes separately from the worker's change.
Main supplies or obtains the corresponding complete candidate identity before integration.

Research also pins each external source's revision or version and access date.
Test also identifies the reviewed candidate, test configuration, observer, and expected evidence.
Doc also identifies the implementation and verification records used to update current contracts.
Coding also identifies copied material, effective build inputs, and unresolved implementation assumptions.

Store raw verification output in the assigned ignored evidence location.
Return reports through the host when a role has no permitted artifact path.
Research and doc retain their directory boundaries even for auxiliary output.
No worker claims another worker's checks as its own execution.
No worker declares global completion or publishes changes.

## Accept or return

Main checks ownership, identities, dependencies, and whether all required outcomes have evidence.
Send technical findings back to the relevant worker.
A source or configuration change makes affected conclusions stale.
Ask test to determine the necessary repeat checks against the new candidate.
A corrected document or a longer test list alone does not close a behavioral defect.

For review findings, require a concrete trigger, consequence, supporting location, and proposed correction.
Keep introduced defects, pre-existing defects, optional improvements, and verification gaps distinct.
A serious correctness defect remains blocking regardless of when a reviewer noticed it.
On repeat review, check prior findings and the changed diff.
Expand only for affected dependencies, new evidence, or a demonstrated correctness defect.
