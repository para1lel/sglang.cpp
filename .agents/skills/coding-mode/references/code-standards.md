# Implementation standards

Coding owns these implementation rules.
Project-specific contracts can add relevant constraints.
Read affected consumers before changing their interface or representation.

## Keep modules coherent

Give each responsibility one authoritative implementation.
Prefer extending the existing owner to creating a parallel version.
Keep interfaces small enough to state their inputs, effects, errors, and ownership.
Add an abstraction when it removes meaningful caller complexity or repeated invariant handling.
Avoid layers with no caller or operational purpose.

Fix the cause at the narrowest owner that protects related paths.
Preserve established semantics outside the requested change.
Use explicit types, units, interval endpoints, and valid-state rules where ambiguity can change behavior.
Keep mutable state transitions coherent across success, error, cancellation, and retry.
Choose names and structure that expose the invariant without explanatory repetition.

Separate correctness eligibility from selection preferences.
A cost policy chooses among legal implementations.
It does not make an otherwise supported input illegal.
Keep a correct fallback when the optimized path cannot meet its contract.
Reject unsupported inputs before unsafe arithmetic, allocation, or submission.
Check direct internal entries when public validation can be bypassed.

## Preserve semantics

Define conversion, arithmetic, accumulation, result storage, and nonfinite behavior for numerical operations.
Use the operation's requirements to justify precision and tolerances.
Keep reference and candidate precision policies compatible with the claim.
A wider accumulator does not undo a required input conversion.
Find the first divergent value or state before changing a numerical policy.

Derive indices, spans, layouts, and allocation sizes from authoritative representations.
Check overflow before products or narrowing conversions.
Treat aliases as actual overlapping storage spans.
Different argument names do not establish separate memory.
Validate empty work, tails, strides, alignment, and maximum supported geometry where relevant.

Preserve order when a schema, protocol, parser, ABI, or consumer makes it observable.
Carry termination causes, errors, and partially completed state through each boundary.
Keep failure behavior consistent between streaming, blocking, eager, and replay paths when the contract requires it.
Use supplied domain guidance for model mathematics, compiler legality, or device synchronization when relevant.

## Maintain layout and dependencies

Put files in the repository's established layout.
Avoid duplicate implementation files named as temporary, backup, or alternate versions.
Keep generated disposable output outside the maintained tree.
Remove obsolete code and unused imports within the assigned change.
Keep debugging scaffolds out of delivered library code.
Give deferred work a tracked issue or other project-approved owner.

Declare new dependencies and pin them according to project policy.
Record necessary license and source attribution for copied material.
Retain actual dependencies in build and link rules.
Make optional dependency failure explicit.
Avoid silently using a different installed package or stale compiled artifact.

For source and artifact identity, read [execution identity](execution-identity.md).
For native ownership, asynchronous reads, and partial cleanup, read [resource lifetimes](resource-lifetimes.md).

## Maintained size limits

Read the target repository's `.agents/gates.json` when it exists.
Its frozen `limits` values govern directory entries, code-file lines, and function lines.
The configuration is a target input; the five-mode package requires no installation-mode directory.

| Item | Configured limit | Fallback when configuration is absent | Counting rule |
| --- | --- | --- | --- |
| Code file | `limits.code_file_lines` | 1,000 lines | Count physical lines, including blanks and comments. |
| Function | `limits.function_lines` | 100 lines | Count from the first decorator or declaration through its end. |
| Directory | `limits.directory_entries` | 30 immediate entries | Count files and subdirectories, including dotfiles. |
| Code line | Fixed rule | 100 characters | Include indentation; exclude the line ending. |

Use the three fallback values only when the configuration is absent.
An invalid or incomplete existing configuration is a verification failure.
Preserve the recorded adoption source, digest, baseline maxima, and limits.
Do not refresh the baseline or raise limits to accept repository growth.
When a frozen limit would be exceeded, split at a meaningful ownership boundary.
An intentional policy migration requires separate review; ordinary development cannot rebaseline the repository.

Apply code limits to maintained source, templates, and generated code checked into the project.
Use the language's parser or verifier for function boundaries.
For directory counts, count each immediate subdirectory once.
Exclude Git metadata and ignored disposable caches, builds, and artifacts.
Tracked content still counts when an ignore rule matches it.
Include nonignored untracked files in a working-tree count.
Use the selected index tree for a commit gate.

Split content at a meaningful ownership boundary before exceeding a limit.
Keep each rule with its reason.
Compiler transformations need not restrict disposable generated output to maintained-source limits.
Document size and language rules belong to [doc's writing reference](../../doc-mode/references/writing.md).
Mechanically testable requirements belong in [incremental gates](../../test-mode/references/evidence.md#examine-the-selected-index).

## Edit instruction files

When assigned a skill edit, keep its purpose and essential workflow in `SKILL.md`.
Put substantial conditional detail in local references with clear reading conditions.
Keep each rule with its owning role rather than creating another universal rule guide.
For explicit-only modes, preserve `disable-model-invocation: true` and `policy.allow_implicit_invocation: false`.
Resolve resource links within the deployed package and validate its required copied set.
Supply worker instructions by local path or body instead of requiring a global skill or proprietary invocation tool.

## Comments and self-review

Use code and names to show what happens.
Add a comment for a non-obvious reason, invariant, or external constraint.
Keep historical change descriptions in Git or the handoff.
Check existing comments and docstrings when their function changes.
Remove references to deleted or renamed symbols.
Ask doc to repair durable documents and incoming links outside coding's ownership.

Before returning, examine correctness, simplicity, public reachability, compatibility, and failure behavior.
Check that each new implementation has a supported path from a real consumer.
A passing nearby fixture does not establish production reachability.
Return unsupported assumptions as limitations.
Writing a rule or increasing test count cannot substitute for correcting the implementation.
