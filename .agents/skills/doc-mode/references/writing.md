# Write maintained engineering text

Doc owns these language, document-size, and source-link rules.
Research and coding can read them without acquiring doc's write scope.
Apply them to new documents, comments, messages, commit text, issues, and pull request text.

## Use clear English

Use ASD-STE100 Issue 9 as the language reference.
Use approved words with their approved meanings.
Use permitted technical names consistently with the project glossary.
If the project needs a new technical name, define it at its owning contract.
Have doc update the glossary when main assigns that change.

Give each instruction one action.
Keep procedure sentences within 20 words.
Keep descriptive sentences within 25 words.
Use active voice where the standard requires it.
Place a condition before the action that depends on it.
Use concrete names and verbs.
Keep a rule, its scope, and its reason together.

Review dictionary meanings and grammar against the official standard when claiming language compliance.
A sentence-length check cannot certify compliance.
Report a limited language review as limited.
Keep mechanical results separate from semantic and language judgments.

## Keep documents small

Limit each document to 3,000 visible words.
Count non-whitespace groups in rendered text.
Include headings, table cells, list items, link labels, and visible code blocks.
Exclude hidden comments, link destinations, and undisplayed metadata.
Split at a meaningful question, component, or ownership boundary before exceeding the limit.

Keep the entry document useful for the action the reader must take.
Move substantial conditional details to linked references.
A pointer states what the reference owns and when to read it.
Keep one authoritative location for each rule.
Repeated summaries must not become competing specifications.

Use tables for parallel responsibilities or comparisons.
Use a diagram when relationships or state changes become clearer.
Keep an example only when it explains a non-obvious requirement.
Avoid execution histories and copied tool manuals in current contract documents.

## Make code links verifiable

Every link to project code needs a fixed `#L<number>` anchor.
For a historical version, also fix the source commit.
Select a nonblank target line.
Immediately append a hidden prefix comment to the link:

```text
<!-- prefix: "the first 12" -->
```

The comment contains a JSON string.
Its value is the first 12 characters of the target line after removing leading whitespace.
For a shorter line, use all remaining characters.
For a range, use the first line.
Escape angle brackets in the JSON string as `\u003c` and `\u003e`.
Escape other characters as JSON requires.

Examine the complete destination, line number, and prefix against the intended source revision.
Check incoming references when a target moves, changes, or is removed.
Update the explanation if the linked code's meaning changes.
A matching prefix alone does not establish that the statement remains correct.

For local resource links in an installed skill, resolve paths relative to the containing file.
Keep required targets within the deployed skill package.
Target repository documents and source paths are explicit task inputs, not installed resources.
Do not make an installed mode depend on the library author's repository root.
For external technical claims, use the pinned primary sources supplied by research.

## State supported claims

Separate a requirement from an implementation fact.
Separate an observation from an inference.
Identify the evidence and assumptions behind an inference.
Record unverified behavior as remaining work or a limitation.
Avoid turning one workload result into a general capability claim.

Give a procedure its prerequisites, command, expected behavior, and failure interpretation.
Use Linux Bash for command examples in this library.
Use explicit virtual-environment interpreter paths when an environment is required.
Quote paths with spaces.
Use a backslash for Bash line continuation and keep it last on the line.
Keep shared helpers independent of shell syntax.

Keep durable component structure in [doc mode](../SKILL.md#keep-one-authoritative-contract).
Keep run records and recoverable evidence under [test's evidence rules](../../test-mode/references/evidence.md#preserve-results).
Keep implementation-size rules under [coding's standards](../../coding-mode/references/code-standards.md#maintained-size-limits).
