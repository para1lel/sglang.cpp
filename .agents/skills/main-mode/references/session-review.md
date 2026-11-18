# Review a development session

Main owns the fixed session record and review schedule.
Workers perform the technical review and document changes.
`REPO` names the source repository selected for this session.
Project document and evidence paths are relative to `REPO`.
They are task inputs and outputs, separate from installed skill resources.

## Review a failure immediately

Start an immediate review after a developer correction, reproduced defect, failed gate, or recurring cause.
Pause further affected work until the failure and its scope are understood.
Keep the scheduled commit base and interval unchanged.
An immediate review does not complete an unexamined commit batch.

Assign test to preserve the failing input, original source identity, expected result, and actual result.
Do this before correction removes the failure.
Assign the relevant worker to identify the first incorrect assumption or missing contract.
Examine related callers, control flow, representations, and backends for the same cause.
Use the existing lesson when the finding repeats that cause.
Repeated causes require correction to the owning invariant or prevention mechanism.
Another narrow instruction is insufficient when the invariant remains unprotected.

Have coding correct the implementation.
Have test preserve an exposing regression and a valid control when they add distinct coverage.
Use an instrument that can expose the failure.
Numerical success cannot close a race finding without relevant ordering evidence.
Use [test selection](../../test-mode/references/test-selection.md) to choose the affected module and necessary consumers.
Have doc update durable lessons and changed contracts.

## Record the session base

Use the bundled [session helper](../scripts/session_review.py#L1)<!-- prefix: "import argpa" --> before development.
It requires Python 3.11 or later and Git.
It uses the Python standard library.
Use an existing interpreter environment.
The examples use Linux Bash from the installed `main-mode` directory.
Replace `PYTHON_ENV` with the absolute path to a suitable virtual environment.
For a read-only installation, keep that environment in a writable location.

```bash
"PYTHON_ENV/bin/python" "scripts/session_review.py" start --repo "REPO"
```

Keep the returned `state_path` in the session handoff.
The helper stores one state file per session under local Git metadata.
Use a separate file for each concurrent session.
Main is its only writer.
The helper does not change project files, create commits, or perform remote operations.

The initial `base` is the commit before development starts.
If that base was recorded elsewhere, supply `--base COMMIT`.
Change the default interval of 50 only when the developer requests it.
Use `--interval N` for that interval.
Resume an existing session with the same `--state PATH`.
An existing record preserves its base and interval.
Relative state and evidence paths resolve against the repository selected by `--repo`.

For an unborn repository, the helper records a null base.
Its count is zero before the first commit.
Afterward, it uses `git rev-list --count HEAD`.

## Examine status

Use the saved state path:

```bash
"PYTHON_ENV/bin/python" "scripts/session_review.py" status \
    --repo "REPO" --state "STATE"
```

For an existing base, count with:

```text
git rev-list --count BASE..HEAD
```

Follow all parent links.
Merged branches contribute each commit included by this command once.
Branch names and first-parent position do not change the rule.
Keep the session base as an ancestor of `HEAD`.

The result includes total count, due batches, next batch, and batch digest.
The next batch contains the oldest unreviewed commits in Git's reverse topological order.
Each complete batch contains `n` commits.
Reviewed commit identities remain recorded when later merges reorder other commits.

Examine status after authorized commits and before handoff.
If several batches are due, review one complete batch at a time.
Examine status again after each completion.
Keep the final partial batch for the next boundary.
At handoff, record partial progress and due reviews.

If a rebase or reset changes the base or reviewed history, resolve the mapping first.
Keep the original record.
Get developer approval for a recovered base or replacement session.
Do not reset the count to clear a due review.

## Dispatch the batch review

Supply test with the exact selected commit list, diffs, requirements, and earlier findings.
Use research for unresolved source or design questions.
Use coding for implementation analysis and corrections.
Use doc for lesson, standard, roadmap, and contract changes.
Keep their writable paths disjoint through [the delegation contract](delegation.md).

Examine available review comments, failures, regressions, and checks.
Read earlier open findings and relevant lessons before searching for new findings.
Assess whether earlier corrections still protect related paths changed by this batch.
Look for repeated corrections, weak verification, and rules that caused unnecessary work.
Classify an existing rule as absent, unclear, incorrect, or not obeyed when evidence supports that distinction.
Also consider architecture, types, tools, and automated checks as correction owners.

Have doc follow [the lessons procedure](../../doc-mode/SKILL.md#maintain-lessons-and-accepted-rules) for durable findings.
Use the dispositions open, corrected, deferred, or rejected in the review report.
Record each occurrence's date, source identity, and check results in that report.
For a correction, identify changed source and evidence of corrected behavior.
For a deferral, identify remaining risk, responsible developer, and agreed next step.
For rejection, preserve the evidence that disproves the finding.
Distinguish a defect in the fixed batch from a later correction.
A written rule or increased test count does not establish correction.

Separate independent occurrences from repeated descriptions of one incident.

Repeated evidence can support a proposed project standard.
One serious verified incident can also justify a proposal.
Review applicability and adverse effects before acceptance.
Have doc put accepted standards in `REPO/docs/code-standards.md`, with their supporting lessons.
Change or remove standards when their evidence or scope changes.
Update roadmap and design documents when findings change a persistent obligation or design.
Reusable library proposals belong to a separately authorized meta workflow.

A review can have no new findings.
Record that result when the evidence supports it.
Do not invent a lesson to fill the report.

## Preserve evidence and complete the record

Assign test a unique report path in the project's ignored evidence directory.
Use `REPO/artifacts/reviews/` when no location exists.
Include reviewed commits, actual check results, resulting document changes, finding dispositions, and unresolved delivery restrictions.
Keep reusable conclusions in lessons, standards, and current component documents.
Require this marker with the digest returned by `status`:

```text
<!-- review-batch: DIGEST -->
```

After technical review and document updates, main records completion:

```bash
"PYTHON_ENV/bin/python" "scripts/session_review.py" complete \
    --repo "REPO" --state "STATE" --head "HEAD" --batch "DIGEST" --evidence "REPORT"
```

Replace uppercase values with actual identifiers and paths.
Use the full head identifier returned by `status`.
The helper checks the expected head, batch, report marker, and presence of report text.
It records the report hash.
These checks establish record identity, not review quality or correction of every finding.
Keep serious reproduced defects open until correction or agreed deferral has sufficient evidence.
Affected delivery requires developer agreement while a serious defect remains unresolved.

Keep original report bytes available while the session record refers to them.
If evidence moves, preserve its hash and exact recovery location.
For another checkout, use an approved archive or exact recoverable Git revision.
Have doc retain a compact reproduction command and expected failure in the durable lesson.
A local ignored path or ephemeral worker output alone is insufficient evidence for another checkout.

If `HEAD` changes before completion, examine the new state first.
A repeat completion for an already recorded digest does not advance progress again.

## Retain the adaptation boundary

The earlier general procedure borrowed lesson review from Magpie.
Its source revision was `ef176e21f6c40e5a623709bb0e369647173964de` in `yetone/magpie`.
The source records were `LESSONS.md` and `AGENTS.md`.
That design used nightly reviews and observations on three separate review days.
This procedure uses the developer's fixed session base and commit interval.
It does not impose a universal three-day promotion threshold.
It also does not restore the retired runtime's row-based schedule.
