# Development lessons

Record verified failures from [session review](../.agents/skills/main-mode/references/session-review.md#dispatch-the-batch-review) here. Merge recurrences under the same identifier. Keep exact inputs, commands, and results in the review record.

## L001 · Header placement

**Corrected.** The prefix change put `decoder_prefix_cache.h` in `src/`. Eight other headers were already outside `include/sglang/`. Local includes let the build succeed despite the [documented layout](development.md#scope-at-adoption).

The relocation moved all nine headers under `include/sglang/`. The first gate let an unchanged misplaced header pass when only Markdown was staged.

The [commit gate](../.agents/checks/staged_check.py#L173)<!-- prefix: "def check_he" --> now checks every entry in the selected index. It rejects project headers outside `include/sglang/`. The [retained-header regression](../tests/test_gate_contract.py#L138)<!-- prefix: "def test_hea" --> passed.

## L002 · Remapped KV pages

**Corrected.** Prefix store and restore assumed that logical and physical page indices matched. The identity case passed. With logical page zero at physical page one, the remapped case reported `Key payload mismatch`.

The [page-map reader](../src/decoder_prefix_cache.cpp#L17)<!-- prefix: "PageMapping " --> now supplies physical indices for both transfers. Store gathers into logical order. Restore scatters into the destination mapping. The selected correction passed the remap and guard checks.

Run `prefix-state --remapped` and `prefix-state --guards` after changes to KV copies. Use the [B200 build and run procedure](../tests/prefix/README.md). Keep independent source and destination permutations and check all four KV planes and untouched tail pages.

## L003 · Patch and anchor provenance

**Corrected.** The first prefix candidate left 53 stale or blank code anchors. The prefix docs also needed rebasing after finite EBNF changed their source preimage. Earlier checks covered a different file layout.

Record the exact base, ordered patch hashes, and resulting tree in the handoff. Replay patches on that base and compare the result with the selected tree. Check incoming code links again after overlapping source edits. The selected prefix tree passed the staged maintenance checks after anchor repair.
