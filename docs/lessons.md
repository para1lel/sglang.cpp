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

## L004 · Finite-input NaNs in page-parallel GQA

**Correction verified on the isolated W2 candidate.** At 1,536 visible tokens, finite inputs overflowed FP32 QK scores or page-local value sums. Both cases returned NaNs in eager and graph modes while ordered replay and the FP64 oracle returned finite outputs.

The [merge guard](batch-graphs.md#page-parallel-gqa-decode) sends an affected head and channel tile through ordered replay. The correction matched the ordered and FP64 outputs on both cases. Keep both cases in the GQA regression and run them with eager execution and graph replay.

## L005 · HTTP request parsing on the accept thread

**Corrected.** The cache-enabled listener accepted and parsed requests on one thread. A stalled body held that thread until the read deadline and delayed unrelated controls and completions.

The initial correction used a reader pool. At width one, a CPU loopback regression paused one reader after parsing a complete request. Separate health and completion requests returned HTTP 200. Shutdown joined a held reader and discarded its request. The loopback suite also passed checks for queue overflow, busy reset, cancellation, and cache accounting.

The [current HTTP reader](http-server.md#http-admission-and-shutdown) advances multiple incomplete requests from one poll loop.

## L008 · Buffered tokens suppress SSE keepalives

The original batch consumer deferred keepalives when a decoder step was pending and reset the timer when it handled that step. Stop matching or chat parsing can accept a token without writing an SSE frame. One B200 probe delivered six decode steps about three seconds apart. A partial stop string held all six tokens until completion. The first data frame arrived 18.349 seconds after the HTTP 200 headers, with zero comments. The no-step control saw a comment near 15 seconds on benchmark, text, and chat streams.

The [corrected writer](http-server.md#completion-streams) measures 15 seconds from the last successful header, data, or comment write. A step that only buffers tokens leaves the deadline unchanged. The coding rerun sent a comment after 15.006 seconds, then delivered all six exclamation marks, finish, usage, and `[DONE]`. The no-step controls and first-token timing checks also passed. Independent acceptance is in progress.

Run the scripted CPU regression from the repository root in the managed environment. Set `out` to an existing output directory and `checkpoint` to the local Qwen3.8-27B-FP8 checkpoint. Use a fresh `$out/keepalive` directory. [Scripted regression](../tests/text/check_stream_keepalive_cpu.py#L58)<!-- prefix: "def main():" -->

```bash
bash tests/text/build_chat_cpu.sh "$PWD" - "$out/chat-cpu"
python3 -IB tests/text/check_stream_keepalive_cpu.py \
    --binary "$out/chat-cpu" --checkpoint "$checkpoint" --evidence "$out/keepalive"
```

The scripted clock advances by three seconds per decode step. Both text and chat must send two comments, including one after the deadline restarts on visible data. The test also checks role, content, finish, usage, and `[DONE]`. The matched build and wire evidence are in `artifacts/resume-stream-fix/handoff.md` in the scheduler worktree.
