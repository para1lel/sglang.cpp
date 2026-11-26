# Development lessons

Record verified failures from [session review](../.agents/skills/main-mode/references/session-review.md#dispatch-the-batch-review) here. Merge recurrences under the same identifier. Keep exact inputs, commands, and results in the review record.

## L001 · Header placement

**Corrected.** The prefix change put `decoder_prefix_cache.h` in `src/`. Eight other headers were already outside `include/sglang/`. Local includes let the build succeed despite the [documented layout](development.md#scope-at-adoption).

The relocation moved all nine headers under `include/sglang/`. The first gate let an unchanged misplaced header pass when only Markdown was staged.

The [commit gate](../.agents/checks/staged_check.py#L173)<!-- prefix: "def check_he" --> now checks every entry in the selected index. It rejects project headers outside `include/sglang/`. The [retained-header regression](../tests/test_gate_contract.py#L140)<!-- prefix: "def test_hea" --> passed.

## L002 · Remapped KV pages

**Corrected.** Prefix store and restore assumed that logical and physical page indices matched. The identity case passed. With logical page zero at physical page one, the remapped case reported `Key payload mismatch`.

The former resident cache fixed both transfers with one physical page-map reader. Store gathered into logical order. Restore scattered into the destination mapping. The fix passed the remap and guard checks. [Historical page-map reader](https://github.com/para1lel/sglang.cpp/blob/a7583f7b6346b5eb91aba2cba4748a6461e60fb4/src/decoder_prefix_cache.cpp#L151)<!-- prefix: "GpuPrefixSna" -->

Keep independent source and destination permutations in KV-copy tests. Compare all four KV planes and untouched tail pages. [Former B200 remap and guard procedure](https://github.com/para1lel/sglang.cpp/blob/a7583f7b6346b5eb91aba2cba4748a6461e60fb4/tests/prefix/README.md#cuda-state-and-routes) · [Current LMCache verification](prefix-cache.md#verification)

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

## L006 · FP16 operand loss in split GQA

**Correction verified in Candidate B.** Candidate A divided Q by its largest channel and narrowed to FP16. With Q channels `1` and `2^-40`, and K nonzero only in the second channel at `±448 × 2^33`, the scaled scores are `±0.21875`. The split returned zero instead of the FP64 result `tanh(0.21875) = 0.215326`. Ordered BF16 returned `0.214844`. The well-conditioned `2^-16` control used K at `±448 × 2^9` and returned `0.215820` in the split. PV had a second loss when narrowing probability times relative V scale to FP16.

The repair preserves Q in BF16, expands unscaled E4M3 K and V exactly to BF16, and rounds the weighted V operands to BF16. It uses FP32 accumulation and sends a flagged head to FP64 replay when range checks fire. The corrected split returns `0.215820` for both Q controls. Run from the repository root in the managed environment on a reserved B200. [Query load](../src/text/gqa_decode_split.cu#L174)<!-- prefix: "__device__ v" --> · [Regression cases](../tests/decoder/gqa_split_fp64_check.cpp#L181)<!-- prefix: "int main() t" -->

```bash
out="$PWD/build/long-decode"
xmake f -P "$PWD" -o "$out" -p linux -a x86_64 -m release --long-decode-check=y
xmake build -P "$PWD" gqa-split-fp64-check
CUDA_VISIBLE_DEVICES="${GPU_UUID:?Set the reserved B200 UUID}" \
    "$out/bin/gqa-split-fp64-check"
```

The test compares well-conditioned, query-range, and weighted-value cases with FP64. The `packed_fp64.cpp` probe under `artifacts/coding-fix-a/` exercises the same cases through the production fused step, with three live lanes, one padded lane, KV append, and a gate of 0.5. Candidate A returned zero in eager and replay for the two range cases with gated FP64 references `0.107663` and `0.000162406`. Candidate B passed all three in both modes. Use the probe's `packed-fp64-{fixed,frozen}-build.log` and corresponding run logs. See the [packed drift evidence](long-paged-decode.md#packed-route-drift-evidence) for drift that remained after the operand repair.

## L007 · Singleton route parity

**Correction verified in Candidate B.** Candidate A changed tensor-core singleton graph attention to the fused split and kept legacy attention in eager. On identical token histories, all 78 eager rows matched HEAD bitwise and 274 replay rows differed. A link-only control restored legacy attention and recovered all 44 sampled rows. Candidate B keeps legacy singleton attention in eager, capture, replay, and eager continuation after an eligible launch rejection. The checkpoint rerun recovered all 352 decode logit rows bitwise. [Bound attention](../include/sglang/runtime/execution/decoder_forward_layers.h#L694)<!-- prefix: "inline void " -->

Build using the [batch graph runbook](../tests/batch/README.md). Set `out` to the Release build directory, `checkpoint` to the local checkpoint, and `GPU_UUID` to a reserved B200. Use a fresh `$out/singleton` directory.

```bash
GPU_UUID="${GPU_UUID:?Set the reserved B200 UUID}" \
    bash tests/batch/run_batch_graph.sh "$out" "$checkpoint" "$out/singleton" singleton
python3 -IB tests/batch/check_graph_evidence.py "$out/singleton" --singleton-only
```

Check slot 31, replacement, and injected graph launch failure. The initial Candidate B verifier saw 13 of 17 BF16 rows per arm because it missed four packed-pair rows. The fixed observer recorded packed winner rows and checked subsequent width-two replay. The final verifier matched all 17 complete BF16 rows across eager, graph, and launch-rejection arms. The checkpoint route rerun separately observed zero fused attention launches in each singleton phase. [Historical singleton assertions](https://github.com/para1lel/sglang.cpp/blob/93648269d852e0fee78f7af4d1c3ffe4897aa361/tests/batch/check_graph_evidence.py#L406)<!-- prefix: "def verify_s" -->

The source records are `artifacts/test-accept-a/report.md` for Candidate A, `artifacts/coding-fix-a/report.md` for the Candidate B rebuild and initial verifier, and `artifacts/resume-verifier/evidence/final-check.log` for the final verifier. Candidate B is based on `aa4ec04b63c381cf11fdb4a3801fb925bacb5135`. Its tracked patch SHA-256 is `ba9502803ff42da44100f1e71e4ba37485ca1b7b79b2332f6775dc646beb986d`, and its untracked archive SHA-256 is `565193d94d7a562257733ad810bbab1f60aec7e44513fe8e345cfcfcf6b5e7aa`.

## L008 · Buffered tokens suppress SSE keepalives

The original batch consumer deferred keepalives when a decoder step was pending and reset the timer when it handled that step. Stop matching or chat parsing can accept a token without writing an SSE frame. One B200 probe delivered six decode steps about three seconds apart. A partial stop string held all six tokens until completion. The first data frame arrived 18.349 seconds after the HTTP 200 headers, with zero comments. The no-step control saw a comment near 15 seconds on benchmark, text, and chat streams.

The [corrected writer](http-server.md#completion-streams) measures 15 seconds from the last successful header, data, or comment write. A step that only buffers tokens leaves the deadline unchanged. The coding rerun sent a comment after 15.006 seconds, then delivered all six exclamation marks, finish, usage, and `[DONE]`. The no-step controls and first-token timing checks also passed on the standalone fix.

Run the scripted CPU regression from the repository root in the managed environment. Set `out` to an existing output directory and `checkpoint` to the local Qwen3.8-27B-FP8 checkpoint. Use a fresh `$out/keepalive` directory. [Scripted regression](../tests/text/check_stream_keepalive_cpu.py#L58)<!-- prefix: "def main():" -->

```bash
bash tests/text/build_chat_cpu.sh "$PWD" - "$out/chat-cpu"
python3 -IB tests/text/check_stream_keepalive_cpu.py \
    --binary "$out/chat-cpu" --checkpoint "$checkpoint" --evidence "$out/keepalive"
```

The scripted clock advances by three seconds per decode step. Both text and chat must send two comments, including one after the deadline restarts on visible data. The test also checks role, content, finish, usage, and `[DONE]`. The matched build and wire evidence are in `artifacts/resume-stream-fix/handoff.md` in the scheduler worktree.

## L009 · Terminal error during first packed decode setup

The original first-use handler caught every CUDA error from packed setup. An injected illegal-address status at setup synchronization reached legacy dispatch. The repair permits legacy dispatch only for packed scratch allocation OOM after the CUDA error check and release of partial allocations. A terminal status reaches the batch handler and disables the batch before any legacy submission or decode publication. [Batch error handler](../src/decoder_batch.cpp#L744)<!-- prefix: "} catch (con" -->

The initial repair still attempted setup for two one-row prefills or one decode with one one-row prefill. The independent fault probe disabled the batch in both cases. The second repair selects at least two packable lanes and a bucket before requesting scratch. It uses the same plan for binding. Both corrected controls published their expected results without hitting the fault. [Setup order](https://github.com/para1lel/sglang.cpp/blob/93648269d852e0fee78f7af4d1c3ffe4897aa361/src/text/decoder_graph_packed.cpp#L524)<!-- prefix: "if (!PackedD" -->

Run this procedure on [revision `9364826`](https://github.com/para1lel/sglang.cpp/tree/93648269d852e0fee78f7af4d1c3ffe4897aa361). Use that revision's [batch graph runbook](https://github.com/para1lel/sglang.cpp/blob/93648269d852e0fee78f7af4d1c3ffe4897aa361/tests/batch/README.md) to build Release tests. Set `out` to that build, `checkpoint` to the local checkpoint, and `GPU_UUID` to a reserved B200. Use a fresh evidence directory.

```bash
GPU_UUID="${GPU_UUID:?Set the reserved B200 UUID}" \
    bash tests/batch/run_batch_graph.sh "$out" "$checkpoint" "$out/packed-setup" packed-setup
python3 -IB tests/batch/check_graph_evidence.py "$out/packed-setup" --packed-setup-only
```

The eligible terminal arm requires two prefill score rows, no decode result, and rejection of later decode and admission. CUPTI must show no launches after the fault and no graph lifecycle calls. The OOM arm requires full BF16 parity with eager, sparse legacy replay after slot reuse, and zero request bytes at cleanup. Ineligible arms require full BF16 parity with their healthy controls and no injection, packed submission, or graph activity. [Terminal assertions](https://github.com/para1lel/sglang.cpp/blob/93648269d852e0fee78f7af4d1c3ffe4897aa361/tests/batch/check_graph_evidence.py#L432)<!-- prefix: "def verify_p" --> · [Eligibility assertions](https://github.com/para1lel/sglang.cpp/blob/93648269d852e0fee78f7af4d1c3ffe4897aa361/tests/batch/check_graph_evidence.py#L453)<!-- prefix: "def verify_p" -->

The independent `artifacts/resume-verifier/packed-fix-accept/report.md` records the terminal, OOM, and eligibility failures and controls. The OOM arm replayed sparse masks `0x5` and `0x6` after cleanup and retired with zero request bytes. The corrected eligibility reruns are in `artifacts/resume-packed-repair/eligibility-fix/report.md`. They identify the corrected source by SHA-256 `05b30ab632f7cd615379ea485126675774dda197b6b767c7ae332135f4460cef`.

## L010 · Rejected packed winner pin release

The original packed winner buffer used a deleter that logged a rejected `cudaFreeHost` and lost the pointer. The independent fault probe recovered the live pin by retrying from its observer. The repair gives the pin to [batch execution](batch-graphs.md#constraints-and-failure-behavior), where drain and graph teardown precede release and a rejected free leaves the pin eligible for deferred reclamation.

The `packed-setup` procedure in [L009](#l009--terminal-error-during-first-packed-decode-setup) runs the rejected-free arm. Its observer rejects the first release of the winner-copy destination and requires a successful retry on that same address. The checker also requires a packed replay mask of `0x5`, six token IDs, and valid BF16 score records. [Fault arm](https://github.com/para1lel/sglang.cpp/blob/93648269d852e0fee78f7af4d1c3ffe4897aa361/tests/batch/run_batch_graph.sh#L95)<!-- prefix: "run_lanes pa" --> · [Assertions](https://github.com/para1lel/sglang.cpp/blob/93648269d852e0fee78f7af4d1c3ffe4897aa361/tests/batch/check_graph_evidence.py#L394)<!-- prefix: "def verify_p" -->

## L011 · Saved benchmark counts concealed mismatched controls

The paired launcher accepted a saved summary when its cohort counts matched,
before resolving the target GPU. The review reproduced acceptance with an
unrelated GPU and fixture. Resolve the UUID first, compare the saved process
receipts and fixed inputs, and require native process evidence before labeling
a completed run. Keep the published cross-cap results under their own label.
See the [paired benchmark procedure](../benchmarks/paired/README.md).

The offline regression runs with `python3 -B -m unittest discover -s tests -p test_paired_identity.py`.
It uses saved-run receipts and the fixed streaming requests and checks that
unchanged counts cannot hide identity differences.

Archived K4 timing replay can reproduce all medians with an altered `--mtp-steps 0`
process receipt, or with every recorded native depth-four round reduced to depth
three. On the reviewed revision, the replay and the launcher's server-info
predicate also accepted a copied receipt claiming the distinct long split route.
The live argv and activity checks caught the first two controls. The formal W1
and W2 rows retain their original live witnesses. [Paired modes and controls](../benchmarks/paired/README.md#matched-k4-timing-protocol)

Record the strict native-versus-vLLM wall-time result separately from campaign
qualification. Before publishing a new row or using it as a predecessor, bind
the campaign, workload, frozen request, executable, physical GPU and arm to
observed argv, dispatch policy and runtime evidence of depth-four activity in
both engines. Check both bounded and long route state. Treat a missing long-route
field as disabled only with an identified pre-long source or receipt contract.
Reproduce a passing archive and keep negative controls for zero depth, missing
fourth-position activity, wrong row or profile, and long-route substitution.
The private reproduction and its output are retained in
`artifacts/session-1122-batch1-review/check-offline-mode.py` and `offline-mode-negative.log`.

## L012 · Manual CUDA launch arguments drifted from the kernel signature

In the unpublished long-target composition, `exact_scores_kernel` gained a
sixth parameter, `split_end`. The chunked draft path still packed five arguments
and crashed after one token per lane. The selected repair derives the host
argument pack from the kernel function pointer and passes `0` explicitly for
the chunked draft path. [Typed launcher](../include/sglang/text/gqa_decode_split.cuh#L18)<!-- prefix: "cudaError_t " --> · [Draft call](../include/sglang/text/mtp/gqa_long.cuh#L163)<!-- prefix: "auto status " -->

Keep the launch typed when changing a migrated kernel signature. In the selected
`8e4f210b` verification, removing the sixth argument failed NVCC compilation
before an object was emitted. The positive Release exercised the chunked draft
path and ordinary generation. The retained compile control is
`artifacts/continue-w3-long-target-independent/main-selected-gate.IahPUCPH/arity/compile.log`.

## L013 · Shared HTTP cache fixture changes missed public runners

The LMCache migration updated constraints, batch admission, and sampled MTP in `f38f03a`.
The next fix updated the media runner in `7e349bb`. The batch lifecycle runner still passed
the former resident entry count `4` to `server_policy`. Its Boolean identity check rejected
even a healthy daemon before the first W1 turn. Later assertions still expected resident
store events on individual scheduler steps. [First migration](https://github.com/para1lel/sglang.cpp/commit/f38f03af03450904c1fd4a8499687ee69d214cb4) · [Lifecycle call at the reviewed revision](https://github.com/para1lel/sglang.cpp/blob/7e349bb224fbb5715a2c5ff937d513d860fe6d01/tests/batch/checkpoint_lifecycle.py#L246)<!-- prefix: "http.server_" --> · [Review finding](https://github.com/para1lel/sglang.cpp/pull/1#discussion_r4694496532)

When the shared fixture changes, trace its imports and calls across test directories. Exercise
each affected public entry point in its documented cache mode. For this change, the media
runner checks both daemon-enabled and explicit cache-off workers. The batch runner requires a
dedicated daemon and proves that the saved W1 endpoint restores the exact rows and produces
the same continuation IDs as a cold request. [Media controls](../tests/media/http-contract.md) · [Lifecycle reproducer](../tests/batch/README.md#checkpoint-lifecycle)
