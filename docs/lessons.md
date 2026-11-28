# Development lessons

Record verified failures from [session review](../.agents/skills/main-mode/references/session-review.md#dispatch-the-batch-review) here. Merge recurrences under the same identifier. Keep exact inputs, commands, and results in the review record.

## L001 · Header placement

Local includes hid nine project headers outside `include/sglang/`. The original placement gate missed unchanged files. The declarations gate checks indexed project headers at every path, including `.ipp` and `.tpp` files. [Original placement regression](https://github.com/para1lel/sglang.cpp/blob/572627c81b89b1c75c7bcda1c2e9c23a4c8e95f6/docs/lessons.md#l001--header-placement) · [Current rules](../AGENTS.md#integrating-changes)

## L002 · Remapped KV pages

**Corrected.** Prefix store and restore assumed that logical and physical page indices matched. The identity case passed. With logical page zero at physical page one, the remapped case reported `Key payload mismatch`.

The former resident cache fixed both transfers with one physical page-map reader. Store gathered into logical order. Restore scattered into the destination mapping. The fix passed the remap and guard checks. [Historical page-map reader](https://github.com/para1lel/sglang.cpp/blob/a7583f7b6346b5eb91aba2cba4748a6461e60fb4/src/decoder_prefix_cache.cpp#L151)<!-- prefix: "GpuPrefixSna" -->

Keep independent source and destination permutations in KV-copy tests. Compare all four KV planes and untouched tail pages. [Former B200 remap and guard procedure](https://github.com/para1lel/sglang.cpp/blob/a7583f7b6346b5eb91aba2cba4748a6461e60fb4/tests/prefix/README.md#cuda-state-and-routes) · [Current LMCache verification](prefix-cache.md#verification)

## L003 · Patch and anchor provenance

A composed patch left stale and blank code anchors after source and layout changes. Record base, ordered patch digests, and resulting tree. Compare the composed tree before qualification and recheck incoming anchors after overlapping source edits. [Frozen review](https://github.com/para1lel/sglang.cpp/blob/a5974ba4eb99bd8e8f890c2e7262a27dbc3aace1/docs/lessons.md#l003--patch-and-anchor-provenance)

## L004 · Finite-input NaNs in page-parallel GQA

**Correction verified.** At 1,536 visible tokens, finite inputs overflowed FP32 QK scores or page-local value sums. Both cases returned NaNs in eager and graph modes while ordered replay and the FP64 oracle returned finite outputs.

The [merge guard](batch-graphs.md#page-parallel-gqa-decode) sends an affected head and channel tile through ordered replay. The correction matched the ordered and FP64 outputs on both cases. Keep both cases in the GQA regression and run them with eager execution and graph replay.

## L005 · HTTP request parsing on the accept thread

A stalled HTTP body previously blocked health and completions on the accept thread. Check independent HTTP 200 replies with one held reader, then verify shutdown joins the reader and discards its request. The current reader progresses incomplete requests from one poll loop. [Reader contract](http-server.md#http-admission-and-shutdown) · [Frozen regression](https://github.com/para1lel/sglang.cpp/blob/a5974ba4eb99bd8e8f890c2e7262a27dbc3aace1/docs/lessons.md#l005--http-request-parsing-on-the-accept-thread)

## L006 · FP16 operand loss in split GQA

FP16 narrowing lost small Q channels before multiplying large K values and lost the weighted V operand in a second case. Keep BF16 Q and decoded KV, FP32 accumulation, and per-head FP64 replay on unsafe range. Use the reserved B200 regression from the repository root. [Query load](../src/text/gqa_decode_split.cu#L175)<!-- prefix: "__device__ v" --> · [Regression cases](../tests/decoder/gqa_split_fp64_check.cpp#L181)<!-- prefix: "int main() t" --> · [Frozen range probes and results](https://github.com/para1lel/sglang.cpp/blob/6debe50edccdeda34ff307235ed38e74b6b8f2ee/docs/lessons.md#l006--fp16-operand-loss-in-split-gqa)

```bash
out="$PWD/build/long-decode"
xmake f -P "$PWD" -o "$out" -p linux -a x86_64 -m release --long-decode-check=y
xmake build -P "$PWD" gqa-split-fp64-check
CUDA_VISIBLE_DEVICES="${GPU_UUID:?Set the reserved B200 UUID}" \
    "$out/bin/gqa-split-fp64-check"
```

Compare well-conditioned, query-range, and weighted-value cases with FP64. The former eager/replay packed probe and its receipts are in the archive. See the [packed drift evidence](long-paged-decode.md#packed-route-drift-evidence) for residual drift.

## L007 · Singleton route parity

Singleton graph attention diverged from eager when capture used a different operator. Keep the same singleton route before and after eligible launch rejection. The correction recovered all 352 fixed-history decode rows bitwise. [Bound attention](../src/runtime/execution/decoder_forward_attention.cpp#L180)<!-- prefix: "void attend_" --> · [Frozen candidate and control results](https://github.com/para1lel/sglang.cpp/blob/6debe50edccdeda34ff307235ed38e74b6b8f2ee/docs/lessons.md#l007--singleton-route-parity)

Build using the [batch graph runbook](../tests/batch/README.md). Set `out` to the Release build directory, `checkpoint` to the local checkpoint, and `GPU_UUID` to a reserved B200. Use a fresh `$out/singleton` directory.

```bash
GPU_UUID="${GPU_UUID:?Set the reserved B200 UUID}" \
    bash tests/batch/run_batch_graph.sh "$out" "$checkpoint" "$out/singleton" singleton
python3 -IB tests/batch/check_graph_evidence.py "$out/singleton" --singleton-only
```

Check slot 31, replacement, launch rejection, and later width-two replay. Require all 17 complete BF16 rows per arm, including four packed pairs missed by the initial observer. [Frozen observer, checkpoint rerun, and build identities](https://github.com/para1lel/sglang.cpp/blob/a5974ba4eb99bd8e8f890c2e7262a27dbc3aace1/docs/lessons.md#l007--singleton-route-parity)

## L008 · Buffered tokens suppress SSE keepalives

A pending decode step can buffer IDs without sending SSE data. Measure the 15-second deadline from the last successful write. The archived B200 probe stalled 18.349 seconds before the fix and sent a comment at 15.006 seconds after it. [corrected writer](http-server.md#completion-streams) · [Frozen wire controls](https://github.com/para1lel/sglang.cpp/blob/6debe50edccdeda34ff307235ed38e74b6b8f2ee/docs/lessons.md#l008--buffered-tokens-suppress-sse-keepalives)

Run the scripted CPU regression from the repository root in the managed environment. Set `out` to an existing output directory and `checkpoint` to the local Qwen3.8-27B-FP8 checkpoint. Use a fresh `$out/keepalive` directory. [Scripted regression](../tests/text/check_stream_keepalive_cpu.py#L58)<!-- prefix: "def main():" -->

```bash
bash tests/text/build_chat_cpu.sh "$PWD" - "$out/chat-cpu"
python3 -IB tests/text/check_stream_keepalive_cpu.py \
    --binary "$out/chat-cpu" --checkpoint "$checkpoint" --evidence "$out/keepalive"
```

The scripted clock advances three seconds per step. Text and chat must send two comments, including one after visible data resets the deadline, and match role, content, finish, usage, and `[DONE]`.

## L009 · Terminal error during first packed decode setup

The former packed setup swallowed terminal CUDA failures and fell through to legacy dispatch. Only clean allocation OOM may retry after error checks and cleanup. Select an eligible bucket before requesting packed scratch. The frozen runbook at revision `9364826` includes runnable fault arms and the exact no-publication, launch, graph and BF16 checks. [revision `9364826`](https://github.com/para1lel/sglang.cpp/tree/93648269d852e0fee78f7af4d1c3ffe4897aa361) · [batch graph runbook](https://github.com/para1lel/sglang.cpp/blob/93648269d852e0fee78f7af4d1c3ffe4897aa361/tests/batch/README.md) · [Frozen terminal, OOM, and eligibility receipts](https://github.com/para1lel/sglang.cpp/blob/a5974ba4eb99bd8e8f890c2e7262a27dbc3aace1/docs/lessons.md#l009--terminal-error-during-first-packed-decode-setup)

## L010 · Rejected packed winner pin release

The original packed winner buffer used a deleter that logged a rejected `cudaFreeHost` and lost the pointer. The independent fault probe recovered the live pin by retrying from its observer. The repair gives the pin to [batch execution](batch-graphs.md#constraints-and-failure-behavior), where drain and graph teardown precede release and a rejected free leaves the pin eligible for deferred reclamation.

The `packed-setup` procedure in [L009](#l009--terminal-error-during-first-packed-decode-setup) runs the rejected-free arm. Its observer rejects the first release of the winner-copy destination and requires a successful retry on that same address. The checker also requires a packed replay mask of `0x5`, six token IDs, and valid BF16 score records. [Fault arm](https://github.com/para1lel/sglang.cpp/blob/93648269d852e0fee78f7af4d1c3ffe4897aa361/tests/batch/run_batch_graph.sh#L95)<!-- prefix: "run_lanes pa" --> · [Assertions](https://github.com/para1lel/sglang.cpp/blob/93648269d852e0fee78f7af4d1c3ffe4897aa361/tests/batch/check_graph_evidence.py#L394)<!-- prefix: "def verify_p" -->

## L011 · Saved benchmark counts concealed mismatched controls

Saved counts and medians hid differences in device, inputs, and draft policy. An altered receipt could also claim a different route. Bind results to the physical GPU UUIDs, executable and dependency hashes, observed arguments, resolved policy, actual dispatch, and scored request cohort. Use route checks before interpreting missing activity. [Execution identity](../.agents/skills/coding-mode/references/execution-identity.md) · [AgentX evidence](agentx-256k.md#scoring-and-evidence)

## L012 · Manual CUDA launch arguments drifted from the kernel signature

In the unpublished long-target composition, `exact_scores_kernel` gained a
sixth parameter, `split_end`. The chunked draft path still packed five arguments
and crashed after one token per lane. The selected repair derives the host
argument pack from the kernel function pointer and passes `0` explicitly for
the chunked draft path. [Typed launcher](https://github.com/para1lel/sglang.cpp/blob/8e4f210b7cde78ab935faabaa0e3693467fa8b5b/include/sglang/text/gqa_decode_split.cuh#L17)<!-- prefix: "cudaError_t " --> · [Draft call](https://github.com/para1lel/sglang.cpp/blob/8e4f210b7cde78ab935faabaa0e3693467fa8b5b/include/sglang/text/mtp/gqa_long.cuh#L104)<!-- prefix: "auto status " -->

Keep launch arguments typed. At `8e4f210b`, omitting the sixth argument failed NVCC before object generation. The positive Release exercised chunked drafting and ordinary generation. [Frozen compile-control record](https://github.com/para1lel/sglang.cpp/blob/6debe50edccdeda34ff307235ed38e74b6b8f2ee/docs/lessons.md#l012--manual-cuda-launch-arguments-drifted-from-the-kernel-signature)

## L013 · Shared HTTP cache fixture changes missed public runners

The LMCache migration updated constraints, batch admission, and sampled MTP in `f38f03a`.
The next fix updated the media runner in `7e349bb`. The batch lifecycle runner still passed
the former resident entry count `4` to `server_policy`. Its Boolean identity check rejected
even a healthy daemon before the first request. Later assertions still expected resident
store events on individual scheduler steps. [First migration](https://github.com/para1lel/sglang.cpp/commit/f38f03af03450904c1fd4a8499687ee69d214cb4) · [Lifecycle call at the reviewed revision](https://github.com/para1lel/sglang.cpp/blob/7e349bb224fbb5715a2c5ff937d513d860fe6d01/tests/batch/checkpoint_lifecycle.py#L246)<!-- prefix: "http.server_" --> · [Review finding](https://github.com/para1lel/sglang.cpp/pull/1#discussion_r4694496532)

When the shared fixture changes, trace its imports and calls across test directories. Exercise
each affected public entry point in its documented cache mode. For this change, the media
runner checks both daemon-enabled and explicit cache-off workers. The batch runner requires a
dedicated daemon and proves that restoring the stored endpoint recovers the exact rows and produces
the same continuation IDs as a cold request. [Media controls](../tests/media/http-contract.md) · [Lifecycle reproducer](../tests/batch/README.md#checkpoint-lifecycle)

## L014 · Score scratch limits hid graph fallback

**Open.** TP1 MTP verification reused a 248,320-element BF16 logits buffer for FP32 GQA scores. With 24 query heads and 128-token pages, only 40 whole pages fit. The verifier quietly ran eager when its exclusive row end exceeded 5,120. The score-slab check asserted the inline limit and exercised separate long scratch for ordinary batch decode. It did not exercise the MTP verifier at that boundary. [Inline score capacity](https://github.com/para1lel/sglang.cpp/blob/6cf81a27829354551397d463113be002ea025343/src/runtime/model/qwen_request.cpp#L256)<!-- prefix: "ExactGqaWork" --> · [Verifier gate](https://github.com/para1lel/sglang.cpp/blob/6cf81a27829354551397d463113be002ea025343/src/decoder_verify.cpp#L401)<!-- prefix: "eager_worksp" --> · [Score-slab check](https://github.com/para1lel/sglang.cpp/blob/6cf81a27829354551397d463113be002ea025343/tests/batch/check_score_slab.cpp#L141)<!-- prefix: "require(requ" -->

The default server enabled a separate TP2 verifier option when eager mode was off. Both C++ graph flags defaulted to `false`. TP2 verification graph dispatch read `tp_verification_graphs` independently of `cuda_graphs`. The TP2 graph required two lanes and ten rows and had a separate score limit of 256 positions. Long-history TP2 replay needed a direct graph check. [Server option wiring](https://github.com/para1lel/sglang.cpp/blob/6cf81a27829354551397d463113be002ea025343/apps/sglang-server.cpp#L405)<!-- prefix: "arguments.ma" --> · [C++ options](https://github.com/para1lel/sglang.cpp/blob/6cf81a27829354551397d463113be002ea025343/include/sglang/decoder_batch.h#L70)<!-- prefix: "bool tp_veri" --> · [TP2 position gate](https://github.com/para1lel/sglang.cpp/blob/6cf81a27829354551397d463113be002ea025343/src/runtime/execution/tp_verify_graph.cpp#L128)<!-- prefix: "row.position" -->

Check capture and warm replay through default HTTP and both explicit C++ graph options. Exercise row ends immediately below, at, and above each scratch limit. Include replay near the 262,144-token context limit and vary the active group and sampling policy. Observe which graph launched and record the reason for each supported decline. Compare IDs and committed state with eager execution at the same positions. Keep allocation and replay checks on the same verifier path. [Optimized route verification rules](../.agents/skills/coding-mode/SKILL.md#prepare-the-candidate)

## L015 · Target parallelism hid rank 0 drafting

**Open.** The TP2 constructor loaded sharded target layers on both devices and assigned MTP weights only to the root model. Batch admission created a single draft request, with its draft KV owned by rank 0. Rank 1 built a target verification round without a draft request. Eligible peer repair prepared work from the root draft weights. That work assisted repair while proposal and persistent draft state remained on rank 0. [Draft weight ownership](https://github.com/para1lel/sglang.cpp/blob/6cf81a27829354551397d463113be002ea025343/src/decoder.cpp#L520)<!-- prefix: "model_-\u003emtp_" --> · [Draft request](https://github.com/para1lel/sglang.cpp/blob/6cf81a27829354551397d463113be002ea025343/src/decoder_batch.cpp#L108)<!-- prefix: "lane.draft =" --> · [Peer round](https://github.com/para1lel/sglang.cpp/blob/6cf81a27829354551397d463113be002ea025343/src/decoder_tp.cpp#L167)<!-- prefix: "return {null" --> · [Peer repair](https://github.com/para1lel/sglang.cpp/blob/6cf81a27829354551397d463113be002ea025343/src/decoder_tp.cpp#L181)<!-- prefix: "peer_repair " -->

Validation checked the target communicator and rank identity. Output agreement and target collectives could therefore pass with draft state and proposal work confined to one rank.

For a sharded draft policy, prove which weights each rank loads and which useful draft work each rank executes. Check proposal, acceptance and rejection repair, and committed KV on both ranks. Check cold seeding and restoration of the same endpoint from LMCache. Use controls that expose missing peer draft work and stale peer state. Report draft topology separately from target topology and measure its memory and latency costs. [Optimized route verification rules](../.agents/skills/coding-mode/SKILL.md#prepare-the-candidate)
