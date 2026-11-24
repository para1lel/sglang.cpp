# Native MTP

Multi-token prediction (MTP) verifies draft candidates in order against the
target model. It supports greedy decoding and sampling.

## Responsibilities and sources of truth

| Part | Responsibility | Source |
| --- | --- | --- |
| Generator and server | Set the proposal limit and pass request options to the decoder. | [Generator options](../apps/sglang-generate.cpp#L32)<!-- prefix: "void usage(s" --> · [Server dispatch](../apps/sglang-server.cpp#L298)<!-- prefix: "BatchService" --> |
| MTP round | Draft proposals and commit target outputs in order. | [Proposals](../src/decoder_verify.cpp#L190)<!-- prefix: "bool MtpRoun" --> · [Commit](../src/decoder_verify.cpp#L312)<!-- prefix: "bool MtpRoun" --> |
| Target verifier | Process causal rows by layer and commit each request's consumed prefix. | [Forward](../src/decoder_verify.cpp#L641)<!-- prefix: "bool Decoder" --> · [State commit](../src/decoder_verify.cpp#L529)<!-- prefix: "void TargetV" --> |

## Runtime path

`sglang-server` and `sglang-generate` default to MTP depth four (K4). Use `--mtp-steps 0` to disable MTP (K0). Values `1`–`4` limit draft proposals per round. Both commands accept `--mtp-k` as an alias. C++ `DecoderOptions` defaults to K0. [Depth option](../apps/sglang-server.cpp#L149)<!-- prefix: "} else if (o" --> · [Server default](../apps/sglang-server.cpp#L38)<!-- prefix: "int mtp_step" --> · [Generator default](../apps/sglang-generate.cpp#L124)<!-- prefix: "arguments.op" --> · [C++ options](../include/sglang/decoder.h#L40)<!-- prefix: "int mtp_step" -->

At capacities 1–32, text, media, and native generation use configured MTP depth. Greedy [token logprobs](native-generate.md#token-logprobs) can use MTP and score committed target rows. Sampling and [constraints](output-constraints.md) use their own verification paths. Pointwise scores and target-only options use K0. `Decoder::generate` uses one batch lane for MTP. [Effective depth](../src/decoder_batch.cpp#L410)<!-- prefix: "if (options." --> · [Verifier scores](../src/decoder_verify.cpp#L482)<!-- prefix: "void TargetV" -->

MTP loads draft weights once and shares the target embedding and head.
Each request owns its target state, draft MXFP8 KV, and verifier storage.
C++ callers set depth per lane. Positive-depth lanes can [restore paired prefixes](prefix-cache.md#checkpoint-compatibility-and-resume).
[Weights](../src/decoder.cpp#L453)<!-- prefix: "Decoder::Dec" --> · [Admission](../src/decoder_batch.cpp#L397)<!-- prefix: "int DecoderB" -->

On B200, loaded draft weights support [INT8 draft scoring](native-model-info.md#int8-draft-head).

Eligible B200 MTP prompt tiles hold up to 1,024 rows. Other devices cap tiles at 128. Draft seeding pairs each final-normalized target row with the next prompt token's embedding. A media lane supplies its own embedding and rotary coordinates. Single rows or unavailable scratch use scalar prefill. [Tile cap](../src/runtime/execution/decoder_execution_policy.cpp#L25)<!-- prefix: "int DecoderE" --> · [Draft input](../src/decoder_prefill.cpp#L681)<!-- prefix: "seed.seed_dr" -->

B200 Tensor Core prefill can prefer TMA and `tcgen05` for draft seed FC on prompts of at least 4,096 tokens. Each FC call needs 64–128 effective rows, 10,240 input and 5,120 output channels, and 16-byte-aligned input and weights. Other shapes and input alignments use SIMT row GEMV. [Prefill selection](../src/decoder_prefill.cpp#L679)<!-- prefix: "const bool w" --> · [FC dispatch](../src/text/draft_seed_fc.cu#L124)<!-- prefix: "cudaError_t " -->

TMA requires 4-byte-aligned output, a matching image and tensor-map encoder, and enough shared memory. Unaligned output or unsupported preflight selects the 32×64 WMMA tile. CUDA errors, including stream/device mismatch and launch failure, propagate without retry. SIMT accumulation can change draft KV and proposals. [TMA preflight](../src/text/draft_seed_fc_sm100.cu#L195)<!-- prefix: "std::optiona" --> · [Submission](../src/text/draft_seed_fc_sm100.cu#L241)<!-- prefix: "return submi" --> · [WMMA tile](../src/text/draft_seed_fc.cu#L24)<!-- prefix: "constexpr in" -->

Tiled and row-wise seeding normalize and concatenate both inputs before the draft input and KV projections. Both normalize and rotate keys and write MXFP8 KV at the target positions. Both skip query projection, draft attention, MLP, and scoring. [Draft seed](../src/decoder_mtp.cpp#L789)<!-- prefix: "void MtpDraf" -->

For a multi-row seed tile on B200, each request prepares FP8 activations in
slices of up to 128 rows on the prefill stream. One exact FP8 kernel projects
K and V in separate grid banks for the tile. One key normalization and RoPE
launch and one MXFP8 append follow. Grouped prefill seeds each request
separately. Single-row tiles and other devices project and append each slice
separately. Seed scratch comes from the existing prefill workspace.
[Slice preparation](../src/decoder_mtp.cpp#L833)<!-- prefix: "for (int off" --> ·
[Paired projection](../src/text_ops.cu#L789)<!-- prefix: "cudaError_t " --> ·
[Grid banks](../include/sglang/text/prefill_fp8_exact.cuh#L209)<!-- prefix: "__global__ v" --> ·
[Grouped seed](../src/decoder_prefill.cpp#L681)<!-- prefix: "seed.seed_dr" -->

The target scores the last prompt row and selects the first token. If generation continues, that token and the last normalized target row start the draft round at `L-1`. Later proposals use the previous draft's final-normalized row and prediction. The proposal loop synchronizes the stream before reading winners. If any lane retries INT8 scoring in BF16, the loop synchronizes again before reading the retry results. [Proposal loop](../src/decoder_verify.cpp#L190)<!-- prefix: "bool MtpRoun" -->

Eligible B200 draft waves select [grouped online attention](native-model-info.md#draft-attention) automatically.

### Draft winner readback

On B200, if every lane has mapped pinned host storage, two to four unconstrained draft lanes share two argmax launches. The first launch reduces tiles within each lane. The second writes each lane's winning ID and score directly to that request's host record. The selection rules match the per-lane path, including lowest-ID ties and invalidation for nonfinite INT8 scores. [Dispatch](../src/decoder_mtp.cpp#L439)<!-- prefix: "void MtpDraf" --> · [Launcher](../src/text_ops.cu#L862)<!-- prefix: "cudaError_t " --> · [Selection](../src/text_ops.cu#L430)<!-- prefix: "__device__ T" -->

Each request can reuse its mapping during stream capture. Outside capture, a request can allocate mapped storage on demand. Singleton and constrained lanes use per-lane selection and copy the winning ID. If capture prevents allocation or pinned allocation fails on any lane, the group uses per-lane selection. Other devices also use per-lane selection. Other CUDA errors propagate. [Mapping](../src/decoder_mtp.cpp#L419)<!-- prefix: "bool MtpDraf" --> · [Allocation](../src/decoder_verify.cpp#L565)<!-- prefix: "std::shared_" --> · [Per-lane selection](../src/decoder_mtp.cpp#L397)<!-- prefix: "void MtpDraf" -->

### Target round

Each target round starts with the latest committed token. Proposals supply subsequent inputs. The batch limits proposals to `K`, remaining outputs minus one, and assigned target rows minus one. The row grant also respects [checkpoint boundaries](prefix-cache.md#checkpoint-boundaries). A one-row verification makes no proposals. One remaining output uses ordinary eager decode. Zero budget without scoring allocates no request state. [Batch assignment](../src/decoder_batch.cpp#L601)<!-- prefix: "if (results[" --> · [Zero-budget finish](../src/decoder_batch.cpp#L433)<!-- prefix: "if (options." -->

The batch forms packets of up to four lanes in slot order under one step budget. Ordinary packets interleave causal rows by position, verify up to 20 rows, and commit separately. Serial generation can read raw BF16 logits. Eligible eager greedy rounds use the [certified target head](native-model-info.md#certified-tensor-core-target-head). [Target round](../src/decoder_verify.cpp#L601)<!-- prefix: "bool Decoder" -->

On B200, a batch with draft weights and certified vocabulary scratch needs capacity eight or more and a step cap of at least 40. It combines adjacent eager packets into 8, 16, or 32 lanes with 40, 80, or 160 assigned rows. Each lane needs five assigned rows, at least five remaining outputs, and greedy unconstrained K4 without raw logits. Sparse slots preserve packet order. The [wide FP8 projections](fp8-prefill-projections.md#multi-lane-target-verification) share a pass. The [certified head](native-model-info.md#certified-tensor-core-target-head) can score pairs of full packets. [Draft projections](native-model-info.md#grouped-draft-projections) and KV repairs can use complete 16-lane groups. [Long GQA](singleton-gqa-decode.md#tensor-core-reads-across-long-k4-packets) can share a launch across packets. [Compact GDN](gdn-prefill.md#grouped-target-verification) can share 80- or 160-row launches. Short drafts fall back to eligible aligned eight-lane groups or ordinary packets. [Planner](../include/sglang/text/decoder_batch_plan.h#L172)<!-- prefix: "inline WideM" -->

The controller commits matching proposals in order. The first mismatch commits a target correction and ends the round. It leaves later proposals unconsumed. Full acceptance can commit a target bonus within the stop rule and output budget. A stop match takes precedence over length. A correction or bonus becomes the next target input if generation continues. Draft KV rewinds to the first target position of the round. Continuing lanes rebuild only the accepted draft KV rows from target-normalized rows. [Round commit](../src/decoder_verify.cpp#L312)<!-- prefix: "bool MtpRoun" -->

The verifier writes tentative GDN state into an alternate bank. Full consumption swaps that bank into the request. Partial consumption replays saved inputs for the consumed prefix from the prior bank, then swaps. Zero consumption preserves the initial bank. The target KV prefix advances by the consumed rows. Later forwards bound reads by position and overwrite rejected slots before they become visible. [Causal attention](../src/decoder_forward.cpp#L504)<!-- prefix: "cudaError_t " --> · [State commit](../src/decoder_verify.cpp#L529)<!-- prefix: "void TargetV" -->

C++ callers consume `token`, then the first `extra_token_count` entries of `extra_tokens`. `extra_token` mirrors the first extra for depth-one callers. HTTP workers receive output after the batch commits state. Per-token callbacks deliver IDs in order and check cancellation between tokens. Native [step delivery](native-generate.md#text-stops) can send several IDs in one cumulative event. Eligible per-token writers use [bounded decode output](http-server.md#bounded-decode-output). [Batch API](../include/sglang/decoder_batch.h#L84)<!-- prefix: "class Decode" --> · [Token delivery](../include/sglang/server/output_mailbox.h#L30)<!-- prefix: "inline bool " --> · [Step delivery](../apps/server/batch_service.cpp#L648)<!-- prefix: "bool BatchSe" -->

### Target verification graph

The default [graph policy](batch-graphs.md) permits verification replay for one unconstrained text lane with two target rows. Both positions must fit inline GQA scores. The last eligible pair is 5,118 and 5,119 (zero-based). Replay and eager verification share two-row projection grouping. Replay uses bound GQA dispatch, including the [split-KV position check](singleton-gqa-decode.md#split-kv-target-attention) when the split policy is active. The first eligible group runs eagerly. The next captures the verifier on the batch stream. [Group dispatch](../src/decoder_verify.cpp#L665)<!-- prefix: "if (graph &&" --> · [Row eligibility](../src/decoder_verify.cpp#L420)<!-- prefix: "bool TargetV" --> · [Capture](../src/decoder_verify.cpp#L666)<!-- prefix: "std::exchang" -->

Each launch refreshes tokens, positions, KV page-table descriptors and cache pointers, GDN bank pointers, and GQA score bindings. The executable reuses fixed batch-owned scratch. Draft proposals, sampling, result readback, acceptance, and state commit run outside it. C++ reports replayed slots in `verification_graph_slots`. The HTTP `graph_slots` trace reports K0 replay. [Bindings](../src/decoder_verify.cpp#L71)<!-- prefix: "void Verific" --> · [Slot mask](../include/sglang/decoder_batch.h#L54)<!-- prefix: "std::uint32_" --> · [HTTP trace](../include/sglang/server/batch_trace.h#L51)<!-- prefix: "if (stats.gr" -->

Direct `Decoder::generate` verification runs eagerly. Ineligible batch groups run eagerly. A K4 round with four proposals has five target rows and runs eagerly. With one output token left, the lane uses ordinary eager decode. Other groups in the step can still qualify. `--eager` disables replay for all groups.

### Sampled MTP

An item with active [output penalties](output-penalties.md), [logit bias](serving/openai-logprobs.md#logit-bias), or a positive [output minimum](output-penalties.md#minimum-output) uses K0. Eligible peers in the same batch keep their configured depth. [Target-only selection](../include/sglang/decoder.h#L76)<!-- prefix: "bool require" -->

The draft proposes greedy IDs. The target uses the [normal sampler](text-serving.md#sampling) on the first score and each verification row. It keys each draw by the request's resolved seed and the row's prospective output index. Discarded rows leave subsequent output indices available for recomputation on the corrected history. The same index drives the draw on that new row. [Verifier sampler](../src/decoder_verify.cpp#L460)<!-- prefix: "void TargetV" -->

For a valid causal row with filtered target law `P`, the target draws `T` from `P`. It emits `T`, reusing proposal state when `T` equals the proposed ID. A fixed proposal `d` is therefore accepted with probability `P(d)`. Ratio-based speculative sampling uses `min(1, P(d)/Q(d))` and a residual draw on rejection. Here the target draw itself supplies the correction. Full acceptance permits a target bonus under the usual budget and stop rules.

Each emitted token has the target law for its causal row. A fixed seed gives the same sampled IDs as K0 when both paths reach identical effective rows and use the same sampler and output indices. Prefill, forward precision, and batch grouping can change the logits and therefore the draws.

### Constrained MTP

The first target output uses the [ordinary constraint selection](output-constraints.md#selection-and-finish). Each round starts from the committed cursor. The draft chooses the highest-scoring allowed ID and advances a private cursor. Each row uses its own remaining output budget. If a proposal completes the constraint, the target verifies it and skips the bonus row. [Round cursor](../src/decoder_verify.cpp#L100)<!-- prefix: "void MtpCons" -->

The target filters each verified row with the allowed IDs for that candidate prefix. It uses the ordinary greedy rule or samples with the prospective output index. Only emitted target IDs advance the live cursor. A mismatch or finish discards later proposals and their cursor states. A correction or bonus can also complete the constraint. Identical score rows, cursors, sampler options, and output indices give identical target decisions. [Target selection](../src/decoder_verify.cpp#L470)<!-- prefix: "void TargetV" -->

## Constraints and failure behavior

The verifier reserves up to `K+1` rows, limited to the initial output budget minus one. The step cap can also limit the computed row count. The [resident batch path](batch-graphs.md#resident-memory-admission) allocates verifier storage on admission. Other paths allocate it before the first verification. Five rows add **167,355,584 logical device bytes (159.60 MiB)** per request, independent of prompt length. This includes the alternate GDN bank, saved replay inputs, and row workspaces. Existing target and draft storage is separate. Five-row pinned results use 40 bytes for winners or 2,483,240 bytes with serial `--top-logits`. The batch API rejects top logits. Constrained verification also reserves up to 16,384 pinned host bytes per row for IDs or 31,040 for a wide mask. [Constraint staging](../src/decoder_verify.cpp#L85)<!-- prefix: "MtpConstrain" --> · [Row bound](../src/decoder_resources.cpp#L342)<!-- prefix: "int reserved" --> · [Storage](../src/decoder_verify.cpp#L353)<!-- prefix: "TargetVerifi" -->

MTP releases temporary prefill scratch after processing each batch step. Direct generation checks its cancellation callback while advancing the lane. Batch cancellation can leave work from a completed round unused by the client. A failed speculative batch step fails the current jobs. If the stream drains, the service can admit replacements. If completion is uncertain, the batch is disabled and retains its storage. [Step recovery](../src/decoder_batch.cpp#L955)<!-- prefix: "} catch (con" --> · [Cleanup](../src/decoder_resources.cpp#L225)<!-- prefix: "CleanupError" -->

The draft prefix advances after a successful KV append enqueue. If preparation
fails in a later slice, the paired path attempts to append the prepared prefix
before reporting the error. A draft preparation, projection, or append error
marks the draft failed. A target final-normalization error leaves the draft's
failure flag clear if the prepared prefix flush succeeds.
[Append](../src/decoder_mtp.cpp#L812)<!-- prefix: "check(mxfp8_" --> ·
[Target normalization failure](../src/decoder_mtp.cpp#L845)<!-- prefix: "append_prepa" --> ·
[Draft preparation failure](../src/decoder_mtp.cpp#L870)<!-- prefix: "append_prepa" -->

For a recoverable verification graph setup failure, eager fallback requires ended capture, a drained stream, and successful handle cleanup. This disables only the verifier executable. A binding upload failure follows speculative step recovery without replaying the failed round. A verification launch error or unconfirmed capture cleanup disables the batch without reevaluating the target. Resources stay live until safe cleanup. [Graph recovery](../src/decoder_resources.cpp#L610)<!-- prefix: "const auto f" --> · [Launch](../src/decoder_resources.cpp#L687)<!-- prefix: "const auto l" -->

If a constrained draft row has no valid winner, drafting stops and the target can still select an output. For an invalid winner on a consumed constrained row, direct generation throws `DecoderConstraintError`. The batch reports `constraint_failed` and retires the lane. For an invalid unconstrained sample, direct generation throws `DecoderSamplingError`. The batch reports `sampling_failed`. HTTP drops the failed lane's unpublished step tokens. Rows discarded after a mismatch or stop have no effect. [Batch consumption](../src/decoder_batch.cpp#L82)<!-- prefix: "void finish_" --> · [HTTP failure](../apps/server/batch_service.cpp#L370)<!-- prefix: "if (step.sam" -->

## Verification

Run the [MTP acceptance procedure](../tests/mtp/README.md) with both release binaries, the pinned checkpoint, and a reserved B200. Require exit status zero. Match target dispatch and prompt tiles for the K0 control. Compare default generation with explicit K4. Check IDs, finish reasons, usage, and SSE at wider capacities. For seed parity, pass a row-wise seed generator with the same target verifier to `--reference-generate`. [Runner](../tests/mtp/run_native.py#L1)<!-- prefix: "\"\"\"Opt-in re" -->

Compare complete BF16 draft vocabulary rows bitwise with per-lane scoring. Compare winner IDs and scores exactly in waves of two, three, and four. Include cross-tile ties and nonfinite inputs. Test eager launches and graph replay with mapped storage. Trace both grouped argmax launches and require zero explicit winner copies. Check per-lane fallback and the BF16 retry after an invalid INT8 winner.

Compare paired RMSNorm and key-only preprocessing with the single-row operations. At matched compute boundaries, compare full logits, final-normalized rows, target state, and draft KV where arithmetic agrees. Check subsequent draft and verifier logits. Exercise 1, 127, 128, 129, and 257 positions and a longer multi-tile prompt. Cover cold and restored prompts, media, and scored generation. Keep the K0 graph control. Disconnect during prefill and verify queued-request parity after retirement. [Paired RMSNorm](../src/text_normalization.cu#L525)<!-- prefix: "cudaError_t " --> · [Key preprocessing](../src/text_attention_gqa.cu#L266)<!-- prefix: "cudaError_t " -->

Compare paired K and V BF16 projections byte for byte with the exact per-slice
projections before key preprocessing. Trace one paired projection and one
append per multi-row B200 seed tile. Include a prefix restore and a tile with
a partial final slice. Inject a seed FC launch error in the second slice.
Check that the first slice extends the draft prefix, the failed request retires,
and the batch admits a replacement.

Compare TMA seed FC output byte for byte with WMMA at every eligible row count. Trace TMA launches and check WMMA fallback before submission, SIMT at the row boundaries, and retirement after a launch error.

Compare BF16 logits, normalized rows, target KV, and GDN with row-wise eager verification through zero, partial, and full acceptance. Check draft KV and continuation after rewind, KV crossings, mixed depths, and the exact GQA score-capacity boundary. Reset a lane and check retained and replacement streams. Require zero request bytes after retirement. Check recovery from result-sync failure and real launches for K0 and positive-K graphs. Use the [wide projection controls](fp8-prefill-projections.md#wide-verifier-checks) for 40-, 80-, and 160-row routes.

At widths 16 and 32, compare dense draft with four-lane controls at fixed grants and target dispatch. Cover full and sparse proposal and repair depths. Dense BF16 FC can change draft KV and acceptance. On fixed histories, compare draft projections and KV separately from target state and continuation IDs. Check clean setup decline, error propagation, and replacement.

At server capacity 32, check K1 with more than four verifying lanes. Check sparse slot selections. Match committed IDs, finish reasons, and usage against same-policy controls. Check the combined target-row count against the step cap. Include a lane that receives one verification row and makes zero proposals. Check final-token eager decode, zero-budget completion, and reset recovery. Run [performance measurement](performance.md#native-mtp) separately.

### Verification graph checks

Follow the [batch graph runbook](../tests/batch/README.md) on a reserved B200. Compare eager and replayed verifier BF16 rows and committed IDs. Check sparse slots, replacement, mixed K0 and K1 steps, and eager group exclusions. Exercise recoverable capture setup and terminal launch or cleanup failures.

The [HTTP probe](../tests/mtp/graph_http.py#L213)<!-- prefix: "def main():" --> sends `POST /v1/completions` and `POST /v1/chat/completions` with SSE. Compare output IDs, finish reasons, usage, and verifier counters. Correlate CUPTI device activity with each request interval. Check the 127/128 page crossing, the last eligible pair, the 5,119/5,120 straddle, and positions beyond the bound. Check the eager warmup, full acceptance, sampled rejection, stop, EOS, the final single-output round, and reset with replacement. Also check a higher depth with its step budget limited to two target rows. Use the [constraint checks](#constrained-verification) for constrained dispatch.

### Sampled verification

Compare a finite target-law oracle with target-draw verification at depths 0–4. Cover first rejection, partial and full acceptance, stop, and budget exhaustion. Check sampler filters separately against a CPU row oracle. On a reserved B200, compare K0 and K1–K4 with fixed seeds, the same prefill policy, and the same prompt and budget. Use `sglang-generate --top-logits 32` to inspect score order and margins at the first differing ID. Compare full BF16 rows on shared histories to distinguish a score change from a draw or state error.

With `CHECKPOINT` set to the pinned model directory and the reserved GPU visible as device zero, run each depth with the same options:

```sh
for depth in 0 1 2 3 4; do
  sglang-generate --checkpoint "$CHECKPOINT" --device 0 \
    --prompt-ids 48,25,220,16,10,16,28 --max-new-tokens 21 \
    --temperature 1.8 --top-p 0.89 --top-k 56 --min-p 0.04 \
    --seed -9223372036854775808 --top-logits 32 --mtp-steps "$depth" \
    > "sampled-k${depth}.json" || break
done
```

Require 21 outputs and `length` at each depth. Compare `output_token_ids` and `step_top_logits`. Check K0 has no draft counters and positive depths record drafting and rejection.

Run both HTTP text routes at configured capacities one and four. Match JSON and SSE IDs, usage, and finish reasons for the same seeded cases. Exercise different seeds, greedy peers, rejection, EOS, and a one-token budget. Trace draft counts and dispatch separately from timing.

### Constrained verification

Follow the [positive-depth constraint checks](output-constraints.md#positive-depth-checks) for the HTTP and C++ routes. Check target state and continuation as for other verifier changes.

### Measured trade-offs

Layer-major verification, tiled prefill seeding, and speculative depth each changed the best choice by prompt and workload. Early rejection can leave already computed rows unused and require GDN replay. Measure the intended prompt mix, depth, verifier, and seeding against K0 before changing a default.
