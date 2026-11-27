# Native MTP

Multi-token prediction (MTP) verifies draft candidates in order against the
target model. It supports greedy decoding and sampling.

## Responsibilities and sources of truth

| Part | Responsibility | Source |
| --- | --- | --- |
| Generator and server | Set the proposal limit and pass request options to the decoder. | [Generator options](../apps/sglang-generate.cpp#L57)<!-- prefix: "void usage(s" --> · [Server dispatch](../apps/sglang-server.cpp#L410)<!-- prefix: "BatchService" --> |
| MTP round | Draft proposals and commit target outputs in order. | [Proposals](../src/decoder_verify.cpp#L157)<!-- prefix: "bool MtpRoun" --> · [Commit](../src/decoder_verify.cpp#L279)<!-- prefix: "bool MtpRoun" --> |
| Target verifier | Process causal rows by layer and commit each request's consumed prefix. | [Forward](../src/decoder_verify.cpp#L885)<!-- prefix: "bool Decoder" --> · [State commit](../src/decoder_verify.cpp#L466)<!-- prefix: "void TargetV" --> |

## Runtime path

`sglang-server` and `sglang-generate` default to MTP depth four (K4). Use `--mtp-steps 0` to disable MTP (K0). Values `1`–`4` limit draft proposals per round. Both commands accept `--mtp-k` as an alias. C++ `DecoderOptions` defaults to K0. [Depth option](../apps/sglang-server.cpp#L152)<!-- prefix: "} else if (o" --> · [Server default](../apps/sglang-server.cpp#L51)<!-- prefix: "int mtp_step" --> · [Generator default](../apps/sglang-generate.cpp#L150)<!-- prefix: "arguments.op" --> · [C++ options](../include/sglang/decoder.h#L62)<!-- prefix: "int mtp_step" -->

At capacities 1–32, text, media, and native generation use configured MTP depth. Greedy [token logprobs](native-generate.md#token-logprobs) can use MTP and score committed target rows. Sampling and [constraints](output-constraints.md) use their own verification paths. Pointwise scores and target-only options use K0. `Decoder::generate` uses one batch lane for MTP. [Effective depth](../src/decoder_batch.cpp#L148)<!-- prefix: "if (options." --> · [Verifier scores](../src/decoder_verify.cpp#L466)<!-- prefix: "void TargetV" -->

MTP loads draft weights once and shares the target embedding and head.
Each request owns its target state, draft MXFP8 KV, and verifier storage.
C++ callers set depth per lane. Positive-depth lanes can [restore paired prefixes](prefix-cache.md#checkpoint-compatibility-and-resume).
[Weights](../src/decoder.cpp#L458)<!-- prefix: "Decoder::Dec" --> · [Admission](../src/decoder_batch.cpp#L135)<!-- prefix: "int DecoderB" -->

On B200, loaded draft weights support [INT8 draft scoring](native-model-info.md#int8-draft-head).

Eligible B200 MTP prompt tiles hold up to 1,024 rows. Other devices cap tiles at 128. Draft seeding pairs each final-normalized target row with the next prompt token's embedding. A media lane supplies its own embedding and rotary coordinates. Single rows or unavailable scratch use scalar prefill. [Tile cap](../src/runtime/execution/decoder_execution_policy.cpp#L25)<!-- prefix: "int DecoderE" --> · [Draft input](../src/decoder_prefill.cpp#L75)<!-- prefix: "seed.seed_dr" -->

B200 Tensor Core prefill can prefer TMA and `tcgen05` for draft seed FC on prompts of at least 4,096 tokens. Each FC call needs 64–128 effective rows, 10,240 input and 5,120 output channels, and 16-byte-aligned input and weights. Other shapes and input alignments use SIMT row GEMV. [Prefill selection](../src/decoder_prefill.cpp#L73)<!-- prefix: "const bool w" --> · [FC dispatch](../src/text/draft_seed_fc.cu#L124)<!-- prefix: "cudaError_t " -->

TMA requires 4-byte-aligned output, a matching image and tensor-map encoder, and enough shared memory. Unaligned output or unsupported preflight selects the 32×64 WMMA tile. CUDA errors, including stream/device mismatch and launch failure, propagate without retry. SIMT accumulation can change draft KV and proposals. [TMA preflight](../src/text/draft_seed_fc_sm100.cu#L195)<!-- prefix: "std::optiona" --> · [Submission](../src/text/draft_seed_fc_sm100.cu#L241)<!-- prefix: "return submi" --> · [WMMA tile](../src/text/draft_seed_fc.cu#L24)<!-- prefix: "constexpr in" -->

Tiled and row-wise seeding normalize and concatenate both inputs before the draft input and KV projections. Both normalize and rotate keys and write MXFP8 KV at the target positions. Both skip query projection, draft attention, MLP, and scoring. [Draft seed](../src/decoder_mtp.cpp#L686)<!-- prefix: "void MtpDraf" -->

For a multi-row seed tile on B200, each request prepares FP8 activations in
slices of up to 128 rows on the prefill stream. One exact FP8 kernel projects
K and V in separate grid banks for the tile. One key normalization and RoPE
launch and one MXFP8 append follow. Grouped prefill seeds each request
separately. Single-row tiles and other devices project and append each slice
Seed scratch comes from the [existing prefill workspace](native-prefill-workspace.md#seed-scratch).
[Slice preparation](../src/decoder_mtp.cpp#L730)<!-- prefix: "for (int off" --> ·
[Paired projection](../src/text_ops.cu#L748)<!-- prefix: "cudaError_t " --> ·
[Grid banks](../include/sglang/text/prefill_fp8_exact.cuh#L209)<!-- prefix: "__global__ v" --> ·
[Grouped seed](../src/decoder_prefill.cpp#L75)<!-- prefix: "seed.seed_dr" -->

The target scores the last prompt row and selects the first token. If generation continues, that token and the last normalized target row start the draft round at `L-1`. Later proposals use the previous draft's final-normalized row and prediction. The proposal loop synchronizes the stream before reading winners and again before reading BF16 retry results. [Proposal loop](../src/decoder_verify.cpp#L157)<!-- prefix: "bool MtpRoun" -->

Eligible B200 draft lanes select [grouped online attention](native-model-info.md#draft-attention) automatically.

### Draft winner readback

On B200, two to four unconstrained lanes with mapped pinned storage share two argmax launches. The first reduces tiles. The second writes each winner directly to the mapped record. Both follow per-lane tie and nonfinite rules. [Dispatch](../src/decoder_mtp.cpp#L289)<!-- prefix: "void MtpDraf" --> · [Launcher](../src/text_ops.cu#L821)<!-- prefix: "cudaError_t " --> · [Selection](../src/text_ops.cu#L377)<!-- prefix: "__device__ T" -->

Capture can reuse mappings. Eager requests can allocate them on demand. Singletons, constrained lanes, and other devices use per-lane selection and copy. If any mapping allocation fails or capture prevents it, the group uses per-lane selection. Other CUDA errors propagate. [Mapping](../src/decoder_mtp.cpp#L269)<!-- prefix: "bool MtpDraf" --> · [Allocation](../src/decoder_verify.cpp#L559)<!-- prefix: "std::shared_" --> · [Per-lane selection](../src/decoder_mtp.cpp#L289)<!-- prefix: "void MtpDraf" -->

### Target round

Each target round starts with the latest committed token. Proposals supply subsequent inputs. The batch limits proposals to `K`, remaining outputs minus one, and assigned target rows minus one. The row grant also respects [checkpoint boundaries](prefix-cache.md#checkpoint-boundaries). A one-row verification makes no proposals. One remaining output uses ordinary eager decode. Zero budget without scoring allocates no request state. [Batch assignment](../src/decoder_batch.cpp#L327)<!-- prefix: "if (results[" --> · [Zero-budget finish](../src/decoder_batch.cpp#L175)<!-- prefix: "if (options." -->

The batch forms packets of up to four lanes in slot order under one step budget. Ordinary packets interleave causal rows by position, verify up to 20 rows, and commit separately. Verification packets of two to twenty rows can score through one packed BF16 head projection, falling back to the grouped path on decline; operator errors propagate. Serial generation can read raw BF16 logits. Eligible eager greedy rounds use the [certified target head](native-model-info.md#certified-tensor-core-target-head). [Target round](../src/decoder_verify.cpp#L691)<!-- prefix: "bool Decoder" --> · [Packed head](../include/sglang/text/bf16_head_projection.h#L15)<!-- prefix: "std::optiona" -->

On B200, a batch with draft weights and certified vocabulary scratch needs capacity eight or more and a step cap of at least 40. Adjacent eager packets combine into 8, 16, or 32 lanes with 40, 80, or 160 rows; two ranks stop at sixteen. Each lane needs five assigned rows, at least five remaining outputs, and greedy unconstrained K4 without raw logits. Sparse slots preserve packet order. The [wide FP8 projections](fp8-prefill-projections.md#multi-lane-target-verification) share a pass. The [certified head](native-model-info.md#certified-tensor-core-target-head) can score pairs of full packets. [Draft projections](native-model-info.md#grouped-draft-projections) and KV repairs can use complete 8- or 16-lane groups. [Long GQA](singleton-gqa-decode.md#tensor-core-reads-across-long-k4-packets) can share a launch across packets. [Compact GDN](gdn-prefill.md#grouped-target-verification) can share 80- or 160-row launches. Short drafts fall back to eligible aligned eight-lane groups or ordinary packets. [Planner](../include/sglang/text/decoder_batch_plan.h#L55)<!-- prefix: "inline WideM" -->

The controller commits matching proposals in order. The first mismatch commits a target correction and ends the round. It leaves later proposals unconsumed. Full acceptance can commit a target bonus within the stop rule and output budget. A stop match takes precedence over length. A correction or bonus becomes the next target input if generation continues. Draft KV rewinds to the first target position of the round. Continuing lanes rebuild only the accepted draft KV rows from target-normalized rows. [Round commit](../src/decoder_verify.cpp#L279)<!-- prefix: "bool MtpRoun" -->

The verifier writes tentative GDN state into an alternate bank. Full consumption swaps that bank into the request. Partial consumption replays saved inputs for the consumed prefix from the prior bank, then swaps. Zero consumption preserves the initial bank. The target KV prefix advances by the consumed rows. Later forwards bound reads by position and overwrite rejected slots before they become visible. [Causal attention](../include/sglang/runtime/execution/decoder_forward_layers.h#L634)<!-- prefix: "cudaError_t " --> · [State commit](../src/decoder_verify.cpp#L466)<!-- prefix: "void TargetV" -->

C++ callers consume `token`, then the first `extra_token_count` entries of `extra_tokens`. `extra_token` mirrors the first extra for depth-one callers. HTTP workers receive output after the batch commits state. Per-token callbacks deliver IDs in order and check cancellation between tokens. Native [step delivery](native-generate.md#text-stops) can send several IDs in one cumulative event. Eligible per-token writers use [bounded decode output](http-server.md#bounded-decode-output). [Batch API](../include/sglang/decoder_batch.h#L92)<!-- prefix: "class Decode" --> · [Token delivery](../include/sglang/server/detail/output_mailbox.h#L30)<!-- prefix: "inline bool " --> · [Step delivery](../apps/server/batch_service.cpp#L573)<!-- prefix: "bool BatchSe" -->

### Target verification graph

TP1 [replay](batch-graphs.md) supports one unconstrained text lane with two target rows. Both positions must fit inline GQA scores. The last eligible pair is 5,118 and 5,119 (zero-based). Replay and eager verification share two-row projection grouping. Replay uses bound GQA dispatch, including the [split-KV position check](singleton-gqa-decode.md#split-kv-target-attention) when the split policy is active. The first eligible group runs eagerly. The next captures the verifier on the batch stream. [Group dispatch](../src/decoder_verify.cpp#L909)<!-- prefix: "if (graph &&" --> · [Row eligibility](../src/decoder_verify.cpp#L386)<!-- prefix: "bool TargetV" --> · [Capture](../src/decoder_verify.cpp#L910)<!-- prefix: "std::exchang" -->

Each launch refreshes tokens, positions, KV page tables and pointers, GDN banks, and GQA score bindings. The executable reuses fixed batch-owned scratch. Draft proposals, sampling, result readback, acceptance, and state commit run outside it. C++ reports replayed slots in `verification_graph_slots`. The HTTP `graph_slots` trace reports K0 replay. [Bindings](../src/decoder_verify.cpp#L38)<!-- prefix: "void Verific" --> · [Slot mask](../include/sglang/decoder_batch.h#L53)<!-- prefix: "std::uint32_" --> · [HTTP trace](../include/sglang/server/detail/batch_trace.h#L51)<!-- prefix: "if (stats.gr" -->

TP1 K4 full rounds, direct `Decoder::generate`, and ineligible groups run eagerly. One output remaining uses ordinary decode. Other groups can still qualify. [TP2 verification replay](batch-graphs.md#tp2-target-verification-graph) has separate eligibility. `--eager` disables both.

### Sampled MTP

An item with active [output penalties](output-penalties.md), [logit bias](serving/openai-logprobs.md#logit-bias), or a positive [output minimum](output-penalties.md#minimum-output) uses K0. Eligible peers in the same batch keep their configured depth. [Target-only selection](../include/sglang/decoder.h#L104)<!-- prefix: "bool require" -->

The draft proposes greedy IDs. The target uses the [normal sampler](text-serving.md#sampling) on the first score and each verification row. It keys each draw by the request's resolved seed and the row's prospective output index. Discarded rows leave subsequent output indices available for recomputation on the corrected history. The same index drives the draw on that new row. [Verifier sampler](../src/decoder_verify.cpp#L444)<!-- prefix: "void TargetV" -->

For a valid causal row with filtered target law `P`, the target draws `T` from `P`. It emits `T`, reusing proposal state when `T` equals the proposed ID. A fixed proposal `d` is therefore accepted with probability `P(d)`. Ratio-based speculative sampling uses `min(1, P(d)/Q(d))` and a residual draw on rejection. Here the target draw itself supplies the correction. Full acceptance permits a target bonus under the usual budget and stop rules.

Each emitted token has the target law for its causal row. A fixed seed gives the same sampled IDs as K0 when both paths reach identical effective rows and use the same sampler and output indices. Prefill, forward precision, and batch grouping can change the logits and therefore the draws.

### Constrained MTP

The first target output uses the [ordinary constraint selection](output-constraints.md#selection-and-finish). Each round starts from the committed cursor. The draft chooses the highest-scoring allowed ID and advances a private cursor. Each row uses its own remaining output budget. If a proposal completes the constraint, the target verifies it and skips the bonus row. [Round cursor](../src/decoder_verify.cpp#L67)<!-- prefix: "void MtpCons" -->

The target filters each verified row with the allowed IDs for that candidate prefix. It uses the ordinary greedy rule or samples with the prospective output index. Only emitted target IDs advance the live cursor. A mismatch or finish discards later proposals and their cursor states. A correction or bonus can also complete the constraint. Identical score rows, cursors, sampler options, and output indices give identical target decisions. [Target selection](../src/decoder_verify.cpp#L454)<!-- prefix: "void TargetV" -->

## Constraints and failure behavior

The verifier reserves up to `K+1` rows, limited to the initial output budget minus one. The step cap can also limit the computed row count. The [resident batch path](batch-graphs.md#resident-memory-admission) allocates verifier storage on admission. Other paths allocate it before the first verification. Five rows add **167,355,584 logical device bytes (159.60 MiB)** per request, independent of prompt length. This includes the alternate GDN bank, saved replay inputs, and row workspaces. Existing target and draft storage is separate. Five-row pinned results use 40 bytes for winners or 2,483,240 bytes with serial `--top-logits`. The batch API rejects top logits. Constrained verification also reserves up to 16,384 pinned host bytes per row for IDs or 31,040 for a wide mask. [Constraint staging](../src/decoder_verify.cpp#L52)<!-- prefix: "MtpConstrain" --> · [Row bound](../src/runtime/model/qwen_request.cpp#L69)<!-- prefix: "int reserved" --> · [Storage](../src/decoder_verify.cpp#L321)<!-- prefix: "TargetVerifi" -->

MTP releases temporary prefill scratch after each batch step. Direct generation checks cancellation while advancing and while waiting for [prefix operations](prefix-cache.md#constraints-and-failure-behavior). A failed speculative step fails the current jobs. After a recoverable failure and stream drain, slots with outstanding restores retire when polled to completion. Other slots can admit replacements. If GPU execution remains uncertain, the batch retains its storage and disables further work. [Step recovery](../src/decoder_batch.cpp#L804)<!-- prefix: "} catch (con" --> · [Cleanup](../src/runtime/backend/cuda_resources.cpp#L247)<!-- prefix: "CleanupError" -->

The draft prefix advances after a successful KV append enqueue. If preparation
fails in a later slice, the paired path attempts to append the prepared prefix
before reporting the error. A draft preparation, projection, or append error
marks the draft failed. A target final-normalization error leaves the draft's
failure flag clear if the prepared prefix flush succeeds.
[Append](../src/decoder_mtp.cpp#L709)<!-- prefix: "check(mxfp8_" --> ·
[Target normalization failure](../src/decoder_mtp.cpp#L742)<!-- prefix: "append_prepa" --> ·
[Draft preparation failure](../src/decoder_mtp.cpp#L767)<!-- prefix: "append_prepa" -->

For recoverable TP1 verifier setup failures, eager fallback requires ended capture, a drained stream, and successful handle cleanup. This disables only the verifier executable. A binding upload failure follows speculative step recovery without replaying the failed round. A verification launch error or unconfirmed capture cleanup disables the batch without reevaluating the target. Resources stay live until safe cleanup. [Graph recovery](../src/runtime/backend/cuda_resources.cpp#L568)<!-- prefix: "const auto f" --> · [Launch](../src/runtime/backend/cuda_resources.cpp#L645)<!-- prefix: "const auto l" -->

If a constrained draft row has no valid winner, drafting stops and the target can still select an output. For an invalid winner on a consumed constrained row, direct generation throws `DecoderConstraintError`. The batch reports `constraint_failed` and retires the lane. For an invalid unconstrained sample, direct generation throws `DecoderSamplingError`. The batch reports `sampling_failed`. HTTP drops the failed lane's unpublished step tokens. Rows discarded after a mismatch or stop have no effect. [Batch consumption](../include/sglang/runtime/model/decoder_batch_lane.h#L75)<!-- prefix: "void finish_" --> · [HTTP failure](../apps/server/batch_service.cpp#L284)<!-- prefix: "if (step.sam" -->

## Verification

Run the [MTP acceptance procedure](../tests/mtp/README.md) with both release binaries, the pinned checkpoint, and a reserved B200. Require exit status zero. Match target dispatch and prompt tiles for the K0 control. Compare default generation with explicit K4. Check IDs, finish reasons, usage, and SSE at wider capacities. For seed parity, pass a row-wise seed generator with the same target verifier to `--reference-generate`. [Runner](../tests/mtp/run_native.py#L1)<!-- prefix: "\"\"\"Opt-in re" -->

Compare grouped argmax with per-lane scoring at widths two through four. Require identical BF16 rows, winners, and scores through ties and nonfinite inputs. Trace both launches, mapped writes without copies, per-lane fallback, and BF16 retry in eager and graph modes.

Compare paired RMSNorm, key preprocessing, paired K/V projection, and TMA seed FC with their single-row, per-slice, and WMMA controls. Require byte parity where arithmetic is preserved, including target state, draft KV, and subsequent logits. Cover positions 1, 127, 128, 129, and 257, long and partial tiles, prefix restore, media, and scores. Trace one paired projection and append per multi-row B200 seed tile. Inject the second-slice seed launch failure and verify prefix advance only through the first slice, retirement, and replacement. Check pre-submission WMMA fallback and boundary SIMT.

Check zero, partial, and full acceptance against row-wise verification, including logits, target KV and GDN, rewind, KV crossings, mixed depths, and exact score capacity. At widths 8, 16, and 32, compare dense drafting with four-lane drafting at fixed grants and target dispatch. Compare draft drift separately from committed target state on fixed histories. Exercise full and sparse depths, clean decline, hard errors, and replacement. Use the [wide projection controls](fp8-prefill-projections.md#wide-verifier-checks).

At capacity 32, check K1 with more than four verifying lanes and sparse slot selections. Match committed IDs, finish, and usage. Enforce the target-row cap, including zero proposals on a one-row grant. Check final-token eager decode, zero budget, reset recovery, and zero bytes after retirement. [Full MTP probe matrix](https://github.com/para1lel/sglang.cpp/blob/5ee79416d3e30cd178abab69598d047c4393793d/docs/mtp.md#verification) · [Timing](performance.md#native-mtp)

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

Early rejection can waste precomputed rows and require GDN replay. Measure the intended prompts, depth, verifier, and seed policy against K0 before changing a default.

[Frozen draft-head operator, decode-step, and ranked payload replays](https://github.com/para1lel/sglang.cpp/blob/a5974ba4eb99bd8e8f890c2e7262a27dbc3aace1/docs/mtp.md#measured-trade-offs)
