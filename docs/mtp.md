# Native MTP

Multi-token prediction (MTP) verifies draft candidates in order against the
target model. It supports greedy decoding and sampling.

## Responsibilities and sources of truth

| Part | Responsibility | Source |
| --- | --- | --- |
| Generator and server | Set the proposal limit and pass request options to the decoder. | [Generator options](../apps/sglang-generate.cpp#L31)<!-- prefix: "void usage(s" --> · [Server dispatch](../apps/sglang-server.cpp#L294)<!-- prefix: "BatchService" --> |
| MTP round | Draft proposals and commit target outputs in order. | [Proposals](../src/decoder_verify.cpp#L163)<!-- prefix: "bool MtpRoun" --> · [Commit](../src/decoder_verify.cpp#L215)<!-- prefix: "bool MtpRoun" --> |
| Target verifier | Process causal rows by layer and commit each request's consumed prefix. | [Forward](../src/decoder_forward.cpp#L772)<!-- prefix: "bool Decoder" --> · [State commit](../src/decoder_verify.cpp#L396)<!-- prefix: "void TargetV" --> |

## Runtime path

`sglang-server` and `sglang-generate` default to MTP depth four (K4). Use `--mtp-steps 0` to disable MTP (K0). Values `1`–`4` limit draft proposals per round. Both commands accept `--mtp-k` as an alias. C++ `DecoderOptions` defaults to K0. [Depth option](../apps/sglang-server.cpp#L149)<!-- prefix: "} else if (o" --> · [Server default](../apps/sglang-server.cpp#L37)<!-- prefix: "int mtp_step" --> · [Generator default](../apps/sglang-generate.cpp#L123)<!-- prefix: "arguments.op" --> · [C++ options](../include/sglang/decoder.h#L36)<!-- prefix: "int mtp_step" -->

At capacities 1–32, the server supports MTP for text completions, text-only chat, and native prompts, including sampling and [constraints](output-constraints.md). If you omit the depth option, jobs with [token logprobs](native-generate.md#token-logprobs) use K0. At explicit positive depth, these requests return HTTP 400. [Media requests](multimodal-decoder.md) use K0. A [prefix-cache-enabled batch](prefix-cache.md#runtime-path) uses exact prompt projections. `Decoder::generate` uses one batch lane for MTP. [Logprob policy](../include/sglang/server/batch_service.h#L126)<!-- prefix: "bool support" --> · [Single-prompt selection](../apps/server/batch_service.cpp#L894)<!-- prefix: "job->options" --> · [Batch selection](../apps/server/batch_service.cpp#L597)<!-- prefix: "job->options" -->

MTP loads draft weights once and shares the target embedding and head.
Each request owns its target state, draft MXFP8 KV, and verifier storage.
C++ callers set depth per lane. Exact batches can [restore paired prefixes](prefix-cache.md#checkpoint-compatibility-and-resume).
[Weights](../src/decoder.cpp#L430)<!-- prefix: "Decoder::Dec" --> · [Admission](../src/decoder_batch.cpp#L313)<!-- prefix: "int DecoderB" -->

On B200, loaded draft weights support [INT8 draft scoring](native-model-info.md#int8-draft-head).

Eligible B200 MTP prompt tiles hold up to 1,024 rows. Prefix reuse and other devices cap MTP tiles at 128. Draft seeding pairs each final-normalized target row with the next prompt token embedding. Single rows or unavailable scratch use scalar prefill. [Tile cap](../src/decoder_batch.cpp#L644)<!-- prefix: "const int li" --> · [Row alignment](../src/decoder_prefill.cpp#L896)<!-- prefix: "chunk.seed_d" -->

On the B200 Tensor Core text path, draft seed FC can use WMMA for prompts of at least 4,096 tokens and aligned chunks of 64–128 rows. BF16 inputs accumulate in FP32 and round to BF16. Accumulation order can change draft KV and proposals. Ineligible chunks use SIMT seed FC.

Tiled and row-wise seeding normalize and concatenate both inputs before the draft input and KV projections. Both normalize and rotate keys and write MXFP8 KV at the target positions. Both skip query projection, draft attention, MLP, and scoring. [Draft seed](../src/decoder_mtp.cpp#L597)<!-- prefix: "void MtpDraf" -->

The target scores the last prompt row and selects the first token. If generation continues, that token and the last normalized target row start the draft round at `L-1`. Later proposals use the previous draft's final-normalized row and prediction. The proposal loop synchronizes once per wave across the group. [Proposal loop](../src/decoder_verify.cpp#L163)<!-- prefix: "bool MtpRoun" -->

Eligible B200 draft waves select [grouped online attention](native-model-info.md#draft-attention) automatically.

Each target round starts with the latest committed token. Proposals supply subsequent inputs. The batch limits proposals to `K`, remaining outputs minus one, and assigned target rows minus one. Each proposal can add one target row. A lane assigned one verification row makes zero proposals. One remaining output uses ordinary eager decode. Zero budget finishes without allocating model state or running a forward. See [target row accounting](batch-graphs.md#runtime-path). [Batch assignment](../src/decoder_batch.cpp#L577)<!-- prefix: "if (results[" --> · [Zero-budget admission](../src/decoder_batch.cpp#L770)<!-- prefix: "if (const in" -->

The batch processes lanes with reserved verification rows in ascending slot order, up to four lanes per group. All groups share one step budget. The verifier interleaves causal rows by position and processes up to 20 rows per layer. With target Tensor Cores enabled, four eager five-row lanes batch the BF16 GDN gate projections, convolution, recurrence, and finish. Each lane keeps its own causal state and uses the same commit rules. Each group reads back its winners and commits before the next group starts. `Decoder::generate` can read raw BF16 rows for top-logit output. [Grouping](../src/decoder_batch.cpp#L706)<!-- prefix: "while (next_" --> · [Shared verification](../src/decoder_forward.cpp#L739)<!-- prefix: "bool Decoder" --> · [Direct adapter](../src/decoder.cpp#L661)<!-- prefix: "SerialOutput" -->

The controller commits matching proposals in order. The first mismatch commits a target correction and ends the round. It leaves later proposals unconsumed. Full acceptance can commit a target bonus within the stop rule and output budget. A stop match takes precedence over length. A correction or bonus becomes the next target input if generation continues. Draft KV rewinds to the first target position of the round. Continuing lanes rebuild only the accepted draft KV rows from target-normalized rows. [Round commit](../src/decoder_verify.cpp#L215)<!-- prefix: "bool MtpRoun" -->

The verifier writes tentative GDN state into an alternate bank. Full consumption swaps that bank into the request. Partial consumption replays saved inputs for the consumed prefix from the prior bank, then swaps. Zero consumption preserves the initial bank. The target KV prefix advances by the consumed rows. Later forwards bound reads by position and overwrite rejected slots before they become visible. [Causal attention](../src/decoder_forward.cpp#L412)<!-- prefix: "cudaError_t " --> · [State commit](../src/decoder_verify.cpp#L396)<!-- prefix: "void TargetV" -->

C++ callers consume `token`, then the first `extra_token_count` entries of `extra_tokens`. `extra_token` mirrors the first extra for depth-one callers. The HTTP worker receives tokens after the batch completes its state commits. Token callbacks deliver in order and check cancellation between tokens. A [`/generate` batch stream](native-generate.md#json-and-sse) or a scalar stream with [native text stops](native-generate.md#text-stops) can include several IDs from one step in a cumulative event. A native text match clips the published IDs through the matching token. A blocked response holds its slot while ready peers advance. A reset or failed write retires the affected request's active lanes and drops their undelivered tokens. [Batch API](../include/sglang/decoder_batch.h#L79)<!-- prefix: "class Decode" --> · [Token delivery](../include/sglang/server/output_mailbox.h#L30)<!-- prefix: "inline bool " --> · [Step delivery](../apps/server/batch_service.cpp#L645)<!-- prefix: "bool BatchSe" -->

### Target verification graph

The default [graph policy](batch-graphs.md) permits verification replay for one unconstrained text lane with two target rows. Both positions must fit inline GQA scores. The last eligible pair is 5,118 and 5,119 (zero-based). Replay and eager verification share ordered exact GQA and two-row projection grouping. The first eligible group runs eagerly. The next captures the verifier on the batch stream. [Group dispatch](../src/decoder_forward.cpp#L792)<!-- prefix: "if (graph &&" --> · [Row eligibility](../src/decoder_verify.cpp#L297)<!-- prefix: "bool TargetV" --> · [Capture](../src/decoder_forward.cpp#L793)<!-- prefix: "std::exchang" -->

Each launch refreshes tokens, positions, KV page-table descriptors and cache pointers, GDN bank pointers, and GQA score bindings. The executable reuses fixed batch-owned scratch. Draft proposals, sampling, result readback, acceptance, and state commit run outside it. C++ reports replayed slots in `verification_graph_slots`. The HTTP `graph_slots` trace reports K0 replay. [Bindings](../src/decoder_verify.cpp#L67)<!-- prefix: "void Verific" --> · [Slot mask](../include/sglang/decoder_batch.h#L49)<!-- prefix: "std::uint32_" --> · [HTTP trace](../include/sglang/server/batch_trace.h#L49)<!-- prefix: "if (stats.gr" -->

Direct `Decoder::generate` verification runs eagerly. Ineligible batch groups run eagerly. A K4 round with four proposals has five target rows and runs eagerly. With one output token left, the lane uses ordinary eager decode. Other groups in the step can still qualify. `--eager` disables replay for all groups.

### Sampled MTP

An item with active [output penalties](output-penalties.md), [logit bias](serving/openai-logprobs.md#logit-bias), or a positive [output minimum](output-penalties.md#minimum-output) uses K0. Eligible peers in the same batch keep their configured depth. [Target-only selection](../include/sglang/decoder.h#L70)<!-- prefix: "bool require" -->

The draft proposes greedy IDs. The target uses the [normal sampler](text-serving.md#sampling) on the first score and each verification row. It keys each draw by the request's resolved seed and the row's prospective output index. Discarded rows leave subsequent output indices available for recomputation on the corrected history. The same index drives the draw on that new row. [Verifier sampler](../src/decoder_verify.cpp#L337)<!-- prefix: "void TargetV" -->

For a valid causal row with filtered target law `P`, the target draws `T` from `P`. It emits `T`, reusing proposal state when `T` equals the proposed ID. A fixed proposal `d` is therefore accepted with probability `P(d)`. Ratio-based speculative sampling uses `min(1, P(d)/Q(d))` and a residual draw on rejection. Here the target draw itself supplies the correction. Full acceptance permits a target bonus under the usual budget and stop rules.

Each emitted token has the target law for its causal row. A fixed seed gives the same sampled IDs as K0 when both paths reach identical effective rows and use the same sampler and output indices. Prefill, forward precision, and batch grouping can change the logits and therefore the draws.

### Constrained MTP

The first target output uses the [ordinary constraint selection](output-constraints.md#selection-and-finish). Each round starts from the committed cursor. The draft chooses the highest-scoring allowed ID and advances a private cursor. Each row uses its own remaining output budget. If a proposal completes the constraint, the target verifies it and skips the bonus row. [Round cursor](../src/decoder_verify.cpp#L95)<!-- prefix: "void MtpCons" -->

The target filters each verified row with the allowed IDs for that candidate prefix. It uses the ordinary greedy rule or samples with the prospective output index. Only emitted target IDs advance the live cursor. A mismatch or finish discards later proposals and their cursor states. A correction or bonus can also complete the constraint. Identical score rows, cursors, sampler options, and output indices give identical target decisions. [Target selection](../src/decoder_verify.cpp#L347)<!-- prefix: "void TargetV" -->

## Constraints and failure behavior

The verifier reserves up to `K+1` rows, limited to the initial output budget minus one. The step cap can also limit the computed row count. The [resident batch path](batch-graphs.md#resident-memory-admission) allocates verifier storage on admission. Other paths allocate it before the first verification. Five rows add **167,355,584 logical device bytes (159.60 MiB)** per request, independent of prompt length. This includes the alternate GDN bank, saved replay inputs, and row workspaces. Existing target and draft storage is separate. Five-row pinned results use 40 bytes for winners or 2,483,240 bytes with serial `--top-logits`. The batch API rejects top logits. Constrained verification also reserves up to 16,384 pinned host bytes per row for IDs or 31,040 for a wide mask. [Constraint staging](../src/decoder_verify.cpp#L80)<!-- prefix: "MtpConstrain" --> · [Row bound](../src/decoder_resources.cpp#L342)<!-- prefix: "int reserved" --> · [Storage](../src/decoder_verify.cpp#L250)<!-- prefix: "TargetVerifi" -->

MTP releases temporary prefill scratch after processing each batch step. Direct generation checks its cancellation callback while advancing the lane. Batch cancellation can leave work from a completed round unused by the client. A failed speculative batch step fails the current jobs. If the stream drains, the service can admit replacements. If completion is uncertain, the batch is disabled and retains its storage. [Step recovery](../src/decoder_batch.cpp#L844)<!-- prefix: "} catch (con" --> · [Cleanup](../src/decoder_resources.cpp#L225)<!-- prefix: "CleanupError" -->

For a recoverable verification graph setup failure, eager fallback requires ended capture, a drained stream, and successful handle cleanup. This disables only the verifier executable. A binding upload failure follows speculative step recovery without replaying the failed round. A verification launch error or unconfirmed capture cleanup disables the batch without reevaluating the target. Resources stay live until safe cleanup. [Graph recovery](../src/decoder_resources.cpp#L610)<!-- prefix: "const auto f" --> · [Launch](../src/decoder_resources.cpp#L687)<!-- prefix: "const auto l" -->

If a constrained draft row has no valid winner, drafting stops and the target can still select an output. For an invalid winner on a consumed constrained row, direct generation throws `DecoderConstraintError`. The batch reports `constraint_failed` and retires the lane. For an invalid unconstrained sample, direct generation throws `DecoderSamplingError`. The batch reports `sampling_failed`. HTTP drops the failed lane's unpublished step tokens. Rows discarded after a mismatch or stop have no effect. [Batch consumption](../src/decoder_batch.cpp#L68)<!-- prefix: "void finish_" --> · [HTTP failure](../apps/server/batch_service.cpp#L351)<!-- prefix: "if (step.sam" -->

## Verification

Run the [MTP acceptance procedure](../tests/mtp/README.md) with both release binaries, the pinned checkpoint, and a reserved B200. Require exit status zero. Use explicit K0 with matching target dispatch and prompt tiles as the control for IDs, finish reasons, and BF16 logits. Compare default generation with explicit K4. The runner checks budgets, stops, page crossings, counters, and capacity-one HTTP, including K4 disconnect recovery. At wider capacities, check SSE, finish reasons, usage, and full joint target dispatch. For seeding parity, pass a row-wise seed generator with the same target verifier to `--reference-generate`. [Runner](../tests/mtp/run_native.py#L1)<!-- prefix: "\"\"\"Opt-in re" -->

For seeding changes, compare paired RMSNorm and key-only preprocessing with the single-row operations. Compare full target logits, final-normalized rows, committed target KV and GDN state, and seeded draft KV byte for byte where the seed arithmetic agrees. Check later draft and verification logits and continued output. Exercise 1, 127, 128, 129, and 257 prompt positions and a longer multi-tile case. Cover the cache-enabled exact prefill path and the B200 Tensor Core path. Keep the MTP-off graph control. Disconnect during prefill before the first output. Check queued-request parity after retirement. [Paired RMSNorm](../src/text_normalization.cu#L439)<!-- prefix: "cudaError_t " --> · [Key preprocessing](../src/text_attention_gqa.cu#L128)<!-- prefix: "cudaError_t " -->

For verifier changes, match full BF16 logits and final-normalized rows against row-wise eager execution. Match target KV and GDN after zero, partial, and full acceptance. Check draft KV and continuation after rewind. Exercise mixed depths, budget stops, KV page crossings, and the exact GQA score-capacity boundary. Reset an active lane and check retained and replacement streams. Require zero request bytes after retirement. Check recovery after a recoverable result-sync failure. Count real launches for K0 and positive-K graph-flag controls. Keep source, binary, checkpoint, fixtures, and raw results together.

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

### Measured verifier trade-offs

These verifier studies predate tiled MTP prefill seeding.

On B200 GPU3, the serial W1 HTTP comparison used one 128-token prompt and 128 output tokens. It used two warmups and five measured requests per verifier and depth. At depth four (K4), the median rate fell from 50.696 to 45.677 output tokens/s (9.9%) with the layer-major verifier. The paired generator runs accepted 75 of 206 proposals (36.4%) with either verifier. HTTP depth zero (K0) had the highest median rate in both arms. This native comparison used the W1 request bytes. Official leaderboard timing follows the separate [frozen protocol](../benchmarks/streaming/README.md#frozen-measurement-protocol).

On B200 GPU6, an independent C++ library probe timed `Decoder::generate` on the prior and selected runtimes. With models already loaded, it measured three requests per depth, prompt, and verifier. Each of the two held-out prompts generated 64 tokens. At K4, the code prompt accepted 49 of 60 proposals. Its median fell from 740.2 to 593.6 ms with layer-major verification. The Spanish field prompt accepted 36 of 108 proposals. Its median rose from 800.2 to 1,009.6 ms. The selected runtime's K0 medians were 544.3 ms for code and 562.1 ms for Spanish.

Early rejection can leave already computed rows unused and require GDN replay. These results support measuring each intended prompt mix and depth against both the prior verifier and K0.

### Measured seeding trade-offs

The tiled-seeding release comparison used the pinned checkpoint and exact FP8 policy on B200 GPU0. Both positive-depth builds used the same layer-major target verifier. The generator used held-out prefixes, 64 output tokens, one warmup, and three measured requests per arm. Model load was outside the timed interval. IDs and finish reasons matched across arms. Draft counters matched between the depth-one (K1) builds.

| Prompt tokens | Row-wise K1 median (ms) | Tiled K1 median (ms) | K0 graph median (ms) | Tiled K1 latency above K0 |
| ---: | ---: | ---: | ---: | ---: |
| 5 | 1,916.1 | 1,068.4 | 498.7 | 114.2% |
| 129 | 2,143.4 | 1,003.6 | 890.6 | 12.7% |
| 695 | 11,267.7 | 2,890.7 | 2,762.2 | 4.7% |

The five-token K1 samples varied from 587.6 to 2,017.2 ms with row-wise seeding and 610.4 to 1,411.8 ms with tiled seeding. Their ranges overlap. The 129- and 695-token cases reduced median K1 latency by 53.2% and 74.3%, respectively.

A separate C++ probe timed prefill through the first target score once per path and length. At 129 tokens, it measured 1,290.3 ms row-wise and 391.0 ms tiled. At 695 tokens, it measured 10,545.3 ms row-wise and 2,146.7 ms tiled. See `artifacts/mtp-seeding-promotion-prep/gpu-all.Oftu8wNv/inputs.sha256`, `duration.log`, and `timing-summary.json` for the inputs and raw measurements.

### Measured continuous MTP trade-offs

The earlier public-runtime study timed eager batches on B200 GPU5 with tracing off. Each arm used one warmup and three measured bursts with 64 output IDs per request. K0 used the published server. Timed and separately traced bursts matched IDs and finish reasons. The table reports median aggregate output IDs/s.

| Width | Held-out fixture | K0 | K1 | K2 | K3 | K4 | K4 accepted/drafted |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | Code | 110.86 | 93.89 | 112.99 | 94.77 | 55.50 | 96/118 |
| 2 | Field | 105.50 | 90.41 | 84.58 | 71.91 | 70.11 | 72/214 |
| 4 | Code | 128.32 | 112.15 | 134.19 | 120.80 | 94.20 | 192/236 |
| 4 | Field | 124.79 | 122.89 | 71.48 | 81.64 | 60.74 | 144/428 |

At width four, K2 increased code throughput by 4.6% and reduced field throughput by 42.7% against K0. On the field fixture, K0 led at both widths. At width four, K0 recorded 606.60 MiB of live request storage. Separately, peak K0 scratch was 4,991,360 bytes for code and 6,239,200 bytes for field. The peak request counters recorded 1,218.03 MiB at K1 and 1,248.73 MiB at K4. The shared draft model added 477,199,744 device bytes separately. These are held-out results from before standalone loading and long-score graphs. See `artifacts/mtp-continuous-public/timing-final-public.swigsmps/performance.json`.
