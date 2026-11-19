# Native MTP

Multi-token prediction (MTP) drafts candidates and verifies them in order against the target model. It supports greedy decoding and seeded or unseeded sampling.

## Responsibilities and sources of truth

| Part | Responsibility | Source |
| --- | --- | --- |
| Generator and server | Set the proposal limit and select serial or batch execution. | [Generator options](../apps/sglang-generate.cpp#L34)<!-- prefix: "void usage(s" --> · [Server dispatch](../apps/sglang-server.cpp#L256)<!-- prefix: "if (argument" --> |
| MTP controllers | Seed draft KV, collect proposals, and publish target outputs. | [Serial controller](../src/decoder.cpp#L340)<!-- prefix: "bool prefill" --> · [Batch rounds](../src/decoder_batch.cpp#L827)<!-- prefix: "void Decoder" --> |
| Target verifier | Process causal rows by layer and commit each request's consumed prefix. | [Forward](../src/decoder_forward.cpp#L379)<!-- prefix: "void Decoder" --> · [State commit](../src/decoder_verify.cpp#L146)<!-- prefix: "void TargetV" --> |

## Runtime path

Set `--mtp-steps` or `--mtp-k` to `1`–`4` to enable MTP and limit draft proposals per round. Positive depths work at server capacities 1–32 with prefix reuse off. The server applies the depth to unconstrained text and text-only chat, including sampled requests. HTTP [constrained requests](text-serving.md#finite-choice-constraints) and [media requests](multimodal-decoder.md) use K0. The generator and width-one server use the serial controller. A batch step that executes work for an MTP lane runs eagerly. [Depth option](../apps/sglang-server.cpp#L105)<!-- prefix: "} else if (o" --> · [HTTP selection](../apps/server/batch_service.cpp#L561)<!-- prefix: "if (media.em" -->

MTP loads its draft weights once and shares the target embedding and LM head. Each positive-budget MTP request owns target state, draft MXFP8 KV, and any verifier storage it allocates. C++ batch callers can set different proposal limits per lane. Positive-depth lanes skip [prefix reuse](prefix-cache.md). [Weights](../src/decoder.cpp#L419)<!-- prefix: "Decoder::Dec" --> · [Admission](../src/decoder_batch.cpp#L449)<!-- prefix: "int DecoderB" -->

Positive-depth MTP prefills the target in tiles of up to 128 rows using exact SIMT projections under either FP8 prefill policy. Single rows and unavailable scratch use row-wise prefill. For a prompt of length `L`, the draft seeds `L-1` positions starting at zero. At position `p`, it combines the final-normalized target row with the embedding of prompt token `p+1`, including across tile boundaries. [Prefill](../src/decoder.cpp#L340)<!-- prefix: "bool prefill" --> · [Row alignment](../src/decoder_prefill.cpp#L417)<!-- prefix: "void Decoder" -->

The tiled seed normalizes and concatenates the two inputs, then applies the draft input and KV projections. It normalizes and rotates keys and writes MXFP8 keys and values at their target positions. Tiled seeding skips query projection, draft attention, MLP, and scoring. The row-wise fallback uses the same shift and runs the unscored draft forward. [Draft seed](../src/decoder_mtp.cpp#L309)<!-- prefix: "void MtpDraf" -->

The target scores the last prompt row and selects the first token. If generation continues, that token and the last normalized target row start the draft round at `L-1`. Later proposals use the previous draft's final-normalized row and prediction. Serial execution synchronizes after each draft proposal. Each batch group synchronizes once per proposal wave. [Serial proposals](../src/decoder.cpp#L264)<!-- prefix: "bool propose" --> · [Batch proposals](../src/decoder_batch.cpp#L883)<!-- prefix: "void Decoder" -->

Each target round starts with the latest committed token. Proposals supply subsequent inputs. Serial verification evaluates up to five rows, bounded by the remaining output budget and the configured step cap. That cap also limits serial prompt tiles. A batch lane verifies one row plus its proposals. The proposal count is bounded by `K`, remaining output minus one, and reserved target rows minus one. A lane with more than one output left verifies with zero proposals if it receives one target row. A lane with one output left uses ordinary eager decode. A zero-budget batch lane holds no model state and finishes without a forward. See [target row accounting](batch-graphs.md#runtime-path). [Serial cap](../src/decoder.cpp#L382)<!-- prefix: "const int pr" --> · [Batch assignment](../src/decoder_batch.cpp#L712)<!-- prefix: "if (results[" --> · [Zero-budget admission](../src/decoder_batch.cpp#L534)<!-- prefix: "if (options." -->

The batch processes lanes with reserved verification rows in ascending slot order, up to four lanes per group. All groups share one step budget. In each group, the verifier interleaves causal rows by position and processes up to 20 rows per layer. Eligible FP8 and BF16 projections share weight reads in groups of up to five rows. Attention and GDN use each request's own causal state. Each group reads back its winners and completes its state commits before the next group starts. Serial verification can also read full BF16 logits. [Grouping](../src/decoder_batch.cpp#L835)<!-- prefix: "while (next_" --> · [Group commits](../src/decoder_batch.cpp#L874)<!-- prefix: "TargetVerifi" --> · [Serial round](../src/decoder.cpp#L307)<!-- prefix: "std::optiona" -->

The controller commits matching proposals in order. The first mismatch commits a target correction and ends the round. It leaves later proposals unconsumed. Full acceptance can commit a target bonus within the stop rule and output budget. A stop match takes precedence over length. A correction or bonus becomes the next target input if generation continues. Draft KV rewinds to the first target position of the round. Continuing lanes rebuild the accepted prefix from target-normalized rows. [Serial consumption](../src/decoder.cpp#L285)<!-- prefix: "std::optiona" --> · [Batch consumption](../src/decoder_batch.cpp#L73)<!-- prefix: "void finish_" -->

The verifier writes tentative GDN state into an alternate bank. Full consumption swaps that bank into the request. Partial consumption replays saved inputs for the consumed prefix from the prior bank, then swaps. Zero consumption preserves the initial bank. The target KV prefix advances by the consumed rows. Later forwards bound reads by position and overwrite rejected slots before they become visible. [Causal attention](../src/decoder_forward.cpp#L179)<!-- prefix: "cudaError_t " --> · [State commit](../src/decoder_verify.cpp#L146)<!-- prefix: "void TargetV" -->

C++ callers consume `token`, then the first `extra_token_count` entries of `extra_tokens`. `extra_token` mirrors the first extra for depth-one callers. The HTTP worker publishes after the batch completes its state commits. It delivers tokens in order and checks cancellation between tokens. A blocked response holds its slot while ready peers advance. A reset or failed write retires the affected lane and drops its undelivered tokens. [Batch API](../include/sglang/decoder_batch.h#L67)<!-- prefix: "class Decode" --> · [Publication](../apps/server/batch_service.cpp#L72)<!-- prefix: "bool emit_st" -->

### Sampled MTP

The draft proposes greedy IDs. The target uses the [normal sampler](text-serving.md#sampling) on the first score and each verification row. It keys each draw by the request's resolved seed and the row's prospective output index. Discarded rows leave subsequent output indices available for recomputation on the corrected history. The same index drives the draw on that new row. [Verifier sampler](../src/decoder_verify.cpp#L100)<!-- prefix: "void TargetV" -->

For a valid causal row with filtered target law `P`, the target draws `T` from `P`. It emits `T`, reusing proposal state when `T` equals the proposed ID. A fixed proposal `d` is therefore accepted with probability `P(d)`. Ratio-based speculative sampling uses `min(1, P(d)/Q(d))` and a residual draw on rejection. Here the target draw itself supplies the correction. Full acceptance permits a target bonus under the usual budget and stop rules.

Each emitted token has the target law for its causal row. A fixed seed gives the same sampled IDs as K0 when both paths reach identical effective rows and use the same sampler and output indices. Prefill, forward precision, and batch grouping can change the logits and therefore the draws.

## Constraints and failure behavior

The verifier reserves up to `K+1` rows, limited to the initial output budget minus one. The serial step cap can reduce this number further. The [resident batch path](batch-graphs.md#resident-memory-admission) allocates verifier storage on admission. Other paths allocate it before the first verification. Five rows add **167,355,584 logical device bytes (159.60 MiB)** per request, independent of prompt length. This includes the alternate GDN bank, saved replay inputs, and row workspaces. Existing target and draft storage is separate. Five-row pinned results use 40 bytes for winners or 2,483,240 bytes with serial `--top-logits`. The batch API rejects top logits. [Row bound](../src/decoder_resources.cpp#L310)<!-- prefix: "int reserved" --> · [Serial sizing](../src/decoder.cpp#L382)<!-- prefix: "const int pr" --> · [Storage](../src/decoder_verify.cpp#L27)<!-- prefix: "TargetVerifi" -->

Serial MTP releases prefill scratch before the first target score. With a cancellation callback, it waits for each tile and polls between tiles. Batch cancellation can leave work from a completed round unused by the client. A failed speculative batch step fails the current jobs. If the stream drains, the service can admit replacements. If completion is uncertain, the batch is disabled and retains its storage. [Step recovery](../src/decoder_batch.cpp#L985)<!-- prefix: "drain(\"Drain" --> · [Cleanup](../src/decoder_resources.cpp#L211)<!-- prefix: "CleanupError" -->

If an invalid sample reaches consumption, serial generation throws `DecoderSamplingError`. The batch reports `sampling_failed` and retires that lane. HTTP drops that lane's unpublished step tokens. Discarded speculative rows after a mismatch or stop have no effect on the result. [Batch consumption](../src/decoder_batch.cpp#L73)<!-- prefix: "void finish_" --> · [HTTP failure](../apps/server/batch_service.cpp#L276)<!-- prefix: "if (step.sam" -->

## Verification

Run the [public MTP acceptance procedure](../tests/mtp/README.md) with both release binaries, the pinned checkpoint, and a reserved B200. Require zero exit status. The runner compares generator depths 0–4 with default K0 for exact IDs, finish reasons, and ordered top-64 BF16 logits. It checks budget, stop, page crossing, counters, serial HTTP, and serial K4 disconnect recovery. All 12 batch width/depth combinations check exact SSE, finish reasons, usage, and a full joint target dispatch. To isolate seeding parity, use a row-wise seed generator with the same target verifier as `--reference-generate`. [Runner](../tests/mtp/run_native.py#L1)<!-- prefix: "\"\"\"Opt-in re" -->

For seeding changes, compare paired RMSNorm and key-only preprocessing with the single-row operations. Compare full target logits, final-normalized rows, committed target KV and GDN state, and seeded draft KV byte for byte. Check later draft and verification logits and IDs. Check continued output. Exercise 1, 127, 128, 129, and 257 prompt positions and a longer multi-tile case. Check both requested FP8 policies with MTP enabled. Keep the MTP-off graph control. Disconnect during prefill before the first output. Check queued-request parity after retirement. [Paired RMSNorm](../src/text_normalization.cu#L286)<!-- prefix: "cudaError_t " --> · [Key preprocessing](../src/text_attention_gqa.cu#L128)<!-- prefix: "cudaError_t " -->

For verifier changes, match full BF16 logits and final-normalized rows against row-wise eager execution. Match target KV and GDN after zero, partial, and full acceptance. Check draft KV and continuation after rewind. Exercise mixed depths, stop at the budget, KV page crossings, and the exact GQA score-capacity boundary. Reset an active lane and check retained and replacement streams. Require zero request bytes after retirement. Check recovery after a recoverable result-sync failure. Count real launches for K0 and positive-K graph-flag controls. Keep source, binary, checkpoint, fixtures, and raw results together.

At server capacity 32, check K1 with more than four verifying lanes. Check sparse slot selections. Match committed IDs, finish reasons, and usage against same-policy controls. Check the combined target-row count against the step cap. Include a lane that receives one verification row and makes zero proposals. Check final-token eager decode, zero-budget completion, and reset recovery. Run [performance measurement](performance.md#native-mtp) separately.

### Sampled verification

Compare a finite target-law oracle with target-draw verification at depths 0–4. Cover first rejection, partial and full acceptance, stop, and budget exhaustion. Check sampler filters separately against a CPU row oracle. On a reserved B200, compare K0 and K1–K4 with fixed seeds, the same prefill policy, and the same prompt and budget. Use `sglang-generate --top-logits 32` to inspect score order and margins at the first differing ID. Compare full BF16 rows on shared histories to distinguish a score change from a draw or state error.

With `CHECKPOINT` set to the pinned model directory and the reserved GPU visible as device zero, run each depth with the same options:

```sh
for depth in 0 1 2 3 4; do
  sglang-generate --checkpoint "$CHECKPOINT" --device 0 --fp8-prefill exact \
    --prompt-ids 48,25,220,16,10,16,28 --max-new-tokens 21 \
    --temperature 1.8 --top-p 0.89 --top-k 56 --min-p 0.04 \
    --seed -9223372036854775808 --top-logits 32 --mtp-steps "$depth" \
    > "sampled-k${depth}.json" || break
done
```

Require 21 outputs and `length` at each depth. Compare `output_token_ids` and `step_top_logits`. Check K0 has no draft counters and positive depths record drafting and rejection.

Run both HTTP text routes at configured capacities one and four. Match JSON and SSE IDs, usage, and finish reasons for the same seeded cases. Exercise different seeds, greedy peers, rejection, EOS, and a one-token budget. Trace separately to check draft counts and eager dispatch on steps that execute MTP work.

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
