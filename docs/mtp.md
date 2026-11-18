# Native greedy MTP

Multi-token prediction (MTP) drafts candidates and uses the target model to verify them in greedy order.

## Responsibilities and sources of truth

| Part | Responsibility | Source |
| --- | --- | --- |
| Generator and server | Set the proposal limit. The server admits one active request with MTP. | [Generator options](../apps/sglang-generate.cpp#L34)<!-- prefix: "void usage(s" --> · [Server limit](../apps/sglang-server.cpp#L84)<!-- prefix: "if (argument" --> |
| MTP controller | Seed the draft, collect proposals, and publish greedy target outputs. | [Controller](../src/decoder.cpp#L228)<!-- prefix: "bool prefill" --> |
| Target verifier | Process causal rows by layer, read the results, and commit the consumed state prefix. | [Forward](../src/decoder_forward.cpp#L249)<!-- prefix: "void Decoder" --> · [State commit](../src/decoder_verify.cpp#L112)<!-- prefix: "void TargetV" --> |

## Runtime path

The default `--mtp-steps 0` keeps serial CUDA Graph decode. Set `--mtp-steps` to `1`–`4` to cap draft proposals per round. Positive steps apply server-wide and require `--max-active 1`. MTP loads its draft weights once and shares the target embedding and LM head. Each positive-budget MTP request owns eager target state and draft MXFP8 KV. [Weights](../src/decoder.cpp#L303)<!-- prefix: "Decoder::Dec" --> · [Request](../src/decoder.cpp#L322)<!-- prefix: "DecoderFinis" -->

Positive-depth MTP prefills the target in tiles of up to 128 rows using exact SIMT projections under either FP8 prefill policy. Single rows and unavailable scratch use row-wise prefill. For a prompt of length `L`, the draft seeds `L-1` positions starting at zero. At position `p`, it combines the final-normalized target row with the embedding of prompt token `p+1`, including across tile boundaries. [Prefill](../src/decoder.cpp#L228)<!-- prefix: "bool prefill" --> · [Row alignment](../src/decoder_prefill.cpp#L379)<!-- prefix: "void Decoder" -->

The tiled seed normalizes and concatenates the two inputs, then applies the draft input and KV projections. It normalizes and rotates keys and writes MXFP8 keys and values at their target positions. Tiled seeding skips query projection, draft attention, MLP, and scoring. The row-wise fallback uses the same shift and runs the unscored draft forward. [Draft seed](../src/decoder_mtp.cpp#L309)<!-- prefix: "void MtpDraf" -->

The target scores the last prompt row to emit the first greedy token. If generation continues, that token and the last normalized target row start the draft round at `L-1`. Later proposals use the previous draft's final-normalized row and prediction. Each proposal readback synchronizes the stream. [Proposals](../src/decoder.cpp#L161)<!-- prefix: "bool propose" -->

Each round starts with the latest emitted token as the first target input. The verifier takes successive proposals as the next inputs. It evaluates up to five rows, capped by the remaining output budget and the proposal count plus one. Each target layer advances those causal rows before the next layer begins. Eligible FP8 and BF16 projections share weight reads across rows. One target result synchronization collects all winners and optional full BF16 logit rows. [Round](../src/decoder.cpp#L204)<!-- prefix: "std::optiona" --> · [Readback](../src/decoder_verify.cpp#L93)<!-- prefix: "void TargetV" -->

The controller emits matching proposals in order. The first mismatch emits a target correction and ends the round. Full acceptance can emit a target bonus if the stop rule and budget allow. The correction or bonus becomes the next target input if generation continues. The controller rewinds draft KV before verification and refreshes continuing draft state with final-normalized target rows from the accepted prefix. [Result consumption](../src/decoder.cpp#L182)<!-- prefix: "std::optiona" --> · [Draft rewind](../src/decoder_mtp.cpp#L361)<!-- prefix: "void MtpDraf" -->

The verifier writes tentative GDN state into an alternate bank. Full consumption swaps that bank into the request. Partial consumption replays saved inputs for the consumed prefix from the prior bank, then swaps. Zero consumption preserves the initial bank. The target KV prefix advances by the consumed rows. Later forwards bound reads by position and overwrite rejected slots before they become visible. [Causal attention](../src/decoder_forward.cpp#L143)<!-- prefix: "cudaError_t " --> · [State commit](../src/decoder_verify.cpp#L112)<!-- prefix: "void TargetV" -->

## Constraints and failure behavior

Verifier storage is allocated after the initial greedy token if generation continues. Its row capacity is the lesser of the proposal limit plus one and the initial output budget minus one. Five rows add **167,355,584 logical device bytes (159.60 MiB)** per request, independent of prompt length. This includes the alternate GDN bank, saved replay inputs, and row workspaces. Existing target and draft storage is separate. Five-row pinned results use 40 bytes for winners or 2,483,240 bytes with `--top-logits` enabled. [Sizing](../src/decoder.cpp#L268)<!-- prefix: "const int ro" --> · [Storage](../src/decoder_verify.cpp#L24)<!-- prefix: "TargetVerifi" -->

MTP releases its private prefill workspace before queuing the first score. Requests with a cancellation callback wait for each prefill tile and poll between tiles. Stop, budget, and cancellation bound output consumption. Cancellation can leave already computed rows unused. Allocation or execution errors fail the request. Request cleanup retains storage and pinned results while stream completion is uncertain. [Cleanup](../src/decoder_resources.cpp#L160)<!-- prefix: "CleanupError" -->

## Verification

Run the [public MTP acceptance procedure](../tests/mtp/README.md) with both release binaries, the pinned checkpoint, and a reserved B200. Require zero exit status. The runner compares explicit depths 0–4 with default MTP off. It requires exact IDs, finish reasons, and ordered top-64 BF16 logits. It checks budget, stop, page crossing, counters, serial HTTP tokens and usage, and batched MTP rejection. At depth four, it checks retirement and continuation after a decode disconnect. To isolate seeding parity, pass a row-wise seed generator with the same target verifier to `--reference-generate`. [Runner](../tests/mtp/run_native.py#L1)<!-- prefix: "\"\"\"Opt-in re" -->

For seeding changes, compare paired RMSNorm and key-only preprocessing with the single-row operations. Compare full target logits, final-normalized rows, committed target KV and GDN state, and seeded draft KV byte for byte. Check later draft and verification logits and IDs. Check continued output. Exercise 1, 127, 128, 129, and 257 prompt positions and a longer multi-tile case. Check both requested FP8 policies with MTP enabled. Keep the MTP-off graph control. Disconnect during prefill before the first output. Check queued-request parity after retirement. [Paired RMSNorm](../src/text_ops.cu#L952)<!-- prefix: "cudaError_t " --> · [Key preprocessing](../src/text_attention_gqa.cu#L113)<!-- prefix: "cudaError_t " -->

For verifier changes, compare each full BF16 logit row and final-normalized row with the row-wise eager target. Compare committed target KV and GDN bytes after zero, partial, and full acceptance. Check draft KV and continuation after rewind. Exercise stop, cancellation, and recoverable result-sync failure. Check decoder reuse with MTP enabled and with graphs. Include a KV page crossing and the exact GQA score-capacity boundary. Keep source, binary, checkpoint, fixtures, and raw results together. Run [performance measurement](performance.md#native-serial-mtp) separately.

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
