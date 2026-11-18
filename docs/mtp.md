# Native greedy MTP

Multi-token prediction (MTP) drafts candidates and uses the target model to verify them in greedy order.

## Responsibilities and sources of truth

| Part | Responsibility | Source |
| --- | --- | --- |
| Generator and server | Set the proposal limit. The server admits one active request with MTP. | [Generator options](../apps/sglang-generate.cpp#L34)<!-- prefix: "void usage(s" --> · [Server limit](../apps/sglang-server.cpp#L84)<!-- prefix: "if (argument" --> |
| MTP controller | Seed the draft, collect proposals, and publish greedy target outputs. | [Controller](../src/decoder.cpp#L228)<!-- prefix: "DecoderFinis" --> |
| Target verifier | Process causal rows by layer, read the results, and commit the consumed state prefix. | [Forward](../src/decoder_forward.cpp#L234)<!-- prefix: "void Decoder" --> · [State commit](../src/decoder_verify.cpp#L112)<!-- prefix: "void TargetV" --> |

## Runtime path

The default `--mtp-steps 0` keeps serial CUDA Graph decode. Set `--mtp-steps` to `1`–`4` to cap draft proposals per round. Positive steps apply server-wide and require `--max-active 1`. MTP loads its draft weights once and shares the target embedding and LM head. Each positive-budget MTP request owns eager target state and draft MXFP8 KV. [Weights](../src/decoder.cpp#L281)<!-- prefix: "Decoder::Dec" --> · [Request](../src/decoder.cpp#L300)<!-- prefix: "DecoderFinis" -->

MTP prefills the target one position at a time under both FP8 prefill policies. During seeding, the draft receives the final-normalized target row at `p` and the shifted prompt token at `p+1`. The target emits the first greedy token. That token and the last normalized target row start the draft round. Later proposals use the previous draft's final-normalized row and prediction. Each proposal readback synchronizes the stream. [Prefill](../src/decoder.cpp#L228)<!-- prefix: "DecoderFinis" --> · [Proposals](../src/decoder.cpp#L161)<!-- prefix: "bool propose" -->

Each round starts with the latest emitted token as the first target input. The verifier takes successive proposals as the next inputs. It evaluates up to five rows, capped by the remaining output budget and the proposal count plus one. Each target layer advances those causal rows before the next layer begins. Eligible FP8 and BF16 projections share weight reads across rows. One target result synchronization collects all winners and optional full BF16 logit rows. [Round](../src/decoder.cpp#L204)<!-- prefix: "std::optiona" --> · [Readback](../src/decoder_verify.cpp#L93)<!-- prefix: "void TargetV" -->

The controller emits matching proposals in order. The first mismatch emits a target correction and ends the round. Full acceptance can emit a target bonus if the stop rule and budget allow. The correction or bonus becomes the next target input if generation continues. The controller rewinds draft KV before verification and refreshes continuing draft state with final-normalized target rows from the accepted prefix. [Result consumption](../src/decoder.cpp#L182)<!-- prefix: "std::optiona" --> · [Draft rewind](../src/decoder_mtp.cpp#L321)<!-- prefix: "void MtpDraf" -->

The verifier writes tentative GDN state into an alternate bank. Full consumption swaps that bank into the request. Partial consumption replays saved inputs for the consumed prefix from the prior bank, then swaps. Zero consumption preserves the initial bank. The target KV prefix advances by the consumed rows. Later forwards bound reads by position and overwrite rejected slots before they become visible. [Causal attention](../src/decoder_forward.cpp#L128)<!-- prefix: "cudaError_t " --> · [State commit](../src/decoder_verify.cpp#L112)<!-- prefix: "void TargetV" -->

## Constraints and failure behavior

Verifier storage is allocated after the initial greedy token if generation continues. Its row capacity is the lesser of the proposal limit plus one and the initial output budget minus one. Five rows add **167,355,584 logical device bytes (159.60 MiB)** per request, independent of prompt length. This includes the alternate GDN bank, saved replay inputs, and row workspaces. Existing target and draft storage is separate. Five-row pinned results use 40 bytes for winners or 2,483,240 bytes with `--top-logits` enabled. [Sizing](../src/decoder.cpp#L249)<!-- prefix: "const int ro" --> · [Storage](../src/decoder_verify.cpp#L24)<!-- prefix: "TargetVerifi" -->

Stop, budget, and cancellation bound output consumption. Cancellation can leave already computed rows unused. Allocation or execution errors fail the request. Request cleanup retains storage and pinned results while stream completion is uncertain. [Cleanup](../src/decoder_resources.cpp#L160)<!-- prefix: "CleanupError" -->

## Verification

Run the [public MTP acceptance procedure](../tests/mtp/README.md) with both release binaries, the pinned checkpoint, and a reserved B200. Require zero exit status. The runner compares explicit depths 0–4 with default MTP off using exact IDs, finish reasons, and ordered top-64 BF16 logits. It checks budget, stop, page crossing, counters, serial HTTP tokens and usage, and batched MTP rejection. At depth four, it checks retirement and continuation on the same server after a disconnect. The optional prior generator checks row-wise verifier parity for IDs, logits, and MTP counters. [Runner](../tests/mtp/run_native.py#L1)<!-- prefix: "\"\"\"Opt-in re" -->

For verifier changes, compare each full BF16 logit row and final-normalized row with the row-wise eager target. Compare committed target KV and GDN bytes after zero, partial, and full acceptance. Check draft KV and continuation after rewind. Exercise stop, cancellation, and recoverable result-sync failure. Check decoder reuse with MTP enabled and with graphs. Include a KV page crossing and the exact GQA score-capacity boundary. Keep source, binary, checkpoint, fixtures, and raw results together. Run [performance measurement](performance.md#native-serial-mtp) separately.

### Measured verifier trade-offs

On B200 GPU3, the serial W1 HTTP comparison used one 128-token prompt and 128 output tokens. It used two warmups and five measured requests per verifier and depth. At depth four (K4), the median rate fell from 50.696 to 45.677 output tokens/s (9.9%) with the layer-major verifier. The paired generator runs accepted 75 of 206 proposals (36.4%) with either verifier. HTTP depth zero (K0) had the highest median rate in both arms. This native comparison used the W1 request bytes. Official leaderboard timing follows the separate [frozen protocol](../benchmarks/streaming/README.md#frozen-measurement-protocol).

On B200 GPU6, an independent C++ library probe timed `Decoder::generate` on the prior and selected runtimes. With models already loaded, it measured three requests per depth, prompt, and verifier. Each of the two held-out prompts generated 64 tokens. At K4, the code prompt accepted 49 of 60 proposals. Its median fell from 740.2 to 593.6 ms with layer-major verification. The Spanish field prompt accepted 36 of 108 proposals. Its median rose from 800.2 to 1,009.6 ms. The selected runtime's K0 medians were 544.3 ms for code and 562.1 ms for Spanish.

Early rejection can leave already computed rows unused and require GDN replay. These results support measuring each intended prompt mix and depth against both the prior verifier and K0.
