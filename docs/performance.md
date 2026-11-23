# Single-B200 benchmark agreement

Compare both engines on the same physical B200 and immutable checkpoint. Freeze the workload and controls before timing. Follow the matched K4 driver's arm order for each row. [Paired driver](../benchmarks/paired/README.md#matched-k4-timing-protocol)

## Implementations and controls

| Part | Responsibility | Source |
| --- | --- | --- |
| Independent baseline | Run vLLM in its own environment. Record the checkout, installed wheel and native binary identities, dependency versions, and effective launch settings. | `vllm-project/vllm` at `81a8cb063e6247fe254d7caf7e98a6691d71022a`. |
| Candidate | Run the project binary built by XMake with the default Pixi toolchain. Record source, build options, and executable identity. | [Development contract](development.md). |
| Architectural reference | Interpret service and cache behavior. Any separate SGLang timing needs its own result label. | `sgl-project/sglang` at `f04d886bf79aa3d6237cbb12a3311838eb9f9bd8`. |

Run each engine alone on the selected UUID in its own environment. Record drivers, runtimes, host and client settings, clocks, power, other GPU activity, and software differences.

Match checkpoint, tokenizer, template, processor, weight precision, sampling, cache state, MTP, and media preprocessing. Freeze request bytes, input IDs, arrivals, seeds, and stops. Record dispatch and token counts. Mark missing modes unsupported.

## Scheduler and reporting policy

Use MTP4, an 8,192-token step budget, and a 32-request cap for W1–W6. Enable prefix caching on both engines and measure cold prefixes.

- **Pinned vLLM.** Set `--max-num-batched-tokens 8192 --max-num-scheduled-tokens 8192 --max-num-seqs 32 --async-scheduling`. Verify MTP4 and the effective aligned GDN cache settings at the API and worker.
- **Native server.** Use the Release defaults and pass only `--port` for timing. Verify four prefix-cache slots, MTP4, the scheduler limits, and CUDA graphs. [Server information](native-model-info.md#server-information)

Use the [paired launchers](../benchmarks/paired/README.md). Label other caps exploratory or historical. Preserve historical evidence and explicit operator microbenchmark settings.

Report nondefault settings, effective limits, offered and observed concurrency, input lengths, context and output budgets, and [admission failures](batch-graphs.md#resident-memory-admission) separately from timing. Acceptance requires a fresh matched run with every required mode.

## GQA cache comparison

Use the candidate's [native MXFP8 GQA path](development.md#p2-gqa-cache-contract). For the pinned vLLM comparator, request `kv_cache_dtype="fp8_e4m3"` (`--kv-cache-dtype fp8_e4m3` on the server CLI). Pair the runs only after vLLM dispatches with plain E4M3 GQA KV on the required workload. The pin supports per-tensor K and V scales and defaults both to `1.0` when the checkpoint supplies neither. The local checkpoint index contains no attention scales. Verify the effective scale policy in the run. [vLLM cache setting](https://github.com/vllm-project/vllm/blob/81a8cb063e6247fe254d7caf7e98a6691d71022a/vllm/config/cache.py#L251)<!-- prefix: "CUDA 11.8+ s" --> · [vLLM scale initialization](https://github.com/vllm-project/vllm/blob/81a8cb063e6247fe254d7caf7e98a6691d71022a/vllm/model_executor/layers/quantization/kv_cache.py#L118)<!-- prefix: "elif layer.k" -->

The native path stores E4M3 values with E8M0 scales per 32 channels. The pinned comparator uses E4M3 values with per-tensor K and V scales. These precision policies can produce different logits and greedy IDs. Report cross-engine output and logit comparisons for diagnosis only. Require each engine to satisfy the workload's token counts, finish conditions, and stream contract. Record each engine's requested and effective KV format and the scale dtype, granularity, and source. Record the effective attention Q dtype, any Q scaling, and the attention backend and fallback status. Match and record the GDN state dtypes. The checkpoint FP8 weights and their scales remain a separate control.

## Historical offline smoke

The archived eight-output greedy BF16 KV fixture has a separate offline recipe. Use streaming for W1–W6 TTFT. [Small workload](../benchmarks/vllm/README.md#small-workload) · [Historical BF16 result](../benchmarks/vllm/README.md#historical-bf16-result)

## Named workloads

Lengths are exact post-template tokens per request. Concurrency is the offered client load. Use the frozen requests, arrival pattern, and stop settings. Test normal EOS separately. [Concurrent fixture manifest](../benchmarks/concurrent/fixture-manifest.json#L2)<!-- prefix: "\"seed\": \"sgl" -->

| Workload | Input tokens per request | Output tokens per request | Offered concurrency |
| --- | ---: | ---: | ---: |
| W1 | 128 | 128 | 1 |
| W2 | 2,048 | 512 | 1 |
| W3 | 8,192 | 1,024 | 4 |
| W4 | 32,768 | 1,024 | 16 |
| W5 | 32,768 | 4,096 | 32 |
| W6 | 131,072 | 1,024 | 4 |

Use the [server-defaults protocol](../benchmarks/paired/README.md#native-server-defaults-against-matched-vllm-k4) for `matched-k4-native-defaults-cache-on-v2`. Keep the [warm-prefix diagnostic](../benchmarks/paired/README.md#prefix-reuse-diagnostic) and media variants separate. Freeze media preprocessing, positions, and budgets. Retain failed runs under their original workload names.

## Measurement

**Paired W1–W6 acceptance.** Require a strictly lower native median client makespan on every row with matched settings and work on the same B200. For W1 and W2, run three single-request warmups and nine measured cohorts. For W3–W6, run two concurrent warmups and five measured cohorts. Progress in row order and stop after the first failure. Validate launch and device identity and every stream's request, token counts, finish condition, usage, and drain. Use each recorded profile's native ID checks and evidence requirements. The pinned vLLM arm supplies timing, and its output IDs can vary across valid streams. [Matched K4 timing protocol](../benchmarks/paired/README.md#matched-k4-timing-protocol)

Use one streaming client and monotonic clock. Measure from earliest submit to last valid `[DONE]`. Preconnect lanes and release them with one barrier. Send all bodies before the first valid completion. Limit the submit span to 50 ms. Retain submit times, headers, token events, completion times, token counts, and failures. Exclude [SSE keepalives](http-server.md#completion-streams) and other empty events from token counts.

Report output tokens/s, TTFT, TPOT, end-to-end latency, and whether every lane starts generating before the first completion for diagnosis.

| Metric | Definition for one measured cohort |
| --- | --- |
| Output tokens/s | Sum of emitted output tokens divided by elapsed cohort seconds. Exclude input, padding, and rejected drafts. |
| TTFT (ms) | Submission to first output token, including queueing and prefill. Unavailable for zero output tokens. [Client calculation](../benchmarks/streaming/client.py#L202)<!-- prefix: "\"ttft_ms\": (" --> |
| TPOT (ms/token) | `(last token time − first token time) / (output token count − 1)`, per request. Mark unavailable for fewer than two tokens. |
| End-to-end latency (ms) | Time from request submission to its terminal response. Record stream chunking when multiple tokens arrive together. |
| Requests/s and aggregate tokens/s | Completed requests per cohort second, and total input plus output tokens per cohort second, respectively. |

Exclude startup, model load, JIT compilation, and graph capture from steady measurements. Before every warmup, timed cohort, and separate W1 trace, drain both engines. Send `{}` as `application/json` to each idle reset route. Require success and record cache counters. Start the timer after reset. Require zero cached tokens per request and unchanged prefix-hit counters. Report cohort median and range for makespan and output throughput. Report request p50 and p95 for TTFT, TPOT, and end-to-end latency, with sample counts and the percentile method.

Report one row per engine, workload, and mode with absolute metrics and successful and attempted request counts. Use paired throughput ratios and latency differences for diagnosis. Set any undefined workload margins from baseline repeats before candidate timing.

For offline timing, omit `--top-logits`. Match `--warmup` and `--repeats` to the reference workload. `load_ms` covers model initialization. `request_ms` includes request allocation and graph capture. Allocation metrics include resident request storage on the serial graph path. Keep JSON, invocation, binary hash, and device UUID. [Generator measurement](../apps/sglang-generate.cpp#L216)<!-- prefix: "Measurements" -->

### Operator qualification

Operators may reorder arithmetic within their numerical contract. Compare outputs and state with an independent mathematical reference under the operation's established error criteria. Check causality where the operation uses token history. Check model quality on frozen inputs and shared token histories. Record free-running divergence separately. Require bitwise equality for copied bytes, exact replay, and comparisons that preserve arithmetic.

Use repeated matched measurements to choose the fastest validated implementation for each supported case. Keep one production implementation per case. Use validated fallbacks where shape, device, or available resources prevent the primary route. Keep frozen fixtures and timing gates unchanged.

### Native text completions

Reuse frozen W1–W6 `benchmark` requests, token-ID SSE clients, and timing definitions under the [scheduler policy](#scheduler-and-reporting-policy). Compare cross-engine output IDs for diagnosis. Run [text verification](text-serving.md#verification) separately.

For raw text, freeze UTF-8 requests, checkpoint and tokenizer hashes, serving flags, output budget, and stop IDs. Match prompt IDs across engines. Time cold tokenizer load separately from steady tokenization. Record streaming, usage, and diagnostic ID settings. Text can lag IDs. A text event can span several IDs, and an ID can produce no text. Count tokens from usage or diagnostic IDs. Calculate TTFT and TPOT from ID timestamps when available. Otherwise, mark them unavailable. Report time to first visible text separately. Give raw text its own result rows.

### Serial W1 and W2 comparisons

The historical GPU3 serial protocol uses its own lane limits, throughput ratio, and p95 latency ceilings. Keep these diagnostics under their original settings and labels. [Frozen GPU3 protocol](../benchmarks/streaming/README.md#frozen-measurement-protocol)

### Fused FP8 decode

Match native control IDs and full BF16 logits on W1 and W2 with the same checkpoint, precision, and GQA path. Check eager decode and fresh and reused graphs. Exercise exact and direct GQA at their dispatch bounds. [Singleton attention](singleton-gqa-decode.md) · [Prefill](development.md#chunked-prefill-verification) · [Streaming](development.md#streaming-verification) · [MTP](mtp.md#verification)

Verify fused projections for eligible singleton decode and legacy groups of two through five, including mixed decode and single-row prefill. Check [packed K0 decode](batch-graphs.md#packed-paged-gqa-decode) by bucket. Compare shape and alignment fallbacks against separate native ops. Match execution mode, context, and KV capacity for timing. Record row modes, group sizes, captures, and replays.

Compare fused SwiGLU BF16 results with identical and disjoint gate and up output spans. Require `cudaErrorInvalidValue` before launch for identical gate and up spans in fallback, partial output overlaps, and output overlaps with inputs.

Qualify the Release build with the serial and concurrent fixtures under the applicable launch policy.

### Exact decode bandwidth

Match the published native control's BF16 projections, E4M3 payloads, and FP32 scales byte for byte. Include nonfinite inputs and shape and alignment fallbacks. Trace fresh and reused graphs. Compare full BF16 logits and greedy IDs on shared W1/W2 histories within each policy. Run the [MTP checks](mtp.md#verification).

For serial HTTP comparisons, use `--max-active 1 --mtp-steps 0` and compare within each prefill policy. Apply the [serial W1 and W2 criteria](#serial-w1-and-w2-comparisons).

Record visible lengths and KV capacity. Report ordered fallback timing separately if measured.

### Chunked prefill

Run the [prefill acceptance checks](development.md#chunked-prefill-verification) before timing. For exact prefill, match full BF16 logits and committed KV and GDN state with the row-wise control. Time device prompt processing with CUDA events through the operation before the first LM head. Measure HTTP TTFT from submission to the first token. Record tile sizes, tails, active lanes, and decode token gaps during mixed steps.

Generator `peak_device_allocation_bytes` counts logical model, request, and scratch bytes. Batch `workspace_bytes` tracks the largest active prefill workspace. `request_bytes` counts live request storage after each step. These counters exclude idle backing. Live reservations can exceed the pool release thresholds. [Prefill workspace accounting](native-prefill-workspace.md#constraints-and-failure-behavior)

Record pool used and reserved bytes after synchronization. Require zero live scratch after retirement, including recovered free-enqueue failures. Report peak HBM, headroom, lane mix, and KV capacity.

Use a fresh process for cold allocator timing. Warm pools with a multirow request. Record tiles, lanes, and order. Report model load, tokenizer warmup, graph state, and prefix state separately.

Qualify each production configuration under its applicable protocol.

### FP8 tensor-core prefill

For the native control, use `--mtp-steps 0`. Record commands, binary and source
hashes, toolchain, driver, and dispatch. Trace projection shapes and fallback.

Run the [prefill checks](development.md#chunked-prefill-verification) for exact scoring and cache storage. Compare Tensor Core projections with an independent reference under predefined tolerances. Match fallback and exact projections bitwise. Cover full chunks, tails, mixed steps, and unsupported dispatch. Check repeated tokens and finish reasons.

For arithmetic-preserving prefill fusions, match full BF16 logits, committed KV and GDN state, and staged BF16 and FP8 values bitwise. Hold dispatch and tile schedule fixed. Include raw MXFP8 exponents 0 and 255. Cover single rows, two- and four-row tiles, page crossings, positive MTP depth, and graph and eager decode. [Fused activation quantization](fp8-prefill-projections.md#fused-activation-quantization)

Use the [GQA verification criteria and measurements](flash-prefill.md#verification) for flash, ordered fallback, and protected single rows.

For cross-policy diagnostics, compare full BF16 logits on identical histories and prefill boundaries. Use shared teacher-forced tokens after greedy divergence. Report logit errors, ranks, top-two margins, labeled-token negative log likelihood, free-running IDs, and first divergence.

Compare with pinned vLLM on identical histories for model diagnostics. Record its execution mode and KV and GDN policies. Record how diagnostic rows were selected. Use frozen fixtures for workload-level results.

Run [serial tile checks](text-serving.md#serial-prefill-verification) before timing. Match BF16 scores across tiles on identical inputs when arithmetic stays the same. Follow the [prefill protocol](#chunked-prefill). Report native deltas and gates per policy.

### Native concurrent workloads

Verify offered and observed concurrency, streaming, and decode width and replay under the applicable scheduler policy. Keep historical K0 diagnostics with their original controls. [Streaming verification](development.md#streaming-verification) · [Batch timing](batch-graphs.md#timing) · [Wide decode checks](long-paged-decode.md#verification)

### Native MTP

Run the [MTP checks](mtp.md#verification) before timing. Use `--mtp-steps 0` for the eager control at the same width. Measure K0 replay separately. Match prompt IDs, arrivals, budgets, stop policy, and outputs. Record binary and checkpoint hashes, GPU UUID, dispatch, and device and pinned host bytes. Disable tracing and `--top-logits`.

To isolate layer-major verification, compare with row-wise MTP at the same depth. To isolate tiled seeding, keep the verifier, exact precision, and depth fixed. Time target prefill and draft KV seeding through workspace release, before the first LM head. Record HTTP TTFT and request latency separately. Include short, tile-boundary, and long prompts. Record tile sizes, pool state, and output length.

Alternate paired arms on one reserved B200. Match warmups and repeats. Report samples, medians, ranges, and paired ratios. The W3-MTP variant uses four 8,192-token prompts with 1,024 outputs each. Its K1 start exceeds the verification graph's visible limit. [W3-MTP fixture](../benchmarks/variants/README.md) · [Graph limit](mtp.md#target-verification-graph)

Positive depth adds counters for the first measured request. `mtp_drafted_tokens` counts proposals read back. `mtp_accepted_tokens` counts matching proposals appended to the output. `mtp_rounds` counts rounds that produced at least one proposal. Count committed `output_token_ids` including first, correction, and bonus tokens. Report accepted/drafted and accepted/rounds for positive denominators. [JSON fields](../apps/sglang-generate.cpp#L242)<!-- prefix: "void write_r" --> · [Proposal loop](../src/decoder_verify.cpp#L190)<!-- prefix: "bool MtpRoun" --> · [Generator counters](../src/decoder_batch.cpp#L981)<!-- prefix: "result.mtp_r" -->

Use `--trace-batches` at any capacity in a separate run. Check `accepted ≤ compared ≤ drafted` and `drafted ≤ rounds × K`. Count the first mismatch as compared. Computed rows can exceed consumed rows after mismatch or stop. In normal completion, consumed decode rows equal committed outputs minus the initial token. Zero or one output requires zero drafts. [Consumption](../src/decoder_verify.cpp#L293)<!-- prefix: "bool MtpRoun" -->

Batch traces sum verifier lanes and computed rows across all groups in the step. Per slot, `drafted` counts proposals read back and `accepted` counts matching outputs committed before publication. `verified` counts computed target rows. `replayed` counts rows replayed on partial consumption. A one-output eager `boundary` records zero speculative counts. Client-visible output can be shorter after cancellation. [Batch trace](../include/sglang/server/batch_trace.h#L12)<!-- prefix: "inline void " -->

See the [verifier storage bound](mtp.md#constraints-and-failure-behavior). The generator adds peak request, prefill, and selection bytes to model bytes. Batch `peak_request_bytes` includes prefill scratch or live request storage before retirement. Record peak HBM separately. [Generator accounting](../src/decoder_batch.cpp#L978)<!-- prefix: "result.peak_" --> · [Batch accounting](../src/decoder_batch.cpp#L901)<!-- prefix: "current.peak" -->

## Correctness, memory, and diagnosis

Performance acceptance requires [native implementation correctness](development.md#verification), repeatability, and comparable memory and feature settings. Run correctness checks separately when instrumentation changes timing. Apply the remaining proposed checks at their applicable milestones.

- **Memory.** Record HBM after load and peak HBM during prefill and decode. Budget weights and scales, GQA KV pages, GDN live state and checkpoints, activations, workspaces, graph pools, vision weights and buffers, and runtime allocations. State the measurement method, capacity, and headroom for each input length and concurrency.
- **Cache.** Record reusable prefix tokens divided by eligible prompt tokens, the initial cache state, page occupancy, evictions, and recurrent checkpoint restores and replay lengths. Compare cold and primed continuations with fresh execution. Check reclamation after cancellation and memory pressure.
- **MTP.** Check target verification and recurrent rollback.
- **Sampling.** Run the [sampling checks](text-serving.md#verification) and [sampled MTP checks](mtp.md#sampled-verification). Report rates separately by width, depth, and arm order.
- **Finite choices.** Run the [constraint checks](text-serving.md#finite-choice-verification). Record grammar, candidate counts, effective dispatch, and [batch scratch](batch-graphs.md#runtime-path) separately from frozen workloads.
- **Multimodal.** Verify image and video processing, visual token positions, and reference output behavior on each fixture. Include client-visible preprocessing and transport in end-to-end measurements. Record server processing and vision-encoder time separately when observable.
- **Profiling.** Record GPU utilization, kernel, JIT, scheduler, and queue times, plus collection methods. Profile separately if tracing changes throughput. Collect [native scheduler samples](prefix-cache.md#native-http-controls) during diagnostics. [Full vLLM verification](../benchmarks/variants/verify.py#L390)<!-- prefix: "def verify(r" --> needs pinned worker metadata and PyTorch traces. Link optimizations to bottlenecks, correctness, and results.

Retain the run manifest, commands, environment, fixture hashes, raw samples, and failures in ignored `artifacts/` storage. A missing counter is unavailable. A missing baseline, failed correctness check, or changed workload leaves that performance claim open.

## Matched K4 evidence

The cache-on `matched-k4-native-defaults-cache-on-v2` campaign failed W1 on 2026-11-23. Both engines ran on the same B200 with MTP4, an 8,192-token step budget, and a 32-request cap. Each ran three warmups and nine timed requests. Each cohort began after an acknowledged idle reset. All 25 streams, including the separate native trace, reported zero cached tokens. Median client makespan was **658.323903 ms native** versus **627.697053 ms pinned vLLM**. Native was 30.626850 ms slower (4.88%). The campaign stopped, leaving W2–W6 unmeasured.

The measured prerelease native server SHA-256 was `dcac77a6012b2990d189dd67229a0e696e5c5d5716fcc66dae967a6271c6cb3c`. Its frozen source manifest SHA-256 was `e22a47bf8a509535e333fd8d59d05bb8d7dec35649b8cee811f3b4f382319156` on base `28148cda233e5522274e258d5eb5caa9ab83e313`. This run predates the published runtime. The vLLM pin was `81a8cb063e6247fe254d7caf7e98a6691d71022a`. The comparison, row receipt, and provenance are under `artifacts/handover-w3/cache-on-pin/campaign-v2-gpu0.kn5ms1zc/`.

The archived cache-off `matched-k4-native-defaults-v1` campaign passed W1–W3 on a reserved B200 on 2026-11-23, 10:41:24–11:03:51 UTC. W4–W6 were unmeasured. Both engines used MTP4, a 32-request cap, and an 8,192-token step budget. Native used the Release defaults at the time and CUDA graphs. All streams reported zero cached tokens and streamed greedy IDs to their fixed output caps. [Measured revision](https://github.com/para1lel/sglang.cpp/commit/c3c6271fc9be52dcadea4c074cebd87d68355518)

The pinned vLLM arm used a 2,560-token context cap and 10 GiB KV budget for W1/W2, and 132,096 tokens and 128 GiB for W3. Each arm ran three warmups and nine measured serial requests per W1/W2 row. W3 used two warmups and five measured four-request cohorts per arm. W1 and W3 timed vLLM first. W2 timed native first.

| Workload | Native median, ms | Pinned vLLM median, ms | Native reduction |
| --- | ---: | ---: | ---: |
| W1 | 605.388949 | 723.791664 | 16.36% |
| W2 | 1,348.773244 | 1,934.857093 | 30.29% |
| W3 | 3,950.509358 | 4,280.063137 | 7.70% |

Both engines repeated full IDs on W1/W2. Cross-engine IDs differed for both serial rows and matched all 28 W3 lane streams. The separate native W1 trace matched timed IDs and verified depth four in 49 of 50 rounds. The archive retains the exact launch settings, build and runner manifests, and separate score smoke. [Cache-off v1 evidence](https://github.com/para1lel/sglang.cpp/blob/f00cb0cc976e1a20bf85fd30030f259a7e11a7c6/docs/performance.md#matched-k4-evidence)

Earlier native-default campaigns retain their own settings, medians, and digests. [Seed FC campaign](https://github.com/para1lel/sglang.cpp/blob/0b49cd18212d3535305593bc1c5f5c4c88393e0a/docs/performance.md#matched-k4-evidence) · [Archived native-default report](https://github.com/para1lel/sglang.cpp/blob/4dd8c887c278c67744ed80e7feb2a825a92197a2/docs/serving/paired-native-defaults-results.md)

Use the historical K0 and bounded split-attention K4 reports only under their
recorded settings. [W1–W6 K0 report](https://github.com/para1lel/sglang.cpp/blob/8b4e3f62e0a7a053ffdefe85da99a841fabeb5d3/docs/performance-comparison.md) · [Measurement ledger](https://github.com/para1lel/sglang.cpp/blob/8b4e3f62e0a7a053ffdefe85da99a841fabeb5d3/docs/serving/paired-measurement-ledger.md) · [Bounded split K4 report](https://github.com/para1lel/sglang.cpp/blob/8b4e3f62e0a7a053ffdefe85da99a841fabeb5d3/docs/serving/paired-split-attention-results.md)
