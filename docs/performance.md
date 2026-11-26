# Matched B200 benchmark agreement

Use the same B200 devices and immutable checkpoint. Freeze workloads, controls, and arm order. The [TP2 target](serving/tp2.md) covers two-GPU acceptance: all five rows miss at `168fdb1a`. [Paired driver](../benchmarks/paired/README.md#matched-k4-timing-protocol)

## Implementations and controls

| Part | Responsibility | Source |
| --- | --- | --- |
| Independent baseline | Run vLLM in its own environment. Record the checkout, installed wheel and native binary identities, dependency versions, and effective launch settings. | `vllm-project/vllm` at `81a8cb063e6247fe254d7caf7e98a6691d71022a`. |
| Candidate | Run the project binary built by XMake with the default Pixi toolchain. Record source, build options, and executable identity. | [Development contract](development.md). |
| Architectural reference | Interpret service and cache behavior. Any separate SGLang timing needs its own result label. | `sgl-project/sglang` at `f04d886bf79aa3d6237cbb12a3311838eb9f9bd8`. |

Run each engine alone on the selected UUIDs in its own environment. Record drivers, runtimes, host and client settings, clocks, power, competing GPU activity, and software differences.

Match checkpoint, tokenizer, template, processor, weight precision, sampling, cache, MTP, and media preprocessing. Freeze request bytes, input IDs, arrivals, seeds, and stops. Record dispatch, token counts, and unsupported modes.

## Scheduler and reporting policy

Use MTP4, 32,768 tokens per step, eight slots per GPU, and cold prefixes.

- **Pinned vLLM.** Set `--max-num-batched-tokens 32768 --max-num-scheduled-tokens 32768` and enable `--async-scheduling`. Match `--max-num-seqs` to the active capacity. Verify MTP4 and the effective aligned GDN cache settings at the API and worker.
- **Native.** Use the default direct GPU LMCache path. The paired driver starts a private LMCache 0.5.8 daemon with 64 GiB CPU L1 on the same physical GPU. [Daemon launcher](../benchmarks/paired/lmcache_arm.sh#L36)<!-- prefix: "start_lmcach" -->

The pinned vLLM arm uses its local prefix cache. Verify `kv_transfer=false` and `kv_offloading_size=null` in the worker receipt. The same check requires zero CPU weight offload. [Worker receipt check](https://github.com/para1lel/sglang.cpp/blob/03ee22d849e6ef7bca723595a7c8e63c47187749/benchmarks/paired/saved_evidence.py#L223)<!-- prefix: "and worker[\"" -->

The pinned vLLM scheduler scans its running list before its FCFS waiting queue. That list can include an unfinished prompt. Native [reserves ready decode rows before FCFS prefill](batch-graphs.md#runtime-path). Match workload and capacity settings across the engines. [Pinned vLLM running scan](https://github.com/vllm-project/vllm/blob/81a8cb063e6247fe254d7caf7e98a6691d71022a/vllm/v1/core/sched/scheduler.py#L767) · [Waiting scan](https://github.com/vllm-project/vllm/blob/81a8cb063e6247fe254d7caf7e98a6691d71022a/vllm/v1/core/sched/scheduler.py#L1029)

Use the [paired launchers](../benchmarks/paired/README.md). Label other caps exploratory or historical.

Report nondefault settings, effective limits, offered and observed concurrency, input lengths, context and output budgets, and [admission failures](batch-graphs.md#resident-memory-admission) separately from timing. Acceptance requires a fresh matched run with every required mode.

## GQA cache comparison

Use the candidate's [native MXFP8 GQA path](development.md#p2-gqa-cache-contract). For the pinned vLLM comparator, request `kv_cache_dtype="fp8_e4m3"` (`--kv-cache-dtype fp8_e4m3` on the server CLI). Pair the runs only after vLLM dispatches with plain E4M3 GQA KV on the required workload. The pin supports per-tensor K and V scales and defaults both to `1.0` when the checkpoint supplies neither. The local checkpoint index contains no attention scales. Verify the effective scale policy in the run. [vLLM cache setting](https://github.com/vllm-project/vllm/blob/81a8cb063e6247fe254d7caf7e98a6691d71022a/vllm/config/cache.py#L251)<!-- prefix: "CUDA 11.8+ s" --> · [vLLM scale initialization](https://github.com/vllm-project/vllm/blob/81a8cb063e6247fe254d7caf7e98a6691d71022a/vllm/model_executor/layers/quantization/kv_cache.py#L118)<!-- prefix: "elif layer.k" -->

The native path stores E4M3 values with E8M0 scales per 32 channels. The pinned comparator uses E4M3 values with per-tensor K and V scales. These precision policies can produce different logits and greedy IDs. Report cross-engine output and logit comparisons for diagnosis only. Require each engine to satisfy the workload's token counts, finish conditions, and stream contract. Record each engine's requested and effective KV format and the scale dtype, granularity, and source. Record the effective attention Q dtype, any Q scaling, and the attention backend and fallback status. Match and record the GDN state dtypes. The checkpoint FP8 weights and their scales remain a separate control.

## Historical offline smoke

The archived eight-output greedy BF16 KV fixture has a separate offline recipe. Use streaming for TTFT. [Small workload](../benchmarks/vllm/README.md#small-workload) · [Historical BF16 result](../benchmarks/vllm/README.md#historical-bf16-result)

## Named workloads

Lengths are exact post-template tokens per request. Scale offered requests by GPU count. The driver uses one B200. Freeze IDs, arrivals, and stops. Test normal EOS separately.

| Workload | Input tokens per request | Output tokens per request | Offered requests per GPU |
| --- | ---: | ---: | ---: |
| W1 | 128 | 128 | 1 |
| W2 | 2,048 | 512 | 1 |
| W3 | 8,192 | 1,024 | 4 |
| W4 | 32,768 | 4,096 | 8 |
| W5 | 131,072 | 1,024 | 4 |

Use the [matched K4 protocol](../benchmarks/paired/README.md#matched-k4-timing-protocol). W4 uses the first eight requests from archived W5. W5 uses all four archived W6 requests. Keep the [warm-prefix diagnostic](../benchmarks/paired/README.md#prefix-reuse-diagnostic) and media variants separate.

## Measurement

Require native makespan ≤1.10× pinned vLLM on each W1–W5 row. Stop tuning when all pass. Freeze scheduling and workloads. Tune kernels, programmatic dependent launch (PDL), streams, and CUDA Graph replay.

Run one warmup and one measured cohort per item and engine. Use the measured verdict and retain the first miss. Profile long rows separately. W4 can open a campaign. W5 requires a W4 pass in that campaign. Run W1–W3 in order, advancing on each pass. Validate device, launch settings, and each stream’s request, counts, finish, usage, and drain. Under TP1, measured IDs must match the engine’s warmup; [TP2](serving/tp2.md) content is reference only. [Matched K4 timing protocol](../benchmarks/paired/README.md#matched-k4-timing-protocol)

Use one streaming client and monotonic clock from earliest submission to last valid `[DONE]`. Preconnect lanes, release one barrier, and send all bodies within 50 ms and before any completion. Retain submit times, headers, token events, completion times, counts, and failures. Exclude [SSE keepalives](http-server.md#completion-streams) and empty events from token counts.

Report output tokens/s, TTFT, TPOT, end-to-end latency, and whether every lane starts generating before the first completion for diagnosis.

| Metric | Definition for one measured cohort |
| --- | --- |
| Output tokens/s | Sum of emitted output tokens divided by elapsed cohort seconds. Exclude input, padding, and rejected drafts. |
| TTFT (ms) | Submission to first output token, including queueing and prefill. Unavailable for zero output tokens. [Client calculation](../benchmarks/streaming/client.py#L202)<!-- prefix: "\"ttft_ms\": (" --> |
| TPOT (ms/token) | `(last token time − first token time) / (output token count − 1)`, per request. Mark unavailable for fewer than two tokens. |
| End-to-end latency (ms) | Time from request submission to its terminal response. Record stream chunking when multiple tokens arrive together. |
| Requests/s and aggregate tokens/s | Completed requests per cohort second, and total input plus output tokens per cohort second, respectively. |

Exclude startup, model load, JIT, and graph capture. Drain both engines and require acknowledged resets before each cohort, zero cached tokens per request, and unchanged hit counters. Direct GPU replay checks periodic and terminal store counts, written positions, transfer counters, zero GQA/GDN/draft reuse, and no new declines or I/O errors. Report makespan and throughput as point estimates. Give the latency sample count and percentile method. [Cold reset](../benchmarks/paired/cold_cache.py#L112)<!-- prefix: "def before_c" --> · [Post-stream drain](../benchmarks/paired/cold_cache.py#L127)<!-- prefix: "def after_co" --> · [Drain verification](../benchmarks/paired/cold_cache.py#L134)<!-- prefix: "def verify_l" -->

After stream capture, the client records `settled_ns` when running requests, waiting requests, and `lmcache_pending_stores` reach zero. Polling contributes to this interval. Use settled counters for cache accounting and stream timestamps for the timing ratio.

Report absolute metrics and successful/attempted request counts per engine, workload, and mode. Diagnose with paired throughput ratios and latency differences. Archived receipts retain their original checks.

For new offline timing, use `--warmup 1 --repeats 1` and omit `--top-logits`. `load_ms` covers model initialization. `request_ms` includes request allocation and graph capture. Allocation metrics include resident storage on the serial graph path. Keep JSON, invocation, binary hash, and device UUID. [Generator measurement](../apps/sglang-generate.cpp#L247)<!-- prefix: "Measurements" -->

### Operator qualification

Operators may reorder arithmetic within their numerical contract. Compare outputs and state with an independent mathematical reference under the operation's established error criteria. Check causality where the operation uses token history, and model quality on frozen inputs and shared token histories. Record free-running divergence separately. Require bitwise equality for copied bytes, exact replay, and comparisons that preserve arithmetic.

Use repeated matched measurements to choose the fastest validated implementation per supported case. Keep one production implementation per case. Use validated fallbacks where shape, device, or resources prevent the primary route. Keep frozen fixtures and timing gates unchanged.

### Native text completions

Reuse the current workload `benchmark` requests, token-ID SSE clients, and timing definitions under the [scheduler policy](#scheduler-and-reporting-policy). Compare cross-engine output IDs for diagnosis. Run [text verification](text-serving.md#verification) separately.

For raw text, freeze UTF-8 requests, checkpoint and tokenizer hashes, serving flags, output budget, and stop IDs. Match prompt IDs across engines. Time cold tokenizer load separately from steady tokenization. Record streaming, usage, and diagnostic ID settings. A text event can span several IDs, and an ID can produce no text. Count tokens from usage or diagnostic IDs. Calculate TTFT and TPOT from ID timestamps when available. Otherwise, mark them unavailable. Report time to first visible text separately. Give raw text its own result rows.

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

Verify offered and observed concurrency, streaming, and decode width and replay under the fixed scheduler. Keep historical K0 diagnostics with their original controls. [Streaming verification](development.md#streaming-verification) · [Batch timing](batch-graphs.md#timing) · [Wide decode checks](long-paged-decode.md#verification)

### Native MTP

Run the [MTP checks](mtp.md#verification) before timing. Use `--mtp-steps 0` for the eager control at the same width. Measure K0 replay separately. Match prompt IDs, arrivals, budgets, stop policy, and outputs. Record binary and checkpoint hashes, GPU UUID, dispatch, and device and pinned host bytes. Disable tracing and `--top-logits`.

To isolate layer-major verification, compare with row-wise MTP at the same depth. To isolate tiled seeding, keep the verifier, exact precision, and depth fixed. Time target prefill and draft KV seeding through workspace release, before the first LM head. Record HTTP TTFT and request latency separately. Include short, tile-boundary, and long prompts. Record tile sizes, pool state, and output length.

Alternate paired arms on one reserved B200. Match warmups and repeats. Report samples, medians, ranges, and paired ratios. The W3-MTP variant uses four 8,192-token prompts with 1,024 outputs each. Its K1 start exceeds the verification graph's visible limit. [W3-MTP fixture](../benchmarks/variants/README.md) · [Graph limit](mtp.md#target-verification-graph)

Positive depth adds counters for the first measured request. `mtp_drafted_tokens` counts proposals read back. `mtp_accepted_tokens` counts matching proposals appended to the output. `mtp_rounds` counts rounds that produced at least one proposal. Count committed `output_token_ids` including first, correction, and bonus tokens. Report accepted/drafted and accepted/rounds for positive denominators. [JSON fields](../apps/sglang-generate.cpp#L273)<!-- prefix: "void write_r" --> · [Proposal loop](../src/decoder_verify.cpp#L156)<!-- prefix: "bool MtpRoun" --> · [Generator counters](../src/decoder_batch.cpp#L783)<!-- prefix: "result.mtp_r" -->

Use `--trace-batches` at any capacity in a separate run. Check `accepted ≤ compared ≤ drafted` and `drafted ≤ rounds × K`. Count the first mismatch as compared. Computed rows can exceed consumed rows after mismatch or stop. In normal completion, consumed decode rows equal committed outputs minus the initial token. Zero or one output requires zero drafts. [Consumption](../src/decoder_verify.cpp#L278)<!-- prefix: "bool MtpRoun" -->

Batch traces sum verifier lanes and computed rows across all groups in the step. Per slot, `drafted` counts proposals read back and `accepted` counts matching outputs committed before publication. `verified` counts computed target rows. `replayed` counts rows replayed on partial consumption. A one-output eager `boundary` records zero speculative counts. Client-visible output can be shorter after cancellation. [Batch trace](../include/sglang/server/batch_trace.h#L14)<!-- prefix: "inline void " -->

See the [verifier storage bound](mtp.md#constraints-and-failure-behavior). The generator adds peak request, prefill, and selection bytes to model bytes. Batch `peak_request_bytes` includes prefill scratch or live request storage before retirement. Record peak HBM separately. [Generator accounting](../src/decoder_batch.cpp#L780)<!-- prefix: "result.peak_" --> · [Batch accounting](../src/decoder_batch.cpp#L681)<!-- prefix: "if (stats) c" -->

## Correctness, memory, and diagnosis

Performance acceptance requires [native implementation correctness](development.md#verification), repeatability, and comparable memory and feature settings. Run correctness checks separately when instrumentation changes timing. Apply remaining proposed checks at their applicable milestones.

- **Memory.** Record HBM after load and peak HBM during prefill and decode. Budget weights and scales, GQA KV pages, GDN live state and checkpoints, activations, workspaces, graph pools, vision weights and buffers, and runtime allocations. State the measurement method, capacity, and headroom for each input length and concurrency.
- **Cache.** Record reusable prefix tokens divided by eligible prompt tokens, the initial cache state, page occupancy, evictions, and recurrent checkpoint restores and replay lengths. Compare cold and primed continuations with fresh execution. Check reclamation after cancellation and memory pressure.
- **MTP.** Check target verification and recurrent rollback.
- **Sampling.** Run the [sampling checks](text-serving.md#verification) and [sampled MTP checks](mtp.md#sampled-verification). Report rates separately by width, depth, and arm order.
- **Finite choices.** Run the [constraint checks](text-serving.md#finite-choice-verification). Record grammar, candidate counts, effective dispatch, and [batch scratch](batch-graphs.md#runtime-path) separately from frozen workloads.
- **Multimodal.** Verify image and video processing, visual token positions, and reference output behavior on each fixture. Include client-visible preprocessing and transport in end-to-end measurements. Record server processing and vision-encoder time separately when observable.
- **Profiling.** Record GPU utilization, kernel, JIT, scheduler, and queue times, plus collection methods. Profile separately if tracing changes throughput. Collect [native scheduler samples](prefix-cache.md#native-http-controls) during diagnostics. [Full vLLM verification](../benchmarks/variants/verify.py#L407)<!-- prefix: "def verify(r" --> needs pinned worker metadata and PyTorch traces. Link optimizations to bottlenecks, correctness, and results.

Retain the run manifest, commands, environment, fixture hashes, raw samples, and failures in ignored `artifacts/` storage. A missing counter is unavailable. A missing baseline, failed correctness check, or changed workload leaves that performance claim open.

## Matched K4 evidence

All five rows passed (native/vLLM ≤1.10). Both engines used one B200, [cold MTP4](#scheduler-and-reporting-policy), eight slots, and 32,768 tokens per step. Each row had one warmup and one measurement per engine. Native used LMCache 0.5.8, vLLM its local cache. [Native release](https://github.com/para1lel/sglang.cpp/commit/c701d294b3823c655c6adc38bbd87d077147bbcc)

| Row | Native (s) | Pinned vLLM (s) | Native/vLLM |
| --- | ---: | ---: | ---: |
| W1 | 0.575548348 | 0.621156755 | 0.926575045 |
| W2 | 1.369969434 | 1.686546247 | 0.812292836 |
| W3 | 3.965594404 | 3.954927060 | 1.002697229 |
| W4 | 26.809497401 | 26.564937688 | 1.009206109 |
| W5 | 38.396171943 | 35.940895116 | 1.068314293 |

SSE replay independently reproduced makespans. Each engine repeated its warmup output IDs. Across engines, W1/W2 first differed at zero-based indices 28/0. All 16 W3–W5 lanes matched. Native W5 acknowledged 68 stores covering 528,380 positions. Its drain ended 0.113431875 seconds after the last `[DONE]`, outside makespan. See `artifacts/resume-paired-count2-official-B9PfsPRu/formal-summary.json`.

### Traced one-output prefill diagnostic

Traced cold MTP4, B200 GPU 6, four 131,072×1 requests. Each measurement recorded 15 mixed-start steps, 64 stores, and 524,288 written positions.

| Reader (ELF) | Measured attention (s) | Client makespan (warmup / measured, s) |
| --- | ---: | ---: |
| Parent `e2e07a51` | 30.225588 | 34.103112 / 47.770832 |
| [Singleton](flash-prefill.md#runtime-path) `719a25ab` | 16.827783 | 35.260373 / 36.015000 |
| Paired `0d0a7132` | 16.023583 | 34.400717 / 34.439601 |

Pairing cut attention by 0.804200 s. Time outside server GPU work also fell.

<a id="current-single-round-v5"></a>
### Archived single-round v5

The former native device-resident prefix cache ran W1–W3 on GPU 7 and W4–W5 on GPU 6. Both engines used MTP4, 32,768 tokens per step, eight slots, cold resets, one warmup, and one measurement.

| Row | Native (s) | Pinned vLLM (s) | Native / vLLM | Verdict |
| --- | ---: | ---: | ---: | --- |
| W1 | 0.579101967 | 0.621863994 | 0.931236 | Pass |
| W2 | 1.411212457 | 1.686342932 | 0.836848 | Pass |
| W3 | 3.823016226 | 4.037761004 | 0.946816 | Pass |
| W4 | 26.368189095 | 26.589718570 | 0.991669 | Pass |
| W5 | 39.399096915 | 35.972035063 | 1.095270 | Pass |

W5 followed W4 and cleared the gate by 0.170142 seconds. Replay verified 73 cold SSE streams, per-engine repeat IDs, and W1 trace IDs. Across engines, all W3–W5 lanes matched. W1 and W2 differed.

Release `ee39aef6c625c50d98eede8103a2b53e798f589d7c0e916db7f913bcb6d94a6d` matched 226 production and 22 protocol files. [Native release](https://github.com/para1lel/sglang.cpp/commit/6865c175da560021f5724a869cb7f39061e1675c) · [vLLM pin](https://github.com/vllm-project/vllm/commit/81a8cb063e6247fe254d7caf7e98a6691d71022a)

### Archived multi-cohort v4

Final v4 passed all five rows. W1–W2 used three warmups and nine measurements, W3–W5 two and five. W5 medians were 38.170671560 seconds native and 35.974923597 seconds vLLM (1.061036×). [Full v4 results](https://github.com/para1lel/sglang.cpp/blob/6f4d54d3a07ea8ad5bae68ea020105b8ecaeba93/docs/performance.md#matched-k4-evidence) · [Measured v4 release](https://github.com/para1lel/sglang.cpp/commit/b8764ee28523c7223865a0e1de8ebc8041ca857a)

The previous v4 campaign passed W1–W4 and missed W5. [Prior v4 evidence](https://github.com/para1lel/sglang.cpp/blob/17d8634f1f9677bf4b2bc3dea93b7e5ca58eca5a/docs/performance.md#matched-k4-evidence)
