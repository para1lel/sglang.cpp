# Single-B200 benchmark agreement

The project goal is higher inference throughput than pinned vLLM on the **same physical B200 and the same immutable checkpoint**. A workload earns a speedup only after the candidate passes its correctness checks. The measured candidate/vLLM output-token throughput ratio must exceed the agreed threshold above 1.0. The candidate must also meet the agreed latency ceilings. Measure vLLM first. Freeze the workloads, repetition policy, noise margin, and latency ceilings before timing the candidate.

## Implementations and controls

| Part | Responsibility | Source |
| --- | --- | --- |
| Independent baseline | Run vLLM in its own environment. Record the checkout, installed wheel and native binary identities, dependency versions, and effective launch settings. | `vllm-project/vllm` at `81a8cb063e6247fe254d7caf7e98a6691d71022a`. |
| Candidate | Run the project binary built by XMake with the default Pixi toolchain. Record source, build options, and executable identity. | [Development contract](development.md). |
| Architectural reference | Interpret service and cache behavior. Any separate SGLang timing needs its own result label. | `sgl-project/sglang` at `f04d886bf79aa3d6237cbb12a3311838eb9f9bd8`. |

Run each engine alone on the selected UUID. Record driver, CUDA runtimes, host and client configuration, clocks, power settings, and other GPU activity. Keep isolation between the baseline environment and the candidate build. Disclose required software differences and their effect on the comparison.

Match model revision, weight shards and scales, tokenizer, chat template, processor, effective weight precision, and sampling policy. Use the precision policies in the [GQA cache comparison](#gqa-cache-comparison). Freeze request bytes, input token IDs, arrival order, concurrency, seeds, output limits, and stop rules. Match prefix-cache state, MTP mode, and media preprocessing. Check actual dispatch and token counts. If either engine lacks a required mode, mark that row unsupported and leave the comparison open.

## GQA cache comparison

Use the candidate's [native MXFP8 GQA path](development.md#p2-gqa-cache-contract). For the pinned vLLM comparator, request `kv_cache_dtype="fp8_e4m3"` (`--kv-cache-dtype fp8_e4m3` on the server CLI). Pair the runs only after vLLM dispatches with plain E4M3 GQA KV on the required workload. The pin supports per-tensor K and V scales and defaults both to `1.0` when the checkpoint supplies neither. The local checkpoint index contains no attention scales. Verify the effective scale policy in the run. [vLLM cache setting](https://github.com/vllm-project/vllm/blob/81a8cb063e6247fe254d7caf7e98a6691d71022a/vllm/config/cache.py#L251)<!-- prefix: "CUDA 11.8+ s" --> · [vLLM scale initialization](https://github.com/vllm-project/vllm/blob/81a8cb063e6247fe254d7caf7e98a6691d71022a/vllm/model_executor/layers/quantization/kv_cache.py#L118)<!-- prefix: "elif layer.k" -->

The native path stores E4M3 values with E8M0 scales per 32 channels. The pinned comparator uses E4M3 values with per-tensor K and V scales. These precision policies can produce different logits and greedy IDs. Report cross-engine output and logit comparisons for diagnosis only. Require each engine to satisfy the workload's token counts, finish conditions, and stream contract. Record each engine's requested and effective KV format and the scale dtype, granularity, and source. Record the effective attention Q dtype, any Q scaling, and the attention backend and fallback status. Match and record the GDN state dtypes. The checkpoint FP8 weights and their scales remain a separate control.

## First baseline run

From the repository root, prepare the isolated vLLM environment. Set `GPU_INDEX` to the reserved B200's physical index. Run the supplied token-ID fixture against `/data0/shared/Qwen3.8-27B-FP8` and record the resolved UUID.

```bash
bash benchmarks/vllm/setup.sh
GPU="${GPU_INDEX:?set the reserved B200 index}" bash benchmarks/vllm/run.sh benchmarks/vllm/small.json
```

The offline smoke case requests eight output tokens using checkpoint FP8 weights and BF16 KV. Inspect the reported run directory for the device UUID, effective configuration, output IDs, and individual request timings. Repeats of this vLLM run must finish at the requested length with identical greedy IDs. This historical diagnostic measures whole-request latency and output tokens/s. Paired acceptance requires a separate run under the GQA cache comparison. W1–W6 require separate runs with a streaming client to measure first-token latency.

## Named workloads

**Proposed starting matrix.** Lengths are exact tokens **per request**, after template expansion. Concurrency is the number of active requests. Approve the matrix, prompt corpus, request count, arrival schedule, and stop policy before timing. Use fixed-length text fixtures for these rows. Test normal EOS and stop behavior separately.

| Workload | Input tokens per request | Output tokens per request | Concurrency |
| --- | ---: | ---: | ---: |
| W1 | 128 | 128 | 1 |
| W2 | 2,048 | 512 | 1 |
| W3 | 8,192 | 1,024 | 4 |
| W4 | 32,768 | 1,024 | 16 |
| W5 | 32,768 | 4,096 | 32 |
| W6 | 131,072 | 1,024 | 4 |

**Proposed variants.** Start W1–W6 with disjoint prefixes and MTP off. For `W3-prefix`, prime a recorded shared prefix before timing. For `W3-MTP`, repeat W3 with native speculation enabled. For `W1-image` and `W1-video`, freeze media bytes, dimensions, sampled frames, timestamps, preprocessing, effective input positions, and output budget. Report each variant separately. Select actual fixtures and confirm supported lengths against the checkpoint in P0. Preserve OOM, timeout, early stop, and unsupported outcomes under the original workload name.

## Measurement

**Required reporting.** Give output tokens/s, first-token latency, per-output-token latency, and end-to-end latency for each named workload and engine. Retain actual input and output counts and all failed requests.

**Proposed timing protocol.** Use the same streaming client and monotonic clock for both engines. Time a cohort from the first request submission to the last terminal response. Preserve per-request submission, first output token, subsequent token events when observable, final output token, and completion times. Exclude empty stream events from token counts. A fixed-length comparison requires all requests to complete at the specified lengths.

| Metric | Definition for one measured cohort |
| --- | --- |
| Output tokens/s | Sum of emitted output tokens divided by elapsed cohort seconds. Exclude input, padding, and rejected drafts. |
| TTFT (ms) | Time from request submission to its first output token, including queueing and prefill. |
| TPOT (ms/token) | `(last token time − first token time) / (output token count − 1)`, per request. Mark unavailable for fewer than two tokens. |
| End-to-end latency (ms) | Time from request submission to its terminal response. Record stream chunking when multiple tokens arrive together. |
| Requests/s and aggregate tokens/s | Completed requests per cohort second, and total input plus output tokens per cohort second, respectively. |

Separate startup, model load, JIT compilation, and graph capture from steady execution. Warm both engines under the declared cache policy. Reset reusable prefixes before a cold-prefix measurement. Use the same workload order or a recorded counterbalanced order. **Proposed repeat policy.** Collect at least five measured cohorts per engine and workload after warmup. Preserve each cohort's output tokens/s. Report the median and range across cohorts. Report p50 and p95 for TTFT, TPOT, and end-to-end latency over requests per workload, with sample counts and the percentile method.

Report one row per engine, workload, and mode. Include absolute metrics, successful and attempted requests, and the agreed limits. For paired runs, also report the candidate/baseline throughput ratio and latency differences. For workloads with open limits, set numerical margins from repeated baseline measurements before candidate timing.

For offline timing, omit `--top-logits`. Match `--warmup` and `--repeats` to the reference workload. `load_ms` covers model initialization. `request_ms` includes request allocation and any graph capture. Allocation metrics include resident request storage on the serial graph path. Keep the JSON, invocation, binary hash, and device UUID. [Generator measurement](../apps/sglang-generate.cpp#L177)<!-- prefix: "Measurements" -->

### Native text completions

W1–W6 keep the frozen `benchmark` model name, prompt ID arrays, and token-ID SSE. Use those request bytes, clients, and timing definitions for acceptance. Run [text verification](text-serving.md#verification) separately.

For a raw text experiment, freeze UTF-8 request bytes, checkpoint and tokenizer hashes, serving flags, output budget, and stop IDs. Require both engines to encode the same prompt IDs for a paired comparison. Record cold tokenizer load separately from steady tokenization. Declare streaming, usage, and diagnostic ID settings. The first visible text can arrive after the first output ID. A text event can contain bytes from several IDs, and an ID can produce no text. Get output-token counts from usage or diagnostic IDs. Use diagnostic ID timestamps for TTFT and TPOT. Mark those metrics unavailable if the client cannot observe the corresponding IDs. Report time to first visible text separately. Keep raw text measurements in separate result rows.

### Frozen W1 and W2 timing

Use the [frozen streaming protocol](../benchmarks/streaming/README.md#frozen-measurement-protocol) and its GPU3 reference for W1 and W2. Run the native server in default or explicit `--max-active 1` mode. Preserve the device, fixtures, request bytes, warmups, repetitions, and timing definitions. For native timing, run the frozen client with `--url` and a new `--output` directory. Omit `--reference`, which couples timing evaluation to the frozen vLLM IDs. The client then checks stream validity and exact greedy IDs across native repetitions. Verify the client, fixture, and request hashes against the frozen manifest.

For each workload, require every warmup and measured request to pass stream validation. Divide the native median output tokens/s by the frozen baseline median. Require that ratio to exceed `strict_throughput_ratio_gt` in the frozen reference. Compare native p95 TTFT, TPOT, and end-to-end latency with `p95_latency_ceilings_ms` from the same reference. Require each value to meet its ceiling. Retain the frozen output IDs as historical diagnostics.

### Fused FP8 decode

Compare with the published native control under the same checkpoint and precision policy. Require identical greedy IDs and full-vocabulary BF16 logit bits for W1 and W2. Check eager decode and fresh and reused graphs. Exercise exact GQA and direct fallback at their [dispatch bounds](development.md#runtime-path). Run the [prefill](development.md#chunked-prefill-verification), [streaming](development.md#streaming-verification), and [MTP](development.md#serial-greedy-mtp) checks before timing.

Trace actual forward group sizes and row modes. Confirm fused launches for an eligible single target decode row and packed projections for eligible groups of two to four rows. Include a group containing decode and single-row prefill. Compare shape and alignment fallbacks with the separate native operations. For steady decode timing, match execution mode, context length, and KV capacity. Record graph captures and replays.

Compare fused SwiGLU BF16 results with identical and disjoint gate and up output spans. Require `cudaErrorInvalidValue` before launch for identical gate and up spans in fallback, partial output overlaps, and output overlaps with inputs.

The studies in `artifacts/fp8-decode-next/report.md` and `artifacts/decode-combined/report.md` used earlier source. Qualify the promoted build under the frozen serial and concurrent protocols.

### Exact decode bandwidth

Match the published native control's BF16 projections, E4M3 payloads, and FP32 scales byte for byte. Include nonfinite inputs and shape and alignment fallbacks. Trace fresh and reused graphs. Compare full BF16 logits and greedy IDs on shared W1/W2 histories within each policy. Run the [MTP checks](mtp.md#verification).

Time serial HTTP with `--max-active 1 --mtp-steps 0`. Compare within each prefill policy. Apply the [frozen GPU3 criteria](#frozen-w1-and-w2-timing) separately for pinned vLLM.

An earlier GPU1 run compared the unsplit build with the Unit13 runtime. Each arm used three warmups and nine measured requests per workload. W1/W2 median output tokens/s rose from 91.88/35.16 to 95.59/35.70 for exact and from 122.84/57.27 to 129.51/58.70 for tensor-core. All four passed native timing checks. P95 TPOT and end-to-end latency fell throughout. P95 TTFT rose up to 1.00 ms. See `artifacts/exact-decode-after-seed/runs/http.6acmFcnP/`.

### Chunked prefill

Before timing, run the [prefill acceptance checks](development.md#chunked-prefill-verification). Compare full BF16 logit rows and committed KV and GDN state with the row-wise native control under the same precision policy. For device prefill time, place CUDA events around prompt processing. End the interval before the first LM head. For HTTP TTFT, measure from submission to the first token. Record chunk sizes, tails, and active lanes. Measure decode token gaps while mixed steps process prompt chunks.

The 128-row tile requests 31,944,704 scratch bytes. One row requires zero scratch bytes. Serial generator `peak_device_allocation_bytes` counts logical model, request, and transient scratch allocations. Batch `workspace_bytes` counts scratch sized for the largest lane tile in the step. `request_bytes` counts live request storage after the step. These counters exclude idle pool backing. The [32 MiB release threshold](development.md#constraints-and-failure-behavior) is a retention target after synchronization. CUDA can reserve more backing during live work.

Record pool used and reserved bytes separately after synchronization. Check that retired requests leave no live scratch, including after a recovered free-enqueue failure. Record observed peak HBM and headroom with allocator and graph pool effects. Include lane mix and KV capacity.

Use a fresh process for cold allocator measurements. For warm measurements, prime with a multirow request and wait for completion. Record its tile sizes, lane count, and order relative to the measured requests. Record model load, tokenizer warmup, graph state, and prefix-cache state separately. Keep cold and warm results in separate rows.

The historical serial prototype reduced CUDA prefill time against the row-wise native control on W1 and W2. It missed all four paired GPU6 vLLM targets on both workloads. Frozen GPU3 W1/W2 acceptance and W3/W6 concurrent acceptance require measurements of the promoted implementation under their respective protocols.

### FP8 tensor-core prefill

Compare `--fp8-prefill tensor-core` with explicit `--fp8-prefill exact` on the same native binary with MTP off. Positive MTP depth uses exact projections under either flag. Match the checkpoint, GPU UUID, and KV and GDN policies. Freeze request bytes, arrival order, concurrency, output budgets, stop rules, warmups, and repetitions. Put the full launch commands and flags in the immutable run manifest. Record source and binary hashes, toolchain and driver identities, and the requested mode. Trace effective dispatch, row counts, shapes, and fallbacks separately from timing.

Keep the [exact prefill checks](development.md#chunked-prefill-verification) as the regression gate. Compare tensor-core projections with an independent reference using predefined tolerances. Require bitwise equality between fallback and exact projections on identical inputs. Check default and explicit `exact` in the generator and server. Exercise full chunks, tails, mixed prefill and decode, and unsupported dispatch conditions. Check tokens and finish reasons across repetitions within each mode under a fixed schedule.

Compare full-vocabulary BF16 logits at the same positions with identical token histories. Use a shared teacher-forced continuation after greedy outputs diverge. Keep the same prefill and decode boundaries. Report logit errors, top ranks, greedy IDs, and top-two margins for both modes. Keep free-running IDs and the first divergence separately. Reduction differences can accumulate across layers and change the highest-scoring token even when the lead is large. Apply the predefined numerical tolerances to cross-mode score comparisons.

Gate external parity separately against the pinned serving reference on identical histories. Record the reference execution mode and KV and GDN policies. Report targeted diagnostic rows with their selection criteria and use the frozen fixtures for workload-level results.

Measure prompt device time and HTTP TTFT as specified for [chunked prefill](#chunked-prefill). Retain TPOT, end-to-end latency, and output tokens/s under the applicable frozen protocol. Use separate result rows for the two native modes and preserve the vLLM controls. Qualify the promoted backend with its own measurements.

### Native concurrent workloads

W3 and W6 each require four active requests. Use `--max-active 4` after the [native streaming checks](development.md#streaming-verification) pass. Verify full-width decode and memory capacity at each workload's lengths. Use the [concurrent baseline procedure](../benchmarks/concurrent/README.md#conditions-and-gates) for paired timing and acceptance. W4 and W5 exceed the native server's four-lane limit and remain unsupported at their specified concurrency.

### Native serial MTP

Run the [native MTP correctness checks](mtp.md#verification) before timing. Compare layer-major verification with the prior row-wise MTP verifier at the same proposal depth to isolate the verifier change. Use a separate MTP-off graph control to measure the effect of enabling speculation. Match prompt IDs, output budgets, stop policy, and committed outputs. Record binary and checkpoint hashes, GPU UUID, effective dispatch, and device and pinned host allocations. Omit `--top-logits` during timing.

To isolate tiled seeding, compare with row-wise seeding under the same layer-major verifier, exact precision, and positive depth. Time target prefill and draft KV seeding together through prefill workspace release. For an isolated seeding measurement, stop before the first LM head. Record HTTP TTFT and full request latency separately. Include a short prompt, a tile-boundary case, and a longer prompt. Record actual tile sizes, warmup, pool state, output length, and MTP counters.

Alternate arm order across paired runs on the same reserved B200. Match warmups and repeats. Retain each sample and report medians, range, and paired ratios. Earlier short-prompt runs showed substantial variation. Keep serial results scoped to the measured lengths and depth. The [frozen W3-MTP variant](../benchmarks/variants/README.md) requires four active requests. Native MTP supports one.

Positive `--mtp-steps` adds three generator counters from the first measured request. `mtp_drafted_tokens` counts proposals read back. `mtp_accepted_tokens` counts matching proposals appended to the output. `mtp_rounds` counts draft rounds started. Count committed tokens from `output_token_ids`, including the initial token and any corrections or bonuses. Report accepted/drafted and accepted/rounds when their denominators are positive. [JSON fields](../apps/sglang-generate.cpp#L200)<!-- prefix: "void write_r" --> · [Proposal loop](../src/decoder.cpp#L161)<!-- prefix: "bool propose" --> · [Output commit](../src/decoder.cpp#L135)<!-- prefix: "std::optiona" -->

In a separate verification trace, count computed rows, compared proposals, and committed outputs. Count the first mismatch as compared. Computed rows can exceed consumed rows after a mismatch, stop, or cancellation. Check `accepted ≤ compared ≤ drafted` and `drafted ≤ rounds × mtp_steps`. For normal completion, consumed rows equal committed outputs minus the initial token when present. Zero or one output on normal completion requires zero drafts, accepts, and rounds. [Result consumption](../src/decoder.cpp#L182)<!-- prefix: "std::optiona" -->

The [verifier memory bound](mtp.md#constraints-and-failure-behavior) gives logical device and pinned result sizes. The generator adds the larger of prefill scratch and verifier device storage to the base model and request bytes. Record peak HBM separately, including retained pool backing. [Device accounting](../src/decoder.cpp#L273)<!-- prefix: "result.peak_" -->

## Correctness, memory, and diagnosis

Performance acceptance requires [native implementation correctness](development.md#verification), repeatability, and comparable memory and feature settings. Run correctness checks separately when instrumentation changes timing. Apply the remaining proposed checks at their applicable milestones.

- **Memory.** Record HBM after load and peak HBM during prefill and decode. Budget weights and scales, GQA KV pages, GDN live state and checkpoints, activations, workspaces, graph pools, vision weights and buffers, and runtime allocations. State the measurement method, capacity, and headroom for each input length and concurrency.
- **Cache.** Record reusable prefix tokens divided by eligible prompt tokens, the initial cache state, page occupancy, evictions, and recurrent checkpoint restores and replay lengths. Compare cold and primed continuations with fresh execution. Check reclamation after cancellation and memory pressure.
- **MTP.** Check actual draft and target execution. For future sampled paths, compare distributions with MTP off. When target rollback is available, exercise rejection across a recurrent checkpoint boundary.
- **Multimodal.** Verify image and video processing, visual token positions, and reference output behavior on each fixture. Include client-visible preprocessing and transport in end-to-end measurements. Record server processing and vision-encoder time separately when observable.
- **Profiling.** Record GPU utilization, kernel time, JIT time, scheduler time, and queue delay with their collection methods. Profile a separate run if tracing changes throughput. Tie each optimization to its bottleneck, correctness result, and end-to-end result on the same workload.

Retain the run manifest, commands, environment, fixture hashes, raw samples, and failures in ignored `artifacts/` storage. A missing counter is unavailable. A missing baseline, failed correctness check, or changed workload leaves that performance claim open.
