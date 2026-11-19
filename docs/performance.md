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

Start W1–W6 with disjoint prefixes and MTP off. Follow the [W3-prefix protocol](prefix-cache.md#w3-prefix) for native prefix reuse. For `W3-MTP`, repeat W3 with native speculation enabled. For proposed `W1-image` and `W1-video`, freeze media bytes, dimensions, sampled frames, timestamps, preprocessing, effective input positions, and output budget. Report each variant separately. Preserve OOM, timeout, early stop, and unsupported outcomes under the original workload name.

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

Follow the [frozen streaming protocol](../benchmarks/streaming/README.md#frozen-measurement-protocol) and GPU3 reference. Use default or explicit `--max-active 1`. Preserve the device, requests, warmups, repeats, and timing definitions. Verify client, fixture, and request hashes. Use `--url` and a new `--output` directory. Omit `--reference` for native timing. The client checks streams and greedy IDs across native repetitions.

Require every warmup and measured stream to pass validation. The native-to-baseline median throughput ratio must exceed `strict_throughput_ratio_gt`. Native p95 TTFT, TPOT, and end-to-end latency must meet `p95_latency_ceilings_ms`. Retain frozen vLLM IDs for diagnosis.

### Fused FP8 decode

Match native control IDs and full BF16 logits on W1 and W2 with the same checkpoint, precision, and [GQA path](batch-graphs.md#page-parallel-gqa-decode). Check eager decode and fresh and reused graphs. Exercise exact and direct GQA at their [dispatch bounds](development.md#runtime-path). Run the [prefill](development.md#chunked-prefill-verification), [streaming](development.md#streaming-verification), and [MTP](mtp.md#verification) checks.

Trace actual forward group sizes and row modes. Confirm fused launches for an eligible single target decode row. Check packed projections for eligible groups of two through five rows. Include a group containing decode and single-row prefill. Compare shape and alignment fallbacks with the separate native operations. For steady decode timing, match execution mode, context length, and KV capacity. Record graph captures and replays.

Compare fused SwiGLU BF16 results with identical and disjoint gate and up output spans. Require `cudaErrorInvalidValue` before launch for identical gate and up spans in fallback, partial output overlaps, and output overlaps with inputs.

The studies in `artifacts/fp8-decode-next/report.md` and `artifacts/decode-combined/report.md` used earlier source. Qualify the promoted build under the frozen serial and concurrent protocols.

### Exact decode bandwidth

Match the published native control's BF16 projections, E4M3 payloads, and FP32 scales byte for byte. Include nonfinite inputs and shape and alignment fallbacks. Trace fresh and reused graphs. Compare full BF16 logits and greedy IDs on shared W1/W2 histories within each policy. Run the [MTP checks](mtp.md#verification).

Time serial HTTP with `--max-active 1 --mtp-steps 0`. Compare within each prefill policy. Apply the [frozen GPU3 criteria](#frozen-w1-and-w2-timing) separately for pinned vLLM.

Record visible lengths and KV capacity. Report ordered fallback timing separately if measured.

An earlier GPU1 run compared the unsplit build with the Unit13 runtime. Each arm used three warmups and nine measured requests per workload. W1/W2 median output tokens/s rose from 91.88/35.16 to 95.59/35.70 for exact and from 122.84/57.27 to 129.51/58.70 for tensor-core. All four passed native timing checks. P95 TPOT and end-to-end latency fell throughout. P95 TTFT rose up to 1.00 ms. See `artifacts/exact-decode-after-seed/runs/http.6acmFcnP/`.

### Chunked prefill

Run the [prefill acceptance checks](development.md#chunked-prefill-verification) before timing. For exact prefill, match full BF16 logits and committed KV and GDN state with the row-wise control. Time device prompt processing with CUDA events through the operation before the first LM head. Measure HTTP TTFT from submission to the first token. Record tile sizes, tails, active lanes, and decode token gaps during mixed steps.

The 128-row tile requests 31,944,704 scratch bytes. One row requires zero. Generator `peak_device_allocation_bytes` counts logical model, request, and scratch bytes. Batch `workspace_bytes` sizes scratch for the largest lane tile. `request_bytes` counts live request storage after the step. These counters exclude idle pool backing. The [32 MiB release threshold](development.md#constraints-and-failure-behavior) applies after synchronization. Live work can reserve more.

Record pool used and reserved bytes after synchronization. Require zero live scratch after retirement, including recovered free-enqueue failures. Report peak HBM, headroom, lane mix, and KV capacity.

Use a fresh process for cold allocator timing. For warm timing, complete a multirow request first. Record its tile sizes, lane count, and order. Record model load, tokenizer warmup, graph state, and prefix-cache state separately. Report cold and warm rows.

The historical GPU6 prototype reduced W1 and W2 CUDA prefill time against the row-wise native control. It missed all four paired vLLM targets on both workloads. Qualify each production configuration under its applicable protocol.

### FP8 tensor-core prefill

Compare `--fp8-prefill tensor-core` with explicit `--fp8-prefill exact` on one binary with MTP off. Positive MTP depth uses exact projections under either flag. Match measurement controls and KV and GDN policies. Record commands, flags, source and binary hashes, toolchain, and driver in the immutable manifest. Trace effective dispatch, shapes, and fallbacks separately from timing.

Run the [exact prefill checks](development.md#chunked-prefill-verification) with default and explicit `exact` in both binaries. Compare tensor-core projections with an independent reference using predefined tolerances. Match fallback and exact projections bitwise on identical inputs. Exercise full chunks, tails, mixed steps, and unsupported dispatch. Check tokens and finish reasons across repetitions within each policy.

For the [prefill fusions](development.md#runtime-path), match full BF16 logits and committed KV and GDN state bitwise against the pre-fusion implementation with the same policy and tile schedule. Match staged operator results for BF16 outputs, FP8 payloads and scales, including raw MXFP8 exponents 0 and 255. Check one-row fallback, two-row tiles, page crossings, positive MTP depth under both flags, and graph and eager decode.

Compare full BF16 logits within predefined tolerances on identical histories and prefill boundaries. Use shared teacher-forced tokens after greedy divergence. Report logit errors, ranks, top-two margins, free-running IDs, and first divergence. Reduction differences can change the greedy winner even when the lead is large.

Check parity against pinned vLLM on identical histories. Record its execution mode and KV and GDN policies. Record how diagnostic rows were selected. Use frozen fixtures for workload-level results.

Use [chunked prefill measurements](#chunked-prefill) and the [frozen timing protocol](#frozen-w1-and-w2-timing). Report each precision policy separately.

The post-chat GPU3 run used tensor-core prefill, serial CUDA graphs, and MTP off. Each arm and workload used three warmups and nine measured requests. The chat control and fusion build returned identical IDs on the frozen token-ID requests.

| Workload and arm | Median output tok/s | P95 TTFT (ms) | P95 TPOT (ms/token) | P95 E2E (ms) | Gates met |
| --- | ---: | ---: | ---: | ---: | ---: |
| W1 chat control | 128.976 | 27.798 | 7.596 | 992.525 | 3/4 |
| W1 fusion | 129.935 | 20.427 | 7.609 | 986.680 | 4/4 |
| W1 frozen limits | >117.037 | ≤21.637 | ≤8.625 | ≤1116.170 | |
| W2 chat control | 58.526 | 1148.397 | 14.878 | 8750.750 | 0/4 |
| W2 fusion | 61.250 | 753.935 | 14.890 | 8361.881 | 0/4 |
| W2 frozen limits | >115.815 | ≤80.305 | ≤8.671 | ≤4510.901 | |

The verifier used unrounded limits. W2 still misses all four. See `artifacts/tc-prefill-fusion-after-chat/results/performance/comparison.json`.

### Native concurrent workloads

After [streaming verification](development.md#streaming-verification), set `--max-active` to the row's lane count and use K0. W4 uses 16 lanes and W5 uses 32. Wait for [HTTP readiness](development.md#native-streaming-server). In a separate traced run, verify full-width decode and record effective [graph dispatch](batch-graphs.md#timing).

Follow the [frozen protocol](../benchmarks/concurrent/README.md#conditions-and-gates) for acceptance. The post-loader [W3 smoke](batch-graphs.md#verification) used shortened budgets and instrumentation. W4 and W5 acceptance remains pending.

A pre-loader GPU6 comparison ran one cold full W3 cohort per arm with tensor-core prefill, batch graphs, and MTP off. Unit16 used eager long decode. The candidate replayed it. All 4,096 IDs, finish reasons, and usage matched. Throughput rose from 5.69 to 14.91 IDs/s. P95 TPOT fell from 624.40 to 189.36 ms. See `artifacts/long-batch-graph-release/report.md`.

The historical `w3-fused` experiment used additional attention and projection fusions. It reached 144.84 output IDs/s and missed all four [frozen W3 gates](../benchmarks/concurrent/README.md#conditions-and-gates).

### Native MTP

Run the [MTP checks](mtp.md#verification) before timing. Use same-width eager K0 to isolate speculation and a separate K0 graph arm to measure the deployment trade-off. Match prompt IDs, arrivals, output budgets, stop policy, and committed outputs. Record binary and checkpoint hashes, GPU UUID, effective dispatch, and device and pinned host allocations. Disable tracing and `--top-logits` for timing.

To isolate layer-major verification, compare with row-wise MTP at the same depth. To isolate tiled seeding, keep the verifier, exact precision, and depth fixed. Time target prefill and draft KV seeding through workspace release, before the first LM head. Record HTTP TTFT and request latency separately. Include short, tile-boundary, and long prompts. Record tile sizes, pool state, and output length.

Alternate paired arms on the same reserved B200. Match warmups and repeats. Report individual samples, medians, ranges, and paired ratios. The [frozen W3-MTP variant](../benchmarks/variants/README.md) requires four active requests and K1. Keep that workload separate from the [historical eager bursts](mtp.md#measured-continuous-mtp-trade-offs).

Positive depth adds counters for the generator's first measured request. `mtp_drafted_tokens` counts proposals read back. `mtp_accepted_tokens` counts matching proposals appended to the output. `mtp_rounds` counts rounds started. Count committed `output_token_ids` including first, correction, and bonus tokens. Report accepted/drafted and accepted/rounds for positive denominators. [JSON fields](../apps/sglang-generate.cpp#L200)<!-- prefix: "void write_r" --> · [Proposal loop](../src/decoder.cpp#L235)<!-- prefix: "bool propose" --> · [Output commit](../src/decoder.cpp#L209)<!-- prefix: "std::optiona" -->

Trace separately. Check `accepted ≤ compared ≤ drafted` and `drafted ≤ rounds × K`. Count the first mismatch as compared. Computed rows can exceed consumed rows after mismatch or stop. In normal completion, consumed decode rows equal committed outputs minus the initial token. Zero or one output requires zero drafts. [Consumption](../src/decoder.cpp#L256)<!-- prefix: "std::optiona" -->

Batch traces report joint verifier lanes and computed rows. Per slot, `drafted` counts proposals read back and `accepted` counts matching outputs committed before publication. `verified` counts computed target rows. `replayed` counts rows replayed on partial consumption. A one-output eager `boundary` records zero speculative counts. Client-visible output can be shorter after cancellation. [Batch trace](../apps/server/batch_service.cpp#L52)<!-- prefix: "void trace_v" -->

The [verifier storage bound](mtp.md#constraints-and-failure-behavior) covers device and pinned results. The generator adds the larger of prefill scratch and verifier storage to base model and request bytes. Batch `peak_request_bytes` includes transient prefill scratch or live request storage before retirement. Record peak HBM separately. [Generator accounting](../src/decoder.cpp#L347)<!-- prefix: "result.peak_" --> · [Batch accounting](../src/decoder_batch.cpp#L622)<!-- prefix: "current.peak" -->

## Correctness, memory, and diagnosis

Performance acceptance requires [native implementation correctness](development.md#verification), repeatability, and comparable memory and feature settings. Run correctness checks separately when instrumentation changes timing. Apply the remaining proposed checks at their applicable milestones.

- **Memory.** Record HBM after load and peak HBM during prefill and decode. Budget weights and scales, GQA KV pages, GDN live state and checkpoints, activations, workspaces, graph pools, vision weights and buffers, and runtime allocations. State the measurement method, capacity, and headroom for each input length and concurrency.
- **Cache.** Record reusable prefix tokens divided by eligible prompt tokens, the initial cache state, page occupancy, evictions, and recurrent checkpoint restores and replay lengths. Compare cold and primed continuations with fresh execution. Check reclamation after cancellation and memory pressure.
- **MTP.** Check greedy draft and target execution and recurrent rollback.
- **Sampling.** Run the [sampling checks](text-serving.md#verification). Record K0 sampling controls, effective dispatch, and [batch scratch](batch-graphs.md#runtime-path) separately from frozen greedy workloads.
- **Finite choices.** Run the [constraint checks](text-serving.md#finite-choice-verification). Record grammar, candidate counts, effective dispatch, and [batch scratch](batch-graphs.md#runtime-path) separately from frozen workloads.
- **Multimodal.** Verify image and video processing, visual token positions, and reference output behavior on each fixture. Include client-visible preprocessing and transport in end-to-end measurements. Record server processing and vision-encoder time separately when observable.
- **Profiling.** Record GPU utilization, kernel, JIT, scheduler, and queue times, plus collection methods. Profile separately if tracing changes throughput. Collect [native scheduler samples](prefix-cache.md#native-http-controls) during diagnostics. [Full vLLM verification](../benchmarks/variants/verify.py#L390)<!-- prefix: "def verify(r" --> needs pinned worker metadata and PyTorch traces. Link optimizations to bottlenecks, correctness, and results.

Retain the run manifest, commands, environment, fixture hashes, raw samples, and failures in ignored `artifacts/` storage. A missing counter is unavailable. A missing baseline, failed correctness check, or changed workload leaves that performance claim open.
