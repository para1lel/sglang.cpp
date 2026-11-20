# Single-B200 benchmark agreement

Compare correct implementations on the **same physical B200 and immutable checkpoint**. Measure pinned vLLM first. Freeze workloads, repeats, throughput thresholds above 1.0, and latency ceilings before candidate timing. See the [fixed W1–W6 comparison](performance-comparison.md).

## Implementations and controls

| Part | Responsibility | Source |
| --- | --- | --- |
| Independent baseline | Run vLLM in its own environment. Record the checkout, installed wheel and native binary identities, dependency versions, and effective launch settings. | `vllm-project/vllm` at `81a8cb063e6247fe254d7caf7e98a6691d71022a`. |
| Candidate | Run the project binary built by XMake with the default Pixi toolchain. Record source, build options, and executable identity. | [Development contract](development.md). |
| Architectural reference | Interpret service and cache behavior. Any separate SGLang timing needs its own result label. | `sgl-project/sglang` at `f04d886bf79aa3d6237cbb12a3311838eb9f9bd8`. |

Run each engine alone on the selected UUID in its own environment. Record drivers, runtimes, host and client settings, clocks, power, other GPU activity, and software differences.

Match checkpoint, tokenizer, template, processor, weight precision, sampling, cache state, MTP, and media preprocessing. Freeze request bytes, input IDs, arrivals, seeds, and stops. Record dispatch and token counts. Mark missing modes unsupported.

## Scheduler and reporting policy

Launch both engines with **32,768 aggregate scheduled tokens per step** and **32 maximum in-service requests** for every new performance run, including diagnostics.

- **Pinned vLLM.** Set `--max-num-batched-tokens 32768 --max-num-seqs 32`. Leave `--max-num-scheduled-tokens` unset or set it to `32768`. Verify the effective limits.
- **Native server.** Set `--max-active 32 --max-num-batched-tokens 32768`. Verify both defaults in [`GET /health`](http-server.md). Record admission failures separately from timing under the [resident memory limit](batch-graphs.md#resident-memory-admission).

In every chat report, list each engine's nondefault launch, environment, client, and request settings, including diagnostics. Report effective server maxima, offered and observed concurrency, input lengths, context limits, and output budgets. Preserve frozen run settings, identities, and gates. Acceptance requires a new matched run under this policy with every required mode available.

## GQA cache comparison

Use the candidate's [native MXFP8 GQA path](development.md#p2-gqa-cache-contract). For the pinned vLLM comparator, request `kv_cache_dtype="fp8_e4m3"` (`--kv-cache-dtype fp8_e4m3` on the server CLI). Pair the runs only after vLLM dispatches with plain E4M3 GQA KV on the required workload. The pin supports per-tensor K and V scales and defaults both to `1.0` when the checkpoint supplies neither. The local checkpoint index contains no attention scales. Verify the effective scale policy in the run. [vLLM cache setting](https://github.com/vllm-project/vllm/blob/81a8cb063e6247fe254d7caf7e98a6691d71022a/vllm/config/cache.py#L251)<!-- prefix: "CUDA 11.8+ s" --> · [vLLM scale initialization](https://github.com/vllm-project/vllm/blob/81a8cb063e6247fe254d7caf7e98a6691d71022a/vllm/model_executor/layers/quantization/kv_cache.py#L118)<!-- prefix: "elif layer.k" -->

The native path stores E4M3 values with E8M0 scales per 32 channels. The pinned comparator uses E4M3 values with per-tensor K and V scales. These precision policies can produce different logits and greedy IDs. Report cross-engine output and logit comparisons for diagnosis only. Require each engine to satisfy the workload's token counts, finish conditions, and stream contract. Record each engine's requested and effective KV format and the scale dtype, granularity, and source. Record the effective attention Q dtype, any Q scaling, and the attention backend and fallback status. Match and record the GDN state dtypes. The checkpoint FP8 weights and their scales remain a separate control.

## Historical offline smoke

From the repository root, set `GPU_INDEX` to the reserved B200 index. These commands reproduce the archived offline diagnostic.

```bash
bash benchmarks/vllm/setup.sh
GPU="${GPU_INDEX:?set the reserved B200 index}" bash benchmarks/vllm/run.sh benchmarks/vllm/small.json
```

The fixture uses eight greedy output tokens and BF16 KV. Check repeat IDs, UUID, effective settings, request latency, and throughput. Use streaming for W1–W6 TTFT.

## Named workloads

**Proposed starting matrix.** Lengths are exact post-template tokens per request. Concurrency is client offered load. Approve prompts, request counts, arrivals, and stops before timing. Test normal EOS separately.

| Workload | Input tokens per request | Output tokens per request | Offered concurrency |
| --- | ---: | ---: | ---: |
| W1 | 128 | 128 | 1 |
| W2 | 2,048 | 512 | 1 |
| W3 | 8,192 | 1,024 | 4 |
| W4 | 32,768 | 1,024 | 16 |
| W5 | 32,768 | 4,096 | 32 |
| W6 | 131,072 | 1,024 | 4 |

Start W1–W6 with disjoint prefixes and MTP off. Keep [W3-prefix](prefix-cache.md#w3-prefix), `W3-MTP`, and media variants separate. Freeze media bytes, dimensions, sampled frames, timestamps, preprocessing, positions, and output budget. Retain failed outcomes under the original workload name.

## Measurement

**Required reporting.** For each workload and engine report output tokens/s, TTFT, TPOT, and end-to-end latency. Retain input and output counts and failed requests.

**Proposed timing protocol.** Use one streaming client and monotonic clock for both engines. Time cohorts from first submission to last terminal response. Preserve submission, first and last token, intermediate token events when observable, and completion times per request. Exclude empty events from token counts. Require every request to finish at its specified length.

| Metric | Definition for one measured cohort |
| --- | --- |
| Output tokens/s | Sum of emitted output tokens divided by elapsed cohort seconds. Exclude input, padding, and rejected drafts. |
| TTFT (ms) | Time from request submission to its first output token, including queueing and prefill. |
| TPOT (ms/token) | `(last token time − first token time) / (output token count − 1)`, per request. Mark unavailable for fewer than two tokens. |
| End-to-end latency (ms) | Time from request submission to its terminal response. Record stream chunking when multiple tokens arrive together. |
| Requests/s and aggregate tokens/s | Completed requests per cohort second, and total input plus output tokens per cohort second, respectively. |

Separate startup, model load, JIT compilation, and graph capture from steady execution. Warm both engines under the declared cache policy. Reset reusable prefixes before a cold-prefix measurement. Use the same workload order or a recorded counterbalanced order. **Proposed repeat policy.** Collect at least five measured cohorts per engine and workload after warmup. Preserve each cohort's output tokens/s. Report the median and range across cohorts. Report p50 and p95 for TTFT, TPOT, and end-to-end latency over requests per workload, with sample counts and the percentile method.

Report one row per engine, workload, and mode. Include absolute metrics, successful and attempted requests, and the agreed limits. For paired runs, also report the candidate/baseline throughput ratio and latency differences. For workloads with open limits, set numerical margins from repeated baseline measurements before candidate timing.

For offline timing, omit `--top-logits`. Match `--warmup` and `--repeats` to the reference workload. `load_ms` covers model initialization. `request_ms` includes request allocation and any graph capture. Allocation metrics include resident request storage on the serial graph path. Keep the JSON, invocation, binary hash, and device UUID. [Generator measurement](../apps/sglang-generate.cpp#L221)<!-- prefix: "Measurements" -->

### Native text completions

Reuse frozen W1–W6 `benchmark` requests, token-ID SSE clients, and timing definitions under the [new launch policy](#scheduler-and-reporting-policy). Check repeat IDs within each engine. Compare cross-engine output IDs only for diagnosis. Run [text verification](text-serving.md#verification) separately.

For raw text, freeze UTF-8 request bytes, checkpoint and tokenizer hashes, serving flags, output budget, and stop IDs. Match prompt IDs across engines. Time cold tokenizer load separately from steady tokenization. Declare streaming, usage, and diagnostic ID settings. Visible text can follow the first output ID. A text event can span several IDs, and an ID can produce no text. Count output tokens from usage or diagnostic IDs. Use diagnostic ID timestamps for TTFT and TPOT. Mark both metrics unavailable when the corresponding timestamps are missing. Report time to first visible text separately. Keep raw text measurements in separate result rows.

### Frozen W1 and W2 timing

The [GPU3 protocol](../benchmarks/streaming/README.md#frozen-measurement-protocol) used native `--max-active 1` and vLLM `--max-num-batched-tokens 2560 --max-num-seqs 1`. Verify input hashes. For native runs, set `--url` and a fresh `--output`. Omit `--reference`.

Validate every stream and match repeat IDs within each engine. Require the native-to-vLLM median throughput ratio to exceed `strict_throughput_ratio_gt`. Keep native p95 TTFT, TPOT, and end-to-end latency within `p95_latency_ceilings_ms`.

On November 19, 2026, the candidate matched `60677ec` IDs. Both passed W1. W2 missed only p95 TTFT (base 395.348 ms, candidate 366.781 ms, limit 80.305 ms).

On November 20, 2026, `9d047fe` passed all four W1 gates on repeat after a TTFT miss. W2 missed only p95 TTFT (368.736 ms against 80.258 ms).

On B200 GPU 6, the four-row W2 candidate matched the base's 512 full BF16 score rows and greedy IDs. In clean native-only serving, p95 TTFT fell from 367.158 to 277.963 ms and median throughput rose 2.16%.

### Fused FP8 decode

Match native control IDs and full BF16 logits on W1 and W2 with the same checkpoint, precision, and [GQA path](batch-graphs.md#page-parallel-gqa-decode). Check eager decode and fresh and reused graphs. Exercise exact and direct GQA at their [dispatch bounds](development.md#runtime-path). Run the [prefill](development.md#chunked-prefill-verification), [streaming](development.md#streaming-verification), and [MTP](mtp.md#verification) checks.

Trace actual forward group sizes and row modes. Confirm fused launches for an eligible single target decode row. Check packed projections for eligible groups of two through five rows. Include a group containing decode and single-row prefill. Compare shape and alignment fallbacks with the separate native operations. For steady decode timing, match execution mode, context length, and KV capacity. Record graph captures and replays.

Compare fused SwiGLU BF16 results with identical and disjoint gate and up output spans. Require `cudaErrorInvalidValue` before launch for identical gate and up spans in fallback, partial output overlaps, and output overlaps with inputs.

The studies in `artifacts/fp8-decode-next/report.md` and `artifacts/decode-combined/report.md` used earlier source. Qualify the promoted build with the serial and concurrent fixtures under the new launch policy.

### Exact decode bandwidth

Match the published native control's BF16 projections, E4M3 payloads, and FP32 scales byte for byte. Include nonfinite inputs and shape and alignment fallbacks. Trace fresh and reused graphs. Compare full BF16 logits and greedy IDs on shared W1/W2 histories within each policy. Run the [MTP checks](mtp.md#verification).

Historical serial HTTP used `--max-active 1 --mtp-steps 0` and compared within each prefill policy. Use the [frozen GPU3 criteria](#frozen-w1-and-w2-timing) to reproduce that comparison.

Record visible lengths and KV capacity. Report ordered fallback timing separately if measured.

An earlier GPU1 run compared the unsplit build with the Unit13 runtime. Each arm used three warmups and nine measured requests per workload. W1/W2 median output tokens/s rose from 91.88/35.16 to 95.59/35.70 for exact and from 122.84/57.27 to 129.51/58.70 for tensor-core. All four passed native timing checks. P95 TPOT and end-to-end latency fell throughout. P95 TTFT rose up to 1.00 ms. See `artifacts/exact-decode-after-seed/runs/http.6acmFcnP/`.

### Chunked prefill

Run the [prefill acceptance checks](development.md#chunked-prefill-verification) before timing. For exact prefill, match full BF16 logits and committed KV and GDN state with the row-wise control. Time device prompt processing with CUDA events through the operation before the first LM head. Measure HTTP TTFT from submission to the first token. Record tile sizes, tails, active lanes, and decode token gaps during mixed steps.

Scratch uses 31,944,704 bytes at 128 rows and zero at one row. Generator `peak_device_allocation_bytes` counts logical model, request, and scratch bytes. Batch `workspace_bytes` tracks the largest lane tile's scratch. `request_bytes` counts live request storage after each step. These counters exclude idle backing. Live reservations can exceed [pool thresholds](development.md#constraints-and-failure-behavior).

Record pool used and reserved bytes after synchronization. Require zero live scratch after retirement, including recovered free-enqueue failures. Report peak HBM, headroom, lane mix, and KV capacity.

Use a fresh process for cold allocator timing. Warm pools with a multirow request. Record tiles, lanes, and order. Report model load, tokenizer warmup, graph state, and prefix state separately.

The historical GPU6 prototype reduced W1 and W2 CUDA prefill time against the row-wise native control. It missed all four paired vLLM targets on both workloads. Qualify each production configuration under its applicable protocol.

### FP8 tensor-core prefill

Compare both `--fp8-prefill` policies on one binary with MTP off. Match measurement controls and KV and GDN policies. Record commands, source and binary hashes, toolchain, and driver. Trace projection shapes, grids, shared memory, and fallback separately.

Run [exact prefill checks](development.md#chunked-prefill-verification) with default and explicit `exact` in both binaries. Compare tensor-core projections with an independent reference under predefined tolerances. Match fallback and exact projections bitwise. Cover full chunks, tails, mixed steps, and unsupported dispatch. Check repeated tokens and finish reasons.

For arithmetic-preserving [prefill fusions](development.md#runtime-path), match full BF16 logits, committed KV and GDN state, and staged BF16 and FP8 values bitwise. Hold the policy and tile schedule fixed. Include raw MXFP8 exponents 0 and 255. Cover single rows, two- and four-row tiles, page crossings, positive MTP depth under both policies, and graph and eager decode.

Use the [GQA verification criteria and measurements](flash-prefill.md#verification) for flash, ordered fallback, and protected single rows.

For cross-policy comparisons, check full BF16 logits within predefined tolerances on identical histories and prefill boundaries. Use shared teacher-forced tokens after greedy divergence. Report logit errors, ranks, top-two margins, free-running IDs, and first divergence.

Check parity against pinned vLLM on identical histories. Record its execution mode and KV and GDN policies. Record how diagnostic rows were selected. Use frozen fixtures for workload-level results.

Run [serial tile checks](text-serving.md#serial-prefill-verification) before timing. Match BF16 across projection tiles on identical inputs. Follow the [prefill protocol](#chunked-prefill). Report native deltas and gates per policy. Keep [frozen timings](#frozen-w1-and-w2-timing) historical.

The post-chat GPU3 run used tensor-core prefill, serial graphs, and MTP off. Each workload and arm used three warmups and nine measurements. Both arms returned identical IDs.

| Workload and arm | Median output tok/s | P95 TTFT (ms) | P95 TPOT (ms/token) | P95 E2E (ms) | Gates met |
| --- | ---: | ---: | ---: | ---: | ---: |
| W1 chat control | 128.976 | 27.798 | 7.596 | 992.525 | 3/4 |
| W1 fusion | 129.935 | 20.427 | 7.609 | 986.680 | 4/4 |
| W1 frozen limits | >117.037 | ≤21.637 | ≤8.625 | ≤1116.170 | |
| W2 chat control | 58.526 | 1148.397 | 14.878 | 8750.750 | 0/4 |
| W2 fusion | 61.250 | 753.935 | 14.890 | 8361.881 | 0/4 |
| W2 frozen limits | >115.815 | ≤80.305 | ≤8.671 | ≤4510.901 | |

The verifier used unrounded limits. Both W2 arms missed all four gates in this run. See `artifacts/tc-prefill-fusion-after-chat/results/performance/comparison.json`.

### Native concurrent workloads

Verify [streaming](development.md#streaming-verification), [K0 scheduling](#scheduler-and-reporting-policy), offered concurrency, and [decode width and replay](batch-graphs.md#timing).

Historical W3 missed all gates on GPU2 (63.779 IDs/s, 28,672 matching IDs), `w3-fused` (144.84), and GPU3 `9d047fe` (90.519 versus vLLM’s 373.931, 4,096 matching IDs). [Historical protocol](../benchmarks/concurrent/README.md#conditions-and-gates) · `artifacts/takeover-w3/report.md`

GPU6 W3 replay rose from 5.69 to 14.91 IDs/s (4,096 matching IDs). `artifacts/long-batch-graph-release/report.md`

GPU3 W3 pairing raised median IDs/s from 103.724 to 111.340 (+7.34%). Pinned vLLM reached 374.427. All 4,096 IDs matched. Both native arms missed all four frozen gates. `artifacts/takeover-w3-gqa-perf/report.md`

On GPU4, the `dad987dd` W3 [grouped head](batch-graphs.md#runtime-path) raised median throughput by 3.39–3.63% to 138.843–138.849 IDs/s versus pinned vLLM's 374.187. All 143,360 IDs matched. Both candidate arms missed all four gates. `artifacts/takeover-w3-lmhead-code/flash-release/REPORT.md`

The prior W4 singleton spent 54.245 of 58.657 s TTFT in protected single-row GQA. `artifacts/takeover-w4-diagnostic/report.md`

GPU3 singleton median W4 TTFT fell from 58.701 to 29.214 s across three fresh-process pairs. First ID `760` matched.

vLLM completed GPU3 W4. Both 16-lane native runs timed out at 90 s with zero output. Completed prefill rose from 163,840 to 196,608 of 524,288 rows. Native gates stayed unscored. See `artifacts/takeover-w4-long-tiled/report.md`.

GPU5 operator median GQA fell from 222.465 to 101.679 ms for 1,024 rows at 32,768 visible tokens. See `artifacts/takeover-long-prefill-code/handoff.md`.

[Wide long decode checks](long-paged-decode.md#verification) cover direct controls and HTTP cohorts.

See [W6 prefill diagnostics](flash-prefill.md#w6-warmups-and-diagnostics), [experimental TMEM quality limits](flash-prefill.md#tmem-checkpoint-verification), and the [TMEM prototype](flash-prefill.md#double-buffered-tmem-prototype).

### Native MTP

Run the [MTP checks](mtp.md#verification) before timing. Use same-width eager K0 to isolate speculation. Measure K0 replay separately. Match prompt IDs, arrivals, budgets, stop policy, and outputs. Record binary and checkpoint hashes, GPU UUID, dispatch, and device and pinned host bytes. Disable tracing and `--top-logits`.

To isolate layer-major verification, compare with row-wise MTP at the same depth. To isolate tiled seeding, keep the verifier, exact precision, and depth fixed. Time target prefill and draft KV seeding through workspace release, before the first LM head. Record HTTP TTFT and request latency separately. Include short, tile-boundary, and long prompts. Record tile sizes, pool state, and output length.

Alternate paired arms on one reserved B200. Match warmups and repeats. Report samples, medians, ranges, and paired ratios. [W3-MTP](../benchmarks/variants/README.md) uses four 8,192-token prompts with 1,024 outputs each. K1 starts beyond the [graph limit](mtp.md#target-verification-graph). The matched run remains pending. Use `--max-active 32 --max-num-batched-tokens 32768`.

Positive depth adds counters for the generator's first measured request. `mtp_drafted_tokens` counts proposals read back. `mtp_accepted_tokens` counts matching proposals appended to the output. `mtp_rounds` counts rounds that produced at least one proposal. Count committed `output_token_ids` including first, correction, and bonus tokens. Report accepted/drafted and accepted/rounds for positive denominators. [JSON fields](../apps/sglang-generate.cpp#L247)<!-- prefix: "void write_r" --> · [Proposal loop](../src/decoder_verify.cpp#L130)<!-- prefix: "bool MtpRoun" --> · [Generator counters](../src/decoder_batch.cpp#L980)<!-- prefix: "result.mtp_r" -->

Use `--trace-batches` at any capacity in a separate run. Check `accepted ≤ compared ≤ drafted` and `drafted ≤ rounds × K`. Count the first mismatch as compared. Computed rows can exceed consumed rows after mismatch or stop. In normal completion, consumed decode rows equal committed outputs minus the initial token. Zero or one output requires zero drafts. [Consumption](../src/decoder_verify.cpp#L180)<!-- prefix: "bool MtpRoun" -->

Batch traces sum verifier lanes and computed rows across all groups in the step. Per slot, `drafted` counts proposals read back and `accepted` counts matching outputs committed before publication. `verified` counts computed target rows. `replayed` counts rows replayed on partial consumption. A one-output eager `boundary` records zero speculative counts. Client-visible output can be shorter after cancellation. [Batch trace](../apps/server/batch_service.cpp#L55)<!-- prefix: "void trace_v" -->

See the [verifier storage bound](mtp.md#constraints-and-failure-behavior). The generator adds peak request, prefill, and selection bytes to model bytes. Batch `peak_request_bytes` includes prefill scratch or live request storage before retirement. Record peak HBM separately. [Generator accounting](../src/decoder_batch.cpp#L977)<!-- prefix: "result.peak_" --> · [Batch accounting](../src/decoder_batch.cpp#L902)<!-- prefix: "current.peak" -->

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
