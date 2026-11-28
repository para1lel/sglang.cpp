# Matched B200 benchmark agreement

Compare native TP1 and TP2 independently against pinned vLLM at the same width. Use the same B200 devices and immutable checkpoint. Freeze workloads, controls, and arm order. [Paired driver](../benchmarks/paired/README.md#matched-k4-timing-protocol) · [TP2 topology and corpus](serving/tp2.md#acceptance-target)

## Implementations and controls

| Part | Responsibility | Source |
| --- | --- | --- |
| Independent baseline | Run vLLM in its own environment. Record the checkout, installed wheel and native binary identities, dependency versions, and effective launch settings. | `vllm-project/vllm` at `81a8cb063e6247fe254d7caf7e98a6691d71022a`. |
| Candidate | Run the project binary built by XMake with the default Pixi toolchain. Record source, build options, and executable identity. | [Development contract](development.md). |
| Architectural reference | Interpret service and cache behavior. Any separate SGLang timing needs its own result label. | `sgl-project/sglang` at `f04d886bf79aa3d6237cbb12a3311838eb9f9bd8`. |

Run each engine alone on the selected UUIDs in its own environment. Record drivers, runtimes, host and client settings, clocks, power, competing GPU activity, and software differences.

Match checkpoint, tokenizer, template, processor, weight precision, sampling, cache, MTP, and media preprocessing. Freeze request bytes, input IDs, arrivals, seeds, and stops. Record dispatch, token counts, and unsupported modes.

## Scheduler and reporting policy

Use MTP4, 32,768 tokens per step, and eight slots per GPU. W1–W5 start each cohort with cold prefixes. W6 starts cold and reuses prefixes inside its measured chain.

- **Pinned vLLM.** Set `--max-num-batched-tokens 32768 --max-num-scheduled-tokens 32768` and enable `--async-scheduling`. Match `--max-num-seqs` to the active capacity. Verify MTP4 and the effective aligned GDN cache settings at the API and worker.
- **Native.** Use the default 0.88 GPU memory utilization and automatic GQA page and GDN checkpoint pool capacities. The private LMCache 0.5.8 daemon uses 64 GiB CPU L1 at TP1 and 128 GiB shared by both UUIDs at TP2. [Campaign defaults](../benchmarks/paired/workload_suite.py#L37)<!-- prefix: "HBM_POOL_DEF" --> · [TP1 daemon launcher](../benchmarks/paired/lmcache_arm.sh#L36)<!-- prefix: "start_lmcach" -->

The pinned vLLM arm uses its local prefix cache. Verify `kv_transfer=false` and `kv_offloading_size=null` in the worker receipt. The same check requires zero CPU weight offload. [Worker receipt check](https://github.com/para1lel/sglang.cpp/blob/03ee22d849e6ef7bca723595a7c8e63c47187749/benchmarks/paired/saved_evidence.py#L223)<!-- prefix: "and worker[\"" -->

vLLM scans running requests before FCFS admission, including unfinished prompts. Native [reserves ready decode rows before FCFS prefill](batch-graphs.md#runtime-path). [Pinned vLLM scheduler](https://github.com/vllm-project/vllm/blob/81a8cb063e6247fe254d7caf7e98a6691d71022a/vllm/v1/core/sched/scheduler.py#L767)

Report nondefault settings, effective limits, offered and observed concurrency, input lengths, context and output budgets, and [admission failures](batch-graphs.md#resident-memory-admission) separately from timing. Acceptance requires a fresh matched run with every required mode.

## GQA cache comparison

Native [MXFP8 GQA](development.md#p2-gqa-cache-contract) uses E4M3 payloads with E8M0 scales per 32 channels. Pass `--kv-cache-dtype fp8_e4m3` to vLLM and verify plain E4M3 GQA dispatch. The pin defaults per-tensor K and V scales to `1.0` when the checkpoint supplies neither. This checkpoint has no attention scales. [vLLM cache setting](https://github.com/vllm-project/vllm/blob/81a8cb063e6247fe254d7caf7e98a6691d71022a/vllm/config/cache.py#L251)<!-- prefix: "CUDA 11.8+ s" -->

Record actual KV format and scale policy, Q dtype and scaling, attention backend and fallback, and GDN state dtypes. Match GDN dtypes and record FP8 weight scales separately. Compare cross-engine logits and output IDs for diagnosis. Require each engine to meet token counts, finish conditions, and the stream contract. [vLLM scale initialization](https://github.com/vllm-project/vllm/blob/81a8cb063e6247fe254d7caf7e98a6691d71022a/vllm/model_executor/layers/quantization/kv_cache.py#L118)<!-- prefix: "elif layer.k" -->

## Named workloads

Lengths are exact post-template token counts. Scale W1–W5 request counts by GPU count. Use distinct frozen lanes and test normal EOS separately.

| Workload | Input tokens per request | Output tokens per request | Offered requests per GPU |
| --- | ---: | ---: | ---: |
| W1 | 128 | 128 | 1 |
| W2 | 2,048 | 512 | 1 |
| W3 | 8,192 | 1,024 | 4 |
| W4 | 32,768 | 4,096 | 8 |
| W5 | 131,072 | 1,024 | 4 |

TP1 W4 uses the first eight archived W5 requests. TP1 W5 uses all four archived W6 requests. Use the frozen additive corpus for [TP2](serving/tp2.md#acceptance-target). Give the [warm-prefix diagnostic](../benchmarks/paired/README.md#prefix-reuse-diagnostic) and media variants separate rows.

### W6 serial history

Use `k = 1,024` tokens. Each arm at each width sends one independent warmup with 203,776 input tokens and 1,024 output tokens. Drain and reset prefix state after warmup. The measured chain starts with empty history.

For each of 20 requests, append 9,216 frozen IDs to that engine’s history. Append its 1,024 emitted IDs after the response. Input `i` contains `10,240 × i − 1,024` IDs, from 9,216 to 203,776. The final request reaches 204,800 tokens with output. Both engines share the frozen additions and sampling controls.

Validate usage and `[DONE]`. Dispatch immediately after HTTP EOF, keeping prefix state. Time the first POST through the twentieth EOF on one monotonic clock. Retain warmup separately, actual IDs and usage, and both `[DONE]` and EOF clocks. An invalid response invalidates the chain.

Use the [paired driver’s shared controls](../benchmarks/paired/README.md#matched-k4-timing-protocol) and the same campaign as that width’s passing W5.

Prepare and verify the W6 fixture once in `PAIRED_FIXTURES`. [Fixture validation](../benchmarks/serial/fixture.py#L45)<!-- prefix: "def load_inp" --> · [Capture and replay](../benchmarks/serial/chain.py#L157)<!-- prefix: "def handle(" -->

```bash
python -B -m benchmarks.serial.fixture prepare "$CHECKPOINT/tokenizer.json" "$PAIRED_FIXTURES"
python -B -m benchmarks.serial.fixture verify "$PAIRED_FIXTURES"
bash benchmarks/paired/run_k4.sh "$NATIVE_BINARY" "$GPU0" "$TP1_CAMPAIGN" W6
bash benchmarks/tp2/run.sh measure "$GPU0" "$GPU1" W6 "$NATIVE_BINARY" "$TP2_CAMPAIGN"
```

Keep each engine's `warmup-01.json.gz` and `measured-01.json.gz` with the comparison and cache receipts.

## Measurement

Require native makespan ≤1.10× pinned vLLM for every row at each width. Run one warmup and one measurement per arm. Stop at the first miss. W4 may open the campaign and must pass before W5. Advance through W1–W3 in order on passes. Verify devices, settings, stream identities, counts, finish, usage, and drain. TP1 W1–W5 require each arm’s measured IDs to match its warmup. TP2 output content is diagnostic. [Matched K4 timing protocol](../benchmarks/paired/README.md#matched-k4-timing-protocol)

Use `benchmarks/tp2/run.sh qualify` separately for readiness and a single native W1-shaped functional request. Use `measure` for full row and rank-cache receipts. W1 includes a separate native trace arm for K4 route activity. [TP2 cohort assessment](../benchmarks/tp2/cohorts.py#L138)<!-- prefix: "def assess(d" -->

For W1–W5, time the earliest submission through the last valid `[DONE]` on one monotonic clock. Preconnect lanes and release one barrier. Send all bodies within 50 ms and before any completion. Retain submissions, headers, token events, counts, failures, and EOFs. EOF closes the stream and bounds cache accounting. Exclude [keepalives](http-server.md#completion-streams) and empty events from token counts. W6 uses the [first POST through final EOF](#w6-serial-history). [Cohort score](../benchmarks/paired/makespan.py#L21)<!-- prefix: "def cohort_m" --> · [Cache receipt interval](../benchmarks/paired/cold_cache.py#L258)<!-- prefix: "def replay(" -->

Report output tokens/s, TTFT, TPOT, end-to-end latency, and whether every lane starts generating before the first completion for diagnosis.

| Metric | Definition for one measured cohort |
| --- | --- |
| Output tokens/s | Sum of emitted output tokens divided by elapsed cohort seconds. Exclude input, padding, and rejected drafts. |
| TTFT (ms) | Submission to first output token, including queueing and prefill. Unavailable for zero output tokens. [Client calculation](../benchmarks/streaming/client.py#L207)<!-- prefix: "\"ttft_ms\": (" --> |
| TPOT (ms/token) | `(last token time − first token time) / (output token count − 1)`, per request. Mark unavailable for fewer than two tokens. |
| End-to-end latency (ms) | Time from request submission to its terminal response. Record stream chunking when multiple tokens arrive together. |
| Requests/s and aggregate tokens/s | Completed requests per cohort second, and total input plus output tokens per cohort second, respectively. |

Exclude startup, model load, JIT, and graph capture. Drain and acknowledge resets before each W1–W5 cohort and before the W6 warmup and measured chain. TP2 resets its shared daemon once. W1–W5 require zero cached tokens and unchanged hit counters. Use settled counters for cache accounting and stream timestamps for timing. [Cold reset](../benchmarks/paired/cold_cache.py#L155)<!-- prefix: "def before_c" -->

Native pool receipts require zero occupied entries and pages after reset and unchanged positive pool capacities. Require no pending stores at settlement and no daemon restores for a cold cohort. W6 can record cache hits and restores inside its measured chain. Report LMCache activity separately from GPU-resident reuse. [Pool receipt](../benchmarks/paired/cold_cache.py#L234)<!-- prefix: "def verify_p" -->

Report absolute metrics and successful and attempted requests for each engine, row, and mode. Give the latency sample count and percentile method.

For new offline timing, use `--warmup 1 --repeats 1` and omit `--top-logits`. `load_ms` covers model initialization. `request_ms` includes request allocation and graph capture. Allocation metrics include resident storage on the serial graph path. Keep JSON, invocation, binary hash, and device UUID. [Generator measurement](../apps/sglang-generate.cpp#L256)<!-- prefix: "Measurements" -->

### GPU pool engineering checks

Use the [prefix cache fixtures](prefix-cache.md#verification) at TP1 and TP2. Check cold prefill, HBM-only reuse, mixed-tier restore, and CPU reload on the same token history. Hold one TP2 STORE reply to verify that the backup stays pending until both ranks acknowledge it.

Check physical device allocations and frees in the server and daemon from readiness to shutdown start. Count IPC maps and module-static symbols separately. Retain the route receipts and exported trace outside the timed campaign.

### Operator qualification

Operators may reorder arithmetic within their numerical contract. Compare outputs and state with an independent mathematical reference under the operation's established error criteria. Check causality where the operation uses token history, and model quality on frozen inputs and shared token histories. Record free-running divergence separately. Require bitwise equality for copied bytes, exact replay, and comparisons that preserve arithmetic.

Use repeated matched measurements to choose the fastest validated implementation per supported case. Keep one production implementation per case. Use validated fallbacks where shape, device, or resources prevent the primary route. Keep frozen fixtures and timing gates unchanged.

### Native text completions

Run [text verification](text-serving.md#verification) separately. Freeze raw UTF-8 requests, tokenizer and checkpoint identities, settings, budgets, and stops. Match prompt IDs and separate cold tokenizer load. Count tokens from usage or diagnostic IDs. Calculate TTFT and TPOT from ID timestamps when available. Report first visible text separately.

### Serial W1 and W2 comparisons

The historical GPU3 serial protocol uses its own lane limits, throughput ratio, and p95 latency ceilings. Keep these diagnostics under their original settings and labels. [Frozen GPU3 protocol](../benchmarks/streaming/README.md#frozen-measurement-protocol)

### Fused FP8 decode

Match BF16 logits and IDs with native controls on W1 and W2. Compare eager, fresh graphs, and reused graphs at matching visible lengths and KV capacities. Cover exact and direct GQA bounds, groups of two through five, mixed steps, [packed K0 buckets](batch-graphs.md#packed-paged-gqa-decode), and shape and alignment fallbacks.

Compare fused SwiGLU BF16 results with identical and disjoint gate and up output spans. Require `cudaErrorInvalidValue` before launch for identical gate and up spans in fallback, partial output overlaps, and output overlaps with inputs.

### Exact decode bandwidth

Match BF16 projections, E4M3 payloads, and FP32 scales bitwise against the native control. Include nonfinite inputs, shape and alignment fallbacks, and fresh and reused graphs. Compare full logits and greedy IDs on shared W1/W2 histories. Run the [MTP checks](mtp.md#verification). For serial HTTP, use `--max-active 1 --mtp-steps 0` and apply the [serial criteria](#serial-w1-and-w2-comparisons) within each prefill policy.

### Chunked prefill

Run the [prefill checks](development.md#chunked-prefill-verification). Match exact-policy BF16 logits, KV, and GDN state with row-wise control. Measure device prefill through the operation before the first LM head and measure HTTP TTFT separately. Record tiles, tails, active lanes, and decode gaps.

Record peak HBM and [logical workspace and request bytes](native-prefill-workspace.md#constraints-and-failure-behavior). Record pool used and reserved bytes after synchronization. Require zero live scratch after retirement, including recovered free-enqueue failures. Use a fresh process for cold allocation timing.

### FP8 tensor-core prefill

Use `--mtp-steps 0` for the native control. Compare reordered arithmetic with an independent reference and require bitwise parity for exact projections and arithmetic-preserving fusions. Cover chunks, tails, mixed steps, unsupported dispatch, raw MXFP8 exponents 0 and 255, page crossings, and eager and graph decode. For changed fusions, verify staged values and committed state, including positive MTP depth. [Projection checks](fp8-prefill-projections.md#verification) · [GQA checks](flash-prefill.md#verification)

For cross-policy diagnostics, compare logits on identical token histories and prefill boundaries. After greedy divergence, use shared teacher-forced tokens. Report logit error, ranks, winner margins, labeled-token NLL, and first divergence. Run [serial tile checks](text-serving.md#serial-prefill-verification) before timing.

### Native concurrent workloads

Verify offered and observed concurrency, streamed output, decode width, and graph replay under the fixed scheduler. Use [streaming checks](development.md#streaming-verification), [batch timing](batch-graphs.md#timing), and [wide decode checks](long-paged-decode.md#verification).

### Native MTP

Run [MTP verification](mtp.md#verification). Compare with `--mtp-steps 0` at the same width and measure K0 graph replay separately. Match prompt IDs, arrivals, budgets, stops, and precision. Disable tracing and `--top-logits` for timing. Alternate arms and report samples, medians, ranges, and paired ratios.

For layer-major verification, keep depth fixed against a row-wise control. For tiled draft seeding, keep verifier and depth fixed. Time target prefill and draft seeding through workspace release before the first LM head. Include short, boundary, and long prompts. [Archived W3-MTP fixture](https://github.com/para1lel/sglang.cpp/blob/a5974ba4eb99bd8e8f890c2e7262a27dbc3aace1/benchmarks/variants/README.md) · [Graph limit](mtp.md#target-verification-graph)

In separate `--trace-batches` runs, require `accepted ≤ compared ≤ drafted` and `drafted ≤ rounds × K`. A first mismatch counts as compared. Zero or one output requires zero drafts. With normal completion, consumed decode rows equal committed outputs minus the initial token. Computed or replayed rows can exceed consumed rows. Report accepted/drafted and accepted/rounds for positive denominators. [Batch trace](../apps/server/detail/batch_trace.cpp#L11)<!-- prefix: "void trace_v" --> · [Archived counter definitions and probes](https://github.com/para1lel/sglang.cpp/blob/a5974ba4eb99bd8e8f890c2e7262a27dbc3aace1/docs/performance.md#native-mtp)

## Correctness, memory, and diagnosis

Run [native correctness checks](development.md#verification) separately from timed runs when instrumentation affects latency. Record HBM after load and peak, capacity and headroom, initial cache state, evictions, replay, and reclamation. Keep sampling, grammar, and media diagnostics separate from frozen workloads. [Sampling](text-serving.md#verification) · [Constraints](text-serving.md#finite-choice-verification) · [Multimodal](multimodal-decoder.md#verification)

Collect kernel, scheduler, queue, and utilization diagnostics separately. Record the collection method and link each change to its measured bottleneck. [Native scheduler samples](prefix-cache.md#native-http-controls) · [Archived vLLM trace verifier](https://github.com/para1lel/sglang.cpp/blob/a5974ba4eb99bd8e8f890c2e7262a27dbc3aace1/benchmarks/variants/verify.py#L407)<!-- prefix: "def verify(r" -->

Retain the run manifest, commands, environment, fixture hashes, raw samples, and failures in ignored `artifacts/` storage. A missing counter is unavailable. A missing baseline, failed correctness check, or changed workload leaves that performance claim open.

## Matched K4 evidence

Native source `b0596c25eab50dfbe4b5a340da3a06f84f89b339` used server SHA-256 `59748dd6424d97019d49360fe42d87ea9d6aaaf5472ed98cdfc19b41a3a806d8`. The reference was vLLM `81a8cb063e6247fe254d7caf7e98a6691d71022a`.

TP1 used B200 0 on NUMA 0, with server CPUs 24–33 and client CPUs 34–35. TP2 used B200s 4→5 on NUMA 1, with server CPUs 60–69 and client CPUs 70–71. Both engines used the same devices and CPU sets at each width.

Both engines used MTP4, a 32,768-token step cap, and eight slots per GPU. Native used 0.88 GPU memory utilization, automatic GPU pools, and one LMCache daemon per width. The daemon used 64 GiB at TP1 and 128 GiB at TP2. vLLM used local prefix caching with KV offload disabled.

All 12 measured native/vLLM ratios were within the 1.10× makespan limit.

| Width | Row | Requests completed per arm | Native (ns) | vLLM (ns) | Native / vLLM | Timing gate |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| TP1 | W1 | 1 | 587,873,203 | 717,222,183 | 0.819653× | Pass |
| TP1 | W2 | 1 | 1,386,329,951 | 1,948,254,502 | 0.711575× | Pass |
| TP1 | W3 | 4 | 4,260,391,054 | 4,135,764,552 | 1.030134× | Pass |
| TP1 | W4 | 8 | 26,417,963,763 | 27,929,400,877 | 0.945884× | Pass |
| TP1 | W5 | 4 | 38,807,917,973 | 36,727,422,658 | 1.056647× | Pass |
| TP1 | W6 | 20 | 71,744,251,350 | 66,839,293,143 | 1.073384× | Pass |
| TP2 | W1 | 2 | 515,768,012 | 560,978,526 | 0.919408× | Pass |
| TP2 | W2 | 2 | 1,381,815,049 | 1,312,484,124 | 1.052824× | Pass |
| TP2 | W3 | 8 | 3,922,045,906 | 4,115,852,250 | 0.952912× | Pass |
| TP2 | W4 | 16 | 27,621,177,840 | 26,302,791,966 | 1.050123× | Pass |
| TP2 | W5 | 8 | 41,007,480,757 | 39,870,936,066 | 1.028506× | Pass |
| TP2 | W6 | 20 | 53,711,372,451 | 52,972,333,926 | 1.013951× | Pass |

The campaign directory `artifacts/pool-w6/campaign/official-b0596c25eab5-59748dd6424d/` contains the saved streams, clock receipts, and placements. The independent replay record is `artifacts/pool-w6/verification/audit-formal-digest-b0596c25eab5.json`.
