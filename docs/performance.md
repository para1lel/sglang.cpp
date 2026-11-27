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

Use MTP4, 32,768 tokens per step, eight slots per GPU, and cold prefixes.

- **Pinned vLLM.** Set `--max-num-batched-tokens 32768 --max-num-scheduled-tokens 32768` and enable `--async-scheduling`. Match `--max-num-seqs` to the active capacity. Verify MTP4 and the effective aligned GDN cache settings at the API and worker.
- **Native.** Use the default direct GPU LMCache path. The paired driver starts a private LMCache 0.5.8 daemon with 64 GiB CPU L1 on the same physical GPU. [Daemon launcher](../benchmarks/paired/lmcache_arm.sh#L36)<!-- prefix: "start_lmcach" -->

The pinned vLLM arm uses its local prefix cache. Verify `kv_transfer=false` and `kv_offloading_size=null` in the worker receipt. The same check requires zero CPU weight offload. [Worker receipt check](https://github.com/para1lel/sglang.cpp/blob/03ee22d849e6ef7bca723595a7c8e63c47187749/benchmarks/paired/saved_evidence.py#L223)<!-- prefix: "and worker[\"" -->

vLLM scans running requests before FCFS admission, including unfinished prompts. Native [reserves ready decode rows before FCFS prefill](batch-graphs.md#runtime-path). [Pinned vLLM scheduler](https://github.com/vllm-project/vllm/blob/81a8cb063e6247fe254d7caf7e98a6691d71022a/vllm/v1/core/sched/scheduler.py#L767)

Report nondefault settings, effective limits, offered and observed concurrency, input lengths, context and output budgets, and [admission failures](batch-graphs.md#resident-memory-admission) separately from timing. Acceptance requires a fresh matched run with every required mode.

## GQA cache comparison

Native [MXFP8 GQA](development.md#p2-gqa-cache-contract) uses E4M3 payloads with E8M0 scales per 32 channels. Pass `--kv-cache-dtype fp8_e4m3` to vLLM and verify plain E4M3 GQA dispatch. The pin defaults per-tensor K and V scales to `1.0` when the checkpoint supplies neither. This checkpoint has no attention scales. [vLLM cache setting](https://github.com/vllm-project/vllm/blob/81a8cb063e6247fe254d7caf7e98a6691d71022a/vllm/config/cache.py#L251)<!-- prefix: "CUDA 11.8+ s" -->

Record actual KV format and scale policy, Q dtype and scaling, attention backend and fallback, and GDN state dtypes. Match GDN dtypes and record FP8 weight scales separately. Compare cross-engine logits and output IDs for diagnosis. Require each engine to meet token counts, finish conditions, and the stream contract. [vLLM scale initialization](https://github.com/vllm-project/vllm/blob/81a8cb063e6247fe254d7caf7e98a6691d71022a/vllm/model_executor/layers/quantization/kv_cache.py#L118)<!-- prefix: "elif layer.k" -->

## Named workloads

Lengths are exact post-template token counts. Scale request counts by GPU count. Use distinct frozen lanes and test normal EOS separately.

| Workload | Input tokens per request | Output tokens per request | Offered requests per GPU |
| --- | ---: | ---: | ---: |
| W1 | 128 | 128 | 1 |
| W2 | 2,048 | 512 | 1 |
| W3 | 8,192 | 1,024 | 4 |
| W4 | 32,768 | 4,096 | 8 |
| W5 | 131,072 | 1,024 | 4 |

TP1 W4 uses the first eight archived W5 requests. TP1 W5 uses all four archived W6 requests. Use the frozen additive corpus for [TP2](serving/tp2.md#acceptance-target). Give the [warm-prefix diagnostic](../benchmarks/paired/README.md#prefix-reuse-diagnostic) and media variants separate rows.

## Measurement

Require native makespan ≤1.10× pinned vLLM on every W1–W5 row at each TP width. Run one warmup and one measured cohort per row and engine. Apply the gate to the measured cohort and stop at the first miss. W4 can open a campaign. Run W5 only after W4 passes in that campaign. Advance through W1–W3 in order on passes. Verify device, effective controls, stream identities, counts, finish, usage, and drain. Require TP1 measured IDs to match each engine's warmup IDs. TP2 output content is diagnostic. [Matched K4 timing protocol](../benchmarks/paired/README.md#matched-k4-timing-protocol)

Before timing each TP2 row, run `benchmarks/tp2/run.sh qualify` on a separate untimed cohort and require the native route coverage checks to pass.

Use one streaming client and monotonic clock from earliest submission to last valid `[DONE]`. Preconnect lanes, release one barrier, and send all bodies within 50 ms and before any completion. Retain submit times, headers, token events, completion times, counts, and failures. Exclude [SSE keepalives](http-server.md#completion-streams) and empty events from token counts.

Report output tokens/s, TTFT, TPOT, end-to-end latency, and whether every lane starts generating before the first completion for diagnosis.

| Metric | Definition for one measured cohort |
| --- | --- |
| Output tokens/s | Sum of emitted output tokens divided by elapsed cohort seconds. Exclude input, padding, and rejected drafts. |
| TTFT (ms) | Submission to first output token, including queueing and prefill. Unavailable for zero output tokens. [Client calculation](../benchmarks/streaming/client.py#L207)<!-- prefix: "\"ttft_ms\": (" --> |
| TPOT (ms/token) | `(last token time − first token time) / (output token count − 1)`, per request. Mark unavailable for fewer than two tokens. |
| End-to-end latency (ms) | Time from request submission to its terminal response. Record stream chunking when multiple tokens arrive together. |
| Requests/s and aggregate tokens/s | Completed requests per cohort second, and total input plus output tokens per cohort second, respectively. |

Exclude startup, model load, JIT, and graph capture. Drain and acknowledge resets before each cohort. Require zero cached tokens and unchanged hit counters. For each native LMCache daemon, verify periodic and terminal stores, written positions, transfers, zero GQA/GDN/draft reuse, and no new declines or I/O errors. Use settled counters for cache accounting and stream timestamps for the timing ratio. [Cold reset](../benchmarks/paired/cold_cache.py#L112)<!-- prefix: "def before_c" --> · [Drain verification](../benchmarks/paired/cold_cache.py#L134)<!-- prefix: "def verify_l" -->

Report absolute metrics and successful and attempted requests for each engine, row, and mode. Give the latency sample count and percentile method.

For new offline timing, use `--warmup 1 --repeats 1` and omit `--top-logits`. `load_ms` covers model initialization. `request_ms` includes request allocation and graph capture. Allocation metrics include resident storage on the serial graph path. Keep JSON, invocation, binary hash, and device UUID. [Generator measurement](../apps/sglang-generate.cpp#L247)<!-- prefix: "Measurements" -->

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

In separate `--trace-batches` runs, require `accepted ≤ compared ≤ drafted` and `drafted ≤ rounds × K`. A first mismatch counts as compared. Zero or one output requires zero drafts. With normal completion, consumed decode rows equal committed outputs minus the initial token. Computed or replayed rows can exceed consumed rows. Report accepted/drafted and accepted/rounds for positive denominators. [Batch trace](../include/sglang/server/batch_trace.h#L14)<!-- prefix: "inline void " --> · [Archived counter definitions and probes](https://github.com/para1lel/sglang.cpp/blob/a5974ba4eb99bd8e8f890c2e7262a27dbc3aace1/docs/performance.md#native-mtp)

## Correctness, memory, and diagnosis

Run [native correctness checks](development.md#verification) separately from timed runs when instrumentation affects latency. Record HBM after load and peak, capacity and headroom, initial cache state, evictions, replay, and reclamation. Keep sampling, grammar, and media diagnostics separate from frozen workloads. [Sampling](text-serving.md#verification) · [Constraints](text-serving.md#finite-choice-verification) · [Multimodal](multimodal-decoder.md#verification)

Collect kernel, scheduler, queue, and utilization diagnostics separately. Record the collection method and link each change to its measured bottleneck. [Native scheduler samples](prefix-cache.md#native-http-controls) · [Archived vLLM trace verifier](https://github.com/para1lel/sglang.cpp/blob/a5974ba4eb99bd8e8f890c2e7262a27dbc3aace1/benchmarks/variants/verify.py#L407)<!-- prefix: "def verify(r" -->

Retain the run manifest, commands, environment, fixture hashes, raw samples, and failures in ignored `artifacts/` storage. A missing counter is unavailable. A missing baseline, failed correctness check, or changed workload leaves that performance claim open.

## Matched K4 evidence

The archived TP1 release `c701d294b3823c655c6adc38bbd87d077147bbcc` passed all five rows with direct GPU LMCache 0.5.8. W5 took 38.396171943 s native and 35.940895116 s pinned vLLM (1.068314293×). [Frozen TP1 results and SSE replay](https://github.com/para1lel/sglang.cpp/blob/a5974ba4eb99bd8e8f890c2e7262a27dbc3aace1/docs/performance.md#matched-k4-evidence)

<a id="traced-one-output-prefill-diagnostic"></a>
The traced prefill comparison is diagnostic. [Frozen one-output W5 diagnostic](https://github.com/para1lel/sglang.cpp/blob/a5974ba4eb99bd8e8f890c2e7262a27dbc3aace1/docs/performance.md#traced-one-output-prefill-diagnostic) · [TP2 run evidence](serving/tp2.md)

<a id="current-single-round-v5"></a>
### Archived single-round v5

The former device-resident cache also passed W1–W5. Keep this result under its original cache control. [Frozen v5 results](https://github.com/para1lel/sglang.cpp/blob/a5974ba4eb99bd8e8f890c2e7262a27dbc3aace1/docs/performance.md#archived-single-round-v5)

### Archived multi-cohort v4

Final v4 passed all five rows. W1–W2 used three warmups and nine measurements, W3–W5 two and five. W5 medians were 38.170671560 seconds native and 35.974923597 seconds vLLM (1.061036×). [Full v4 results](https://github.com/para1lel/sglang.cpp/blob/6f4d54d3a07ea8ad5bae68ea020105b8ecaeba93/docs/performance.md#matched-k4-evidence) · [Measured v4 release](https://github.com/para1lel/sglang.cpp/commit/b8764ee28523c7223865a0e1de8ebc8041ca857a)

The previous v4 campaign passed W1–W4 and missed W5. [Prior v4 evidence](https://github.com/para1lel/sglang.cpp/blob/17d8634f1f9677bf4b2bc3dea93b7e5ca58eca5a/docs/performance.md#matched-k4-evidence)
