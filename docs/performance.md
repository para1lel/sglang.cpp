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

Each arm sends 20 measured requests. For request `i` from 1 through 20, append 9,216 frozen input IDs to that engine's complete history. After its response, append all 1,024 emitted output IDs to the history. Inputs contain `10,240 × i − 1,024` IDs, from 9,216 to 203,776. The final input and output total 204,800 tokens. Both engines use the same frozen added IDs and sampling controls. Each arm carries its own emitted output IDs forward.

Validate final usage and `[DONE]`, then send the next request immediately after HTTP EOF. Keep prefix state between measured requests. Time from the first POST submission to the twentieth HTTP EOF on one monotonic clock, including handoffs. Save warmup separately. Retain added IDs, actual prompt and output IDs, usage, and separate `[DONE]` and EOF timestamps. A missing or invalid response invalidates the chain.

Use the same campaign identity as the passing W5 row at the corresponding width. The [paired driver runbook](../benchmarks/paired/README.md#matched-k4-timing-protocol) supplies common environment and checkpoint controls.

Prepare the W6 fixture once with the checkpoint tokenizer and verify it before reusing it. Use `PAIRED_FIXTURES` as the shared fixture directory. Measure each width in its existing campaign after that width's W5 pass. [Fixture validation](../benchmarks/serial/fixture.py#L45)<!-- prefix: "def load_inp" --> · [Capture and replay](../benchmarks/serial/chain.py#L157)<!-- prefix: "def handle(" -->

```bash
python -B -m benchmarks.serial.fixture prepare "$CHECKPOINT/tokenizer.json" "$PAIRED_FIXTURES"
python -B -m benchmarks.serial.fixture verify "$PAIRED_FIXTURES"
bash benchmarks/paired/run_k4.sh "$NATIVE_BINARY" "$GPU0" "$TP1_CAMPAIGN" W6
bash benchmarks/tp2/run.sh measure "$GPU0" "$GPU1" W6 "$NATIVE_BINARY" "$TP2_CAMPAIGN"
```

Keep each engine's `warmup-01.json.gz` and `measured-01.json.gz` with the comparison and cache receipts.

## Measurement

Require native makespan ≤1.10× pinned vLLM on every W1–W6 row independently at each TP width. Use one warmup and one measured cohort or chain per engine. Apply the gate to the measured makespan and stop at the first miss. W4 can open a campaign. Run W5 only after W4 passes in that campaign. Advance through W1–W3 in order on passes. Verify device, effective controls, stream identities, counts, finish, usage, and drain. For W1–W5 at TP1, require measured IDs to match each engine's warmup IDs. TP2 output content is diagnostic. [Matched K4 timing protocol](../benchmarks/paired/README.md#matched-k4-timing-protocol)

Use `benchmarks/tp2/run.sh qualify` separately for readiness and a single native W1-shaped functional request. Use `measure` for full row and rank-cache receipts. W1 includes a separate native trace arm for K4 route activity. [TP2 cohort assessment](../benchmarks/tp2/cohorts.py#L138)<!-- prefix: "def assess(d" -->

For W1–W5, measure from the earliest submission to the last valid `[DONE]` on one monotonic clock. Record each HTTP EOF separately to validate stream completion and place cache accounting after the wire closes. Preconnect lanes, release one barrier, and send all bodies within 50 ms and before any completion. Retain submit times, headers, token events, completion times, counts, and failures. Exclude [SSE keepalives](http-server.md#completion-streams) and empty events from token counts. W6 uses the [first POST through final EOF](#w6-serial-history). [Cohort score](../benchmarks/paired/makespan.py#L21)<!-- prefix: "def cohort_m" --> · [Cache receipt interval](../benchmarks/paired/cold_cache.py#L258)<!-- prefix: "def replay(" -->

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

The historical `drain-suffix` receipt requires acknowledgement of every planned prompt checkpoint and distinct terminal store. It checks zero resident checkpoints, declines, store failures, restores, read bytes, and I/O errors. Short GQA chunks can overlap at different endpoints, so validate writes against the physical transfer bound. At TP2, both ranks acknowledge every store. GQA and GDN write bytes must be positive and equal across ranks. Rank 0 alone writes draft bytes. [Drain verification](https://github.com/para1lel/sglang.cpp/blob/3b138b699bdf47b02a7dde8ef5e07142ee699d3f/benchmarks/paired/cold_cache.py#L153)<!-- prefix: "def verify_l" --> · [Rank receipts](https://github.com/para1lel/sglang.cpp/blob/3b138b699bdf47b02a7dde8ef5e07142ee699d3f/benchmarks/tp2/cache_activity.py#L16)<!-- prefix: "def replay(r" -->

Report absolute metrics and successful and attempted requests for each engine, row, and mode. Give the latency sample count and percentile method.

For new offline timing, use `--warmup 1 --repeats 1` and omit `--top-logits`. `load_ms` covers model initialization. `request_ms` includes request allocation and graph capture. Allocation metrics include resident storage on the serial graph path. Keep JSON, invocation, binary hash, and device UUID. [Generator measurement](../apps/sglang-generate.cpp#L256)<!-- prefix: "Measurements" -->

### GPU pool engineering checks

Controlled TP1 and TP2 tests used the [submitted release](#submitted-hbm-pool-release). Each B200 reported a 168,524,838,338-byte budget and a 168,522,940,416-byte committed arena. The tests set the GQA pool to 16 pages and the GDN pool to four checkpoints to exercise pressure.

| Width | Completed backups | Backed evictions | Unbacked evictions | CUDA IPC map and unmap events | Module-static bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| TP1 | 64 | 62 | 0 | 132 | 12 |
| TP2 | 64 | 62 | 0 | 224 | 41 |

The cold, HBM-only, mixed-tier, and CPU-reload checks all returned greedy ID 17. At TP2, rank 0 completed while the fixture held rank 1's STORE reply. The backup stayed pending until the fixture released the reply.

Nsight Systems captured zero physical device allocation or free calls and zero corresponding events from readiness to shutdown start in the server and daemon. IPC maps committed backing. CUDA loads the module-static symbols automatically. The table reports fixture counts outside the timed campaign.

`artifacts/pool-w6/verification/official-verification.sha256` identifies each `routes/run.json`, `api-summary.json`, and exported trace. The release proof directories are `tp1-official-0788856-proof/` and `tp2-official-0788856-proof-b/`.

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

### Submitted HBM pool release

Native source `0788856a586a6f484135bc627270ddc509a29687` produced server SHA-256 `6874dc330b996616128a897d7f26896b2891fc3485652879f41d585a61585a78`. Native used LMCache. Pinned vLLM used local prefix caching with KV offload disabled. The arms in each pair ran on the same B200 devices.

The first attempt measured all six TP1 rows. W1–W5 passed. W6 reached 1.210904283× and missed the 1.10× limit. This build has no TP2 timings. Each arm ran one warmup and one timed cohort or chain per row. The ratio divides native makespan by vLLM makespan.

| Width | Row | Completed / attempted per arm | Native (ns) | vLLM (ns) | Ratio | Gate |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| TP1 | W1 | 1 / 1 | 572,171,374 | 641,618,759 | 0.891762× | Pass |
| TP1 | W2 | 1 / 1 | 1,383,434,795 | 1,952,919,939 | 0.708393× | Pass |
| TP1 | W3 | 4 / 4 | 4,305,343,466 | 4,911,358,733 | 0.876609× | Pass |
| TP1 | W4 | 8 / 8 | 27,103,415,230 | 27,924,394,892 | 0.970600× | Pass |
| TP1 | W5 | 4 / 4 | 38,855,159,394 | 36,724,431,609 | 1.058019× | Pass |
| TP1 | W6 | 20 / 20 | 88,663,772,515 | 73,221,123,885 | 1.210904× | Miss |

W1–W5 span the earliest submission through the last valid `[DONE]`. W6 spans the first POST through the twentieth HTTP EOF. Each comparison has one measured makespan per arm. The W1–W5 `median_ns` fields hold that single sample. Warmups and occupancy sampling are outside the makespan calculation.

Each TP1 W6 arm served 2,129,920 measured input tokens and emitted 20,480 output tokens. Native reported 1,945,581 cached prompt tokens. vLLM reported 1,925,248. The occupancy audit observed no foreign compute owner. The comparison SHA-256 is `658f2635617fdeb382dfeb315caf43437dd873034567e0bb45ab68abaae88c8f`.

The campaign root is `artifacts/pool-w6/campaign/official-0788856a586a-6874dc330b99/`. Retained `comparison.json` and arm receipts give exact timings, streams, cache counters, and launch controls.

### Warm GQA diagnostics

A separate TP1 replay of turn 20 matched all 1,024 output IDs from the timed native run. Its two prefill grants spanned 3.168 s. The trace attributed 1.074 s to 16 tiled singleton calls and 1.724 s to 144 raw paired calls. Together those readers accounted for 2.798 s of prefill. The replay receipt is `artifacts/pool-w6/verification/w6-prefill-nsys.H85aZmFJ/turn20-coverage.txt`.

A focused TP1 replay with the chunked singleton reader preserved the cache hits and first eight output IDs on turns 6 and 20. Turn 20 time to first token fell from 3,271.929 to 2,394.973 ms. Its first 2,049-row grant fell from 1,560.879 to 689.398 ms. The second stayed near 1,604 ms. The short replay receipts are under `artifacts/pool-w6/hotpool/control.FmgaURB3/` and `artifacts/pool-w6/hotpool/singleton-generation.2prfpSKU/`.

The resident staged reader passed 24 eager TP1 and TP2 rank-local cases against an independent FP64 reference. Across 198,180,864 outputs, maximum absolute error was 0.00866110 and maximum normalized RMSE was 0.00230620. For 1,024 query rows with 195,583 visible tokens, the three-trial median fell from 12.324 to 5.656 ms at TP1. At TP2, the rank-local median fell from 6.191 to 2.750 ms. These isolated readings compare raw paired attention with the staged reader. The `exact-final-resident-parity-tp{1,2}.log` files are under `artifacts/pool-w6/tmem/w6-profile.kecUkXXS/`.

### Historical PR #3 campaign

The historical PR #3 campaign covered W1–W5. Native release `739413015202541bdb4bdabcc1580aceb381e86b` passed the 1.10× gate independently on all five rows at both widths. Each native arm used one LMCache daemon with a 64 GiB CPU L1 at TP1 and 128 GiB at TP2. [PR #3 campaign evidence](https://github.com/para1lel/sglang.cpp/pull/3)

| Row | TP1 native (ns) | TP1 vLLM (ns) | TP1 ratio | TP2 native (ns) | TP2 vLLM (ns) | TP2 ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| W1 | 565,244,798 | 644,300,564 | 0.87730× | 495,706,506 | 563,151,345 | 0.88024× |
| W2 | 1,288,408,231 | 1,779,069,188 | 0.72420× | 1,235,557,398 | 2,017,694,876 | 0.61236× |
| W3 | 3,950,017,357 | 4,442,892,822 | 0.88906× | 3,877,101,752 | 3,868,120,684 | 1.00232× |
| W4 | 26,478,343,436 | 27,963,477,695 | 0.94689× | 27,371,598,131 | 25,600,380,451 | 1.06919× |
| W5 | 38,757,275,298 | 36,979,270,671 | 1.04808× | 41,606,364,017 | 40,096,070,795 | 1.03767× |

The scored TP1 W4 pair excludes an earlier pair with external GPU PID 833580 during measured vLLM requests. The independent replay receipt `consolidated.json` has SHA-256 `53fd6f8b86df28787269dbf0a48d30f2183122e2922ecf70fb7aee72cdf0b181`.

<a id="traced-one-output-prefill-diagnostic"></a>
<a id="current-single-round-v5"></a>
<a id="archived-single-round-v5"></a>
<a id="archived-multi-cohort-v4"></a>

[Archived TP1 `c701d294` results and SSE replay](https://github.com/para1lel/sglang.cpp/blob/a5974ba4eb99bd8e8f890c2e7262a27dbc3aace1/docs/performance.md#matched-k4-evidence) · [Archived v4, v5, and one-output diagnostics](https://github.com/para1lel/sglang.cpp/blob/5ee79416d3e30cd178abab69598d047c4393793d/docs/performance.md#matched-k4-evidence) · [TP2 campaign record](serving/tp2.md#campaign-record)
