# Performance and diagnostic checks

## AgentX performance standard

AgentX 256k is the sole formal performance benchmark. Compare native with LMCache
against pinned vLLM with Simple KV Offload at TP1 and TP2 on matching B200 devices.
Use the same Qwen3.8-27B-FP8 checkpoint, tokenizer, and pinned replay inputs.
[Frozen inputs](agentx-256k.md#frozen-inputs) · [Run matrix](agentx-256k.md#run-matrix)

Build the native Release server and run vLLM in its separate environment. Record
the exact revision, effective configuration, and executable or wheel hash for
each arm. Reserve the GPU UUIDs and CPU sets. Complete the affected functional
checks before measuring. [Native build](../README.md#build-on-the-project-host) ·
[Development verification](development.md#development-workflow)

Use the pinned AgentX client for the 1,800-second closed-loop replay at eight live
agent clients per GPU. Follow the documented setup and warmup, dispatch, and drain
boundaries. Report P90 normalized interactivity, input and output tokens per
second per chip, actual and theoretical KV hit rates, and request coverage.
[Execution](agentx-256k.md#execution) · [Scoring and evidence](agentx-256k.md#scoring-and-evidence)

The measured results identify the source and settings of each scored run.
[Timed AgentX results](agentx-256k.md#timed-agentx-results)

## GQA cache comparison

Native [MXFP8 GQA](development.md#p2-gqa-cache-contract) uses E4M3 payloads with
E8M0 scales per 32 channels. The pinned vLLM comparison uses `fp8_e4m3` and plain
E4M3 GQA dispatch. It defaults the per-tensor K and V scales to `1.0` when the
checkpoint supplies neither. This checkpoint has no attention scales.
[vLLM cache setting](https://github.com/vllm-project/vllm/blob/81a8cb063e6247fe254d7caf7e98a6691d71022a/vllm/config/cache.py#L251)<!-- prefix: "CUDA 11.8+ s" -->

Record the actual KV format and scale policy, Q dtype and scaling, attention
backend and fallback, and GDN state dtypes. Match GDN dtypes for controlled
comparisons and record FP8 weight scales separately. Compare logits and output
IDs on shared histories for diagnosis. [vLLM scale initialization](https://github.com/vllm-project/vllm/blob/81a8cb063e6247fe254d7caf7e98a6691d71022a/vllm/model_executor/layers/quantization/kv_cache.py#L118)<!-- prefix: "elif layer.k" -->

## Streaming diagnostics

Use one monotonic clock. Capture request submission, headers, output frames,
terminal usage, `[DONE]`, and EOF. Read token counts from usage or diagnostic IDs.
Exclude keepalive comments and empty events from token counts. Record stream
chunking and the first visible text separately. [Text verification](text-serving.md#verification) · [Completion streams](http-server.md#completion-streams)

| Metric | Definition |
| --- | --- |
| TTFT | Time from request submission to the first output token, including queueing and prefill. Unavailable for zero output tokens. |
| TPOT | Time from first to last output token divided by the number of output tokens minus one. Unavailable for fewer than two tokens. |
| End-to-end latency | Time from request submission to its terminal response. |

## Native diagnostics

### GPU pool engineering checks

Use the [prefix cache fixtures](prefix-cache.md#verification) at TP1 and TP2. Check cold prefill, HBM-only reuse, mixed-tier restore, and CPU reload on the same token history. Hold one TP2 STORE reply to verify that the backup stays pending until both ranks acknowledge it.

Check physical device allocations and frees in the server and daemon from readiness to shutdown start. Count IPC maps and module-static symbols separately. Retain the route receipts and exported trace.

### Operator qualification

Operators may reorder arithmetic within their numerical contract. Compare outputs and state with an independent mathematical reference under the operation's established error criteria. Check causality where the operation uses token history, and model quality on frozen inputs and shared token histories. Record free-running divergence separately. Require bitwise equality for copied bytes, exact replay, and comparisons that preserve arithmetic.

Use repeated matched measurements to choose the fastest validated implementation per supported case. Keep one production implementation per case. Use validated fallbacks where shape, device, or resources prevent the primary route. Freeze diagnostic inputs and numerical expectations before the comparison.

### Fused FP8 decode

Match BF16 logits and IDs with native controls. Compare eager, fresh graphs, and reused graphs at matching visible lengths and KV capacities. Cover exact and direct GQA bounds, groups of two through five, mixed steps, [packed K0 buckets](batch-graphs.md#packed-paged-gqa-decode), and shape and alignment fallbacks.

Compare fused SwiGLU BF16 results with identical and disjoint gate and up output spans. Require `cudaErrorInvalidValue` before launch for identical gate and up spans in fallback, partial output overlaps, and output overlaps with inputs.

### Exact decode bandwidth

Match BF16 projections, E4M3 payloads, and FP32 scales bitwise against the native control. Include nonfinite inputs, shape and alignment fallbacks, and fresh and reused graphs. Compare full logits and greedy IDs on shared histories. Run the [MTP checks](mtp.md#verification).

### Chunked prefill

Run the [prefill checks](development.md#chunked-prefill-verification). Match exact-policy BF16 logits, KV, and GDN state with row-wise control. Measure device prefill through the operation before the first LM head and measure HTTP TTFT separately. Record tiles, tails, active lanes, and decode gaps.

Record peak HBM and [logical workspace and request bytes](native-prefill-workspace.md#constraints-and-failure-behavior). Record pool used and reserved bytes after synchronization. Require zero live scratch after retirement, including recovered free-enqueue failures. Use a fresh process for cold allocation timing.

### FP8 tensor-core prefill

Use `--mtp-steps 0` for the native control. Compare reordered arithmetic with an independent reference and require bitwise parity for exact projections and arithmetic-preserving fusions. Cover chunks, tails, mixed steps, unsupported dispatch, raw MXFP8 exponents 0 and 255, page crossings, and eager and graph decode. For changed fusions, verify staged values and committed state, including positive MTP depth. [Projection checks](fp8-prefill-projections.md#verification) · [GQA checks](flash-prefill.md#verification)

For cross-policy diagnostics, compare logits on identical token histories and prefill boundaries. After greedy divergence, use shared teacher-forced tokens. Report logit error, ranks, winner margins, labeled-token NLL, and first divergence. Run [serial tile checks](text-serving.md#serial-prefill-verification) before timing.

### Native concurrent workloads

Verify offered and observed concurrency, streamed output, decode width, and graph replay under the fixed scheduler. Use [streaming checks](development.md#streaming-verification) and [wide decode checks](long-paged-decode.md#verification).

### Native MTP

Run [MTP verification](mtp.md#verification). Compare with `--mtp-steps 0` at the same width and measure K0 graph replay separately. Match prompt IDs, arrivals, budgets, stops, and precision. Disable tracing and `--top-logits` for timing. Alternate arms and report samples, medians, ranges, and paired ratios.

For layer-major verification, keep depth fixed against a row-wise control. For tiled draft seeding, keep verifier and depth fixed. Time target prefill and draft seeding through workspace release before the first LM head. Include short, boundary, and long prompts. [Graph limit](mtp.md#target-verification-graph)

In separate `--trace-batches` runs, require `accepted ≤ compared ≤ drafted` and `drafted ≤ rounds × K`. A first mismatch counts as compared. Zero or one output requires zero drafts. With normal completion, consumed decode rows equal committed outputs minus the initial token. Computed or replayed rows can exceed consumed rows. Report accepted/drafted and accepted/rounds for positive denominators. [Batch trace](../apps/server/detail/batch_trace.cpp#L11)<!-- prefix: "void trace_v" -->

## Correctness, memory, and diagnosis

Run [native correctness checks](development.md#verification) separately from timed runs when instrumentation affects latency. Record HBM after load and peak, capacity and headroom, initial cache state, evictions, replay, and reclamation. [Sampling](text-serving.md#verification) · [Constraints](text-serving.md#finite-choice-verification) · [Multimodal](multimodal-decoder.md#verification)

Collect kernel, scheduler, queue, and utilization diagnostics separately. Record the collection method and link each change to its measured bottleneck. [Native scheduler samples](prefix-cache.md#native-http-controls)

Retain the run manifest, commands, environment, fixture hashes, raw samples, and failures in ignored `artifacts/` storage. A missing counter is unavailable. A missing baseline, failed correctness check, or changed workload leaves that performance claim open.

The campaign directory `artifacts/pool-w6/campaign/official-b0596c25eab5-59748dd6424d/` contains the saved streams, clock receipts, and placements. The independent replay record is `artifacts/pool-w6/verification/audit-formal-digest-b0596c25eab5.json`.