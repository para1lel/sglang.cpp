# Single-B200 benchmark agreement

The project goal is higher inference throughput than pinned vLLM on the **same physical B200 and the same immutable checkpoint**. A workload earns a speedup only when the candidate passes its correctness checks, its measured candidate/vLLM output-token throughput ratio exceeds 1.0, and it meets the agreed latency ceilings. Measure vLLM first. Freeze the workloads, repetition policy, noise margin, and latency ceilings before timing the candidate.

## Implementations and controls

| Part | Responsibility | Source |
| --- | --- | --- |
| Independent baseline | Run vLLM in its own environment. Record the checkout, installed wheel and native binary identities, dependency versions, and effective launch settings. | `vllm-project/vllm` at `81a8cb063e6247fe254d7caf7e98a6691d71022a`. |
| Candidate | Run the project binary built by XMake with the default Pixi toolchain. Record source, build options, and executable identity. | [Development contract](development.md). |
| Architectural reference | Interpret service and cache behavior. Any separate SGLang timing needs its own result label. | `sgl-project/sglang` at `f04d886bf79aa3d6237cbb12a3311838eb9f9bd8`. |

Run each engine alone on the selected UUID. Record driver, CUDA runtimes, host and client configuration, clocks, power settings, and other GPU activity. Keep isolation between the baseline environment and the candidate build. Disclose required software differences and their effect on the comparison.

Match model revision, weight shards and scales, tokenizer, chat template, processor, effective weight and KV precision, and sampling policy. Freeze request bytes, token IDs, arrival order, concurrency, seeds, output limits, and stop rules. Match prefix-cache state, MTP mode, and media preprocessing. Check actual dispatch and token counts. If either engine lacks a required mode, mark that row unsupported and leave the comparison open.

## First baseline run

From the repository root, prepare the isolated vLLM environment. Set `GPU_INDEX` to the reserved B200's physical index. Run the supplied token-ID fixture against `/data0/shared/Qwen3.8-27B-FP8` and record the resolved UUID.

```bash
bash benchmarks/vllm/setup.sh
GPU="${GPU_INDEX:?set the reserved B200 index}" bash benchmarks/vllm/run.sh benchmarks/vllm/small.json
```

The offline smoke case requests eight output tokens. Inspect the reported run directory for the device UUID, effective configuration, output IDs, and individual request timings. Repeats must finish at the requested length with identical greedy token IDs. This case measures whole-request latency and output tokens/s. W1–W6 require separate runs with a streaming client to measure first-token latency.

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

Report one row per engine, workload, and mode. Include absolute metrics, successful and attempted requests, and the agreed limits. For paired runs, also report the candidate/baseline throughput ratio and latency differences. At adoption, the required improvement above parity and the latency budgets still need numerical margins from repeated baseline measurements.

## Correctness, memory, and diagnosis

Performance acceptance requires repeatability, correctness, and comparable memory and feature settings. The following checks are proposed for the applicable milestones.

- **Correctness.** Freeze high-precision operator references, logits tolerances, greedy token fixtures, and sampling distribution tests in advance. Validate independent outputs before timing. Check EOS, stop strings, constraints, cancellation, and stream termination. Use a separate correctness run when instrumentation changes timing.
- **Memory.** Record HBM after load and peak HBM during prefill and decode. Budget weights and scales, GQA KV pages, GDN live state and checkpoints, activations, workspaces, graph pools, vision weights and buffers, and runtime allocations. State the measurement method, capacity, and headroom for each input length and concurrency.
- **Cache.** Record reusable prefix tokens divided by eligible prompt tokens, the initial cache state, page occupancy, evictions, and recurrent checkpoint restores and replay lengths. Compare cold and primed continuations with fresh execution. Check reclamation after cancellation and memory pressure.
- **MTP.** Record drafted and accepted tokens, verification rounds, acceptance rate (accepted/drafted), and accepted length per round. Check actual draft and target execution. Compare greedy continuations and sampled distributions with MTP off. Exercise rejection across a recurrent checkpoint boundary.
- **Multimodal.** Verify image and video processing, visual token positions, and reference output behavior on each fixture. Include client-visible preprocessing and transport in end-to-end measurements. Record server processing and vision-encoder time separately when observable.
- **Profiling.** Record GPU utilization, kernel time, JIT time, scheduler time, and queue delay with their collection methods. Profile a separate run if tracing changes throughput. Tie each optimization to its bottleneck, correctness result, and end-to-end result on the same workload.

Retain the run manifest, commands, environment, fixture hashes, raw samples, and failures in ignored `artifacts/` storage. A missing counter is unavailable. A missing baseline, failed correctness check, or changed workload leaves that performance claim open.
