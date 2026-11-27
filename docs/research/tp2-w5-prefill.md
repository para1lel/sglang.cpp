# Why long, partial prefill tiles cost time in TP2 W5

Eight long, partial prefill tiles spent **12.891 seconds on rank 0 and 12.897 seconds on rank 1** in protected grouped-query attention (GQA). Both ranks processed the same requests concurrently. The measurement covers an instrumented eight-request run with 160 outputs per request. Confidence is high in the operator attribution. The full W5 campaign used 1,024 outputs and recorded client clocks without these operator events.

## Campaign baseline

W1–W5 measure earliest request submission through the last streamed `[DONE]`, with one cold measured cohort after warmup. Each row has an independent 1.10× limit. Native used ranked LMCache and MTP4. The pinned vLLM TP2 reference disabled KV offload. Both used B200 devices 4 and 5. [TP2 campaign and acceptance contract](../serving/tp2.md#campaign-record)

| Row | Native seconds | vLLM seconds | Native / vLLM |
| --- | ---: | ---: | ---: |
| W1 | 0.592804 | 0.558432 | 1.061551 |
| W2 | 1.699082 | 2.000014 | 0.849535 |
| W3 | 6.862179 | 4.113793 | 1.668091 |
| W4 | 57.338228 | 26.346666 | 2.176299 |
| W5 | 66.235146 | 39.459475 | 1.678561 |

The clocks belong to native release `337edce79e46222c90ad700c468b0c5468729463`, executable SHA-256 `0100f4e2065977bd57c3c75f76cdebd01bb0102776ac33c3fca6d9f75bf59bb7`, and vLLM `81a8cb063e6247fe254d7caf7e98a6691d71022a`. This report reads the same native operator and scheduler code at `26380a7ffef839f47037b8f80c619367f513167e`. Only benchmark and documentation files differ between those revisions. W1–W2 ran on November 25, 2026 UTC. W3–W5 ran on November 26 UTC. W2 uses the complete measured reference stream through `[DONE]` because its collector stopped before retirement.

## What changed when requests kept generating

Each request had a different 131,072-token W5 prompt. All times below are seconds from the earliest submission. The first two timing columns end at the earliest and latest first output across requests.

| Engine and observation | Requests × outputs | Device pair | First output | Latest first output | Last `[DONE]` |
| --- | ---: | --- | ---: | ---: | ---: |
| Native, events disabled | 2 × 4 | 0/3 | 10.283 | 10.698 | 10.776 |
| vLLM, clean | 2 × 4 | 4/5 | 5.181 | 8.963 | 8.986 |
| Native, events disabled | 8 × 4 | 0/3 | 19.992 | 43.295 | 43.414 |
| Native, events enabled | 8 × 4 | 0/3 | 20.569 | 45.481 | 45.612 |
| Native, mixed prefill and verification trace | 8 × 160 | 4/5 | 22.258 | 59.410 | 60.449 |
| Native, official | 8 × 1,024 | 4/5 | 19.944 | 60.010 | 66.235 |
| vLLM, official | 8 × 1,024 | 4/5 | 5.149 | 36.172 | 39.459 |

Both device pairs use NV18 NVLink. The two- and eight-request native four-output controls loaded observer-capable executable `97b4662de6e9867ad515bf9522ab7f844bd397875924e419c0415d78528ea570`. Both kept `--trace-batches` active. One observer switch enabled CUDA events and host phase, cache-export and RPC timing for their second cohorts. The mixed trace also kept batch tracing active and loaded `817f9e94ebdf20f447bead97d7e8914473c23de8401c2bcb4b66fa5e8904bf6a`, linked from the release objects. The official native runs used the plain release executable without `--trace-batches`. The mixed capture ran from **01:27:40.957731 to 01:28:41.416456 UTC on November 26**. Client capture brackets include observer overhead outside the submission-to-`[DONE]` interval.

## Which operator ran

The scheduler reserves one target row plus legal proposals for each ready decoder. It spends the remaining 32,768-row budget on ready prompts in admission order, stopping at 8,192-token checkpoint boundaries. [Release row reservation](https://github.com/para1lel/sglang.cpp/blob/26380a7ffef839f47037b8f80c619367f513167e/include/sglang/text/decoder_batch_plan.h#L31)<!-- prefix: "const int re" -->

The mixed trace recorded 17 prompt-only advances and 24 advances with both prompt and verification work. These processed all 1,048,576 prompt rows, plus 575 verified rows during the mixed advances. Of 136 prompt grants, 120 contained 8,192 rows. One grant contained 8,187 rows, seven contained 8,172, and the same late request then needed one five-row and seven 20-row grants to reach its checkpoints. Within each 8,172-row grant, seven full 1,024-row tiles left a 1,004-row tile.

The dispatcher allows a full 1,024-row tile to attend through 131,072 visible tokens. Partial tiles have a 32,768-token limit on the fast route. Beyond that limit, the release selected `protected_prefill_attention` for the observed partial tiles. [Visibility limits](https://github.com/para1lel/sglang.cpp/blob/26380a7ffef839f47037b8f80c619367f513167e/src/mxfp8_gqa_prefill.cu#L10)<!-- prefix: "constexpr in" --> · [Row-dependent limit](https://github.com/para1lel/sglang.cpp/blob/26380a7ffef839f47037b8f80c619367f513167e/include/sglang/text/gqa/multirow_prefill.cuh#L19)<!-- prefix: "const int fl" --> · [Protected launch](https://github.com/para1lel/sglang.cpp/blob/26380a7ffef839f47037b8f80c619367f513167e/include/sglang/text/gqa/multirow_prefill.cuh#L79)<!-- prefix: "return cudaL" -->

| Protected tile rows | Starting token positions | Calls per rank | Rank 0 seconds | Rank 1 seconds |
| ---: | --- | ---: | ---: | ---: |
| 1,004 | 39,936 / 48,128 / 56,320 / 64,512 | 64 | 11.868098 | 11.876424 |
| 20 | 40,940 / 49,132 / 57,324 / 65,516 | 64 | 1.022665 | 1.020869 |
| **Total** | **Eight tile positions** | **128** | **12.890763** | **12.897292** |

Each protected tile ran once per GQA layer on each rank, across all 16 GQA layers. CUDA events enclosed every irregular dispatcher call. One-in-eight underlying kernel samples identified the protected kernel at each listed position. The eight groups occupied advances 26–33. Earlier partial tiles through 32,768 visible tokens used faster kernels. Only the mixed trace recorded the per-dispatch durations in the table.

Host prefill can return while kernels are still running. A later stream synchronization or cache export can therefore include preceding attention work. Both ranks acknowledged 136 cache stores. Before the latest first output, the union of cache RPC intervals was 9.294 seconds, including 0.417 seconds outside decoder advances. Overlap does not establish how much any request waited for a store. The 13.929-second difference between the two eight-lane traced latest-first times also includes device-pair and instrumentation differences.

The first operator experiment should preserve FIFO scheduling and compare a long-partial fast path with protected attention at the eight recorded positions. Check causal outputs, paged KV and numerical recovery on both ranks before measuring the same request stream. A separate scheduler experiment could align truncated FIFO grants to avoid partial tiles. It would need to measure first-output latency for each request as well as cohort makespan.

## Reproducible evidence

Paths below are relative to `artifacts/tp2-bench/`. The diagnostic manifest indexes the actual request streams, rank identities, operator events and analysis inputs. The official receipt contains each arm's UTC cross-clock sample and monotonic timestamps.

| Record | SHA-256 |
| --- | --- |
| `timing-only-0825/qualified-five-rows.json` | `d48b1591a01f6dd27ab6b2a6045841d33ae999f93dccaebd81bdf6f9d76189e3` |
| `w5-attribution-0855/w5-evidence.sha256` | `43de64a42f1ee16b5f5722ab42508f918de3fe6083b4e4759bd2dae8be60e55e` |
| `w5-attribution-0855/w5-budget.json` | `bdbfc02ba97f39dc5d4d2df2fa039aed4e98bccf6ed48b6133e588196d33127d` |
| `w5-attribution-0855/mixed-summary.json` | `228e93deaa43bb69925aed232780ed5f978265260465e10f957739be7361cee0` |

Under `w5-attribution-0855/mixed/W5/native-k4-trace/`, `grants.tsv` records request positions and row counts. `ops.tsv` records device events, `phases.tsv` identifies dispatch boundaries, and `put.tsv` records cache RPCs. `observed.json.gz` contains the original client clocks. `native-summary.json`, `native8-summary.json` and `reference-summary.json` in `w5-attribution-0855/` describe the four-output controls.
