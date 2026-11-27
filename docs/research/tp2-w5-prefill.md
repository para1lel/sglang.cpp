# Why long, partial prefill tiles cost time in TP2 W5

Eight long, partial prefill tiles spent 12.890763 seconds on rank 0 and 12.897292 seconds on rank 1 in protected grouped-query attention (GQA) in an instrumented eight-request run with 160 outputs per request. Both ranks processed the same requests concurrently. The official 1,024-output W5 cohort recorded client clocks only.

## Campaign baseline

Native used ranked LMCache and MTP4. Pinned vLLM disabled KV offload. Both used B200s 4 and 5 and one cold measured cohort after warmup. Makespan runs from earliest submission to last streamed `[DONE]`, with an independent 1.10× limit per row. [Campaign contract](../serving/tp2.md#acceptance-target)

| Row | Native seconds | vLLM seconds | Native / vLLM |
| --- | ---: | ---: | ---: |
| W1 | 0.592804 | 0.558432 | 1.061551 |
| W2 | 1.699082 | 2.000014 | 0.849535 |
| W3 | 6.862179 | 4.113793 | 1.668091 |
| W4 | 57.338228 | 26.346666 | 2.176299 |
| W5 | 66.235146 | 39.459475 | 1.678561 |

Native `337edce79e46222c90ad700c468b0c5468729463`, executable SHA-256 `0100f4e2065977bd57c3c75f76cdebd01bb0102776ac33c3fca6d9f75bf59bb7`, ran against vLLM `81a8cb063e6247fe254d7caf7e98a6691d71022a`. The report cites identical operator and scheduler code at `26380a7ffef839f47037b8f80c619367f513167e`. W1–W2 ran November 25, 2026 UTC and W3–W5 November 26. W2 uses the complete measured reference stream through `[DONE]` because its collector stopped before retirement.

## What changed when requests kept generating

Each request had a distinct 131,072-token prompt. Clocks are seconds from earliest submission.

| Engine and observation | Requests × outputs | Device pair | First output | Latest first output | Last `[DONE]` |
| --- | ---: | --- | ---: | ---: | ---: |
| Native, events disabled | 2 × 4 | 0/3 | 10.283 | 10.698 | 10.776 |
| vLLM, clean | 2 × 4 | 4/5 | 5.181 | 8.963 | 8.986 |
| Native, events disabled | 8 × 4 | 0/3 | 19.992 | 43.295 | 43.414 |
| Native, events enabled | 8 × 4 | 0/3 | 20.569 | 45.481 | 45.612 |
| Native, mixed prefill and verification trace | 8 × 160 | 4/5 | 22.258 | 59.410 | 60.449 |
| Native, official | 8 × 1,024 | 4/5 | 19.944 | 60.010 | 66.235 |
| vLLM, official | 8 × 1,024 | 4/5 | 5.149 | 36.172 | 39.459 |

Both pairs use NV18. Native four-output controls loaded observer-capable executable `97b4662de6e9867ad515bf9522ab7f844bd397875924e419c0415d78528ea570` with batch tracing and toggled CUDA events plus host phase and cache timing. The mixed run used batch tracing and release-linked observer `817f9e94ebdf20f447bead97d7e8914473c23de8401c2bcb4b66fa5e8904bf6a`. Capture ran on November 26 from 01:27:40.957731 to 01:28:41.416456 UTC and bracketed the client interval. Official cohorts used the plain release executable.

## Which operator ran

The scheduler reserved verification rows before spending the remaining 32,768-row budget in admission order, stopping at 8,192-token checkpoints. The trace captured 17 prompt-only advances and 24 mixed advances, processing 1,048,576 prompt and 575 verification rows. Of 136 grants, 120 had 8,192 rows, one had 8,187, and seven had 8,172. The same late request then needed one five-row and seven 20-row grants. Each 8,172-row grant left 1,004 rows after seven full 1,024-row tiles. [Row reservation](https://github.com/para1lel/sglang.cpp/blob/26380a7ffef839f47037b8f80c619367f513167e/include/sglang/text/decoder_batch_plan.h#L31)<!-- prefix: "const int re" -->

The frozen fast route supported 131,072 visible tokens for full tiles and 32,768 for partial tiles. The observed long tails used `protected_prefill_attention`. [Visibility limits](https://github.com/para1lel/sglang.cpp/blob/26380a7ffef839f47037b8f80c619367f513167e/src/mxfp8_gqa_prefill.cu#L10)<!-- prefix: "constexpr in" --> · [Protected launch](https://github.com/para1lel/sglang.cpp/blob/26380a7ffef839f47037b8f80c619367f513167e/include/sglang/text/gqa/multirow_prefill.cuh#L79)<!-- prefix: "return cudaL" -->

| Protected tile rows | Starting token positions | Calls per rank | Rank 0 seconds | Rank 1 seconds |
| ---: | --- | ---: | ---: | ---: |
| 1,004 | 39,936 / 48,128 / 56,320 / 64,512 | 64 | 11.868098 | 11.876424 |
| 20 | 40,940 / 49,132 / 57,324 / 65,516 | 64 | 1.022665 | 1.020869 |
| **Total** | **Eight tile positions** | **128** | **12.890763** | **12.897292** |

Each of the eight groups in advances 26–33 ran once per GQA layer. Events enclosed every irregular dispatch and one-in-eight kernel samples identified the protected path. Both ranks acknowledged 136 stores. Before the latest first output, cache RPC intervals covered 9.294 seconds in union, including 0.417 seconds outside decoder advances. Store waits cannot be isolated from those intervals. The 13.929-second latest-first difference between the traced eight-lane controls also spans different device pairs and instrumentation.

## Subsequent coverage

The native staged long route supports 992–1,024 rows through 131,072 visible tokens. At a captured 1,004-row W5 tile it matched all 3,084,288 BF16 outputs per rank and cut median device times from 6.408 to 1.034 ms and 6.405 to 1.029 ms. A two-B200 replay checked the tile inside a mixed 4,096-row projection group and after twelve forced clean pool OOMs. The 20-row tails now qualify for ordered staging with aligned cache buffers. Their six rank-local head pairs and ten two-row tiles yield 60 CTAs per rank. A replay at the four recorded tail positions can measure that route separately from the large-tile gain. Full W5 makespan for these changes remains unmeasured. [Staged long-route evidence](../ordered-gqa-prefill.md#verification) · [Twenty-row dispatch](https://github.com/para1lel/sglang.cpp/blob/a5974ba4eb99bd8e8f890c2e7262a27dbc3aace1/include/sglang/text/gqa/multirow_prefill.cuh#L61)<!-- prefix: "if (rows >= " -->

## Reproducible evidence

Paths are relative to `artifacts/tp2-bench/`. The diagnostic manifest indexes request streams, rank identities, operator events and analysis inputs. The official receipt records UTC and monotonic clocks.

| Record | SHA-256 |
| --- | --- |
| `timing-only-0825/qualified-five-rows.json` | `d48b1591a01f6dd27ab6b2a6045841d33ae999f93dccaebd81bdf6f9d76189e3` |
| `w5-attribution-0855/w5-evidence.sha256` | `43de64a42f1ee16b5f5722ab42508f918de3fe6083b4e4759bd2dae8be60e55e` |
| `w5-attribution-0855/w5-budget.json` | `bdbfc02ba97f39dc5d4d2df2fa039aed4e98bccf6ed48b6133e588196d33127d` |
| `w5-attribution-0855/mixed-summary.json` | `228e93deaa43bb69925aed232780ed5f978265260465e10f957739be7361cee0` |

Under `w5-attribution-0855/mixed/W5/native-k4-trace/`, `grants.tsv` records row positions and counts, `ops.tsv` device events, `phases.tsv` dispatch boundaries, `put.tsv` cache RPCs and `observed.json.gz` client clocks. The four-output controls are in `w5-attribution-0855/{native-summary,native8-summary,reference-summary}.json`.
