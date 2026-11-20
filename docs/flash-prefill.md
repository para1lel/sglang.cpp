# Native MXFP8 flash prefill

## Responsibilities and sources of truth

| Part | Responsibility | Source |
| --- | --- | --- |
| Decoder | Select the target prefill policy and append KV before attention. | [Policy](../src/decoder_prefill.cpp#L451)<!-- prefix: "const bool t" --> · [Attention call](../src/decoder_prefill.cpp#L267)<!-- prefix: "const auto s" --> |
| Native CUDA operator | Select flash, ordered multirow, or protected single-row attention for tensor-core multirow prefill. | [Dispatch](../src/mxfp8_gqa_prefill.cu#L817)<!-- prefix: "cudaError_t " --> |
| Serving | Select actual tile rows before dispatch. | [Serial prefill](text-serving.md#serial-prefill) · [Batch planner](batch-graphs.md#runtime-path) |

## Runtime path

The CLI and C++ default policy is `exact`. Its original MXFP8 attention path retains the exact prefill contract. With `tensor-core`, the decoder selects native flash by default for eligible multirow tiles on the SM100 build.

The flash kernel follows the logical page table. A block handles 64 query rows and two query heads that share a KV head. It processes 64-key tiles and masks each query at its own position. QK and PV use BF16 Tensor Core products with FP32 accumulators. Online softmax keeps FP32 normalization and rounds the probabilities used by PV to BF16. [Flash kernel](../src/mxfp8_gqa_prefill.cu#L732)<!-- prefix: "__global__ v" -->

## Constraints and failure behavior

Flash requires exactly 1,024 query rows and at most 131,072 visible tokens at the final row, including that row. Both K and V payload bases must be 16-byte aligned. The device's opt-in shared memory limit must cover the 154,128 dynamic bytes plus the kernel's static allocation. [Eligibility](../src/mxfp8_gqa_prefill.cu#L827)<!-- prefix: "if (rows == " --> · [Capacity check](../src/mxfp8_gqa_prefill.cu#L775)<!-- prefix: "std::optiona" -->

Other valid shapes, unaligned payloads, and a known shared memory shortfall use the [ordered fallback dispatch](development.md#ordered-multirow-gqa-prefill). Above 32,768 visible tokens, that dispatch uses protected single-row attention. CUDA discovery, configuration, and launch errors propagate. If a query or visible KV value falls outside flash's numerical range, protected attention repairs the affected row and query-head pair. This repair retains selective FP64 recovery.

## Verification

For the explicit tensor-core policy, prioritize measured efficiency over bitwise parity with ordered attention. Flash can change BF16 results, logits, and generated IDs. Compare flash with an independent FP64 reference over the same represented inputs. Check causality, page remaps, output guards, repaired rows, and eager versus CUDA Graph replay. Trace dispatch at the eligibility bounds and through alignment and capacity fallback.

Run [`mxfp8-gqa-check`](development.md#p2-gqa-verification) on a reserved B200. The [flash regression](../tests/mxfp8_gqa_flash_check.cu#L485)<!-- prefix: "void check_f" --> checks capture and replay, sampled FP64 outputs, exact dispatch, and unaligned K and V separately.

The [long-context cases](../tests/mxfp8_gqa_flash_check.cu#L445)<!-- prefix: "void check_l" --> require full-tile flash at 32,768, 32,769, 65,536, and 131,072 visible tokens. They require protected attention at 131,073 and for unaligned full tiles at 32,769. The 1,023-row cases check ordered fallback at 32,768 and protected fallback at 32,769. The upper-bound cases also check the exact route, causality, and score overflow recovery.

Check ordered and protected fallback bitwise against the ordered FP32 reference for finite scores and quotients. Check FP64 recovery independently. Use the [exact prefill runbook](../tests/prefill/README.md) for the separate `exact` contract.

### Earlier 32,768-token measurements

On a reserved B200, default flash sped up nine eligible operator shapes by 8.92–12.77× by paired median kernel time. The operator probe completed 24 graph replays and retained the previous routes at 1,023 and 1,025 rows and 32,769 visible tokens. The earlier arithmetic-identical kernel had 0.158–0.225% normalized RMSE on sampled FP64 comparisons across those eligible shapes.

The native HTTP comparison used a 32-slot batch, a 32,768-token step budget, K0, decode graphs, and prefix reuse off. W1 and W2 used nine measured requests per arm. W3 used five measured four-lane cohorts per arm.

| Workload | Median TTFT, ordered → flash-enabled (ms) | Median output IDs/s, ordered → flash-enabled | Output comparison |
| --- | ---: | ---: | --- |
| W1 | 19.606 → 19.912 | 128.753 → 128.182 | Both used ordered attention. IDs matched. |
| W2 | 272.398 → 176.219 | 119.070 → 121.847 | The first different ID was 314 → 440 at zero-based output index 13. |
| W3, concurrency 4 | 9,284.164 → 3,200.740 | 110.374 → 132.611 | All 28,672 IDs matched, including warmups. |

W1 p95 TTFT rose from 19.793 to 64.086 ms in these runs. On the common W2 teacher-forced history, 62 of 64 top IDs matched and mean NLL increased by 0.00460 nats per token. These measurements compare two attention routes in the same FP8 checkpoint. Both W3 arms missed the frozen external-reference gates.

At W4 concurrency 16, both arms stopped at the first warmup with zero completed streams. Ordered attention emitted zero IDs across all 16 lanes before their 90-second socket timeouts. With flash enabled, five lanes timed out with zero IDs at 90 seconds. The other 11 received HTTP 200 and emitted 39–55 IDs each before the 360-second cohort deadline. W4 produced no scored results. Overall vLLM acceptance remains open.

Timing and dispatch records are in `artifacts/takeover-flash-prefill-default/handoff.md`. Independent numerical review is in `artifacts/takeover-flash-prefill-review/report.md`.

### W6 warmups and diagnostics

The [frozen W6 protocol](../benchmarks/concurrent/README.md#conditions-and-gates) uses four concurrent 131,072-token prompts and 1,024 output IDs per request. The matched GPU6 campaign used 32,768 scheduled tokens per step, 32 active slots, disjoint prefixes, and MTP off. Both native builds used landed GDN. The candidate extended full-tile Flash visibility to 131,072 tokens.

Both native arms stopped at the first warmup. All four streams in each arm reached the 90-second socket timeout before the first output ID. Neither arm produced a scored cohort or a native output comparison.

Pinned vLLM completed two warmups and five measured cohorts. All 28 streams matched the archived output IDs and repeated exactly across cohorts. The maintained gates remain the frozen W6 thresholds.

| Metric | Pinned vLLM | Frozen W6 gate | Result |
| --- | ---: | ---: | --- |
| Median output IDs/s | 45.467107 | > 48.709172 | Fail |
| p95 TTFT (ms) | 75,540.784412 | ≤ 113,118.073927 | Pass |
| p95 TPOT (ms) | 69.018133 | ≤ 104.925433 | Pass |
| p95 E2E (ms) | 90,104.907074 | ≤ 85,806.011537 | Fail |

The same builds ran separate one-lane, two-ID diagnostics with the 32,768-token step cap. They used a 1,800-second socket timeout and a 1,810-second cohort deadline. At 40,960 prompt tokens, control and candidate both returned `16,25`. TTFT fell from 37.386 s to 6.115 s. At 131,072 prompt tokens, the candidate returned `760,45366` with 45.798 s TTFT. The 131,072-token diagnostic tested the candidate only.

See `artifacts/takeover-w6-flash-selected/report.md` for the integrated source identities and run record.

The earlier GPU6 campaign used an 8,192-token step cap. Both native arms stopped at the first warmup with zero IDs on all four lanes. A separate profile spent 74.021 s of its 95.010 s attention interval in protected prefill. See `artifacts/takeover-w6/report.md`.

A separate prototype diagnostic used single requests with two output IDs, the historical 8,192-token step cap, a 1,800-second socket timeout, and a 1,810-second cohort deadline. Both arms completed and agreed on both IDs at 40,960 and 131,072 prompt tokens. The longer Flash route reduced TTFT on those requests. See `artifacts/takeover-w6-flash-code/report.md`.

Report the actual prompt length, concurrency, output budget, and deadlines for short diagnostics. Score W6 from measured cohorts that complete the frozen workload under the [current launch policy](performance.md#scheduler-and-reporting-policy).
