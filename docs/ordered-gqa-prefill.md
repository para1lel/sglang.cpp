# Ordered MXFP8 GQA prefill

## Responsibilities and sources of truth

Model prefill appends uncovered KV and selects the attention path. The native dispatcher picks Flash, staged or direct ordered tiles, or protected single-row attention. Ordered attention owns causal FP32 scoring and token-ordered accumulation. [Caller](../src/runtime/execution/decoder_prefill_tiles.cpp#L346)<!-- prefix: "void Chunk::" --> · [Policy](../src/decoder_prefill.cpp#L659)<!-- prefix: "void Decoder" --> · [Dispatch](../src/text/gqa/ordered_prefill.cu#L382)<!-- prefix: "cudaError_t " --> · [Score](../src/text/gqa/ordered_prefill.cu#L34)<!-- prefix: "__device__ v" --> · [Tile loop](../src/text/gqa/ordered_prefill.cu#L188)<!-- prefix: "for (int fir" -->

## Runtime path

After KV append, Tensor Core uses the [Flash dispatcher](flash-prefill.md#runtime-path) at the same position and row count. Exact multirow tiles use direct MXFP8. Single rows use [ordinary forward and its ordered reader](singleton-gqa-decode.md#scratch-free-ordered-attention). Eligible grouped tiles can use [batched staging](flash-prefill.md#runtime-path).

Staged ordered attention double-buffers packed K/V and scales in 32-key blocks, zero-filling outside visibility. Each row keeps its causal limit. Ordered attention shares decoded K/V across two query heads and preserves the FP32 score tree, token order and BF16 rounding. Skip unit history multipliers only when all active heads have a finite prior maximum and no higher visible score. [Copies](../src/text/gqa/prefill_common.cu#L27)<!-- prefix: "__device__ _" --> · [Warp vote](../src/text/gqa/ordered_prefill.cu#L89)<!-- prefix: "if (isfinite" --> · [Block vote](../src/text/gqa/ordered_prefill.cu#L211)<!-- prefix: "const bool u" -->

### Grouped partial grants

The [partial-grant projection path](native-prefill-workspace.md#partial-grant-projection-groups) appends new KV before selecting a reader. Each lane keeps its own page table, causal mask and unwritten KV/query-tail guards. Staged and raw grouped readers require a common KV head count across lanes. [Grouping](../src/runtime/execution/decoder_prefill_staged_gqa.cpp#L49)<!-- prefix: "for (; first" --> · [Staged guard](../src/text/gqa/tmem_staged.cu#L476)<!-- prefix: "if (rows \u003c m" --> · [KV head guard](../src/text/gqa/tmem_staged.cu#L472)<!-- prefix: "if (sequence" --> · [Buffer validation](../src/text/gqa/prefill_common.cu#L375)<!-- prefix: "bool valid_p" -->

| Reader | Eligibility and sequence |
| --- | --- |
| Staged | Four-lane subgroups, each with any nonnegative start, 256–1,024 rows and ≤32,768 visible tokens. Starts and counts may differ. SM100 requires 16-byte-aligned KV payloads/scales and disjoint scratch. Submit one predecode and one [page-wide attention](flash-prefill.md#runtime-path). |
| Raw grouped | After staging declines, 148-SM SM100 may use raw ordered TMEM for a group divisible by four with equal 256- or 512-row grants and starts ≥8,192. |
| Per-lane | If neither group reader submits, each lane, including leftovers, reads retained and appended KV. |

Queued predecode precedes fallback on the same stream. Pending/unexpected errors propagate. Gate per lane or defer for [fused postgate and quantization](fp8-prefill-projections.md#attention-output-quantization). [Staged launch](../src/runtime/execution/decoder_prefill_staged_gqa.cpp#L62)<!-- prefix: "auto submitt" --> · [Lane fallback](../src/runtime/execution/decoder_prefill_staged_gqa.cpp#L72)<!-- prefix: "if (!submitt" --> · [KV tail](../src/text/gqa/tmem_staged_kernels.cu#L81)<!-- prefix: "if (logical " -->

### Long serial TP2 tiles

The ordinary long serial TP2 route stages 992–1,024 rows at 32,769–131,072 visible tokens, including new rows. Each rank has 12 Q heads and two KV heads of dimension 256. The page reader stages BF16 K/V, masks causal tails, and writes only live rows. See [warm cached GQA](flash-prefill.md#warm-cached-gqa-through-the-full-context) for the dedicated route. [Shape guard](../src/text/gqa/geometry.cpp#L48)<!-- prefix: "bool wants_l" --> · [Serial reader](../src/text/gqa/tmem_staged.cu#L604)<!-- prefix: "std::optiona" -->

A mixed 4,096-row group can use this reader after grouped readers decline. Contiguous causal groups use their [causal reader](flash-prefill.md#same-request-causal-gqa). Clean setup decline uses ordinary per-lane GQA. CUDA errors propagate. [Lane fallback](../src/runtime/execution/decoder_prefill_staged_gqa.cpp#L72)<!-- prefix: "if (!submitt" --> · [Reader dispatch](../src/runtime/execution/decoder_prefill_tiles.cpp#L328)<!-- prefix: "if (long_sta" -->

For the ordinary serial route, optional [dedicated scratch](native-prefill-workspace.md#dedicated-long-context-staging) follows the largest eligible span per step. It uses up to 268,443,648 B per rank for page-rounded K/V and metadata. On clean OOM the failing rank retries the base arena and both ranks use ordinary attention. [Span planning](../src/runtime/execution/decoder_prefill_plan.cpp#L202)<!-- prefix: "int long_ser" --> · [Both-rank gate](../src/runtime/execution/decoder_prefill_tiles.cpp#L304)<!-- prefix: "PrefillGqaSt" -->

## Constraints and failure behavior

Visibility is `position + rows`.

| Ordered fallback | Eligibility and tiling |
| --- | --- |
| Ragged TMEM | B200 text, 96–128 rows at 28,672–32,768 visible, with valid aligned buffers. Two physical tiles cover 128 queries with row-local causality. [Guard](../src/text/gqa/prefill_common.cu#L392)<!-- prefix: "bool default" --> |
| Warm staged ordered | 64–143 rows, start at least 512 tokens (four pages), and fewer than 8,192 visible tokens. K/V payloads require 16-byte alignment and scales require 8-byte alignment. Use four-row tiles. For 64–127 rows, a two-row tile handles the first two leftovers when available, and a protected CTA handles any final row. For 128–143 rows, use protected single-row CTAs for all leftover rows. [Dispatch](../src/text/gqa/ordered_prefill.cu#L356)<!-- prefix: "if (stage_pa" --> · [Tail handling](../src/text/gqa/ordered_prefill.cu#L287)<!-- prefix: "if constexpr" --> · [Alignment](../src/text/gqa/ordered_prefill.cu#L395)<!-- prefix: "const bool a" --> |
| Staged ordered | 2–143 rows at 8,192–262,144 visible tokens. Align both KV payloads to 16 B and both scales to 8 B. Use four-row tiles from 32 rows, two-row tiles for even 12–30 rows, otherwise one row. Protect each four-row remainder. [Eligibility](../src/text/gqa/ordered_prefill.cu#L362)<!-- prefix: "if (stage_pa" --> · [Tile size](../src/text/gqa/ordered_prefill.cu#L363)<!-- prefix: "if (rows \u003e= " --> · [Remainder](../src/text/gqa/ordered_prefill.cu#L298)<!-- prefix: "protected_at" --> |
| Direct ordered | Through 32,768 visible, use two-row tiles for 64–127 rows and four-row from 128. Smaller grants, leftovers and longer prefixes use protected single-row if earlier routes decline. Staged and direct paths recover nonfinite visible scores by head and nonfinite quotients by channel in FP64. [Fallback](../src/text/gqa/ordered_prefill.cu#L371)<!-- prefix: "if (rows \u003c m" --> · [Recovery](../include/sglang/mxfp8_gqa.h#L117)<!-- prefix: "cudaError_t " --> |

Warm staged tiles double-buffer packed K/V in static block shared memory and use zero dynamic shared memory. Both tile shapes preserve the ordered FP32 score and accumulation path, including protected FP64 repair. In the measured linked SM100 build, the four-row kernel used 40,288 B of static shared memory per block and the specialization with a two-row tail used 41,856 B. Both had 128 registers per thread, a 144 B stack frame, and zero reported local memory. [Tile and repair](../src/text/gqa/ordered_prefill.cu#L253)<!-- prefix: "run_ordered_" --> · [Launch](../src/text/gqa/ordered_prefill.cu#L338)<!-- prefix: "cudaError_t " -->

Zero rows succeed. For other inputs, negative positions, negative rows, and position-plus-row overflow return `cudaErrorInvalidValue`.

## Verification

Run the [GQA checker](development.md#gqa-verification) on a reserved B200. The [public multirow overflow control](../tests/mxfp8_gqa_prefill_check.cu#L691)<!-- prefix: "void check_o" --> covers 65 rows at 65, 32,768, 32,769, 131,072, and 131,073 visible tokens. It compares aligned graph replay against an FP64 known answer and requires bitwise parity with unaligned K. The [evidence gate](../tests/batch/check_graph_evidence.py#L346)<!-- prefix: "def verify_o" --> requires the matching CUPTI graph sequence and CTA counts of 33, 17, 2, 2, and 2.

Replay identical checkpoint Q and KV with remapped pages and guarded outputs. Compare complete BF16 outputs and the corresponding rows from a shortened causal grant. Run memcheck, initcheck, and synccheck on eager replays with unwritten tail slots.

For grouped staging, compare with ordinary per-lane dispatch. Use unequal 256–1,024-row grants with distinct starts and remapped page tables. Include position zero, non-page-aligned starts, partial KV and query tails, and a group with leftover lanes. Check visibility at 32,768 and immediately above it. Compare all BF16 outputs and verify that inactive rows stay unwritten.

To isolate the page-wide reader, keep the grants, staged K/V, and page remaps fixed. Compare all BF16 outputs bitwise with the prior 64-key staged reader. Include a final page with only one live 64-key chunk, an invalid visible KV entry, and an unwritten KV tail. Check lane-local repair and output canaries. [64-key control](https://github.com/para1lel/sglang.cpp/blob/0d03fd96f1f5d6d25322913bcfb3732e9c6472e2/include/sglang/text/mxfp8_gqa_prefill_paired.cuh#L384)<!-- prefix: "__global__ _" -->

Before the final conversion of recovery-mask arithmetic to unsigned integers, the B200 attention fixture matched baseline BF16 digests in all 508 cases across TP1 geometry and TP2 rank-local geometry, eager and captured dispatch, and protected and misaligned controls. In 27 eligible captured cases per geometry, the ratios of baseline to warm-kernel median time were 1.15–1.51× for TP1 and 1.15–1.48× for TP2 rank-local geometry. The native TP1 and TP2 checks compared 127- and 128-row warm continuations and matched token IDs, reported logprobs, usage, and finish reasons. Integrated prefill latency, serving throughput, and AgentX were unmeasured.

Use identical scheduled grants and token histories to compare full BF16 vocabulary scores and greedy IDs. For isolated operator timing, report the B200 rank configuration, row count, and visible span.

Compare both B200 rank shards at 1,004 rows and 65,516 visible tokens. Restore a 65,536-token prefix and compare 1- and 32-token continuations and rank-local GQA, GDN, and draft state. Inject clean pool OOM on either GPU and require both ranks to select ordinary attention.

Reject staging on insufficient scratch and check the eligible raw grouped or per-lane fallback. Inject an unexpected CUDA launch error and require propagation without a second attention launch. Include invalid visible KV and check lane-local repair. Follow the [numerical and quality checks](performance.md#operator-qualification).
