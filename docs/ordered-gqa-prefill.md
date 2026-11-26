# Ordered MXFP8 GQA prefill

## Responsibilities and sources of truth

| Part | Responsibility | Source |
| --- | --- | --- |
| Model prefill | Append uncovered KV and select the attention path. | [Caller](../include/sglang/runtime/execution/decoder_prefill_tiles.h#L333)<!-- prefix: "const auto s" --> · [Policy](../src/decoder_prefill.cpp#L613)<!-- prefix: "const bool t" --> |
| Native dispatcher | Select Flash, staged or direct ordered tiles, or protected single-row attention. | [Dispatch](../include/sglang/text/gqa/multirow_prefill.cuh#L12)<!-- prefix: "cudaError_t " --> |
| Ordered tile | Apply causal FP32 scoring and token-ordered accumulation. | [Score](../src/mxfp8_gqa_prefill.cu#L35)<!-- prefix: "__device__ v" --> · [Tile loop](../src/mxfp8_gqa_prefill.cu#L189)<!-- prefix: "for (int fir" --> |

## Runtime path

After KV append, the Tensor Core model path uses native attention dispatch at the same position and row count. Exact multirow tiles use direct MXFP8 attention. See the [Flash dispatcher](flash-prefill.md#runtime-path) for shape and device selection.

[Batched staged attention](flash-prefill.md#runtime-path) is available for eligible full-tile and partial-grant groups. A one-row model tile uses [ordinary forward execution](../src/decoder_prefill.cpp#L582)<!-- prefix: "if (rows == " -->.

Staged tiles double-buffer packed K/V and scales in 32-key blocks. The loads follow the page table and zero-fill slots outside the tile's visible span. Scoring and accumulation apply each row's causal limit. [Copies](../include/sglang/text/mxfp8_gqa_prefill_common.cuh#L112)<!-- prefix: "__device__ v" -->

Ordered tiles share decoded K/V across rows and two query heads. They preserve the FP32 score tree, token order for softmax and value updates, and BF16 rounding. When all active heads have a finite prior maximum and no higher visible score, the tile skips the unit history multipliers. [Warp vote](../src/mxfp8_gqa_prefill.cu#L90)<!-- prefix: "if (isfinite" --> · [Block vote](../src/mxfp8_gqa_prefill.cu#L212)<!-- prefix: "const bool u" -->

### Grouped partial grants

The [partial-grant projection path](native-prefill-workspace.md#partial-grant-projection-groups) stages successive four-lane subgroups. Each lane may start at any nonnegative position and take 256–1,024 rows with at most 32,768 visible tokens. Positions and row counts can differ. SM100 needs 16-byte-aligned KV payload and scale bases and sufficient disjoint scratch. Each lane uses its own page table and causal mask. [Grouping](../include/sglang/runtime/execution/decoder_prefill_tiles.h#L620)<!-- prefix: "for (; first" --> · [Staged guard](../include/sglang/text/mxfp8_gqa_prefill_paired.cuh#L804)<!-- prefix: "if (rows \u003c m" --> · [Buffer validation](../include/sglang/text/mxfp8_gqa_prefill_paired.cuh#L516)<!-- prefix: "std::optiona" -->

The decoder appends each lane’s new KV before trying one predecode and one [page-wide attention](flash-prefill.md#runtime-path) launch per subgroup. The reader masks unwritten KV tails and inactive query rows. If staging declines, raw grouped ordered TMEM remains eligible on 148-SM SM100. That fallback requires a group divisible by four, equal 256- or 512-row grants, and starts at 8,192 or later. If neither reader submits, each lane, including leftovers, uses its own reader against retained and newly appended KV. If predecode has queued, the stream orders fallback after it. Pending and unexpected CUDA errors propagate. The decoder gates each lane afterward or defers gating for [fused postgate and quantization](fp8-prefill-projections.md#attention-output-quantization). [Staged launch](../include/sglang/runtime/execution/decoder_prefill_tiles.h#L633)<!-- prefix: "auto submitt" --> · [Lane fallback](../include/sglang/runtime/execution/decoder_prefill_tiles.h#L643)<!-- prefix: "if (!submitt" --> · [KV tail](../include/sglang/text/mxfp8_gqa_prefill_paired.cuh#L127)<!-- prefix: "if (logical " -->

### Long serial TP2 tiles

On B200, eager Tensor Core TP2 prefill can stage native MXFP8 GQA for 992–1,024 rows and 32,769–131,072 visible tokens. Visibility includes the new rows. Each rank uses 12 query heads and two KV heads of dimension 256. The reader stages BF16 K and V by page, masks each row’s causal tail, and writes only live query rows. [Shape guard](../include/sglang/text/prefill_attn_sm100.h#L43)<!-- prefix: "constexpr bo" --> · [Native reader](../include/sglang/text/mxfp8_gqa_prefill_paired.cuh#L931)<!-- prefix: "std::optiona" -->

A mixed 4,096-row projection group can use this reader for an eligible lane after grouped readers decline. A contiguous 4,096-row causal attention group continues to use its [causal reader](flash-prefill.md#same-request-causal-gqa). A clean reader setup decline resumes ordinary per-lane GQA. CUDA errors propagate. [Lane fallback](../include/sglang/runtime/execution/decoder_prefill_tiles.h#L643)<!-- prefix: "if (!submitt" --> · [Reader dispatch](../include/sglang/runtime/execution/decoder_prefill_tiles.h#L419)<!-- prefix: "void gqa_rea" -->

Each rank sizes optional workspace for the largest eligible visible span in the step. Page-rounded BF16 K and V planes and per-page metadata add up to 268,443,648 bytes per rank. If either rank hits a clean allocation OOM, that rank retries with the base arena. Both ranks then use ordinary attention for these tiles. A rank that allocated the extra space releases it at normal stream-ordered close. The existing pool policy governs retention. [Span planning](../include/sglang/runtime/execution/decoder_prefill_plan.h#L46)<!-- prefix: "inline int l" --> · [Workspace sizing](../src/runtime/execution/decoder_prefill_workspace.cpp#L225)<!-- prefix: "const auto b" --> · [Base retry](../src/runtime/execution/decoder_prefill_workspace.cpp#L237)<!-- prefix: "if (plane &&" --> · [Both-rank gate](../src/decoder_prefill.cpp#L303)<!-- prefix: "PairedPrefil" --> · [Release](../src/runtime/execution/decoder_prefill_workspace.cpp#L266)<!-- prefix: "cudaError_t " -->

## Constraints and failure behavior

The native dispatcher computes visible length as `position + rows`. B200 text can use ragged TMEM for 96–128 rows and 28,672–32,768 visible tokens with valid aligned buffers. Two physical query tiles cover 128 rows and each row keeps its own causal limit. [Guard](../include/sglang/text/mxfp8_gqa_prefill_common.cuh#L478)<!-- prefix: "inline bool " -->

Staged ordered attention accepts 2–143 rows and 8,192–131,072 visible tokens, inclusive. Both KV payload bases need 16-byte alignment. Both scale bases need 8-byte alignment. [Eligibility](../include/sglang/text/gqa/multirow_prefill.cuh#L61)<!-- prefix: "if (rows \u003e= " -->

Staged attention uses four-row tiles at 32 rows or more. It uses two-row tiles for even counts from 12 through 30. Other eligible counts use single-row tiles. Each remainder row from a four-row tile uses protected attention. [Tile size](../include/sglang/text/gqa/multirow_prefill.cuh#L67)<!-- prefix: "const int ti" --> · [Remainder](../src/mxfp8_gqa_prefill.cu#L260)<!-- prefix: "if (tile \u003e= " -->

Through 32,768 visible tokens, direct ordered fallback uses two-row tiles at 64–127 rows and four-row tiles at 128 or more. Smaller grants, leftovers, and longer prefixes use protected single-row attention. Both routes recover nonfinite visible scores by head and nonfinite quotients by channel using FP64. [Fallback](../include/sglang/text/gqa/multirow_prefill.cuh#L76)<!-- prefix: "if (rows < m" --> · [Recovery](../include/sglang/mxfp8_gqa.h#L121)<!-- prefix: "cudaError_t " -->

Zero rows succeed. For other inputs, negative positions, negative rows, and position-plus-row overflow return `cudaErrorInvalidValue`.

## Verification

Run the [P2 GQA checker](development.md#p2-gqa-verification) on a reserved B200. The [overflow control](../tests/mxfp8_gqa_prefill_check.cu#L798)<!-- prefix: "void check_o" --> compares staged graph replay at 65 rows and 32,768 visible tokens with an FP64 known answer. It also checks an unaligned K payload and the 32,769-token fallback.

Replay identical checkpoint Q and KV with remapped pages and guarded outputs. Compare complete BF16 outputs and the corresponding rows from a shortened causal grant. Run memcheck, initcheck, and synccheck on eager replays with unwritten tail slots.

For grouped staging, compare with ordinary per-lane dispatch. Use unequal 256–1,024-row grants with distinct starts and remapped page tables. Include position zero, non-page-aligned starts, partial KV and query tails, and a group with leftover lanes. Check visibility at 32,768 and immediately above it. Compare all BF16 outputs and verify that inactive rows stay unwritten.

To isolate the page-wide reader, keep the grants, staged K/V, and page remaps fixed. Compare all BF16 outputs bitwise with the prior 64-key staged reader. Include a final page with only one live 64-key chunk, an invalid visible KV entry, and an unwritten KV tail. Check lane-local repair and output canaries. [64-key control](https://github.com/para1lel/sglang.cpp/blob/0d03fd96f1f5d6d25322913bcfb3732e9c6472e2/include/sglang/text/mxfp8_gqa_prefill_paired.cuh#L384)<!-- prefix: "__global__ _" -->

Use identical scheduled grants and token histories to compare full BF16 vocabulary scores and greedy IDs. Report operator replay timings and trace-weighted kernel estimates separately from end-to-end serving measurements.

An operator replay on one B200 used captured W5 inputs from both rank shards, with 1,004 rows and 65,516 visible tokens. All 3,084,288 BF16 outputs per rank matched bitwise. Median device time fell from 6.408 to 1.034 ms on rank 0 and from 6.405 to 1.029 ms on rank 1. See `artifacts/tp2-resume/w5-current-native/` (`ragged.log`, `control.log`).

A focused two-B200 decoder replay restored a 65,536-token W5 prefix. It checked 1- and 32-token continuations for serial tiles and a 1,004-row tile inside a mixed 4,096-row projection group. Twelve clean pool OOMs, triggered on either rank, forced both ranks to ordinary attention for the affected tiles. Continuation IDs and rank-local target GQA, GDN, and rank-0 draft state matched byte for byte. W5 serving makespan remains unmeasured for this change. See `artifacts/tp2-resume/w5-current-native/production-e561917/replay-gpu0-6.log`.

Reject staging on insufficient scratch and check the eligible raw grouped or per-lane fallback. Inject an unexpected CUDA launch error and require propagation without a second attention launch. Include invalid visible KV and check lane-local repair. Follow the [numerical and quality checks](performance.md#operator-qualification).
