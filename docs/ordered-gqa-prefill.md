# Ordered MXFP8 GQA prefill

## Responsibilities and sources of truth

| Part | Responsibility | Source |
| --- | --- | --- |
| Model prefill | Append uncovered KV and select the attention path. | [Caller](../src/runtime/execution/decoder_prefill_tiles.cpp#L224)<!-- prefix: "const auto s" --> · [Policy](../src/decoder_prefill.cpp#L613)<!-- prefix: "const bool t" --> |
| Native dispatcher | Select Flash, staged or direct ordered tiles, or protected single-row attention. | [Dispatch](../src/text/gqa/ordered_prefill.cu#L292)<!-- prefix: "cudaError_t " --> |
| Ordered tile | Apply causal FP32 scoring and token-ordered accumulation. | [Score](../src/text/gqa/ordered_prefill.cu#L33)<!-- prefix: "__device__ v" --> · [Tile loop](../src/text/gqa/ordered_prefill.cu#L187)<!-- prefix: "for (int fir" --> |

## Runtime path

After KV append, the Tensor Core model path uses native attention dispatch at the same position and row count. Exact multirow tiles use direct MXFP8 attention. See the [Flash dispatcher](flash-prefill.md#runtime-path) for shape and device selection.

[Batched staged attention](flash-prefill.md#runtime-path) is available for eligible full-tile and partial-grant groups. A one-row model tile uses [ordinary forward execution](../src/decoder_prefill.cpp#L582)<!-- prefix: "if (rows == " -->.

Staged tiles double-buffer packed K/V and scales in 32-key blocks. The loads follow the page table and zero-fill slots outside the tile's visible span. Scoring and accumulation apply each row's causal limit. [Copies](../src/text/gqa/prefill_common.cu#L27)<!-- prefix: "__device__ _" -->

Ordered tiles share decoded K/V across rows and two query heads. They preserve the FP32 score tree, token order for softmax and value updates, and BF16 rounding. When all active heads have a finite prior maximum and no higher visible score, the tile skips the unit history multipliers. [Warp vote](../src/text/gqa/ordered_prefill.cu#L88)<!-- prefix: "if (isfinite" --> · [Block vote](../src/text/gqa/ordered_prefill.cu#L210)<!-- prefix: "const bool u" -->

### Grouped partial grants

The [partial-grant projection path](native-prefill-workspace.md#partial-grant-projection-groups) stages successive four-lane subgroups. Each lane may start at any nonnegative position and take 256–1,024 rows with at most 32,768 visible tokens. Positions and row counts can differ. SM100 needs 16-byte-aligned KV payload and scale bases and sufficient disjoint scratch. Each lane uses its own page table and causal mask. [Grouping](../src/runtime/execution/decoder_prefill_staged_gqa.cpp#L84)<!-- prefix: "for (; first" --> · [Staged guard](../src/text/gqa/tmem_staged.cu#L494)<!-- prefix: "if (rows \u003c m" --> · [Buffer validation](../src/text/mxfp8_gqa_prefill_tmem.cu#L650)<!-- prefix: "std::optiona" -->

The decoder appends each lane’s new KV before trying one predecode and one [page-wide attention](flash-prefill.md#runtime-path) launch per subgroup. The reader masks unwritten KV tails and inactive query rows. If staging declines, raw grouped ordered TMEM remains eligible on 148-SM SM100. That fallback requires a group divisible by four, equal 256- or 512-row grants, and starts at 8,192 or later. If neither reader submits, each lane, including leftovers, uses its own reader against retained and newly appended KV. If predecode has queued, the stream orders fallback after it. Pending and unexpected CUDA errors propagate. The decoder gates each lane afterward or defers gating for [fused postgate and quantization](fp8-prefill-projections.md#attention-output-quantization). [Staged launch](../src/runtime/execution/decoder_prefill_staged_gqa.cpp#L97)<!-- prefix: "auto submitt" --> · [Lane fallback](../src/runtime/execution/decoder_prefill_staged_gqa.cpp#L107)<!-- prefix: "if (!submitt" --> · [KV tail](../src/text/gqa/tmem_staged_kernels.cu#L81)<!-- prefix: "if (logical " -->

### Long serial TP2 tiles

On B200, eager Tensor Core TP2 prefill can stage native MXFP8 GQA for 992–1,024 rows and 32,769–131,072 visible tokens. Visibility includes the new rows. Each rank uses 12 query heads and two KV heads of dimension 256. The reader stages BF16 K and V by page, masks each row’s causal tail, and writes only live query rows. [Shape guard](../src/text/gqa/geometry.cpp#L48)<!-- prefix: "bool wants_l" --> · [Native reader](../src/text/gqa/tmem_staged.cu#L621)<!-- prefix: "std::optiona" -->

A mixed 4,096-row projection group can use this reader for an eligible lane after grouped readers decline. A contiguous 4,096-row causal attention group continues to use its [causal reader](flash-prefill.md#same-request-causal-gqa). A clean reader setup decline resumes ordinary per-lane GQA. CUDA errors propagate. [Lane fallback](../src/runtime/execution/decoder_prefill_staged_gqa.cpp#L107)<!-- prefix: "if (!submitt" --> · [Reader dispatch](../src/runtime/execution/decoder_prefill_tiles.cpp#L309)<!-- prefix: "void Chunk::" -->

Each rank sizes optional workspace for the largest eligible visible span in the step. Page-rounded BF16 K and V planes and per-page metadata add up to 268,443,648 bytes per rank. If either rank hits a clean allocation OOM, that rank retries with the base arena. Both ranks then use ordinary attention for these tiles. A rank that allocated the extra space releases it at normal stream-ordered close. The existing pool policy governs retention. [Span planning](../src/runtime/execution/decoder_prefill_plan.cpp#L28)<!-- prefix: "int long_ser" --> · [Workspace sizing](../src/runtime/execution/decoder_prefill_workspace.cpp#L282)<!-- prefix: "const auto b" --> · [Base retry](../src/runtime/execution/decoder_prefill_workspace.cpp#L294)<!-- prefix: "if (plane &&" --> · [Both-rank gate](../src/decoder_prefill.cpp#L395)<!-- prefix: "PairedPrefil" --> · [Release](../src/runtime/execution/decoder_prefill_workspace.cpp#L413)<!-- prefix: "cudaError_t " -->

## Constraints and failure behavior

The native dispatcher computes visible length as `position + rows`. B200 text can use ragged TMEM for 96–128 rows and 28,672–32,768 visible tokens with valid aligned buffers. Two physical query tiles cover 128 rows and each row keeps its own causal limit. [Guard](../src/text/gqa/prefill_common.cu#L384)<!-- prefix: "bool default" -->

Staged ordered attention accepts 2–143 rows and 8,192–131,072 visible tokens, inclusive. Both KV payload bases need 16-byte alignment. Both scale bases need 8-byte alignment. [Eligibility](../src/text/gqa/ordered_prefill.cu#L341)<!-- prefix: "if (rows \u003e= " -->

Staged attention uses four-row tiles at 32 rows or more. It uses two-row tiles for even counts from 12 through 30. Other eligible counts use single-row tiles. Each remainder row from a four-row tile uses protected attention. [Tile size](../src/text/gqa/ordered_prefill.cu#L347)<!-- prefix: "const int ti" --> · [Remainder](../src/text/gqa/ordered_prefill.cu#L258)<!-- prefix: "if (tile \u003e= " -->

Through 32,768 visible tokens, direct ordered fallback uses two-row tiles at 64–127 rows and four-row tiles at 128 or more. Smaller grants, leftovers, and longer prefixes use protected single-row attention. Both routes recover nonfinite visible scores by head and nonfinite quotients by channel using FP64. [Fallback](../src/text/gqa/ordered_prefill.cu#L356)<!-- prefix: "if (rows < m" --> · [Recovery](../include/sglang/mxfp8_gqa.h#L113)<!-- prefix: "cudaError_t " -->

Zero rows succeed. For other inputs, negative positions, negative rows, and position-plus-row overflow return `cudaErrorInvalidValue`.

## Verification

Run the [P2 GQA checker](development.md#p2-gqa-verification) on a reserved B200. The [overflow control](../tests/mxfp8_gqa_prefill_check.cu#L797)<!-- prefix: "void check_o" --> compares aligned graph replay for 65 rows at 65, 32,768, 32,769, 131,072, and 131,073 visible tokens against an FP64 known answer and requires bitwise parity with unaligned K. The [evidence gate](../tests/batch/check_graph_evidence.py#L293)<!-- prefix: "def verify_o" --> requires the matching CUPTI graph sequence and CTA counts of 33, 17, 17, 17, and 65.

Replay identical checkpoint Q and KV with remapped pages and guarded outputs. Compare complete BF16 outputs and the corresponding rows from a shortened causal grant. Run memcheck, initcheck, and synccheck on eager replays with unwritten tail slots.

For grouped staging, compare with ordinary per-lane dispatch. Use unequal 256–1,024-row grants with distinct starts and remapped page tables. Include position zero, non-page-aligned starts, partial KV and query tails, and a group with leftover lanes. Check visibility at 32,768 and immediately above it. Compare all BF16 outputs and verify that inactive rows stay unwritten.

To isolate the page-wide reader, keep the grants, staged K/V, and page remaps fixed. Compare all BF16 outputs bitwise with the prior 64-key staged reader. Include a final page with only one live 64-key chunk, an invalid visible KV entry, and an unwritten KV tail. Check lane-local repair and output canaries. [64-key control](https://github.com/para1lel/sglang.cpp/blob/0d03fd96f1f5d6d25322913bcfb3732e9c6472e2/include/sglang/text/mxfp8_gqa_prefill_paired.cuh#L384)<!-- prefix: "__global__ _" -->

Use identical scheduled grants and token histories to compare full BF16 vocabulary scores and greedy IDs. Report operator replay timings and trace-weighted kernel estimates separately from end-to-end serving measurements.

Compare both W5 rank shards at 1,004 rows and 65,516 visible tokens. Restore a 65,536-token prefix and compare 1- and 32-token continuations and rank-local GQA, GDN, and draft state. Inject clean pool OOM on either GPU and require both ranks to select ordinary attention. [Frozen operator and decoder replays](https://github.com/para1lel/sglang.cpp/blob/a5974ba4eb99bd8e8f890c2e7262a27dbc3aace1/docs/ordered-gqa-prefill.md#verification)

Reject staging on insufficient scratch and check the eligible raw grouped or per-lane fallback. Inject an unexpected CUDA launch error and require propagation without a second attention launch. Include invalid visible KV and check lane-local repair. Follow the [numerical and quality checks](performance.md#operator-qualification).
