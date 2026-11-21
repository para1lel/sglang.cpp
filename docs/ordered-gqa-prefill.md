# Ordered MXFP8 GQA prefill

## Responsibilities and sources of truth

| Part | Responsibility | Source |
| --- | --- | --- |
| Model prefill | Append KV and select the attention path. | [Caller](../src/decoder_prefill.cpp#L323)<!-- prefix: "const auto s" --> · [Policy](../src/decoder_prefill.cpp#L610)<!-- prefix: "const bool t" --> |
| Native dispatcher | Select Flash, staged or direct ordered tiles, or protected single-row attention. | [Dispatch](../src/mxfp8_gqa_prefill.cu#L839)<!-- prefix: "cudaError_t " --> |
| Ordered tile | Apply causal FP32 scoring and token-ordered accumulation. | [Score](../src/mxfp8_gqa_prefill.cu#L49)<!-- prefix: "__device__ v" --> · [Tile loop](../src/mxfp8_gqa_prefill.cu#L201)<!-- prefix: "for (int fir" --> |

## Runtime path

After KV append, the tensor-core model wrapper tries the [SM100 page-128 opt-ins](sm100-page128-prefill.md). When they decline, the wrapper forwards the same position and row count to native dispatch. Exact and draft multirow tiles use direct MXFP8 attention. [Wrapper](../include/sglang/text/prefill_attn_sm100.h#L38)<!-- prefix: "if (const au" -->

[Grouped text prefill](native-prefill-workspace.md#multi-request-text-prefill) calls attention for each request with its original grant size and position. A one-row model tile uses [ordinary forward execution](../src/decoder_prefill.cpp#L590)<!-- prefix: "if (rows == " -->.

Staged tiles double-buffer packed K/V and scales in 32-key blocks. The loads follow the page table and zero-fill slots outside the tile's visible span. Scoring and accumulation apply each row's causal limit. [Copies](../src/mxfp8_gqa_prefill.cu#L421)<!-- prefix: "__device__ v" -->

Ordered tiles share decoded K/V across rows and two query heads. They preserve the FP32 score tree, token order for softmax and value updates, and BF16 rounding. When all active heads have a finite prior maximum and no higher visible score, the tile skips the unit history multipliers. [Warp vote](../src/mxfp8_gqa_prefill.cu#L104)<!-- prefix: "if (isfinite" --> · [Block vote](../src/mxfp8_gqa_prefill.cu#L224)<!-- prefix: "const bool u" -->

## Constraints and failure behavior

The native dispatcher computes visible length as `position + rows`. Staged ordered attention accepts 2–143 rows and 8,192–32,768 visible tokens, inclusive. Both KV payload bases need 16-byte alignment. Both scale bases need 8-byte alignment. [Eligibility](../src/mxfp8_gqa_prefill.cu#L901)<!-- prefix: "if (rows \u003e= " -->

Staged attention uses four-row tiles at 32 rows or more. It uses two-row tiles for even counts from 12 through 30. Other eligible counts use single-row tiles. Each remainder row from a four-row tile uses protected attention. [Tile size](../src/mxfp8_gqa_prefill.cu#L907)<!-- prefix: "const int ti" --> · [Remainder](../src/mxfp8_gqa_prefill.cu#L274)<!-- prefix: "if (tile \u003e= " -->

Through 32,768 visible tokens, direct ordered fallback uses two-row tiles at 64–127 rows and four-row tiles at 128 or more. Smaller grants, leftovers, and longer prefixes use protected single-row attention. Both routes recover nonfinite visible scores by head and nonfinite quotients by channel using FP64. [Fallback](../src/mxfp8_gqa_prefill.cu#L916)<!-- prefix: "if (rows \u003c m" --> · [Recovery](../include/sglang/mxfp8_gqa.h#L63)<!-- prefix: "/// Shared a" -->

Zero rows succeed. For other inputs, negative positions, negative rows, and position-plus-row overflow return `cudaErrorInvalidValue`.

## Verification

Run the [P2 GQA checker](development.md#p2-gqa-verification) on a reserved B200. The [overflow control](../tests/mxfp8_gqa_prefill_check.cu#L798)<!-- prefix: "void check_o" --> compares staged graph replay at 65 rows and 32,768 visible tokens with an FP64 known answer. It also checks an unaligned K payload and the 32,769-token fallback.

Replay identical checkpoint Q and KV with remapped pages and guarded outputs. Compare complete BF16 outputs and the corresponding rows from a shortened causal grant. Run memcheck, initcheck, and synccheck on eager replays with unwritten tail slots.

Use identical scheduled grants and token histories to compare full BF16 vocabulary scores and greedy IDs. Report operator replay timings and trace-weighted kernel estimates separately from end-to-end serving measurements.
