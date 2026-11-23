# Ordered MXFP8 GQA prefill

## Responsibilities and sources of truth

| Part | Responsibility | Source |
| --- | --- | --- |
| Model prefill | Append KV and select the attention path. | [Caller](../src/decoder_prefill.cpp#L219)<!-- prefix: "const auto s" --> · [Policy](../src/decoder_prefill.cpp#L878)<!-- prefix: "const bool t" --> |
| Native dispatcher | Select Flash, staged or direct ordered tiles, or protected single-row attention. | [Dispatch](../src/mxfp8_gqa_prefill.cu#L924)<!-- prefix: "cudaError_t " --> |
| Ordered tile | Apply causal FP32 scoring and token-ordered accumulation. | [Score](../src/mxfp8_gqa_prefill.cu#L35)<!-- prefix: "__device__ v" --> · [Tile loop](../src/mxfp8_gqa_prefill.cu#L187)<!-- prefix: "for (int fir" --> |

## Runtime path

After KV append, the Tensor Core model path uses native attention dispatch at the same position and row count. Exact multirow tiles use direct MXFP8 attention. See the [Flash dispatcher](flash-prefill.md#runtime-path) for shape and device selection.

[Batched staged attention](flash-prefill.md#runtime-path) is available for eligible full-tile groups. A one-row model tile uses [ordinary forward execution](../src/decoder_prefill.cpp#L847)<!-- prefix: "if (rows == " -->.

Staged tiles double-buffer packed K/V and scales in 32-key blocks. The loads follow the page table and zero-fill slots outside the tile's visible span. Scoring and accumulation apply each row's causal limit. [Copies](../include/sglang/text/mxfp8_gqa_prefill_common.cuh#L112)<!-- prefix: "__device__ v" -->

Ordered tiles share decoded K/V across rows and two query heads. They preserve the FP32 score tree, token order for softmax and value updates, and BF16 rounding. When all active heads have a finite prior maximum and no higher visible score, the tile skips the unit history multipliers. [Warp vote](../src/mxfp8_gqa_prefill.cu#L90)<!-- prefix: "if (isfinite" --> · [Block vote](../src/mxfp8_gqa_prefill.cu#L210)<!-- prefix: "const bool u" -->

### Grouped partial grants

The [partial-grant projection path](native-prefill-workspace.md#partial-grant-projection-groups) tries ordered TMEM on SM100 with 148 SMs. The group needs a multiple of four equal grants. Each grant has 256 or 512 rows. Each lane starts at 8,192 or later and sees at most 32,768 tokens. Each lane uses its own start, page table, and causal mask. Its cache payload bases require 16-byte alignment and its scale bases require eight-byte alignment. [Decoder guard](../src/decoder_prefill.cpp#L384)<!-- prefix: "if (count % " --> · [Reader guard](../src/mxfp8_gqa_prefill.cu#L889)<!-- prefix: "std::optiona" -->

The decoder appends KV separately for four lanes before the shared attention launch. The decoder gates each lane afterward or defers gating for eligible [fused postgate and quantization](fp8-prefill-projections.md#attention-output-quantization). When preflight declines the launch, the decoder reads those lanes separately with the existing workspace and already appended KV. CUDA errors returned by setup or launch propagate without fallback. [Append and dispatch](../src/decoder_prefill.cpp#L397)<!-- prefix: "segment.gqa_" --> · [Per-lane fallback](../src/decoder_prefill.cpp#L408)<!-- prefix: "if (!submitt" --> · [TMEM launch](../include/sglang/text/mxfp8_gqa_prefill_tmem_batch.cuh#L25)<!-- prefix: "std::optiona" -->

## Constraints and failure behavior

The native dispatcher computes visible length as `position + rows`. B200 text can use ragged TMEM for 96–128 rows and 28,672–32,768 visible tokens with valid aligned buffers. Two physical query tiles cover 128 rows and each row keeps its own causal limit. [Guard](../include/sglang/text/mxfp8_gqa_prefill_common.cuh#L469)<!-- prefix: "inline bool " -->

Staged ordered attention accepts 2–143 rows and 8,192–32,768 visible tokens, inclusive. Both KV payload bases need 16-byte alignment. Both scale bases need 8-byte alignment. [Eligibility](../src/mxfp8_gqa_prefill.cu#L974)<!-- prefix: "if (rows \u003e= " -->

Staged attention uses four-row tiles at 32 rows or more. It uses two-row tiles for even counts from 12 through 30. Other eligible counts use single-row tiles. Each remainder row from a four-row tile uses protected attention. [Tile size](../src/mxfp8_gqa_prefill.cu#L980)<!-- prefix: "const int ti" --> · [Remainder](../src/mxfp8_gqa_prefill.cu#L260)<!-- prefix: "if (tile \u003e= " -->

Through 32,768 visible tokens, direct ordered fallback uses two-row tiles at 64–127 rows and four-row tiles at 128 or more. Smaller grants, leftovers, and longer prefixes use protected single-row attention. Both routes recover nonfinite visible scores by head and nonfinite quotients by channel using FP64. [Fallback](../src/mxfp8_gqa_prefill.cu#L989)<!-- prefix: "if (rows \u003c m" --> · [Recovery](../include/sglang/mxfp8_gqa.h#L65)<!-- prefix: "/// Shared a" -->

Zero rows succeed. For other inputs, negative positions, negative rows, and position-plus-row overflow return `cudaErrorInvalidValue`.

## Verification

Run the [P2 GQA checker](development.md#p2-gqa-verification) on a reserved B200. The [overflow control](../tests/mxfp8_gqa_prefill_check.cu#L798)<!-- prefix: "void check_o" --> compares staged graph replay at 65 rows and 32,768 visible tokens with an FP64 known answer. It also checks an unaligned K payload and the 32,769-token fallback.

Replay identical checkpoint Q and KV with remapped pages and guarded outputs. Compare complete BF16 outputs and the corresponding rows from a shortened causal grant. Run memcheck, initcheck, and synccheck on eager replays with unwritten tail slots.

For grouped partial grants, compare all BF16 output rows with per-lane dispatch. Test four equal 256-row or 512-row grants with different positions and page tables. Include unequal grants and starts below 8,192. Check visibility at 32,768 and immediately above it. Inject a setup decline and require per-lane output parity. Inject a launch error and require propagation without a second attention launch.

Use identical scheduled grants and token histories to compare full BF16 vocabulary scores and greedy IDs. Report operator replay timings and trace-weighted kernel estimates separately from end-to-end serving measurements.
