# FP8 prefill projections

## Responsibilities and sources of truth

| Part | Responsibility | Source |
| --- | --- | --- |
| Decoder | Select the prefill policy and supply activation and output buffers. | [Packed call](../src/decoder_prefill.cpp#L115)<!-- prefix: "bool project" --> |
| Fused RMSNorm | Produce BF16 normalized values, E4M3 activations, and FP32 activation scales for prefill and packed decode. | [Launcher](../src/text_normalization.cu#L453)<!-- prefix: "cudaError_t " --> |
| Attention output quantizer | Convert BF16 GDN and GQA output rows for their FP8 output projections. | [Row launcher](../src/text_ops.cu#L659)<!-- prefix: "cudaError_t " --> |
| SM100 operator | Select the tile, manage CTA-local storage, and write BF16 outputs. | [Separate dispatch](../src/prefill_fp8_sm100.cu#L585)<!-- prefix: "if (rows > 1" --> · [Packed dispatch](../src/prefill_fp8_sm100.cu#L694)<!-- prefix: "if (rows < 1" --> |

## Runtime path

### Exact SIMT projections

For eligible wide inputs, the exact projection selects the eight-row launcher at 64 or more prompt rows and more than 1,024 output channels. With 5,120, 6,144, or 17,408 input channels and an output width at most 65,535, the launcher can assign four prompt rows and four adjacent output channels to each warp if the output width is divisible by four. For 64–128 prompt rows and an output width divisible by eight, it assigns eight adjacent channels per warp and reuses each decoded FP8 operand across the corresponding rows or channels. Every output retains its two ordered FP32 dot-product chains, scaling, warp reduction, and BF16 rounding. Other eligible shapes use the existing row tiles. [Row dispatch](../src/text_ops.cu#L762)<!-- prefix: "if (rows \u003e 1" --> · [Tile selection](../include/sglang/text/prefill_fp8_exact.cuh#L225)<!-- prefix: "if constexpr" --> · [Operand reuse](../include/sglang/text/prefill_fp8_exact.cuh#L77)<!-- prefix: "__device__ _" -->

### Tensor-core projections

On B200, the native runtime selects packed Tensor Core projections for eligible prefill and target decode. Dispatch depends on the operation, device, buffers, and row shape. Media embeddings and scoring use the same target layers. [Prefix restore](prefix-cache.md#checkpoint-compatibility-and-resume) supplies lane state before the remaining rows run.

Short target projections use the SM100 Swap-AB FP8 operator for one to five text rows. It reads the FP32 activation scales and BF16 checkpoint weight scales and writes BF16 results to the destination rows. Multi-row calls use a row-major E4M3 activation buffer for an eight-column TMA tile. With five verification rows, the fused quantizers write directly into that buffer and keep the same per-row FP32 scales. Other multi-row calls can pack the activation values before launching the projection. GDN QKV and Z, GQA QKV, and MLP gate and up share one packed activation per group and write to separate BF16 destinations. Scalar text decode uses the same split planner as verification. This grouping changes where the FP32 partials are added, so comparisons between K0 and K4 use the same projection policy and implementation. Unsupported projections and devices use exact target dispatch, including the staged five-row kernel where eligible. A per-lane guard keeps multimodal decode on its mRoPE-aware path, including in a mixed batch.

For eligible GDN verification, one launch projects the BF16 b and a gates for two to five rows. A CTA loads one gate weight row into shared memory and each warp computes it for one verification row. The kernel keeps the two ordered FP32 accumulation chains, warp reduction, and BF16 rounding. Scalar decode, exact verification, and ineligible inputs use the existing gate projections. [Verification dispatch](../src/decoder_forward.cpp#L329)<!-- prefix: "void project" --> · [Gate launcher](../src/text_decode_bandwidth.cu#L565)<!-- prefix: "std::optiona" -->

### Multi-lane target verification

Four-lane packets batch activation quantization at six to twenty rows. With a packed activation buffer, [wide K4 verification](mtp.md#target-round) batches 40, 80, or 160 rows per launch. This covers RMSNorm, attention output, and SwiGLU quantization. [Compact GDN](gdn-prefill.md#grouped-target-verification) writes FP8 during finish, bypassing separate output quantization. Rows keep BF16 rounding, E4M3 values, FP32 scales, and output slots. [RMSNorm](../src/decoder_forward.cpp#L916)<!-- prefix: "if (batch.fu" --> · [Quantization](../src/decoder_forward.cpp#L165)<!-- prefix: "void quantiz" --> · [Wide launcher](../src/text_normalization.cu#L423)<!-- prefix: "cudaError_t " -->

Eligible B200 projections through 20 rows use 8-, 16-, or 32-column MMA tiles. At 5–20 rows, up to 348,160 E4M3 bytes fit in the first 496,640-byte logits buffer. M40/M80/M160 use separate packed storage and 64/80/160 columns. Compact M80/M160 use four/three pipeline slots. Column pairs use rounded FP32 multiply before FMA. K-groups accumulate in order. Split kernels reuse shared tiles for partials. Cluster barriers bracket an out-of-line reducer. The reducer loads four values per rank for output widths divisible by four. Sums keep rank order and round once to BF16. Single splits store directly. [Packing](../src/decoder_forward.cpp#L112)<!-- prefix: "auto* packed" --> · [Compact arithmetic](../src/text/decode_proj_fp8.cu#L71)<!-- prefix: "template <bo" --> · [Compact epilogue](../src/text/decode_proj_fp8.cu#L199)<!-- prefix: "__device__ _" --> · [Wide dispatch](../src/text/decode_proj_fp8.cu#L409)<!-- prefix: "if (count ==" -->

Compact setup decline tries 40 rows, then 20, then groups of at most five. M80/M160 check pending CUDA errors before setup and recorded status after dispatch. An unavailable launch target can fall back when setup returns no status and CUDA records only `cudaErrorNotSupported`. Explicit setup and dispatch errors propagate. [Fallback](../src/decoder_forward.cpp#L274)<!-- prefix: "void project" --> · [CUDA status](../src/text/decode_proj_fp8.cu#L382)<!-- prefix: "std::optiona" -->

The batch lazily allocates stream-owned scratch before submission (87,040 bytes per lane, up to 2,785,280 bytes). Resident batches check the 2 GiB shared reserve plus net growth. Reserve shortage or allocation OOM lowers the width cap and replans. Other CUDA errors propagate. [Scratch reserve](../src/decoder_batch.cpp#L708)<!-- prefix: "bool Decoder" --> · [Width downgrade](../src/decoder_batch.cpp#L738)<!-- prefix: "WideMtpGroup" -->

### Large prefill tiles

Separate tensor-core K0 projections require at least three rows, input widths divisible by 128, and contiguous, 16-byte-aligned E4M3 payloads. The output can end in a partial tile. At 129–1,023 rows, projections with at least 8,192 output channels or 6,144 input channels first try a paired-TMA 128×128 tile. The launch places the row tiles for each column tile next to one another to improve weight cache reuse. [Eligibility](../src/prefill_fp8_sm100.cu#L556)<!-- prefix: "const int pr" --> · [Separate projection tiles](../src/prefill_fp8_sm100.cu#L585)<!-- prefix: "if (rows > 1" -->

Packed projections also require contiguous weight and scale slices and each output slice width to be divisible by 128. For 1,024 rows and input width 5,120, eligible packed GDN QKV/Z (width 16,384) and GQA QKV (width 14,336) first try a 128×256 tile. The wide tile requires each output slice width to be divisible by 256. The CTA owns four TMA stages and its TMEM. Each of eight consumer warps handles 32 rows in one 128-column weight-scale block and reads 16 TMEM columns at a time. The epilogue writes the existing BF16 slices in the decoder workspace. [GDN call](../src/decoder_prefill.cpp#L142)<!-- prefix: "if (!project" --> · [GQA call](../src/decoder_prefill.cpp#L199)<!-- prefix: "if (!project" --> · [Scale block](../src/prefill_fp8_sm100.cu#L324)<!-- prefix: "const int sc" --> · [Output slice](../src/prefill_fp8_sm100.cu#L348)<!-- prefix: "const auto [" -->

At 1,024 rows and input width 5,120, the wide tile also supports eligible MLP gate/up packs with exactly two 17,408-column slices. The MLP launch groups eight row tiles per column tile. The GDN and GQA launches use one row tile per group. The MLP writes BF16 gate/up slices for the separate SwiGLU quantizer, which supplies the down projection. [MLP call](../src/decoder_prefill.cpp#L250)<!-- prefix: "if (!project" --> · [MLP slice check](../src/prefill_fp8_sm100.cu#L664)<!-- prefix: "const bool p" --> · [Row grouping](../src/prefill_fp8_sm100.cu#L689)<!-- prefix: "packed_mlp_c" --> · [SwiGLU quantization](../src/decoder_prefill.cpp#L258)<!-- prefix: "check(tensor" -->

At exactly 768 rows and input width 5,120, an eligible MLP gate/up pack first tries a 128×128 tile with three TMA stages and four consumer warps. The tile uses 256 TMEM columns and reads 16 columns at a time. If the tile is unavailable, dispatch continues to the existing packed fallbacks. At 256–1,023 rows, other eligible packs try paired TMA with a 128×256 tile when every slice width is divisible by 256, or a 128×128 tile otherwise. The launches group the row tiles for each column tile and keep the existing scale types, ordered FP32 K-group accumulation, and BF16 rounding. [Compact MLP dispatch](../src/prefill_fp8_sm100.cu#L676)<!-- prefix: "if ((rows ==" --> · [Paired packed dispatch](../src/prefill_fp8_sm100.cu#L694)<!-- prefix: "if (rows < 1" -->

[Whole-tile MTP pairs](native-prefill-workspace.md#projection-pairs-for-mtp) and
[partial-grant projection groups](native-prefill-workspace.md#partial-grant-projection-groups)
use a grouped dispatch limit of 4,096 rows. Ordinary large-row calls keep
their 2,048-row limit. Eligible grouped gate/up, GDN QKV/Z, and GQA QKV
packs at 2,048 or 4,096 rows and input width 5,120 first try the 128×256 TMA tile.
Dispatch groups all 16 or 32 row tiles per column tile.
For 4,096×6,144 attention output with 5,120 output channels, the same tile
provides a grouped fallback. These kernels preserve the scale types,
FP32 accumulation order, and BF16 rounding.
[Group dispatch](../src/decoder_prefill.cpp#L657)<!-- prefix: "const Chunk " --> · [Grouped packs](../src/prefill_fp8_sm100.cu#L667)<!-- prefix: "if (grouped_" --> · [Grouped attention output](../src/prefill_fp8_sm100.cu#L576)<!-- prefix: "if (grouped_" -->

The 1,024-row down fallback uses paired TMA when large-row prefill is enabled. Its 128×128 CTA tile uses six stages in three pairs. Each pair shares one TMA completion barrier and releases both stages after the second group’s MMA. The launch groups eight row tiles per weight tile. [Down dispatch](../src/prefill_fp8_sm100.cu#L569)<!-- prefix: "if (rows == " --> · [Producer](../src/prefill_fp8_sm100.cu#L89)<!-- prefix: "__device__ v" --> · [Release](../src/prefill_fp8_sm100.cu#L159)<!-- prefix: "commit_mma(&" -->

One CUTLASS/CuTe SM100 collective writes 5,120 BF16 channels. Grouped GQA and GDN output use 4,096 rows and 6,144 input channels. Tensor Core MLP down uses 1,024 or grouped 4,096 rows and 17,408 input channels. Distinct idle MLP gate/up spans hold transposed activation scales and widened FP32 weight scales. The workspace stream orders conversion and the two-CTA GEMM. A clean decline uses existing dispatch, including paired TMA for eligible 1,024-row down. Buffer, CUDA setup, descriptor, and submission failures propagate. [Group guard](../src/decoder_prefill.cpp#L659)<!-- prefix: "const bool s" --> · [Down dispatch](../src/decoder_prefill.cpp#L261)<!-- prefix: "project_scal" --> · [Operator](../src/text/prefill_down_sm100.cu#L151)<!-- prefix: "std::optiona" --> · [Scale preparation](../src/text/prefill_down_sm100.cu#L68)<!-- prefix: "__global__ v" -->

### Attention output quantization

For ordinary tensor-core text prefill, GDN combines gated RMSNorm with FP8 quantization. Each warp rounds one head to BF16 in registers before quantization. The output projection consumes the E4M3 values and FP32 scales directly. In grouped prefill, each grant writes its own rows in the shared activation buffer. The separate GQA gate and the BF16 GDN prefill path pass 6,144 channels per row to `text_quantize_fp8_rows`. [Fused finish](../src/text_attention_gdn.cu#L583)<!-- prefix: "__global__ v" --> · [GDN call](../src/decoder_prefill.cpp#L929)<!-- prefix: "chunk.gdn_at" --> · [GQA call](../src/decoder_prefill.cpp#L924)<!-- prefix: "chunk.projec" -->

On B200, [strict MTP groups](native-prefill-workspace.md#projection-pairs-for-mtp) of four 1,024-row tiles can fuse the GQA postgate and quantizer. [Partial-grant groups](native-prefill-workspace.md#partial-grant-projection-groups) of eight 512-row or sixteen 256-row grants can use the same preparation. Strict groups keep staged or per-lane readers. Partial groups keep staged, raw grouped ordered, or per-lane readers. The fused kernel rounds gated values to BF16 before writing E4M3 values and FP32 scales. If fusion is ineligible, strict groups gate the full buffer, and partial groups gate each grant. Both then use ordinary quantization. CUDA errors propagate. [Preparation policy](../src/decoder_prefill.cpp#L654)<!-- prefix: "const bool p" --> · [Postgate dispatch](../src/decoder_prefill.cpp#L684)<!-- prefix: "if (const au" --> · [Partial fallback](../src/decoder_prefill.cpp#L692)<!-- prefix: "if (wide_par" --> · [Eligibility and launch](../src/text_attention_gqa.cu#L340)<!-- prefix: "std::optiona" -->

For exactly 1,024 rows and 6,144 channels, the launcher uses the warp kernel if the BF16 input and E4M3 output are both 4-byte aligned. A 256-thread CTA handles eight 128-channel groups in one row. Each warp owns one group and each lane processes four values. The grid has six CTAs per row. The warp reduces the absolute maximum and broadcasts the reciprocal without CTA barriers. [Dispatch](../src/text_ops.cu#L665)<!-- prefix: "if (rows == " --> · [Warp kernel](../src/text_ops.cu#L211)<!-- prefix: "__global__ v" -->

The kernel writes row-major E4M3 values and 48 row-major FP32 scales per row. It floors the absolute maximum at `1e-10`, then divides by 448 in FP32 with round-to-nearest. It computes the reciprocal with FP32 round-to-nearest division, multiplies each value by it, and converts with saturated E4M3 rounding. The projection reads the same scale stride. [Scale and conversion](../include/sglang/text/quantize_warp.cuh#L28)<!-- prefix: "const float " --> · [Projection scale row](../src/prefill_fp8_sm100.cu#L322)<!-- prefix: "const auto* " -->

### Fused activation quantization

Packed decode and tensor-core prefill share the fused RMSNorm launcher. For ordinary small tiles, the eight-lane quantizer requires 2–32 launch rows, width 5,120, and a residual buffer. In packed decode, the bucket width sets the launch row count. Branch, residual, and gain buffers require 16-byte alignment. The E4M3 output requires 8-byte alignment. Eligible prefill tiles also use this path. Other valid calls use the existing fused kernel or separate RMSNorm and quantization according to width and input alignment. [Packed decode](https://github.com/para1lel/sglang.cpp/blob/93648269d852e0fee78f7af4d1c3ffe4897aa361/src/text/decoder_graph_packed.cpp#L270)<!-- prefix: "void PackedD" --> · [Prefill](../src/decoder_prefill.cpp#L271)<!-- prefix: "void normali" --> · [Selection](../src/text_normalization.cu#L477)<!-- prefix: "if (residual" -->

These 4,096-row groups can also use the eight-lane RMSNorm quantization epilogue before attention and MLP projections. It supports both residual modes with ordinary loads and stream ordering. The same shape and alignment requirements apply. Other partial groups, ragged tiles, two-grant groups, and singleton tiles retain ordinary preparation. [Layer RMS dispatch](../src/decoder_prefill.cpp#L668)<!-- prefix: "chunk.normal" --> · [4,096-row RMS guard](../src/text_normalization.cu#L467)<!-- prefix: "if (launch =" -->

Packed decode can use [programmatic launch](long-paged-decode.md#packed-rmsnorm-programmatic-launch) to start projection weight loads during fused RMSNorm.

The specialization keeps the FP32 RMS accumulation order and produces BF16 residual and normalized rows. It quantizes the rounded normalized row with one FP32 scale per 128 channels. Scale and reciprocal use FP32 round-to-nearest division. The conversion saturates to finite E4M3. [RMS arithmetic](../src/text_normalization.cu#L293)<!-- prefix: "float values" --> · [Quantizer](../src/text_normalization.cu#L220)<!-- prefix: "__device__ v" -->

## Constraints and failure behavior

The `sm_100a` build uses TMA, TMEM, and `tcgen05`. It requires capability 10.0, a matching image, sufficient opt-in shared memory, and tensor maps. Output slices must be disjoint from all inputs and each other. If the preferred separate tile is unavailable, the launcher tries the ordinary tile selected by shape. If the preferred packed choices are unavailable, the launcher tries the ordinary 128×128 tile. If packing is unavailable, the decoder projects each slice separately. Ineligible tensor-core projections use SIMT. CUDA errors propagate. [Build](../xmake.lua#L88)<!-- prefix: "target(\"sgla" --> · [Narrow packed launch](../src/prefill_fp8_sm100.cu#L707)<!-- prefix: "return launc" --> · [Separate projection](../src/decoder_prefill.cpp#L804)<!-- prefix: "void Prefill" -->

Attention output quantization requires a valid FP32 scale buffer with ordinary float alignment. Other valid shapes or alignments use the scalar kernel, which supports partial groups. The row launcher rejects null pointers, nonpositive widths, and row counts outside 1 through 65,535. The selected launch returns its CUDA status. Fused SwiGLU shares the warp kernel and retains its own dispatch and BF16 rounding. [Row launcher](../src/text_ops.cu#L659)<!-- prefix: "cudaError_t " --> · [SwiGLU dispatch](../src/text_ops.cu#L678)<!-- prefix: "cudaError_t " -->

Recorded terminal CUDA errors outrank nonterminal launch returns. The wide activation launchers consume matching error records and leave unrelated nonterminal records for their callers. [Launch status](../src/text_normalization.cu#L382)<!-- prefix: "cudaError_t " -->

## Verification

On B200, compare complete BF16 slices against the 128×128 control for both QKV checkpoint shapes. Include a ragged tile. Include a slice width with a 128-column remainder modulo 256. Measure projection latency separately with resident buffers.

For gate/up, compare against a 128×128 control with the same eight-row-tile grouping. The standalone change starts from a parent with wide QKV and ungrouped narrow gate/up. Check the BF16 slices, quantized activations and scales, and down output. Include row counts immediately below and above 1,024 to exercise the narrow fallback. Local B200 timings use checkpoint weights in isolated operators. Estimates use a historical 32-request decode trace.

For down, build the paired-mode `tests/text/fp8_prefill_tiling_check.cpp` from the acceptance change against both release libraries. Link with `-Wl,--wrap=cudaLaunchKernel -Wl,--wrap=cudaFuncGetAttributes`. Run both on a reserved B200 with the same checkpoint and a fresh reference directory.

```bash
"$control_check" "$checkpoint" wide-base "$reference_dir"
"$candidate_check" "$checkpoint" paired "$reference_dir"
```

Require bitwise BF16 parity and the 320×1 CTA grid for the paired shape. Check the large-row guard, unavailable-kernel fallback, and propagation of a CUDA launch error. Measure isolated graph replay with checkpoint weights and prepared SwiGLU activations. Keep operator latency and archive-based call exposure in the run record.

For the shared collective, hold grants and attention readers fixed. Cover grouped GQA and GDN output and MLP down for four full tiles and a 4,096-row partial group with unequal grants. Check MLP down on an ordinary 1,024-row tile. Compare normalized rows, committed target and draft state, and continuation according to [operator qualification](performance.md#operator-qualification). Exercise a clean setup decline separately from descriptor, scale preparation, and GEMM failures.

For attention output quantization, compare every E4M3 byte and FP32 scale with the scalar control in eager and graph replay. Check the warp grid at 1,024×6,144. Check the scalar grid for neighboring shapes, partial groups, and misaligned vector inputs or outputs. Include a scale buffer shifted by one float, which must retain the warp grid. Replay checkpoint GDN outputs to check bitwise parity through the output projection and measure isolated quantization time.

For fused RMSNorm parity, compare BF16 rows, E4M3 bytes, and FP32 scales bitwise in eager and graph replay. Exercise the packed buckets and shape and alignment fallbacks. The operator timing probe uses graph replay with checkpoint gains and fixed synthetic activation buffers. Use the [serving workloads](performance.md#native-concurrent-workloads) to measure request-level performance.

Use identical tiles and GQA readers in both arms. Test strict groups of four 1,024-row tiles and partial groups of eight 512-row or sixteen 256-row grants. Exercise staged and per-lane readers for strict groups. Exercise staged, raw grouped, and per-lane readers for partial groups. Compare packed RMSNorm with ordinary RMSNorm quantization, with and without a residual. Compare the fused GQA postgate with separate BF16 gating and FP8 quantization. Require bitwise parity for full BF16 state, E4M3 activations, and FP32 scales. Capture full BF16 vocabulary rows from both eligible generation arms on identical fixed token histories and compare them bitwise. Keep generated logprobs disabled to preserve the eligible target path. Separately compare generated IDs through continuation. Check that scored prompts, cached tiles, and other ineligible groups keep ordinary preparation. Exercise preflight fallback and verify that launch errors propagate.

### Wide verifier checks

On a reserved B200, compare 40 rows with two 20-row packets. Use M40 as the wide-dispatch control. Fix checkpoint, grants, and attention policy. Compare IDs, finish reasons, and drafted, accepted, and replayed counts. Trace projection, draft, GQA, GDN, and vocabulary routes. To isolate the epilogue, fix the split plan and compare BF16 output bits. Time warmed direct K4 decode at 16 and 32 lanes after 32K prefill.

Compare packed and 20-row submissions at fixed verification width. Require bitwise parity for BF16 normalized and residual output, E4M3 values, and FP32 scales. Cover RMSNorm with and without residual. Trace the compact GDN finish bypass. Check terminal status precedence and unrelated record preservation.

For depth-major scoring, fix grants, draft activations, and target verification. Compare full BF16 draft logits and mapped winners with four-lane scoring at widths 8, 16, and 32. Trace one shared head launch per active depth. Inject an invalid INT8 winner and check that drafting continues on a finite BF16 retry. If the retry has no finite winner, check that only the affected lane stops drafting. Compare committed state in both cases. Check packet scoring when depth setup declines. Require hard CUDA errors to propagate. [Packet fallback](../src/decoder_verify.cpp#L242)<!-- prefix: "bool MtpRoun" --> · [Depth loop](../src/decoder_verify.cpp#L254)<!-- prefix: "void MtpRoun" -->

Cover sparse slots, short drafts, partial packets, cancellation, and slot reuse. Check widths one through four, K0–K4, sampling, constraints, and tight step caps. Trace four- and eight-lane dispatch. Force compact setup decline and allocation OOM separately. Check smaller routes and matching outputs. Allow clean setup decline with only recorded `cudaErrorNotSupported`. Require explicit setup and launch errors to propagate. Check zero request bytes after retirement. Retain sources, binaries, inputs, and receipts.

### Held-out model quality

The held-out checkpoint study compared historical exact and Tensor Core
profiles against pinned SGLang on short answers, reasoning, typed JSON,
teacher-forced scores, and seeded sampling. Native Tensor Core and SGLang
answered 11 of 17 unconstrained cases correctly, compared with 10 of 17 for
native exact. Both native profiles matched their own K0 on the tested greedy
and seeded cases. The study leaves broader reasoning, long contexts, and
other sampling settings unmeasured. Its launch flags and default values
describe the measured historical Release. [Held-out results and source identity](https://github.com/para1lel/sglang.cpp/blob/4dd8c887c278c67744ed80e7feb2a825a92197a2/docs/fp8-prefill-projections.md#held-out-model-quality) · [Current launch migration](native-model-info.md#migrate-old-launch-commands)

An earlier ungrouped short-text probe also matched its own K0 and K4 but
diverged from exact. Diagnose differences on identical histories and measure
serving separately. [Archived short text observations](https://github.com/para1lel/sglang.cpp/blob/4dd8c887c278c67744ed80e7feb2a825a92197a2/docs/fp8-prefill-projections.md#historical-short-text-observations)
