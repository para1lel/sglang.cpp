# FP8 prefill projections

## Responsibilities and sources of truth

| Part | Responsibility | Source |
| --- | --- | --- |
| Decoder | Select the prefill policy and supply activation and output buffers. | [Packed call](../src/decoder_prefill.cpp#L247)<!-- prefix: "bool project" --> |
| Fused RMSNorm | Produce BF16 normalized values, E4M3 activations, and FP32 activation scales for prefill and packed decode. | [Launcher](../src/text_normalization.cu#L351)<!-- prefix: "cudaError_t " --> |
| Attention output quantizer | Convert BF16 GDN and GQA output rows for their FP8 output projections. | [Row launcher](../src/text_ops.cu#L612)<!-- prefix: "cudaError_t " --> |
| SM100 operator | Select the tile, manage CTA-local storage, and write BF16 outputs. | [Separate dispatch](../src/prefill_fp8_sm100.cu#L561)<!-- prefix: "if (rows > 1" --> · [Packed dispatch](../src/prefill_fp8_sm100.cu#L661)<!-- prefix: "if (rows \u003c 1" --> |

## Runtime path

### Exact SIMT projections

For eligible wide inputs, the exact projection selects the eight-row launcher at 64 or more prompt rows and more than 1,024 output channels. The launcher can assign up to four prompt rows and four adjacent output channels to each warp. This tile requires 5,120, 6,144, or 17,408 input channels. It requires an output width divisible by four and at most 65,535. It shares activation values, activation scales, and weight scales. Each result keeps its own ordered FP32 sum and BF16 rounding. Other eligible shapes keep the existing row tiles. [Row dispatch](../src/text_ops.cu#L716)<!-- prefix: "if (rows \u003e 1" --> · [Tile selection](../include/sglang/text/prefill_fp8_exact.cuh#L77)<!-- prefix: "if constexpr" --> · [Activation loads](../include/sglang/text/prefill_fp8_exact.cuh#L43)<!-- prefix: "const auto r" -->

### Tensor-core projections

Separate tensor-core K0 projections require at least three rows, input widths divisible by 128, and contiguous, 16-byte-aligned E4M3 payloads. The output can end in a partial tile. At 129–1,023 rows, projections with at least 8,192 output channels or 6,144 input channels first try a paired-TMA 128×128 tile. The launch places the row tiles for each column tile next to one another to improve weight cache reuse. [Eligibility](../src/prefill_fp8_sm100.cu#L542)<!-- prefix: "const int ma" --> · [Separate projection tiles](../src/prefill_fp8_sm100.cu#L561)<!-- prefix: "if (rows > 1" -->

Packed projections also require contiguous weight and scale slices and each output slice width to be divisible by 128. For 1,024 rows and input width 5,120, eligible packed GDN QKV/Z (width 16,384) and GQA QKV (width 14,336) first try a 128×256 tile. The wide tile requires each output slice width to be divisible by 256. The CTA owns four TMA stages and its TMEM. Each of eight consumer warps handles 32 rows in one 128-column weight-scale block and reads 16 TMEM columns at a time. The epilogue writes the existing BF16 slices in the decoder workspace. [GDN call](../src/decoder_prefill.cpp#L276)<!-- prefix: "if (!project" --> · [GQA call](../src/decoder_prefill.cpp#L333)<!-- prefix: "if (!project" --> · [Scale block](../src/prefill_fp8_sm100.cu#L323)<!-- prefix: "const int sc" --> · [Output slice](../src/prefill_fp8_sm100.cu#L347)<!-- prefix: "const auto [" -->

At 1,024 rows and input width 5,120, the wide tile also supports eligible MLP gate/up packs with exactly two 17,408-column slices. The MLP launch groups eight row tiles per column tile. The GDN and GQA launches use one row tile per group. The MLP writes BF16 gate/up slices for the separate SwiGLU quantizer, which supplies the down projection. [MLP call](../src/decoder_prefill.cpp#L378)<!-- prefix: "if (!project" --> · [MLP slice check](../src/prefill_fp8_sm100.cu#L640)<!-- prefix: "const bool p" --> · [Row grouping](../src/prefill_fp8_sm100.cu#L656)<!-- prefix: "packed_mlp_c" --> · [SwiGLU quantization](../src/decoder_prefill.cpp#L386)<!-- prefix: "check(tensor" -->

At exactly 768 rows and input width 5,120, an eligible MLP gate/up pack first tries a 128×128 tile with three TMA stages and four consumer warps. The tile uses 256 TMEM columns and reads 16 columns at a time. If the tile is unavailable, dispatch continues to the existing packed fallbacks. At 256–1,023 rows, other eligible packs try paired TMA with a 128×256 tile when every slice width is divisible by 256, or a 128×128 tile otherwise. The launches group the row tiles for each column tile and keep the existing scale types, ordered FP32 K-group accumulation, and BF16 rounding. [Compact MLP dispatch](../src/prefill_fp8_sm100.cu#L643)<!-- prefix: "if ((rows ==" --> · [Paired packed dispatch](../src/prefill_fp8_sm100.cu#L661)<!-- prefix: "if (rows \u003c 1" -->

[Wide projection groups](native-prefill-workspace.md#multi-request-text-prefill) extend compact gate/up and eligible paired TMA tiles to 2,048 rows. These tiles group row tiles for each weight tile. For a 1,024-row wide gate/up group, dispatch tries the compact tile before the existing 1,024-row choices. [Group flag](../src/decoder_prefill.cpp#L602)<!-- prefix: "const bool g" -->

The separate down projection tries paired TMA at exactly 1,024 rows, 17,408 input channels, and 5,120 output channels when large-row prefill is enabled. Its 128×128 CTA tile uses six shared-memory stages in three pairs. The producer issues four input and weight copies per pair with one TMA completion barrier. The MMA warp waits once per pair and releases both stages after it completes the second group's matrix operations. The launch groups eight row tiles per weight tile for cache reuse. It keeps the checkpoint scale types, FP32 K-group accumulation order, and BF16 rounding. [Down dispatch](../src/prefill_fp8_sm100.cu#L553)<!-- prefix: "if (rows == " --> · [Producer](../src/prefill_fp8_sm100.cu#L88)<!-- prefix: "__device__ v" --> · [Release](../src/prefill_fp8_sm100.cu#L158)<!-- prefix: "commit_mma(&" -->

### Attention output quantization

For ordinary tensor-core text prefill, GDN combines gated RMSNorm with FP8 quantization. Each warp rounds one head to BF16 in registers before quantization. The output projection consumes the E4M3 values and FP32 scales directly. In grouped prefill, each grant writes its own rows in the shared activation buffer. Gated GQA and the BF16 GDN prefill path pass 6,144 channels per row to `text_quantize_fp8_rows`. [Fused finish](../src/text_attention_gdn.cu#L583)<!-- prefix: "__global__ v" --> · [GDN call](../src/decoder_prefill.cpp#L864)<!-- prefix: "chunk.gdn_at" --> · [GQA call](../src/decoder_prefill.cpp#L860)<!-- prefix: "chunk.attent" -->

For exactly 1,024 rows and 6,144 channels, the launcher uses the warp kernel if the BF16 input and E4M3 output are both 4-byte aligned. A 256-thread CTA handles eight 128-channel groups in one row. Each warp owns one group and each lane processes four values. The grid has six CTAs per row. The warp reduces the absolute maximum and broadcasts the reciprocal without CTA barriers. [Dispatch](../src/text_ops.cu#L618)<!-- prefix: "if (rows == " --> · [Warp kernel](../src/text_ops.cu#L207)<!-- prefix: "__global__ v" -->

The kernel writes row-major E4M3 values and 48 row-major FP32 scales per row. It floors the absolute maximum at `1e-10`, then divides by 448 in FP32 with round-to-nearest. It computes the reciprocal with FP32 round-to-nearest division, multiplies each value by it, and converts with saturated E4M3 rounding. The projection reads the same scale stride. [Scale and conversion](../include/sglang/text/quantize_warp.cuh#L46)<!-- prefix: "const float " --> · [Projection scale row](../src/prefill_fp8_sm100.cu#L321)<!-- prefix: "const auto* " -->

### Fused activation quantization

Packed decode and tensor-core prefill share the fused RMSNorm launcher. The eight-lane quantizer requires 2–32 launch rows, width 5,120, and a residual buffer. In packed decode, the bucket width sets the launch row count. Branch, residual, and gain buffers require 16-byte alignment. The E4M3 output requires 8-byte alignment. Eligible prefill tiles also use this path. Other valid calls use the existing fused kernel or separate RMSNorm and quantization according to width and input alignment. [Packed decode](../src/text/decoder_graph_packed.cpp#L270)<!-- prefix: "void PackedD" --> · [Prefill](../src/decoder_prefill.cpp#L397)<!-- prefix: "void normali" --> · [Selection](../src/text_normalization.cu#L365)<!-- prefix: "if (residual" -->

Packed decode can use [programmatic launch](long-paged-decode.md#packed-rmsnorm-programmatic-launch) to start projection weight loads during fused RMSNorm.

The specialization keeps the FP32 RMS accumulation order and produces BF16 residual and normalized rows. It quantizes the rounded normalized row with one FP32 scale per 128 channels. Scale and reciprocal use FP32 round-to-nearest division. The conversion saturates to finite E4M3. [RMS arithmetic](../src/text_normalization.cu#L254)<!-- prefix: "float values" --> · [Quantizer](../src/text_normalization.cu#L182)<!-- prefix: "__device__ v" -->

## Constraints and failure behavior

The `sm_100a` build uses TMA, TMEM, and `tcgen05`. It requires capability 10.0, a matching image, sufficient opt-in shared memory, and tensor maps. Output slices must be disjoint from all inputs and each other. If the preferred separate tile is unavailable, the launcher tries the ordinary tile selected by shape. If the preferred packed choices are unavailable, the launcher tries the ordinary 128×128 tile. If packing is unavailable, the decoder projects each slice separately. Ineligible tensor-core projections use SIMT. CUDA errors propagate. [Build](../xmake.lua#L67)<!-- prefix: "target(\"sgla" --> · [Narrow packed launch](../src/prefill_fp8_sm100.cu#L674)<!-- prefix: "return launc" --> · [Separate projection](../src/decoder_prefill.cpp#L658)<!-- prefix: "void Prefill" -->

Attention output quantization requires a valid FP32 scale buffer with ordinary float alignment. Other valid shapes or alignments use the scalar kernel, which supports partial groups. The row launcher rejects null pointers, nonpositive widths, and row counts outside 1 through 65,535. The selected launch returns its CUDA status. Fused SwiGLU shares the warp kernel and retains its own dispatch and BF16 rounding. [Row launcher](../src/text_ops.cu#L612)<!-- prefix: "cudaError_t " --> · [SwiGLU dispatch](../src/text_ops.cu#L631)<!-- prefix: "cudaError_t " -->

## Verification

On B200, compare complete BF16 slices against the 128×128 control for both QKV checkpoint shapes. Include a ragged tile. Include a slice width with a 128-column remainder modulo 256. Measure projection latency separately with resident buffers.

For gate/up, compare against a 128×128 control with the same eight-row-tile grouping. The standalone change starts from a parent with wide QKV and ungrouped narrow gate/up. Check the BF16 slices, quantized activations and scales, and down output. Include row counts immediately below and above 1,024 to exercise the narrow fallback. Local B200 timings use checkpoint weights in isolated operators. W5 savings are estimates from an archived trace.

For down, build the paired-mode `tests/text/fp8_prefill_tiling_check.cpp` from the acceptance change against both release libraries. Link with `-Wl,--wrap=cudaLaunchKernel -Wl,--wrap=cudaFuncGetAttributes`. Run both on a reserved B200 with the same checkpoint and a fresh reference directory.

```bash
"$control_check" "$checkpoint" wide-base "$reference_dir"
"$candidate_check" "$checkpoint" paired "$reference_dir"
```

Require bitwise BF16 parity and the 320×1 CTA grid for the paired shape. Check the large-row guard, unavailable-kernel fallback, and propagation of a CUDA launch error. Measure isolated graph replay with checkpoint weights and prepared SwiGLU activations. Keep operator latency and archive-based call exposure in the run record.

For attention output quantization, compare every E4M3 byte and FP32 scale with the scalar control in eager and graph replay. Check the warp grid at 1,024×6,144. Check the scalar grid for neighboring shapes, partial groups, and misaligned vector inputs or outputs. Include a scale buffer shifted by one float, which must retain the warp grid. Replay checkpoint GDN outputs to check bitwise parity through the output projection and measure isolated quantization time.

For fused RMSNorm parity, compare BF16 rows, E4M3 bytes, and FP32 scales bitwise in eager and graph replay. Exercise the packed buckets and shape and alignment fallbacks. The operator timing probe uses graph replay with checkpoint gains and fixed synthetic activation buffers. Use the [serving workloads](performance.md#native-concurrent-workloads) to measure request-level performance.
