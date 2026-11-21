# FP8 prefill projections

## Responsibilities and sources of truth

| Part | Responsibility | Source |
| --- | --- | --- |
| Decoder | Select the prefill policy and supply activation and output buffers. | [Packed call](../src/decoder_prefill.cpp#L220)<!-- prefix: "bool project" --> |
| Fused RMSNorm | Produce BF16 normalized values, E4M3 activations, and FP32 activation scales for prefill and packed decode. | [Launcher](../src/text_normalization.cu#L309)<!-- prefix: "cudaError_t " --> |
| SM100 operator | Select the tile, manage CTA-local storage, and write BF16 outputs. | [Down dispatch](../src/prefill_fp8_sm100.cu#L551)<!-- prefix: "if (rows == " --> · [Packed dispatch](../src/prefill_fp8_sm100.cu#L631)<!-- prefix: "if (rows == " --> |

## Runtime path

Separate tensor-core K0 projections require at least three rows, input widths divisible by 128, and contiguous, 16-byte-aligned E4M3 payloads. The output can end in a partial tile. At input widths of at least 8,192, eligible separate projections use 64×128 tiles below 1,024 rows and 128×128 otherwise. [Eligibility](../src/prefill_fp8_sm100.cu#L540)<!-- prefix: "const int ma" --> · [Separate projection tiles](../src/prefill_fp8_sm100.cu#L558)<!-- prefix: "const bool n" -->

Packed projections also require contiguous weight and scale slices and each output slice width to be divisible by 128. For 1,024 rows and input width 5,120, eligible packed GDN QKV/Z (width 16,384) and GQA QKV (width 14,336) first try a 128×256 tile. The wide tile requires each output slice width to be divisible by 256. The CTA owns four TMA stages and its TMEM. Each of eight consumer warps handles 32 rows in one 128-column weight-scale block and reads 16 TMEM columns at a time. The epilogue writes the existing BF16 slices in the decoder workspace. [GDN call](../src/decoder_prefill.cpp#L249)<!-- prefix: "if (!project" --> · [GQA call](../src/decoder_prefill.cpp#L276)<!-- prefix: "if (!project" --> · [Scale block](../src/prefill_fp8_sm100.cu#L321)<!-- prefix: "const int sc" --> · [Output slice](../src/prefill_fp8_sm100.cu#L345)<!-- prefix: "const auto [" -->

At 1,024 rows and input width 5,120, the wide tile also supports eligible MLP gate/up packs with exactly two 17,408-column slices. The MLP launch groups eight row tiles per column tile. The GDN and GQA launches use one row tile per group. The MLP writes BF16 gate/up slices for the separate SwiGLU quantizer, which supplies the down projection. [MLP call](../src/decoder_prefill.cpp#L306)<!-- prefix: "if (!project" --> · [MLP slice check](../src/prefill_fp8_sm100.cu#L628)<!-- prefix: "const bool p" --> · [Row grouping](../src/prefill_fp8_sm100.cu#L635)<!-- prefix: "packed_mlp_c" --> · [SwiGLU quantization](../src/decoder_prefill.cpp#L314)<!-- prefix: "check(tensor" -->

The separate down projection tries paired TMA at exactly 1,024 rows, 17,408 input channels, and 5,120 output channels when large-row prefill is enabled. Its 128×128 CTA tile uses six shared-memory stages in three pairs. The producer issues four input and weight copies per pair with one TMA completion barrier. The MMA warp waits once per pair and releases both stages after it completes the second group's matrix operations. The launch groups eight row tiles per weight tile for cache reuse. It keeps the checkpoint scale types, FP32 K-group accumulation order, and BF16 rounding. [Down dispatch](../src/prefill_fp8_sm100.cu#L551)<!-- prefix: "if (rows == " --> · [Producer](../src/prefill_fp8_sm100.cu#L86)<!-- prefix: "__device__ v" --> · [Release](../src/prefill_fp8_sm100.cu#L156)<!-- prefix: "commit_mma(&" -->

### Fused activation quantization

Packed decode and tensor-core prefill share the fused RMSNorm launcher. The eight-lane quantizer requires 2–32 launch rows, width 5,120, and a residual buffer. In packed decode, the bucket width sets the launch row count. Branch, residual, and gain buffers require 16-byte alignment. The E4M3 output requires 8-byte alignment. Eligible prefill tiles also use this path. Other valid calls use the existing fused kernel or separate RMSNorm and quantization according to width and input alignment. [Packed decode](../src/text/decoder_graph_packed.cpp#L268)<!-- prefix: "void PackedD" --> · [Prefill](../src/decoder_prefill.cpp#L325)<!-- prefix: "void normali" --> · [Selection](../src/text_normalization.cu#L322)<!-- prefix: "if (residual" -->

The specialization keeps the FP32 RMS accumulation order and produces BF16 residual and normalized rows. It quantizes the rounded normalized row with one FP32 scale per 128 channels. Scale and reciprocal use FP32 round-to-nearest division. The conversion saturates to finite E4M3. [RMS arithmetic](../src/text_normalization.cu#L254)<!-- prefix: "float values" --> · [Quantizer](../src/text_normalization.cu#L181)<!-- prefix: "__device__ v" -->

## Constraints and failure behavior

The `sm_100a` build uses TMA, TMEM, and `tcgen05`. It requires capability 10.0, a matching image, sufficient opt-in shared memory, and tensor maps. Output slices must be disjoint from all inputs and each other. If the paired down kernel is unavailable, the launcher tries the ordinary 128×128 kernel. Other eligible packs use 128×128. If the wide launch is unavailable, the launcher tries 128×128. If packing is unavailable, the decoder projects each slice separately. Ineligible tensor-core projections use SIMT. CUDA errors propagate. The wide tile preserves E4M3 operands, FP32 activation scales, BF16 inverse weight scales, K-group accumulation order, and BF16 rounding. [Build](../xmake.lua#L67)<!-- prefix: "target(\"sgla" --> · [Narrow packed launch](../src/prefill_fp8_sm100.cu#L640)<!-- prefix: "return launc" --> · [Separate projection](../src/decoder_prefill.cpp#L346)<!-- prefix: "void Prefill" -->

## Verification

On B200, compare complete BF16 slices against the 128×128 control for both QKV checkpoint shapes. Include a ragged tile. Include a slice width with a 128-column remainder modulo 256. Measure projection latency separately with resident buffers.

For gate/up, compare against a 128×128 control with the same eight-row-tile grouping. The standalone change starts from a parent with wide QKV and ungrouped narrow gate/up. Check the BF16 slices, quantized activations and scales, and down output. Include row counts immediately below and above 1,024 to exercise the narrow fallback. Local B200 timings use checkpoint weights in isolated operators. W5 savings are estimates from an archived trace.

For down, build the paired-mode `tests/text/fp8_prefill_tiling_check.cpp` from the acceptance change against both release libraries. Link with `-Wl,--wrap=cudaLaunchKernel -Wl,--wrap=cudaFuncGetAttributes`. Run both on a reserved B200 with the same checkpoint and a fresh reference directory.

```bash
"$control_check" "$checkpoint" wide-base "$reference_dir"
"$candidate_check" "$checkpoint" paired "$reference_dir"
```

Require bitwise BF16 parity and the 320×1 CTA grid for the paired shape. Check the large-row guard, unavailable-kernel fallback, and propagation of a CUDA launch error. Measure isolated graph replay with checkpoint weights and prepared SwiGLU activations. Keep operator latency and archive-based call exposure in the run record.

For fused RMSNorm parity, compare BF16 rows, E4M3 bytes, and FP32 scales bitwise in eager and graph replay. Exercise the packed buckets and shape and alignment fallbacks. The operator timing probe uses graph replay with checkpoint gains and fixed synthetic activation buffers. Use the [serving workloads](performance.md#native-concurrent-workloads) to measure request-level performance.
