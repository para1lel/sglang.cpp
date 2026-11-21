# FP8 prefill projections

## Responsibilities and sources of truth

| Part | Responsibility | Source |
| --- | --- | --- |
| Decoder | Select the prefill policy and supply activation and output buffers. | [Packed call](../src/decoder_prefill.cpp#L220)<!-- prefix: "bool project" --> |
| SM100 operator | Select the tile, manage CTA-local storage, and write BF16 outputs. | [Packed dispatch](../src/prefill_fp8_sm100.cu#L610)<!-- prefix: "if (rows == " --> |

## Runtime path

Separate tensor-core K0 projections require at least three rows, input widths divisible by 128, and contiguous, 16-byte-aligned E4M3 payloads. The output can end in a partial tile. At input widths of at least 8,192, eligible separate projections use 64×128 tiles below 1,024 rows and 128×128 otherwise. [Eligibility](../src/prefill_fp8_sm100.cu#L526)<!-- prefix: "const int ma" --> · [Separate projection tiles](../src/prefill_fp8_sm100.cu#L537)<!-- prefix: "const bool n" -->

Packed projections also require contiguous weight and scale slices and each output slice width to be divisible by 128. For 1,024 rows and input width 5,120, eligible packed GDN QKV/Z (width 16,384) and GQA QKV (width 14,336) first try a 128×256 tile. The wide tile requires each output slice width to be divisible by 256. The CTA owns four TMA stages and its TMEM. Each of eight consumer warps handles 32 rows in one 128-column weight-scale block and reads 16 TMEM columns at a time. The epilogue writes the existing BF16 slices in the decoder workspace. [GDN call](../src/decoder_prefill.cpp#L249)<!-- prefix: "if (!project" --> · [GQA call](../src/decoder_prefill.cpp#L276)<!-- prefix: "if (!project" --> · [Scale block](../src/prefill_fp8_sm100.cu#L307)<!-- prefix: "const int sc" --> · [Output slice](../src/prefill_fp8_sm100.cu#L331)<!-- prefix: "const auto [" -->

At 1,024 rows and input width 5,120, the wide tile also supports eligible MLP gate/up packs with exactly two 17,408-column slices. The MLP launch groups eight row tiles per column tile. The GDN and GQA launches use one row tile per group. The MLP writes BF16 gate/up slices for the separate SwiGLU quantizer, which supplies the down projection. [MLP call](../src/decoder_prefill.cpp#L306)<!-- prefix: "if (!project" --> · [MLP slice check](../src/prefill_fp8_sm100.cu#L607)<!-- prefix: "const bool p" --> · [Row grouping](../src/prefill_fp8_sm100.cu#L614)<!-- prefix: "packed_mlp_c" --> · [SwiGLU quantization](../src/decoder_prefill.cpp#L314)<!-- prefix: "check(tensor" -->

## Constraints and failure behavior

The `sm_100a` build uses TMA, TMEM, and `tcgen05`. It requires capability 10.0, a matching image, sufficient opt-in shared memory, and tensor maps. Output slices must be disjoint from all inputs and each other. Other eligible packs use 128×128. If the wide launch is unavailable, the launcher tries 128×128. If packing is unavailable, the decoder projects each slice separately. Ineligible tensor-core projections use SIMT. CUDA errors propagate. The wide tile preserves E4M3 operands, FP32 activation scales, BF16 inverse weight scales, K-group accumulation order, and BF16 rounding. [Build](../xmake.lua#L67)<!-- prefix: "target(\"sgla" --> · [Narrow packed launch](../src/prefill_fp8_sm100.cu#L619)<!-- prefix: "return launc" --> · [Separate projection](../src/decoder_prefill.cpp#L346)<!-- prefix: "void Prefill" -->

## Verification

On B200, compare complete BF16 slices against the 128×128 control for both QKV checkpoint shapes. Include a ragged tile. Include a slice width with a 128-column remainder modulo 256. Measure projection latency separately with resident buffers.

For gate/up, compare against a 128×128 control with the same eight-row-tile grouping. The standalone change starts from a parent with wide QKV and ungrouped narrow gate/up. Check the BF16 slices, quantized activations and scales, and down output. Include row counts immediately below and above 1,024 to exercise the narrow fallback. Local B200 timings use checkpoint weights in isolated operators. W5 savings are estimates from an archived trace.
