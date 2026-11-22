# FP8 prefill projections

## Responsibilities and sources of truth

| Part | Responsibility | Source |
| --- | --- | --- |
| Decoder | Select the prefill policy and supply activation and output buffers. | [Packed call](../src/decoder_prefill.cpp#L247)<!-- prefix: "bool project" --> |
| Fused RMSNorm | Produce BF16 normalized values, E4M3 activations, and FP32 activation scales for prefill and packed decode. | [Launcher](../src/text_normalization.cu#L351)<!-- prefix: "cudaError_t " --> |
| Attention output quantizer | Convert BF16 GDN and GQA output rows for their FP8 output projections. | [Row launcher](../src/text_ops.cu#L629)<!-- prefix: "cudaError_t " --> |
| SM100 operator | Select the tile, manage CTA-local storage, and write BF16 outputs. | [Separate dispatch](../src/prefill_fp8_sm100.cu#L561)<!-- prefix: "if (rows > 1" --> · [Packed dispatch](../src/prefill_fp8_sm100.cu#L661)<!-- prefix: "if (rows < 1" --> |

## Runtime path

### Exact SIMT projections

For eligible wide inputs, the exact projection selects the eight-row launcher at 64 or more prompt rows and more than 1,024 output channels. With 5,120, 6,144, or 17,408 input channels and an output width at most 65,535, the launcher can assign four prompt rows and four adjacent output channels to each warp if the output width is divisible by four. For 64–128 prompt rows and an output width divisible by eight, it assigns eight adjacent channels per warp and reuses each decoded FP8 operand across the corresponding rows or channels. Every output retains its two ordered FP32 dot-product chains, scaling, warp reduction, and BF16 rounding. Other eligible shapes use the existing row tiles. [Row dispatch](../src/text_ops.cu#L732)<!-- prefix: "if (rows > 1" --> · [Tile selection](../include/sglang/text/prefill_fp8_exact.cuh#L201)<!-- prefix: "if constexpr" --> · [Operand reuse](../include/sglang/text/prefill_fp8_exact.cuh#L77)<!-- prefix: "__device__ _" -->

### Tensor-core projections

The server and generator default to `--fp8-prefill exact`. Three opt-in policies share token-ordered text attention between singleton decode and MTP verification. `tensor-core-mtp` selects packed Tensor Core prompt projections, `tensor-core-target` selects short target projections, and `tensor-core-mtp-target` selects both. These modes require prefix caching off. Draft seeding and draft decode retain their arithmetic. The score endpoint uses exact prefill. Image and video requests use the exact K0 language path, including visual embedding substitution, chunked prefill, and target decode. The vision encoder is shared across policies. The existing `tensor-core` policy keeps its separate prefill and prefix behavior.

Short target projections use the SM100 Swap-AB FP8 operator for one to five text rows. It reads the FP32 activation scales and BF16 checkpoint weight scales and writes BF16 results to the destination rows. Multi-row calls use a row-major E4M3 activation buffer for an eight-column TMA tile. With five verification rows, the fused quantizers write directly into that buffer and keep the same per-row FP32 scales. Other multi-row calls can pack the activation values before launching the projection. GDN QKV and Z, GQA QKV, and MLP gate and up share one packed activation per group and write to separate BF16 destinations. Scalar text decode uses the same split planner as verification. This grouping changes where the FP32 partials are added, so comparisons between K0 and K4 use the same projection policy and implementation. Unsupported projections and devices use exact target dispatch, including the staged five-row kernel where eligible. A per-lane guard keeps multimodal decode on its mRoPE-aware path, including in a mixed batch.

Under the short target policies, GDN verification projects the BF16 b and a gates for two to five rows in one launch. A CTA loads one gate weight row into shared memory and each warp computes it for one verification row. The kernel keeps the two ordered FP32 accumulation chains, warp reduction, and BF16 rounding. Scalar decode, exact verification, and ineligible inputs use the existing gate projections. [Verification dispatch](../src/decoder_forward.cpp#L241)<!-- prefix: "void project" --> · [Gate launcher](../src/text_decode_bandwidth.cu#L442)<!-- prefix: "std::optiona" -->

On B200 with Qwen3.8-27B-FP8, a single text request with 128 prompt IDs and 128 greedy output IDs produced the same sequence with the ungrouped combined policy at K0 and K4. That sequence first differed from exact at output token 20 and differed on 108 of 128 free-running IDs. A separate ungrouped combined K4 Nsight Systems run recorded 657 ms of verifier GPU activity, including 311 ms in FP8 projection kernels before activation packing. These measurements cover one short checkpoint-backed workload. The change in greedy IDs and the prefix reuse requirement keep `exact` as the default.

Separate tensor-core K0 projections require at least three rows, input widths divisible by 128, and contiguous, 16-byte-aligned E4M3 payloads. The output can end in a partial tile. At 129–1,023 rows, projections with at least 8,192 output channels or 6,144 input channels first try a paired-TMA 128×128 tile. The launch places the row tiles for each column tile next to one another to improve weight cache reuse. [Eligibility](../src/prefill_fp8_sm100.cu#L542)<!-- prefix: "const int ma" --> · [Separate projection tiles](../src/prefill_fp8_sm100.cu#L561)<!-- prefix: "if (rows > 1" -->

Packed projections also require contiguous weight and scale slices and each output slice width to be divisible by 128. For 1,024 rows and input width 5,120, eligible packed GDN QKV/Z (width 16,384) and GQA QKV (width 14,336) first try a 128×256 tile. The wide tile requires each output slice width to be divisible by 256. The CTA owns four TMA stages and its TMEM. Each of eight consumer warps handles 32 rows in one 128-column weight-scale block and reads 16 TMEM columns at a time. The epilogue writes the existing BF16 slices in the decoder workspace. [GDN call](../src/decoder_prefill.cpp#L281)<!-- prefix: "if (!project" --> · [GQA call](../src/decoder_prefill.cpp#L338)<!-- prefix: "if (!project" --> · [Scale block](../src/prefill_fp8_sm100.cu#L323)<!-- prefix: "const int sc" --> · [Output slice](../src/prefill_fp8_sm100.cu#L347)<!-- prefix: "const auto [" -->

At 1,024 rows and input width 5,120, the wide tile also supports eligible MLP gate/up packs with exactly two 17,408-column slices. The MLP launch groups eight row tiles per column tile. The GDN and GQA launches use one row tile per group. The MLP writes BF16 gate/up slices for the separate SwiGLU quantizer, which supplies the down projection. [MLP call](../src/decoder_prefill.cpp#L383)<!-- prefix: "if (!project" --> · [MLP slice check](../src/prefill_fp8_sm100.cu#L640)<!-- prefix: "const bool p" --> · [Row grouping](../src/prefill_fp8_sm100.cu#L656)<!-- prefix: "packed_mlp_c" --> · [SwiGLU quantization](../src/decoder_prefill.cpp#L391)<!-- prefix: "check(tensor" -->

At exactly 768 rows and input width 5,120, an eligible MLP gate/up pack first tries a 128×128 tile with three TMA stages and four consumer warps. The tile uses 256 TMEM columns and reads 16 columns at a time. If the tile is unavailable, dispatch continues to the existing packed fallbacks. At 256–1,023 rows, other eligible packs try paired TMA with a 128×256 tile when every slice width is divisible by 256, or a 128×128 tile otherwise. The launches group the row tiles for each column tile and keep the existing scale types, ordered FP32 K-group accumulation, and BF16 rounding. [Compact MLP dispatch](../src/prefill_fp8_sm100.cu#L643)<!-- prefix: "if ((rows ==" --> · [Paired packed dispatch](../src/prefill_fp8_sm100.cu#L661)<!-- prefix: "if (rows < 1" -->

[Wide projection groups](native-prefill-workspace.md#multi-request-text-prefill) extend compact gate/up and eligible paired TMA tiles to 2,048 rows. These tiles group row tiles for each weight tile. For a 1,024-row wide gate/up group, dispatch tries the compact tile before the existing 1,024-row choices. [Group flag](../src/decoder_prefill.cpp#L607)<!-- prefix: "const bool g" -->

The separate down projection tries paired TMA at exactly 1,024 rows, 17,408 input channels, and 5,120 output channels when large-row prefill is enabled. Its 128×128 CTA tile uses six shared-memory stages in three pairs. The producer issues four input and weight copies per pair with one TMA completion barrier. The MMA warp waits once per pair and releases both stages after it completes the second group's matrix operations. The launch groups eight row tiles per weight tile for cache reuse. It keeps the checkpoint scale types, FP32 K-group accumulation order, and BF16 rounding. [Down dispatch](../src/prefill_fp8_sm100.cu#L553)<!-- prefix: "if (rows == " --> · [Producer](../src/prefill_fp8_sm100.cu#L88)<!-- prefix: "__device__ v" --> · [Release](../src/prefill_fp8_sm100.cu#L158)<!-- prefix: "commit_mma(&" -->

### Attention output quantization

For ordinary tensor-core text prefill, GDN combines gated RMSNorm with FP8 quantization. Each warp rounds one head to BF16 in registers before quantization. The output projection consumes the E4M3 values and FP32 scales directly. In grouped prefill, each grant writes its own rows in the shared activation buffer. Gated GQA and the BF16 GDN prefill path pass 6,144 channels per row to `text_quantize_fp8_rows`. [Fused finish](../src/text_attention_gdn.cu#L583)<!-- prefix: "__global__ v" --> · [GDN call](../src/decoder_prefill.cpp#L870)<!-- prefix: "chunk.gdn_at" --> · [GQA call](../src/decoder_prefill.cpp#L866)<!-- prefix: "chunk.attent" -->

For exactly 1,024 rows and 6,144 channels, the launcher uses the warp kernel if the BF16 input and E4M3 output are both 4-byte aligned. A 256-thread CTA handles eight 128-channel groups in one row. Each warp owns one group and each lane processes four values. The grid has six CTAs per row. The warp reduces the absolute maximum and broadcasts the reciprocal without CTA barriers. [Dispatch](../src/text_ops.cu#L635)<!-- prefix: "if (rows == " --> · [Warp kernel](../src/text_ops.cu#L208)<!-- prefix: "__global__ v" -->

The kernel writes row-major E4M3 values and 48 row-major FP32 scales per row. It floors the absolute maximum at `1e-10`, then divides by 448 in FP32 with round-to-nearest. It computes the reciprocal with FP32 round-to-nearest division, multiplies each value by it, and converts with saturated E4M3 rounding. The projection reads the same scale stride. [Scale and conversion](../include/sglang/text/quantize_warp.cuh#L46)<!-- prefix: "const float " --> · [Projection scale row](../src/prefill_fp8_sm100.cu#L321)<!-- prefix: "const auto* " -->

### Fused activation quantization

Packed decode and tensor-core prefill share the fused RMSNorm launcher. The eight-lane quantizer requires 2–32 launch rows, width 5,120, and a residual buffer. In packed decode, the bucket width sets the launch row count. Branch, residual, and gain buffers require 16-byte alignment. The E4M3 output requires 8-byte alignment. Eligible prefill tiles also use this path. Other valid calls use the existing fused kernel or separate RMSNorm and quantization according to width and input alignment. [Packed decode](../src/text/decoder_graph_packed.cpp#L270)<!-- prefix: "void PackedD" --> · [Prefill](../src/decoder_prefill.cpp#L402)<!-- prefix: "void normali" --> · [Selection](../src/text_normalization.cu#L365)<!-- prefix: "if (residual" -->

Packed decode can use [programmatic launch](long-paged-decode.md#packed-rmsnorm-programmatic-launch) to start projection weight loads during fused RMSNorm.

The specialization keeps the FP32 RMS accumulation order and produces BF16 residual and normalized rows. It quantizes the rounded normalized row with one FP32 scale per 128 channels. Scale and reciprocal use FP32 round-to-nearest division. The conversion saturates to finite E4M3. [RMS arithmetic](../src/text_normalization.cu#L254)<!-- prefix: "float values" --> · [Quantizer](../src/text_normalization.cu#L182)<!-- prefix: "__device__ v" -->

## Constraints and failure behavior

The `sm_100a` build uses TMA, TMEM, and `tcgen05`. It requires capability 10.0, a matching image, sufficient opt-in shared memory, and tensor maps. Output slices must be disjoint from all inputs and each other. If the preferred separate tile is unavailable, the launcher tries the ordinary tile selected by shape. If the preferred packed choices are unavailable, the launcher tries the ordinary 128×128 tile. If packing is unavailable, the decoder projects each slice separately. Ineligible tensor-core projections use SIMT. CUDA errors propagate. [Build](../xmake.lua#L68)<!-- prefix: "target(\"sgla" --> · [Narrow packed launch](../src/prefill_fp8_sm100.cu#L674)<!-- prefix: "return launc" --> · [Separate projection](../src/decoder_prefill.cpp#L663)<!-- prefix: "void Prefill" -->

Attention output quantization requires a valid FP32 scale buffer with ordinary float alignment. Other valid shapes or alignments use the scalar kernel, which supports partial groups. The row launcher rejects null pointers, nonpositive widths, and row counts outside 1 through 65,535. The selected launch returns its CUDA status. Fused SwiGLU shares the warp kernel and retains its own dispatch and BF16 rounding. [Row launcher](../src/text_ops.cu#L629)<!-- prefix: "cudaError_t " --> · [SwiGLU dispatch](../src/text_ops.cu#L648)<!-- prefix: "cudaError_t " -->

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

### Held-out model quality

The native study exercised `exact` and explicit `tensor-core-mtp-target` at K0 and K4. It used the Qwen3.8-27B-FP8 checkpoint on one B200, with graphs enabled, prefix reuse off, 32 slots, and an 8,192-token step cap. It compared the shared text and JSON-schema cases with full-checkpoint SGLang `f04d886bf79aa3d6237cbb12a3311838eb9f9bd8`. SGLang used deterministic eager execution, Triton kernels, FP32 recurrent state, and no speculation.

| Check | Native exact | Native tensor-core target | SGLang |
| --- | ---: | ---: | ---: |
| Correct unconstrained visible answers | 10/17 | 11/17 | 11/17 |
| Correct visible answers with thinking enabled | 4/4 | 4/4 | 4/4 |
| Correct typed JSON under a schema | 2/2 | 2/2 | 2/2 |
| Mean teacher NLL at the first answer token (17 rows, nats) | 1.23333 | 1.14192 | 1.09220 |
| Mean teacher NLL during decode (64 rows, nats) | 0.25424 | 0.22718 | 0.23879 |

Both native K4 policies matched their own K0 on 29 greedy cases and nine seeded cases. The seeded cases used temperature 0.85, top-k 32, and top-p 1. CPU replay matched 144 draws per native K0 policy and 144 SGLang draws under their respective sampler rules. Direct kernel traces confirmed tensor-core dispatch for native constrained generation, chat with a schema, thinking chat, and K0 generated logprobs.

The 17 authored prompts span language, math, code, and five non-English languages. The base prompts used 26–67 input IDs, and the four thinking variants used 70–93. Fixed-history comparisons used one canonical answer per prompt and included terminal IDs. The study graded visible answers and counted latent reasoning IDs separately. It leaves broader reasoning quality, long contexts, and other sampling settings unmeasured.

The measured native tree `c6ef0daf79e2f4f4a495d338dc96bfb81f5bc43d` is the tree of `904e1a12ea45d1696128b6b36160367c9a4f85f7`. Its target forward and prefill sources match publication `99c523d95c47788f6c872c6c5428c28418d395eb`. The study predates the optional INT8 draft head and OpenAI Responses. The server default remains `exact`, with the INT8 draft head opt-in. [Target selection](../src/decoder_forward.cpp#L743)<!-- prefix: "is_b200() &&" --> · [FP8 default](../apps/sglang-server.cpp#L32)<!-- prefix: "sglang::Fp8P" --> · [INT8 default](../apps/sglang-server.cpp#L42)<!-- prefix: "bool mtp_int" -->
