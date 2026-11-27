# FP8 prefill projections

## Responsibilities and sources of truth

The decoder selects precision and buffers. Fused RMSNorm and attention quantizers provide packed activations and scales. The SM100 operator selects and runs the BF16-output tile. [Packed call](../include/sglang/runtime/execution/decoder_prefill_tiles.h#L104)<!-- prefix: "bool project" --> · [Launcher](../src/text_normalization.cu#L539)<!-- prefix: "cudaError_t " --> · [Row launcher](../src/text_ops.cu#L463)<!-- prefix: "cudaError_t " --> · [Separate dispatch](../src/prefill_fp8_sm100.cu#L594)<!-- prefix: "if (rows \u003e 1" --> · [Packed dispatch](../src/prefill_fp8_sm100.cu#L703)<!-- prefix: "if (rows \u003c 1" -->

## Runtime path

### Exact SIMT projections

Exact SIMT tries eight-row projection at ≥64 rows and >1,024 output channels. Inputs of 5,120/6,144/17,408 and output ≤65,535 use four-channel groups if output is divisible by four. At 64–128 rows and width divisible by eight, eight-channel groups reuse decoded operands. Both retain ordered FP32 accumulation and BF16 rounding. [Row dispatch](../src/text/fp8_projection.cu#L679)<!-- prefix: "if (rows \u003e 1" --> · [Tile selection](../src/text/fp8_projection.cu#L437)<!-- prefix: "if constexpr" -->

### Tensor-core projections

B200 packed Tensor Core FP8 includes eligible media and scoring rows. Swap-AB covers 1–5 rows. Multi-row TMA reads row-major E4M3 and FP32 activation/BF16 weight scales into BF16 slices. Fused quantizers or projection packing supply activations; GDN QKV/Z, GQA QKV and gate/up share packing. Split reductions can change scores. [Verification dispatch](../src/runtime/execution/decoder_forward_attention.cpp#L20)<!-- prefix: "void project" -->

Eligible 2–5-row GDN verification fuses BF16 B/A with separate-path arithmetic. Exact, scalar and ineligible calls use ordinary projections. [Gate launcher](../src/text_decode_bandwidth.cu#L720)<!-- prefix: "std::optiona" -->

### Multi-lane target verification

Verification preserves BF16 rounding, E4M3 activations and FP32 scales. Compact GDN may fuse output quantization.

| Route | Activations and B200 MMA geometry |
| --- | --- |
| Four-lane packets | Pack 6–20 rows. Through 20 rows use 8/16/32 MMA columns. |
| [wide K4 verification](mtp.md#target-round) | With adequate scratch, M40/M80/M160 use 48/80/160 columns and 4/4/3 slots. M40 gate/up uses 64 columns; M40 can fall back to 64 columns. Split epilogues sum K groups and ranks in FP32 order and round once to BF16. [Compact arithmetic](../src/text/decode_proj_fp8.cu#L94)<!-- prefix: "template \u003cbo" --> |

Wide fallback tries 40, 20, then ≤5 rows. Pending/recorded CUDA errors propagate, except optional setup reporting no status with only recorded `cudaErrorNotSupported`. [Wide dispatch](../src/text/decode_proj_fp8.cu#L618)<!-- prefix: "if (count ==" -->

Stream-owned scratch is 87,040 B/lane (up to 2,785,280 B). Eight TP2 lanes reserve 696,320 B with 409,600 live for depth scoring. Setup failure leaves ordinary draft packets. Resident shortage or allocation OOM reduces width and replans against the 2 GiB shared reserve. Other CUDA errors propagate. [Scratch reserve](../src/decoder_batch.cpp#L478)<!-- prefix: "bool Decoder" --> · [Width downgrade](../src/decoder_batch.cpp#L508)<!-- prefix: "WideMtpGroup" -->

### Large prefill tiles

Separate K0 Tensor Core projections need at least three rows, input width divisible by 128, and 16-byte-aligned contiguous FP8. At 129–1,023 rows, output width ≥8,192 or input width ≥6,144 selects paired 128×128 TMA. Packed projections need contiguous weights and scales and slices divisible by 128. [Eligibility](../src/prefill_fp8_sm100.cu#L558)<!-- prefix: "const int pr" --> · [Separate projection tiles](../src/prefill_fp8_sm100.cu#L594)<!-- prefix: "if (rows \u003e 1" -->

| Packed shape | First eligible tile |
| --- | --- |
| 1,024×5,120 GDN QKV/Z (16,384 columns), GQA QKV (14,336), or MLP gate/up (2×17,408) | 128×256 when slices are multiples of 256. MLP groups eight row tiles per column tile. |
| 768×5,120 MLP gate/up | 128×128 compact. Other 256–1,023-row packs try paired 128×256 or 128×128 by slice width. |
| Grouped 2,048 or 4,096×5,120 | 128×256 groups all row tiles per column tile. TP2 gate/up requires 4,096 rows and two 8,704-column slices. At 4,096 rows TP2 GDN uses 5,120/3,072-column slices and GQA uses 6,144/512/512, with copy batch one. |
| Grouped 4,096-row GDN/GQA output to 5,120 channels | 128×256 with three stages at 3,072 input channels and four at 6,144. Clean decline uses narrow TMA. |

Grouped routes allow 4,096 rows and ordinary calls 2,048. [Whole-tile MTP pairs](native-prefill-workspace.md#projection-pairs-for-mtp) and [partial-grant projection groups](native-prefill-workspace.md#partial-grant-projection-groups) share this guard. The SM100 CUTLASS/CuTe collective writes 5,120 BF16 channels for grouped GQA/GDN output at 6,144 inputs and MLP down at 1,024 or grouped 4,096 rows (17,408 inputs TP1, 8,704 TP2). It transposes FP32 activation scales and widens BF16 weight scales in idle gate/up storage on the workspace stream. Clean decline uses existing dispatch, including paired 128×128 down at 1,024 rows when large-row prefill is enabled. Setup and submission errors propagate. [Group guard](../src/decoder_prefill.cpp#L467)<!-- prefix: "const bool s" --> · [Shape guard](../src/text/prefill_down_sm100.cu#L146)<!-- prefix: "const bool d" --> · [Tile layout and pipeline details](https://github.com/para1lel/sglang.cpp/blob/6debe50edccdeda34ff307235ed38e74b6b8f2ee/docs/fp8-prefill-projections.md#large-prefill-tiles)

### Attention output quantization

Tensor Core GDN can fuse gated RMSNorm/FP8 output. Separate BF16 GDN and GQA quantize 6,144 channels per row. B200 fused GQA postgate needs eligible 4,096-row strict, partial or causal groups and TP2 rank agreement. Clean decline gates then quantizes at group granularity, per grant for partial groups. [Fused finish](../src/text_attention_gdn.cu#L489)<!-- prefix: "__global__ v" --> · [TP2 dispatch](../src/decoder_prefill.cpp#L287)<!-- prefix: "if (plan.pac" --> · [Eligibility and launch](../src/text_attention_gqa.cu#L395)<!-- prefix: "std::optiona" -->

The 1,024×6,144 warp path needs 4 B alignment for BF16/E4M3, launches six CTAs per row and emits 48 FP32 scales per row. Maxima floor at `1e-10`; scale uses FP32 round-to-nearest division by 448 and saturates E4M3 conversion. [Dispatch](../src/text_ops.cu#L469)<!-- prefix: "if (rows == " --> · [Scale and conversion](../src/text/text_cuda_primitives.cu#L48)<!-- prefix: "const float " -->

### Fused activation quantization

| Fused RMSNorm path | Shape and fallback |
| --- | --- |
| Eight-lane | 2–32 launch rows of width 5,120 with residual. Packed decode uses bucket width for launch rows. Branch/residual/gain need 16 B alignment, E4M3 output 8 B. Other valid shapes use ordinary fused or separate RMSNorm/quantization by width and alignment. [Packed decode](https://github.com/para1lel/sglang.cpp/blob/93648269d852e0fee78f7af4d1c3ffe4897aa361/src/text/decoder_graph_packed.cpp#L270)<!-- prefix: "void PackedD" --> · [Prefill](../src/runtime/execution/decoder_prefill_tiles.cpp#L366)<!-- prefix: "void Chunk::" --> · [Selection](../src/text_normalization.cu#L563)<!-- prefix: "if (residual" --> |
| Grouped | 4,096×5,120, optional residual. Eligible eager B200 TP2 stages asynchronously, quantizes BF16-rounded values and stores normalized BF16 for GDN but skips that store for GQA/MLP. BF16/E4M3/scales need 16/8/4 B alignment. Capture/misalignment uses ordinary normalization with BF16 store. CUDA errors propagate. [Grouped dispatch](../src/runtime/execution/decoder_prefill_tiles.cpp#L379)<!-- prefix: "void Chunk::" --> · [Launcher](../src/text_normalization.cu#L527)<!-- prefix: "std::optiona" --> · [Vector kernel](../src/text_normalization.cu#L89)<!-- prefix: "template \u003cbo" --> · [BF16 consumers](../src/decoder_prefill.cpp#L63)<!-- prefix: "normalize(pr" --> |

Packed decode can use [programmatic launch](long-paged-decode.md#packed-rmsnorm-programmatic-launch) to start projection weight loads during fused RMSNorm.

The specialization preserves FP32 RMS accumulation and BF16 residuals. It quantizes BF16-rounded values with one FP32 scale per 128 channels. Scale and reciprocal use FP32 round-to-nearest division. The conversion saturates to finite E4M3. [RMS arithmetic](../src/text/rms_quantize.cu#L101)<!-- prefix: "float values" --> · [Quantizer](../src/text/rms_quantize.cu#L56)<!-- prefix: "static __dev" -->

## Constraints and failure behavior

The `sm_100a` TMA, TMEM, and `tcgen05` path requires capability 10.0, a matching image, sufficient opt-in shared memory, and tensor maps. Output slices must be disjoint from inputs and each other. Clean separate and packed declines try the ordinary shape-selected tile and 128×128 tile, respectively. If packing declines, the decoder projects slices separately. Ineligible Tensor Core calls use SIMT. CUDA errors propagate. [Build](../xmake.lua#L177)<!-- prefix: "target(\"sgla" --> · [Narrow packed launch](../src/prefill_fp8_sm100.cu#L716)<!-- prefix: "return launc" --> · [Separate projection](../src/runtime/execution/decoder_prefill_tiles.cpp#L479)<!-- prefix: "void Prefill" -->

Attention output quantization requires a valid FP32 scale buffer with ordinary float alignment. Other valid shapes or alignments use the scalar kernel, which supports partial groups. The row launcher rejects null pointers, nonpositive widths, and row counts outside 1 through 65,535. The selected launch returns its CUDA status. Fused SwiGLU shares the warp kernel and retains its own dispatch and BF16 rounding. [Row launcher](../src/text_ops.cu#L463)<!-- prefix: "cudaError_t " --> · [SwiGLU dispatch](../src/text_ops.cu#L483)<!-- prefix: "cudaError_t " -->

Recorded terminal CUDA errors outrank nonterminal launch returns. The wide activation launchers consume matching error records and leave unrelated nonterminal records for their callers. [Launch status](../src/text_normalization.cu#L398)<!-- prefix: "cudaError_t " -->

## Verification

On a reserved B200, compare complete BF16 projection slices with the 128×128 control for both QKV checkpoint shapes, ragged rows, and 128-column tails. Check gate/up, quantized activations and scales, and down outputs around 1,024 rows. For the paired down checker, build `tests/text/fp8_prefill_tiling_check.cpp` against both releases with `-Wl,--wrap=cudaLaunchKernel -Wl,--wrap=cudaFuncGetAttributes`, then run the same checkpoint and fresh reference directory.

```bash
"$control_check" "$checkpoint" wide-base "$reference_dir"
"$candidate_check" "$checkpoint" paired "$reference_dir"
```

Require bitwise BF16 parity and a 320×1 grid, then exercise ineligibility, unavailable kernel, and hard launch errors. Check the collective at fixed grants and readers for four full tiles, unequal 4,096-row groups, and ordinary down. Match normalized and committed target and draft state and continuation. Distinguish clean decline from setup, descriptor, and GEMM errors.

Compare normalization and attention output quantization bitwise for BF16, E4M3, and FP32 scales in eager and graph replay. Test shape and alignment fallbacks, packed buckets, residual and no residual, scalar floor cases, and a scale-buffer shift of one float that retains warp dispatch. Use four 1,024-row tiles for strict groups and eight 512-row or sixteen 256-row grants for partial groups. Cover staged and per-lane readers, and raw grouped reads for partial groups. Compare against ordinary RMSNorm and separate BF16 gating on fixed grants and teacher histories. Require bitwise BF16 state, E4M3, FP32 scale and full BF16 vocabulary parity where arithmetic is preserved. Compare generated IDs separately with generated logprobs disabled. Time resident operators separately from [serving workloads](performance.md#native-concurrent-workloads). [Full projection and quantization matrix](https://github.com/para1lel/sglang.cpp/blob/6debe50edccdeda34ff307235ed38e74b6b8f2ee/docs/fp8-prefill-projections.md#verification)

### Wide verifier checks

At fixed checkpoint, grants, widths, and attention, compare M40 against two M20 packets and wide quantization against packet submission. Require bitwise BF16 parity with a fixed split plan, and bitwise normalized, residual, E4M3, and FP32 outputs with and without residual. Check IDs, finishes, draft and acceptance counts, compact GDN bypass, and actual routes.

Compare depth-major and packet scoring at widths 8/16/32. Match full BF16 scores, winners, and committed state. Trace one head launch per active depth. An invalid INT8 winner retries with BF16 and stops only that lane's draft if no finite winner exists. Cover sparse slots, short drafts, K0–K4, sampling, constraints, caps, cancellation, and slot reuse. Clean decline or allocation OOM uses smaller routes. Explicit setup and launch errors propagate, terminal status wins, and unrelated CUDA records survive. Require zero bytes after retirement. [Full verifier controls and timing](https://github.com/para1lel/sglang.cpp/blob/6debe50edccdeda34ff307235ed38e74b6b8f2ee/docs/fp8-prefill-projections.md#wide-verifier-checks)

### Held-out model quality

The short held-out checkpoint study compared exact and Tensor Core with pinned SGLang on reasoning, typed JSON, teacher scores, and seeded sampling. Use matched histories to diagnose divergence. [Frozen quality results and source identity](https://github.com/para1lel/sglang.cpp/blob/4dd8c887c278c67744ed80e7feb2a825a92197a2/docs/fp8-prefill-projections.md#held-out-model-quality) · [Earlier short-text probe](https://github.com/para1lel/sglang.cpp/blob/4dd8c887c278c67744ed80e7feb2a825a92197a2/docs/fp8-prefill-projections.md#historical-short-text-observations) · [Current flags](native-model-info.md#migrate-old-launch-commands)
