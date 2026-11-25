# Gated DeltaNet prefill

## Responsibilities and sources of truth

| Part | Responsibility | Source |
| --- | --- | --- |
| Decoder | Select the precision policy and provide tile scratch when the GDN layer qualifies. | [Policy](../src/decoder_prefill.cpp#L641)<!-- prefix: "const bool t" --> · [First-layer guard](../include/sglang/runtime/execution/decoder_prefill_tiles.h#L282)<!-- prefix: "const bool c" --> |
| GDN operator | Prepare rows, advance each layer's recurrent state, and finish output and convolution history. | [Prefill submission](../src/text_attention_gdn.cu#L295)<!-- prefix: "cudaError_t " --> |
| Verification | Check ordered parity, chunked numerical error, and causal continuation. | [Segmented probe](../tests/decoder/gdn_check.cu#L424)<!-- prefix: "void check_s" --> · [Chunked probe](../tests/decoder/gdn_check.cu#L551)<!-- prefix: "void check_c" --> · [Default route](../tests/decoder/gdn_routes.cpp#L66)<!-- prefix: "void check_d" --> · [Prefix fallback](../tests/decoder/gdn_routes.cpp#L104)<!-- prefix: "void check_p" --> · [Checkpoint probe](../tests/decoder/gdn_checkpoint.cpp#L101)<!-- prefix: "int main(int" --> |

## Runtime path

Eligible SM100 tensor-core prefill projects B and A together from the normalized BF16 input before GDN input preparation. Each Tensor Core block covers 32 token rows and one of ten contiguous 512-element slices of the input width. It writes FP32 partials for both 48-channel gates. A second kernel adds the ten partials in slice order and rounds each output to BF16. This accumulation order can change rounded gate values and downstream scores. [Partial products](../src/text/prefill_gdn_gates_sm100.cu#L43)<!-- prefix: "__global__ _" --> · [Ordered reduction](../src/text/prefill_gdn_gates_sm100.cu#L106)<!-- prefix: "__global__ v" -->

The fused SIMT fallback shares normalized input loads between B and A and reuses the weights across four rows. It preserves the separate projections' two FP32 accumulation chains, warp reduction, and BF16 rounding. [Fused SIMT projection](../src/text_decode_bandwidth.cu#L102)<!-- prefix: "__global__ v" --> · [Separate projection order](../src/text_ops.cu#L97)<!-- prefix: "for (std::si" -->

Exact prefill uses the original token-ordered GDN recurrence. On eligible B200 text tiles, GDN prepares convolution, normalized Q and K, and gates. The SM100 chunked route then prepares causal matrices in groups of 32 token rows. It uses BF16 Tensor Core products with FP32 accumulation to solve for corrections and advance the carry. The recurrence uses BF16 operands for the incoming carry projection, including between groups. Persistent carry remains FP32. The finish updates BF16 convolution history and supplies the [attention output projection](fp8-prefill-projections.md#attention-output-quantization). [Matrix preparation](../include/sglang/text/gdn/gdn_chunk.cuh#L192)<!-- prefix: "__global__ v" --> · [Chunked recurrence](../include/sglang/text/gdn/gdn_warp_mma.cuh#L348)<!-- prefix: "__global__ v" -->

Grouped input preparation assigns four consecutive rows to each 128-thread convolution block. Each channel reuses the four filter weights and a rolling input window. Each output uses its current sample and the preceding three raw samples. Rows before the tile read the incoming BF16 convolution history. The kernel rounds SiLU to BF16 before Q and K normalization and preserves each row's reduction order. A separate block in the same launch prepares the gates for each four-row group. [Grouped preparation](../include/sglang/text/gdn/gdn_prefill_kernels.cuh#L188)<!-- prefix: "__global__ v" --> · [History read](../include/sglang/text/gdn/gdn_prefill_kernels.cuh#L44)<!-- prefix: "__device__ _" -->

Per-grant matrix preparation uses one 128-thread block per value head and row group. Recurrence uses a `48 × 2` grid with 256 threads per block. Eight warps share the staging work in each block. Each warp processes eight value channels, so the two blocks cover all 128 channels per head. [Matrix launch](../include/sglang/text/gdn/gdn_chunk.cuh#L264)<!-- prefix: "const auto s" --> · [Recurrence launch](../include/sglang/text/gdn/gdn_chunk.cuh#L271)<!-- prefix: "return cudaL" --> · [Channel assignment](../include/sglang/text/gdn/gdn_warp_mma.cuh#L282)<!-- prefix: "const int fe" -->

On SM100, eligible grants can share the preparation, matrix, recurrence, and
finish launches across up to four segments. The large
[projection groups](native-prefill-workspace.md#projection-pairs-for-mtp) pass
four 1,024-row tiles at 4,096 rows or two at 2,048 rows. The regular layerwise
path can group two or three eligible grants. The same provider handles eligible [partial-grant groups](#grouped-partial-grants).
[Segment limit](../include/sglang/decoder_cuda.cuh#L156)<!-- prefix: "inline const" --> · [Decoder dispatch](../src/decoder_prefill.cpp#L380)<!-- prefix: "void prefill" -->

The grid Z coordinate selects a grant with its own state, row slices, and
scratch spans. Each block uses its grant's row count, including when aligned
grant lengths differ. Each grant's absolute start determines bank-zero
eligibility. If any grant starts at position zero, the first GDN layer runs
per grant for the whole group. A declined grouped submission also runs
per grant. A CUDA error from the grouped provider propagates.
[Position guard](../include/sglang/runtime/execution/decoder_prefill_tiles.h#L660)<!-- prefix: "grant.rows %" --> · [Grouped submission](../src/text_attention_gdn.cu#L521)<!-- prefix: "std::optiona" -->

Grouped matrix preparation uses one 384-thread block per Q and K head and row group. The three associated value heads share Q and K staging and the FP32 dot products. Each value head applies its own gates and solves its own triangular system. [Grouped matrix launch](../include/sglang/text/gdn/gdn_chunk.cuh#L282)<!-- prefix: "const auto s" -->

After the dot products, grouped preparation reuses Q and K staging for the BF16 score and solution rows. It pads each shared row to 34 elements for column writes. Threads store adjacent BF16 pairs to the existing 32-element rows in global scratch. The grouped store preserves the solve arithmetic and matrix record layout. [Shared staging](../include/sglang/text/gdn/gdn_chunk.cuh#L144)<!-- prefix: "constexpr bo" --> · [Paired store](../include/sglang/text/gdn/gdn_chunk.cuh#L70)<!-- prefix: "__device__ _" -->

The grouped recurrence reads and writes adjacent BF16 pairs when it scales the keys. It also reads value pairs for the residual and writes output pairs. The per-grant recurrence accesses one BF16 element at a time in these steps. [Grouped recurrence](../include/sglang/text/gdn/gdn_warp_mma.cuh#L354)<!-- prefix: "__global__ _" --> · [Key scaling](../include/sglang/text/gdn/gdn_warp_mma.cuh#L169)<!-- prefix: "if constexpr" --> · [Residual inputs](../include/sglang/text/gdn/gdn_warp_mma.cuh#L199)<!-- prefix: "if constexpr" --> · [Output store](../include/sglang/text/gdn/gdn_warp_mma.cuh#L234)<!-- prefix: "if constexpr" -->

The ordered tensor-core fallback uses the paired kernel for at least 32 rows with 16-byte-aligned recurrence workspace. Otherwise it uses the original recurrence. The paired kernel shares prepared Q, K, and gates between adjacent value channels. SM100 uses packed FP32 arithmetic and preserves each channel's token, update, and warp reduction order. Other compiled architectures use scalar arithmetic in this paired kernel. [Paired kernel](../include/sglang/text/gdn/gdn_prefill_kernels.cuh#L285)<!-- prefix: "__global__ v" --> · [Paired guard](../src/text_attention_gdn.cu#L319)<!-- prefix: "const bool p" -->

### Grouped partial grants

A [partial-grant projection group](native-prefill-workspace.md#partial-grant-projection-groups) can share GDN launches at 4,096 rows on SM100. It needs eight 512-row grants or sixteen 256-row grants. The decoder submits four grants at a time, with separate convolution history, FP32 carry, and scratch for each grant. The groups use the existing prefill arena. [Shape guard](../include/sglang/runtime/execution/decoder_prefill_tiles.h#L700)<!-- prefix: "inline bool " --> · [Four-grant dispatch](../include/sglang/runtime/execution/decoder_prefill_tiles.h#L712)<!-- prefix: "inline void " -->

If any of the four starts at position zero, bank zero uses the existing per-grant path for those four. The decoder also uses that path when the provider declines before GDN submission. A returned CUDA error propagates. The decoder clears the CUDA last-error value if it matches the reported error. [Bank guard](../include/sglang/runtime/execution/decoder_prefill_tiles.h#L660)<!-- prefix: "grant.rows %" --> · [Error handling](../include/sglang/runtime/execution/decoder_prefill_tiles.h#L676)<!-- prefix: "if (status) " -->

## Grouped target verification

With target Tensor Cores enabled, four eager five-row MTP lanes share the grouped GDN verifier launches. Each lane retains its own convolution history and recurrent state. [Dispatch](../include/sglang/runtime/execution/decoder_forward_layers.h#L550)<!-- prefix: "if (batch.pa" --> · [Grouped launch](../src/text/gdn_verify.cu#L283)<!-- prefix: "cudaError_t " -->

Each recurrence warp advances two independent value channels eight channels apart. Eight warps cover 16 channels per block. Eight blocks cover each 128-channel value head. SM100 packs the FP32 multiply and FMA operations across the two channels. Other architectures use scalar arithmetic for each channel. Each channel keeps its token order, accumulation through the four key parts, and warp reduction order. The kernel stores BF16 outputs and FP32 tail state in their existing slots. [Recurrence](../src/text/gdn_verify.cu#L147)<!-- prefix: "__global__ v" --> · [Packed arithmetic](../src/text/gdn_verify.cu#L130)<!-- prefix: "__device__ f" --> · [Output reduction](../src/text/gdn_verify.cu#L187)<!-- prefix: "const float " -->

Packed 80- and 160-row K4 groups can run GDN across 16 or 32 eager lanes. Each layer submits one gate, convolution, recurrence, and finish kernel. Each lane owns its history and recurrent state in the usual banks. Convolution copies raw BF16 taps into the tentative bank. The finish rounds gated RMSNorm outputs to BF16 before writing E4M3 activations and FP32 scales. It uses the existing packed projection buffer and one scale per 128 channels. [Compact dispatch](../include/sglang/runtime/execution/decoder_forward_layers.h#L440)<!-- prefix: "inline bool " --> · [History copy](../src/text/gdn_verify.cu#L62)<!-- prefix: "job.tail.his" --> · [Fused finish](../src/text/gdn_verify.cu#L242)<!-- prefix: "if constexpr" -->

A setup decline before the gate launch uses the same four-lane packets and separate quantization. Pending CUDA errors and submission errors propagate without retry. A recorded terminal error takes precedence over a recoverable submission error. [Gate setup](../src/text_decode_bandwidth.cu#L743)<!-- prefix: "std::optiona" --> · [Packet fallback](../include/sglang/runtime/execution/decoder_forward_layers.h#L550)<!-- prefix: "if (batch.pa" --> · [Launch status](../src/text/gdn_verify.cu#L306)<!-- prefix: "cudaError_t " -->

Use the [batched target GDN check](../tests/decoder/README.md#batched-target-gdn) to compare scores, normalized rows, and committed state against scalar decode. Exercise zero through five consumed rows and a replacement token.

Compare 80- and 160-row launches with four-lane controls on the same verification inputs. Compare BF16 scores, normalized rows, committed carry, FP8 activations, and scales. Match raw BF16 history bitwise. Exercise setup decline, propagation of CUDA errors, and reuse after retirement.

## Constraints and failure behavior

The B/A fast paths require tensor-core prefill policy and two BF16 weight matrices of 48 rows by 5,120 columns. The SM100 projection accepts 512–4,096 rows, including a partial final 32-row block. It needs nonnull buffers, 16-byte-aligned weights, input, and scratch, and supported TMA setup. It borrows 3,840 scratch bytes per token row from the MLP gate buffer. Reduction consumes those bytes on the same stream before recurrence or the MLP reuses them. [Decoder dispatch](../include/sglang/runtime/execution/decoder_prefill_tiles.h#L250)<!-- prefix: "if (tensor_c" --> · [SM100 preflight](../src/text/prefill_gdn_gates_sm100.cu#L209)<!-- prefix: "std::optiona" -->

A clean SM100 setup decline tries the fused SIMT projection. That path accepts multiples of four from 16 through 65,535 rows with aligned weights and input. If it also declines, the decoder projects B and A separately in that order. CUDA errors propagate from either fused path. [SIMT preflight](../src/text_decode_bandwidth.cu#L575)<!-- prefix: "if (rows < 1" --> · [Descriptor errors](../src/text/prefill_gdn_gates_sm100.cu#L138)<!-- prefix: "std::optiona" --> · [Separate fallback](../include/sglang/runtime/execution/decoder_prefill_tiles.h#L271)<!-- prefix: "project(laye" -->

The first GDN layer uses ordered recurrence for a tile that starts at prompt position zero. Later GDN layers in that tile and all GDN layers in subsequent tiles may use chunked recurrence. A tile that resumes a saved prefix uses the absolute restored position. An earlier recurrence experiment on W2 found that reordering only the first GDN call could change later model scores. The guard preserves that call's update order.

Tensor-core prefill uses grouped input preparation at 128 rows and above when the row count is divisible by four. Other counts use one row per preparation block. Preparation selects its route independently of matrix scratch and the recurrence route. The position-zero first GDN layer can therefore use grouped preparation with ordered recurrence. Groups write disjoint workspace rows and gates. The finish saves three raw BF16 samples per channel after preparation and recurrence on the same stream. State ownership and workspace sizes stay the same. [Shape guard](../src/text_attention_gdn.cu#L309)<!-- prefix: "const bool g" --> · [History write](../src/text_attention_gdn.cu#L124)<!-- prefix: "final_histor" -->

Chunked recurrence requires SM100, at least 128 rows per call, and sufficient 32-byte-aligned matrix scratch. The decoder borrows the MLP gate buffer until the recurrence consumes the matrices on the same stream. The later gate projection can then write the buffer. Chunked recurrence leaves request-state and pool sizes unchanged. Calls below 128 rows or with unavailable, short, or unaligned scratch use the ordered fallback. A single-row tail uses the existing single-row path. CUDA discovery, configuration, and launch errors propagate. [Scratch guard](../include/sglang/text/gdn/gdn_chunk.cuh#L227)<!-- prefix: "inline bool " --> · [CUDA configuration](../include/sglang/text/gdn/gdn_chunk.cuh#L233)<!-- prefix: "inline cudaE" -->

Each 32-row group needs one 4,480-byte matrix record per value head. At 1,024 rows, 32 groups across 48 heads use 6,881,280 bytes (6.5625 MiB) of the borrowed buffer. The two causal matrices have a 32-element BF16 row pitch in global scratch and a 40-element pitch in recurrence shared memory. The shared pitch shifts successive rows across bank groups. Each recurrence block uses 45,824 bytes of dynamic shared memory for its two staging buffers. [Global layout](../include/sglang/text/gdn/gdn_chunk.cuh#L23)<!-- prefix: "struct align" --> · [Scratch size](../include/sglang/text/gdn/gdn_chunk.cuh#L221)<!-- prefix: "inline std::" --> · [Shared layout](../include/sglang/text/gdn/gdn_warp_mma.cuh#L114)<!-- prefix: "struct Stage" --> · [Dynamic size](../include/sglang/text/gdn/gdn_warp_mma.cuh#L158)<!-- prefix: "constexpr in" -->

Chunked recurrence can reorder arithmetic. BF16 output, FP32 carry, logits, and generated IDs can differ from ordered execution. Chunked GDN and [flash GQA](flash-prefill.md#runtime-path) qualify independently within the same prefill tile.

## Verification

On a reserved B200, build the segmented probe with `SGLANG_TEST_GDN_TILED` and wrap `cudaLaunchKernel` at link time. Run `segmented paired CAPTURE` on both the packed candidate and the scalar paired control. The probe starts with nonzero history and recurrent state and carries them through eight segments. It checks at least the 31-, 32-, and 33-row boundary and a two-byte workspace offset.

Require bitwise equality of BF16 output, FP32 carry, and BF16 history across the CUDA paths. Require exact history and apply the probe tolerances for output and carry against the scalar FP64 CPU oracle. Run `short paired CAPTURE` under memcheck, racecheck, initcheck, and synccheck. Require zero errors and hazards.

Follow the [CUDA check instructions](../tests/decoder/README.md#gdn-prefill) on a reserved B200. Use `gdn-check chunked` for the 127-, 128-, and 129-row dispatches and the 193-row tail. For eligible calls, check the preparation and recurrence launch dimensions given in [Runtime path](#runtime-path). Check scratch alignment, one-byte shortfall, guards, and causality across a group boundary. Compare against the independent FP64 oracle with the probe's output and carry error limits. Require exact BF16 convolution history and bitwise equality for ordered fallbacks.

Use `gdn-routes` to check the first tile and a tile restored from a prefix while another lane decodes. Expect one ordered and 47 chunked calls on the first eligible tile, and 48 chunked calls on the restored tile. Decode-only steps use no chunked GDN. Check lane retirement and reuse. Compare exact-policy IDs and scores across builds.

Run the checkpoint probe with 2,048 prompt IDs and eight fixed teacher IDs. Leave `SGLANG_SERIAL_PREFILL_ROWS` unset. Require two 1,024-row prefill tiles. For ordered changes, compare carries and histories across all 48 GDN layers after each tile and the decode sequence. Compare full BF16 logits and winners along the fixed history.

If grouped preparation preserves arithmetic, compare output, carry, and history bitwise across builds. Check the 127-, 128-, and 129-row boundary and perturb a later row within a four-row group. Require earlier outputs to match. Measure each stage and the preparation-to-finish interval on the same checkpoint tiles in both run orders. Record any serving estimate with its archived launch count and label it as an estimate.

For grouped projection prefill, compare the full state and draft continuation
with the per-grant control at the same prompt positions. Check bank zero at
position zero and on a later tile. Record grouped and per-grant launches
separately. See the [workspace pressure checks](native-prefill-workspace.md#verification).

On a reserved B200, keep grants, positions, GQA dispatch, and B/A projection dispatch fixed. Change only the GDN submission after projection. Exercise both 4,096-row shapes and a smaller or unequal group. Include bank zero at position zero and at a later tile. Check which grants share launches. Compare BF16 outputs, FP32 carry, and BF16 convolution history bitwise. With positive MTP depth, also compare draft continuation bitwise. Force grouped setup to decline and check per-grant parity. With positive MTP depth, inject a recoverable CUDA setup error and a recoverable launch error separately. Require each error to propagate without retrying the affected grants. Admit a fresh cohort to the same batch after each error and require it to advance.

For the SM100 B/A projection, hold the tile schedule, recurrence, and attention routes fixed against the fused SIMT control. Compare BF16 gate outputs with an FP64 dot-product oracle. Test 511 and 512 rows. Test a partial 32-row block and a 4,096-row group. Check clean fallback and error propagation independently. Time the projection and reduction together under matched cache conditions.

The B200 operator comparison reduced B/A projection time at the tested sizes from 512 to 4,096 rows. Fixed-history checkpoint probes changed some top predictions on natural documents. Both document-level 95% bootstrap intervals for mean loss change included zero. Use the frozen workload protocol for serving timing.

For reordered arithmetic, follow the [operator qualification procedure](performance.md#operator-qualification). Compare logits and labeled-token negative log likelihood on identical prompt and teacher histories. Record greedy divergence separately. Trace the position-zero first-layer guard, eligible layers, short tails, exact, and positive-depth MTP. Use the [frozen serving protocols](performance.md) with matched tile schedules, graph settings, and warmups. To isolate GDN when flash is active, hold flash dispatch fixed across both arms.

For unchanged recurrence arithmetic, match the prior chunked kernel's BF16 output and FP32 carry bitwise from nonzero carry. Time isolated recurrence separately from serving. In the archived W6 diagnostic, recurrence time fell and IDs, usage, and finish reasons matched. Flash GQA dominated GPU kernel time. The single cohort used extended transport timeouts and cannot supply a score under the frozen protocol. [Historical diagnostic](https://github.com/para1lel/sglang.cpp/blob/4dd8c887c278c67744ed80e7feb2a825a92197a2/docs/gdn-prefill.md#verification) · [Frozen W6 protocol](../benchmarks/concurrent/README.md#conditions-and-gates)

The packed ordered implementation preserved BF16 logits and reduced 1,024-row operator time. The chunked comparison with Flash active reduced measured W1 and W2 TTFT. Report warmup and measured samples, variability, and peak device memory separately.

The archived reordered comparison found small mean loss changes on natural documents, with bootstrap intervals spanning zero. The authored W2 fixture changed top predictions on the fixed teacher history and diverged during greedy generation. Keep these two checks separate. [Archived checkpoint measurements](https://github.com/para1lel/sglang.cpp/blob/4dd8c887c278c67744ed80e7feb2a825a92197a2/docs/gdn-prefill.md#verification)
