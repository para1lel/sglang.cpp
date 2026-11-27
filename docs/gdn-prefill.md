# Gated DeltaNet prefill

## Responsibilities and sources of truth

| Part | Responsibility | Source |
| --- | --- | --- |
| Decoder | Select the precision policy and provide tile scratch when the GDN layer qualifies. | [Policy](../src/decoder_prefill.cpp#L626)<!-- prefix: "const bool t" --> · [First-layer guard](../include/sglang/runtime/execution/decoder_prefill_tiles.h#L299)<!-- prefix: "const bool c" --> |
| GDN operator | Prepare rows, advance each layer's recurrent state, and finish output and convolution history. | [Prefill submission](../src/text_attention_gdn.cu#L339)<!-- prefix: "cudaError_t " --> |
| Verification | Check ordered parity, chunked numerical error, and causal continuation. | [Segmented probe](../tests/decoder/gdn_check.cu#L424)<!-- prefix: "void check_s" --> · [Chunked probe](../tests/decoder/gdn_check.cu#L551)<!-- prefix: "void check_c" --> · [Default route](../tests/decoder/gdn_routes.cpp#L66)<!-- prefix: "void check_d" --> · [Prefix fallback](../tests/decoder/gdn_routes.cpp#L104)<!-- prefix: "void check_p" --> · [Checkpoint probe](../tests/decoder/gdn_checkpoint.cpp#L101)<!-- prefix: "int main(int" --> |

## Runtime path

Eligible SM100 tensor-core prefill projects B and A together from the normalized BF16 input before GDN input preparation. Each Tensor Core block covers 32 token rows and one of ten contiguous 512-element slices of the input width. It writes FP32 partials for both 48-channel gates. A second kernel adds the ten partials in slice order and rounds each output to BF16. This accumulation order can change rounded gate values and downstream scores. [Partial products](../src/text/prefill_gdn_gates_sm100.cu#L43)<!-- prefix: "__global__ _" --> · [Ordered reduction](../src/text/prefill_gdn_gates_sm100.cu#L106)<!-- prefix: "__global__ v" -->

The SIMT fallback shares normalized input loads and reuses gate weights across four rows. It preserves both FP32 accumulation chains, warp reduction, and BF16 rounding. [Fused SIMT projection](../src/text_decode_bandwidth.cu#L102)<!-- prefix: "__global__ v" --> · [Separate projection order](../src/text_ops.cu#L100)<!-- prefix: "for (std::si" -->

Exact prefill uses token-ordered recurrence. On eligible B200 text tiles, GDN prepares convolution, normalized Q and K, and gates. SM100 chunked recurrence prepares 32-row causal matrices. BF16 Tensor Core products with FP32 accumulation solve corrections and advance the carry. Carry projection uses BF16 operands between groups. Persistent carry stays FP32. The finish updates BF16 convolution history and supplies the [attention output projection](fp8-prefill-projections.md#attention-output-quantization). [Matrix preparation](../include/sglang/text/gdn/gdn_chunk.cuh#L191)<!-- prefix: "__global__ v" --> · [Chunked recurrence](../include/sglang/text/gdn/gdn_warp_mma.cuh#L420)<!-- prefix: "__global__ v" -->

Each 128-thread grouped convolution block handles four consecutive rows. Channels reuse four filter weights and a rolling window of the current and three preceding raw samples. Earlier rows read incoming BF16 history. SiLU rounds to BF16 before Q and K normalization, which preserves reduction order. A separate block in the same launch prepares the four rows’ gates. [Grouped preparation](../include/sglang/text/gdn/gdn_prefill_kernels.cuh#L185)<!-- prefix: "__global__ v" --> · [History read](../include/sglang/text/gdn/gdn_prefill_kernels.cuh#L41)<!-- prefix: "__device__ _" -->

Single-grant `GdnLocal` uses a `24 × 4` grid per TP2 rank. Four recurrence warps handle eight value channels each. `GdnFull` uses eight warps and a `48 × 2` grid. Preparation shares Q/K dot products across three local value heads and solves each separately. In the pipelined variant, four extra warps stage BF16 values and scaled keys in alternating frames. A producer barrier completes copies before key scaling. A block barrier protects frame reuse. The four recurrence warps retain the same arithmetic. [Matrix launch](../include/sglang/text/gdn/gdn_chunk.cuh#L304)<!-- prefix: "constexpr in" --> · [Recurrence launch](../include/sglang/text/gdn/gdn_chunk.cuh#L315)<!-- prefix: "return cudaL" --> · [Value staging](../include/sglang/text/gdn/gdn_warp_mma.cuh#L165)<!-- prefix: "copy_async(t" --> · [Producer barrier](../include/sglang/text/gdn/gdn_warp_mma.cuh#L222)<!-- prefix: "asm volatile" -->

On SM100, eligible grants can share the preparation, matrix, recurrence, and
finish launches across up to four segments. The large
[projection groups](native-prefill-workspace.md#projection-pairs-for-mtp) pass
four 1,024-row tiles at 4,096 rows or two at 2,048 rows. The regular layerwise
path can group two or three eligible grants. The same provider handles eligible [partial-grant groups](#grouped-partial-grants).
[Segment limit](../include/sglang/decoder_cuda.cuh#L160)<!-- prefix: "inline const" --> · [Decoder dispatch](../src/decoder_prefill.cpp#L363)<!-- prefix: "void prefill" -->

The grid Z coordinate selects a grant with its own state, row slices, and
scratch spans. Each block uses its grant's row count, including when aligned
grant lengths differ. Each grant's absolute start determines bank-zero
eligibility. If any grant starts at position zero, the first GDN layer runs
per grant for the whole group. A declined grouped submission also runs
per grant. A CUDA error from the grouped provider propagates.
[Position guard](../include/sglang/runtime/execution/decoder_prefill_tiles.h#L737)<!-- prefix: "grant.rows %" --> · [Grouped submission](../src/text_attention_gdn.cu#L609)<!-- prefix: "std::optiona" -->

Grouped matrix preparation uses one 384-thread block per Q and K head and row group. The three associated value heads share Q and K staging and the FP32 dot products. Each value head applies its own gates and solves its own triangular system. [Grouped matrix launch](../include/sglang/text/gdn/gdn_chunk.cuh#L343)<!-- prefix: "const auto s" -->

After the dot products, grouped preparation reuses Q and K staging for the BF16 score and solution rows. It pads each shared row to 34 elements for column writes. Threads store adjacent BF16 pairs to the existing 32-element rows in global scratch. The grouped store preserves the solve arithmetic and matrix record layout. [Shared staging](../include/sglang/text/gdn/gdn_chunk.cuh#L143)<!-- prefix: "constexpr bo" --> · [Paired store](../include/sglang/text/gdn/gdn_chunk.cuh#L71)<!-- prefix: "__device__ _" -->

Both grouped recurrence and single-segment `GdnLocal` use adjacent BF16 pairs for key scaling, residual inputs, and output stores. Grouped recurrence uses eight warps per block. Single-segment `GdnFull` uses scalar accesses. [Grouped recurrence](../include/sglang/text/gdn/gdn_warp_mma.cuh#L428)<!-- prefix: "__global__ _" --> · [Key scaling](../include/sglang/text/gdn/gdn_warp_mma.cuh#L201)<!-- prefix: "if constexpr" --> · [Residual inputs](../include/sglang/text/gdn/gdn_warp_mma.cuh#L253)<!-- prefix: "if constexpr" --> · [Output store](../include/sglang/text/gdn/gdn_warp_mma.cuh#L288)<!-- prefix: "if constexpr" -->

Ordered fallback pairs adjacent value channels at 32 or more rows with 16-byte-aligned recurrence workspace. Otherwise it uses the original kernel. Paired channels share Q, K, and gates. SM100 uses packed FP32 and preserves token, update, and warp reduction order. Other architectures use scalar arithmetic. [Paired kernel](../include/sglang/text/gdn/gdn_prefill_kernels.cuh#L282)<!-- prefix: "__global__ v" --> · [Paired guard](../src/text_attention_gdn.cu#L365)<!-- prefix: "const bool p" -->

### Grouped partial grants

A [partial-grant projection group](native-prefill-workspace.md#partial-grant-projection-groups) can share GDN launches at 4,096 rows on SM100. It needs eight 512-row grants or sixteen 256-row grants. The decoder submits four grants at a time, with separate convolution history, FP32 carry, and scratch for each grant. The groups use the existing prefill arena. [Shape guard](../include/sglang/runtime/execution/decoder_prefill_tiles.h#L779)<!-- prefix: "inline bool " --> · [Four-grant dispatch](../include/sglang/runtime/execution/decoder_prefill_tiles.h#L791)<!-- prefix: "inline void " -->

If any of the four starts at position zero, bank zero uses the existing per-grant path for those four. The decoder also uses that path when the provider declines before GDN submission. A returned CUDA error propagates. The decoder clears the CUDA last-error value if it matches the reported error. [Bank guard](../include/sglang/runtime/execution/decoder_prefill_tiles.h#L737)<!-- prefix: "grant.rows %" --> · [Error handling](../include/sglang/runtime/execution/decoder_prefill_tiles.h#L755)<!-- prefix: "if (status) " -->

## Vendor prefill opt-in

An optional vendor kernel takes precedence for eligible rank-local GDN prefill when `SGLANG_GDN_VENDOR_PREFILL` is exactly `1`. The flag is read once per process. Calls outside the opt-in or vendor eligibility use native dispatch. [Opt-in check](../src/runtime/backend/gdn_vendor_prefill.cu#L463)<!-- prefix: "bool gdn_ven" -->

The device must report compute capability 10.0 with 148 streaming multiprocessors, cached per device. [Device support](../src/runtime/backend/gdn_vendor_prefill.cu#L38)<!-- prefix: "bool vendor_" -->

Eligible calls carry positive row counts up to 262,144, in multiples of four. Local GDN layers route BF16 and FP8 advances, grouped segments, and coalesced partial-grant runs through the vendor kernel. [Eligibility](../src/runtime/backend/gdn_vendor_prefill.cu#L500)<!-- prefix: "bool gdn_ven" --> · [FP8 advance](../include/sglang/runtime/execution/decoder_prefill_tiles.h#L304)<!-- prefix: "if (quantize" --> · [Grouped segments](../include/sglang/runtime/execution/decoder_prefill_tiles.h#L704)<!-- prefix: "if (dims.is_" --> · [Partial runs](../include/sglang/runtime/execution/decoder_prefill_tiles.h#L808)<!-- prefix: "chunk.tensor" -->

The workspace reserves the larger of the native and vendor layouts, independent of the runtime flag. Toggling the environment variable needs no rebuild. [Workspace sizing](../src/runtime/backend/gdn_vendor_prefill.cu#L505)<!-- prefix: "std::size_t " -->

Builds without the vendor archive compile the same interfaces as stubs. Stubs report disabled and not-supported, leaving dispatch native. [No-asset stubs](../include/sglang/runtime/backend/gdn_vendor_prefill.h#L46)<!-- prefix: "inline bool " -->

Qualification compared matched-layer BF16 outputs against the pinned vLLM/FlashInfer reference. The vendor kernel was closer on all twelve matched outputs. Native was closer on the one 248,320-entry complete-vocabulary row. No blanket parity or speed claim is made.

## Grouped target verification

With target Tensor Cores enabled, four eager five-row MTP lanes share the grouped GDN verifier launches. Each lane retains its own convolution history and recurrent state. [Dispatch](../include/sglang/runtime/execution/decoder_forward_layers.h#L598)<!-- prefix: "if (batch.pa" --> · [Grouped launch](../src/text/gdn_verify.cu#L283)<!-- prefix: "cudaError_t " -->

Each recurrence warp advances two independent value channels eight channels apart. Eight warps cover 16 channels per block. Eight blocks cover each 128-channel value head. SM100 packs the FP32 multiply and FMA operations across the two channels. Other architectures use scalar arithmetic for each channel. Each channel keeps its token order, accumulation through the four key parts, and warp reduction order. The kernel stores BF16 outputs and FP32 tail state in their existing slots. [Recurrence](../src/text/gdn_verify.cu#L147)<!-- prefix: "__global__ v" --> · [Packed arithmetic](../src/text/gdn_verify.cu#L130)<!-- prefix: "__device__ f" --> · [Output reduction](../src/text/gdn_verify.cu#L187)<!-- prefix: "const float " -->

Packed 80- and 160-row K4 groups can run GDN across 16 or 32 eager lanes. Each layer submits one gate, convolution, recurrence, and finish kernel. Each lane owns its history and recurrent state in the usual banks. Convolution copies raw BF16 taps into the tentative bank. The finish rounds gated RMSNorm outputs to BF16 before writing E4M3 activations and FP32 scales. It uses the existing packed projection buffer and one scale per 128 channels. [Compact dispatch](../include/sglang/runtime/execution/decoder_forward_layers.h#L424)<!-- prefix: "inline bool " --> · [History copy](../src/text/gdn_verify.cu#L62)<!-- prefix: "job.tail.his" --> · [Fused finish](../src/text/gdn_verify.cu#L242)<!-- prefix: "if constexpr" -->

A setup decline before the gate launch uses the same four-lane packets and separate quantization. Pending CUDA errors and submission errors propagate without retry. A recorded terminal error takes precedence over a recoverable submission error. [Gate setup](../src/text_decode_bandwidth.cu#L734)<!-- prefix: "std::optiona" --> · [Packet fallback](../include/sglang/runtime/execution/decoder_forward_layers.h#L598)<!-- prefix: "if (batch.pa" --> · [Launch status](../src/text/gdn_verify.cu#L306)<!-- prefix: "cudaError_t " -->

Use the [batched target GDN check](../tests/decoder/README.md#batched-target-gdn) to compare scores, normalized rows, and committed state against scalar decode. Exercise zero through five consumed rows and a replacement token.

Compare 80- and 160-row launches with four-lane controls on the same verification inputs. Compare BF16 scores, normalized rows, committed carry, FP8 activations, and scales. Match raw BF16 history bitwise. Exercise setup decline, propagation of CUDA errors, and reuse after retirement.

MTP commit-phase GDN prefix replay is dispatched as one batched address-table launch per rank per commit group instead of two launches per layer. Per-layer kernel arguments, order, and results are unchanged. Setting `SGLANG_DISABLE_GDN_REPLAY_BATCH` forces the exact per-layer reference dispatch on both ranks. [Batched replay](../include/sglang/runtime/execution/decoder_batch_state.h#L241)<!-- prefix: "TextGdnRepla" -->

## Constraints and failure behavior

The B/A fast paths require tensor-core prefill policy and two BF16 weight matrices of 48 rows by 5,120 columns. The SM100 projection accepts 512–4,096 rows, including a partial final 32-row block. It needs nonnull buffers, 16-byte-aligned weights, input, and scratch, and supported TMA setup. It borrows 3,840 scratch bytes per token row from the MLP gate buffer. Reduction consumes those bytes on the same stream before recurrence or the MLP reuses them. [Decoder dispatch](../include/sglang/runtime/execution/decoder_prefill_tiles.h#L267)<!-- prefix: "if (tensor_c" --> · [SM100 preflight](../src/text/prefill_gdn_gates_sm100.cu#L209)<!-- prefix: "std::optiona" -->

A clean SM100 setup decline tries the fused SIMT projection. That path accepts multiples of four from 16 through 65,535 rows with aligned weights and input. If it also declines, the decoder projects B and A separately in that order. CUDA errors propagate from either fused path. [SIMT preflight](../src/text_decode_bandwidth.cu#L569)<!-- prefix: "if (rows < 1" --> · [Descriptor errors](../src/text/prefill_gdn_gates_sm100.cu#L138)<!-- prefix: "std::optiona" --> · [Separate fallback](../include/sglang/runtime/execution/decoder_prefill_tiles.h#L288)<!-- prefix: "project(laye" -->

The first GDN layer uses ordered recurrence for a tile that starts at prompt position zero. Later GDN layers in that tile and all GDN layers in subsequent tiles may use chunked recurrence. A tile that resumes a saved prefix uses the absolute restored position. An earlier recurrence experiment on W2 found that reordering only the first GDN call could change later model scores. The guard preserves that call's update order.

Grouped input preparation requires at least 128 rows in multiples of four. Other calls prepare one row per block. Selection is independent of matrix scratch and recurrence. The first GDN layer at position zero can group preparation with ordered recurrence. Groups write disjoint rows and gates. The finish saves three raw BF16 samples per channel on the same stream. Ownership and workspace sizes stay the same. [Shape guard](../src/text_attention_gdn.cu#L355)<!-- prefix: "const bool g" --> · [History write](../include/sglang/text/gdn/gdn_finish_history.cuh#L13)<!-- prefix: "final_histor" -->

Native chunked recurrence requires compute capability 10.0, at least 128 rows, and sufficient 32-byte-aligned matrix scratch borrowed from the MLP gate buffer. Ineligible calls use ordered recurrence. Single-row tails use the single-row path. CUDA errors propagate. [Scratch guard](../include/sglang/text/gdn/gdn_chunk.cuh#L226)<!-- prefix: "inline bool " --> · [Configuration](../include/sglang/text/gdn/gdn_chunk.cuh#L245)<!-- prefix: "inline cudaE" -->

Each 32-row group uses a 4,480-byte matrix record per value head, or 3,440,640 bytes for 1,024 TP2 rows. BF16 matrix row pitches are 32 globally and 40 in shared memory. Eligible `GdnLocal` calls with at least 1,024 rows use four staging warps with at least 77,568 opt-in shared bytes per block and the earlier staged variant at 68,864–77,567 bytes. Other eligible calls use unstaged chunked recurrence with at least 64,768 bytes. Below that, ordered recurrence applies. Grouped and TP1 calls keep their existing dispatch. [Capacity guard](../include/sglang/text/gdn/gdn_chunk.cuh#L274)<!-- prefix: "if (shared_l" --> · [Shared bytes](../include/sglang/text/gdn/gdn_warp_mma.cuh#L183)<!-- prefix: "constexpr in" --> · [Pipeline threshold](../include/sglang/text/gdn/gdn_chunk.cuh#L280)<!-- prefix: "*pipelined =" --> · [Pipeline size](../include/sglang/text/gdn/gdn_warp_mma.cuh#L190)<!-- prefix: "constexpr in" -->

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

### Native TP2 four-warp replay

One B200 replayed each rank's W4 and W5 layer-0 inputs separately, with 28 rotated CUDA-event samples per arm. Warm preparation-to-finish medians compare scalar eight-warp with paired four-warp recurrence.

| Fixture position | Rows | Rank 0 (ms) | Rank 1 (ms) |
| --- | ---: | ---: | ---: |
| W4 4,096 | 4,096 | 0.5892 → 0.4570 | 0.5892 → 0.4580 |
| W4 4,096 | 8,192 | 1.1732 → 0.9090 | 1.1737 → 0.9087 |
| W5 40,960 | 1,024 | 0.1547 → 0.1232 | 0.1544 → 0.1219 |
| W5 40,960 | 4,096 | 0.5868 → 0.4553 | 0.5869 → 0.4549 |
| W5 40,960 | 8,192 | 1.1701 → 0.9053 | 1.1696 → 0.9056 |

Cold and warm 1,024-, 4,096-, and 8,192-row runs matched FP8 codes, FP32 scales and carry, and raw BF16 history bitwise. Split serialization and restore, a 128-row suffix, and one BF16 decode matched. Both W5 shard seeds matched through position 40,960. Intermediate prepared QKV, gates, response, BF16 output, and state matched at 127, 128, 129, 193, 512, 1,004, and 1,024 rows. Logs and probe source are under `artifacts/tp2-resume/long-gdn/native/`.

An independent B200 comparison of the baseline and tracked operator matched 68 SHA-256 digests each for W4 and W5, covering both rank shards, restored 128-row prefill, and one decode. It verified a 96-block launch with 128 threads per block versus 48 blocks with 256 threads, plus ordered fallback below 128 rows, with missing, short, or unaligned scratch, and under injected compute capability 9.0. The receipt is `artifacts/tp2-resume/long-gdn-review/run-gpu4.log`.

The four-warp replay estimates 1.60 s for a W5 cohort at 48 layers and 8 cold plus 248 warm 4,096-row groups, assuming rank 0 recurrence savings stay on the critical path.

### Native TP2 staged value replay

One B200 replayed 8,192 W4/W5 rows in 1,024-row spans for both rank shards and banks 0 and 15. Operator medians fell 14.12–14.31% versus unstaged four-warp recurrence. FP8 codes, BF16 outputs and history, FP32 scales and carry, restore, and continuation matched bitwise. See `artifacts/tp2-resume/long-gdn/recurrent/` (`production-w4-clean-gpu3.log`, `production-w5-last-gpu3.log`, `production-w5-bank0-rank0-gpu3.log`). Full-model parity remains unmeasured.

With four staging warps, mean fenced full-model time fell 1.44–2.15 ms per 4,096-row W4 single-grant call. The two-B200 fixture measured two warm samples per arm at each of four positions. Independent B200 replay matched outputs, state, restore, and continuation bitwise across seven W4 and W5 cases. Serving makespan remains unmeasured. See `artifacts/tp2-resume/long-gdn/recurrent/e912/group-fenced-gpu45.log` and `artifacts/tp2-resume/gdn-producer4-review/receipt.txt`.
