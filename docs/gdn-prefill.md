# Gated DeltaNet prefill

## Responsibilities and sources of truth

| Part | Responsibility | Source |
| --- | --- | --- |
| Decoder | Select the precision policy and provide tile scratch when the GDN layer qualifies. | [Policy](../src/decoder_prefill.cpp#L613)<!-- prefix: "const bool t" --> · [First-layer guard](../src/runtime/execution/decoder_prefill_tiles.cpp#L207)<!-- prefix: "const bool c" --> |
| GDN operator | Prepare rows, advance each layer's recurrent state, and finish output and convolution history. | [Prefill submission](../src/text_attention_gdn.cu#L557)<!-- prefix: "cudaError_t " --> |
| Verification | Check ordered parity, chunked numerical error, and causal continuation. | [Segmented probe](../tests/decoder/gdn_check.cu#L424)<!-- prefix: "void check_s" --> · [Chunked probe](../tests/decoder/gdn_check.cu#L551)<!-- prefix: "void check_c" --> · [Default route](../tests/decoder/gdn_routes.cpp#L66)<!-- prefix: "void check_d" --> · [Prefix fallback](../tests/decoder/gdn_routes.cpp#L104)<!-- prefix: "void check_p" --> · [Checkpoint probe](../tests/decoder/gdn_checkpoint.cpp#L101)<!-- prefix: "int main(int" --> |

## Runtime path

On SM100, each Tensor Core block projects both 48-channel B and A gates across 32 normalized BF16 rows and one 512-element input slice. A second kernel sums ten FP32 partials in slice order and rounds to BF16. The different accumulation order can change gate values and downstream scores. [Partial products](../src/text/prefill_gdn_gates_sm100.cu#L43)<!-- prefix: "__global__ _" --> · [Ordered reduction](../src/text/prefill_gdn_gates_sm100.cu#L105)<!-- prefix: "__global__ v" -->

The SIMT fallback shares normalized input loads and reuses gate weights across four rows. It preserves both FP32 accumulation chains, warp reduction, and BF16 rounding. [Fused SIMT projection](../src/text_decode_bandwidth.cu#L102)<!-- prefix: "__global__ v" --> · [Separate projection order](../src/text_ops.cu#L96)<!-- prefix: "for (std::si" -->

Exact prefill uses token-ordered recurrence. On eligible B200 text tiles, GDN prepares convolution, normalized Q and K, and gates. SM100 chunked recurrence prepares 32-row causal matrices. BF16 Tensor Core products with FP32 accumulation solve corrections and advance the carry. Carry projection uses BF16 operands between groups. Persistent carry stays FP32. The finish updates BF16 convolution history and supplies the [attention output projection](fp8-prefill-projections.md#attention-output-quantization). [Matrix preparation](../src/text/gdn_chunk.cu#L164)<!-- prefix: "__global__ v" --> · [Chunked recurrence](../src/text/gdn_chunk.cu#L605)<!-- prefix: "__global__ v" -->

Grouped SM100 convolution processes four rows per 128-thread block, retaining three raw BF16 history samples and ordered Q/K normalization. TP2 `GdnLocal` uses a 24×4 grid and four recurrence warps, and `GdnFull` uses 48×2 and eight warps. Three value heads share Q/K staging and FP32 products and solve separately. Optional staging adds four producer warps. [Matrix launch](../src/text/gdn_chunk.cu#L721)<!-- prefix: "constexpr in" --> · [Recurrence launch](../src/text/gdn_chunk.cu#L732)<!-- prefix: "return cudaL" --> · [Value staging](../src/text/gdn_chunk.cu#L350)<!-- prefix: "copy_async(t" -->

Eligible 2,048/4,096-row projection groups submit two/four 1,024-row tiles together. The layerwise path can group two or three grants. Each of at most four segments keeps its own row count, state, and scratch, including unequal grants. If any starts at zero, the first GDN layer uses per-grant recurrence throughout the group. Clean decline also uses per-grant dispatch. CUDA errors propagate. [projection groups](native-prefill-workspace.md#projection-pairs-for-mtp) · [Segment limit](../include/sglang/decoder_cuda.cuh#L156)<!-- prefix: "inline const" --> · [Decoder dispatch](../src/decoder_prefill.cpp#L449)<!-- prefix: "void prefill" --> · [Position guard](../src/runtime/execution/decoder_prefill_staged_gqa.cpp#L189)<!-- prefix: "grant.rows %" --> · [Grouped submission](../src/text_attention_gdn.cu#L843)<!-- prefix: "std::optiona" --> · [Grouped staging and matrix layout](https://github.com/para1lel/sglang.cpp/blob/6debe50edccdeda34ff307235ed38e74b6b8f2ee/docs/gdn-prefill.md#runtime-path)

Ordered fallback pairs adjacent value channels from 32 rows with 16-byte-aligned workspace. Other calls use the original kernel. Paired channels share Q, K, and gates. SM100 packs FP32 arithmetic with the same token, update, and warp reduction order. Other architectures use scalar arithmetic. [Paired kernel](../src/text/gdn_prefill.cu#L98)<!-- prefix: "__global__ v" --> · [Paired guard](../src/text/gdn_prefill.cu#L201)<!-- prefix: "const bool p" -->

Native TP2 can combine four contiguous 1,024-row text tiles in one 4,096-row grant at a tile-aligned start with Tensor Core FP8 and rank-local GDN. At position zero, bank zero keeps the ordered first tile and combines the remaining 3,072 rows. Other eligible banks and starts combine all 4,096. [Run detection](../src/runtime/execution/decoder_prefill_staged_gqa.cpp#L256)<!-- prefix: "while (index" --> · [Native gate](../src/runtime/execution/decoder_prefill_staged_gqa.cpp#L262)<!-- prefix: "if (activati" --> · [Bank zero](../src/runtime/execution/decoder_prefill_staged_gqa.cpp#L266)<!-- prefix: "const bool o" -->

### Grouped partial grants

A [partial-grant projection group](native-prefill-workspace.md#partial-grant-projection-groups) can share GDN launches at 4,096 rows on SM100. It needs eight 512-row grants or sixteen 256-row grants. The decoder submits four grants at a time, with separate convolution history, FP32 carry, and scratch for each grant. The groups use the existing prefill arena. [Shape guard](../src/runtime/execution/decoder_prefill_staged_gqa.cpp#L223)<!-- prefix: "bool is_full" --> · [Four-grant dispatch](../src/runtime/execution/decoder_prefill_staged_gqa.cpp#L235)<!-- prefix: "void prefill" -->

If any of the four starts at position zero, bank zero uses the existing per-grant path for those four. The decoder also uses that path when the provider declines before GDN submission. A returned CUDA error propagates. The decoder clears the CUDA last-error value if it matches the reported error. [Bank guard](../src/runtime/execution/decoder_prefill_staged_gqa.cpp#L189)<!-- prefix: "grant.rows %" --> · [Error handling](../src/runtime/execution/decoder_prefill_staged_gqa.cpp#L207)<!-- prefix: "if (status) " -->

## Former vendor prefill opt-in

Rank-local GDN prefill uses the native BF16 or FP8 path. [Frozen former opt-in and reference comparisons](https://github.com/para1lel/sglang.cpp/blob/a5974ba4eb99bd8e8f890c2e7262a27dbc3aace1/docs/gdn-prefill.md#vendor-prefill-opt-in)

## Grouped target verification

With target Tensor Cores, four eager five-row MTP lanes share GDN launches while retaining lane-local state. Compact 80/160-row groups submit one gate, convolution, recurrence and finish per layer for 16/32 lanes. The tentative bank keeps raw BF16 history. Compact finish rounds to BF16 before E4M3 quantization with an FP32 scale per 128 channels. Clean setup decline before the gate uses four-lane packets and separate quantization. Pending and submitted errors propagate, and terminal status takes precedence. [Dispatch](../src/runtime/execution/decoder_forward_attention.cpp#L546)<!-- prefix: "if (batch.pa" --> · [Grouped launch](../src/text/gdn_verify.cu#L317)<!-- prefix: "cudaError_t " --> · [Compact dispatch](../src/runtime/execution/decoder_forward_attention.cpp#L90)<!-- prefix: "bool run_com" --> · [Fused finish](../src/text/gdn_verify.cu#L276)<!-- prefix: "if constexpr" --> · [Packet fallback](../src/runtime/execution/decoder_forward_attention.cpp#L546)<!-- prefix: "if (batch.pa" --> · [Launch status](../src/text/gdn_verify.cu#L351)<!-- prefix: "cudaError_t " -->

Use the [batched target GDN check](../tests/decoder/README.md#batched-target-gdn) against scalar decode for zero through five consumed rows and a replacement. Compare compact against four-lane controls on identical verification inputs. Match BF16 scores, normalized output and raw history, FP32 carry and scales, and FP8 bytes. Check decline, error propagation, and reuse.

MTP commit-phase GDN prefix replay is dispatched as one batched address-table launch per rank per commit group instead of two launches per layer. Per-layer kernel arguments, order, and results are unchanged. Setting `SGLANG_DISABLE_GDN_REPLAY_BATCH` forces the exact per-layer reference dispatch on both ranks. [Batched replay](../src/runtime/execution/decoder_batch_state.cpp#L146)<!-- prefix: "TextGdnRepla" -->

## Constraints and failure behavior

The B/A fast paths require tensor-core prefill policy and two BF16 weight matrices of 48 rows by 5,120 columns. The SM100 projection accepts 512–4,096 rows, including a partial final 32-row block. It needs nonnull buffers, 16-byte-aligned weights, input, and scratch, and supported TMA setup. It borrows 3,840 scratch bytes per token row from the MLP gate buffer. Reduction consumes those bytes on the same stream before recurrence or the MLP reuses them. [Decoder dispatch](../src/runtime/execution/decoder_prefill_tiles.cpp#L175)<!-- prefix: "if (tensor_c" --> · [SM100 preflight](../src/text/prefill_gdn_gates_sm100.cu#L208)<!-- prefix: "std::optiona" -->

A clean SM100 setup decline tries the fused SIMT projection. That path accepts multiples of four from 16 through 65,535 rows with aligned weights and input. If it also declines, the decoder projects B and A separately in that order. CUDA errors propagate from either fused path. [SIMT preflight](../src/text_decode_bandwidth.cu#L569)<!-- prefix: "if (rows < 1" --> · [Descriptor errors](../src/text/prefill_gdn_gates_sm100.cu#L137)<!-- prefix: "std::optiona" --> · [Separate fallback](../src/runtime/execution/decoder_prefill_tiles.cpp#L197)<!-- prefix: "project(laye" -->

The first GDN layer uses ordered recurrence for a tile that starts at prompt position zero. Later GDN layers in that tile and all GDN layers in subsequent tiles may use chunked recurrence. A tile that resumes a saved prefix uses the absolute restored position. An earlier recurrence experiment on W2 found that reordering only the first GDN call could change later model scores. The guard preserves that call's update order.

Grouped input preparation requires at least 128 rows in multiples of four. Other calls prepare one row per block. Selection is independent of matrix scratch and recurrence. The first GDN layer at position zero can group preparation with ordered recurrence. Groups write disjoint rows and gates. The finish saves three raw BF16 samples per channel on the same stream. Ownership and workspace sizes stay the same. [Shape guard](../src/text_attention_gdn.cu#L573)<!-- prefix: "const bool g" --> · [History write](../src/text_attention_gdn.cu#L194)<!-- prefix: "final_histor" -->

Native chunked recurrence requires compute capability 10.0, at least 128 rows, and sufficient 32-byte-aligned matrix scratch borrowed from the MLP gate buffer. Ineligible calls use ordered recurrence. Single-row tails use the single-row path. CUDA errors propagate. [Scratch guard](../src/text/gdn_chunk.cu#L649)<!-- prefix: "bool eligibl" --> · [Configuration](../src/text/gdn_chunk.cu#L663)<!-- prefix: "cudaError_t " -->

Each 32-row group uses a 4,480-byte matrix record per value head, or 3,440,640 bytes for 1,024 TP2 rows. BF16 matrix row pitches are 32 globally and 40 in shared memory. Eligible `GdnLocal` calls with at least 1,024 rows use four staging warps with at least 77,568 opt-in shared bytes per block and the earlier staged variant at 68,864–77,567 bytes. Other eligible calls use unstaged chunked recurrence with at least 64,768 bytes. Below that, ordered recurrence applies. Grouped and TP1 calls keep their existing dispatch. [Capacity guard](../src/text/gdn_chunk.cu#L691)<!-- prefix: "if (shared_l" --> · [Shared bytes](../src/text/gdn_chunk.cu#L368)<!-- prefix: "constexpr in" --> · [Pipeline threshold](../src/text/gdn_chunk.cu#L697)<!-- prefix: "*pipelined =" --> · [Pipeline size](../src/text/gdn_chunk.cu#L375)<!-- prefix: "constexpr in" -->

Chunked recurrence can reorder arithmetic. BF16 output, FP32 carry, logits, and generated IDs can differ from ordered execution. Chunked GDN and [flash GQA](flash-prefill.md#runtime-path) qualify independently within the same prefill tile.

## Verification

On a reserved B200, build the segmented probe with `SGLANG_TEST_GDN_TILED` and wrap `cudaLaunchKernel` at link time. Run `segmented paired CAPTURE` on both the packed candidate and the scalar paired control. The probe starts with nonzero history and recurrent state and carries them through eight segments. It checks at least the 31-, 32-, and 33-row boundary and a two-byte workspace offset.

Require bitwise equality of BF16 output, FP32 carry, and BF16 history across the CUDA paths. Require exact history and apply the probe tolerances for output and carry against the scalar FP64 CPU oracle. Run `short paired CAPTURE` under memcheck, racecheck, initcheck, and synccheck. Require zero errors and hazards.

Follow the [CUDA check instructions](../tests/decoder/README.md#gdn-prefill) on a reserved B200. Use `gdn-check chunked` for the 127-, 128-, and 129-row dispatches and the 193-row tail. For eligible calls, check the preparation and recurrence launch dimensions given in [Runtime path](#runtime-path). Check scratch alignment, one-byte shortfall, guards, and causality across a group boundary. Compare against the independent FP64 oracle with the probe's output and carry error limits. Require exact BF16 convolution history and bitwise equality for ordered fallbacks.

Use `gdn-routes` to check the first tile and a tile restored from a prefix while another lane decodes. Expect one ordered and 47 chunked calls on the first eligible tile, and 48 chunked calls on the restored tile. Decode-only steps use no chunked GDN. Check lane retirement and reuse. Compare exact-policy IDs and scores across builds.

Run a 2,048-ID checkpoint prompt with eight fixed teacher IDs and `SGLANG_SERIAL_PREFILL_ROWS` unset. Require two 1,024-row tiles and, for ordered changes, match carry and history across all 48 GDN layers plus decode, full BF16 logits and winners.

Keep GQA, grants, positions and B/A dispatch fixed for grouped recurrence checks. Compare both 4,096-row shapes, smaller or unequal groups, and bank zero at zero and later starts against per-grant execution. Match BF16 output/history, FP32 carry, and positive-depth draft continuation bitwise where arithmetic is preserved. At 127/128/129 rows, perturb a later row and check causality. Trace submission grouping, clean decline, independent setup and launch errors without retry, and a fresh cohort after recovery. [workspace pressure checks](native-prefill-workspace.md#verification)

For B/A, compare fused SM100 and SIMT with an FP64 oracle at 511/512 rows, a partial 32-row block and 4,096 rows. Keep recurrence and attention fixed. Check fallback and errors and time projection plus reduction together. [Frozen B/A projection and checkpoint comparisons](https://github.com/para1lel/sglang.cpp/blob/a5974ba4eb99bd8e8f890c2e7262a27dbc3aace1/docs/gdn-prefill.md#verification) · [Expanded checkpoint and grouped probes](https://github.com/para1lel/sglang.cpp/blob/6debe50edccdeda34ff307235ed38e74b6b8f2ee/docs/gdn-prefill.md#verification)

For reordered arithmetic, follow the [operator qualification procedure](performance.md#operator-qualification). Compare logits and labeled-token negative log likelihood on identical prompt and teacher histories. Record greedy divergence separately. Trace the position-zero first-layer guard, eligible layers, short tails, exact, and positive-depth MTP. Use the [frozen serving protocols](performance.md) with matched tile schedules, graph settings, and warmups. To isolate GDN when flash is active, hold flash dispatch fixed across both arms.

For unchanged arithmetic, compare BF16 output, FP32 carry, and BF16 history bitwise from nonzero carry. Time isolated recurrence separately from serving. Report samples, variability, and peak device memory. [Frozen W6 and checkpoint diagnostics](https://github.com/para1lel/sglang.cpp/blob/4dd8c887c278c67744ed80e7feb2a825a92197a2/docs/gdn-prefill.md#verification)

### Native TP2 four-warp replay

Compare preparation through finish on both rank shards at 1,024, 4,096, and 8,192 rows and check the 127/128/129 dispatch boundary. Match FP8 codes, FP32 scales and carry, and raw BF16 history. Cover split restore, a 128-row suffix, and BF16 decode. Force missing, short, and unaligned scratch and a pre-SM100 device. [Frozen fixtures, geometry, medians, and projection](https://github.com/para1lel/sglang.cpp/blob/a5974ba4eb99bd8e8f890c2e7262a27dbc3aace1/docs/gdn-prefill.md#native-tp2-four-warp-replay)

### Native TP2 staged value replay

Compare FP8 codes, BF16 output and history, and FP32 scales and carry against unstaged four-warp recurrence. Cover W4/W5 layers, bank zero and later banks, restore, and continuation across both ranks. Time isolated recurrence and fenced full-model prefill separately. [Frozen operator and caller receipts](https://github.com/para1lel/sglang.cpp/blob/a5974ba4eb99bd8e8f890c2e7262a27dbc3aace1/docs/gdn-prefill.md#native-tp2-staged-value-replay)
