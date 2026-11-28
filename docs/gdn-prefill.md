# Gated DeltaNet prefill

## Responsibilities and sources of truth

The decoder selects precision policy and lends scratch. The operator prepares GDN rows, advances recurrence and writes output/history. Verification covers ordered parity, chunked error and continuation. [Policy](../src/decoder_prefill.cpp#L636)<!-- prefix: "const bool t" --> · [First-layer guard](../src/runtime/execution/decoder_prefill_tiles.cpp#L218)<!-- prefix: "const bool c" --> · [Prefill submission](../src/text_attention_gdn.cu#L296)<!-- prefix: "cudaError_t " --> · [Segmented probe](../tests/decoder/gdn_check.cu#L424)<!-- prefix: "void check_s" --> · [Chunked probe](../tests/decoder/gdn_check.cu#L551)<!-- prefix: "void check_c" --> · [Default route](../tests/decoder/gdn_routes.cpp#L66)<!-- prefix: "void check_d" --> · [Prefix fallback](../tests/decoder/gdn_routes.cpp#L104)<!-- prefix: "void check_p" -->

## Runtime path

SM100 B/A projects both 48-channel gates in blocks of 32 normalized BF16 rows by 512 input elements. A second kernel sums ten FP32 partials in slice order and rounds to BF16. This can change gates and scores. Fused SIMT shares input/weights across four rows and preserves each FP32 accumulation chain, warp reduction and BF16 rounding. [Partial products](../src/text/prefill_gdn_gates_sm100.cu#L43)<!-- prefix: "__global__ _" --> · [Ordered reduction](../src/text/prefill_gdn_gates_sm100.cu#L105)<!-- prefix: "__global__ v" --> · [Fused SIMT projection](../src/text_decode_bandwidth.cu#L102)<!-- prefix: "__global__ v" --> · [Separate projection order](../src/text_ops.cu#L96)<!-- prefix: "for (std::si" -->

Exact prefill retains token-ordered recurrence. Eligible B200 GDN prepares convolution, normalized Q/K and gates, then solves 32-row causal matrices using BF16 Tensor Cores with FP32 accumulation. Carry projection uses BF16 operands and persistent carry stays FP32. Finish writes BF16 history and feeds the [attention output projection](fp8-prefill-projections.md#attention-output-quantization). [Matrix preparation](../src/text/gdn_chunk.cu#L165)<!-- prefix: "__global__ v" --> · [Chunked recurrence](../src/text/gdn_chunk.cu#L675)<!-- prefix: "__global__ v" -->

| Native route | Shape and state |
| --- | --- |
| Grouped preparation and recurrence | Convolution handles four rows per 128 threads with three raw BF16 history samples and ordered Q/K norm. TP2 `GdnLocal` uses 24×4 with four recurrence warps; `GdnFull` uses 48×2 with eight. Three value heads share Q/K staging and FP32 products, then solve separately. Optional staging adds four producer warps. [Matrix launch](../src/text/gdn_chunk.cu#L928)<!-- prefix: "cudaLaunchKe" --> · [Recurrence launch](../src/text/gdn_chunk.cu#L936)<!-- prefix: "return cudaL" --> · [Value staging](../src/text/gdn_chunk.cu#L398)<!-- prefix: "__device__ _" --> |
| Projection-group recurrence | Two/four 1,024-row tiles in [projection groups](native-prefill-workspace.md#projection-pairs-for-mtp), or two/three layerwise grants. Up to four segments retain their own row count, state and scratch, including unequal grants. Any start at zero forces per-grant recurrence for the first GDN layer. Clean decline uses per-grant dispatch. CUDA errors propagate. [Segment limit](../include/sglang/decoder_cuda.cuh#L152)<!-- prefix: "inline const" --> · [Decoder dispatch](../src/decoder_prefill.cpp#L454)<!-- prefix: "void prefill" --> · [Position guard](../src/runtime/execution/decoder_prefill_staged_gqa.cpp#L203)<!-- prefix: "grant.rows %" --> · [Grouped submission](../src/text_attention_gdn.cu#L441)<!-- prefix: "std::optiona" --> · [Grouped staging and matrix layout](https://github.com/para1lel/sglang.cpp/blob/6debe50edccdeda34ff307235ed38e74b6b8f2ee/docs/gdn-prefill.md#runtime-path) |
| Ordered fallback | From 32 rows, 16-byte-aligned scratch enables pairing adjacent channels, sharing Q/K and gates. SM100 packed FP32 preserves token, update and reduction order. Other architectures use scalar arithmetic; other shapes use the original kernel. [Paired kernel](../src/text/gdn_prefill.cu#L470)<!-- prefix: "__global__ v" --> · [Paired guard](../src/text/gdn_prefill.cu#L579)<!-- prefix: "const bool p" --> |
| TP2 within one grant | Four contiguous 1,024-row tiles, tile-aligned text start, Tensor Core FP8 and rank-local GDN. Bank zero at position zero keeps the ordered first tile and groups the remaining 3,072 rows. Otherwise group all 4,096. [Run detection](../src/runtime/execution/decoder_prefill_staged_gqa.cpp#L276)<!-- prefix: "while (index" --> · [Native gate](../src/runtime/execution/decoder_prefill_staged_gqa.cpp#L285)<!-- prefix: "if (activati" --> · [Bank zero](../src/runtime/execution/decoder_prefill_staged_gqa.cpp#L289)<!-- prefix: "const bool o" --> |

### Grouped partial grants

SM100 [partial-grant projection group](native-prefill-workspace.md#partial-grant-projection-groups) shares GDN at 4,096 rows for 8×512 or 16×256 grants. Submit four grants at a time in the existing arena with separate history, FP32 carry and scratch. If any starts at zero, bank zero uses per-grant recurrence for those four, as on a clean pre-submission decline. CUDA errors propagate and matching last-error records clear. [Shape guard](../src/runtime/execution/decoder_prefill_staged_gqa.cpp#L246)<!-- prefix: "bool is_full" --> · [Four-grant dispatch](../src/runtime/execution/decoder_prefill_staged_gqa.cpp#L258)<!-- prefix: "void prefill" --> · [Bank guard](../src/runtime/execution/decoder_prefill_staged_gqa.cpp#L203)<!-- prefix: "grant.rows %" --> · [Error handling](../src/runtime/execution/decoder_prefill_staged_gqa.cpp#L230)<!-- prefix: "if (status) " -->

## Former vendor prefill opt-in

Rank-local GDN prefill uses the native BF16 or FP8 path. [Frozen former opt-in and reference comparisons](https://github.com/para1lel/sglang.cpp/blob/a5974ba4eb99bd8e8f890c2e7262a27dbc3aace1/docs/gdn-prefill.md#vendor-prefill-opt-in)

## Grouped target verification

Grouped target verification projects GDN inputs for each lane, then hands ordered rows to [ReplaySSM decode](gdn-decode.md#packed-eager-and-captured-verification). That path owns tentative history and commits each lane separately. Prefill builds the exact recurrent state for each needed prompt checkpoint. [Forward](../src/runtime/execution/decoder_forward_attention.cpp#L463)<!-- prefix: "void run_gdn" --> · [Capture](../src/runtime/execution/decoder_prefill_tiles.cpp#L493)<!-- prefix: "void capture" -->

Use the [batched target GDN check](../tests/decoder/README.md#batched-target-gdn) against scalar decode for zero through five consumed rows and a replacement. Compare compact against four-lane controls on identical verification inputs. Match BF16 scores, normalized output and raw history, FP32 carry and scales, and FP8 bytes. Check decline, error propagation, and reuse.

The [frozen dense verifier and compact check](https://github.com/para1lel/sglang.cpp/blob/fad939383840a545e03ae2d2fe17277b52403cf3/docs/gdn-prefill.md#grouped-target-verification) describes the earlier alternate-bank implementation and its test receipts.

## Constraints and failure behavior

B/A fast paths need Tensor Core prefill and two BF16 48×5,120 weight matrices. CUDA errors propagate from both fused paths.

| B/A projection | Eligibility and fallback |
| --- | --- |
| SM100 | 512–4,096 rows, including partial 32-row tails. Nonnull buffers, weights/input/scratch aligned to 16 B, and supported TMA. Borrow 3,840 B per row from MLP gate; consume on the same stream before reuse. [Decoder dispatch](../src/runtime/execution/decoder_prefill_tiles.cpp#L186)<!-- prefix: "if (tensor_c" --> · [SM100 preflight](../src/text/prefill_gdn_gates_sm100.cu#L197)<!-- prefix: "std::optiona" --> |
| Fused SIMT, then separate | On clean SM100 decline, try multiples of four from 16–65,535 rows with aligned weights/input. A further decline projects B then A separately. [SIMT preflight](../src/text_decode_bandwidth.cu#L569)<!-- prefix: "if (rows < 1" --> · [Descriptor errors](../src/text/prefill_gdn_gates_sm100.cu#L137)<!-- prefix: "std::optiona" --> · [Separate fallback](../src/runtime/execution/decoder_prefill_tiles.cpp#L207)<!-- prefix: "project(laye" --> |

At absolute position zero the first GDN layer uses ordered recurrence. Later layers and tiles may use chunked recurrence. Resume uses the restored absolute position. [Frozen first-call W2 observation](https://github.com/para1lel/sglang.cpp/blob/57ac5fbbdf20a5a80c5bc079b49073a44c06680f/docs/gdn-prefill.md#constraints-and-failure-behavior)

| Recurrence stage | Eligibility and resources |
| --- | --- |
| Grouped preparation | ≥128 rows, multiple of four. Otherwise prepare one row/block. Independent of recurrence and matrix scratch, including ordered first layer at zero. Groups write disjoint rows and gates. Same-stream finish saves three raw BF16 samples per channel. Ownership and workspace sizes stay fixed. [Shape guard](../src/text/gdn_prefill.cu#L528)<!-- prefix: "const bool g" --> · [History write](../src/text/gdn_prefill.cu#L161)<!-- prefix: "__device__ v" --> |
| Native chunked | Capability 10.0, ≥128 rows and sufficient 32-byte-aligned matrix scratch in MLP gate. Use ordered on decline and single-row for tails. CUDA errors propagate. Per 32 rows, 4,480 B per value head (3,440,640 B at 1,024 TP2 rows). BF16 pitches are 32 globally and 40 in shared. [Scratch guard](../src/text/gdn_chunk.cu#L748)<!-- prefix: "bool eligibl" --> · [Configuration](../src/text/gdn_chunk.cu#L802)<!-- prefix: "cudaError_t " --> |
| TP2 value staging | Eligible ungrouped `GdnLocal` with ≥1,024 rows uses three frames at ≥111,232 opt-in shared B per block and two at 77,568–111,231 B, both with four producer warps. At 68,864–77,567 B it uses earlier staging. The third frame changes prefetch and shared storage while preserving arithmetic and global scratch. Other eligible calls use unstaged at ≥64,768 B, otherwise ordered. Grouped and TP1 dispatch is unchanged. [Capacity guard](../src/text/gdn_chunk.cu#L781)<!-- prefix: "if (shared_l" --> · [Frame selection](../src/text/gdn_chunk.cu#L789)<!-- prefix: "if (shared_l" --> · [Dynamic frame bytes](../src/text/gdn_chunk.cu#L368-L369)<!-- prefix: "constexpr in" --> · [Static plus dynamic bytes](../src/text/gdn_chunk.cu#L375-L376)<!-- prefix: "constexpr in" --> |

Chunked recurrence can reorder arithmetic. BF16 output, FP32 carry, logits, and generated IDs can differ from ordered execution. Chunked GDN and [flash GQA](flash-prefill.md#runtime-path) qualify independently within the same prefill tile.

## Verification

On a reserved B200, build the segmented probe with `SGLANG_TEST_GDN_TILED` and wrap `cudaLaunchKernel` at link time. Run `segmented paired CAPTURE` on both the packed candidate and the scalar paired control. The probe starts with nonzero history and recurrent state and carries them through eight segments. It checks at least the 31-, 32-, and 33-row boundary and a two-byte workspace offset.

Require bitwise equality of BF16 output, FP32 carry, and BF16 history across the CUDA paths. Require exact history and apply the probe tolerances for output and carry against the scalar FP64 CPU oracle. Run `short paired CAPTURE` under memcheck, racecheck, initcheck, and synccheck. Require zero errors and hazards.

Follow the [CUDA check instructions](../tests/decoder/README.md#gdn-prefill) on a reserved B200. Use `gdn-check chunked` for the 127-, 128-, and 129-row dispatches and the 193-row tail. For eligible calls, check the preparation and recurrence launch dimensions given in [Runtime path](#runtime-path). Check scratch alignment, one-byte shortfall, guards, and causality across a group boundary. Compare against the independent FP64 oracle with the probe's output and carry error limits. Require exact BF16 convolution history and bitwise equality for ordered fallbacks.

Use `gdn-routes` to check the first tile and a tile restored from a prefix while another lane decodes. Expect one ordered and 47 chunked calls on the first eligible tile, and 48 chunked calls on the restored tile. Decode-only steps use no chunked GDN. Check lane retirement and reuse. Compare exact-policy IDs and scores across builds.

Use the [LMCache device state and graph checks](../tests/prefix/README.md#device-state-and-graph-controls) to check checkpoint payloads, split GQA/GDN coverage, and continuation scores. The [archived teacher-forced probe](https://github.com/para1lel/sglang.cpp/blob/8b4e3f62e0a7a053ffdefe85da99a841fabeb5d3/tests/decoder/gdn_checkpoint.cpp#L67-L112) used two 1,024-row prefill tiles and eight fixed teacher IDs with `SGLANG_SERIAL_PREFILL_ROWS` unset. It captured all 48 GDN layers' carry and history after each tile and the decode sequence, plus full BF16 logits and winners.

Keep GQA, grants, positions and B/A dispatch fixed for grouped recurrence checks. Compare both 4,096-row shapes, smaller or unequal groups, and bank zero at zero and later starts against per-grant execution. Match BF16 output/history, FP32 carry, and positive-depth draft continuation bitwise where arithmetic is preserved. At 127/128/129 rows, perturb a later row and check causality. Trace submission grouping, clean decline, independent setup and launch errors without retry, and a fresh cohort after recovery. [workspace pressure checks](native-prefill-workspace.md#verification)

For B/A, compare fused SM100 and SIMT with an FP64 oracle at 511/512 rows, a partial 32-row block and 4,096 rows. Keep recurrence and attention fixed. Check fallback and errors and time projection plus reduction together. [Frozen B/A projection and checkpoint comparisons](https://github.com/para1lel/sglang.cpp/blob/a5974ba4eb99bd8e8f890c2e7262a27dbc3aace1/docs/gdn-prefill.md#verification) · [Expanded checkpoint and grouped probes](https://github.com/para1lel/sglang.cpp/blob/6debe50edccdeda34ff307235ed38e74b6b8f2ee/docs/gdn-prefill.md#verification)

For reordered arithmetic, follow the [operator qualification procedure](performance.md#operator-qualification). Compare logits and labeled-token negative log likelihood on identical prompt and teacher histories. Record greedy divergence separately. Trace the position-zero first-layer guard, eligible layers, short tails, exact, and positive-depth MTP. Use the [frozen serving protocols](performance.md) with matched tile schedules, graph settings, and warmups. To isolate GDN when flash is active, hold flash dispatch fixed across both arms.

For unchanged arithmetic, compare BF16 output, FP32 carry, and BF16 history bitwise from nonzero carry. Time isolated recurrence separately from serving. Report samples, variability, and peak device memory. [Frozen W6 and checkpoint diagnostics](https://github.com/para1lel/sglang.cpp/blob/4dd8c887c278c67744ed80e7feb2a825a92197a2/docs/gdn-prefill.md#verification)

### Native TP2 four-warp replay

Compare preparation through finish on both rank shards at 1,024, 4,096, and 8,192 rows and check the 127/128/129 dispatch boundary. Match FP8 codes, FP32 scales and carry, and raw BF16 history. Cover split restore, a 128-row suffix, and BF16 decode. Force missing, short, and unaligned scratch and a pre-SM100 device. [Frozen fixtures, geometry, medians, and projection](https://github.com/para1lel/sglang.cpp/blob/a5974ba4eb99bd8e8f890c2e7262a27dbc3aace1/docs/gdn-prefill.md#native-tp2-four-warp-replay)

### Native TP2 staged value replay

Compare FP8 codes, BF16 output and history, and FP32 scales and carry against unstaged four-warp recurrence. Cover W4/W5 layers, bank zero and later banks, restore, and continuation across both ranks. Time isolated recurrence and fenced full-model prefill separately. [Frozen operator and caller receipts](https://github.com/para1lel/sglang.cpp/blob/a5974ba4eb99bd8e8f890c2e7262a27dbc3aace1/docs/gdn-prefill.md#native-tp2-staged-value-replay)

For three frames, compare the original and candidate with fixed inputs, states and grants. Require bitwise BF16 output and history, FP32 carry and scales, and E4M3 codes in eager and graph replay, including changed inputs at stable addresses. Wrap CUDA device queries, kernel configuration and submissions. Check the 24×4 recurrence grid, dynamic and total shared bytes, 1,023/1,024 rows, and both sides of each capacity threshold. Exercise old devices, scratch declines, and injected query, attribute, matrix and recurrence launch failures. Clean eligibility decline uses the documented fallback. CUDA errors propagate without retry.
