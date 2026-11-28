# Wide long paged decode

This guide covers wide TP2 projections and draft repair. [Current target attention](singleton-gqa-decode.md#split-kv-target-attention)

## Responsibilities and sources of truth

The verifier owns rank-local projection dispatch and KV repair. [Packed projection caller](../src/runtime/execution/decoder_forward_layers.cpp#L258)<!-- prefix: "bool LayerBa" --> · [Former packed K0 source map](https://github.com/para1lel/sglang.cpp/blob/a5974ba4eb99bd8e8f890c2e7262a27dbc3aace1/docs/long-paged-decode.md#responsibilities-and-sources-of-truth)

## Runtime path

### Fused packed GQA

The former K0 path fused KV append, split attention and gating with FP64 head recovery. [Frozen runtime and staging](https://github.com/para1lel/sglang.cpp/blob/a5974ba4eb99bd8e8f890c2e7262a27dbc3aace1/docs/long-paged-decode.md#fused-packed-gqa) · [Operand range regression](lessons.md#l006--fp16-operand-loss-in-split-gqa)

### Packed FP8 projections

Packed eager and replay share bucket-width FP8 dispatch. Four-stage gate/up requires 5,120 inputs, two 17,408-wide outputs, and 16–32 rows. Bucket 16 uses 16 columns. Buckets 24 and 32 use 32. [Packed MLP](https://github.com/para1lel/sglang.cpp/blob/93648269d852e0fee78f7af4d1c3ffe4897aa361/src/text/decoder_graph_packed.cpp#L345)<!-- prefix: "check(text_f" --> · [Shape filter](../src/text/decode_proj_fp8.cu#L511)<!-- prefix: "bool use_mlp" --> · [Tile width](../src/text/decode_proj_fp8.cu#L731)<!-- prefix: "const int co" --> · [Pipeline dispatch](../src/text/decode_proj_fp8.cu#L747)<!-- prefix: "const bool s" -->

Four stages reduce TMEM and shared storage. Other ordinary packed FP8 shapes use eight stages. A clean SM100 decline uses per-target GEMV. [Pipeline details](https://github.com/para1lel/sglang.cpp/blob/a5974ba4eb99bd8e8f890c2e7262a27dbc3aace1/docs/long-paged-decode.md#packed-fp8-projections) · [Fallback](../src/text/fp8_projection.cu#L767)<!-- prefix: "const auto s" -->

#### Compact forty-row projections

B200 packs 40 rows into 48 MMA columns, reading 16+16+8 results through four stages. K-group order and BF16 rounding are preserved. TP1/TP2 gate/up and clean declines use 64 columns. CUDA errors propagate. [Result loads](../src/text/decode_proj_fp8.cu#L163)<!-- prefix: "constexpr in" --> · [Store](../src/text/decode_proj_fp8.cu#L450)<!-- prefix: "} else if co" --> · [Dispatch](../src/text/decode_proj_fp8.cu#L632)<!-- prefix: "if (count ==" -->

[Frozen W3 projection coverage, rank parity, and direct step timing](https://github.com/para1lel/sglang.cpp/blob/a5974ba4eb99bd8e8f890c2e7262a27dbc3aace1/docs/long-paged-decode.md#compact-forty-row-projections)

#### Eighty-row TP2 projections

For 80 rows on B200, native FP8 covers TP2 rank-local GDN QKV/Z, GQA Q/gate and K/V, attention output, MLP gate/up, and down. Each CTA uses 320 threads instead of 192. Two consumer warp groups read 40 columns each, preserving the 80-column MMA tile, four stages, K splits, FP32 order, and BF16 rounding. Clean declines and other M80 shapes use ordinary M80. CUDA errors propagate. [Shapes](../src/text/decode_proj_fp8.cu#L518)<!-- prefix: "bool tp2_m80" --> · [Consumers](../src/text/decode_proj_fp8.cu#L195)<!-- prefix: "__device__ v" --> · [Dispatch](../src/text/decode_proj_fp8.cu#L619)<!-- prefix: "if (tp2_m80_" -->

[Frozen W4 bitwise projection replay, FP32-order oracle, and fallback](https://github.com/para1lel/sglang.cpp/blob/a5974ba4eb99bd8e8f890c2e7262a27dbc3aace1/docs/long-paged-decode.md#eighty-row-tp2-projections)

#### Packed RMSNorm programmatic launch

The former packed K0 overlapped consumer weight staging with RMSNorm/FP8 preparation. [Frozen producer, graph edge, and consumer details](https://github.com/para1lel/sglang.cpp/blob/a5974ba4eb99bd8e8f890c2e7262a27dbc3aace1/docs/long-paged-decode.md#packed-rmsnorm-programmatic-launch) · [Launch attribute](../src/text/decode_proj_host.cu#L243)<!-- prefix: "attributes[0" -->

#### Ten-row TP2 verification projections

Two five-row K4 lanes share one B200 TP2 target forward. Packed projections require output slices in multiples of 128. Consumers read eight plus two live columns per 16-column MMA tile. K-group order and BF16 rounding stay fixed. Clean setup decline and standalone TP2 gate/up use the ordinary 16-column path. [Packed dispatch](../src/text/decode_proj_fp8.cu#L657)<!-- prefix: "if (count ==" --> · [Ten result columns](../src/text/decode_proj_fp8.cu#L137)<!-- prefix: "__device__ _" --> · [K-group accumulation](../src/text/decode_proj_fp8.cu#L164)<!-- prefix: "for (int ind" -->

At ten rows, TP2 can fuse 5,120-to-8,704 gate and up with SwiGLU quantization. It needs narrow scratch, contiguous FP8 inputs, one K split, and occupancy for all two-CTA clusters. Each pair rounds gate and up to BF16 in shared memory before BF16 SwiGLU and E4M3 quantization. Separate logits scratch holds E4M3 values and FP32 scales for down. Unsupported setup uses separate projections and quantization. Explicit CUDA errors propagate. [Verifier caller](../src/runtime/execution/decoder_forward_layers.cpp#L258)<!-- prefix: "bool LayerBa" --> · [Cluster launch](../src/text/decode_proj_fp8.cu#L678)<!-- prefix: "std::optiona" --> · [Occupancy guard](../src/text/decode_proj_host.cu#L183)<!-- prefix: "std::optiona" --> · [BF16 epilogue](../src/text/decode_proj_fp8.cu#L339)<!-- prefix: "__device__ v" --> · [Fallback](../src/runtime/execution/decoder_forward_layers.cpp#L448)<!-- prefix: "if (wide && " -->

Compare checkpoint shards across eager, capture, and clean fallback. Check full BF16 outputs and scratch guards. Time rotating and hot weights separately. [Frozen B200 operator medians and W1 call weighting](https://github.com/para1lel/sglang.cpp/blob/a5974ba4eb99bd8e8f890c2e7262a27dbc3aace1/docs/long-paged-decode.md#ten-row-tp2-verification-projections)

### Sixteen-row TP2 draft repair

Rank 0 batches TP2 draft KV repair for 16 lanes when its packed activation buffer fits all lanes. [Peer staging](serving/request-memory.md#peer-draft-repair) can overlap projection with scoring. Graph support can stay enabled for eager verification groups. A verifier capture excludes peer repair. Each repair depth needs 16 active rows with ordinary scalar rotary positions below `text_attention_max_tokens`. The effective rotary position uses the row position when no override is set. Partial depths, MRoPE, or a rotary position outside that limit use four-lane packets. [Commit guard](../src/decoder_batch.cpp#L733)<!-- prefix: "const bool e" --> · [Depth guard](../src/decoder_mtp.cpp#L692)<!-- prefix: "const bool g" --> · [Capture guard](../src/decoder_tp.cpp#L198)<!-- prefix: "auto* repair" -->

The wide repair groups input normalization, BF16 FC, FP8 K and V projections, rotary, and MXFP8 cache append. The 16-row SIMT projection dispatch is limited to the draft FC and KV dimensions and retains the vector reduction and rounding used by packet repair. [Projection selection](../src/text/draft_projection.cpp#L76)<!-- prefix: "const int wi" --> · [BF16 FC](../src/text_ops.cu#L112)<!-- prefix: "__global__ v" --> · [FP8 shape](../src/text/fp8_projection.cu#L623)<!-- prefix: "const bool e" -->

Match MXFP8 KV payloads, scales, and continuation proposals. Cover eager and graph, permuted and partial lanes, MRoPE, and missing scratch. [Frozen checkpoint replay and independent state receipts](https://github.com/para1lel/sglang.cpp/blob/a5974ba4eb99bd8e8f890c2e7262a27dbc3aace1/docs/long-paged-decode.md#sixteen-row-tp2-draft-repair)

### Legacy wide eager decode

The current [grouped split reader](singleton-gqa-decode.md#split-kv-target-attention) also supports [long draft dispatch](native-model-info.md#draft-attention). [Frozen compaction, split, and scratch layout](https://github.com/para1lel/sglang.cpp/blob/a5974ba4eb99bd8e8f890c2e7262a27dbc3aace1/docs/long-paged-decode.md#legacy-wide-eager-decode)

## Constraints and failure behavior

[Frozen eligibility, validation, and operand guards](https://github.com/para1lel/sglang.cpp/blob/a5974ba4eb99bd8e8f890c2e7262a27dbc3aace1/docs/long-paged-decode.md#constraints-and-failure-behavior) records the former K0 launcher and BF16 numerator rounding.

## Verification

Capture the RMSNorm and projection calls on a reserved B200 with checkpoint gains, weights, and scales. Inspect the dependency type and ports with `cudaGraphGetEdges` or CUDA graph DOT output. The SM100 control and early-trigger arms both have a programmatic edge. Test GDN and GQA at small and full bucket widths, with and without residual accumulation. Check that an unfused producer omits the early RMS trigger and forced GEMV uses ordinary edges. For gate and up, check that the early trigger requires the aligned 32-row residual path.

Compare complete BF16 residual, normalized, and projection outputs, E4M3 activations, and FP32 scales bitwise in eager and replay. Change the input between replays. Alternate arm order and time the pair with CUDA events under both warm and swept L2 conditions. Keep the sweep outside the timed interval. Collect node traces separately to measure overlap. Use event timings to measure latency.

For bounded decoder checks, use the same prompt and continuation IDs, lane bindings, and step budget in both arms. Compare complete BF16 scores for short and split-attention histories in eager and replay. Include a step with eager prefill alongside packed decode. Compare matched decode ticks separately from the mixed prefill step.

Use a Release SM100 build and one reserved B200. Run the [GQA verifier](development.md#p2-gqa-verification) and the [wide K0 runbook](../tests/batch/README.md#wide-k0-serving) for the retained routes.

Build and run both checks in the [long decode test runbook](../tests/decoder/README.md#wide-long-context-decode). The operator check uses shuffled pages, distinct lane positions, masked tails, and guarded scratch and output. It compares with ordered attention and FP64 on identical packed KV and checks recovery, invalid descriptors, and injected launch rejections. Run the [split FP64 regression](lessons.md#l006--fp16-operand-loss-in-split-gqa) for the operand-range controls. [Operator verifier](../tests/decoder/long_decode_check.cu#L398)<!-- prefix: "void run(Mod" --> · [Error cases](../tests/decoder/long_decode_check.cu#L200)<!-- prefix: "void check_e" -->

For a staging comparison, use prepared queries and packed KV from checkpoint captures. Include long batches at widths 24 and 32, full and padded masks, and short contexts near tile and page boundaries. Replay one graph per shape with two bindings that change cache pointers, page tables, and positions. Check complete BF16 output and inactive sentinels against eager and the fixed control. Time only split and combine with B200 CUDA events. Keep binding uploads and output readback outside the timed interval. These measurements cover the isolated attention operator.

For four-stage FP8, compare BF16 outputs and padding at widths 16, 24, and 32 in eager and graph. Cover widths 15, 17, and 33, a partial output tile, and misalignment. Use down projection as the split-K control. [Frozen projection and trace coverage results](https://github.com/para1lel/sglang.cpp/blob/a5974ba4eb99bd8e8f890c2e7262a27dbc3aace1/docs/long-paged-decode.md#verification)

The checkpoint route check exercises sparse slots, mixed context lengths, fixed-seed sampling, and slot reuse. It counts ordered split and combine launches for long lanes and checks graph capture and replay with a short lane. Use the optional BF16 capture and forced-token input to compare greedy choices on identical histories. [Route verifier](../tests/decoder/long_decode_routes.cpp#L815)<!-- prefix: "int main(int" -->

For the direct checkpoint control, admit all 16 prompts from historical W4 before advancing. Keep prompt IDs, decode histories, and sparse logical slots fixed across arms. Exercise active widths 16, 11, 5, 4, and 1. Trace the packed buckets, singleton route, and any legacy fallback. Compare each emitted ID and record decode step time separately from prefill. To measure numerical drift on identical histories, capture complete logits and force the same continuation IDs in both arms. Report top-1, KL, and teacher-forced negative log-likelihood deltas with the tested token count.

Reproduce old concurrent timing with the [frozen driver and workload](https://github.com/para1lel/sglang.cpp/blob/e0f45046af0f859806c531497e194318a23d9a75/benchmarks/concurrent/README.md#regenerate-and-measure) and [original launch controls and limits](https://github.com/para1lel/sglang.cpp/blob/a5974ba4eb99bd8e8f890c2e7262a27dbc3aace1/docs/long-paged-decode.md#verification). Use the [current agreement](performance.md#measurement) for acceptance.

Record prefill overlays as separate experiments. Use identical prefill code on both sides of an isolated long-decode comparison. The [Flash prefill contract](flash-prefill.md#constraints-and-failure-behavior) defines the selected 144-row floor.

### Packed route drift evidence

Historical packed runs diverged above width one. Fixed-input probes implicated FP8 projection reduction order and BF16 softmax-weight rounding. [Frozen drift details](https://github.com/para1lel/sglang.cpp/blob/6debe50edccdeda34ff307235ed38e74b6b8f2ee/docs/long-paged-decode.md#packed-route-drift-evidence)

The graph requirement is compare complete BF16 score rows bitwise across every eligible width and both low/high slot masks. Warmup must not consume a required mask. Full serving performance qualification remains open.
