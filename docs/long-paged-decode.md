# Wide long paged decode

## Responsibilities and sources of truth

| Part | Responsibility | Source |
| --- | --- | --- |
| Decoder batch | Mark K0 text decode lanes under the tensor-core policy. | [Lane eligibility](../src/decoder_batch.cpp#L671)<!-- prefix: "const bool p" --> |
| Packed input projection | Pair layer-input RMSNorm and FP8 quantization with the GDN or GQA input projection. | [Packed call](../src/text/decoder_graph_packed.cpp#L278)<!-- prefix: "void PackedD" --> · [Projection API](../src/text_ops.cu#L906)<!-- prefix: "cudaError_t " --> |
| Eager forward | Compact eligible attention reads and order KV append, attention, and gating on the batch stream. | [Forward dispatch](../src/decoder_forward.cpp#L304)<!-- prefix: "void attend_" --> |
| CUDA operator | Validate the group, select a split, merge partials, and recover individual heads. | [Launch](../include/sglang/text/mxfp8_gqa_decode.cuh#L380)<!-- prefix: "cudaError_t " --> · [Tensor-core split](../include/sglang/text/mxfp8_gqa_wide_tc.cuh#L178)<!-- prefix: "__global__ v" --> · [Paired split](../include/sglang/text/mxfp8_gqa_decode.cuh#L274)<!-- prefix: "__global__ v" --> · [Merge](../src/mxfp8_gqa.cu#L658)<!-- prefix: "__global__ v" --> · [Recovery](../include/sglang/text/mxfp8_gqa_decode.cuh#L332)<!-- prefix: "__global__ v" --> |

## Runtime path

Start the batch server with `--fp8-prefill tensor-core` to enable [packed K0 text decode](batch-graphs.md#runtime-path). The fused GQA step requires at least two eligible decode lanes, with a live position, KV cache, and page table per lane. The same operator runs in eager and replay. Negative positions mark inactive rows. See the [batch failure contract](batch-graphs.md#constraints-and-failure-behavior) for first-use packed scratch errors. [Packed dispatch](../src/text/decoder_graph_packed.cpp#L506)<!-- prefix: "bool run_pac" --> · [Fused API](../include/sglang/text/gqa_decode_step.h#L28)<!-- prefix: "cudaError_t " -->

### Fused packed GQA

The fused step prepares Q and K, appends the new packed KV, and runs attention and output gating on the batch stream. The split uses whole 64-token chunks, with up to 16 partials per lane and query head. It preserves the original BF16 Q and expands raw E4M3 operands exactly to BF16. QK and PV use BF16 MMA with FP32 accumulation. PV rounds the probability times the relative V scale to BF16. Range checks cover operands, scale updates, and the merge. A flagged head replays attention in FP64. The fast path can still lose precision through cancellation. See [the regression inputs and results](lessons.md#l006--fp16-operand-loss-in-split-gqa). [Fused step](../src/text/gqa_decode_prepare.cu#L77)<!-- prefix: "cudaError_t " --> · [Split selection](../include/sglang/text/gqa_decode_split.cuh#L41)<!-- prefix: "__host__ __d" --> · [Merge](../include/sglang/text/gqa_decode_split.cuh#L102)<!-- prefix: "__device__ i" -->

The split uses two shared-memory stages. Each of four warps copies and consumes its own 16-token slice of the chunk. The payload copies use 16-byte `cp.async.cg` and zero-fill tokens past the visible tail. The scale copies use 8-byte `cp.async.ca` to give each warp disjoint scale writes. [Stage count](../src/text/gqa_decode_split.cu#L18)<!-- prefix: "constexpr in" --> · [Copy ownership](../src/text/gqa_decode_split.cu#L134)<!-- prefix: "__device__ v" --> · [Payload copy](../src/text/gqa_decode_split.cu#L94)<!-- prefix: "asm volatile" --> · [Tail predicate](../src/text/gqa_decode_split.cu#L149)<!-- prefix: "const bool v" --> · [Scale copy](../src/text/gqa_decode_split.cu#L167)<!-- prefix: "asm volatile" -->

Warp zero loads BF16 query fragments and their range flags into shared memory. The same CTA barrier publishes the query and page indices. During scoring, each of the four warps reads one 32-channel query group at a time. [Page indices](../src/text/gqa_decode_split.cu#L455)<!-- prefix: "for (int ind" --> · [Query publication](../src/text/gqa_decode_split.cu#L463)<!-- prefix: "__syncthread" --> · [Query group read](../src/text/gqa_decode_split.cu#L267)<!-- prefix: "const auto w" -->

In the tile loop, lanes wait for their copy groups and synchronize within the warp before they refill one stage and consume the other. All lanes drain their copies and synchronize across the CTA before they reuse shared memory for the merge. A final CTA barrier publishes the warp results before the block writes split partials. [Tile wait](../src/text/gqa_decode_split.cu#L476)<!-- prefix: "wait_pending" --> · [Merge storage](../src/text/gqa_decode_split.cu#L489)<!-- prefix: "auto& merge " -->

### Packed FP8 projections

Packed eager and replay use the same FP8 projection dispatch and pass the bucket width as the row count. SM100 uses 8-, 16-, 32-, or 64-column activation tiles. The four-stage pipeline requires 5,120 input features, two targets with 17,408 output features each, and 16–32 rows. Packed MLP gate and up projections meet this shape at buckets 16, 24, and 32. The 16-row call uses a 16-column tile. Calls with 17–32 rows use a 32-column tile. [Packed MLP](../src/text/decoder_graph_packed.cpp#L345)<!-- prefix: "check(text_f" --> · [Shape filter](../src/text/decode_proj_fp8.cu#L203)<!-- prefix: "bool use_mlp" --> · [Tile width](../src/text/decode_proj_fp8.cu#L225)<!-- prefix: "const int co" --> · [Pipeline dispatch](../src/text/decode_proj_fp8.cu#L243)<!-- prefix: "const bool s" -->

The FP8 copy and consumer warps wait on the same MMA completion phase. The copy warp refills shared memory and signals copy readiness. Consumer warps read TMEM and release the accumulator slot. The MMA warp waits for copy readiness and slot release before reuse. The four-stage pipeline halves the TMEM allocation and lowers dynamic shared memory at the cost of fewer stages for overlapping copies with compute. Its shared-memory request fits two CTAs per SM on B200. The split planner still uses the 64-column kernel and eight-stage footprint, preserving the split-K reduction order and BF16 rounding. [Copy completion](../include/sglang/text/decode_proj_sm100.cuh#L177)<!-- prefix: "const auto* " --> · [Accumulator release](../src/text/decode_proj_fp8.cu#L101)<!-- prefix: "release_slot" --> · [Reuse guard](../include/sglang/text/decode_proj_sm100.cuh#L198)<!-- prefix: "wait_barrier" --> · [Split planning](../src/text/decode_proj_fp8.cu#L233)<!-- prefix: "auto status " --> · [Shared-memory request](../src/text/decode_proj_fp8.cu#L264)<!-- prefix: "const int sh" -->

Other eligible FP8 shapes, including the MLP down projection, use eight stages. BF16 retains separate completion signals. If the SM100 dispatcher declines the call, the entry point uses per-target GEMV. [Fallback](../src/text_ops.cu#L888)<!-- prefix: "const auto s" -->

#### Packed RMSNorm programmatic launch

Packed K0 uses one layer-input producer for GDN QKV and Z or for GQA Q/gate, K, and V. The first layer normalizes the embedding rows without a residual add. Later layers add the previous MLP output to the residual. [Layer input](../include/sglang/decoder_internal.h#L223)<!-- prefix: "const auto i" --> · [GDN targets](../src/text/decoder_graph_packed.cpp#L295)<!-- prefix: "const TextFp" --> · [GQA targets](../src/text/decoder_graph_packed.cpp#L332)<!-- prefix: "const TextFp" -->

Once the SM100 dispatcher prepares the projection, it requests the producer's early trigger before launching the consumer. The fused producer issues the trigger after input staging and CTA synchronization, before RMS reduction and output stores. This also applies to the fused variants without a residual or eight-lane quantization. If RMSNorm requires separate normalization and quantization, those launches use ordinary stream ordering and omit the early RMS trigger. If projection falls back to GEMV, the dispatcher requests ordinary production and uses ordinary projection edges. [Consumer dispatch](../src/text/decode_proj_fp8.cu#L268)<!-- prefix: "if (producer" --> · [Fused selection](../src/text_normalization.cu#L312)<!-- prefix: "cudaError_t " --> · [Trigger](../src/text_normalization.cu#L255)<!-- prefix: "asm volatile" --> · [GEMV producer call](../src/text_ops.cu#L881)<!-- prefix: "const auto s" -->

The SM100 launch requests programmatic stream serialization. Capture creates a programmatic kernel edge from the producer's programmatic port to consumer port zero. The trigger permits independent consumer work after all producer blocks reach the trigger or exit. The copy warp issues TMA copies of immutable weight tiles before its dependency wait. The consumer warps load BF16 weight scales before their wait. The waits still gate activation TMA copies and FP32 activation-scale loads until the producer completes. [Launch attribute](../src/text/decode_proj_host.cu#L189)<!-- prefix: "attributes[0" --> · [Weight tile copies](../include/sglang/text/decode_proj_sm100.cuh#L170)<!-- prefix: "const int pr" --> · [Weight scale loads](../src/text/decode_proj_fp8.cu#L138)<!-- prefix: "const auto* " -->

The post-attention RMSNorm that feeds MLP gate and up uses the same trigger point. It continues to require the 32-row bucket and the [aligned residual path](fp8-prefill-projections.md#fused-activation-quantization). [Packed selection](../src/text/decoder_graph_packed.cpp#L274)<!-- prefix: "RmsQuantizeL" --> · [Row guard](../src/text_normalization.cu#L327)<!-- prefix: "const bool e" -->

### Legacy wide eager decode

The legacy wide eager forward can still compact long attention reads at widths above four. That operator covers 32,768–262,144 visible tokens including the token being appended. [Legacy dispatch](../src/decoder_forward.cpp#L307)<!-- prefix: "const bool w" --> · [Position check](../include/sglang/decoder_cuda.cuh#L119)<!-- prefix: "inline const" --> · [Context ceiling](../include/sglang/text_attention.h#L9)<!-- prefix: "inline const" -->

Each layer appends the new KV before submitting that lane's grouped attention read. The tensor-core split requires at least five compacted reads and 16-byte-aligned K and V payloads for every lane in the group. The paired split handles groups of one through four or groups with an unaligned payload. Each split divides the visible sequence into 16 page-aligned partitions and clamps the final interval to the visible length. The paired split shares K and V across two query heads of one KV head. [Compaction and launch](../src/decoder_forward.cpp#L327)<!-- prefix: "if (lane.mod" --> · [Split selection](../include/sglang/text/mxfp8_gqa_decode.cuh#L386)<!-- prefix: "bool tensor " -->

Each tensor-core block serves one lane, one partition, and all six query heads of a KV head. Four warps process disjoint 16-token tiles. They decode visible E4M3 payloads and their E8M0 scales into BF16 and share K and V across the six heads. BF16 matrix instructions calculate QK scores and weighted V with FP32 accumulation. The softmax maximum, rescaling, and mass use FP32. Only the numerator multiplies BF16-rounded softmax weights by V. The block merges warp partials in FP32. [KV stage](../include/sglang/text/mxfp8_gqa_wide_tc.cuh#L106)<!-- prefix: "__device__ _" --> · [MMA](../include/sglang/text/mxfp8_gqa_wide_tc.cuh#L50)<!-- prefix: "__device__ _" --> · [Weight rounding](../include/sglang/text/mxfp8_gqa_wide_tc.cuh#L250)<!-- prefix: "const float " -->

Both splits write the same per-head FP32 partials. The common merge rescales the 16 partitions against the global maximum and rounds the quotient to BF16. The recovery launch runs before attention gating on the same stream. [Merge](../src/mxfp8_gqa.cu#L658)<!-- prefix: "__global__ v" --> · [Recovery](../include/sglang/text/mxfp8_gqa_decode.cuh#L332)<!-- prefix: "__global__ v" -->

Each long lane uses 399,360 bytes of its existing 496,640-byte logits allocation for partials. The alias requires 16-byte alignment and remains live until the attention launches consume it. The stream orders later scoring writes to that allocation. [Scratch alias](../src/decoder_forward.cpp#L248)<!-- prefix: "PagedDecodeB" --> · [Scratch size](../include/sglang/decoder_cuda.cuh#L104)<!-- prefix: "inline const" -->

## Constraints and failure behavior

Default `exact` policy, media, MTP, and prefill rows retain their legacy routes. Out-of-window lanes in the legacy wide forward use the row dispatch. The legacy long primitive accepts one through 32 descriptors, independently of the serving width requirement. [Eligibility](../src/decoder_batch.cpp#L671)<!-- prefix: "const bool p" --> · [Row dispatch](../src/decoder_forward.cpp#L335)<!-- prefix: "check(read_g" --> · [Primitive](../src/mxfp8_gqa.cu#L975)<!-- prefix: "cudaError_t " -->

The launcher checks the complete group before submission. An invalid count, null required pointer, unaligned scratch address, out-of-window position, or insufficient declared cache or page-table capacity returns `cudaErrorInvalidValue`. The caller supplies valid device page entries and enough scratch for each lane. The launcher stops at the first synchronous CUDA launch error. Earlier launches can still be in flight. The forward propagates the error through the batch execution path. [Validation and launch](../include/sglang/text/mxfp8_gqa_decode.cuh#L380)<!-- prefix: "cudaError_t " --> · [Cache coverage](../src/mxfp8_gqa.cu#L158)<!-- prefix: "bool cache_c" --> · [Forward error](../src/decoder_forward.cpp#L342)<!-- prefix: "check(status" -->

The tensor-core split flags a query head if an input is nonfinite, a BF16 subnormal, or has magnitude above `2^30`. Visible nonfinite K or V flags all six query heads of that KV group. A nonzero decoded K or V magnitude below `2^-126` or above `2^30` also flags the group. The paired split flags nonfinite visible scores or partials and visible V scale bytes above 160. Both routes flag invalid merged maxima, mass, numerators, or BF16 results. A flagged head replays the visible tokens in order in FP64 from the original packed KV. Other heads keep their split results. [Query guard](../include/sglang/text/mxfp8_gqa_wide_tc.cuh#L200)<!-- prefix: "const bool i" --> · [KV range guard](../include/sglang/text/mxfp8_gqa_wide_tc.cuh#L64)<!-- prefix: "const bool i" --> · [Merge guard](../src/mxfp8_gqa.cu#L697)<!-- prefix: "const bool i" --> · [Head replay](../src/mxfp8_gqa.cu#L708)<!-- prefix: "__device__ _" -->

Both splits change the reduction order. The tensor-core split also rounds numerator weights to BF16. Either choice can change BF16 attention output, logits, and generated IDs, including the greedy winner in a near tie.

## Verification

Capture the RMSNorm and projection calls on a reserved B200 with checkpoint gains, weights, and scales. Inspect the dependency type and ports with `cudaGraphGetEdges` or CUDA graph DOT output. The SM100 control and early-trigger arms both have a programmatic edge. Test GDN and GQA at small and full bucket widths, with and without residual accumulation. Check that an unfused producer omits the early RMS trigger and forced GEMV uses ordinary edges. For gate and up, check that the early trigger requires the aligned 32-row residual path.

Compare complete BF16 residual, normalized, and projection outputs, E4M3 activations, and FP32 scales bitwise in eager and replay. Change the input between replays. Alternate arm order and time the pair with CUDA events under both warm and swept L2 conditions. Keep the sweep outside the timed interval. Collect node traces separately to measure overlap. Use event timings to measure latency.

For bounded decoder checks, use the same prompt and continuation IDs, lane bindings, and step budget in both arms. Compare complete BF16 scores for short and split-attention histories in eager and replay. Include a step with eager prefill alongside packed decode. Compare matched decode ticks separately from the mixed prefill step.

Use a Release SM100 build and one reserved B200. Run the [GQA verifier](development.md#p2-gqa-verification) and the [wide K0 runbook](../tests/batch/README.md#wide-k0-serving) for the retained routes.

Build and run both checks in the [long decode test runbook](../tests/decoder/README.md#wide-long-context-decode). The operator check uses shuffled pages, distinct lane positions, masked tails, and guarded scratch and output. It compares with ordered attention and FP64 on identical packed KV and checks recovery, invalid descriptors, and injected launch rejections. Run the [split FP64 regression](lessons.md#l006--fp16-operand-loss-in-split-gqa) for the operand-range controls. [Operator verifier](../tests/decoder/long_decode_check.cu#L402)<!-- prefix: "void run(Mod" --> · [Error cases](../tests/decoder/long_decode_check.cu#L204)<!-- prefix: "void check_e" -->

For a staging comparison, use prepared queries and packed KV from checkpoint captures. Include long batches at widths 24 and 32, full and padded masks, and short contexts near tile and page boundaries. Replay one graph per shape with two bindings that change cache pointers, page tables, and positions. Check complete BF16 output and inactive sentinels against eager and the fixed control. Time only split and combine with B200 CUDA events. Keep binding uploads and output readback outside the timed interval. These measurements cover the isolated attention operator.

The B200 FP8 replay used four checkpoint layers and batches assembled from archived activation tiles. At widths 16, 24, and 32, gate and up graph launches took 9.1–12.2% less time than the eight-stage control. Complete BF16 outputs and padding sentinels matched across eager and graph runs. The down projection served as a split-K control. Shape guards covered widths 15, 17, and 33 and a partial output tile. The probe also checked a misaligned activation. Eligible gate and up launches account for 12.509 of 33.849 seconds (37%) of FP8 projection kernel activity in the archived W5 decode tail. This trace share measures coverage of the prior workload. The replay measured isolated projection latency. CTA residency and serving latency for the four-stage pipeline remain unmeasured.

The checkpoint route check exercises sparse slots, mixed context lengths, fixed-seed sampling, slot reuse, and eager and graph paths. Under the Candidate B tensor-core route, five active long lanes use the padded width-eight bucket. The width-one phases retain legacy attention. The `exact` control uses short prompts. Compare emitted IDs and use the optional BF16 capture and forced-token input to examine changed greedy choices on identical histories. [Route verifier](../tests/decoder/long_decode_routes.cpp#L404)<!-- prefix: "int main(int" -->

For the direct checkpoint control, admit all 16 W4 requests before advancing. Keep prompt IDs, decode histories, and sparse logical slots fixed across arms. Exercise active widths 16, 11, 5, 4, and 1. Trace the packed buckets, singleton route, and any legacy fallback. Compare each emitted ID and record decode step time separately from prefill. To measure numerical drift on identical histories, capture complete logits and force the same continuation IDs in both arms. Report top-1, KL, and teacher-forced negative log-likelihood deltas with the tested token count.

For serving, use the [frozen protocol](../benchmarks/concurrent/README.md#regenerate-and-measure) and the same client and server policy in both arms. Keep graph-enabled K0, capacity 32, a 32,768-token step cap, tensor-core prefill, and prefix caching disabled. W4 submits 16 prompts of 32,768 IDs and requests 1,024 output IDs each. W5 extends coverage to 32 prompts with 4,096 output IDs each. Complete two warmups and five measured cohorts for a valid row, under the original 90-second socket and 360-second cohort limits. Preserve partial IDs, completion counts, and deadline failures for invalid rows. Score throughput and latency gates only over complete valid cohorts.

Record prefill overlays as separate experiments. Use identical prefill code on both sides of an isolated long-decode comparison. The [Flash prefill contract](flash-prefill.md#constraints-and-failure-behavior) defines the selected 144-row floor.

### Candidate B evidence

The B200 probe compared Candidate B with HEAD `aa4ec04` on 32 prompts and 352 teacher-forced decode rows per width. All 128 prefill rows matched bitwise. KL is `KL(HEAD || Candidate B)`. NLL deltas are Candidate B minus HEAD in nats per token.

| Width | Exact decode rows | Greedy IDs matched | Mean KL | Mean NLL delta | 95% interval for NLL delta |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 352/352 | 352/352 | 0 | 0 | [0, 0] |
| 4 | 0/352 | 332/352 | 0.010905 | +0.014366 | [−0.003239, +0.033153] |
| 16 | 0/352 | 333/352 | 0.011003 | +0.015422 | [−0.001699, +0.032493] |
| 32 | 0/352 | 333/352 | 0.011003 | +0.015422 | [−0.001699, +0.032493] |

Peak packed KL was 0.419241. The packed NLL intervals include zero. A separate `exact` prefill control had width-four mean KL 0.016527 for Candidate B and 0.018013 for HEAD. The prompt-bootstrap 95% interval for that difference was [−0.004703, +0.001276].

Same-input FP64 probes and operator substitutions implicate both FP8 projection reduction order and BF16 rounding of scaled softmax weights before packed GQA's value multiply. Across 1,466,368 projection outputs, 192 differed and 181 of those were adjacent BF16 values. Across two checkpoint layers and four positions through 32,768, packed GQA relative L2 error against FP64 was 0.001665–0.001734. Legacy error was 0.001633–0.001711. Gate staging matched all 49,152 tested outputs from the packed ungated results. At width four, substituting legacy projections alone raised greedy agreement from 332 to 338 of 352. Legacy attention alone raised it to 337. Restoring the full legacy operator set recovered all 352 decode and 32 prefill rows in that ablation. The probes attribute measured drift within the sampled operators and histories. See `artifacts/resume-drift/report.md` for the fixed inputs and reproduction commands.

Separately, the final verifier compared **1,696 complete BF16 score rows bitwise** between same-input eager and graph runs. The sweep covered widths 2–32, seven graphs, and 88 completed replays. Each width replayed low and high slot masks, which coincide at width 32. The inherited sweep failed that check because warmup consumed the only low-mask step at width two. See `artifacts/resume-verifier/evidence/` for `final-check.log`, `packed-bucket-replays.log`, and `inherited-coverage-gap.log`.

The paired W3 diagnostic on one B200 used one warmup and three measured cohorts per arm. All 12,288 measured output IDs matched across arms. Candidate B measured 346.53 output IDs/s versus HEAD's 139.43. Candidate B p95 TTFT, TPOT, and E2E were 2,587.87 ms, 9.97 ms/ID, and 12,721.06 ms versus 2,679.32 ms, 26.18 ms/ID, and 29,427.85 ms. Full performance qualification under the frozen serving protocol remains open. See `artifacts/resume-drift/report.md`.

### Isolated tensor-core results

In the GPU5 prototype, direct W4 checkpoint decode took 169.254 ms per median step with the paired split and 105.867 ms with the tensor-core split at width 16. Width 11 fell from 118.674 to 73.230 ms and width five from 56.815 to 33.544 ms. The width-four and singleton routes stayed the same. All 127 generated IDs and graph masks matched across the tested widths.

On one W4 cohort with fixed histories, all 240 decode top-1 predictions matched. Across 59,596,800 decode logits, RMSE was 0.334726 and the largest absolute change was 10.648438. Mean teacher-forced negative log-likelihood changed by +0.000000014 nats. These are direct decode measurements and a short matched-history quality check. See `artifacts/takeover-wide-decode-next/handoff.md` for source identities, traces, and measurements.

### Combined 256-row Flash experiment

The earlier combined build used the 256-row Flash floor, landed GDN, and wide long decode. In W4, it completed all 112 HTTP requests and matched all 114,688 frozen greedy IDs. It missed all four external gates.

| W4 metric | Combined build | Frozen gate |
| --- | ---: | ---: |
| Median output IDs/s | 62.209 | > 358.723 |
| p95 TTFT (s) | 81.627 | ≤ 50.227 |
| p95 TPOT (ms) | 200.896 | ≤ 59.994 |
| p95 E2E (s) | 261.143 | ≤ 46.590 |

On W5, the same build completed zero of 32 requests on the first warmup. All 32 reached the 90-second socket timeout before the first ID. The client skipped measured cohorts, leaving the W5 throughput and latency gates unscored.

See `artifacts/takeover-w4-integrated/handoff.md` for the combined build identities and raw evidence.
