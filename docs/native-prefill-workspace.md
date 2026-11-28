# Native prefill workspace

## Responsibilities and sources of truth

The decoder lends arena spans to singleton and grouped GQA. Attention dispatch selects staged, raw grouped, or per-lane work. The paired launcher checks eligibility and submits on the workspace stream. [Tile layout](../src/runtime/execution/decoder_prefill_tiles.cpp#L72)<!-- prefix: "TileBuffers:" --> · [Scratch view](../src/runtime/execution/decoder_prefill_tiles.cpp#L97)<!-- prefix: "PairedPrefil" --> · [Singleton selection](../src/text/gqa/ordered_prefill.cu#L301)<!-- prefix: "std::optiona" --> · [Grouped dispatch](../src/decoder_prefill.cpp#L401)<!-- prefix: "void prefill" --> · [Staged launch](../src/text/gqa/tmem_staged.cu#L450)<!-- prefix: "std::optiona" -->

## Runtime path

B200 lends projection workspace to eligible eager 1,024-row text tiles and [partial-grant subgroups](ordered-gqa-prefill.md#grouped-partial-grants). A 1,024-row tile at start zero uses raw paired attention. [Attention dispatch](flash-prefill.md#runtime-path) · [Singleton bound](../src/text/gqa/tmem_staged.cu#L545)<!-- prefix: "if (position" -->

| Scratch | Physical span (inclusive start, exclusive end) | One tile capacity | Contents at start 31,744 | Four-tile capacity | Subgroup maximum contents |
| --- | --- | ---: | ---: | ---: | ---: |
| BF16 K | `mlp_gate` through `mlp_up`, bounded by `q_gate` | 68 MiB | 64 MiB | 272 MiB | 256 MiB |
| BF16 V and invalid-token indices | `gdn_qkv`, `gdn_z`, `gdn_b`, `gdn_a` and GDN scratch, bounded by `activation.values` | 64.1875 MiB | 64 MiB + 4 KiB | 256.75 MiB | 256 MiB + 16 KiB |

Each disjoint span allows 127 bytes of 128-byte alignment padding. A [staged subgroup](flash-prefill.md#runtime-path) covers up to 131,072 visible tokens total. Partial groups round each span to whole pages and assign lane offsets. Both spans must fit the combined page count. Subgroups reuse them on one stream without device allocation. GQA keeps exclusive use through the last read, then MLP/GDN can overwrite them. Allocation release follows on the same stream. [Arena allocation](../src/runtime/execution/decoder_prefill_workspace.cpp#L66)<!-- prefix: "PrefillWorks" --> · [Scratch sizing](../src/text/gqa/tmem_staged.cu#L504)<!-- prefix: "const auto p" --> · [Layer loop](../src/decoder_prefill.cpp#L438)<!-- prefix: "run_layers(" --> · [Release](../src/runtime/execution/decoder_prefill_workspace.cpp#L214)<!-- prefix: "cudaError_t " -->

Predecode expands paged MXFP8 to BF16 and zero-fills partial-group lanes past their visible end. See [the paired kernel](flash-prefill.md#long-tile-tmem-pipeline) for separate K/V TMA producers and alternating stages. [Predecode](../src/text/gqa/tmem_staged_kernels.cu#L53)<!-- prefix: "__global__ v" --> · [Producers](../src/text/gqa/tmem_staged_kernels.cu#L141)<!-- prefix: "__device__ v" -->

The [CUDA 13.4 stream ordering guide](https://docs.nvidia.com/cuda/archive/13.4.3/cuda-programming-guide/02-basics/asynchronous-execution.html#cuda-stream-ordering) defines the execution order. The [CUTLASS SM100 MLA change](https://github.com/NVIDIA/cutlass/pull/2472) illustrates a two-CTA asynchronous attention pipeline. The native route uses its own MXFP8 layout and arithmetic.

### Dedicated long-context staging

The prefill workspace takes one general-arena lease for row scratch and optional BF16 K/V staging. The staging planes are disjoint from each other and from the row scratch. Both start on 128-byte boundaries. The V span ends with one `int` per page and local KV head for invalid-token metadata. [Layout and lease](../src/runtime/execution/decoder_prefill_workspace.cpp#L82)<!-- prefix: "const auto b" --> · [Plane views](../src/runtime/execution/decoder_prefill_workspace.cpp#L114)<!-- prefix: "auto* keys =" -->

Cap visibility at 262,144 and round up to 128-token pages. At that limit, TP1's four local KV heads require 512 MiB per BF16 plane plus 32 KiB of metadata. TP2's two local KV heads require 256 MiB per plane plus 16 KiB of metadata per rank. These sizes exclude row scratch and its alignment padding. [Page and head sizing](../src/runtime/execution/decoder_prefill_workspace.cpp#L85)<!-- prefix: "const auto v" -->

On clean allocation pressure the workspace retries with row scratch only. TP2 uses staging only when both ranks have the planes. The arena releases the combined lease on its owning stream after all users finish. [Base retry](../src/runtime/execution/decoder_prefill_workspace.cpp#L97)<!-- prefix: "} catch (con" --> · [Both-rank gate](../src/runtime/execution/decoder_prefill_tiles.cpp#L304)<!-- prefix: "PrefillGqaSt" --> · [Stream-ordered release](../src/runtime/execution/decoder_prefill_workspace.cpp#L214)<!-- prefix: "cudaError_t " -->

## Constraints and failure behavior

`workspace_bytes` includes active staging spans leased from the shared [device arena](serving/request-memory.md). The backing allocation stays committed for later leases. [Workspace accounting](../src/runtime/execution/decoder_prefill_workspace.cpp#L120)<!-- prefix: "std::size_t " -->

| Condition | Outcome |
| --- | --- |
| Arena exhaustion | The workspace retries base scratch when optional staging cannot fit. Ordinary prefill tries 128 rows on a large decline, then goes row-wise if scratch remains unavailable. Single rows use row-wise. Other allocation errors propagate. [Base retry](../src/runtime/execution/decoder_prefill_workspace.cpp#L97)<!-- prefix: "} catch (con" --> · [Ordinary retry](../src/decoder_prefill.cpp#L678)<!-- prefix: "if (!workspa" --> |
| Release event failure | The arena fences the owning stream before returning the span. Explicit workspace close checks the release status. [Arena release](../src/runtime/backend/device_arena.cpp#L460)<!-- prefix: "cudaError_t " --> · [Workspace close](../src/runtime/execution/decoder_prefill_workspace.cpp#L224)<!-- prefix: "void Prefill" --> |
| Missing/undersized scratch, capture or optional setup rejection | Decline before staging (including resource, tensor-map and descriptor OOM setup rejection). Batch requires an explicit current-context stream and disjoint output/scratch/live ranges in each subgroup. [Capture check](../src/text/gqa/tmem_staged.cu#L519)<!-- prefix: "const auto c" --> · [Stream check](../src/text/gqa/tmem_staged.cu#L216)<!-- prefix: "std::optiona" --> · [Range check](../src/text/gqa/tmem_staged.cu#L531)<!-- prefix: "if (!valid_s" --> · [Tensor maps](../src/text/gqa/tmem_staged.cu#L399)<!-- prefix: "const auto e" --> |
| Recognized staged launch rejection | A clean staged decline uses the eligible raw or lane-local route. Already submitted predecode orders fallback and reuse on the same stream. Unexpected and pending submission errors propagate. [Submission handling](../src/text/gqa/tmem_staged.cu#L245)<!-- prefix: "std::optiona" --> · [Full-tile fallback](../src/runtime/execution/decoder_prefill_staged_gqa.cpp#L329)<!-- prefix: "const int vi" --> |
| Paired graph capture | Record raw with maps by value and no staging scratch. Keep cache, page table, queries and output valid through replay. [Raw launch](../src/text/mxfp8_gqa_prefill_tmem.cu#L710)<!-- prefix: "std::optiona" --> · [CUDA graph capture](https://docs.nvidia.com/cuda/archive/13.4.2/cuda-programming-guide/04-special-topics/cuda-graphs.html#stream-capture) |

## Multi-request text prefill

The [batch planner](batch-graphs.md#runtime-path) sends generation and scoring to one prefill executor, separate from decode. Compatible work groups into tiles of at most 1,024 rows. Order, KV, recurrence, rotary positions, media, scores and outputs stay request-local. On [Cache restore](prefix-cache.md#checkpoint-compatibility-and-resume), fully covered GQA tiles project only queries and gates. Others append only uncovered KV. [Attention dispatch](flash-prefill.md#runtime-path) · [Step dispatch](../src/decoder_batch.cpp#L496)<!-- prefix: "if (max_rows" --> · [Grant assembly](../src/decoder_batch.cpp#L527)<!-- prefix: "void Decoder" --> · [Cached GQA append](../src/runtime/execution/decoder_prefill_tiles.cpp#L265)<!-- prefix: "void Chunk::" -->

### Projection pairs for MTP

The planner visits the first unconsumed grant. It first tries the eligible [resident causal preference](#causal-segments-within-one-grant), then a ranked cold pair, distinct full tiles, three or four causal tiles within one grant, or compatible terminal partials. Each tile carries its source grant and offset. Each lane keeps its own KV, recurrence, absolute position and draft state. [Cursor](../src/runtime/execution/decoder_prefill_plan.cpp#L158)<!-- prefix: "bool next(Pr" -->

| Group per tile offset | Eligibility and example | Active workspace |
| --- | --- | ---: |
| 4,096 rows | Up to four distinct full tiles, two cold TP2 grants contributing two tiles each, or four causal tiles from one grant. | 1,022,230,528 B |
| 2,048 rows | Two independent 1,024-row tiles. The planner can choose a nonadjacent matching pair; eligibility is local to the tiles available for the step. | 511,115,264 B |
| 1,024 rows | Ordinary tile. | 255,557,632 B |

The scheduler sizes one reusable workspace for the largest planned group or ordinary tile. It tries a larger lease when the resident-first plan needs more rows, and uses that plan only with the required resident planes. On a clean decline it tries the ordinary group plan with its original workspace size. If that lease also declines, it replans with a 2,048-row group cap or returns to ordinary prefill. TP2 acquires both rank-local leases before grouped submission and releases provisional leases before retrying. [Sizing and fallback](../src/runtime/execution/decoder_prefill_plan.cpp#L328)<!-- prefix: "const auto p" --> · [Plan retry](../src/runtime/execution/decoder_prefill_plan.cpp#L341)<!-- prefix: "for (;;) {" --> · [Ranked lease](../src/decoder_prefill.cpp#L640)<!-- prefix: "[&](int capa" -->

[Grouped GDN](gdn-prefill.md#runtime-path) uses disjoint request spans. Four aligned tiles can use [batched staged GQA](flash-prefill.md#runtime-path). Mixed groups can choose nonadjacent long pairs and retain per-lane dispatch for uncovered tiles. [Layerwise plan](../src/decoder_prefill.cpp#L122)<!-- prefix: "PrefillGroup" --> · [Attention partners](../src/runtime/execution/decoder_prefill_staged_gqa.cpp#L352)<!-- prefix: "std::optiona" -->

### Paired 2,048-token TP2 prompts

B200 TP2 can group the first two 1,024-row tiles from each of two distinct cold text grants starting at zero. Each grant may continue beyond the selected 2,048 rows. Attention and recurrence advance in request order. A clean peer allocation decline releases the provisional primary workspace before a narrower retry. Unexpected CUDA errors propagate. [Cold pair selection](../src/runtime/execution/decoder_prefill_plan.cpp#L57)<!-- prefix: "bool cold_pa" --> · [Ranked schedule](../src/decoder_prefill.cpp#L609)<!-- prefix: "bool try_pre" --> · [Clean allocation fallback](../src/runtime/execution/decoder_prefill_plan.cpp#L329)<!-- prefix: "workspace.em" -->

### Seed scratch

TP2 seed projects and appends the rank-local KV shard on each rank and validates equal starting draft prefixes. Draft rows use the shifted prompt tokens and matched rank-local seed buffers. Verify both ranks’ seeds at the prefill/first-target boundary. [Ranked draft seed](../src/text/draft_projection.cpp#L575)<!-- prefix: "void MtpDraf" -->

### Partial-grant projection groups

After full-tile and causal options, the planner can group compatible terminal partials from distinct grants in source order. Each partial fits within one tile, starts at the grant’s unconsumed offset, and has 256–1,024 rows on TP1 or 128–1,024 on TP2. The total fits the selected group cap, up to 4,096 rows. [Partial cursor](../src/runtime/execution/decoder_prefill_plan.cpp#L178)<!-- prefix: "add(step, fi" -->

Groups share projections and MLP. Uniform 8×512 or 16×256 groups can share [RMSNorm and GQA postgate preparation](fp8-prefill-projections.md#attention-output-quantization). [Grouped staged and ordered GQA](ordered-gqa-prefill.md#grouped-partial-grants) and [grouped GDN](gdn-prefill.md#grouped-partial-grants) qualify separately, with per-lane fallback. [Layerwise dispatch](../src/decoder_prefill.cpp#L401)<!-- prefix: "void prefill" -->

Eligible groups can use [dedicated staging](#dedicated-long-context-staging). If the base workspace cannot be leased for the selected plan, the executor declines before grouped submission. [Workspace selection](../src/runtime/execution/decoder_prefill_plan.cpp#L325)<!-- prefix: "int max_rows" -->

With `--trace-batches`, `layerwise_prefill_groups` counts groups, `rows` counts combined rows and `workspace_bytes` reports the step’s largest active prefill workspace. [Batch trace](../apps/server/detail/batch_trace.cpp#L41)<!-- prefix: "if (stats.la" -->

### Causal segments within one grant

The resident preference gives a causal group priority over pairing when the first unconsumed text grant has three or four complete 1,024-row tiles and starts on a 1,024-token boundary. The request must have at least 32,768 cached GQA tokens, the group must start at or beyond 32,768 and end by 262,144, and the lease must provide the dedicated planes on both TP2 ranks. Otherwise the ordinary order applies: pair eligible grants first, then try the causal group. GDN advances in causal order. A clean staged decline reads the prepared segments individually. Ordinary prefill resumes at the first unprocessed row. TP2 mirrors the source grant and offset. [Resident eligibility](../src/runtime/execution/decoder_prefill_plan.cpp#L149)<!-- prefix: "bool prefer_" --> · [Selection order](../src/runtime/execution/decoder_prefill_plan.cpp#L158)<!-- prefix: "bool next(Pr" --> · [Causal GQA](../src/runtime/execution/decoder_prefill_staged_gqa.cpp#L87)<!-- prefix: "bool prefill" --> · [Ranked schedule](../src/decoder_prefill.cpp#L609)<!-- prefix: "bool try_pre" -->

In a scoped B200 comparison of a 65,536-token cached prefix and a 3,072-row suffix, TP1 and TP2 each matched all 258 output IDs, text, and usage against the prior planner. The resident-first group increased transient prefill workspace by 255,557,632 B per rank. In the separate TP2 case with 64,896 cached tokens and a 3,072-row continuation, the retained schedule used 640- and 2,432-row grants. Neither contains three full tiles, so the new preference cannot apply to that decomposition. See `artifacts/generalization/proof/wide-warm-intersection.gwnaDf24/{combined-warm-attribution,aligned-physical-attention}.log`.

### TP2 causal prefill reduction fusion

Eager 4,096-row TP2 Tensor Core causal groups can fuse reduction, RMSNorm and E4M3 quantization. Require alignment, two distinct SM100 devices, bidirectional peer access at performance rank 0 with native peer atomics, and access to both pools. First-input normalization stays local and final MLP reduces separately. Capture, unsupported setup or misalignment uses NCCL and local norm. [Caller gate](../src/decoder_prefill.cpp#L368)<!-- prefix: "if (plan.ful" --> · [Peer setup](../src/runtime/execution/decoder_prefill_workspace.cpp#L142)<!-- prefix: "void Prefill" --> · [NCCL fallback](../src/decoder_prefill.cpp#L279)<!-- prefix: "if (peer_nor" -->

Both streams record projection-ready before waiting for each other, and read-complete before buffer reuse. Each rank rounds the joint sum to BF16 before residual, normalization and quantization. Submission errors synchronize both streams and prioritize terminal CUDA status. [Event ordering](../src/runtime/execution/decoder_prefill_workspace.cpp#L167)<!-- prefix: "bool Prefill" --> · [BF16 sum](../src/text_normalization.cu#L37)<!-- prefix: "__device__ _" --> · [Error handling](../src/runtime/execution/decoder_prefill_workspace.cpp#L199)<!-- prefix: "check(termin" -->

Compare fused and ordinary reduction at 4,096 rows of causal prefill on both B200 ranks. Reuse a prefix and compare the rank-local state and continuation on identical inputs.

## Verification

The `decoder-batch-plan-check` CPU target checks decode reservations, MTP budget clipping, prompt order, slot reuse, and full occupancy. Enable it with `xmake f --decoder-batch-plan-check=y`. [Planner assertions](../tests/decoder/decoder_batch_plan_check.cpp#L24)<!-- prefix: "void check_d" --> · [Planner target](../xmake.lua#L406)<!-- prefix: "name = \"deco" -->

On a reserved B200, compare staged and raw readers on checkpoint Q, MXFP8 KV, and remapped pages. Cover bounds, nonzero page offsets, masks, short tails, invalid V, and repair. Require canaries, raw parity with scratch one byte short or rejected setup, one raw node at cold capture, and parity after selector changes. Count allocations and time eager predecode plus attention in A/B/B/A order.

Compare four-, two-, and one-lane staging bitwise against per-lane dispatch. Cover mixed starts, invalid KV, aliasing, stream rejection, and either launch rejected. Later groups may still stage after a clean decline. Unexpected errors after predecode must propagate without relaunch. Record allocation counts and arena high-water marks.

Fix admissions and grants. Compare scores, KV and scales, GDN carry and history, hidden and residual values, IDs and terminal records. Include unequal long grants with nonadjacent matching tiles, trailing partials, restored unaligned starts, mixed decode, and four 8,192-row grants under the 32,768-row budget. Deny the 4,096-row workspace and then both workspace sizes. Require a 2,048-row replan before grouped work or ordinary prefill on clean decline. Measure active and retained memory separately.

### TP2 paired prefill replay

Compare state and first-token logits on both ranks, proposals, acceptances, and each rank’s target and draft cache objects. Apply allocation pressure separately to both ranks and require zero transient pool bytes after synchronization.

<a id="tp2-grouped-rmsnorm-replay"></a>
<a id="tp2-causal-gqa-operator-replay"></a>
<a id="tp2-grouped-gateup-replay"></a>
<a id="tp2-packed-qkv-replay"></a>
<a id="tp2-grouped-attention-output-replay"></a>
<a id="tp2-long-down-projection-replay"></a>

Compare both rank shards for grouped RMSNorm, causal GQA, gate/up, packed QKV, attention output, and the long down projection. Match BF16 tensors, quantized values and scales, and teacher logits where arithmetic is preserved. Exercise eager, replay, clean fallback, and injected submission failures. Include zero and nonzero cached rows, two 4,096-row grants, and 4,096/1,024-row down with a 1,004-row decline. Check committed state and MTP continuation after priming.
