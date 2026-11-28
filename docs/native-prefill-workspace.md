# Native prefill workspace

## Responsibilities and sources of truth

The decoder lends arena spans to singleton and grouped GQA. Attention dispatch selects staged, raw grouped, or per-lane work. The paired launcher checks eligibility and submits on the workspace stream. [Tile layout](../src/runtime/execution/decoder_prefill_tiles.cpp#L84)<!-- prefix: "TileBuffers:" --> · [Scratch view](../src/runtime/execution/decoder_prefill_tiles.cpp#L109)<!-- prefix: "PairedPrefil" --> · [Singleton selection](../src/text/gqa/ordered_prefill.cu#L282)<!-- prefix: "std::optiona" --> · [Grouped dispatch](../src/decoder_prefill.cpp#L454)<!-- prefix: "void prefill" --> · [Staged launch](../src/text/gqa/tmem_staged.cu#L452)<!-- prefix: "std::optiona" -->

## Runtime path

B200 lends projection workspace to eligible eager 1,024-row text tiles and [partial-grant subgroups](ordered-gqa-prefill.md#grouped-partial-grants). A 1,024-row tile at start zero uses raw paired attention. [Attention dispatch](flash-prefill.md#runtime-path) · [Singleton bound](../src/text/gqa/tmem_staged.cu#L547)<!-- prefix: "if (position" -->

| Scratch | Physical span (inclusive start, exclusive end) | One tile capacity | Contents at start 31,744 | Four-tile capacity | Subgroup maximum contents |
| --- | --- | ---: | ---: | ---: | ---: |
| BF16 K | `mlp_gate` through `mlp_up`, bounded by `q_gate` | 68 MiB | 64 MiB | 272 MiB | 256 MiB |
| BF16 V and invalid-token indices | `gdn_qkv`, `gdn_z`, `gdn_b`, `gdn_a` and GDN scratch, bounded by `activation.values` | 64.1875 MiB | 64 MiB + 4 KiB | 256.75 MiB | 256 MiB + 16 KiB |

Each disjoint span allows 127 bytes of 128-byte alignment padding. A [staged subgroup](flash-prefill.md#runtime-path) covers up to 131,072 visible tokens total. Partial groups round each span to whole pages and assign lane offsets. Both spans must fit the combined page count. Subgroups reuse them on one stream without device allocation. GQA keeps exclusive use through the last read, then MLP/GDN can overwrite them. Allocation release follows on the same stream. [Arena allocation](../src/runtime/execution/decoder_prefill_workspace.cpp#L88)<!-- prefix: "PrefillWorks" --> · [Scratch sizing](../src/text/gqa/tmem_staged.cu#L506)<!-- prefix: "const auto p" --> · [Layer loop](../src/decoder_prefill.cpp#L649)<!-- prefix: "run_layers(" --> · [Release](../src/runtime/execution/decoder_prefill_workspace.cpp#L238)<!-- prefix: "cudaError_t " -->

Predecode expands paged MXFP8 to BF16 and zero-fills partial-group lanes past their visible end. See [the paired kernel](flash-prefill.md#long-tile-tmem-pipeline) for separate K/V TMA producers and alternating stages. [Predecode](../src/text/gqa/tmem_staged_kernels.cu#L53)<!-- prefix: "__global__ v" --> · [Producers](../src/text/gqa/tmem_staged_kernels.cu#L141)<!-- prefix: "__device__ v" -->

The [CUDA 13.4 stream ordering guide](https://docs.nvidia.com/cuda/archive/13.4.3/cuda-programming-guide/02-basics/asynchronous-execution.html#cuda-stream-ordering) defines the execution order. The [CUTLASS SM100 MLA change](https://github.com/NVIDIA/cutlass/pull/2472) illustrates a two-CTA asynchronous attention pipeline. The native route uses its own MXFP8 layout and arithmetic.

### Dedicated long-context staging

The prefill workspace takes one general-arena lease for row scratch and optional BF16 K/V staging. The staging planes are disjoint from each other and from the row scratch. Both start on 128-byte boundaries. The V span ends with one `int` per page and local KV head for invalid-token metadata. [Layout and lease](../src/runtime/execution/decoder_prefill_workspace.cpp#L104)<!-- prefix: "const auto b" --> · [Plane views](../src/runtime/execution/decoder_prefill_workspace.cpp#L137)<!-- prefix: "auto* keys =" -->

Cap visibility at 262,144 and round up to 128-token pages. At that limit, TP1's four local KV heads require 512 MiB per BF16 plane plus 32 KiB of metadata. TP2's two local KV heads require 256 MiB per plane plus 16 KiB of metadata per rank. These sizes exclude row scratch and its alignment padding. [Page and head sizing](../src/runtime/execution/decoder_prefill_workspace.cpp#L107)<!-- prefix: "const auto v" -->

On clean allocation pressure the workspace retries with row scratch only. TP2 uses staging only when both ranks have the planes. The arena releases the combined lease on its owning stream after all users finish. [Base retry](../src/runtime/execution/decoder_prefill_workspace.cpp#L119)<!-- prefix: "} catch (con" --> · [Both-rank gate](../src/runtime/execution/decoder_prefill_tiles.cpp#L324)<!-- prefix: "PrefillGqaSt" --> · [Stream-ordered release](../src/runtime/execution/decoder_prefill_workspace.cpp#L238)<!-- prefix: "cudaError_t " -->

## Constraints and failure behavior

`workspace_bytes` includes active staging spans leased from the shared [device arena](serving/request-memory.md). The backing allocation stays committed for later leases. [Workspace accounting](../src/runtime/execution/decoder_prefill_workspace.cpp#L143)<!-- prefix: "std::size_t " -->

| Condition | Outcome |
| --- | --- |
| Arena exhaustion | The workspace retries base scratch when optional staging cannot fit. Ordinary prefill tries 128 rows on a large decline, then goes row-wise if scratch remains unavailable. Single rows use row-wise. Other allocation errors propagate. [Base retry](../src/runtime/execution/decoder_prefill_workspace.cpp#L119)<!-- prefix: "} catch (con" --> · [Ordinary retry](../src/decoder_prefill.cpp#L939)<!-- prefix: "if (!workspa" --> |
| Release event failure | The arena fences the owning stream before returning the span. Explicit workspace close checks the release status. [Arena release](../src/runtime/backend/device_arena.cpp#L460)<!-- prefix: "cudaError_t " --> · [Workspace close](../src/runtime/execution/decoder_prefill_workspace.cpp#L248)<!-- prefix: "void Prefill" --> |
| Missing/undersized scratch, capture or optional setup rejection | Decline before staging (including resource, tensor-map and descriptor OOM setup rejection). Batch requires an explicit current-context stream and disjoint output/scratch/live ranges in each subgroup. [Capture check](../src/text/gqa/tmem_staged.cu#L521)<!-- prefix: "const auto c" --> · [Stream check](../src/text/gqa/tmem_staged.cu#L210)<!-- prefix: "std::optiona" --> · [Range check](../src/text/gqa/tmem_staged.cu#L533)<!-- prefix: "if (!valid_s" --> · [Tensor maps](../src/text/gqa/tmem_staged.cu#L395)<!-- prefix: "const auto e" --> |
| Recognized staged launch rejection | Singleton tries raw paired. Full-tile groups use per-lane dispatch. Partial groups try [raw grouped ordered TMEM](ordered-gqa-prefill.md#grouped-partial-grants), then per lane. Predecode already enqueued orders fallback and reuse. The [long whole-tile page reader](flash-prefill.md#long-tile-tmem-pipeline) propagates submission errors, as do all routes for pending or unexpected errors. [Submission handling](../src/text/gqa/tmem_staged.cu#L239)<!-- prefix: "std::optiona" --> · [Full-tile fallback](../src/runtime/execution/decoder_prefill_staged_gqa.cpp#L62)<!-- prefix: "for (int ind" --> |
| Paired graph capture | Record raw with maps by value and no staging scratch. Keep cache, page table, queries and output valid through replay. [Raw launch](../src/text/mxfp8_gqa_prefill_tmem.cu#L706)<!-- prefix: "std::optiona" --> · [CUDA graph capture](https://docs.nvidia.com/cuda/archive/13.4.2/cuda-programming-guide/04-special-topics/cuda-graphs.html#stream-capture) |

## Multi-request text prefill

The [batch planner](batch-graphs.md#runtime-path) sends generation and scoring to one prefill executor, separate from decode. Compatible work groups into tiles of at most 1,024 rows. Order, KV, recurrence, rotary positions, media, scores and outputs stay request-local. On [Cache restore](prefix-cache.md#checkpoint-compatibility-and-resume), fully covered GQA tiles project only queries and gates. Others append only uncovered KV. [Attention dispatch](flash-prefill.md#runtime-path) · [Step dispatch](../src/decoder_batch.cpp#L495)<!-- prefix: "if (max_rows" --> · [Grant assembly](../src/decoder_batch.cpp#L527)<!-- prefix: "void Decoder" --> · [Cached GQA append](../src/runtime/execution/decoder_prefill_tiles.cpp#L285)<!-- prefix: "void Chunk::" -->

### Projection pairs for MTP

B200 prioritizes whole-tile grouping for equal grants made entirely of 1,024-row tiles. Each lane keeps its own absolute start, KV, recurrent and draft state. [Tile eligibility](../src/runtime/execution/decoder_prefill_workspace.cpp#L66)<!-- prefix: "bool decoder" --> · [Group selection](../src/decoder_prefill.cpp#L552)<!-- prefix: "bool decoder" --> · [Tile offsets](../src/decoder_prefill.cpp#L532)<!-- prefix: "for (int off" --> · [Draft seed](../src/decoder_prefill.cpp#L198)<!-- prefix: "auto seed = " -->

| Group per tile offset | Eligibility and example | Active workspace |
| --- | --- | ---: |
| 4,096 rows | Exactly four grants. Four 8,192-row grants run eight groups in a single 32,768-row step. | 1,022,230,528 B |
| 2,048 rows | Even grant count and ≤8,192 rows total. Two 4,096-row grants run four groups. [`paired_prefill` field](native-model-info.md#server-information) · [Pair limit](../src/decoder_prefill.cpp#L562)<!-- prefix: "if (grants[0" --> | 511,115,264 B |
| 1,024 rows | Ordinary tile. | 255,557,632 B |

One workspace serves every group. If 4,096-row allocation declines, try pairs within the pair limit, otherwise the partial/causal planner, which can request its own 4,096-row arena. An unavailable eligible pair or ungrouped work goes to ordinary prefill. [Group workspace](../src/decoder_prefill.cpp#L527)<!-- prefix: "PrefillWorks" --> · [Partial and causal fallback](../src/decoder_prefill.cpp#L114)<!-- prefix: "return execu" -->

[Grouped GDN](gdn-prefill.md#runtime-path) uses existing disjoint spans for both group sizes. Four aligned grants can use [batched staged GQA](flash-prefill.md#runtime-path). For mixed starts, eligible distinct text pairs or unmatched long tiles stage in the same 4,096-row workspace with owner-local pages and outputs. [Attention dispatch](../src/decoder_prefill.cpp#L140)<!-- prefix: "plan.staged_" -->

### Paired 2,048-token TP2 prompts

B200 TP2 groups two distinct complete cold 2,048-token grants with Tensor Core prefill and native GDN on both ranks. Four tiles share projections. Attention and recurrence advance in request order. MTP seed retains stock draft FC and cache format. Each rank leases 1,022,230,528 B from its device arena before submission. Peer allocation decline returns rank 0’s provisional lease after the stream fence. CUDA errors propagate. [Eligibility](../src/runtime/execution/decoder_prefill_plan.cpp#L96)<!-- prefix: "bool can_pac" --> · [Dispatch](../src/decoder_prefill.cpp#L844)<!-- prefix: "bool try_pre" --> · [Workspace](../src/runtime/execution/decoder_prefill_workspace.cpp#L88)<!-- prefix: "PrefillWorks" -->

### Seed scratch

MTP draft seed uses full-geometry scratch on the primary rank outside the rank-local layout. Paired seed reserves `concatenated` for up to 128 rows and `keys`, `values` and `rope_keys` for every launch row. Each seed path throws `runtime_error` if scratch does not fit. [Seed scratch](../src/runtime/execution/decoder_prefill_tiles.cpp#L72)<!-- prefix: "std::size_t " -->

### Partial-grant projection groups

Group consecutive complete grants in order, at least two, ≤4,096 rows total. Each fits one tile at 256–1,024 rows on TP1 or 128–1,024 on TP2. Ungrouped grants stay ordinary. [Plan and dispatch](../src/runtime/execution/decoder_prefill_plan.cpp#L126)<!-- prefix: "bool execute" -->

Groups share projections and MLP. Uniform 8×512 or 16×256 groups can share [RMSNorm and GQA postgate preparation](fp8-prefill-projections.md#attention-output-quantization). [Grouped staged and ordered GQA](ordered-gqa-prefill.md#grouped-partial-grants) and [grouped GDN](gdn-prefill.md#grouped-partial-grants) qualify separately, with per-lane fallback. Positions, KV, recurrence and draft seed stay lane-local. [Layer dispatch](../src/decoder_prefill.cpp#L454)<!-- prefix: "void prefill" --> · [GQA selection](../src/runtime/execution/decoder_prefill_staged_gqa.cpp#L335)<!-- prefix: "} else if (p" -->

Reuse one workspace sized for the largest group or ordinary tile on the same stream. Eligible steps add [dedicated staging](#dedicated-long-context-staging). No group or unavailable base scratch falls back before grouped submission. [Allocation and fallback](../src/runtime/execution/decoder_prefill_plan.cpp#L148)<!-- prefix: "PrefillWorks" -->

With `--trace-batches`, `layerwise_prefill_groups` counts groups, `rows` counts combined rows and `workspace_bytes` reports the step’s largest active prefill workspace. [Batch trace](../apps/server/detail/batch_trace.cpp#L41)<!-- prefix: "if (stats.la" -->

### Causal segments within one grant

After whole-tile pairs and distinct-grant groups, a Tensor Core grant can pack four remaining 1,024-row tiles or three when exactly three remain. Each layer shares projections and MLP while GDN follows prompt order. Four tiles can try [staged same-request GQA](flash-prefill.md#same-request-causal-gqa) even with unequal peer grants. When staging declines, read KV per segment. Ordinary prefill resumes at the first unprocessed row. Reuse the arena sizing and allocation fallback. Unexpected CUDA errors propagate. TP2 mirrors segments by source grant index and offset. [Segment selection](../src/runtime/execution/decoder_prefill_plan.cpp#L22)<!-- prefix: "PrefillGroup" --> · [GQA selection](../src/runtime/execution/decoder_prefill_staged_gqa.cpp#L334)<!-- prefix: "if (prefill_" --> · [GDN order](../src/decoder_prefill.cpp#L170)<!-- prefix: "prefill_part" --> · [Native remainder](../src/runtime/execution/decoder_prefill_plan.cpp#L179)<!-- prefix: "if (offset <" --> · [Peer segments](../src/runtime/execution/decoder_prefill_plan.cpp#L118)<!-- prefix: "void peer_ca" -->

Eligible [warm cached grants](flash-prefill.md#warm-cached-gqa-through-the-full-context) can stage either three or four tiles with the same lease and schedule. Both TP2 ranks must have the dedicated planes.

### TP2 causal prefill reduction fusion

Eager 4,096-row TP2 Tensor Core causal groups can fuse reduction, RMSNorm and E4M3 quantization. Require alignment, two distinct SM100 devices, bidirectional peer access at performance rank 0 with native peer atomics, and access to both pools. First-input normalization stays local and final MLP reduces separately. Capture, unsupported setup or misalignment uses NCCL and local norm. [Caller gate](../src/decoder_prefill.cpp#L421)<!-- prefix: "if (plan.ful" --> · [Peer setup](../src/runtime/execution/decoder_prefill_workspace.cpp#L166)<!-- prefix: "void Prefill" --> · [NCCL fallback](../src/decoder_prefill.cpp#L259)<!-- prefix: "if (peer_nor" -->

Both streams record projection-ready before waiting for each other, and read-complete before buffer reuse. Each rank rounds the joint sum to BF16 before residual, normalization and quantization. Submission errors synchronize both streams and prioritize terminal CUDA status. [Event ordering](../src/runtime/execution/decoder_prefill_workspace.cpp#L191)<!-- prefix: "bool Prefill" --> · [BF16 sum](../src/text_normalization.cu#L37)<!-- prefix: "__device__ _" --> · [Error handling](../src/runtime/execution/decoder_prefill_workspace.cpp#L223)<!-- prefix: "check(termin" -->

Compare fused and ordinary reduction at 4,096 rows of causal prefill on both B200 ranks. Reuse a prefix and compare the rank-local state and continuation on identical inputs.

## Verification

The `decoder-batch-plan-check` CPU target checks decode reservations, MTP budget clipping, prompt order, slot reuse, and full occupancy. Enable it with `xmake f --decoder-batch-plan-check=y`. [Planner assertions](../tests/decoder/decoder_batch_plan_check.cpp#L24)<!-- prefix: "void check_d" --> · [Planner target](../xmake.lua#L446)<!-- prefix: "target(\"deco" -->

On a reserved B200, compare staged and raw readers on checkpoint Q, MXFP8 KV, and remapped pages. Cover bounds, nonzero page offsets, masks, short tails, invalid V, and repair. Require canaries, raw parity with scratch one byte short or rejected setup, one raw node at cold capture, and parity after selector changes. Count allocations and time eager predecode plus attention in A/B/B/A order.

Compare four-, two-, and one-lane staging bitwise against per-lane dispatch. Cover mixed starts, invalid KV, aliasing, stream rejection, and either launch rejected. Later groups may still stage after a clean decline. Unexpected errors after predecode must propagate without relaunch. Record allocation counts and arena high-water marks.

Fix admissions and grants. Compare scores, KV and scales, GDN carry and history, hidden and residual values, IDs, and terminal records. Include singleton, unequal, trailing, split, restored, and mixed decode grants, both offsets of whole-tile pairs, two 256-row K0 grants, and four 8,192-row grants under the 32,768-row budget. Deny grouped scratch, including 4,096-row workspace at and above the 8,192-row pair limit. Require paired fallback at the limit, causal and ordinary fallback above it, and recovery. Measure active and retained memory separately.

### TP2 paired prefill replay

Compare state and first-token logits on both ranks, proposals and acceptances, and ranked cache objects including root draft data. Apply allocation pressure to each rank and require zero transient pool bytes used or reserved after synchronization.

<a id="tp2-grouped-rmsnorm-replay"></a>
<a id="tp2-causal-gqa-operator-replay"></a>
<a id="tp2-grouped-gateup-replay"></a>
<a id="tp2-packed-qkv-replay"></a>
<a id="tp2-grouped-attention-output-replay"></a>
<a id="tp2-long-down-projection-replay"></a>

Compare both rank shards for grouped RMSNorm, causal GQA, gate/up, packed QKV, attention output, and the long down projection. Match BF16 tensors, quantized values and scales, and teacher logits where arithmetic is preserved. Exercise eager, replay, clean fallback, and injected submission failures. Include zero and nonzero cached rows, two 4,096-row grants, and 4,096/1,024-row down with a 1,004-row decline. Check committed state and MTP continuation after priming.
