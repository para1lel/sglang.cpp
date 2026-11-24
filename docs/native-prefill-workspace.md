# Native prefill workspace

## Responsibilities and sources of truth

| Part | Responsibility | Source |
| --- | --- | --- |
| Decoder prefill | Lay out the arena and lend two scratch spans during GQA. | [Tile layout](../src/decoder_prefill.cpp#L41)<!-- prefix: "TileBuffers(" --> · [Scratch view](../src/decoder_prefill.cpp#L63)<!-- prefix: "PairedPrefil" --> |
| Attention dispatch | Stage eligible singleton and four-lane groups. Partial-grant groups can use staged, raw grouped, or per-lane attention. | [Singleton selection](../src/mxfp8_gqa_prefill.cu#L948)<!-- prefix: "const bool d" --> · [Grouped dispatch](../src/decoder_prefill.cpp#L585)<!-- prefix: "void prefill" --> |
| Paired launcher | Check bounds and capture state, then submit predecode and attention on the caller's stream. | [Staged launch](../include/sglang/text/mxfp8_gqa_prefill_paired.cuh#L709)<!-- prefix: "std::optiona" --> |

## Runtime path

The B200 text dispatcher lends these spans to eligible eager 1,024-row tiles and [partial-grant subgroups](ordered-gqa-prefill.md#grouped-partial-grants). Ordinary 1,024-row dispatch at start zero uses raw paired attention and leaves staging scratch unused. See [attention dispatch](flash-prefill.md#runtime-path) for full-tile bounds. [Singleton bound](../include/sglang/text/mxfp8_gqa_prefill_paired.cuh#L790)<!-- prefix: "if (position" -->

The decoder lends two disjoint spans of its existing `PrefillWorkspace` allocation. The first address is inclusive and the end address is exclusive. These capacities apply to a 1,024-row arena.

| Scratch use | Physical span | Capacity | Largest contents at start 31,744 |
| --- | --- | ---: | ---: |
| BF16 K | `mlp_gate` through the end of `mlp_up`, bounded by `q_gate` | 68 MiB | 64 MiB |
| BF16 V and invalid-token indices | `gdn_qkv`, `gdn_z`, `gdn_b`, `gdn_a`, and GDN scratch, bounded by `activation.values` | 64.1875 MiB | 64 MiB + 4 KiB |

Both spans allow up to 127 bytes of 128-byte alignment padding. The four-tile arena supplies 272 MiB for K and 256.75 MiB for V and invalid-token indices. A [staged subgroup](flash-prefill.md#runtime-path) covers at most 131,072 visible tokens in total. It uses up to 256 MiB for K and 256 MiB plus 16 KiB for V and indices before padding. The grouped partial reader rounds each visible span up to full pages and puts each lane at its own offset. Both scratch spans must fit the combined page count. Subgroups reuse both spans on one stream without another device allocation. [Arena allocation](../src/runtime/execution/decoder_prefill_workspace.cpp#L178)<!-- prefix: "PrefillWorks" --> · [Scratch sizing](../include/sglang/text/mxfp8_gqa_prefill_paired.cuh#L749)<!-- prefix: "const auto p" -->

The MLP projection writes its gate and up buffers after GQA. GDN layers write their projection and scratch inputs before use. Predecode, paired attention, later layer work, and workspace release use the workspace's execution stream. Stream ordering puts the last attention read before the next write to either span. Keep both spans exclusive to that stream until the queued work completes. [Layer loop](../src/decoder_prefill.cpp#L864)<!-- prefix: "execute_laye" --> · [Release](../src/runtime/execution/decoder_prefill_workspace.cpp#L233)<!-- prefix: "cudaError_t " -->

Predecode follows each lane’s page table and expands visible MXFP8 entries to BF16. Grouped partial staging zero-fills entries beyond that lane’s visible end. Separate K and V TMA producers load alternating shared-memory stages. See [the paired kernel](flash-prefill.md#long-tile-tmem-pipeline) for arithmetic and tile geometry. [Predecode](../include/sglang/text/mxfp8_gqa_prefill_paired.cuh#L99)<!-- prefix: "__global__ v" --> · [Producers](../include/sglang/text/mxfp8_gqa_prefill_paired.cuh#L186)<!-- prefix: "__device__ v" -->

The [CUDA 13.4 stream ordering guide](https://docs.nvidia.com/cuda/archive/13.4.3/cuda-programming-guide/02-basics/asynchronous-execution.html#cuda-stream-ordering) defines the execution order. The [CUTLASS SM100 MLA change](https://github.com/NVIDIA/cutlass/pull/2472) illustrates a two-CTA asynchronous attention pipeline. The native route uses its own MXFP8 layout and arithmetic.

## Constraints and failure behavior

The workspace allocator reuses private CUDA pools by device and size class, separating workspaces at or below 128 rows from larger ones. At most one pool per device and class uses a non-default release threshold, 32 MiB for small pools and 1 GiB for large ones. CUDA can retain that backing after synchronization and request retirement. `workspace_bytes` counts active workspace bytes, including staging spans. Use device memory measurements to observe retained backing. [Pool policy](../src/runtime/execution/decoder_prefill_workspace.cpp#L50)<!-- prefix: "void configu" --> · [Allocation](../src/runtime/execution/decoder_prefill_workspace.cpp#L178)<!-- prefix: "PrefillWorks" -->

Out-of-memory or unsupported pool creation or allocation leaves the workspace unavailable. Ordinary prefill retries an unavailable large allocation with 128 rows, then uses row-wise prefill if scratch remains unavailable. Single-row tiles also use row-wise prefill. Other CUDA allocation errors and outstanding errors propagate. Release enqueues a free on the workspace stream. If enqueue fails, recovery synchronizes the owning device, checks used bytes, and waits for a retried free when required. Explicit close reports the original failure. [Unavailable statuses](../src/runtime/execution/decoder_prefill_workspace.cpp#L161)<!-- prefix: "bool prefill" --> · [Ordinary retry](../src/decoder_prefill.cpp#L924)<!-- prefix: "if (!workspa" --> · [Release](../src/runtime/execution/decoder_prefill_workspace.cpp#L233)<!-- prefix: "cudaError_t " --> · [Recovery](../src/runtime/execution/decoder_prefill_workspace.cpp#L144)<!-- prefix: "void recover" -->

Missing or undersized scratch, capture, and known setup rejection decline staging before submission. Setup rejection includes unavailable kernel resources and tensor-map rejection or descriptor out-of-memory. Batch staging also requires an explicit CUDA stream in the current context. Its output and scratch ranges must be disjoint from each other and from live inputs within a subgroup. [Capture check](../include/sglang/text/mxfp8_gqa_prefill_paired.cuh#L764)<!-- prefix: "const auto c" --> · [Stream check](../include/sglang/text/mxfp8_gqa_prefill_staged.cuh#L133)<!-- prefix: "std::optiona" --> · [Range check](../include/sglang/text/mxfp8_gqa_prefill_paired.cuh#L776)<!-- prefix: "if (!valid_s" --> · [Tensor maps](../include/sglang/text/mxfp8_gqa_prefill_paired.cuh#L672)<!-- prefix: "const auto e" -->

A recognized rejection can decline either staged launch. Full-tile subgroups fall back lane by lane. Ordinary singleton staging still falls back to raw paired attention. Partial subgroups try [raw grouped ordered TMEM](ordered-gqa-prefill.md#grouped-partial-grants) when eligible, then fall back lane by lane. If attention declines after predecode queues, the stream orders fallback and later reuse after predecode. Pending and unexpected CUDA errors propagate. [Submission handling](../include/sglang/text/mxfp8_gqa_prefill_staged.cuh#L173)<!-- prefix: "std::optiona" --> · [Full-tile fallback](../src/decoder_prefill.cpp#L479)<!-- prefix: "for (int ind" -->

A paired graph capture records the raw kernel with maps by value and uses zero staging scratch. Keep the captured cache, page table, queries, and output valid through replay. [Raw launch](../include/sglang/text/mxfp8_gqa_prefill_paired.cuh#L608)<!-- prefix: "std::optiona" --> · [CUDA graph capture](https://docs.nvidia.com/cuda/archive/13.4.2/cuda-programming-guide/04-special-topics/cuda-graphs.html#stream-capture)

## Multi-request text prefill

The [batch planner](batch-graphs.md#runtime-path) assigns rows. The executor splits grants into physical tiles of up to 1,024 rows and groups compatible work. Each request keeps its token order, KV pages, recurrent state, and output rows. [Attention dispatch](flash-prefill.md#runtime-path)

The earlier K0 planner tested fair-share, tiled, and grouped schedules. Its measured cases used the historical selector and K0 rules. Recover the planner and run notes at the original revisions. [K0 campaign](https://github.com/para1lel/sglang.cpp/blob/8b4e3f62e0a7a053ffdefe85da99a841fabeb5d3/docs/serving/paired-measurement-ledger.md) · [Previous workspace guide](https://github.com/para1lel/sglang.cpp/blob/4dd8c887c278c67744ed80e7feb2a825a92197a2/docs/native-prefill-workspace.md#tiled-prefill-under-saturation)

The scheduler sends generation and scoring grants to one prefill executor. B200 can group compatible Tensor Core tiles. Cache restore supplies the starting state. Media embeddings, rotary positions, and score captures belong to each lane. Decode work has separate dispatch in the same step. [Step dispatch](../src/decoder_batch.cpp#L620)<!-- prefix: "if (max_rows" --> · [Grant assembly](../src/decoder_batch.cpp#L651)<!-- prefix: "void Decoder" -->

### Projection pairs for MTP

Whole-tile grouping has priority. It requires an even number of equal grants made of full 1,024-row tiles, with at most 8,192 rows across the grants. The 32,768-row scheduler budget is a separate limit. The [`paired_prefill` field](native-model-info.md#server-information) reports startup capability. Each lane uses its own draft state if its depth is positive. [Grant eligibility](../include/sglang/decoder_prefill.h#L86)<!-- prefix: "inline bool " --> · [Group priority](../src/decoder_prefill.cpp#L726)<!-- prefix: "if (model.ba" -->

Exactly four eligible grants first try one 4,096-row group per tile offset.
Four 2,048-row grants use two groups in one step. Two 4,096-row grants use
four 2,048-row groups. Other eligible grant counts also use pairs.
Lanes share dense projections and the MLP. Attention and GDN keep each lane's
1,024-row tiles, positions, KV pages, and recurrent state. Draft seeding uses
the lane's own hidden and residual rows and shifted token IDs.
[Group selection](../src/decoder_prefill.cpp#L727)<!-- prefix: "if (count ==" --> · [Draft seed](../src/decoder_prefill.cpp#L671)<!-- prefix: "auto seed = " -->

[Grouped GDN](gdn-prefill.md#runtime-path) reuses the existing disjoint row
spans for both projection group sizes. The group uses the workspace stream.
Eligible four-grant groups can also use [batched staged GQA](flash-prefill.md#runtime-path).

| Workspace | Active bytes |
| --- | ---: |
| Ordinary 1,024-row tile | 255,557,632 |
| Two-grant, 2,048-row group | 511,115,264 |
| Four-grant, 4,096-row group | 1,022,230,528 |

The step reuses one workspace across all groups. Four-grant grouping adds
511,115,264 active bytes over pairs. If the four-grant pool or allocation is
unavailable, dispatch tries pairs before submitting grouped work. If the pair
allocation is also unavailable, dispatch uses ordinary per-grant prefill.
Unexpected CUDA errors propagate. [Group workspace](../src/decoder_prefill.cpp#L692)<!-- prefix: "PrefillWorks" -->

### Partial-grant projection groups

The partial-grant planner can pack consecutive complete grants of 256–1,024 rows. Each grant must fit its own native tile. A group needs at least two grants and fits at most 4,096 rows. The planner leaves grant order intact. Ungrouped grants use ordinary prefill. [Plan and dispatch](../include/sglang/runtime/execution/decoder_prefill_plan.h#L65)<!-- prefix: "bool execute" -->

The group shares dense projections and MLP work. Uniform 4,096-row groups of eight 512-row or sixteen 256-row grants can also share [RMSNorm and GQA postgate preparation](fp8-prefill-projections.md#attention-output-quantization). [Grouped staged and ordered GQA](ordered-gqa-prefill.md#grouped-partial-grants) and [grouped GDN](gdn-prefill.md#grouped-partial-grants) qualify independently. Other attention and recurrence use per-lane dispatch. Each lane keeps its positions, KV pages, and recurrent state. All work uses the same stream and existing workspace. Draft seeding uses the lane's own rows where applicable. [Layer dispatch](../src/decoder_prefill.cpp#L585)<!-- prefix: "void prefill" --> · [GQA selection](../src/decoder_prefill.cpp#L617)<!-- prefix: "if (causal |" -->

One arena fits the largest planned group and ordinary tile in the step. The executor reuses it for the other grants. If no group qualifies or its allocation is unavailable, the executor falls back before submitting grouped work. [Allocation and fallback](../include/sglang/runtime/execution/decoder_prefill_plan.h#L85)<!-- prefix: "PrefillWorks" -->

With `--trace-batches`, `layerwise_prefill_groups` counts the groups,
`rows` counts their combined rows, and `workspace_bytes` records the
largest active prefill workspace in the step.
[Batch trace](../include/sglang/server/batch_trace.h#L46)<!-- prefix: "if (stats.la" -->

### Causal segments within one grant

Whole-tile pairing and complete groups of distinct grants take precedence. This path requires a Tensor Core grant with a native tile size of 1,024 rows. The executor packs four complete tiles when at least four remain. It packs three when exactly three remain. On the workspace stream, each layer shares dense projections and MLP work across the combined rows. It appends and reads GQA KV and advances GDN state one segment at a time in prompt order. The scheduler’s grant stays intact. Ordinary prefill resumes at the first unprocessed row.
[Segment selection](../include/sglang/runtime/execution/decoder_prefill_plan.h#L39)<!-- prefix: "inline Prefi" --> · [GQA order](../src/decoder_prefill.cpp#L617)<!-- prefix: "if (causal |" --> · [GDN order](../src/decoder_prefill.cpp#L656)<!-- prefix: "prefill_part" --> · [Native remainder](../include/sglang/runtime/execution/decoder_prefill_plan.h#L115)<!-- prefix: "if (offset <" -->

Causal segments reuse the same arena sizing and allocation fallback. Unexpected CUDA errors propagate.

## Verification

The `decoder-batch-plan-check` CPU target checks decode reservations, MTP budget clipping, prompt order, slot reuse, and full occupancy. Enable it with `xmake f --decoder-batch-plan-check=y`. [Planner assertions](../tests/decoder/decoder_batch_plan_check.cpp#L24)<!-- prefix: "void check_d" --> · [Planner target](../xmake.lua#L309)<!-- prefix: "target(\"deco" -->

On one reserved B200, replay identical checkpoint Q, MXFP8 payloads and scales, and remapped page tables. Exercise both bounds and nonzero page offsets. Compare full BF16 output with raw paired and archived wide Flash. Include a short tail, causal masks, invalid full- and partial-page V, and repair.

Check each scratch span's bounds and canaries, including a span one byte shorter than the aligned requirement. Check descriptor rejection, empty cluster occupancy, and unavailable predecode. Require raw output parity in those cases.

For four-grant staging, compare full BF16 outputs bitwise with the ordinary per-lane dispatcher. Exercise four-, two-, and one-lane subgroups, lane-specific invalid KV, aliasing, mismatched starts, and stream rejection. Reject each launch in turn with a known configuration status. Check the affected subgroup falls back and later eligible work still stages. Inject an unexpected attention launch error after predecode and require propagation without a second attention launch. Compare allocation counts and the arena high-water mark.

Check a cold capture with scratch supplied. Require one raw kernel node and compare replay after a selector change. Check explicit `paired` and the default dispatch boundaries. Count allocation calls on each arm.

Time eager predecode and attention together with resident inputs in base, candidate, candidate, base order. Archived checkpoint captures showed bitwise BF16 parity and lower operator latency. Verify layer lifetime and request performance with separate checks. [Operator evidence](https://github.com/para1lel/sglang.cpp/blob/4dd8c887c278c67744ed80e7feb2a825a92197a2/docs/native-prefill-workspace.md#verification)

On B200, hold admissions and grants fixed. Compare full score rows, per-lane state, generated IDs, and terminal records with ordinary dispatch. Include unequal and trailing grants, a singleton control, and two 256-row grants at K0. Include a mixed prefill and decode step. Check `layerwise_prefill_groups` and grouped `rows` to confirm which steps group. Reject the grouped workspace allocation and check ordinary prefill completes with the smaller workspace. Compare active workspace statistics and retained GPU memory after retirement separately.

For whole-tile pairs, check both lane offsets and cover mixed, partial, and per-lane split grants. Compare complete logits and recurrent state in addition to sampled hidden and residual values.

For four-grant grouping, use the default 32,768-row scheduler budget. Check four equal grants and a two-grant control on the same native policy. Compare target hidden rows, logits, KV, recurrence, and subsequent draft state. Check uneven grants and a mixed decode step. Verify that whole-tile pairing declines when grants total more than 8,192 rows. Reject the 4,096-row allocation and confirm paired fallback. Remove the failure and confirm four-grant recovery. Record output IDs, active workspace bytes, and physical GPU memory separately.

Historical K0 comparisons exercised pairing on the reduced-output W4 fixture. W3 cohorts exercised no pairs. W5 and W6 used host replay. Inspect traced grant sizes and admission and decode overlap before attributing serving timing changes to pairing. [Historical pairing evidence](https://github.com/para1lel/sglang.cpp/blob/4dd8c887c278c67744ed80e7feb2a825a92197a2/docs/native-prefill-workspace.md#verification)
