# Native prefill workspace

## Responsibilities and sources of truth

| Part | Responsibility | Source |
| --- | --- | --- |
| Decoder prefill | Own the arena and lend two scratch spans during GQA. | [Tile layout](../src/decoder_prefill.cpp#L177)<!-- prefix: "TileBuffers(" --> · [Scratch view](../src/decoder_prefill.cpp#L199)<!-- prefix: "PairedPrefil" --> |
| Attention dispatch | Try singleton staging within the default paired route and batch eligible four-grant groups. | [Singleton selection](../src/mxfp8_gqa_prefill.cu#L933)<!-- prefix: "const bool d" --> · [Batch selection](../src/decoder_prefill.cpp#L627)<!-- prefix: "if (grouped " --> |
| Paired launcher | Check bounds and capture state, then submit predecode and attention on the caller's stream. | [Staged launch](../include/sglang/text/mxfp8_gqa_prefill_paired.cuh#L421)<!-- prefix: "std::optiona" --> |

## Runtime path

The B200 text dispatcher lends these spans to eligible eager 1,024-row tiles. The first eligible start is 1,024. At start zero, raw paired attention leaves staging scratch unused. See the [attention dispatch](flash-prefill.md#runtime-path) for singleton and batch eligibility. [Staging bounds](../include/sglang/text/mxfp8_gqa_prefill_paired.cuh#L426)<!-- prefix: "if (position" -->

The decoder lends two disjoint spans of its existing `PrefillWorkspace` allocation. The first address is inclusive and the end address is exclusive. These capacities apply to a 1,024-row arena.

| Scratch use | Physical span | Capacity | Largest contents at start 31,744 |
| --- | --- | ---: | ---: |
| BF16 K | `mlp_gate` through the end of `mlp_up`, bounded by `q_gate` | 68 MiB | 64 MiB |
| BF16 V and invalid-token indices | `gdn_qkv`, `gdn_z`, `gdn_b`, `gdn_a`, and GDN scratch, bounded by `activation.values` | 64.1875 MiB | 64 MiB + 4 KiB |

Each span has room for up to 127 bytes of padding for 128-byte alignment. The launcher checks both spans separately. A four-grant group partitions both spans by lane, including the invalid-token indices. At the latest eligible start, each lane sees 8,192 tokens. Together they need 64 MiB for K and 64 MiB plus 4 KiB for V and flags. Both routes use the existing arena and add zero device allocation bytes. [Arena allocation](../src/decoder_prefill.cpp#L744)<!-- prefix: "PrefillWorks" --> · [Bounds](../include/sglang/text/mxfp8_gqa_prefill_paired.cuh#L434)<!-- prefix: "const int fu" -->

The MLP projection writes its gate and up buffers after GQA. GDN layers write their projection and scratch inputs before use. Predecode, paired attention, later layer work, and workspace release use the workspace's execution stream. Stream ordering puts the last attention read before the next write to either span. Keep both spans exclusive to that stream until the queued work completes. [Layer loop](../src/decoder_prefill.cpp#L911)<!-- prefix: "execute_laye" --> · [Release](../src/decoder_prefill.cpp#L799)<!-- prefix: "cudaError_t " -->

Predecode follows each lane's page table and expands each complete MXFP8 page once. Separate K and V TMA producers load alternating shared-memory stages. A singleton with a partial page uses the native masked conversion. See [the paired kernel](flash-prefill.md#long-tile-tmem-pipeline) for arithmetic and tile geometry. [Predecode](../include/sglang/text/mxfp8_gqa_prefill_paired.cuh#L97)<!-- prefix: "__global__ v" --> · [Producers](../include/sglang/text/mxfp8_gqa_prefill_paired.cuh#L168)<!-- prefix: "__device__ v" -->

The [CUDA 13.4 stream ordering guide](https://docs.nvidia.com/cuda/archive/13.4.3/cuda-programming-guide/02-basics/asynchronous-execution.html#cuda-stream-ordering) defines the execution order. The [CUTLASS SM100 MLA change](https://github.com/NVIDIA/cutlass/pull/2472) illustrates a two-CTA asynchronous attention pipeline. The native route uses its own MXFP8 layout and arithmetic.

## Constraints and failure behavior

Missing or undersized scratch, capture, and known setup rejection decline staging before submission. Setup rejection includes unavailable kernel resources and tensor-map rejection or descriptor out-of-memory. Batch staging also requires an explicit CUDA stream in the current context. Its output and scratch ranges must be disjoint from each other and from live inputs across all four lanes. [Capture check](../include/sglang/text/mxfp8_gqa_prefill_paired.cuh#L450)<!-- prefix: "const auto c" --> · [Stream check](../include/sglang/text/mxfp8_gqa_prefill_staged.cuh#L132)<!-- prefix: "std::optiona" --> · [Range check](../include/sglang/text/mxfp8_gqa_prefill_paired.cuh#L476)<!-- prefix: "if (!valid_s" --> · [Tensor maps](../include/sglang/text/mxfp8_gqa_prefill_paired.cuh#L497)<!-- prefix: "const auto e" -->

A recognized submission rejection declines staging at either launch. A batch decline runs the usual singleton dispatcher on the same stream. If the attention launch declined after predecode, that stream completes the queued predecode before a fallback can reuse the spans. A singleton decline tries raw paired attention. Pending and unexpected CUDA errors propagate. [Submission handling](../include/sglang/text/mxfp8_gqa_prefill_staged.cuh#L163)<!-- prefix: "std::optiona" --> · [Batch fallback](../src/decoder_prefill.cpp#L549)<!-- prefix: "for (int lan" -->

A paired graph capture records the raw kernel with maps by value and uses zero staging scratch. Keep the captured cache, page table, queries, and output valid through replay. [Raw launch](../include/sglang/text/mxfp8_gqa_prefill_paired.cuh#L390)<!-- prefix: "std::optiona" --> · [CUDA graph capture](https://docs.nvidia.com/cuda/archive/13.4.2/cuda-programming-guide/04-special-topics/cuda-graphs.html#stream-capture)

## Tiled prefill under saturation

The tiled planner in the earlier B200 K0 runtime required a positive step cap, prefix reuse off, and the wide SM100 override off. Selected media, draft, or restored-prefix lanes disabled tiling. See pinned Git history at `8b4e3f62e0a7a053ffdefe85da99a841fabeb5d3` for that measured source and settings.

The planner reserved one row per ready decode lane. It used tiled scheduling when the budget fit at least one full tile and more lanes could supply full tiles. The planner scanned from the rotating cursor and granted one full tile or finishing tail per lane when the rows fit. The next step started at the first deferred lane.

With an 8,192-row budget and no decode reservations, a saturated step can dispatch eight physical 1,024-row tiles. Fair share gives 512 rows to each of 16 equally ready lanes or 256 to each of 32. Full tiles pack the same number of rows into fewer grants. At eligible positions, a full grant can use the [1,024-row attention path](#runtime-path). The planner uses ready work and the remaining budget. Per-lane token order stays fixed, and lanes can finish prefill in a different order.

## Multi-request text prefill

The earlier K0 layerwise path grouped consecutive text grants in slot order. Each grant fit one original tile and contributed at least 256 rows in multiples of 128. Each initial group had at least two grants and at most 1,023 rows. At exactly 8,192 prompt rows, consecutive groups could share projections and the MLP across up to 2,048 rows. Each grant kept separate positions, recurrent state, KV pages, and final hidden and residual rows.

Earlier mixed steps could also group whole grants of at least 256 rows. Group selection rounded each grant up to a multiple of 128 when checking the 1,023-row limit. Projections used the sum of real rows. GDN used each segment's original history and carry.

The earlier step reused one stream-ordered arena. If wide allocation was unavailable, the step retried the initial group capacity and then the scalar capacity. These retries occurred before prefill submission.

Compatible GQA grants could share one CUDA launch. Each grant kept its query tiles, page table, cache position, and output slice. On B200, two or three 256-row grants could try ordered ragged TMEM when every start was at least 4,096.

### Whole-tile projection pairs

In the earlier K0 path, a pure 8,192-row prefill step could pair two consecutive 1,024-row grants. A 2,048-row grant with two 1,024-row tiles ran per lane. Mixed steps and steps with fewer rows used the 1,023-row group limit.

The earlier pair shared FP8 projections and the MLP across 2,048 rows. It ran
GQA and GDN one lane at a time in stream order. The lane views started at rows
0 and 1,024. The current projection path is described under
[Projection pairs for MTP](#projection-pairs-for-mtp).

Eligible GQA lanes reuse the base MLP and GDN scratch spans for staged attention. The 2,048-row arena supplies 136 MiB for K and 128.375 MiB for V and flags. The spans lie outside the live query, key, value, and output slices. The next lane reuses them in stream order. [Scratch spans](../src/decoder_prefill.cpp#L199)<!-- prefix: "PairedPrefil" -->

The step reuses one 487.44 MiB arena across all pairs. A 1,024-row arena uses 243.72 MiB. If wide pool creation or allocation reports `cudaErrorMemoryAllocation` or `cudaErrorNotSupported`, the step lowers the group limit to 1,023 and retries before submission. A second failure can reduce capacity to the largest per-lane tile. Without a workspace, non-cached work uses the single-row path. Tensor-prefix caching requires a workspace for tiles larger than one row. Unexpected CUDA errors propagate. [Unavailable allocation](../src/decoder_prefill.cpp#L160)<!-- prefix: "bool prefill" --> · [Fallback](https://github.com/para1lel/sglang.cpp/blob/93648269d852e0fee78f7af4d1c3ffe4897aa361/src/decoder_prefill.cpp#L947)<!-- prefix: "if (!workspa" -->

### Projection pairs for MTP

On B200 with prefix reuse off, a pure tensor-core text prefill step can share
FP8 projections across grants. The configured step budget must be positive
and at most 8,192 tokens. The grants must total at most 8,192 rows.
The step needs an even number of grants with draft state. All grants must have
the same length in full 1,024-row tiles. A step with decode, restored prompt work,
media, label scoring, prompt-logprob capture, exact grants, or irregular
grant lengths follows ordinary dispatch.
[Step classification](../src/decoder_batch.cpp#L596)<!-- prefix: "if (max_rows" --> · [Step dispatch](../src/decoder_batch.cpp#L661)<!-- prefix: "if (count &&" --> · [Grant eligibility](../include/sglang/decoder_prefill.h#L84)<!-- prefix: "inline bool " -->

Exactly four eligible grants first try one 4,096-row group per tile offset.
Four 2,048-row grants use two groups in one step. Two 4,096-row grants use
four 2,048-row groups. Other eligible grant counts also use pairs.
Lanes share dense projections and the MLP. Attention and GDN keep each lane's
1,024-row tiles, positions, KV pages, and recurrent state. Draft seeding uses
the lane's own hidden and residual rows and shifted token IDs.
[Group selection](../src/decoder_prefill.cpp#L701)<!-- prefix: "if (count ==" --> · [Draft seed](../src/decoder_prefill.cpp#L654)<!-- prefix: "auto seed = " -->

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
Unexpected CUDA errors propagate. One large workspace pool per device can
retain backing across synchronization with a 1 GiB CUDA release threshold.
The threshold controls backing retention. Request workspace accounting counts
the active allocation.
[Workspace](../src/decoder_prefill.cpp#L673)<!-- prefix: "PrefillWorks" --> · [Allocation](../src/decoder_prefill.cpp#L744)<!-- prefix: "PrefillWorks" --> · [Pool threshold](../src/decoder_prefill.cpp#L60)<!-- prefix: "std::uint64_" -->

With `--trace-batches`, `layerwise_prefill_groups` counts the groups,
`rows` counts their combined rows, and `workspace_bytes` records the
largest active prefill workspace in the step.
[Batch trace](../include/sglang/server/batch_trace.h#L44)<!-- prefix: "if (stats.la" -->

## Verification

The `decoder-batch-plan-check` CPU target checks the step budget, decode reservations, finishing tails, rotation under replenished load, and fair-share fallback. Enable it with `xmake f --decoder-batch-plan-check=y`. [Planner test target](../xmake.lua#L285)<!-- prefix: "target(\"deco" -->

On one reserved B200, replay identical checkpoint Q, MXFP8 payloads and scales, and remapped page tables. Exercise both bounds and nonzero page offsets. Compare full BF16 output with raw paired and archived wide Flash. Include a short tail, causal masks, invalid full- and partial-page V, and repair.

Check each scratch span's bounds and canaries, including a span one byte shorter than the aligned requirement. Check descriptor rejection, empty cluster occupancy, and unavailable predecode. Require raw output parity in those cases.

For four-grant staging, compare each lane's BF16 output bitwise with singleton staging at each eligible start. Cover lane-specific invalid KV and repair, aliased outputs or scratch, a mismatched start, and a rejected stream. Reject each launch in turn with a known configuration status and compare fallback output. Inject an unexpected attention launch error after predecode and require propagation without a second attention launch.

Check a cold capture with scratch supplied. Require one raw kernel node and compare replay after a selector change. Check explicit `paired` and the default dispatch boundaries. Count allocation calls on each arm.

Time eager predecode and attention together with resident inputs in base, candidate, candidate, base order. Nine matched checkpoint captures gave byte-identical BF16 and 13.51–21.93% lower operator latency. This evidence covers operator output, dispatch, and allocation calls. Decoder layer lifetime and request performance require separate evidence.

For mixed projection groups, hold admissions, grants, and generated history fixed while comparing full BF16 score rows and terminal records. Repeat with poisoned workspace allocations, cancel a request after it participates in a ragged group, then reuse its lane with a different prompt. Use independent streaming POSTs to confirm the mixed route activates and record their actual grants when comparing output or timing.

For wide groups, compare complete decoder advances at 8,192 prompt rows. Include unequal, ragged, singleton, and partial final grants. Compare full BF16 scores and per-grant state, and reject the wide arena allocation to exercise the initial groups.

For whole-tile pairs, check both lane offsets and cover mixed, partial, and per-lane split grants. The host check modeled allocation failures and checked grouping against the scheduler-only path. A device audit matched 16,384,000 sampled BF16 hidden and residual values across aligned and shifted positions at four layers. Full logits and recurrent state require separate checks. See `artifacts/takeover-native-tile-pair/review.md`.

For four-grant grouping, check four equal grants and a two-grant control on
the same native policy. Compare target hidden rows, logits, KV, recurrence,
and subsequent draft state. Check uneven grants, a mixed decode step,
and a configured budget above 8,192. Reject the 4,096-row allocation, confirm
paired fallback, then remove the failure and confirm four-grant recovery.
Record output IDs, active workspace bytes, and physical GPU memory separately.

Two mirrored W4 comparisons on B200 GPU 7 kept the tiled scheduler in both arms. Each arm used 16 lanes with 32,768 prompt IDs and a 512-output cap. With pairing, the last prefill completed 2.176 s and 1.261 s earlier. Client makespan fell by 0.178 s and 1.110 s. All 32,768 returned IDs across the four arms matched the frozen prefixes. Admission and decode overlap varied across arms. W3 exercised zero pairs across 12 cohorts. W5 and W6 pairing evidence in this review comes from host replay. See `artifacts/takeover-w4-combined/half-output-report.md` and `artifacts/takeover-native-tile-pair/review.md`.
