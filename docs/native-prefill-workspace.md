# Native prefill workspace

## Responsibilities and sources of truth

| Part | Responsibility | Source |
| --- | --- | --- |
| Decoder prefill | Own the arena and lend two scratch spans during GQA. | [Tile layout](../src/decoder_prefill.cpp#L175)<!-- prefix: "TileBuffers(" --> · [Scratch view](../src/decoder_prefill.cpp#L197)<!-- prefix: "PairedPrefil" --> |
| Native Flash dispatcher | Try staged attention within the default paired route. | [Selection](../src/mxfp8_gqa_prefill.cu#L938)<!-- prefix: "const bool d" --> |
| Paired launcher | Check bounds and capture state, then submit predecode and attention on the caller's stream. | [Staged launch](../include/sglang/text/mxfp8_gqa_prefill_paired.cuh#L389)<!-- prefix: "std::optiona" --> |

## Runtime path

Under `tensor-core`, the native dispatcher can stage a 1,024-row tile at starts 8,192 through 31,744 inclusive. The start must be a multiple of 1,024 or have a nonzero offset within a 128-token page. This uses the default paired selection with `SGLANG_EXPERIMENTAL_TMEM_PREFILL` unset or set to `ragged`. The [SM100 page-128 opt-ins](sm100-page128-prefill.md) run before native dispatch. [Decoder call](../src/decoder_prefill.cpp#L363)<!-- prefix: "const auto s" -->

The decoder lends two disjoint spans of its existing `PrefillWorkspace` allocation. The first address is inclusive and the end address is exclusive. These capacities apply to a 1,024-row arena.

| Scratch use | Physical span | Capacity | Largest contents at start 31,744 |
| --- | --- | ---: | ---: |
| BF16 K | `mlp_gate` through the end of `mlp_up`, bounded by `q_gate` | 68 MiB | 64 MiB |
| BF16 V and invalid-token indices | `gdn_qkv`, `gdn_z`, `gdn_b`, `gdn_a`, and GDN scratch, bounded by `activation.values` | 64.1875 MiB | 64 MiB + 4 KiB |

Each span has room for up to 127 bytes of padding for 128-byte alignment. The launcher checks both spans separately. Workspace allocation size, memory pool setup, and KV cache capacity stay the same. The staged route adds zero device allocation bytes. [Arena allocation](../src/decoder_prefill.cpp#L689)<!-- prefix: "PrefillWorks" --> · [Bounds](../include/sglang/text/mxfp8_gqa_prefill_paired.cuh#L398)<!-- prefix: "const int fu" -->

The MLP projection writes its gate and up buffers after GQA. GDN layers write their projection and scratch inputs before use. Predecode, paired attention, later layer work, and workspace release use the workspace's execution stream. Stream ordering puts the last attention read before the next write to either span. Keep both spans exclusive to that stream until the queued work completes. [Layer loop](../src/decoder_prefill.cpp#L855)<!-- prefix: "execute_laye" --> · [Release](../src/decoder_prefill.cpp#L744)<!-- prefix: "cudaError_t " -->

Predecode follows the page table and expands each complete MXFP8 page once. Separate K and V TMA producers load alternating shared-memory stages. A partial page uses the native masked conversion. The paired kernel retains the ordered BF16 operands, FP32 softmax and accumulation, and protected repair. [Predecode](../include/sglang/text/mxfp8_gqa_prefill_paired.cuh#L77)<!-- prefix: "__global__ v" --> · [Producers](../include/sglang/text/mxfp8_gqa_prefill_paired.cuh#L146)<!-- prefix: "__device__ v" -->

The [CUDA 13.4 stream ordering guide](https://docs.nvidia.com/cuda/archive/13.4.3/cuda-programming-guide/02-basics/asynchronous-execution.html#cuda-stream-ordering) defines the execution order. The [CUTLASS SM100 MLA change](https://github.com/NVIDIA/cutlass/pull/2472) illustrates a two-CTA asynchronous attention pipeline. The native route uses its own MXFP8 layout and arithmetic.

## Constraints and failure behavior

Missing or undersized scratch, capture, and a known staging setup rejection select raw paired attention before staged work enters the stream. Setup rejection includes unavailable kernel resources, supported predecode unavailability statuses, and tensor-map rejection or descriptor out-of-memory. The raw path retains its [existing fallback rules](flash-prefill.md#constraints-and-failure-behavior). [Capture check](../include/sglang/text/mxfp8_gqa_prefill_paired.cuh#L415)<!-- prefix: "const auto c" --> · [Setup and submission](../include/sglang/text/mxfp8_gqa_prefill_paired.cuh#L435)<!-- prefix: "const auto e" -->

Unexpected CUDA failures return to the caller. After predecode submission succeeds, the attention launch status also returns directly. A paired graph capture records the original raw kernel with maps by value and borrows neither span. Keep the captured cache, page table, queries, and output valid through replay. [Raw launch](../include/sglang/text/mxfp8_gqa_prefill_paired.cuh#L361)<!-- prefix: "std::optiona" --> · [CUDA graph capture](https://docs.nvidia.com/cuda/archive/13.4.2/cuda-programming-guide/04-special-topics/cuda-graphs.html#stream-capture)

## Tiled prefill under saturation

The tiled planner applies to chunked B200 `tensor-core` prefill with a positive step cap, prefix caching off, and the wide SM100 prefill opt-in off. Selected media, draft, or restored-prefix lanes disable tiling. [Route selection](../src/decoder_batch.cpp#L875)<!-- prefix: "const bool t" --> · [Lane checks](../src/decoder_batch.cpp#L259)<!-- prefix: "if (lane->dr" -->

The planner reserves one row per ready decode lane. It uses tiled scheduling when the remaining budget can fit at least one full tile. More lanes must be ready to supply full tiles than the budget can fit. The planner scans from the rotating cursor and grants one full tile or finishing tail per lane when the rows fit. The next step starts at the first deferred lane. Other capped steps use fair share or the existing cache planner. [Tiled planner](../include/sglang/text/decoder_batch_plan.h#L24)<!-- prefix: "inline std::" --> · [Fair share](../include/sglang/text/decoder_batch_plan.h#L52)<!-- prefix: "inline StepP" -->

With an 8,192-row budget and no decode reservations, a saturated step can dispatch eight physical 1,024-row tiles. Fair share gives 512 rows to each of 16 equally ready lanes or 256 to each of 32. Full tiles pack the same number of rows into fewer grants. At eligible positions, a full grant can use the [1,024-row attention path](#runtime-path). The planner uses ready work and the remaining budget. Per-lane token order stays fixed, and lanes can finish prefill in a different order. See the [short W4 and W5 pilots](performance-comparison.md#tiled-prefill-scheduler-pilots).

## Multi-request text prefill

On B200, a tensor-core prefill-only step with a positive budget of at most 8,192 rows can group consecutive text grants in slot order. Each grant must fit one of its original tiles and contribute at least 256 rows in multiples of 128. Each initial group has at least two grants and at most 1,023 rows. When the step contains exactly 8,192 prompt rows, consecutive initial groups can share dense projections and the MLP across up to 2,048 rows. Other prefill-only steps keep the initial group limit. Within a projection group, GDN and GQA use the initial group finder. Each grant keeps its positions, recurrent state, KV pages, and final hidden and residual row. [Initial groups](../src/decoder_prefill.cpp#L465)<!-- prefix: "PrefillGroup" --> · [Projection groups](../src/decoder_prefill.cpp#L482)<!-- prefix: "PrefillGroup" --> · [Layer loop](../src/decoder_prefill.cpp#L597)<!-- prefix: "void prefill" -->

Eligible text steps that contain both prefill and decode can also group whole grants of at least 256 rows. For mixed steps, group selection rounds each grant up to a multiple of 128 when checking the 1,023-row limit. Embedding gathers write consecutive real rows, and projections use their exact sum. GDN uses each original segment's history and carry. A single grant or a group with an irregular segment uses lane-local GDN recurrence. The worker retains the original physical tiles, token positions, and decode selection. [Mixed step guard](../src/decoder_batch.cpp#L757)<!-- prefix: "const bool m" --> · [Group limit](../src/decoder_prefill.cpp#L465)<!-- prefix: "PrefillGroup" --> · [Real row offsets](../src/decoder_prefill.cpp#L505)<!-- prefix: "void each_pr" --> · [GDN eligibility](../src/decoder_prefill.cpp#L563)<!-- prefix: "bool grouped" -->

The step reuses one stream-ordered arena. Its row capacity covers the largest projection group or original scalar tile. Grouped work writes every real row before use and needs no per-segment padding clears. If the wide allocation is unavailable, the step retries the initial group capacity. If that allocation is also unavailable, it retries the scalar capacity. These retries happen before prefill submission. Tensor-prefix caching, MTP, media, and ineligible grants retain their existing paths. Use `--trace-batches` to inspect projection group counts and rows alongside the original grants. [Step guard](../src/decoder_batch.cpp#L758)<!-- prefix: "const bool l" --> · [Allocation fallback](../src/decoder_prefill.cpp#L898)<!-- prefix: "if (!workspa" -->

Within an initial group, compatible GQA grants can share one CUDA launch. Each grant keeps its query tiles, page table, cache position, and output slice. On B200, two or three 256-row grants can try ordered ragged TMEM when every start is at least 4,096. When the early route is unavailable, dispatch resumes grouped Flash, grouped TMEM, and lane-local selection. The group uses the same arena. [Group dispatch](../src/decoder_prefill.cpp#L524)<!-- prefix: "void read_pa" --> · [Kernel selection](../src/mxfp8_gqa_prefill.cu#L815)<!-- prefix: "std::optiona" -->

### Whole-tile projection pairs

Within the wide B200 tensor-core path, a pure 8,192-row prefill step can pair two consecutive grants in compacted slot order. Each grant must have 1,024 rows and a tile size of 1,024 rows. A 2,048-row grant that splits into two 1,024-row tiles runs per lane. Mixed steps and steps with fewer prefill rows use the 1,023-row group limit. The existing layerwise guard excludes tensor-prefix cache, draft, and media work. [Shared tile size](../include/sglang/decoder_prefill.h#L9)<!-- prefix: "inline const" --> · [Wide limit](../src/decoder_prefill.cpp#L893)<!-- prefix: "int max_rows" --> · [Pair selection](../src/decoder_prefill.cpp#L484)<!-- prefix: "if (!mixed &" -->

The pair shares FP8 projections and the MLP across 2,048 rows. GQA and GDN run one lane at a time in stream order. The lane views start at rows 0 and 1,024. Each GQA lane keeps its position, page table, KV cache, and output slice. Each GDN lane uses its own recurrent state and row slices for the projected inputs and FP8 output. [Attention offsets](../src/decoder_prefill.cpp#L519)<!-- prefix: "offset += gr" --> · [GDN dispatch](../src/decoder_prefill.cpp#L563)<!-- prefix: "bool grouped" -->

Eligible GQA lanes reuse the base MLP and GDN scratch spans for staged attention. The 2,048-row arena supplies 136 MiB for K and 128.375 MiB for V and flags. The spans lie outside the live query, key, value, and output slices. The next lane reuses them in stream order. [Scratch spans](../src/decoder_prefill.cpp#L197)<!-- prefix: "PairedPrefil" --> · [Singleton GQA](../src/decoder_prefill.cpp#L555)<!-- prefix: "read_paged_s" -->

The step reuses one 487.44 MiB arena across all pairs. A 1,024-row arena uses 243.72 MiB. If wide pool creation or allocation reports `cudaErrorMemoryAllocation` or `cudaErrorNotSupported`, the step lowers the group limit to 1,023 and retries before submission. A second failure can reduce capacity to the largest per-lane tile. Without a workspace, non-cached work uses the single-row path. Tensor-prefix caching requires a workspace for tiles larger than one row. Unexpected CUDA errors propagate. [Unavailable allocation](../src/decoder_prefill.cpp#L158)<!-- prefix: "bool prefill" --> · [Fallback](../src/decoder_prefill.cpp#L898)<!-- prefix: "if (!workspa" -->

## Verification

The `decoder-batch-plan-check` CPU target checks the step budget, decode reservations, finishing tails, rotation under replenished load, and fair-share fallback. Enable it with `xmake f --decoder-batch-plan-check=y`. [Planner test target](../xmake.lua#L281)<!-- prefix: "target(\"deco" -->

On one reserved B200, replay identical checkpoint Q, MXFP8 payloads and scales, and remapped page tables. Exercise both bounds and nonzero page offsets. Compare full BF16 output with raw paired and archived wide Flash. Include a short tail, causal masks, invalid full- and partial-page V, and repair.

Check each scratch span's bounds and canaries, including a span one byte shorter than the aligned requirement. Check descriptor rejection, empty cluster occupancy, and unavailable predecode. Require raw output parity in those cases. Inject a launch failure after predecode and require the error with no second attention launch.

Check a cold capture with scratch supplied. Require one raw kernel node and compare replay after a selector change. Check explicit `paired` and the default dispatch boundaries. Count allocation calls on each arm.

Time eager predecode and attention together with resident inputs in base, candidate, candidate, base order. Nine matched checkpoint captures gave byte-identical BF16 and 13.51–21.93% lower operator latency. This evidence covers operator output, dispatch, and allocation calls. Decoder layer lifetime and request performance require separate evidence.

For mixed projection groups, hold admissions, grants, and generated history fixed while comparing full BF16 score rows and terminal records. Repeat with poisoned workspace allocations, cancel a request after it participates in a ragged group, then reuse its lane with a different prompt. Use independent streaming POSTs to confirm the mixed route activates and record their actual grants when comparing output or timing.

For wide groups, compare complete decoder advances at 8,192 prompt rows. Include unequal, ragged, singleton, and partial final grants. Compare full BF16 scores and per-grant state, and reject the wide arena allocation to exercise the initial groups.

For whole-tile pairs, check both lane offsets and cover mixed, partial, and per-lane split grants. The host check modeled allocation failures and checked grouping against the scheduler-only path. A device audit matched 16,384,000 sampled BF16 hidden and residual values across aligned and shifted positions at four layers. Full logits and recurrent state require separate checks. See `artifacts/takeover-native-tile-pair/review.md`.

Two mirrored W4 comparisons on B200 GPU 7 kept the tiled scheduler in both arms. Each arm used 16 lanes with 32,768 prompt IDs and a 512-output cap. With pairing, the last prefill completed 2.176 s and 1.261 s earlier. Client makespan fell by 0.178 s and 1.110 s. All 32,768 returned IDs across the four arms matched the frozen prefixes. Admission and decode overlap varied across arms. W3 exercised zero pairs across 12 cohorts. W5 and W6 pairing evidence in this review comes from host replay. See `artifacts/takeover-w4-combined/half-output-report.md` and `artifacts/takeover-native-tile-pair/review.md`.
