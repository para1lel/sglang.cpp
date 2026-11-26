# Native prefill workspace

## Responsibilities and sources of truth

| Part | Responsibility | Source |
| --- | --- | --- |
| Decoder prefill | Lay out the arena and lend two scratch spans during GQA. | [Tile layout](../include/sglang/runtime/execution/decoder_prefill_tiles.h#L139)<!-- prefix: "TileBuffers(" --> · [Scratch view](../include/sglang/runtime/execution/decoder_prefill_tiles.h#L162)<!-- prefix: "PairedPrefil" --> |
| Attention dispatch | Stage eligible singleton and four-lane groups. Partial-grant groups can use staged, raw grouped, or per-lane attention. | [Singleton selection](../include/sglang/text/gqa/multirow_prefill.cuh#L35)<!-- prefix: "const bool d" --> · [Grouped dispatch](../src/decoder_prefill.cpp#L363)<!-- prefix: "void prefill" --> |
| Paired launcher | Check bounds and capture state, then submit predecode and attention on the caller's stream. | [Staged launch](../include/sglang/text/mxfp8_gqa_prefill_paired.cuh#L784)<!-- prefix: "std::optiona" --> |

## Runtime path

The B200 text dispatcher lends these spans to eligible eager 1,024-row tiles and [partial-grant subgroups](ordered-gqa-prefill.md#grouped-partial-grants). Ordinary 1,024-row dispatch at start zero uses raw paired attention and leaves staging scratch unused. See [attention dispatch](flash-prefill.md#runtime-path) for full-tile bounds. [Singleton bound](../include/sglang/text/mxfp8_gqa_prefill_paired.cuh#L872)<!-- prefix: "if (position" -->

The decoder lends two disjoint spans from the existing 1,024-row workspace. Starts are inclusive and ends are exclusive.

| Scratch use | Physical span | Capacity | Largest contents at start 31,744 |
| --- | --- | ---: | ---: |
| BF16 K | `mlp_gate` through the end of `mlp_up`, bounded by `q_gate` | 68 MiB | 64 MiB |
| BF16 V and invalid-token indices | `gdn_qkv`, `gdn_z`, `gdn_b`, `gdn_a`, and GDN scratch, bounded by `activation.values` | 64.1875 MiB | 64 MiB + 4 KiB |

Both spans allow up to 127 bytes of 128-byte alignment padding. The four-tile arena supplies 272 MiB for K and 256.75 MiB for V and invalid-token indices. A [staged subgroup](flash-prefill.md#runtime-path) covers at most 131,072 visible tokens in total. It uses up to 256 MiB for K and 256 MiB plus 16 KiB for V and indices before padding. The grouped partial reader rounds each visible span up to full pages and puts each lane at its own offset. Both scratch spans must fit the combined page count. Subgroups reuse both spans on one stream without another device allocation. [Arena allocation](../src/runtime/execution/decoder_prefill_workspace.cpp#L214)<!-- prefix: "PrefillWorks" --> · [Scratch sizing](../include/sglang/text/mxfp8_gqa_prefill_paired.cuh#L831)<!-- prefix: "const auto p" -->

GQA scratch stays exclusive to the workspace stream through the last read. MLP and GDN overwrite their buffers later on that stream. The workspace also releases on that stream. [Layer loop](../src/decoder_prefill.cpp#L638)<!-- prefix: "execute_laye" --> · [Release](../src/runtime/execution/decoder_prefill_workspace.cpp#L382)<!-- prefix: "cudaError_t " -->

Predecode follows each lane’s page table and expands visible MXFP8 entries to BF16. Grouped partial staging zero-fills entries beyond that lane’s visible end. Separate K and V TMA producers load alternating shared-memory stages. See [the paired kernel](flash-prefill.md#long-tile-tmem-pipeline) for arithmetic and tile geometry. [Predecode](../include/sglang/text/mxfp8_gqa_prefill_paired.cuh#L99)<!-- prefix: "__global__ v" --> · [Producers](../include/sglang/text/mxfp8_gqa_prefill_paired.cuh#L187)<!-- prefix: "__device__ v" -->

The [CUDA 13.4 stream ordering guide](https://docs.nvidia.com/cuda/archive/13.4.3/cuda-programming-guide/02-basics/asynchronous-execution.html#cuda-stream-ordering) defines the execution order. The [CUTLASS SM100 MLA change](https://github.com/NVIDIA/cutlass/pull/2472) illustrates a two-CTA asynchronous attention pipeline. The native route uses its own MXFP8 layout and arithmetic.

## Constraints and failure behavior

The workspace allocator reuses private CUDA pools by device and size class, separating workspaces at or below 128 rows from larger ones. At most one pool per device and class uses a non-default release threshold, 32 MiB for small pools and 1 GiB for large ones. CUDA can retain that backing after synchronization and request retirement. `workspace_bytes` counts active workspace bytes, including staging spans. Use device memory measurements to observe retained backing. [Pool policy](../src/runtime/execution/decoder_prefill_workspace.cpp#L50)<!-- prefix: "void configu" --> · [Allocation](../src/runtime/execution/decoder_prefill_workspace.cpp#L214)<!-- prefix: "PrefillWorks" -->

Out-of-memory or unsupported pool creation or allocation leaves the workspace unavailable. Ordinary prefill retries an unavailable large allocation with 128 rows, then uses row-wise prefill if scratch remains unavailable. Single-row tiles also use row-wise prefill. Other CUDA failures propagate. A terminal direct status takes precedence over a distinct recorded status. Otherwise a distinct recorded failure takes precedence over an allocation decline. Release enqueues a free on the workspace stream. If enqueue fails, recovery synchronizes the owning device, checks used bytes, and waits for a retried free when required. Explicit close reports the original failure. [Unavailable statuses](../src/runtime/execution/decoder_prefill_workspace.cpp#L163)<!-- prefix: "bool prefill" --> · [Status precedence](../src/runtime/execution/decoder_prefill_workspace.cpp#L41)<!-- prefix: "void check_p" --> · [Ordinary retry](../src/decoder_prefill.cpp#L920)<!-- prefix: "if (!workspa" --> · [Release](../src/runtime/execution/decoder_prefill_workspace.cpp#L382)<!-- prefix: "cudaError_t " --> · [Recovery](../src/runtime/execution/decoder_prefill_workspace.cpp#L146)<!-- prefix: "void recover" -->

Missing or undersized scratch, capture, and known setup rejection decline staging before submission. Ordinary staging setup can decline for unavailable kernel resources, tensor-map rejection, or descriptor out-of-memory. Batch staging also requires an explicit CUDA stream in the current context. Its output and scratch ranges must be disjoint from each other and from live inputs within a subgroup. [Capture check](../include/sglang/text/mxfp8_gqa_prefill_paired.cuh#L846)<!-- prefix: "const auto c" --> · [Stream check](../include/sglang/text/mxfp8_gqa_prefill_staged.cuh#L140)<!-- prefix: "std::optiona" --> · [Range check](../include/sglang/text/mxfp8_gqa_prefill_paired.cuh#L858)<!-- prefix: "if (!valid_s" --> · [Tensor maps](../include/sglang/text/mxfp8_gqa_prefill_paired.cuh#L727)<!-- prefix: "const auto e" -->

The [long whole-tile page reader](flash-prefill.md#long-tile-tmem-pipeline) propagates submission errors. Other staged routes can decline on a recognized launch rejection. Full-tile subgroups fall back lane by lane. Ordinary singleton staging still falls back to raw paired attention. Partial subgroups try [raw grouped ordered TMEM](ordered-gqa-prefill.md#grouped-partial-grants) when eligible, then fall back lane by lane. If attention declines after predecode queues, the stream orders fallback and later reuse after predecode. Pending and unexpected CUDA errors propagate. [Submission handling](../include/sglang/text/mxfp8_gqa_prefill_staged.cuh#L180)<!-- prefix: "std::optiona" --> · [Full-tile fallback](../include/sglang/runtime/execution/decoder_prefill_staged_gqa.h#L41)<!-- prefix: "for (int ind" -->

A paired graph capture records the raw kernel with maps by value and uses zero staging scratch. Keep the captured cache, page table, queries, and output valid through replay. [Raw launch](../include/sglang/text/mxfp8_gqa_prefill_paired.cuh#L609)<!-- prefix: "std::optiona" --> · [CUDA graph capture](https://docs.nvidia.com/cuda/archive/13.4.2/cuda-programming-guide/04-special-topics/cuda-graphs.html#stream-capture)

## Multi-request text prefill

The [batch planner](batch-graphs.md#runtime-path) assigns rows. The executor splits grants into physical tiles of up to 1,024 rows and groups compatible work. Each request keeps its token order, KV pages, recurrent state, and output rows. [Attention dispatch](flash-prefill.md#runtime-path)

The earlier K0 planner tested fair-share, tiled, and grouped schedules. Its measured cases used the historical selector and K0 rules. Recover the planner and run notes at the original revisions. [K0 campaign](https://github.com/para1lel/sglang.cpp/blob/8b4e3f62e0a7a053ffdefe85da99a841fabeb5d3/docs/serving/paired-measurement-ledger.md) · [Previous workspace guide](https://github.com/para1lel/sglang.cpp/blob/4dd8c887c278c67744ed80e7feb2a825a92197a2/docs/native-prefill-workspace.md#tiled-prefill-under-saturation)

The scheduler sends generation and scoring grants to one prefill executor. B200 can group compatible Tensor Core tiles. [Cache restore](prefix-cache.md#checkpoint-compatibility-and-resume) supplies the starting state. A fully covered GQA tile projects only queries and gate values. Other tiles append KV only beyond the restored coverage. Media, rotary positions, and score captures stay lane-local. Decode work has separate dispatch. [Step dispatch](../src/decoder_batch.cpp#L380)<!-- prefix: "if (max_rows" --> · [Grant assembly](../src/decoder_batch.cpp#L411)<!-- prefix: "void Decoder" --> · [Read-only GQA](../include/sglang/runtime/cache/prefill_cached_gqa.h#L14)<!-- prefix: "void prepare" -->

### Projection pairs for MTP

Whole-tile grouping takes priority on B200. The grants must be equal and consist entirely of native 1,024-row tiles. With exactly four grants, the decoder first tries one 4,096-row projection group at each tile offset. Four 8,192-row grants use eight successive groups in one 32,768-row step. Each lane uses its own absolute start, KV pages, recurrent state, and draft state. [Tile eligibility](../include/sglang/decoder_prefill.h#L120)<!-- prefix: "inline bool " --> · [Group selection](../src/decoder_prefill.cpp#L490)<!-- prefix: "bool decoder" --> · [Tile offsets](../src/decoder_prefill.cpp#L464)<!-- prefix: "for (int off" --> · [Draft seed](../src/decoder_prefill.cpp#L69)<!-- prefix: "auto seed = " -->

Pairs require an even grant count and at most 8,192 rows across all grants. Two 4,096-row grants use four 2,048-row groups. The [`paired_prefill` field](native-model-info.md#server-information) reports startup capability. [Pair limit](../src/decoder_prefill.cpp#L500)<!-- prefix: "if (grants[0" -->

[Grouped GDN](gdn-prefill.md#runtime-path) reuses the existing disjoint row
spans for both projection group sizes. The group uses the workspace stream.
Four grants with one aligned start can use [batched staged GQA](flash-prefill.md#runtime-path). With mixed starts, eligible pairs of distinct text requests stage together. An unmatched long tile can stage alone. Both routes reuse the 4,096-row projection workspace and keep each owner’s page table and output rows. [Attention dispatch](../src/decoder_prefill.cpp#L52)<!-- prefix: "plan.staged_" -->

| Workspace | Active bytes |
| --- | ---: |
| Ordinary 1,024-row tile | 255,557,632 |
| Two-grant, 2,048-row group | 511,115,264 |
| Four-grant, 4,096-row group | 1,022,230,528 |

The step reuses one workspace across all groups. Four-grant grouping adds 511,115,264 active bytes over pairs. If the 4,096-row workspace is unavailable, the decoder tries pairs only within the 8,192-row pair limit. If the grants exceed that limit, dispatch tries the existing partial-grant and causal-segment planner. That planner can request its own 4,096-row arena. Ordinary prefill takes over if grouping remains unavailable. If an eligible pair cannot allocate its workspace, dispatch goes straight to ordinary prefill. [Group workspace](../src/decoder_prefill.cpp#L459)<!-- prefix: "PrefillWorks" --> · [Partial and causal fallback](../src/decoder_prefill.cpp#L505)<!-- prefix: "return execu" -->

### Paired 2,048-token TP2 prompts

B200 TP2 groups two distinct complete 2,048-token cold grants with Tensor Core prefill and native GDN on both ranks. Four 1,024-row tiles share projections. Attention and recurrence advance in request order. MTP seed retains stock draft FC and cache format. [Eligibility](../include/sglang/runtime/execution/decoder_prefill_plan.h#L82)<!-- prefix: "inline bool " --> · [Dispatch](../src/decoder_prefill.cpp#L829)<!-- prefix: "bool try_pre" -->

Each rank allocates 1,022,230,528 transient bytes before submission and frees them on its stream. If peer allocation declines, rank 0 releases its provisional workspace and synchronizes before fallback. CUDA errors propagate. [Workspace](../src/runtime/execution/decoder_prefill_workspace.cpp#L217)<!-- prefix: "PrefillWorks" -->

### Seed scratch

MTP draft seeding uses a full-geometry scratch outside the rank-local tile layout. The draft model runs full geometry on the primary rank. Paired seeding allocates `concatenated` for up to 128 rows and `keys`, `values`, and `rope_keys` for every row of one paired seed launch. Each seeding path throws `runtime_error` when the scratch does not fit the workspace. [Seed scratch](../include/sglang/runtime/execution/decoder_prefill_tiles.h#L126)<!-- prefix: "inline std::" -->

### Partial-grant projection groups

The partial-grant planner can pack consecutive complete grants up to 1,024 rows each. The minimum is 256 rows on one rank and 128 rows on two ranks. Each grant must fit its own native tile. A group needs at least two grants and fits at most 4,096 rows. The planner leaves grant order intact. Ungrouped grants use ordinary prefill. [Plan and dispatch](../include/sglang/runtime/execution/decoder_prefill_plan.h#L111)<!-- prefix: "bool execute" -->

The group shares dense projections and MLP work. Uniform 4,096-row groups of eight 512-row or sixteen 256-row grants can also share [RMSNorm and GQA postgate preparation](fp8-prefill-projections.md#attention-output-quantization). [Grouped staged and ordered GQA](ordered-gqa-prefill.md#grouped-partial-grants) and [grouped GDN](gdn-prefill.md#grouped-partial-grants) qualify independently. Other attention and recurrence use per-lane dispatch. Each lane keeps its positions, KV pages, and recurrent state. All work uses the same stream and existing workspace. Draft seeding uses the lane's own rows where applicable. [Layer dispatch](../src/decoder_prefill.cpp#L363)<!-- prefix: "void prefill" --> · [GQA selection](../include/sglang/runtime/execution/decoder_prefill_tiles.h#L858)<!-- prefix: "} else if (p" -->

One arena fits the largest planned group and ordinary tile in the step. The executor reuses it across grants. Eligible TP2 tiles can add [long GQA staging](ordered-gqa-prefill.md#long-serial-tp2-tiles). If no group qualifies or its allocation is unavailable, the executor falls back before submitting grouped work. [Allocation and fallback](../include/sglang/runtime/execution/decoder_prefill_plan.h#L133)<!-- prefix: "PrefillWorks" -->

With `--trace-batches`, `layerwise_prefill_groups` counts the groups,
`rows` counts their combined rows, and `workspace_bytes` records the
largest active prefill workspace in the step.
[Batch trace](../include/sglang/server/batch_trace.h#L46)<!-- prefix: "if (stats.la" -->

### Causal segments within one grant

Whole-tile pairs and complete distinct-grant groups take precedence. A Tensor Core grant with 1,024-row tiles packs four when at least four remain, or three when exactly three remain. Each layer shares projections and MLP work. GDN advances in prompt order. Eligible four-tile groups try [staged same-request GQA](flash-prefill.md#same-request-causal-gqa), even if peer grant sizes differ. Other shapes append and read GQA KV per segment. Ordinary prefill resumes at the first unprocessed row within the same grant.
[Segment selection](../include/sglang/runtime/execution/decoder_prefill_plan.h#L40)<!-- prefix: "inline Prefi" --> · [GQA selection](../include/sglang/runtime/execution/decoder_prefill_tiles.h#L857)<!-- prefix: "if (prefill_" --> · [GDN order](../src/decoder_prefill.cpp#L437)<!-- prefix: "prefill_part" --> · [Native remainder](../include/sglang/runtime/execution/decoder_prefill_plan.h#L163)<!-- prefix: "if (offset \u003c" -->

Causal segments reuse the same arena sizing and allocation fallback. Unexpected CUDA errors propagate.

For TP2, the peer uses the source grant index and offset to mirror each tile’s token span. [Peer segments](../include/sglang/runtime/execution/decoder_prefill_plan.h#L102)<!-- prefix: "inline void " -->

### TP2 causal prefill reduction fusion

Eager TP2 fuses reduction, RMSNorm, and E4M3 quantization for 4,096-row Tensor Core causal groups. It needs aligned operands, two distinct SM100 devices, bidirectional CUDA peer access at performance rank 0 with native peer atomics, and access to both workspace pools. First-input normalization stays local. The final MLP reduces separately. Capture, unsupported setup, or misalignment retains ordinary NCCL and local normalization. [Caller gate](../src/decoder_prefill.cpp#L330)<!-- prefix: "if (plan.ful" --> · [Peer setup](../src/runtime/execution/decoder_prefill_workspace.cpp#L310)<!-- prefix: "void Prefill" --> · [NCCL fallback](../src/decoder_prefill.cpp#L128)<!-- prefix: "if (peer_nor" -->

Both streams record projection-ready events before either waits for its peer. Each rank reads both projections and rounds their sum to BF16 before residual addition, normalization, and quantization. Both streams record read-complete events before either can reuse its projection buffer. Static review confirmed that submission errors synchronize both streams and prioritize terminal CUDA status. [Event ordering](../src/runtime/execution/decoder_prefill_workspace.cpp#L335)<!-- prefix: "bool Prefill" --> · [BF16 sum](../include/sglang/text/quantization/prefill_rms_vector.cuh#L28)<!-- prefix: "__device__ _" --> · [Error handling](../src/runtime/execution/decoder_prefill_workspace.cpp#L367)<!-- prefix: "check(termin" -->

A bounded HTTP audit replayed two cold 32,768-token requests to six output tokens each. Median summed prefill advance fell from 928.239 to 896.783 ms. Output IDs matched, and all 401 ranked state records matched bitwise across 1,948,759,040 bytes. A separate CUDA trace replaced 1,016 of 1,024 NCCL epochs. Median time from the last projection to both next kernels fell from 134.974 to 89.727 µs. The full W4 makespan remains unmeasured for this change. See `artifacts/tp2-resume/prefill-transport-bc6cce91/` (`serving-final-audit-comparison.json`, `serving-final-trace-comparison.json`).

## Verification

The `decoder-batch-plan-check` CPU target checks decode reservations, MTP budget clipping, prompt order, slot reuse, and full occupancy. Enable it with `xmake f --decoder-batch-plan-check=y`. [Planner assertions](../tests/decoder/decoder_batch_plan_check.cpp#L24)<!-- prefix: "void check_d" --> · [Planner target](../xmake.lua#L321)<!-- prefix: "target(\"deco" -->

On one reserved B200, replay checkpoint Q, MXFP8 payloads and scales, and remapped pages. Compare full BF16 output with raw paired and archived wide Flash. Cover both bounds, nonzero page offsets, a short tail, masks, invalid full- and partial-page V, and repair.

Check span bounds, canaries, and scratch one byte short. Require raw output parity after descriptor rejection, empty cluster occupancy, or unavailable predecode.

Compare four-grant staging bitwise with per-lane dispatch. Cover four-, two-, and one-lane groups, invalid KV, aliasing, mixed starts, and stream rejection. Reject each launch in turn. Check subgroup fallback and later staging. An unexpected attention launch error after predecode must propagate without a second launch. Compare allocation counts and arena high-water marks.

Check a cold capture with scratch supplied. Require one raw kernel node and compare replay after a selector change. Check explicit `paired` and the default dispatch boundaries. Count allocation calls on each arm.

Time eager predecode plus attention in base, candidate, candidate, base order. Archived checkpoint replay matched BF16 bitwise and reduced operator time. Check lifetime and serving separately. [Operator evidence](https://github.com/para1lel/sglang.cpp/blob/4dd8c887c278c67744ed80e7feb2a825a92197a2/docs/native-prefill-workspace.md#verification)

On B200, fix admissions and grants. Compare scores, state, generated IDs, and terminal records against ordinary dispatch. Cover unequal and trailing grants, a singleton, two 256-row grants at K0, and mixed prefill and decode. Check group and row counters. Deny grouped allocation and check ordinary prefill with smaller scratch. Measure active and retained memory separately.

Check whole-tile pairs at both lane offsets, including mixed, partial, and split grants. Compare full logits and recurrent state plus sampled hidden and residual values.

Use the default 32,768-row budget. Check four 8,192-row grants and equal grants with distinct restored starts. Compare scores, target and draft KV and scales, GDN carry and history, hidden, residual, and IDs. Include unequal grants and mixed decode. Deny the 4,096-row workspace at and above the 8,192-row pair limit. Check paired fallback at the limit, causal and ordinary fallback above it, and recovery. Record groups, active workspace, and retained GPU memory.

Historical K0 pairing used reduced-output W4. W3 exercised no pairs. W5 and W6 used host replay. Check grant sizes and admission/decode overlap before attributing serving gains to pairing. [Historical pairing evidence](https://github.com/para1lel/sglang.cpp/blob/4dd8c887c278c67744ed80e7feb2a825a92197a2/docs/native-prefill-workspace.md#verification)

### TP2 paired prefill replay

On B200s 0/6, five warm direct trials per arm gave medians of 206.956 ms ordinary and 159.788 ms grouped. Actual pressure on either rank preserved first tokens. Transient pools used and reserved zero bytes after synchronization. See `artifacts/tp2-resume/w2-group-audit/direct.RtENIMZH/`.

The direct replay matched 455,028,608 state bytes, 2,483,200 BF16 first-logit values, and 64 continuation IDs with proposal and acceptance histories. All seven ranked-cache HTTP phases matched, including 16 rank 0 draft objects. An earlier altered-FC trial changed 233,046 of 8,646,528 draft bytes while target GQA and GDN matched. See `artifacts/tp2-resume/w2-group-audit/` (`direct.RtENIMZH/candidate/state.log`, `http.mZP03dnI/comparison.json`, `state.6XeMtH/candidate.log`). Official W2 makespan remains unmeasured for this release.

### TP2 grouped RMSNorm replay

On B200s 0/6, five paired direct W4 tile trials saved a median 1.519 ms per 4,096-row tile. Each rank omitted 80 of 128 normalized BF16 stores. An independent replay matched 30 teacher score entries and five full tensor samples per rank, including forced FP8 projection fallback. W4 HTTP makespan remains unmeasured. See `artifacts/tp2-resume/w4-rms-quant/production/tile-0-6.log` and `artifacts/tp2-resume/w4-rms-vector-review/probe-final-0-6.log`.

### TP2 causal GQA operator replay

On one B200, the native page reader matched all 12,582,912 BF16 outputs per W5 rank shard and replay against the previous staged reader. Query and KV inputs stayed byte-identical. At 8,192 visible tokens, median time fell from 489 µs on each rank to 427 and 421 µs. At 32,768, it fell from 1,904 and 1,900 µs to 1,478 and 1,465 µs. Separate W4 captures at 32,768 tokens matched bitwise, with median times 20–22% lower across four GQA layers. W4 and W5 serving makespan remain unmeasured. See `artifacts/tp2-resume/native-long-gqa/` (`candidate-final-gpu0.log`, `candidate-w4-max-gpu0.log`, `decode-review/bench-gpu4.log`).

On one B200, a separate 4,096-row probe with 12 query heads and two KV heads timed cold preprocessing and gate plus quantization at 196 to 109 µs. It skipped attention and output projection. Eight complete planes matched bitwise at 0, 1,171, and 4,096 reused rows, including rotated Q/K, KV payloads and scales, and activation codes and scales. See `artifacts/tp2-resume/native-causal-fuse/probe-ordered.log`.

The integrated caller check used two logical ranks on one B200 with the collective replaced by a trace. All 20 baseline/candidate full-plane digests matched across the 12Q/2KV and 24Q/4KV layouts. The digests cover rotated Q/K, activation codes and scales, projected outputs, and both owners’ KV planes. Injected staged-reader and epilogue errors propagated. See `artifacts/tp2-resume/native-causal-integrated/hardware-gpu0.log` and its paired result logs. Serving makespan remains unmeasured.

### TP2 grouped gate/up replay

The [native 4,096-row gate/up route](fp8-prefill-projections.md#large-prefill-tiles) matched every BF16 output for checkpoint layers 0, 3, and 32 on both rank shards. With captured W4 activations on one B200, median operator time fell from 356.736 to 314.432 µs for rank 0 and from 356.288 to 314.240 µs for rank 1.

On two B200s, warm direct decoder priming of one 32,768-token W4 prompt used eight 4,096-row groups. Time fell from 1,087.120 to 1,058.156 ms, a 2.66% reduction. The harness recorded rank hidden and residual state in both arms. Those states, token IDs, and MTP acceptance matched. HTTP serving makespan remains unmeasured. See `artifacts/tp2-resume/native-prefill-gate/` (`gpu-0.jI3uJ3/rank-{0,1}.log`, `prime-4-5.dYtCZR/prime-{0,4}.log`).

### TP2 grouped attention output replay

The [native 3,072-input route](fp8-prefill-projections.md#large-prefill-tiles) matched every BF16 output for early and late W4 and W5 captures. Replays covered both TP2 rank shards and checkpoint layers 0, 3, and 32. Eager, CUDA graph, and clean fallback comparisons matched.

On two B200s, warm direct eager decoder priming for one 32,768-token W4 prompt fell from 999.533 to 981.236 ms. The ABBA and BAAB sequences had four observations per arm. Sampled rank hidden and residual state, token IDs, and MTP acceptance matched. For the 131,072-token W5 prompt, caller state and token records matched. The single observation per arm was unpaired. W4 HTTP makespan remains unmeasured. See `artifacts/tp2-resume/native-prefill-output/receipt.txt`.

### TP2 long down projection replay

The [CUTLASS down route](fp8-prefill-projections.md#large-prefill-tiles) replayed each rank shard separately on a single B200 using Qwen3.8-27B-FP8 checkpoint layers 0, 3, and 32 and fixed synthetic input activations. Eight alternating CUDA-event trials measured warm graph replay per layer from residual reset and RMSNorm through gate/up, SwiGLU, and down, including scale preparation. Eager compared all three layers, and graph replay compared layer 32. BF16 down outputs, SwiGLU E4M3 values, and FP32 scales matched bitwise.

| Rows | Rank shard | Native median (µs) | CUTLASS median (µs) |
| ---: | ---: | ---: | ---: |
| 4,096 | 0 | 730.556 | 691.593 |
| 4,096 | 1 | 730.008 | 686.147 |
| 1,024 | 1 | 202.757 | 192.767 |

The forced fallback linked the pre-change archive and required the scaled launcher to decline. At 4,096 rows, its medians were within 2.765 µs of native on both shards, with bitwise parity. The shape guard also declined 1,004 rows on rank 1, with native and fallback medians of 204.579 and 205.278 µs and bitwise parity. See `artifacts/tp2-resume/long-gemm/receipts/` (`summary.json`, `ab-*-down-only.log`, `ab-*-down-control.log`). W4 and W5 serving makespan remain unmeasured for this route.
