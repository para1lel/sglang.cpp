# Batch CUDA graphs

## Responsibilities and sources of truth

| Part | Responsibility | Source |
| --- | --- | --- |
| Server | Resolve graph policy and pass the single-device executor to the batch service. | [Effective policy](../apps/sglang-server.cpp#L58)<!-- prefix: "bool enable_" --> · [Startup composition](../apps/sglang-server.cpp#L355)<!-- prefix: "auto executo" --> · [Trace](../apps/server/detail/batch_trace.cpp#L65)<!-- prefix: "void trace_b" --> |
| Decoder batch | Select eligible decode steps, bind lane state, and return winners by slot. | [Batch options](../include/sglang/decoder_batch.h#L64)<!-- prefix: "struct Decod" --> · [Dispatch](../src/decoder_batch.cpp#L391)<!-- prefix: "run = run_de" --> · [Readback](../src/decoder_batch.cpp#L478)<!-- prefix: "void Decoder" --> |
| CUDA resources | Recover from eligible pre-submission failures and retain resources while completion is uncertain. | [Fallback](../src/runtime/backend/cuda_resources.cpp#L597)<!-- prefix: "bool Allocat" --> · [Cleanup](../src/runtime/backend/cuda_resources.cpp#L278)<!-- prefix: "CleanupError" --> |

## Runtime path

C++ defaults to eight lanes and 32,768 rows per step. Reserve rows for ready decode lanes in admission order. Within the step budget, reserve up to one plus the legal [MTP proposals](mtp.md#runtime-path) per lane. Give the oldest ready prompt the remaining rows up to its end or the next [prefix-cache checkpoint](prefix-cache.md#checkpoint-boundaries). A later prompt can use any remainder. Plain C++ `advance` limits each prompt to one row. The C++ zero-cap mode plans with 128 rows and reserves one per ready decode lane. Execution can still verify the lane's legal proposals. [Batch options](../include/sglang/decoder_batch.h#L64)<!-- prefix: "struct Decod" --> · [Budget planner](../src/text/decoder_batch_plan.cpp#L9)<!-- prefix: "StepPlan pla" --> · [Legal rows](../src/runtime/execution/decoder_batch_state.cpp#L28)<!-- prefix: "StepPlan pla" --> · [Zero-cap execution](../src/decoder_batch.cpp#L322)<!-- prefix: "const int de" -->

HTTP arrival order starts when the server publishes work to the model-ready queue after validation and tokenization. Direct C++ callers establish order on successful admission. Restores and later chunks keep the original order. A new request in a reused slot joins the tail. Decode priority controls row grants. The executor can enqueue prefill before MTP verification in a mixed step. [Scalar publication](../apps/server/batch_service.cpp#L915)<!-- prefix: "state_->admi" --> · [Batch publication](../apps/server/batch_service.cpp#L547)<!-- prefix: "admission.ad" --> · [Direct admission](../src/decoder_batch.cpp#L217)<!-- prefix: "state_->stor" --> · [Submission](../src/decoder_batch.cpp#L289)<!-- prefix: "void Decoder" -->

`--trace-batches` reports `target_rows` and `step_budget`. Charge executed prompt rows (including the first-output row), one row per ordinary decode lane, and computed MTP target verification rows. Prefix restores, score-only hits, draft seeding and proposals, commit replays, vision encoding, and emitted IDs are outside the cap. `prefill_rows` lists configured slots in order, with at least four entries; `width` includes selected lanes with zero rows, and `decode` counts executed decode lanes. Active K0 steps report `peak_request_bytes`. [Step trace](../apps/server/detail/batch_trace.cpp#L39)<!-- prefix: "std::cerr <<" -->

`sglang-server` defaults to TP1 batch replay and [TP2 verifier replay](#tp2-target-verification-graph). `--batch-cuda-graphs` still enables TP1 replay. `--eager` disables both independently of prefix reuse. C++ batch options default both to `false`. [CLI flags](../apps/sglang-server.cpp#L132)<!-- prefix: "if (option =" --> · [Batch options](../include/sglang/decoder_batch.h#L64)<!-- prefix: "struct Decod" -->

TP1 replays direct decode lanes. Separate prefill and full-hit scoring can share the step. Media binds independent rotary positions. [TP1 MTP verification](mtp.md#target-verification-graph) has its own executable. [Step selection](../src/decoder_batch.cpp#L383)<!-- prefix: "const bool r" --> · [Graph binding](../src/text/decoder_graph.cpp#L67)<!-- prefix: "const int ro" -->

TP1 runs each eligible shape eagerly before thread-local capture. Replay binds live tokens, positions, and request state at fixed addresses with device page tables. It runs embedding, decoder layers, and scoring. Greedy, sampled, and [constrained selection](text-serving.md#finite-choice-constraints) share row logits.

B200 K0 graph replay covers eligible text singletons. Other devices can replay eligible widths 1–4. Wider steps use eager forward. See [grouped split target attention](singleton-gqa-decode.md#split-kv-target-attention) and [attention and score limits](singleton-gqa-decode.md#constraints-and-failure-behavior). [Selection](../src/text/decoder_graph.cpp#L98)<!-- prefix: "const bool g" -->

In `--trace-batches`, `graph_slots` sets one bit per replayed K0 logical slot, excluding padding. `request_bytes` includes legacy lane scores. Batch scratch, graphs, and prefix storage count separately. Active steps report prefill `workspace_bytes` and allocated `sampling_workspace_bytes`. [Slot mask](../src/decoder_batch.cpp#L414)<!-- prefix: "stats.graph_" --> · [Request accounting](../src/runtime/execution/decoder_batch_state.cpp#L15)<!-- prefix: "std::size_t " --> · [Sampling trace](../apps/server/detail/batch_trace.cpp#L49)<!-- prefix: "if (stats.sa" -->

After the batch allocates constraint scratch, active steps report `constraint_workspace_bytes`. The batch stream shares 24,576 bytes for IDs and scores. A wide constraint adds a vocabulary mask and masked scores, for 552,256 bytes total. Each selection uploads up to 16,384 bytes of IDs or a 31,040-byte mask. The lane keeps its host buffers live through the step. Constrained speculation keeps [separate host inputs for each row](mtp.md#constraints-and-failure-behavior). Sampling uses separate scratch. [Constraint scratch](../src/decoder_constraint.cpp#L21)<!-- prefix: "std::size_t " --> · [Constraint trace](../apps/server/detail/batch_trace.cpp#L53)<!-- prefix: "if (stats.co" -->

### TP2 target verification graph

B200 TP2 replay requires two distinct complete greedy K4 text lanes, ten unpacked rows at contiguous per-lane positions within 0–255, valid pages, vector GQA and rank-local GDN on both ranks, and healthy MNNVL. Eager lanes, sampling, constraints, requested scores, and other shapes stay eager. [Eligibility](../src/runtime/execution/tp_verify_graph.cpp#L117)<!-- prefix: "gdn_replay_s" --> · [Batch gate](../src/decoder_batch.cpp#L682)<!-- prefix: "const bool c" -->

The TP runtime owns one executable for its verifier stream pair. Each replay refreshes row, rotary, GQA and ReplaySSM bindings from live requests, including after restore or rebinding. Both streams complete before releasing host bindings or destroying the context. The graph captures only target forward. Recoverable capture setup refusal disables this context and falls back to eager. Submission and completion errors make the TP transport terminal. [Binding upload](../src/runtime/execution/tp_verify_graph.cpp#L247)<!-- prefix: "return true;" --> · [Teardown](../src/decoder_batch.cpp#L39)<!-- prefix: "DecoderBatch" -->

### Eager TP2 singleton verification

On B200 TP2, one unbound greedy MTP lane with three to five target rows and Tensor Core vocabulary scratch can write packed FP8 directly if both ranks select Tensor Core projections and neither binds an external packed buffer. Sampling, constraints, and requested logits or logprobs bypass this singleton route. Healthy eager MNNVL can also fuse reduction with RMSNorm and quantization. A transport decline keeps ordinary allreduce and local fused preparation. The route preserves BF16 rounding, E4M3 values, and FP32 scales. [Caller](../src/runtime/execution/tp_verify_graph.cpp#L409)<!-- prefix: "const auto f" --> · [Selection](../src/runtime/execution/decoder_forward_tp.cpp#L224)<!-- prefix: "const bool n" --> · [Partial RMSNorm](../src/text_normalization.cu#L464)<!-- prefix: "cudaError_t " -->

Require 127 fused sites per rank for eligible singleton widths three through five and the two-lane ten-row control. Require zero for two-row, sampled, and scored singleton controls. Compare IDs, both ranks’ KV, GDN, hidden and residual state, and winner scores. [Frozen two-B200 and W1 HTTP replays](https://github.com/para1lel/sglang.cpp/blob/a5974ba4eb99bd8e8f890c2e7262a27dbc3aace1/docs/batch-graphs.md#eager-tp2-singleton-verification)

### Eight-row TP2 draft repair

For a greedy eight-lane TP2 group, rank 0 collects draft KV repairs across both four-lane commits when no output callback is active and packed scratch covers all lanes. Media, sampling, and constraints retain packet repair. Each depth needs all eight repair rows and scalar rotary positions below `text_attention_max_tokens`. Partial depths, MRoPE, and out-of-range positions use four-lane packets. Batched BF16 FC and FP8 K/V keep the packet arithmetic and MXFP8 cache layout. [Group guard](../src/decoder_batch.cpp#L721)<!-- prefix: "const bool g" --> · [Depth guard](../src/decoder_mtp.cpp#L670)<!-- prefix: "const bool g" --> · [BF16 FC](../src/text_ops.cu#L112)<!-- prefix: "__global__ v" --> · [FP8 K/V](../src/text/fp8_projection.cu#L623)<!-- prefix: "const bool e" -->

Match hidden and KV repair records, proposals, and output IDs through full acceptance and a forced rejection. Require 28 captured nodes for eight and sixteen eligible lanes. Check ordinary MRoPE dispatch and zero live request bytes after slot reuse. [Frozen W3 replay and capture evidence](https://github.com/para1lel/sglang.cpp/blob/a5974ba4eb99bd8e8f890c2e7262a27dbc3aace1/docs/batch-graphs.md#eight-row-tp2-draft-repair)

## Resident memory admission

The HTTP batch service derives a fixed request limit from GPU memory free after initialization, reserving up to 2 GiB for shared allocations. Admission checks active reservations and current device memory, including [reclaimable retired blocks](serving/request-memory.md). The [row and lane caps](development.md#native-streaming-server) apply separately. C++ callers can opt in with `DecoderBatchOptions::resident_memory_limit` in bytes. The effective limit is the smaller of that value and the startup limit. [HTTP setup](../src/runtime/server/single_device_batch_executor.cpp#L87)<!-- prefix: "config.max_n" --> · [C++ option](../include/sglang/decoder_batch.h#L69)<!-- prefix: "std::optiona" --> · [Startup limit](../src/runtime/execution/decoder_batch_state.cpp#L87)<!-- prefix: "if (options." --> · [Admission check](../src/decoder_batch.cpp#L67)<!-- prefix: "std::size_t " -->

Each positive-budget request reserves page-rounded KV for the encoded or expanded prompt length plus the output budget minus one. The charge includes target KV, GDN state, workspace allowances, and any MTP draft KV, verifier, or media position storage. Admission initializes live storage and prepares shared sampling and constraint buffers when needed. The [daemon](prefix-cache.md#lmcache-daemon) owns cross-request payloads in CPU memory. A zero-budget request without scoring skips the reservation. [Charge calculation](../src/runtime/model/qwen_request.cpp#L99)<!-- prefix: "std::size_t " --> · [Initialization](../src/decoder_batch.cpp#L106)<!-- prefix: "if (lane.res" --> · [Zero budget](../src/decoder_batch.cpp#L185)<!-- prefix: "if (options." -->

Admission can reduce an [implicit chat budget](chat-serving.md#sampling-defaults-and-output-budget) to the fixed limit. Keep explicit budgets. Exceeding the limit returns HTTP 413; temporary shortage or allocation OOM with successful cleanup returns 503. These failures leave peers running and occur before SSE headers. C++ reports `DecoderCapacityError` with `retryable`. Terminal CUDA status or failed drain disables the batch and fails current jobs. See [readiness](http-server.md#health-and-readiness) and [stream errors](http-server.md#completion-streams). [Limit checks](../src/runtime/model/qwen_request.cpp#L112)<!-- prefix: "int limit_re" --> · [Admission recovery](../src/decoder_batch.cpp#L219)<!-- prefix: "} catch (con" --> · [Drain](../src/runtime/execution/decoder_batch_state.cpp#L110)<!-- prefix: "void Decoder" --> · [C++ error](../include/sglang/decoder.h#L110)<!-- prefix: "class Decode" --> · [SSE error](../apps/server/completions.cpp#L160)<!-- prefix: "if (stream.w" -->

## Constraints and failure behavior

Each batch caches at most one executable per eligible K0 width. Sparse
selections can rebind after retirement. Prefill runs outside these executables.
[Capture and binding](../src/text/decoder_graph.cpp#L40)<!-- prefix: "return execu" --> · [Zero-budget completion](../src/decoder_batch.cpp#L784)<!-- prefix: "result.finis" -->

[Former packed K0 setup and failure rules](https://github.com/para1lel/sglang.cpp/blob/6debe50edccdeda34ff307235ed38e74b6b8f2ee/docs/batch-graphs.md#constraints-and-failure-behavior)

Batch execution owns the winner pin and allocates it on first eligible use. Clean allocation OOM keeps per-lane reads for that batch. Other errors reach batch recovery. Cleanup frees the pin after stream drain and graph teardown. A rejected free retains ownership for device-local deferred reclamation. Terminal errors retain the resources. [Allocation](https://github.com/para1lel/sglang.cpp/blob/93648269d852e0fee78f7af4d1c3ffe4897aa361/src/decoder_resources.cpp#L556)<!-- prefix: "TextLogitCan" --> · [Release](../src/runtime/backend/cuda_resources.cpp#L312)<!-- prefix: "CleanupError" --> · [Retry](../src/runtime/backend/cuda_resources.cpp#L161)<!-- prefix: "static void " -->

Legacy long-score OOM forces eager for that step and later steps needing the failed request's scores. Other requests retain allocations, and recycled slots can replay. Width-local graph scratch OOM or eligible capture, instantiation, or synchronous launch rejection can disable one width. Before eager fallback, end capture, drain the stream, and destroy graph handles. Other widths remain eligible. A target-only execution error outside graph fallback or [prefix recovery](prefix-cache.md#constraints-and-failure-behavior) disables the batch. Invalid selections retire only their lane. Cleanup retains resources while completion is uncertain. [Step recovery](../src/decoder_batch.cpp#L832)<!-- prefix: "} catch (con" -->

### Score-backed singleton GQA decode

See [singleton dispatch and score storage](singleton-gqa-decode.md).

### Page-parallel GQA decode

See [generic SM100 singleton dispatch](singleton-gqa-decode.md#generic-sm100-singleton) for the one direct K0 lane, 1,536–2,560 visible tokens including current, and workspace guard. A width-one graph uses live positions to select inside its attention nodes. The standalone serial route uses [scratch-free attention](singleton-gqa-decode.md#scratch-free-ordered-attention).

Each 128-token page produces an FP32 maximum, mass and value numerator per head. Merge rescales against the global maximum, divides and rounds to BF16. Reduction can change logits or IDs. Page scratch needs 495,360 of the lane's 496,640 logits bytes. Outside the window or without scratch, use ordered replay.

Nonfinite scores invalidate the mass. The merge requires positive finite mass and finite channel numerators. Otherwise the affected head and tile recompute QK and replay in order, within the same graph node. Ordered replay can still produce nonfinite output. [Eager selection](../src/text/decoder_graph.cpp#L94)<!-- prefix: "first.narrow" --> · [Bound attention](../src/runtime/execution/decoder_forward_attention.cpp#L203)<!-- prefix: "if (batch.si" --> · [Position window](../src/mxfp8_gqa.cu#L377)<!-- prefix: "__host__ __d" --> · [Page reduction](../src/mxfp8_gqa.cu#L381)<!-- prefix: "__device__ v" --> · [Workspace](../src/runtime/model/qwen_request.cpp#L226)<!-- prefix: "return {rein" --> · [Merge guard](../src/mxfp8_gqa.cu#L499)<!-- prefix: "if (__syncth" --> · [Replay selection](../src/mxfp8_gqa.cu#L553)<!-- prefix: "__global__ v" -->

### Packed paged GQA decode

[Former packed K0 dispatch and bindings](https://github.com/para1lel/sglang.cpp/blob/6debe50edccdeda34ff307235ed38e74b6b8f2ee/docs/batch-graphs.md#packed-paged-gqa-decode)

### Legacy narrow paged GQA decode

Legacy width-two through width-four graphs select split attention per lane at 8,192–16,384 visible tokens. The captured sequence appends KV, attends, and gates, with ordered score and replay outside the window. Eager groups eligible decode lanes after append. Width four shares decoded KV across adjacent heads in a 16×12×4 grid with 256 threads per block. Widths one to three launch one block per query head and partition. Each split uses 16 partitions of 128-token pages and 399,360 bytes of the lane's 496,640-byte logits arena. The merge rescales FP32 numerators and masses then rounds to BF16. [Bounds](../src/text/gqa/device_geometry.cu#L22)<!-- prefix: "SGLANG_GQA_H" --> · [Bound forward](../src/runtime/execution/decoder_forward_attention.cpp#L184)<!-- prefix: "void attend_" --> · [Eager forward](../src/runtime/execution/decoder_forward_attention.cpp#L271)<!-- prefix: "void attend_" --> · [Paired split](../src/text/gqa/paged_decode.cu#L449)<!-- prefix: "__global__ v" --> · [Scratch alias](../src/text/gqa/verify_batch.cpp#L8)<!-- prefix: "PagedDecodeB" -->

Nonfinite scores, visible V scale above 160, or invalid partials or merge trigger ordered FP64 replay per affected head. The merge requires finite numerators and outputs and a positive finite weight sum. Capture includes head and lane fallback. Reduction order can change logits. [Head replay](../src/text/gqa/paged_decode.cu#L199)<!-- prefix: "__device__ v" --> · [Replay dispatch](../src/text/gqa/paged_decode.cu#L283)<!-- prefix: "__global__ v" -->

## Verification

Build the release server and both probes with `--mxfp8-gqa-check=y` and `--batch-graph-lanes=y` using the [runbook](../tests/batch/README.md). Check bound operators, reuse, recovery, wire parity, and CPU rejection of masks containing a zero-budget slot.

On a reserved B200, compare greedy IDs and full BF16 rows at eager widths 2–32 and singleton replay. Check rebinding, a zero-budget peer, cancellation, first-use score decline, stable score storage and zero retired request bytes. Match eager after launch rejection and across 5,120/5,121 visible tokens. Use [packed-route drift](long-paged-decode.md#packed-route-drift-evidence) and [singleton gates](singleton-gqa-decode.md#verification).

Run [K0 streaming fixtures](development.md#streaming-verification) with default selection and `--eager`. Match IDs, finishes and usage. Trace one executable per eligible width, masks matching real launches, and zero graph launches on eager steps. Run [wide K0](../tests/batch/README.md#wide-k0-serving) at 16 and 32 lanes, including narrower replay after reuse and slot 31. For positive depth use the [MTP graph checks](mtp.md#verification-graph-checks).

The [grouped head checks](../tests/decoder/README.md#grouped-decode-lm-head) require BF16, greedy and seeded parity during capture, replay, mixed steps, and reuse. Require one winner copy per eligible group, or lane copies for pin OOM and diagnostic, sampled, or constrained rows. Cover [rejected pin release](lessons.md#l010--rejected-packed-winner-pin-release).

Use [scheduler admission](../tests/batch/README.md#scheduler-admission) for caps, queueing, reuse, row accounting, and the [admission failure contract](#resident-memory-admission). Verify peer IDs, subsequent completion and readiness after recoverable OOM. On terminal fault, require 500 for new completions, 503 from readiness, and unchanged health limits. Pre-submission rejection and capture failure use eager. Post-submission failure propagates and retains resources until safe. Check reuse under Compute Sanitizer.

Use an FP64 oracle with packed KV for both edges and adjacent lengths of page-parallel and legacy narrow dispatch. Check graph/eager parity, remaps, rebound positions, ordered fallback, masks, scratch guards, and unaffected heads through score overflow and cancellation. [GQA verifier](development.md#p2-gqa-verification) · [Full graph probe matrix](https://github.com/para1lel/sglang.cpp/blob/5ee79416d3e30cd178abab69598d047c4393793d/docs/batch-graphs.md#verification) [Test target](../xmake.lua#L428)<!-- prefix: "target(\"batc" -->

## Timing

Compare default graph selection with `--eager` on one binary, with fixed prefill and prefix reuse off. Freeze requests, arrivals, device, and warmups. Measure request latency, replay, and cold capture separately. Verify widths in untimed traces.

Before timing, repeat each native route for stable output IDs. If native builds diverge, compare full BF16 logit rows at the first differing step on their common history. Record the winner margin. Use the [strict concurrent policy](performance.md#native-concurrent-workloads) for HTTP serving.
