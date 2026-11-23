# Batch CUDA graphs

## Responsibilities and sources of truth

| Part | Responsibility | Source |
| --- | --- | --- |
| Server | Set graph policy, construct the batch service, and report batch steps. | [Effective policy](../apps/sglang-server.cpp#L44)<!-- prefix: "bool enable_" --> · [Batch service](../apps/sglang-server.cpp#L294)<!-- prefix: "BatchService" --> · [Trace](../include/sglang/server/batch_trace.h#L68)<!-- prefix: "inline void " --> |
| Decoder batch | Select eligible decode steps, bind lane state, and return winners by slot. | [Batch options](../include/sglang/decoder_batch.h#L54)<!-- prefix: "struct Decod" --> · [Dispatch](../src/decoder_batch.cpp#L592)<!-- prefix: "const auto r" --> · [Readback](../src/decoder_batch.cpp#L670)<!-- prefix: "void Decoder" --> |
| CUDA resources | Recover from eligible pre-submission failures and retain resources while completion is uncertain. | [Fallback](../src/decoder_resources.cpp#L595)<!-- prefix: "bool Allocat" --> · [Cleanup](../src/decoder_resources.cpp#L225)<!-- prefix: "CleanupError" --> |

## Runtime path

The [step cap](development.md#native-streaming-server) covers prefill and decode. C++ defaults to four lanes and an 8,192-token cap. A zero cap selects legacy planning. Fair share rotates lane priority and gives MTP verification priority over extra prefill. Plain `advance` allows one prompt row per lane. Physical tiles can hold up to 1,024 text rows. Media and MTP with prefix reuse cap at 128 rows. See [MTP prefill](mtp.md#runtime-path) and [prefill workspace](native-prefill-workspace.md). [Batch options](../include/sglang/decoder_batch.h#L54)<!-- prefix: "struct Decod" --> · [Planner entry](../src/decoder_batch.cpp#L189)<!-- prefix: "StepPlan pla" -->

`--trace-batches` reports `target_rows` and `step_budget`. The total charges executed prompt rows, including the first-output row, one row per ordinary decode lane, and computed target verification rows for MTP. Prefix restores, score-only full hits, draft seeding and proposals, commit replays, vision encoding, and emitted IDs stay outside the total. `prefill_rows` lists configured slots in order, with at least four entries. `width` counts selected lanes, including lanes assigned no rows. `decode` counts executed decode lanes. K0 reports `peak_request_bytes` on active steps. [Step trace](../include/sglang/server/batch_trace.h#L42)<!-- prefix: "std::cerr <<" -->

`sglang-server` enables CUDA graphs by default. It still accepts `--batch-cuda-graphs`. `--eager` disables replay and requires prefix reuse off. The C++ option `DecoderBatchOptions::cuda_graphs` defaults to `false`. [CLI flags](../apps/sglang-server.cpp#L93)<!-- prefix: "if (option =" --> · [Batch options](../include/sglang/decoder_batch.h#L54)<!-- prefix: "struct Decod" -->

K0 replay requires text decode without prefill rows or MTP work. Eligible [MTP groups](mtp.md#target-verification-graph) use a separate executable. [Step selection](../src/decoder_batch.cpp#L591)<!-- prefix: "const bool r" -->

Each eligible shape runs eagerly before thread-local capture on the batch stream. Replay binds the live tokens, positions, and request state at fixed addresses. Page tables reside on the device. Replay runs embedding, decoder layers, and scoring. Greedy, sampled, and [constrained selection](text-serving.md#finite-choice-constraints) share per-row logits.

B200 K0 graph replay covers eligible singleton text decode. Other devices can
replay eligible widths one through four. Wider steps use eager forward. See
[grouped split target attention](singleton-gqa-decode.md#split-kv-target-attention).
The earlier packed K0 runtime used padded buckets through width 32.
[Selection](../src/text/decoder_graph.cpp#L92)<!-- prefix: "const bool g" -->

Width one has [attention and score limits](singleton-gqa-decode.md#constraints-and-failure-behavior).

The earlier packed K0 graph read eligible greedy winners in one asynchronous copy. Sampled, constrained, and diagnostic rows kept per-lane readback. [Historical readback](https://github.com/para1lel/sglang.cpp/blob/93648269d852e0fee78f7af4d1c3ffe4897aa361/src/text/decoder_graph_packed.cpp#L531)<!-- prefix: "const bool r" --> · [Copy](https://github.com/para1lel/sglang.cpp/blob/93648269d852e0fee78f7af4d1c3ffe4897aa361/src/text/decoder_graph_packed.cpp#L388)<!-- prefix: "check(cudaMe" -->

In `--trace-batches`, `graph_slots` sets one bit per replayed K0 logical slot, excluding padding. `request_bytes` includes legacy lane scores. Batch scratch, graphs, and prefix storage count separately. Active steps report prefill `workspace_bytes` and allocated `sampling_workspace_bytes`. [Slot mask](../src/decoder_batch.cpp#L606)<!-- prefix: "stats.graph_" --> · [Request accounting](../src/decoder_batch.cpp#L120)<!-- prefix: "std::size_t " --> · [Sampling trace](../include/sglang/server/batch_trace.h#L52)<!-- prefix: "if (stats.sa" -->

After the batch allocates constraint scratch, active steps report `constraint_workspace_bytes`. The batch stream shares 24,576 bytes for IDs and scores. A wide constraint adds a vocabulary mask and masked scores, for 552,256 bytes total. Each selection uploads up to 16,384 bytes of IDs or a 31,040-byte mask. The lane keeps its host buffers live through the step. Constrained speculation keeps [separate host inputs for each row](mtp.md#constraints-and-failure-behavior). Sampling uses separate scratch. [Constraint scratch](../src/decoder_constraint.cpp#L21)<!-- prefix: "std::size_t " --> · [Constraint trace](../include/sglang/server/batch_trace.h#L56)<!-- prefix: "if (stats.co" -->

## Resident memory admission

The HTTP batch service derives a fixed request limit from GPU memory free after initialization, reserving up to 2 GiB for shared allocations. Admission checks active reservations and current device memory, including [reclaimable retired blocks](serving/request-memory.md). The [row and lane caps](development.md#native-streaming-server) apply separately. C++ callers can opt in with `DecoderBatchOptions::resident_memory_limit` in bytes. The effective limit is the smaller of that value and the startup limit. [HTTP setup](../apps/server/batch_service.cpp#L143)<!-- prefix: "executor = s" --> · [C++ option](../include/sglang/decoder_batch.h#L59)<!-- prefix: "std::optiona" --> · [Startup limit](../src/decoder_batch.cpp#L226)<!-- prefix: "if (options." --> · [Admission check](../src/decoder_batch.cpp#L321)<!-- prefix: "std::size_t " -->

Each positive-budget request reserves page-rounded KV for the encoded or expanded prompt length plus the output budget minus one. The charge includes target KV, GDN state, workspace allowances, and any MTP draft KV, verifier, or media position storage. The resident path initializes request storage at admission and prepares shared sampling and constraint buffers when needed. A zero-budget request skips the reservation. [Charge calculation](../src/decoder_resources.cpp#L346)<!-- prefix: "std::size_t " --> · [Initialization](../src/decoder_batch.cpp#L357)<!-- prefix: "if (lane.res" --> · [Zero budget](../src/decoder_batch.cpp#L772)<!-- prefix: "if (const in" -->

The batch can reduce a [chat request's implicit output budget](chat-serving.md#sampling-defaults-and-output-budget) to fit the fixed limit. It keeps explicit budgets. HTTP returns 413 above the fixed limit or 503 when current capacity is too low. A CUDA allocation failure during initialization returns 503 if cleanup succeeds. The service retires the failed lane and keeps active peers. These failures return JSON before [SSE opens on admission](http-server.md#completion-streams). C++ callers receive `DecoderCapacityError` with `retryable`. A terminal CUDA status or failed drain disables the batch. The service fails current jobs and reports [worker failure](http-server.md). An open writable stream sends an SSE error and ends without `[DONE]`. [Limit checks](../src/decoder_resources.cpp#L358)<!-- prefix: "int limit_re" --> · [Admission recovery](../src/decoder_batch.cpp#L438)<!-- prefix: "} catch (con" --> · [Drain](../src/decoder_batch.cpp#L243)<!-- prefix: "void drain(c" --> · [C++ error](../include/sglang/decoder.h#L90)<!-- prefix: "class Decode" --> · [SSE error](../apps/server/completions.cpp#L160)<!-- prefix: "if (stream.w" -->

## Constraints and failure behavior

Each batch caches at most one executable per eligible K0 width. Sparse
selections can rebind after retirement. Prefill and media run eagerly.
Full prefix hits score eagerly while eligible peers replay.
[Capture and binding](../src/text/decoder_graph.cpp#L40)<!-- prefix: "return execu" --> · [Full hit](../src/decoder_batch.cpp#L552)<!-- prefix: "if (lane.res" --> · [Zero-budget completion](../src/decoder_batch.cpp#L800)<!-- prefix: "result.finis" -->

The historical ordered graph used slot logits at positions 0–5,119 and could allocate request-owned scores at widths 2–4. Each 128-position page needed 12,288 bytes for 24 FP32 scores per position, up to 24 MiB per lane. [Historical score binding](https://github.com/para1lel/sglang.cpp/blob/93648269d852e0fee78f7af4d1c3ffe4897aa361/src/text/decoder_graph_packed.cpp#L466)<!-- prefix: "scores = req" --> · [Score allocation](../src/decoder_resources.cpp#L839)<!-- prefix: "ExactGqaWork" -->

The historical packed implementation fell back after first-use scratch OOM only when the CUDA error check and partial cleanup succeeded. A terminal setup error disabled the batch before decode publication. See [the fault regression](lessons.md#l009--terminal-error-during-first-packed-decode-setup). [Historical scratch](https://github.com/para1lel/sglang.cpp/blob/93648269d852e0fee78f7af4d1c3ffe4897aa361/src/text/decoder_graph_packed.cpp#L142)<!-- prefix: "Element* Pac" --> · [Historical fallback](https://github.com/para1lel/sglang.cpp/blob/93648269d852e0fee78f7af4d1c3ffe4897aa361/src/text/decoder_graph_packed.cpp#L509)<!-- prefix: "PackedDecode" -->

Batch execution owns the winner pin and allocates it on first eligible use. Clean allocation OOM keeps per-lane reads for that batch. Other errors reach batch recovery. Cleanup frees the pin after stream drain and graph teardown. A rejected free retains ownership for device-local deferred reclamation. Terminal errors retain the resources. [Allocation](https://github.com/para1lel/sglang.cpp/blob/93648269d852e0fee78f7af4d1c3ffe4897aa361/src/decoder_resources.cpp#L556)<!-- prefix: "TextLogitCan" --> · [Release](../src/decoder_resources.cpp#L259)<!-- prefix: "CleanupError" --> · [Retry](../src/decoder_resources.cpp#L108)<!-- prefix: "static void " -->

Legacy long-score OOM forces eager for that step and later steps needing the failed request's scores. Other requests retain allocations, and recycled slots can replay. Width-local graph scratch OOM or eligible capture, instantiation, or synchronous launch rejection can disable one width. Before eager fallback, end capture, drain the stream, and destroy graph handles. Other widths remain eligible. A target-only execution error outside graph fallback or [prefix recovery](prefix-cache.md#constraints-and-failure-behavior) disables the batch. Invalid selections retire only their lane. Cleanup retains resources while completion is uncertain. [Step recovery](../src/decoder_batch.cpp#L846)<!-- prefix: "} catch (con" -->

### Score-backed singleton GQA decode

See [singleton dispatch and score storage](singleton-gqa-decode.md).

### Page-parallel GQA decode

Singleton K0 decode can use page-parallel attention with 1,536–2,560 visible tokens, including the current token. Both prefill policies share decode. Serial eager and serial graph calls can use this path when they select exact attention. Serial graph exact attention requires page-rounded KV capacity of at most 5,120 tokens. [Row dispatch](../src/decoder_forward.cpp#L413)<!-- prefix: "cudaError_t " --> · [Serial graph attention](../src/mxfp8_gqa.cu#L910)<!-- prefix: "cudaError_t " -->

Eager can use this path with one ordinary decode row and zero prefill rows, including independent full-prefix-hit scoring. The width-one graph captures two attention nodes. The live position selects the reduction inside those nodes. A larger KV reservation still permits short replay. [Eager dispatch](https://github.com/para1lel/sglang.cpp/blob/93648269d852e0fee78f7af4d1c3ffe4897aa361/src/text/decoder_graph_packed.cpp#L486)<!-- prefix: "first.narrow" --> · [Bound attention](../src/decoder_forward.cpp#L487)<!-- prefix: "if (batch.si" -->

Each logical 128-token KV page produces an FP32 maximum, denominator, and value numerator per query head. The merge rescales both accumulators against the global maximum, divides, and rounds to BF16. This reduction order can change logits and greedy IDs. The path needs 495,360 bytes of the existing 496,640-byte logits allocation. Outside the window or with insufficient workspace, the exact path uses ordered replay. [Position window](../src/mxfp8_gqa.cu#L206)<!-- prefix: "__host__ __d" --> · [Page reduction](../src/mxfp8_gqa.cu#L243)<!-- prefix: "__device__ v" --> · [Workspace](../src/decoder_resources.cpp#L829)<!-- prefix: "return {rein" -->

A nonfinite visible score makes the merged denominator nonfinite. The merge checks the denominator and each channel numerator before writing output. The denominator must be finite and positive. Every numerator in the tile must be finite. If a check fails, the tile recomputes QK scores for that head and replays the tokens in order. The fallback stays in the same graph node. It follows the ordered operator's numerical rules, which can also produce nonfinite output. [Merge guard](../src/mxfp8_gqa.cu#L356)<!-- prefix: "if (__syncth" --> · [Replay selection](../src/mxfp8_gqa.cu#L516)<!-- prefix: "__global__ v" -->

### Packed paged GQA decode

Packed buckets bind live positions, KV, and page tables. Negative positions mark padding. The [fused GQA step](long-paged-decode.md#fused-packed-gqa) appends KV, attends, and gates. Split partials use batch-owned logits scratch, avoiding request-owned long scores. Warmup, replay, and eligible pre-submission fallback share one enqueue path. [Packed attention](https://github.com/para1lel/sglang.cpp/blob/93648269d852e0fee78f7af4d1c3ffe4897aa361/src/text/decoder_graph_packed.cpp#L311)<!-- prefix: "void PackedD" --> · [Packed dispatch](https://github.com/para1lel/sglang.cpp/blob/93648269d852e0fee78f7af4d1c3ffe4897aa361/src/text/decoder_graph_packed.cpp#L521)<!-- prefix: "bool run_pac" -->

### Legacy narrow paged GQA decode

The legacy width-two through width-four graph selects split attention per lane at 8,192 through 16,384 visible tokens. It appends the new KV, runs attention, and gates the outputs on the batch stream. Ordered score and replay nodes handle lanes outside the window. Legacy eager forwards group in-window decode lanes after KV append. Other eager rows at those widths follow the row dispatch. [Bounds](../include/sglang/decoder_cuda.cuh#L236)<!-- prefix: "inline const" --> · [Bound forward](../src/decoder_forward.cpp#L469)<!-- prefix: "void attend_" --> · [Eager forward](../src/decoder_forward.cpp#L550)<!-- prefix: "void attend_" --> · [Bound fallback](../src/mxfp8_gqa.cu#L781)<!-- prefix: "__global__ v" -->

The split assigns 128-token KV pages to 16 partitions. It clamps the last to visible length. Bound width four and eager width four after compaction use a `(16, 12, 4)` grid with 256 threads per block. Blocks share decoded K and V between two adjacent query heads of one KV head. Split widths one through three use one block per query head and partition. [Split launch](../include/sglang/text/mxfp8_gqa_decode.cuh#L214)<!-- prefix: "void launch_" --> · [Paired split](../include/sglang/text/mxfp8_gqa_decode.cuh#L151)<!-- prefix: "__global__ v" -->

Four eight-thread subgroups per warp score four channel groups per pass. Two passes preserve each head's FP32 score tree and ascending group sum. Eight warps accumulate values in the original FMA order. Each head writes separate FP32 partials. The merge rescales the weight sum and value numerator to the global maximum, divides in FP32, and rounds to BF16. Split and ordered attention can produce different logits. Each lane uses 399,360 of its 496,640 logits bytes for 16-byte-aligned partials. In-window lanes skip request-owned score binding. [Paired scores](../include/sglang/text/mxfp8_gqa_decode.cuh#L48)<!-- prefix: "__device__ v" --> · [Merge](../src/mxfp8_gqa.cu#L668)<!-- prefix: "__global__ v" --> · [Scratch alias](../include/sglang/text/gqa_verify_batch.h#L57)<!-- prefix: "inline Paged" -->

A nonfinite visible score, visible V scale byte above 160, or invalid partial or merged result triggers FP64 replay for that query head. The merge requires finite numerators and BF16 outputs and a finite positive weight sum. Replay widens the decoded FP32 KV, processes the visible sequence in order, and rounds to BF16. Other heads keep their results. The captured sequence includes both FP64 head replay and ordered lane fallback. [Head replay](../src/mxfp8_gqa.cu#L716)<!-- prefix: "__device__ _" --> · [Replay dispatch](../src/mxfp8_gqa.cu#L791)<!-- prefix: "__global__ v" -->

## Verification

Follow the [batch graph test runbook](../tests/batch/README.md) to build the release server, `mxfp8-gqa-check`, and `batch-graph-lanes`. Enable `--mxfp8-gqa-check=y` and `--batch-graph-lanes=y` in XMake. The runner checks bound operators, lane reuse, fault recovery, and HTTP parity between default replay and eager execution. Its CPU regression rejects a mask with the correct population count that includes a zero-budget slot. [Test target](../xmake.lua#L270)<!-- prefix: "target(\"batc" -->

On the reserved B200, compare eager and graph greedy IDs and full BF16 rows on identical inputs. Check singleton replay, request rebinding, and a zero-budget peer. The width sweep checks eager execution at logical widths two through 32 for the fixture. Follow the [packed route drift rule](long-paged-decode.md#packed-route-drift-evidence). [Lane assertions](../tests/batch/check_graph_evidence.py#L161)<!-- prefix: "def verify_l" -->

For legacy narrow replay, exercise inline and long strides in one graph. Reject first-use long-score allocation at different lanes. Check retained bytes, eager continuation, and executable reuse after cancellation and slot reuse. Verify repeated bindings reuse existing allocations and request bytes return to zero after retirement.

Run the singleton probe for slot 31 warmup, replacement, and launch rejection. Require matching eager output after rejection. Check both singleton narrow-window edges and adjacent lengths. Require replay at 5,120 visible tokens, eager attention at 5,121, and steady request-owned score bytes. See [singleton parity](lessons.md#l007--singleton-route-parity).

Follow the [score-backed singleton checks](singleton-gqa-decode.md#verification).

Set `--mtp-steps 0` for the [streaming fixtures](development.md#streaming-verification) and [text checks](text-serving.md#verification). Compare default selection with `--eager`. Match IDs, finish reasons, and usage. Correlate K0 masks with CUDA graph launches and batch-stream activity. Require one executable per eligible width and zero launches on eager steps. Check singleton decode at capacity one and wider capacities. For positive depths, follow the [verification graph checks](mtp.md#verification-graph-checks).

Follow the [wide K0 runbook](../tests/batch/README.md#wide-k0-serving). Check full-width eager decode and narrower replay after slot reuse at capacities 16 and 32. Require a replay with logical slot 31 in the mask at capacity 32.

On the fixed checkpoint, follow the [grouped head runbook](../tests/decoder/README.md#grouped-decode-lm-head). Match full BF16 logits and greedy and seeded IDs. Check masks during capture, replay, mixed steps, and slot reuse. Require one winner copy per eligible group and per-lane copies after pin OOM or with sampled, constrained, or diagnostic rows. Run the [rejected pin release check](lessons.md#l010--rejected-packed-winner-pin-release).

Use the [admission runbook](../tests/batch/README.md). Check resident pressure separately from timing. Verify the caps, extra-client wait, slot reuse, and row accounting at a binding override. Inject an admission allocation failure with an active peer. Require HTTP 503 for the failed request and unchanged peer IDs. Require HTTP 200 from `/ready` and the next completion. Inject a terminal CUDA error or failed drain in a separate server process. Require HTTP 500 for a new completion, HTTP 503 from `/ready`, and unchanged caps from `/health`. Check capacity one, wider batches, and prefix-cache serving.

Inject capture failure and pre-submission launch rejection. Require eager continuation with matching outputs. Inject a post-submission failure to check service error propagation and resource retention until safe cleanup. Check slot reuse under Compute Sanitizer. Retain the direct C++ graph checks and the [MTP checks](mtp.md#verification).

For singleton page-parallel GQA, check both window edges and their neighboring lengths against an FP64 oracle over the same packed KV bytes. Include page remaps, masked future tokens, and insufficient scratch. Check input bytes and scratch guards. Cross both edges with serial eager and with fresh and reused graphs at matching KV capacity. Match the ordered fallback bitwise. Run the [GQA verifier](development.md#p2-gqa-verification).

Exercise finite Q and KV that overflow FP32 scores or page-local value sums. Build the packed KV with the native append operator. When ordered replay and the FP64 oracle agree on a finite result, require the fallback to match in eager and graph modes.

For legacy narrow decode, replay one executable with rebound page tables, KV, and positions. Mix lanes inside and outside both window edges. Check each result against the FP64 oracle. Require bitwise parity with the corresponding eager operator. The mixed replay test calls the ordered operator explicitly for masked lanes. It checks that only active score storage changes. Check future-token masking and scratch guards. In width-four replay, rotate score overflow across the two members of a pair. Check that only the affected head recovers. Check recovery from large values and cancellation. [Legacy replay test](../tests/mxfp8_gqa_prefill_check.cu#L482)<!-- prefix: "void check_p" --> · [Recovery test](../tests/mxfp8_gqa_prefill_check.cu#L559)<!-- prefix: "void check_p" -->

## Timing

Compare default graph selection with `--eager` on one binary. Keep prefill policy fixed and prefix reuse off. Match graph and prefill policies across releases. Freeze requests, arrivals, device, and warmups. Record cold capture separately. Verify widths in a separate trace. Disable observers and tracing for timing. Measure whole-request latency separately from replay. Frozen W3 and W6 begin decode above inline score capacity.

Before timing, repeat each native route for stable output IDs. If native builds diverge, compare full BF16 logit rows at the first differing step on their common history. Record the winner margin. Use the [strict concurrent policy](performance.md#native-concurrent-workloads) for HTTP serving.
