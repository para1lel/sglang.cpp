# Batch CUDA graphs

## Responsibilities and sources of truth

| Part | Responsibility | Source |
| --- | --- | --- |
| Server | Validate graph options, construct the batch service, and report batch steps. | [CLI option](../apps/sglang-server.cpp#L49)<!-- prefix: "if (option =" --> · [Validation](../apps/sglang-server.cpp#L89)<!-- prefix: "if (argument" --> · [Batch route](../apps/sglang-server.cpp#L216)<!-- prefix: "BatchService" --> · [Trace](../apps/server/batch_service.cpp#L181)<!-- prefix: "void BatchSe" --> |
| Decoder batch | Select eligible decode steps, bind lane state, and return winners by slot. | [Batch options](../include/sglang/decoder_batch.h#L23)<!-- prefix: "struct Decod" --> · [Eligibility](../src/decoder_batch.cpp#L251)<!-- prefix: "const bool g" --> · [Readback](../src/decoder_batch.cpp#L271)<!-- prefix: "for (int slo" --> |
| CUDA resources | Recover from eligible pre-submission failures and retain resources while completion is uncertain. | [Fallback](../src/decoder_resources.cpp#L354)<!-- prefix: "bool Allocat" --> · [Cleanup](../src/decoder_resources.cpp#L210)<!-- prefix: "CleanupError" --> |

## Runtime path

Run `sglang-server` with `--batch-cuda-graphs` and `--max-active 2`, `3`, or `4`. Leave MTP off and omit `--eager`. Either `--fp8-prefill` policy is valid. The batch service passes the flag as `DecoderBatchOptions::cuda_graphs`. Batch execution defaults to eager. MTP-off `--max-active 1` keeps the serial graph path unless `--eager` is set. [Batch construction](../apps/server/batch_service.cpp#L90)<!-- prefix: "batch = std:" -->

The first eligible step per width runs eager. The next eligible step uses thread-local capture and launches on the batch stream. Widths share batch-owned scratch. Before each launch, upload each packed lane's token, position, page table, and KV and GDN pointers. The graph runs embedding, decoder layers, and per-lane scoring. The batch maps winners back to logical slots and synchronizes before publishing or retiring lanes. [Warmup and bindings](../src/decoder_batch.cpp#L199)<!-- prefix: "bool Decoder" --> · [Capture](../src/decoder_forward.cpp#L316)<!-- prefix: "void Decoder" -->

With `--trace-batches`, a replay emits `graph_slots`. Each set bit identifies a replayed slot. The population count (popcount) gives the number of participating lanes. For example, mask `9` identifies slots 0 and 3. The `width` field counts all selected lanes. `request_bytes` excludes batch graph scratch and CUDA graph allocations. [Slot mask](../src/decoder_batch.cpp#L279)<!-- prefix: "stats.graph_" --> · [Request accounting](../src/decoder_batch.cpp#L189)<!-- prefix: "std::size_t " -->

## Constraints and failure behavior

Each batch caches one executable per actual decode width of two, three, or four. Each lane's zero-based decode position must be below its exact GQA score capacity, which is at most 5,120 slots. Larger KV allocations can qualify at early positions. Mixed steps, prefill, and width one use eager execution. Zero-budget lanes can finish on the same step without entering the graph. [Zero-budget completion](../src/decoder_batch.cpp#L313)<!-- prefix: "if (!lane->r" -->

Scratch OOM and recognized capture, instantiation, or synchronous launch rejections can disable one width for the batch lifetime. Eager retry requires ended capture, a drained stream, and destroyed handles. Other widths remain eligible. Unexpected errors or failures after submission disable the batch and fail current and subsequent service requests. Cleanup retains graph resources and request owners while completion is uncertain. [Service failure](../apps/server/batch_service.cpp#L100)<!-- prefix: "failure = st" -->

## Verification

Follow the [batch graph test runbook](../tests/batch/README.md) to build the release server, `mxfp8-gqa-check`, and `batch-graph-lanes`. Enable `--mxfp8-gqa-check=y` and `--batch-graph-lanes=y` in XMake. The runner checks bound operators, lane reuse, fault recovery, and HTTP parity under both prefill policies. Its CPU regression rejects a mask with the correct population count that includes a zero-budget slot. [Test target](../xmake.lua#L158)<!-- prefix: "option(\"batc" -->

On the reserved B200, compare each width against eager with identical inputs under the same prefill policy. Use a C++ probe to compare greedy IDs and a score observer to compare full BF16 rows. Cross positions 127–129. Change slot masks and lane positions at one width. Cancel and reuse a slot. Cross from position 5,119 to 5,120 with enough KV capacity. Select a zero-budget lane alongside replaying lanes. After retirement, require zero request bytes.

Run the [streaming fixtures](development.md#streaming-verification) and [text checks](text-serving.md#verification) on both HTTP modes. Match per-request IDs, finish reasons, and usage. Correlate every replay mask with its CUDA launch and device activity on the batch stream. Require one reused executable per eligible width and zero batch launches on eager steps. Reject CLI combinations with width one, `--eager`, or positive MTP depth.

Inject capture failure and pre-submission launch rejection. Require eager continuation at that width and replay at other widths. Inject a post-submission failure to check service error propagation and resource retention until safe cleanup. Check slot reuse under Compute Sanitizer. Retain the default and explicit width-one graph checks and the [serial MTP checks](mtp.md#verification).

## Timing

Compare graph opt-in with batch eager on the same binary and prefill policy. Freeze requests, arrivals, device, and warmups. Record cold capture separately and verify actual decode widths in a separate trace. Disable observers and trace output for timing. The frozen W3 and W6 prompt lengths put decode beyond the graph position limit.

An earlier run before parallel MTP used one release binary on B200 GPU3 with driver 590.48.01. Both arms used `--max-active 4 --fp8-prefill exact`. The graph arm added `--batch-cuda-graphs`. MTP, observers, and tracing were off. The process order was eager, graph, graph, eager, with two warmups per process and 14 measured cohorts per arm. Each cohort sent four concurrent requests with 19–29 prompt tokens and returned 32 IDs per request. Maximum prompt plus output was 61 tokens. Timed SSE matched. Median output tokens/s increased from 110.83 to 113.65. Median cohort TPOT fell from 26.41 to 24.18 ms/token. The distributions overlapped, and the two eager process medians were 84.28 and 113.20 tokens/s. The historical record is `artifacts/batch-graph-acceptance/report.md`.
