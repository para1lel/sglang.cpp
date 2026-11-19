# Batch CUDA graphs

## Responsibilities and sources of truth

| Part | Responsibility | Source |
| --- | --- | --- |
| Server | Validate graph options, construct the batch service, and report batch steps. | [CLI option](../apps/sglang-server.cpp#L49)<!-- prefix: "if (option =" --> · [Validation](../apps/sglang-server.cpp#L92)<!-- prefix: "if (argument" --> · [Batch route](../apps/sglang-server.cpp#L214)<!-- prefix: "BatchService" --> · [Trace](../apps/server/batch_service.cpp#L228)<!-- prefix: "void trace_b" --> |
| Decoder batch | Select eligible decode steps, bind lane state, and return winners by slot. | [Batch options](../include/sglang/decoder_batch.h#L34)<!-- prefix: "struct Decod" --> · [Eligibility](../src/decoder_batch.cpp#L353)<!-- prefix: "const bool g" --> · [Readback](../src/decoder_batch.cpp#L379)<!-- prefix: "for (int slo" --> |
| CUDA resources | Recover from eligible pre-submission failures and retain resources while completion is uncertain. | [Fallback](../src/decoder_resources.cpp#L354)<!-- prefix: "bool Allocat" --> · [Cleanup](../src/decoder_resources.cpp#L210)<!-- prefix: "CleanupError" --> |

## Runtime path

Run `sglang-server` with `--batch-cuda-graphs`, `--max-active 2`, `3`, or `4`, and `--mtp-steps 0`. Either `--fp8-prefill` policy is valid. Batch execution defaults to eager. The CLI also accepts the graph flag at positive MTP depth, where active MTP lanes keep each step eager. Mixed-depth C++ selections with an active MTP lane also run eager. Width-one K0 uses its serial graph unless `--eager` is set. [Batch construction](../apps/server/batch_service.cpp#L133)<!-- prefix: "batch = std:" -->

The first eligible step per width runs eager. The next eligible step uses thread-local capture on the batch stream. Widths share batch-owned scratch. Capture keeps the device binding addresses fixed. Each launch uploads new tokens, positions, KV page tables, KV and GDN pointers, and GQA score pointers and head strides. The graph runs embedding, decoder layers, and per-lane scoring. The batch maps winners back to logical slots and synchronizes before publishing or retiring lanes. [Warmup and bindings](../src/decoder_batch.cpp#L284)<!-- prefix: "bool Decoder" --> · [Capture](../src/decoder_forward.cpp#L365)<!-- prefix: "void Decoder" -->

With `--trace-batches`, each set bit in `graph_slots` identifies a replayed logical slot. For example, mask `9` identifies slots 0 and 3. `width` counts all selected lanes. `request_bytes` includes request-owned scores and excludes batch scratch and CUDA graph allocations. [Slot mask](../src/decoder_batch.cpp#L387)<!-- prefix: "stats.graph_" --> · [Request accounting](../src/decoder_batch.cpp#L114)<!-- prefix: "std::size_t " -->

## Constraints and failure behavior

Each batch caches one executable per actual decode width of two, three, or four. Sparse selections of the same width reuse it with new bindings. Request retirement leaves the executable available for slot reuse. Mixed steps, prefill, and width one run eager. Zero-budget lanes can finish on the same step without entering the graph. [Zero-budget completion](../src/decoder_batch.cpp#L482)<!-- prefix: "if (!lane->r" -->

Positions 0–5,119 use the logits area in each packed lane's scratch. The first graph binding at a later position allocates scores for that request's page-rounded KV capacity. Each 128-position page needs 12,288 bytes for 24 FP32 scores per position. Capacity is at most 262,144 positions or 24 MiB of scores per lane. W3 reserves 72 pages per lane, or 9,216 positions and 884,736 score bytes. Four W3 lanes add 3,538,944 bytes. The fixed-grid kernels read position and head stride from the current binding. Each request reuses its allocation until retirement after stream completion. [Score allocation](../src/decoder_resources.cpp#L510)<!-- prefix: "ExactGqaWork" --> · [Score launch](../src/mxfp8_gqa.cu#L398)<!-- prefix: "cudaError_t " -->

If the first long-score allocation runs out of memory, the current step runs eager. That request remembers the failure and forces eager on later steps that need its long scores. Allocations already made for other lanes remain owned by those requests and can serve later replays. The executable remains available for other selections and recycled slots. Scratch OOM and recognized capture, instantiation, or synchronous launch rejections can disable one width. That fallback requires ended capture, a drained stream, and destroyed handles. Other widths remain eligible. A target-only step error that escapes fallback disables the batch and fails current and subsequent requests. Cleanup retains resources while completion is uncertain. [Step recovery](../src/decoder_batch.cpp#L460)<!-- prefix: "Steps Decode" -->

## Verification

Follow the [batch graph test runbook](../tests/batch/README.md) to build the release server, `mxfp8-gqa-check`, and `batch-graph-lanes`. Enable `--mxfp8-gqa-check=y` and `--batch-graph-lanes=y` in XMake. The runner checks bound operators, lane reuse, fault recovery, and HTTP parity under both prefill policies. Its CPU regression rejects a mask with the correct population count that includes a zero-budget slot. [Test target](../xmake.lua#L155)<!-- prefix: "option(\"batc" -->

On the reserved B200, compare each width against eager with identical inputs under the same prefill policy. Match greedy IDs and complete BF16 vocabulary rows. Cross positions 127–129 and 5,119–5,121. Exercise inline and long strides in one replay. Change slot masks and request bindings at one width. Reject first-use score allocations at different packed lanes. Check retained bytes, eager continuation, and executable reuse after cancellation and slot reuse. Verify repeated bindings reuse existing allocations. Select a zero-budget lane alongside replaying lanes. After retirement, require zero request bytes.

Run the [streaming fixtures](development.md#streaming-verification) and [text checks](text-serving.md#verification) on both HTTP modes. Match IDs, finish reasons, and usage. Correlate every replay mask with its CUDA launch and device activity on the batch stream. Require one reused executable per eligible width and zero graph launches on eager steps. Check that width one and `--eager` reject the batch graph flag. At positive depths and widths two through four, check that the flag is accepted and produces zero graph launches.

Inject capture failure and pre-submission launch rejection. Require eager continuation at that width and replay at other widths. Inject a post-submission failure to check service error propagation and resource retention until safe cleanup. Check slot reuse under Compute Sanitizer. Retain the default and explicit width-one graph checks and the [MTP checks](mtp.md#verification).

The post-loader smoke used B200 GPU1 and four frozen 8,192-token W3 prompts. Both arms used tensor-core prefill and budgets of 6, 9, 12, and 15. All 42 IDs, finish reasons, and usage matched eager. The IDs matched the frozen output prefixes. CUPTI correlated four replays at width four and two each at widths three and two. The first replay added 798,720 score bytes per lane. Final request bytes were zero. See `artifacts/long-graph-standalone/final-loader/evidence/native/smoke/verified.json`.

## Timing

Compare graph opt-in with batch eager on the same binary and prefill policy. To compare releases, enable batch graphs in both arms and match prefill policy. Freeze requests, arrivals, device, and warmups. Record cold capture separately and verify actual decode widths in a separate trace. Disable observers and trace output for timing. Frozen W3 and W6 begin decode above the inline score capacity.

An earlier run before parallel MTP used one release binary on B200 GPU3 with driver 590.48.01. Both arms used `--max-active 4 --fp8-prefill exact`. The graph arm added `--batch-cuda-graphs`. MTP, observers, and tracing were off. The process order was eager, graph, graph, eager, with two warmups per process and 14 measured cohorts per arm. Each cohort sent four concurrent requests with 19–29 prompt tokens and returned 32 IDs per request. Maximum prompt plus output was 61 tokens. Timed SSE matched. Median output tokens/s increased from 110.83 to 113.65. Median cohort TPOT fell from 26.41 to 24.18 ms/token. The distributions overlapped, and the two eager process medians were 84.28 and 113.20 tokens/s. The historical record is `artifacts/batch-graph-acceptance/report.md`.
