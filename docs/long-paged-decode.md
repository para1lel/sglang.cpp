# Wide long paged decode

## Responsibilities and sources of truth

| Part | Responsibility | Source |
| --- | --- | --- |
| Decoder batch | Mark K0 text decode lanes under the tensor-core policy. | [Lane eligibility](../src/decoder_batch.cpp#L711)<!-- prefix: "work[count -" --> |
| Eager forward | Compact eligible attention reads and order KV append, attention, and gating on the batch stream. | [Forward dispatch](../src/decoder_forward.cpp#L298)<!-- prefix: "void attend_" --> |
| CUDA operator | Validate launch metadata, split the KV scan, merge partials, and recover individual heads. | [Launch](../include/sglang/text/mxfp8_gqa_decode.cuh#L378)<!-- prefix: "cudaError_t " --> · [Split](../include/sglang/text/mxfp8_gqa_decode.cuh#L274)<!-- prefix: "__global__ v" --> · [Merge](../src/mxfp8_gqa.cu#L656)<!-- prefix: "__global__ v" --> · [Recovery](../include/sglang/text/mxfp8_gqa_decode.cuh#L332)<!-- prefix: "__global__ v" --> |

## Runtime path

Start the batch server with `--fp8-prefill tensor-core` to enable the route for ordinary K0 text decode. The eager packed forward must contain more than four rows. Eligible lanes see 32,768–262,144 tokens inclusive, counting the token being appended. The forward compacts only the eligible attention reads, so one eligible lane can use the operator in a wider forward. Each lane supplies its own absolute position, KV cache, and page table. [Packed dispatch](../src/decoder_forward.cpp#L301)<!-- prefix: "const bool w" --> · [Position check](../include/sglang/decoder_cuda.cuh#L107)<!-- prefix: "inline const" --> · [Context ceiling](../include/sglang/text_attention.h#L9)<!-- prefix: "inline const" -->

Each layer appends the new KV before submitting that lane's grouped attention read. Each split block shares MXFP8 K and V loads across two query heads of one KV head. Sixteen page-aligned partitions divide the visible sequence. Within a partition, tiles of up to 1,024 tokens update the running maximum, weight sum, and value numerator in FP32. The merge rescales the partials against the global maximum and rounds the quotient to BF16. The recovery launch runs before attention gating on the same stream. [Compaction and launch](../src/decoder_forward.cpp#L319)<!-- prefix: "if (lane.mod" --> · [Split](../include/sglang/text/mxfp8_gqa_decode.cuh#L274)<!-- prefix: "__global__ v" --> · [Merge](../src/mxfp8_gqa.cu#L656)<!-- prefix: "__global__ v" -->

Each long lane uses 399,360 bytes of its existing 496,640-byte logits allocation for partials. The alias requires 16-byte alignment and remains live until the attention launches consume it. The stream orders later scoring writes to that allocation. [Scratch alias](../src/decoder_forward.cpp#L243)<!-- prefix: "PagedDecodeB" --> · [Scratch size](../include/sglang/decoder_cuda.cuh#L92)<!-- prefix: "inline const" -->

## Constraints and failure behavior

Default `exact` policy, media, MTP, and prefill rows retain their existing routes. Out-of-window lanes in a wide forward use the row dispatch. Widths one through four use the [existing eager and graph paths](batch-graphs.md#page-parallel-gqa-decode). The primitive accepts one through 32 descriptors, independently of the serving width requirement. [Eligibility](../src/decoder_batch.cpp#L711)<!-- prefix: "work[count -" --> · [Row dispatch](../src/decoder_forward.cpp#L327)<!-- prefix: "check(read_g" --> · [Primitive](../src/mxfp8_gqa.cu#L967)<!-- prefix: "cudaError_t " -->

The launcher checks the complete group before submission. An invalid count, null required pointer, unaligned scratch address, out-of-window position, or insufficient declared cache or page-table capacity returns `cudaErrorInvalidValue`. The caller supplies valid device page entries and enough scratch for each lane. The launcher stops at the first synchronous CUDA launch error. Earlier launches can still be in flight. The forward propagates the error through the batch execution path. [Validation and launch](../include/sglang/text/mxfp8_gqa_decode.cuh#L378)<!-- prefix: "cudaError_t " --> · [Cache coverage](../src/mxfp8_gqa.cu#L156)<!-- prefix: "bool cache_c" --> · [Forward error](../src/decoder_forward.cpp#L334)<!-- prefix: "check(status" -->

Nonfinite scores or partials, large visible V scales, and invalid merged output can trigger token-ordered FP64 recovery for the affected head. Unaffected heads keep their split results. The parallel reduction changes the order of operations and can change BF16 logits and generated IDs, including the greedy winner when the ordered route ties. [Merge guard](../src/mxfp8_gqa.cu#L695)<!-- prefix: "const bool i" --> · [Head replay](../src/mxfp8_gqa.cu#L706)<!-- prefix: "__device__ _" -->

## Verification

Use a Release SM100 build and one reserved B200. Run the [GQA verifier](development.md#p2-gqa-verification) and the [wide K0 runbook](../tests/batch/README.md#wide-k0-serving) for the retained routes.

Build and run both checks in the [long decode test runbook](../tests/decoder/README.md#wide-long-context-decode). The operator check uses shuffled pages, distinct lane positions, masked tails, and guarded scratch and output. It compares ordered attention and FP64 on identical packed KV, exercises head-local score overflow and large-value cancellation, and checks invalid descriptors and injected launch rejections. [Operator verifier](../tests/decoder/long_decode_check.cu#L248)<!-- prefix: "void run(boo" --> · [Error cases](../tests/decoder/long_decode_check.cu#L162)<!-- prefix: "void check_e" -->

The checkpoint route check traces a width-five forward with three eligible lanes under the tensor-core policy. It exercises sparse slots, mixed context lengths, fixed-seed sampling, slot reuse, and narrow eager and graph paths. The short-context `exact` control and the pinned baseline exercise the existing route. Compare emitted IDs and use the optional BF16 capture to examine any changed greedy choice or near tie at the three width-five steps. [Route verifier](../tests/decoder/long_decode_routes.cpp#L217)<!-- prefix: "int main(int" -->

For the direct checkpoint control, admit all 16 W4 requests before advancing. Keep prompt IDs, decode histories, and sparse logical slots fixed across arms. Exercise packed widths 16, 11, 5, 4, and 1. Trace the long kernels at eligible widths and the existing eager and replay routes at narrower widths. Compare every emitted ID and record decode step time separately from prefill. The direct full-length control checks all 16 requests through their length stops.

For serving, use the [frozen protocol](../benchmarks/concurrent/README.md#regenerate-and-measure) and the same client and server policy in both arms. Keep graph-enabled K0, capacity 32, a 32,768-token step cap, tensor-core prefill, and prefix caching disabled. W4 submits 16 prompts of 32,768 IDs and requests 1,024 output IDs each. W5 extends coverage to 32 prompts with 4,096 output IDs each. Complete two warmups and five measured cohorts for a valid row, under the original 90-second socket and 360-second cohort limits. Preserve partial IDs, completion counts, and deadline failures for invalid rows. Score throughput and latency gates only over complete valid cohorts.

Record prefill overlays as separate experiments. Use identical prefill code on both sides of an isolated long-decode comparison. The [Flash prefill contract](flash-prefill.md#constraints-and-failure-behavior) defines the selected 144-row floor.

### Combined 256-row Flash experiment

The earlier combined build used the 256-row Flash floor, landed GDN, and wide long decode. In W4, it completed all 112 HTTP requests and matched all 114,688 frozen greedy IDs. It missed all four external gates.

| W4 metric | Combined build | Frozen gate |
| --- | ---: | ---: |
| Median output IDs/s | 62.209 | > 358.723 |
| p95 TTFT (s) | 81.627 | ≤ 50.227 |
| p95 TPOT (ms) | 200.896 | ≤ 59.994 |
| p95 E2E (s) | 261.143 | ≤ 46.590 |

On W5, the same build completed zero of 32 requests on the first warmup. All 32 reached the 90-second socket timeout before the first ID. The client skipped measured cohorts, leaving the W5 throughput and latency gates unscored.

See `artifacts/takeover-w4-integrated/handoff.md` for the combined build identities and raw evidence.
