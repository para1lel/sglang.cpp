# Wide long paged decode

## Responsibilities and sources of truth

| Part | Responsibility | Source |
| --- | --- | --- |
| Decoder batch | Mark K0 text decode lanes under the tensor-core policy. | [Lane eligibility](../src/decoder_batch.cpp#L675)<!-- prefix: "work[count -" --> |
| Eager forward | Compact eligible attention reads and order KV append, attention, and gating on the batch stream. | [Forward dispatch](../src/decoder_forward.cpp#L298)<!-- prefix: "void attend_" --> |
| CUDA operator | Validate the group, select a split, merge partials, and recover individual heads. | [Launch](../include/sglang/text/mxfp8_gqa_decode.cuh#L380)<!-- prefix: "cudaError_t " --> · [Tensor-core split](../include/sglang/text/mxfp8_gqa_wide_tc.cuh#L178)<!-- prefix: "__global__ v" --> · [Paired split](../include/sglang/text/mxfp8_gqa_decode.cuh#L274)<!-- prefix: "__global__ v" --> · [Merge](../src/mxfp8_gqa.cu#L656)<!-- prefix: "__global__ v" --> · [Recovery](../include/sglang/text/mxfp8_gqa_decode.cuh#L332)<!-- prefix: "__global__ v" --> |

## Runtime path

Start the batch server with `--fp8-prefill tensor-core` to enable long paged attention for ordinary K0 text decode. The eager packed forward must contain more than four rows. Eligible lanes see 32,768–262,144 tokens inclusive, counting the token being appended. The forward compacts the eligible attention reads. Each lane supplies its absolute position, KV cache, and page table. [Packed dispatch](../src/decoder_forward.cpp#L301)<!-- prefix: "const bool w" --> · [Position check](../include/sglang/decoder_cuda.cuh#L113)<!-- prefix: "inline const" --> · [Context ceiling](../include/sglang/text_attention.h#L9)<!-- prefix: "inline const" -->

Each layer appends the new KV before submitting that lane's grouped attention read. The tensor-core split requires at least five compacted reads and 16-byte-aligned K and V payloads for every lane in the group. The paired split handles groups of one through four or groups with an unaligned payload. Each split divides the visible sequence into 16 page-aligned partitions and clamps the final interval to the visible length. The paired split shares K and V across two query heads of one KV head. [Compaction and launch](../src/decoder_forward.cpp#L319)<!-- prefix: "if (lane.mod" --> · [Split selection](../include/sglang/text/mxfp8_gqa_decode.cuh#L386)<!-- prefix: "bool tensor " -->

Each tensor-core block serves one lane, one partition, and all six query heads of a KV head. Four warps process disjoint 16-token tiles. They decode visible E4M3 payloads and their E8M0 scales into BF16 and share K and V across the six heads. BF16 matrix instructions calculate QK scores and weighted V with FP32 accumulation. The softmax maximum, rescaling, and mass use FP32. Only the numerator multiplies BF16-rounded softmax weights by V. The block merges warp partials in FP32. [KV stage](../include/sglang/text/mxfp8_gqa_wide_tc.cuh#L106)<!-- prefix: "__device__ _" --> · [MMA](../include/sglang/text/mxfp8_gqa_wide_tc.cuh#L50)<!-- prefix: "__device__ _" --> · [Weight rounding](../include/sglang/text/mxfp8_gqa_wide_tc.cuh#L250)<!-- prefix: "const float " -->

Both splits write the same per-head FP32 partials. The common merge rescales the 16 partitions against the global maximum and rounds the quotient to BF16. The recovery launch runs before attention gating on the same stream. [Merge](../src/mxfp8_gqa.cu#L656)<!-- prefix: "__global__ v" --> · [Recovery](../include/sglang/text/mxfp8_gqa_decode.cuh#L332)<!-- prefix: "__global__ v" -->

Each long lane uses 399,360 bytes of its existing 496,640-byte logits allocation for partials. The alias requires 16-byte alignment and remains live until the attention launches consume it. The stream orders later scoring writes to that allocation. [Scratch alias](../src/decoder_forward.cpp#L243)<!-- prefix: "PagedDecodeB" --> · [Scratch size](../include/sglang/decoder_cuda.cuh#L98)<!-- prefix: "inline const" -->

## Constraints and failure behavior

Default `exact` policy, media, MTP, and prefill rows retain their existing routes. Out-of-window lanes in a wide forward use the row dispatch. Packed widths one through four keep their [eager and graph paths](batch-graphs.md#page-parallel-gqa-decode). The long tensor-core split runs in the eager wide forward. The long primitive accepts one through 32 descriptors, independently of the serving width requirement. [Eligibility](../src/decoder_batch.cpp#L675)<!-- prefix: "work[count -" --> · [Row dispatch](../src/decoder_forward.cpp#L327)<!-- prefix: "check(read_g" --> · [Primitive](../src/mxfp8_gqa.cu#L967)<!-- prefix: "cudaError_t " -->

The launcher checks the complete group before submission. An invalid count, null required pointer, unaligned scratch address, out-of-window position, or insufficient declared cache or page-table capacity returns `cudaErrorInvalidValue`. The caller supplies valid device page entries and enough scratch for each lane. The launcher stops at the first synchronous CUDA launch error. Earlier launches can still be in flight. The forward propagates the error through the batch execution path. [Validation and launch](../include/sglang/text/mxfp8_gqa_decode.cuh#L380)<!-- prefix: "cudaError_t " --> · [Cache coverage](../src/mxfp8_gqa.cu#L156)<!-- prefix: "bool cache_c" --> · [Forward error](../src/decoder_forward.cpp#L334)<!-- prefix: "check(status" -->

The tensor-core split flags a query head if an input is nonfinite, a BF16 subnormal, or has magnitude above `2^30`. Visible nonfinite K or V flags all six query heads of that KV group. A nonzero decoded K or V magnitude below `2^-126` or above `2^30` also flags the group. The paired split flags nonfinite visible scores or partials and visible V scale bytes above 160. Both routes flag invalid merged maxima, mass, numerators, or BF16 results. A flagged head replays the visible tokens in order in FP64 from the original packed KV. Other heads keep their split results. [Query guard](../include/sglang/text/mxfp8_gqa_wide_tc.cuh#L200)<!-- prefix: "const bool i" --> · [KV range guard](../include/sglang/text/mxfp8_gqa_wide_tc.cuh#L64)<!-- prefix: "const bool i" --> · [Merge guard](../src/mxfp8_gqa.cu#L695)<!-- prefix: "const bool i" --> · [Head replay](../src/mxfp8_gqa.cu#L706)<!-- prefix: "__device__ _" -->

Both splits change the reduction order. The tensor-core split also rounds numerator weights to BF16. Either choice can change BF16 attention output, logits, and generated IDs, including the greedy winner in a near tie.

## Verification

Use a Release SM100 build and one reserved B200. Run the [GQA verifier](development.md#p2-gqa-verification) and the [wide K0 runbook](../tests/batch/README.md#wide-k0-serving) for the retained routes.

Build and run both checks in the [long decode test runbook](../tests/decoder/README.md#wide-long-context-decode). The operator check uses shuffled pages, distinct lane positions, masked tails, and guarded scratch and output. It compares ordered attention and FP64 on identical packed KV, exercises head-local and KV-group recovery, and checks invalid descriptors and injected launch rejections. [Operator verifier](../tests/decoder/long_decode_check.cu#L301)<!-- prefix: "void run(Mod" --> · [Error cases](../tests/decoder/long_decode_check.cu#L171)<!-- prefix: "void check_e" -->

The checkpoint route check traces width-five forwards with five and three eligible lanes under the tensor-core policy. The first group can use the tensor-core split. The mixed group uses the paired split and exercises sparse slots, fixed-seed sampling, slot reuse, and narrow eager and graph paths. The short-context `exact` control exercises the existing route. Compare emitted IDs and use the optional BF16 capture and forced-token input to examine changed greedy choices on identical histories. [Route verifier](../tests/decoder/long_decode_routes.cpp#L292)<!-- prefix: "int main(int" -->

For the direct checkpoint control, admit all 16 W4 requests before advancing. Keep prompt IDs, decode histories, and sparse logical slots fixed across arms. Exercise packed widths 16, 11, 5, 4, and 1. Trace both split choices and the retained eager and replay routes. Compare each emitted ID and record decode step time separately from prefill. To measure numerical drift on identical histories, capture complete logits and force the same continuation IDs in both arms. Report top-1 changes and teacher-forced negative log-likelihood deltas with the tested token count.

For serving, use the [frozen protocol](../benchmarks/concurrent/README.md#regenerate-and-measure) and the same client and server policy in both arms. Keep graph-enabled K0, capacity 32, a 32,768-token step cap, tensor-core prefill, and prefix caching disabled. W4 submits 16 prompts of 32,768 IDs and requests 1,024 output IDs each. W5 extends coverage to 32 prompts with 4,096 output IDs each. Complete two warmups and five measured cohorts for a valid row, under the original 90-second socket and 360-second cohort limits. Preserve partial IDs, completion counts, and deadline failures for invalid rows. Score throughput and latency gates only over complete valid cohorts.

Record prefill overlays as separate experiments. Use identical prefill code on both sides of an isolated long-decode comparison. The [Flash prefill contract](flash-prefill.md#constraints-and-failure-behavior) defines the selected 144-row floor.

### Isolated tensor-core results

In the GPU5 prototype, direct W4 checkpoint decode took 169.254 ms per median step with the paired split and 105.867 ms with the tensor-core split at width 16. Width 11 fell from 118.674 to 73.230 ms and width five from 56.815 to 33.544 ms. The width-four and singleton routes stayed the same. All 127 generated IDs and graph masks matched across the tested widths.

On one W4 cohort with fixed histories, all 240 decode top-1 predictions matched. Across 59,596,800 decode logits, RMSE was 0.334726 and the largest absolute change was 10.648438. Mean teacher-forced negative log-likelihood changed by +0.000000014 nats. These are direct decode measurements and a short matched-history quality check. See `artifacts/takeover-wide-decode-next/handoff.md` for source identities, traces, and measurements.

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
