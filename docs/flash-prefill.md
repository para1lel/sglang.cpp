# Native MXFP8 flash prefill

## Responsibilities and sources of truth

| Part | Responsibility | Source |
| --- | --- | --- |
| Decoder | Select the target prefill policy and append KV before attention. | [Policy](../src/decoder_prefill.cpp#L764)<!-- prefix: "const bool t" --> · [Attention call](../src/decoder_prefill.cpp#L219)<!-- prefix: "const auto s" --> |
| Native CUDA operator | Select TMEM Flash, regular Flash, staged or direct ordered tiles, or protected single-row attention. | [Dispatch](../src/mxfp8_gqa_prefill.cu#L905)<!-- prefix: "cudaError_t " --> |
| Long-tile TMEM | Scan the visible prefix in order or merge three partitions. | [Paired attention](../src/text/mxfp8_gqa_prefill_tmem.cu#L633)<!-- prefix: "__global__ v" --> · [Single-CTA attention](../src/text/mxfp8_gqa_prefill_tmem.cu#L527)<!-- prefix: "__global__ v" --> · [Split merge](../src/text/mxfp8_gqa_prefill_tmem.cu#L730)<!-- prefix: "__global__ v" --> |
| Serving | Select actual tile rows before dispatch. | [Serial prefill](text-serving.md#serial-prefill) · [Batch planner](batch-graphs.md#runtime-path) |

## Runtime path

Eligible B200 text prefill uses the native Flash dispatcher. The full-tile
path checks the zero-based start, aligned buffers, and available device
resources. Exact scoring, cached prompt prefill, and media use separate
language prefill paths.

| Automatic TMEM route | Start position for 1,024 query rows |
| --- | --- |
| [Raw paired ordered](../src/mxfp8_gqa_prefill.cu#L930)<!-- prefix: "if (rows == " --> | 0 |
| [Staged or raw paired ordered](../include/sglang/text/mxfp8_gqa_prefill_paired.cuh#L426)<!-- prefix: "if (position" --> | 1,024–7,168, at 1,024-token boundaries |
| Staged or raw paired ordered | 8,192–31,744, at 1,024-token boundaries or inside a 128-token page |
| [Raw paired ordered](../src/mxfp8_gqa_prefill.cu#L942)<!-- prefix: "if (default_" --> | 32,768–130,048, at 1,024-token boundaries |

On 148-SM SM100 devices, automatic dispatch tries ordered TMEM for 256–512 rows, start ≥8,192, and visibility ≤32,768. Both scale bases need eight-byte alignment. [Partial TMEM selection](../src/mxfp8_gqa_prefill.cu#L918)<!-- prefix: "const bool r" -->

The decoder can batch staged paired attention inside a [whole-tile MTP projection group](native-prefill-workspace.md#projection-pairs-for-mtp). All four grants must have 1,024 rows and the same start at a 1,024-token boundary from 1,024 through 7,168. After appending KV for each lane, the decoder tries one predecode launch and one attention launch for the group. A declined batch retries each lane through the ordinary dispatcher on the same stream. [Partial-grant projection groups](native-prefill-workspace.md#partial-grant-projection-groups) run attention per lane. [Shape guard](../src/decoder_prefill.cpp#L408)<!-- prefix: "bool can_bat" --> · [Batch and fallback](../src/decoder_prefill.cpp#L392)<!-- prefix: "if (staged_b" -->

Use the batch planner and serial tile rules for current tile caps. [Batch planner](batch-graphs.md#runtime-path) · [Serial prefill](text-serving.md#serial-prefill)

Flash and TMEM follow the page table and process causal 64-key tiles using BF16 products, FP32 accumulation, and online normalization.

Regular Flash packs BF16 probabilities into PV registers in reduction order. `cp.async` overlaps the next tile's KV loads with computation. Partial blocks stage zeros, mask scores, and skip output and repair for inactive queries. [Regular Flash](../src/mxfp8_gqa_prefill.cu#L672)<!-- prefix: "__global__ v" -->

When regular Flash handles a full tile from position 1,024, it tries one B200 wave of 144 blocks. Each block uses 88 rows and 352 threads. [Wide dispatch](../src/mxfp8_gqa_prefill.cu#L757)<!-- prefix: "if (aligned_" -->

Aligned partial tiles from position 4,096 size CTAs for 12 head pairs and the device SM count. Try 32 or 48 rows if they fit one wave. Try 88 when it fits one wave and 64 needs two. On 148 SMs, these choices cover grants of 144–384, 385–576, and 769–1,023 rows, respectively. The 32- and 48-row CTAs use 256 threads for staging and repair, with 128 and 192 for MMA. [Partial dispatch](../src/mxfp8_gqa_prefill.cu#L764)<!-- prefix: "if (aligned_" -->

The 32-, 48-, and 88-row variants use `ldmatrix.x4` to load two BF16 K or V fragments in MMA order. A warp skips masking if all queries are active and see all 64 keys. It skips PV rescaling if every history multiplier is `+1`. Extra warps in an 88-row block finish before repair. [Fragment loads](../src/mxfp8_gqa_prefill.cu#L349)<!-- prefix: "__device__ _" --> · [Score mask](../src/mxfp8_gqa_prefill.cu#L635)<!-- prefix: "const bool f" --> · [PV rescaling](../src/mxfp8_gqa_prefill.cu#L550)<!-- prefix: "if constexpr" -->

### Long-tile TMEM pipeline

The Blackwell `sm_100a` target uses `tcgen05` and tensor memory (TMEM) for scores and FP32 value accumulators. Regular Flash stays in the `sm_100` target. [TMEM build target](../xmake.lua#L69)<!-- prefix: "target(\"sgla" -->

TMA double-buffers raw KV. Producers pack finite MXFP8 into BF16 at scale exponents 15–148 and check other scales individually.

In the 64-query TMEM tile, four warps compute softmax and eight convert V. PV overlaps the next K conversion. [Pipeline](../src/text/mxfp8_gqa_prefill_tmem.cu#L527)<!-- prefix: "__global__ v" --> · [Conversion](../include/sglang/text/mxfp8_gqa_prefill_tmem_kv.cuh#L74)<!-- prefix: "__device__ _" -->

Single-CTA ordered TMEM uses a four-lane softmax sum, BF16 probabilities, FP32 normalization, and BF16 output. Partial tiles mask inactive queries and load visible keys. [Output](../include/sglang/text/mxfp8_gqa_prefill_staged.cuh#L4)<!-- prefix: "__device__ _" --> · [Partial tile](../src/text/mxfp8_gqa_prefill_tmem.cu#L436)<!-- prefix: "if (end - fi" --> · [Launch](../src/text/mxfp8_gqa_prefill_tmem.cu#L941)<!-- prefix: "if (ordered)" -->

The automatic lane-local 256-row route uses 32 queries per head pair and alternates score, probability, and V buffers. Its issue warp overlaps the next QK with current softmax. Grouped ragged launches use 64-query tiles. [Issuing warp](../include/sglang/text/mxfp8_gqa_prefill_pipeline.cuh#L67)<!-- prefix: "__device__ v" --> · [Grouped launch](../include/sglang/text/mxfp8_gqa_prefill_tmem_batch.cuh#L13)<!-- prefix: "__global__ v" -->

Each paired cluster handles 128 queries and one head pair. TMA refills released full pages. Partial loads zero out-of-bounds operands. [Tail staging](../include/sglang/text/mxfp8_gqa_prefill_tmem_kv.cuh#L135)<!-- prefix: "if (tail) {" --> · [Page transition](../src/text/mxfp8_gqa_prefill_tmem.cu#L708)<!-- prefix: "const bool n" -->

Paired full pages use a 128-byte TMA swizzle for V. Token and vector remapping limits load and store bank conflicts. E8M0 scales cover 32 E4M3 channels per token and KV head. [V map](../src/text/mxfp8_gqa_prefill_tmem.cu#L807)<!-- prefix: "result = enc" --> · [Producer mapping](../include/sglang/text/mxfp8_gqa_prefill_tmem_kv.cuh#L115)<!-- prefix: "const int fi" --> · [Scale indexing](../include/sglang/text/mxfp8_gqa_prefill_common.cuh#L78)<!-- prefix: "__device__ s" -->

Two stages alternate scores, BF16 probabilities, and V. Producers await QK before reusing decoded K. Once both CTAs publish K, a rank-zero warp issues the next QK during softmax. [Stages](../include/sglang/text/mxfp8_gqa_prefill_tmem_kv.cuh#L25)<!-- prefix: "struct align" --> · [Issuer](../src/text/mxfp8_gqa_prefill_tmem.cu#L612)<!-- prefix: "__device__ _" --> · [K handoff](../src/text/mxfp8_gqa_prefill_tmem.cu#L715)<!-- prefix: "pair_publish" -->

PV awaits both CTAs' BF16 probabilities and V, then releases operands. Softmax preserves 64-key reduction order and rescales the FP32 accumulator after the preceding PV. The final wait precedes output, deallocation, and repair. [Operand wait](../src/text/mxfp8_gqa_prefill_tmem.cu#L625)<!-- prefix: "pair_wait(tm" --> · [Correction](../src/text/mxfp8_gqa_prefill_tmem.cu#L684)<!-- prefix: "tmem_correct" --> · [Final wait](../src/text/mxfp8_gqa_prefill_tmem.cu#L690)<!-- prefix: "tmem_mma_wai" --> · [Repair](../src/text/mxfp8_gqa_prefill_tmem.cu#L572)<!-- prefix: "__device__ v" -->

The batched staged kernel uses a separate grid Z index for each lane and keeps its page table, causal limit, and protected repair. Each CTA uses 256 threads with one K and one V producer thread. BF16 output uses aligned 16-byte stores. The ordered softmax and value accumulation follow the singleton staged path. [Kernel](../include/sglang/text/mxfp8_gqa_prefill_paired.cuh#L196)<!-- prefix: "__global__ v" --> · [Output](../include/sglang/text/mxfp8_gqa_prefill_staged.cuh#L4)<!-- prefix: "__device__ _" -->

Raw paired capture records one node with maps by value and zero global scratch. Keep KV, pages, queries, and output valid through replay. Intermediate prefill can skip scoring. [Arguments](../src/text/mxfp8_gqa_prefill_tmem.cu#L633)<!-- prefix: "__global__ v" --> · [Batch scoring](https://github.com/para1lel/sglang.cpp/blob/93648269d852e0fee78f7af4d1c3ffe4897aa361/src/decoder_batch.cpp#L731)<!-- prefix: "if (rows[slo" -->

Split TMEM uses three contiguous partitions per query block and head pair. Its 76,087,296 scratch bytes hold FP32 numerators, maxima, and masses. Merge rescales, combines, normalizes, and rounds to BF16. Allocation and release share one stream. [Scratch allocation](../src/text/mxfp8_gqa_prefill_tmem.cu#L952)<!-- prefix: "status = cud" -->

## Constraints and failure behavior

Regular Flash needs 16-byte-aligned payloads and 144–1,024 rows. Visibility caps at 32,768, or 131,072 for full tiles. Visibility equals start plus rows. One row uses ordinary forward execution. [Eligibility](../src/mxfp8_gqa_prefill.cu#L927)<!-- prefix: "if (rows \u003e= " --> · [Decoder](../src/decoder_prefill.cpp#L733)<!-- prefix: "if (rows == " -->

The 32-, 48-, and 88-row variants need 135,440, 152,464, and 195,024 dynamic shared-memory bytes respectively. The 64-row variants need 169,488 bytes asynchronously or 135,696 synchronously. Each needs static allocation as well. Asynchronous staging needs 8-byte alignment for both scale bases. Capacity or alignment can select another variant when it fits.

TMEM requires compute capability 10.0 and signed 32-bit TMA token coordinates. Single-CTA ordered and split TMEM need 214,016 dynamic shared-memory bytes plus static allocation. Split TMEM needs CUDA memory pools. [Resource checks](../src/text/mxfp8_gqa_prefill_tmem.cu#L875)<!-- prefix: "std::optiona" -->

Raw paired TMEM requires four 16-byte-aligned KV bases, a 416-thread kernel, a resident two-CTA cluster, and 218,112 dynamic shared-memory bytes plus static allocation. [Paired checks](../include/sglang/text/mxfp8_gqa_prefill_paired.cuh#L390)<!-- prefix: "std::optiona" --> · [Device and occupancy](../include/sglang/text/mxfp8_gqa_prefill_paired.cuh#L347)<!-- prefix: "std::optiona" -->

Unsupported raw paired clusters, kernel configurations, encoders, or tensor maps fall through to regular Flash before submission. Missing buffers or insufficient pages return `cudaErrorInvalidValue`. Propagate unexpected CUDA setup errors and the launch status. See [staged scratch and fallback](native-prefill-workspace.md#constraints-and-failure-behavior) for staged submission rules. [Encoder](../src/text/mxfp8_gqa_prefill_tmem.cu#L770)<!-- prefix: "std::optiona" --> · [Submission](../include/sglang/text/mxfp8_gqa_prefill_paired.cuh#L414)<!-- prefix: "return cudaL" -->

Other valid shapes, unaligned payloads, and a known regular Flash shared memory shortfall use the [ordered fallback dispatch](ordered-gqa-prefill.md#constraints-and-failure-behavior). CUDA discovery, configuration, and launch errors propagate.

The automatic 32-query tile retries 64-query TMEM if kernel setup is unsupported. Single-CTA TMEM can fall through before submission on device, resource, encoder, or automatic tensor-map mismatch. Missing buffers or pages and explicit encoding failures return `cudaErrorInvalidValue`. Unexpected CUDA failures propagate. After split allocation, the launcher attempts release and returns the first launch error or release status.

Protected attention repairs affected rows and head pairs when a query or visible KV value falls outside Flash's numerical range. Ordered TMEM repairs after release. Split TMEM marks partitions for repair at merge. Invalid KV affects only queries that can see it. Protected attention retains selective FP64 recovery. [Single-CTA ordered repair](../include/sglang/text/mxfp8_gqa_prefill_pipeline.cuh#L32)<!-- prefix: "__device__ v" -->

### Reference implementations

SGLang supplies separate Q/KV offsets and an optional fixed split size. FlashInfer sizes tiles from GQA-packed lengths and can split and merge KV. CUTLASS SM100 FMHA uses persistent scheduling and specialized warps. Native partial tiling keeps the full visible scan and 64-key reduction order. [SGLang planner](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/python/sglang/srt/layers/attention/flashinfer_backend.py#L2608-L2626) · [FlashInfer planner](https://github.com/flashinfer-ai/flashinfer/blob/77039051b0be921788dc2784ec41423b558878ec/include/flashinfer/attention/scheduler.cuh#L545-L654) · [CUTLASS FMHA](https://github.com/NVIDIA/cutlass/blob/87a5dc63939551f910647e4789a8f8f77ad97003/examples/77_blackwell_fmha/kernel/sm100_fmha_fwd_kernel_tma_warpspecialized.hpp#L164-L178)

FlashInfer's [Blackwell paged context API](https://github.com/flashinfer-ai/flashinfer/blob/77039051b0be921788dc2784ec41423b558878ec/flashinfer/prefill.py#L8577-L8709) exposes NVFP4 block scales. Its public source leaves internal probability precision unspecified. [FlashInfer FA2](https://github.com/flashinfer-ai/flashinfer/blob/77039051b0be921788dc2784ec41423b558878ec/include/flashinfer/attention/prefill.cuh#L2188-L2290) uses BF16 PV operands with BF16 queries and scaled FP8 V. [FlashAttention-4 on SM100](https://github.com/Dao-AILab/flash-attention/blob/aae6770da7dfd2bafd57d2d0ee3d1428b62f4334/flash_attn/cute/flash_fwd_sm100.py#L550-L592) uses BF16 PV with BF16 V and an FP32 accumulator. Its [early probability publication](https://github.com/Dao-AILab/flash-attention/blob/aae6770da7dfd2bafd57d2d0ee3d1428b62f4334/flash_attn/cute/flash_fwd_sm100.py#L2933-L2977) provides a scheduling reference. Compare [DeepGEMM](https://github.com/sgl-project/DeepGEMM/blob/e3f6b84093dab1213f50cd1865d847365697ef23/csrc/apis/gemm.hpp#L194-L217) separately for [FP8 projections](fp8-prefill-projections.md) with matched effective scales.

## Verification

Check BF16 results against independent FP64 and compare logits and IDs. Test causality, remaps, guards, repair, graph replay, and dispatch fallbacks.

Run the [Flash regression](../tests/mxfp8_gqa_flash_check.cu#L879)<!-- prefix: "void check_f" --> on a reserved B200 using the [checker commands](development.md#p2-gqa-verification). Against FP64, require sampled maximum absolute error below 0.08 and normalized RMSE below 0.005. [FP64 limits](../tests/mxfp8_gqa_flash_check.cu#L397)<!-- prefix: "if (!(max_ab" -->

The [partial boundary cases](../tests/mxfp8_gqa_flash_check.cu#L806)<!-- prefix: "void check_p" --> cover the 143/144 and 191/192/193 row boundaries and unaligned K/V scale parity. [Tail recovery](../tests/mxfp8_gqa_flash_check.cu#L839)<!-- prefix: "void check_t" --> confines nonfinite outputs to affected channels on the final active row.

The [long-context cases](../tests/mxfp8_gqa_flash_check.cu#L748)<!-- prefix: "void check_l" --> check 32,768/32,769-token partial tiles, the 131,072-token cap, causality, and overflow recovery. The aligned 1,023-row case checks geometry by SM count and shared memory. At 32,769, compare wide and synchronous Flash.

Check ordered and protected fallback bitwise against the ordered FP32 reference for finite scores and quotients. Check FP64 recovery independently. Use the [exact prefill runbook](../tests/prefill/README.md) for the separate `exact` contract.

For partial Flash changes, match BF16 outputs bitwise on identical checkpoint Q and KV. Match full-vocabulary BF16 logits and greedy IDs across prefill and decode with identical grants and teacher tokens. Cover unaligned starts, remaps, unwritten tails, and unchanged shapes. Weight eager timings by recorded tile counts.

The [lower-edge cases](../tests/mxfp8_gqa_flash_check.cu#L727)<!-- prefix: "void check_p" --> check dispatch and alignment at 4,096. [Graph checks](../tests/mxfp8_gqa_flash_check.cu#L335)<!-- prefix: "void check_d" --> require one node for the automatic path. The [historical split checks](https://github.com/para1lel/sglang.cpp/blob/93648269d852e0fee78f7af4d1c3ffe4897aa361/tests/mxfp8_gqa_flash_check.cu#L352)<!-- prefix: "void check_t" --> verified graph dependencies and scratch lifetime.

### Paired ordered TMEM verification

The [aligned regression](../tests/mxfp8_gqa_flash_check.cu#L624)<!-- prefix: "void check_p" --> checks dispatch, replay, causality, and 1,023-row parity with synchronous 64-row Flash using unaligned V scales. The [unaligned regression](../tests/mxfp8_gqa_flash_check.cu#L706)<!-- prefix: "void check_u" --> checks page offsets 1, 63, 64, and 127, replay, short page counts, and causal repair.

For paired changes, replay aligned and unaligned 1,024-row checkpoint captures with remaps. Compare all BF16 outputs with the prior kernel and wide Flash. Require one paired graph node and eager parity. Mutate one visible V on each of a full and partial page. Check earlier rows and unaffected heads. Run memcheck, initcheck, and synccheck on repair with unwritten tails. Record racecheck launch size and repair coverage.

Time raw paired graph replay in A/B/B/A order. Weight starts by launch count. See [eager staging checks](native-prefill-workspace.md#verification) and [serving gates](performance.md#scheduler-and-reporting-policy).

Check raw dispatch at zero and eager staging at aligned starts from 1,024 through 7,168. Check starts 31,744 and 32,768, the visibility cap, and off-grid starts.

### Ordered TMEM operator verification

Archived remapped operator replays matched wide Flash bitwise. For ordered changes, repeat remaps and causal boundaries. Compare behavior with visible KV NaNs and query infinity. Keep operator parity and timing separate from checkpoint and serving measurements.

### Split TMEM operator verification

Verify graph replay, output guards, and causal invariance at 65,536, 98,304, and 131,072 visible tokens. Compare with FP64 on identical packed KV. Inject unsupported device or resource settings and split or merge failures to check fallback, error propagation, and scratch release. The archived probe measured lower operator latency including the merge.

### TMEM checkpoint verification

The historical split TMEM checkpoint probe changed greedy first tokens at
65,536 and 98,304 prompt tokens. Repeated captures matched within each arm.
The source of the drift remains unresolved. Keep split TMEM operator evidence
separate from full-checkpoint quality and the W6 serving gate. [Archived Flash and TMEM verification and reproduction](https://github.com/para1lel/sglang.cpp/blob/f6c74b50d57dff48137d2e4a19c0a9c99a15bd20/docs/flash-prefill.md#tmem-checkpoint-verification) · [Previous guide with exact operator results](https://github.com/para1lel/sglang.cpp/blob/4dd8c887c278c67744ed80e7feb2a825a92197a2/docs/flash-prefill.md#ordered-tmem-operator-verification) · [Serving gate](performance.md#measurement)

The retired page-128 TMA study matched the prior split producer bitwise and
reduced operator latency. It changed first tokens from wide Flash at 16,384
and 32,768 prompt tokens and left serving qualification open. The archived
study retains its selector, fallback rules, and reproduction checks. [Page-128 study](https://github.com/para1lel/sglang.cpp/blob/4dd8c887c278c67744ed80e7feb2a825a92197a2/docs/sm100-page128-prefill.md) · [Measured selector source](https://github.com/para1lel/sglang.cpp/tree/93648269d852e0fee78f7af4d1c3ffe4897aa361)
