# Native MXFP8 flash prefill

## Responsibilities and sources of truth

| Part | Responsibility | Source |
| --- | --- | --- |
| Decoder | Select the target prefill policy and append KV before attention. | [Policy](../src/decoder_prefill.cpp#L824)<!-- prefix: "const bool t" --> · [Attention call](../src/decoder_prefill.cpp#L352)<!-- prefix: "const auto s" --> |
| Native CUDA operator | Select TMEM Flash, regular Flash, staged or direct ordered tiles, or protected single-row attention. | [Dispatch](../src/mxfp8_gqa_prefill.cu#L908)<!-- prefix: "cudaError_t " --> |
| Long-tile TMEM | Scan the visible prefix in order or merge three partitions. | [Paired attention](../src/text/mxfp8_gqa_prefill_tmem.cu#L654)<!-- prefix: "__global__ v" --> · [Single-CTA attention](../src/text/mxfp8_gqa_prefill_tmem.cu#L545)<!-- prefix: "__global__ v" --> · [Split merge](../src/text/mxfp8_gqa_prefill_tmem.cu#L751)<!-- prefix: "__global__ v" --> |
| Serving | Select actual tile rows before dispatch. | [Serial prefill](text-serving.md#serial-prefill) · [Batch planner](batch-graphs.md#runtime-path) |

## Runtime path

The CLI, server, and C++ default to `exact`. With `tensor-core`, `SGLANG_EXPERIMENTAL_TMEM_PREFILL` selects full-tile TMEM by zero-based query start. [TMEM selection](../src/mxfp8_gqa_prefill.cu#L936)<!-- prefix: "if (rows == " -->

| Setting | TMEM route | Start position for 1,024 query rows |
| --- | --- | --- |
| Unset or `ragged` | Paired ordered | 8,192–31,744 inclusive, at a multiple of 1,024 or inside a 128-token page |
| `paired` | Paired ordered | Any start from 8,192 through 130,048 |
| `ordered` | Single-CTA ordered | At least 8,192, divisible by 64 |
| `1` | Three-part split | At least 32,768, divisible by 64 |
| Any other set value, including empty or `wide` | Disabled | Regular Flash and its fallbacks |

With the variable unset, 148-SM SM100 devices try ordered TMEM for 256–512 rows, start ≥8,192, and visibility ≤32,768. Both scale bases need eight-byte alignment. The `ragged` override tries the same row and visibility range from start 4,096 under the explicit TMEM resource checks. All other set values retain their routes. [Partial TMEM selection](../src/mxfp8_gqa_prefill.cu#L923)<!-- prefix: "const bool d" -->

The [SM100 page-128 opt-ins](sm100-page128-prefill.md) precede native dispatch. Media, MTP, and tensor-core prefix-cache tiles cap at 128 rows. The [workspace](native-prefill-workspace.md) reuses paired staging.

Flash and TMEM follow the logical page table and process causal 64-key tiles. QK and PV use BF16 products, FP32 accumulation, and online normalization.

Regular Flash packs BF16 probabilities into PV registers in reduction order. `cp.async` overlaps the next tile's KV loads with computation. Partial blocks stage zeros, mask scores, and skip output and repair for inactive queries. [Regular Flash](../src/mxfp8_gqa_prefill.cu#L674)<!-- prefix: "__global__ v" -->

From position 1,024, full Flash tiles try one B200 wave of 144 blocks with 88 rows and 352 threads each. [Wide dispatch](../src/mxfp8_gqa_prefill.cu#L759)<!-- prefix: "if (aligned_" -->

Aligned partial tiles from position 4,096 size CTAs for 12 head pairs and the device SM count. Try 32 or 48 rows if they fit one wave. Try 88 when it fits one wave and 64 needs two. On 148 SMs, these choices cover grants of 144–384, 385–576, and 769–1,023 rows, respectively. The 32- and 48-row CTAs use 256 threads for staging and repair, with 128 and 192 for MMA. [Partial dispatch](../src/mxfp8_gqa_prefill.cu#L766)<!-- prefix: "if (aligned_" -->

The 32-, 48-, and 88-row variants use `ldmatrix.x4` to load two BF16 K or V fragments in MMA order. A warp skips masking if all queries are active and see all 64 keys. It skips PV rescaling if every history multiplier is `+1`. Extra warps in an 88-row block finish before repair. [Fragment loads](../src/mxfp8_gqa_prefill.cu#L351)<!-- prefix: "__device__ _" --> · [Score mask](../src/mxfp8_gqa_prefill.cu#L637)<!-- prefix: "const bool f" --> · [PV rescaling](../src/mxfp8_gqa_prefill.cu#L552)<!-- prefix: "if constexpr" -->

### Long-tile TMEM pipeline

The Blackwell `sm_100a` target uses `tcgen05` and tensor memory (TMEM) for scores and FP32 value accumulators. Regular Flash stays in the `sm_100` target. [TMEM build target](../xmake.lua#L67)<!-- prefix: "target(\"sgla" -->

TMA double-buffers raw KV. Producers convert finite MXFP8 to packed BF16 at scale exponents 15–148 and use scalar range checks elsewhere.

In the 64-query TMEM tile, four warps compute softmax and eight convert V. PV overlaps the next K conversion. [Pipeline](../src/text/mxfp8_gqa_prefill_tmem.cu#L545)<!-- prefix: "__global__ v" --> · [Conversion](../include/sglang/text/mxfp8_gqa_prefill_tmem_kv.cuh#L74)<!-- prefix: "__device__ _" -->

Single-CTA ordered TMEM uses a four-lane softmax sum, BF16 probabilities, FP32 normalization, and BF16 output. Partial tiles mask inactive queries and load visible keys. [Output](../src/text/mxfp8_gqa_prefill_tmem.cu#L332)<!-- prefix: "__device__ _" --> · [Partial tile](../src/text/mxfp8_gqa_prefill_tmem.cu#L454)<!-- prefix: "if (end - fi" --> · [Launch](../src/text/mxfp8_gqa_prefill_tmem.cu#L962)<!-- prefix: "if (ordered)" -->

The automatic lane-local 256-row route uses 32 queries per head pair and alternates score, probability, and V buffers. Its issue warp overlaps the next QK with current softmax. Grouped ragged launches use 64-query tiles. [Issuing warp](../include/sglang/text/mxfp8_gqa_prefill_pipeline.cuh#L67)<!-- prefix: "__device__ v" --> · [Grouped launch](../include/sglang/text/mxfp8_gqa_prefill_tmem_batch.cuh#L13)<!-- prefix: "__global__ v" -->

Each paired cluster handles 128 queries and one head pair. TMA refills released full pages. Partial loads zero out-of-bounds operands. [Tail staging](../include/sglang/text/mxfp8_gqa_prefill_tmem_kv.cuh#L135)<!-- prefix: "if (tail) {" --> · [Page transition](../src/text/mxfp8_gqa_prefill_tmem.cu#L729)<!-- prefix: "const bool n" -->

Paired full pages use a 128-byte TMA swizzle for raw V. Remapping token rows and 16-byte vectors limits bank conflicts on raw loads and BF16 stores. E8M0 scales cover 32 E4M3 channels per token and KV head. [V map](../src/text/mxfp8_gqa_prefill_tmem.cu#L828)<!-- prefix: "result = enc" --> · [Producer mapping](../include/sglang/text/mxfp8_gqa_prefill_tmem_kv.cuh#L115)<!-- prefix: "const int fi" --> · [Scale indexing](../include/sglang/text/mxfp8_gqa_prefill_common.cuh#L78)<!-- prefix: "__device__ s" -->

Two stages alternate scores, BF16 probabilities, and V. Producers await QK before reusing decoded K. Once both CTAs publish K, a rank-zero warp issues the next QK during softmax. [Stages](../include/sglang/text/mxfp8_gqa_prefill_tmem_kv.cuh#L25)<!-- prefix: "struct align" --> · [Issuer](../src/text/mxfp8_gqa_prefill_tmem.cu#L633)<!-- prefix: "__device__ _" --> · [K handoff](../src/text/mxfp8_gqa_prefill_tmem.cu#L736)<!-- prefix: "pair_publish" -->

PV awaits both CTAs' BF16 probabilities and V, then releases operands. Softmax preserves 64-key reduction order and rescales the FP32 accumulator after the preceding PV. The final wait precedes output, deallocation, and repair. [Operand wait](../src/text/mxfp8_gqa_prefill_tmem.cu#L646)<!-- prefix: "pair_wait(tm" --> · [Correction](../src/text/mxfp8_gqa_prefill_tmem.cu#L705)<!-- prefix: "tmem_correct" --> · [Final wait](../src/text/mxfp8_gqa_prefill_tmem.cu#L711)<!-- prefix: "tmem_mma_wai" --> · [Repair](../src/text/mxfp8_gqa_prefill_tmem.cu#L590)<!-- prefix: "__device__ v" -->

Capture records one node with maps by value and zero global scratch. Keep KV, pages, queries, and output valid through replay. Intermediate prefill can skip scoring. [Arguments](../src/text/mxfp8_gqa_prefill_tmem.cu#L654)<!-- prefix: "__global__ v" --> · [Batch scoring](../src/decoder_batch.cpp#L750)<!-- prefix: "if (rows[slo" -->

Split TMEM uses three contiguous partitions per query block and head pair. Its 76,087,296 scratch bytes hold FP32 numerators, maxima, and masses. Merge rescales, combines, normalizes, and rounds to BF16. Allocation and release share one stream. [Scratch allocation](../src/text/mxfp8_gqa_prefill_tmem.cu#L973)<!-- prefix: "status = cud" -->

## Constraints and failure behavior

Regular Flash accepts 144–1,024 rows through 32,768 visible tokens and full tiles through 131,072. Visibility equals start plus row count. Both payload bases need 16-byte alignment. One row uses ordinary forward execution. [Eligibility](../src/mxfp8_gqa_prefill.cu#L933)<!-- prefix: "if (rows \u003e= " --> · [Decoder](../src/decoder_prefill.cpp#L794)<!-- prefix: "if (rows == " -->

The 32-, 48-, and 88-row variants need 135,440, 152,464, and 195,024 dynamic shared-memory bytes respectively. The 64-row variants need 169,488 bytes asynchronously or 135,696 synchronously. Each needs static allocation as well. Asynchronous staging needs 8-byte alignment for both scale bases. Capacity or alignment can select another variant when it fits.

TMEM requires compute capability 10.0 and signed 32-bit TMA token coordinates. Single-CTA ordered and split TMEM need 214,016 dynamic shared-memory bytes plus static allocation. Split TMEM needs CUDA memory pools. [Resource checks](../src/text/mxfp8_gqa_prefill_tmem.cu#L896)<!-- prefix: "std::optiona" -->

Raw paired TMEM needs 16-byte alignment for all four KV bases. It needs a 416-thread kernel image, a resident two-CTA cluster, and 218,112 dynamic shared-memory bytes plus static allocation. [Paired checks](../include/sglang/text/mxfp8_gqa_prefill_paired.cuh#L361)<!-- prefix: "std::optiona" -->

Unsupported clusters, kernel configurations, encoders, or tensor maps fall through to regular Flash before submission. Missing buffers or too few pages return `cudaErrorInvalidValue`. Unexpected CUDA setup errors and the paired launch status return to the caller. [Encoder](../src/text/mxfp8_gqa_prefill_tmem.cu#L791)<!-- prefix: "std::optiona" --> · [Submission](../include/sglang/text/mxfp8_gqa_prefill_paired.cuh#L385)<!-- prefix: "return cudaL" -->

Other valid shapes, unaligned payloads, and a known regular Flash shared memory shortfall use the [ordered fallback dispatch](ordered-gqa-prefill.md#constraints-and-failure-behavior). CUDA discovery, configuration, and launch errors propagate.

The automatic 32-query tile retries 64-query TMEM if kernel setup is unsupported. Single-CTA TMEM can fall through before submission on device, resource, encoder, or automatic tensor-map mismatch. Missing buffers or pages and explicit encoding failures return `cudaErrorInvalidValue`. Unexpected CUDA failures propagate. After split allocation, the launcher attempts release and returns the first launch error or release status.

Protected attention repairs affected rows and head pairs when a query or visible KV value falls outside Flash's numerical range. Ordered TMEM repairs after release. Split TMEM marks partitions for repair at merge. Invalid KV affects only queries that can see it. Protected attention retains selective FP64 recovery. [Single-CTA ordered repair](../include/sglang/text/mxfp8_gqa_prefill_pipeline.cuh#L32)<!-- prefix: "__device__ v" -->

### Reference implementations

SGLang supplies separate Q/KV offsets and an optional fixed split size. FlashInfer sizes tiles from GQA-packed lengths and can split and merge KV. CUTLASS SM100 FMHA uses persistent scheduling and specialized warps. Native partial tiling keeps the full visible scan and 64-key reduction order. [SGLang planner](https://github.com/sgl-project/sglang/blob/f04d886bf79aa3d6237cbb12a3311838eb9f9bd8/python/sglang/srt/layers/attention/flashinfer_backend.py#L2608-L2626) · [FlashInfer planner](https://github.com/flashinfer-ai/flashinfer/blob/77039051b0be921788dc2784ec41423b558878ec/include/flashinfer/attention/scheduler.cuh#L545-L654) · [CUTLASS FMHA](https://github.com/NVIDIA/cutlass/blob/87a5dc63939551f910647e4789a8f8f77ad97003/examples/77_blackwell_fmha/kernel/sm100_fmha_fwd_kernel_tma_warpspecialized.hpp#L164-L178)

FlashInfer's [Blackwell paged context API](https://github.com/flashinfer-ai/flashinfer/blob/77039051b0be921788dc2784ec41423b558878ec/flashinfer/prefill.py#L8577-L8709) exposes NVFP4 block scales. Its public source leaves internal probability precision unspecified. [FlashInfer FA2](https://github.com/flashinfer-ai/flashinfer/blob/77039051b0be921788dc2784ec41423b558878ec/include/flashinfer/attention/prefill.cuh#L2188-L2290) uses BF16 PV operands with BF16 queries and scaled FP8 V. [FlashAttention-4 on SM100](https://github.com/Dao-AILab/flash-attention/blob/aae6770da7dfd2bafd57d2d0ee3d1428b62f4334/flash_attn/cute/flash_fwd_sm100.py#L550-L592) uses BF16 PV with BF16 V and an FP32 accumulator. Its [early probability publication](https://github.com/Dao-AILab/flash-attention/blob/aae6770da7dfd2bafd57d2d0ee3d1428b62f4334/flash_attn/cute/flash_fwd_sm100.py#L2933-L2977) provides a scheduling reference. Compare [DeepGEMM](https://github.com/sgl-project/DeepGEMM/blob/e3f6b84093dab1213f50cd1865d847365697ef23/csrc/apis/gemm.hpp#L194-L217) separately for [FP8 projections](fp8-prefill-projections.md) with matched effective scales.

## Verification

Check BF16 results against independent FP64 and compare logits and IDs. Test causality, remaps, guards, repair, graph replay, and dispatch fallbacks.

Run the [Flash regression](../tests/mxfp8_gqa_flash_check.cu#L952)<!-- prefix: "void check_f" --> on a reserved B200 using the [checker commands](development.md#p2-gqa-verification). Against FP64, require sampled maximum absolute error below 0.08 and normalized RMSE below 0.005. [FP64 limits](../tests/mxfp8_gqa_flash_check.cu#L451)<!-- prefix: "if (!(max_ab" -->

The [partial boundary cases](../tests/mxfp8_gqa_flash_check.cu#L879)<!-- prefix: "void check_p" --> cover the 143/144 and 191/192/193 row boundaries and unaligned K/V scale parity. [Tail recovery](../tests/mxfp8_gqa_flash_check.cu#L912)<!-- prefix: "void check_t" --> confines nonfinite outputs to affected channels on the final active row.

The [long-context cases](../tests/mxfp8_gqa_flash_check.cu#L805)<!-- prefix: "void check_l" --> check 32,768/32,769-token partial tiles, the 131,072-token cap, causality, and overflow recovery. The aligned 1,023-row case checks geometry by SM count and shared memory. At 32,768, ordered TMEM matches unaligned-scale output. At 32,769, compare wide and synchronous Flash.

Check ordered and protected fallback bitwise against the ordered FP32 reference for finite scores and quotients. Check FP64 recovery independently. Use the [exact prefill runbook](../tests/prefill/README.md) for the separate `exact` contract.

For partial Flash changes, match BF16 outputs bitwise on identical checkpoint Q and KV. Match full-vocabulary BF16 logits and greedy IDs across prefill and decode with identical grants and teacher tokens. Cover unaligned starts, remaps, unwritten tails, and unchanged shapes. Weight eager timings by recorded tile counts.

The [lower-edge cases](../tests/mxfp8_gqa_flash_check.cu#L786)<!-- prefix: "void check_t" --> check dispatch and alignment at ordered TMEM start 8,192 and split start 32,768. [Graph checks](../tests/mxfp8_gqa_flash_check.cu#L382)<!-- prefix: "void check_d" --> require one ordered node or four split nodes. [Split checks](../tests/mxfp8_gqa_flash_check.cu#L352)<!-- prefix: "void check_t" --> verify dependencies and scratch lifetime.

### Paired ordered TMEM verification

The [aligned regression](../tests/mxfp8_gqa_flash_check.cu#L676)<!-- prefix: "void check_p" --> checks dispatch, replay, causality, and 1,023-row parity with synchronous 64-row Flash using unaligned V scales. The [unaligned regression](../tests/mxfp8_gqa_flash_check.cu#L765)<!-- prefix: "void check_u" --> checks page offsets 1, 63, 64, and 127, replay, short page counts, and causal repair.

For paired changes, replay aligned and unaligned 1,024-row checkpoint captures with remaps. Compare all BF16 outputs with the prior kernel and wide Flash. Require one paired graph node and eager parity. Mutate one visible V on each of a full and partial page. Check earlier rows and unaffected heads. Run memcheck, initcheck, and synccheck on repair with unwritten tails. Record racecheck launch size and repair coverage.

Time raw paired graph replay in A/B/B/A order. Weight starts by launch count. See [eager staging checks](native-prefill-workspace.md#verification) and [serving gates](performance.md#scheduler-and-reporting-policy).

### Ordered TMEM operator verification

Three remapped B200 W5/07 operator replays matched wide Flash bitwise and reduced graph replay time. FP64 sampling covered the late tile. KV NaN and query infinity matched wide Flash and passed memcheck, synccheck, and racecheck. Checkpoint-logit and serving qualification remain open.

### Split TMEM operator verification

The independent B200 check passed long-context dispatch, graph, numerical, and causal cases. At 65,536, 98,304, and 131,072 visible tokens, normalized RMSE against FP64 was 0.224%, 0.219%, and 0.221% over 12,800 values per shape. Replay at the first eligible start restored BF16 outputs after overwrite and preserved guards. The 131,072-token mutation preserved all 399,360 BF16 values in the first 65 rows bitwise.

At 1,024 query rows, the paired median time including the merge fell from 18.206 to 7.171 ms at 65,536 visible tokens and from 37.343 to 8.937 ms at 131,072. Memcheck and synccheck each reported zero errors on the 65,536-token operator probe.

A fault probe verified device and resource fallbacks, CUDA error propagation, and scratch release requests after split or merge failure.

### TMEM checkpoint verification

The Qwen3.8-27B-FP8 probe compared split TMEM (`=1`) with `ec1ca540` Flash on frozen W6 lane 0 at 65,536, 98,304, and 131,072 IDs. Greedy IDs changed at the first two lengths and matched at 131,072. With teacher forcing, second argmax matched at every length. The 193-token `exact` control matched all 496,640 BF16 logits across two steps.

At 98,304 prompt tokens, TMEM ran on 1,024 of 1,536 prefill GQA calls. The first argmax changed from 7,035 to 1,455. Full-logit cosine was 0.821539, top-10 overlap was 1/10, and negative log likelihood for ID 7,035 rose by 20.994807 nats. The second argmax was 29,877 in both arms after forcing ID 7,035, with cosine 0.998866. Repeated captures matched all 496,640 BF16 logits within each arm.

Identical-input replays used the first eligible tile at position 32,768 in layer 3 and the tile with the largest sampled difference at position 79,872 in layer 59, both zero-based. Full-tile TMEM versus regular Flash normalized RMSE was 0.06130% and 0.09469%, respectively. Against independent FP64 over seven query rows, all 24 query heads, and all visible tokens per row, regular Flash and TMEM errors were 0.16567% and 0.16538% on the first tile and 0.16412% and 0.15900% on the later tile. Page remaps and causal-tail mutations passed in both kernels.

Split TMEM on the first eligible call preserved argmax 7,035 (+0.010311 nats NLL). The first 256 eligible calls changed it to 3,470 (+18.999520 nats). The source of full-model drift remains unresolved. Two tile replays and one document leave corpus quality and serving acceptance open. Split TMEM requires opt-in because of this drift.

#### Reproduce the first-token comparison

Build the generator with the [P2 build command](development.md#p2-gqa-verification). Regenerate the [W6 fixtures](../benchmarks/concurrent/README.md#regenerate-and-measure) with the Qwen3.8-27B-FP8 tokenizer. Set `FIXTURES` to that fixture directory, `CHECKPOINT` to the checkpoint, and use the same `BUILD_WORK_DIR` and `GPU_UUID` as for the checker.

```bash
python - "$FIXTURES/W6/00.request.json" > prompt-96k.ids <<'PY'
import json
import sys

with open(sys.argv[1]) as fixture:
    prompt = json.load(fixture)["prompt"]
assert len(prompt) == 131072
print(*prompt[:98304], sep="\n")
PY
sha256sum -c <<'SUM'
735ac7151bc0611bbeee83bacf148ea0b4d83ec0000b1e3312c72cece82e5670  prompt-96k.ids
SUM
export CUDA_VISIBLE_DEVICES="$GPU_UUID"
unset SGLANG_SERIAL_PREFILL_ROWS
for setting in 0 1; do
    SGLANG_EXPERIMENTAL_TMEM_PREFILL="$setting" \
        "$BUILD_WORK_DIR/build/bin/sglang-generate" --checkpoint "$CHECKPOINT" \
        --device 0 --prompt-file prompt-96k.ids --fp8-prefill tensor-core \
        --mtp-steps 0 --temperature 0 --max-new-tokens 1 --top-logits 248320 \
        --warmup 0 --repeats 1 > "prefill-96k-$setting.json"
done
```

Compare `output_token_ids` and `step_top_logits` across arms. The full vocabulary is emitted in score order, so join by `token_id` before computing cosine and NLL. This command tests the common first-token history. The original second-step teacher comparison and tile replay used instrumented probes.

### Partial Flash verification

The B200 regression passed with the partial Flash implementation. On the base build, it stopped at the 144-row dispatch assertion. The 193-row recovery case preserved all 1,179,648 earlier BF16 outputs and produced exactly 262 nonfinite outputs on the final row.

An independent B200 operator probe observed ordered attention at 143 and 1,025 rows, and Flash at 144 rows. It also checked remapped pages, output guards, unaligned K and V separately, shared memory fallback, and CUDA error propagation.

The sampled 144-row case at 8,192 visible tokens had 0.0188% normalized RMSE against independent FP64. At 147 rows and 32,768 visible tokens, all 256 BF16 outputs of the repaired head on the final active row matched exact. The probe's eager and graph outputs matched bitwise, and its memcheck reported zero errors. The raw logs are in `artifacts/takeover-tail-integration/operator/`.

On four documents with 128 teacher-forced tokens each, switching the 144-row tail to flash changed 17 of 512 top IDs and shifted mean NLL by −0.00923 nats per token. Switching the 1,023-row tail changed 27 of 512 top IDs and shifted mean NLL by +0.02774. The 143-row and 1,024-row controls had identical logits on the evaluated rows. See `artifacts/takeover-partial-flash-test/quality/results/summary.json`.

### Regular Flash integration evidence

On B200, late W4 tile replay fell from 8.963 to 6.610 ms with asynchronous staging and direct probability packing. A separate comparison measured 6.611 to 4.667 ms with 88-row blocks. Each change preserved replayed BF16 outputs and two full-vocabulary logit steps on three standalone prompts. Standalone W4/00 profiler runs attributed 1,769.751 and 1,233.610 ms to attention before and after the wide change.

Two native W4 cohorts per arm matched all streamed IDs with asynchronous staging. Pooled p95 TTFT fell from 68.280 to 62.071 s and median throughput rose from 170.408 to 189.908 IDs/s. Pooled p95 TPOT rose from 21.631 to 26.452 ms/ID. The changed arm missed the frozen TTFT ceiling. The wide diagnostic missed the rate and E2E thresholds and lacked an uncontended paired control. [Frozen serving acceptance](performance.md#scheduler-and-reporting-policy) remains open.

Earlier regular Flash sped up nine eligible shapes by 8.92–12.77× against ordered attention. W2 changed a greedy ID and increased mean teacher-forced NLL by 0.00460 nats/token. W3 matched IDs and both arms missed the frozen gates. Both W4 arms stopped at the first warmup with zero complete streams. See `artifacts/takeover-flash-prefill-default/handoff.md` and `artifacts/takeover-flash-prefill-review/report.md`.

### W6 warmups and diagnostics

The [frozen W6 protocol](../benchmarks/concurrent/README.md#conditions-and-gates) uses four 131,072-token prompts and 1,024 IDs per request. GPU6 used 32,768 tokens per step, 32 slots, disjoint prefixes, GDN, and MTP off. The candidate extended full-tile Flash to 131,072 visible tokens. Both native warmups reached the 90-second socket timeout with zero IDs or scored cohorts. Pinned vLLM's two warmups and five measured cohorts matched all 28 archived streams exactly.

| Metric | Pinned vLLM | Frozen W6 gate | Result |
| --- | ---: | ---: | --- |
| Median output IDs/s | 45.467107 | > 48.709172 | Fail |
| p95 TTFT (ms) | 75,540.784412 | ≤ 113,118.073927 | Pass |
| p95 TPOT (ms) | 69.018133 | ≤ 104.925433 | Pass |
| p95 E2E (ms) | 90,104.907074 | ≤ 85,806.011537 | Fail |

One-lane, two-ID diagnostics used the same builds and 32,768-token cap, a 1,800-second timeout, and a 1,810-second deadline. At 40,960 tokens, both returned `16,25` and TTFT fell from 37.386 to 6.115 s. At 131,072, the candidate alone returned `760,45366` with 45.798 s TTFT. See `artifacts/takeover-w6-flash-selected/report.md`.

With an 8,192-token cap, both GPU6 warmups stopped with zero IDs across four lanes. Protected prefill took 74.021 of 95.010 attention seconds in a separate profile. See `artifacts/takeover-w6/report.md`.

Single two-ID prototypes with an 8,192-token cap, 1,800-second timeout, and 1,810-second deadline matched IDs at 40,960 and 131,072 tokens. Longer Flash reduced TTFT. See `artifacts/takeover-w6-flash-code/report.md`.

### Double-buffered TMEM prototype

On GPU6, the prototype compared with sealed TMA at 1,024 query rows with shuffled physical pages. Paired median attention time, including split merge, fell from 12.814 to 4.674 ms at 65,536 visible tokens. At 131,072 visible tokens, it fell from 26.256 to 9.018 ms. Sampled normalized RMSE against FP64 was 0.221% and 0.220%, respectively. The partitioned reduction changed 2,546,820 and 2,555,831 of 6,291,456 BF16 outputs against sealed TMA. Two graph replays per sampled variant matched their own eager outputs bitwise.

The paired two-ID single-lane runs changed greedy IDs at 65,536 and 98,304 prompt tokens. Both IDs matched at 131,072. The prototype and its sealed TMA control used an earlier source revision. See `artifacts/takeover-w6-prefill-pipeline/report.md` for the source identities, diagnostic IDs, and timings.

All four prototype W6 lanes emitted their first ID at 77.243–77.273 s. At the 360-second deadline, the lanes had emitted 705, 705, 705, and 704 IDs. All 2,819 observed IDs matched the frozen prefixes. The first warmup ended with zero completed requests and no scored cohort.

The 84.091 s throughput budget for 4,096 IDs left under 6.82 s for 1,023 decode iterations and delivery. A two-ID trace attributed 9.008 s to first-step decode attention.

Report diagnostic prompt length, concurrency, output budget, and deadlines. W6 acceptance remains open and requires measured cohorts under the [current launch policy](performance.md#scheduler-and-reporting-policy) and [frozen gates](../benchmarks/concurrent/README.md#conditions-and-gates).
