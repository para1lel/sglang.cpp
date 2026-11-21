# Native MXFP8 flash prefill

## Responsibilities and sources of truth

| Part | Responsibility | Source |
| --- | --- | --- |
| Decoder | Select the target prefill policy and append KV before attention. | [Policy](../src/decoder_prefill.cpp#L486)<!-- prefix: "const bool t" --> · [Attention call](../src/decoder_prefill.cpp#L293)<!-- prefix: "const auto s" --> |
| Native CUDA operator | Select TMEM Flash, regular Flash, ordered multirow, or protected single-row attention. | [Dispatch](../src/mxfp8_gqa_prefill.cu#L709)<!-- prefix: "cudaError_t " --> |
| Long-tile TMEM | Scan the visible prefix in order or merge three partitions. | [TMEM attention](../src/text/mxfp8_gqa_prefill_tmem.cu#L384)<!-- prefix: "__global__ v" --> · [Split merge](../src/text/mxfp8_gqa_prefill_tmem.cu#L474)<!-- prefix: "__global__ v" --> |
| Serving | Select actual tile rows before dispatch. | [Serial prefill](text-serving.md#serial-prefill) · [Batch planner](batch-graphs.md#runtime-path) |

## Runtime path

The CLI and C++ default `exact` keeps the original MXFP8 path. With `tensor-core`, set `SGLANG_EXPERIMENTAL_TMEM_PREFILL=ordered` to try ordered TMEM or `1` to try split TMEM. An unset variable or any other value leaves portable dispatch on its regular Flash and fallback routes. Media, MTP, and tensor-core prefix-cache tiles cap at 128 rows.

Packed wide Flash remains the default for eligible tensor-core tiles. See [SM100 page-128 split prefill](sm100-page128-prefill.md) for the `sm100` and `sm100-tma` opt-ins.

Regular Flash and both TMEM modes follow the logical page table and process causal 64-key tiles for query-head pairs. QK and PV use BF16 Tensor Core products with FP32 accumulation and online normalization.

Regular Flash packs rounded BF16 probabilities directly into PV registers and preserves the reduction order. Asynchronous staging uses `cp.async` to load the next tile's MXFP8 payloads and scales during current-tile computation. Partial blocks stage zeros, mask scores, and skip output stores and protected repair for inactive queries. [Regular Flash](../src/mxfp8_gqa_prefill.cu#L566)<!-- prefix: "__global__ v" -->

For a full 1,024-row tile starting at position 1,024 or later, regular Flash tries 88-row, 352-thread blocks. The 144 blocks fit in one wave on the 148-SM B200. Extra warps finish before protected repair. [Wide dispatch](../src/mxfp8_gqa_prefill.cu#L674)<!-- prefix: "if (aligned_" -->

In wide blocks, each `ldmatrix.x4` loads two BF16 K or V fragments for MMA operations in the same order. A warp skips score mask checks when all its queries are active and its first query sees all 64 keys. It skips PV accumulator rescaling when every history multiplier in the warp equals `+1`. [Fragment loads](../src/mxfp8_gqa_prefill.cu#L288)<!-- prefix: "__device__ _" --> · [Score mask](../src/mxfp8_gqa_prefill.cu#L609)<!-- prefix: "const bool f" --> · [PV rescaling](../src/mxfp8_gqa_prefill.cu#L508)<!-- prefix: "if (Rows != " -->

### Long-tile TMEM pipeline

The Blackwell `sm_100a` target uses `tcgen05` and tensor memory (TMEM) for scores and FP32 value accumulators. Regular Flash stays in the `sm_100` target. [TMEM build target](../xmake.lua#L67)<!-- prefix: "target(\"sgla" -->

Tensor Memory Accelerator (TMA) copies raw K and V from the physical pages into two alternating shared-memory buffers. Producer warps convert scaled MXFP8 to BF16. Packed conversion handles finite payloads and scale exponents 15–148. Other values use the scalar conversion and its range checks.

Four warps compute softmax while eight producer warps convert V. PV overlaps conversion of the next K tile. Barriers and proxy fences order transfers, conversions, and tensor operations before buffer reuse. [Pipeline](../src/text/mxfp8_gqa_prefill_tmem.cu#L384)<!-- prefix: "__global__ v" --> · [Conversion](../src/text/mxfp8_gqa_prefill_tmem.cu#L99)<!-- prefix: "__device__ _" -->

Ordered TMEM scans the prefix in order with one CTA per 64 query rows and head pair. It keeps wide Flash's four-lane softmax sum and BF16 probability rounding. It writes BF16 directly from TMEM using FP32 reciprocal normalization. Capture adds one kernel node with zero global scratch. [Output](../src/text/mxfp8_gqa_prefill_tmem.cu#L322)<!-- prefix: "__device__ _" --> · [Launch](../src/text/mxfp8_gqa_prefill_tmem.cu#L585)<!-- prefix: "if (ordered)" -->

Split TMEM assigns three contiguous partitions to three CTAs per query block and head pair. It uses 76,087,296 bytes of scratch for FP32 numerators, maxima, and normalization masses. The merge rescales to the global maximum, combines the results, normalizes, and rounds to BF16. Allocation, attention, merge, and release run on one stream. [Scratch allocation](../src/text/mxfp8_gqa_prefill_tmem.cu#L595)<!-- prefix: "status = cud" -->

## Constraints and failure behavior

Regular Flash accepts 144–1,024 query rows through 32,768 visible tokens and only full tiles through 131,072. Visible tokens equal the zero-based start position plus the row count. Both payload bases must be 16-byte aligned. The decoder sends a one-row tile through ordinary forward execution. [Eligibility](../src/mxfp8_gqa_prefill.cu#L723)<!-- prefix: "if (rows \u003e= " --> · [Decoder](../src/decoder_prefill.cpp#L466)<!-- prefix: "if (rows == " -->

Wide staging requires 195,024 dynamic shared-memory bytes. The 64-row variants need 169,488 bytes with asynchronous staging or 135,696 bytes with synchronous staging. Each variant also needs its static allocation. Asynchronous staging requires 8-byte alignment for both scale bases. A capacity shortfall for wide staging selects the 64-row variants. Unaligned scales or insufficient space for asynchronous staging selects synchronous staging when it fits.

Both TMEM modes share the full-tile Flash visibility and payload alignment limits. They require 1,024 rows and a zero-based start divisible by 64. The minimum start is 8,192 for `ordered` and 32,768 for `1`. Both need capability 10.0 and signed 32-bit TMA token coordinates. Opt-in shared memory must fit 214,016 dynamic bytes plus the kernel's static allocation. Split TMEM also requires CUDA memory pools. [Eligibility](../src/mxfp8_gqa_prefill.cu#L726)<!-- prefix: "if (rows == " --> · [Resource checks](../src/text/mxfp8_gqa_prefill_tmem.cu#L516)<!-- prefix: "std::optiona" -->

A TMEM device or resource mismatch falls through to eligible regular Flash. A successful driver lookup without a tensor-map encoder takes the same fallback.

Other valid shapes, unaligned payloads, and a known regular Flash shared memory shortfall use the [ordered fallback dispatch](development.md#ordered-multirow-gqa-prefill). Above 32,768 visible tokens, that dispatch uses protected single-row attention. CUDA discovery, configuration, and launch errors propagate.

The TMEM launcher returns `cudaErrorInvalidValue` for missing required pointers, insufficient page counts, or tensor-map encoding failure. CUDA discovery, configuration, launch, and split allocation errors propagate. After split allocation, the launcher attempts release and returns the first launch failure or the release status.

Protected attention repairs affected rows and head pairs when a query or visible KV value falls outside Flash's numerical range. Ordered TMEM repairs in the same CTA after releasing tensor memory. Split TMEM marks partitions for repair at merge. An invalid KV position affects only queries that can see it. Protected attention retains selective FP64 recovery. [Ordered repair](../src/text/mxfp8_gqa_prefill_tmem.cu#L456)<!-- prefix: "if constexpr" -->

## Verification

For the explicit tensor-core policy, prioritize measured efficiency over bitwise parity with ordered attention. Flash can change BF16 results, logits, and generated IDs. Compare flash with an independent FP64 reference over the same represented inputs. Check causality, page remaps, output guards, repaired rows, and eager versus CUDA Graph replay. Trace dispatch at the eligibility bounds and through alignment and capacity fallback.

Run the [Flash regression](../tests/mxfp8_gqa_flash_check.cu#L714)<!-- prefix: "void check_f" --> on a reserved B200 using the [three checker commands](development.md#p2-gqa-verification). Require sampled maximum absolute error below 0.08 and normalized RMSE below 0.005 against FP64. [FP64 limits](../tests/mxfp8_gqa_flash_check.cu#L437)<!-- prefix: "if (!(max_ab" -->

The [partial boundary cases](../tests/mxfp8_gqa_flash_check.cu#L638)<!-- prefix: "void check_p" --> cover the 143/144-row floor, the 191/192/193-row boundary, and parity with unaligned K or V scales. [Partial recovery](../tests/mxfp8_gqa_flash_check.cu#L671)<!-- prefix: "void check_p" --> limits nonfinite outputs to affected channels on the final active row.

The [long-context cases](../tests/mxfp8_gqa_flash_check.cu#L568)<!-- prefix: "void check_l" --> check the 32,768/32,769 partial-tile boundary, the 131,072-token cap, causality, and overflow recovery. At 32,768 visible tokens, ordered TMEM preserves BF16 output with unaligned scales. Its 32,769-token control checks bitwise parity between wide Flash and synchronous Flash with unaligned scales.

Check ordered and protected fallback bitwise against the ordered FP32 reference for finite scores and quotients. Check FP64 recovery independently. Use the [exact prefill runbook](../tests/prefill/README.md) for the separate `exact` contract.

The [lower-edge cases](../tests/mxfp8_gqa_flash_check.cu#L549)<!-- prefix: "void check_t" --> check dispatch and payload alignment near starts of 8,192 for ordered TMEM and 32,768 for split TMEM. At the first eligible start, each TMEM mode must replay bitwise after output poisoning. [Graph checks](../tests/mxfp8_gqa_flash_check.cu#L368)<!-- prefix: "void check_d" --> require one ordered kernel node or four split nodes. [Split checks](../tests/mxfp8_gqa_flash_check.cu#L338)<!-- prefix: "void check_t" --> also verify dependencies, scratch size, and the released pointer.

### Ordered TMEM operator verification

On B200, three remapped W5/07 operator replays matched wide Flash bitwise and took less CUDA Graph replay time. A sampled FP64 comparison covered the late tile. An injected KV NaN and query infinity matched wide Flash and passed memcheck, synccheck, and racecheck. Checkpoint-logit and end-to-end serving qualification remain open.

### Split TMEM operator verification

The independent B200 release check passed the long-context dispatch, graph, numerical, and causal cases. At 65,536, 98,304, and 131,072 visible tokens, normalized RMSE against FP64 was 0.224%, 0.219%, and 0.221%, respectively, over 12,800 sampled values per shape. At the first eligible start, repeated replay preserved the BF16 output and its guards after the output buffer was overwritten. The 131,072-token causal mutation preserved all 399,360 BF16 values in the first 65 rows bitwise.

At 1,024 query rows, the paired median time including the merge fell from 18.206 to 7.171 ms at 65,536 visible tokens and from 37.343 to 8.937 ms at 131,072. Memcheck and synccheck each reported zero errors on the 65,536-token operator probe.

A separate launch fault probe exercised the device and resource fallbacks, checked CUDA error propagation, and confirmed that split or merge failure still requests scratch release.

### TMEM checkpoint verification

The Qwen3.8-27B-FP8 probe compared split TMEM (`=1`) with regular Flash from `ec1ca540` on frozen W6 lane 0. It truncated the prompt to 65,536, 98,304, and 131,072 IDs. Greedy IDs changed at the first two lengths and matched at 131,072. On the same teacher histories the second argmax matched at all three lengths. The separate 193-token `exact` control matched all 496,640 BF16 logits across its two continuation steps.

At 98,304 prompt tokens, TMEM ran on 1,024 of 1,536 prefill GQA calls. The first argmax changed from 7,035 to 1,455. Full-logit cosine was 0.821539, top-10 overlap was 1/10, and negative log likelihood for ID 7,035 rose by 20.994807 nats. The second argmax was 29,877 in both arms after forcing ID 7,035, with cosine 0.998866. Repeated captures matched all 496,640 BF16 logits within each arm.

Identical-input replays used the first eligible tile at position 32,768 in layer 3 and the tile with the largest sampled difference at position 79,872 in layer 59, both zero-based. Full-tile TMEM versus regular Flash normalized RMSE was 0.06130% and 0.09469%, respectively. Against independent FP64 over seven query rows, all 24 query heads, and all visible tokens per row, regular Flash and TMEM errors were 0.16567% and 0.16538% on the first tile and 0.16412% and 0.15900% on the later tile. Page remaps and causal-tail mutations passed in both kernels.

Using split TMEM on only the first eligible call preserved first argmax 7,035 with a 0.010311-nat NLL increase. Using it on the first 256 eligible calls changed the argmax to 3,470 with an 18.999520-nat increase. The probes leave the source of the full-model drift unresolved. Two replayed tiles and one document cannot establish corpus quality or serving acceptance. This drift is the reason split TMEM requires an experimental opt-in.

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

The [frozen W6 protocol](../benchmarks/concurrent/README.md#conditions-and-gates) uses four concurrent 131,072-token prompts and 1,024 output IDs per request. The matched GPU6 campaign used 32,768 scheduled tokens per step, 32 active slots, disjoint prefixes, and MTP off. Both native builds used landed GDN. The candidate extended full-tile Flash visibility to 131,072 tokens.

Both native arms stopped at the first warmup. All four streams in each arm reached the 90-second socket timeout before the first output ID. Neither arm produced a scored cohort or a native output comparison.

Pinned vLLM completed two warmups and five measured cohorts. All 28 streams matched the archived output IDs and repeated exactly across cohorts. The maintained gates remain the frozen W6 thresholds.

| Metric | Pinned vLLM | Frozen W6 gate | Result |
| --- | ---: | ---: | --- |
| Median output IDs/s | 45.467107 | > 48.709172 | Fail |
| p95 TTFT (ms) | 75,540.784412 | ≤ 113,118.073927 | Pass |
| p95 TPOT (ms) | 69.018133 | ≤ 104.925433 | Pass |
| p95 E2E (ms) | 90,104.907074 | ≤ 85,806.011537 | Fail |

The same builds ran separate one-lane, two-ID diagnostics with the 32,768-token step cap. They used a 1,800-second socket timeout and a 1,810-second cohort deadline. At 40,960 prompt tokens, control and candidate both returned `16,25`. TTFT fell from 37.386 s to 6.115 s. At 131,072 prompt tokens, the candidate returned `760,45366` with 45.798 s TTFT. The 131,072-token diagnostic tested the candidate only.

See `artifacts/takeover-w6-flash-selected/report.md` for the integrated source identities and run record.

The earlier GPU6 campaign used an 8,192-token step cap. Both native arms stopped at the first warmup with zero IDs on all four lanes. A separate profile spent 74.021 s of its 95.010 s attention interval in protected prefill. See `artifacts/takeover-w6/report.md`.

A separate prototype diagnostic used single requests with two output IDs, the historical 8,192-token step cap, a 1,800-second socket timeout, and a 1,810-second cohort deadline. Both arms completed and agreed on both IDs at 40,960 and 131,072 prompt tokens. The longer Flash route reduced TTFT on those requests. See `artifacts/takeover-w6-flash-code/report.md`.

### Double-buffered TMEM prototype

On GPU6, the prototype compared with sealed TMA at 1,024 query rows with shuffled physical pages. Paired median attention time, including split merge, fell from 12.814 to 4.674 ms at 65,536 visible tokens. At 131,072 visible tokens, it fell from 26.256 to 9.018 ms. Sampled normalized RMSE against FP64 was 0.221% and 0.220%, respectively. The partitioned reduction changed 2,546,820 and 2,555,831 of 6,291,456 BF16 outputs against sealed TMA. Two graph replays per sampled variant matched their own eager outputs bitwise.

The paired two-ID single-lane runs changed greedy IDs at 65,536 and 98,304 prompt tokens. Both IDs matched at 131,072. The prototype and its sealed TMA control used an earlier source revision. See `artifacts/takeover-w6-prefill-pipeline/report.md` for the source identities, diagnostic IDs, and timings.

All four prototype W6 lanes emitted their first ID at 77.243–77.273 s. At the 360-second deadline, the lanes had emitted 705, 705, 705, and 704 IDs. All 2,819 observed IDs matched the frozen prefixes. The first warmup ended with zero completed requests and no scored cohort.

The frozen throughput gate allows less than 84.091 s for all 4,096 IDs. At the observed first-ID latency, less than 6.82 s remained for 1,023 decode iterations and final delivery. A separate two-output trace attributed 9.008 s of the first decode step to attention.

Report prompt length, concurrency, output budget, and deadlines for diagnostics. W6 acceptance requires complete measured cohorts under the [current launch policy](performance.md#scheduler-and-reporting-policy) and all [frozen W6 gates](../benchmarks/concurrent/README.md#conditions-and-gates). The prototype leaves W6 acceptance open.
